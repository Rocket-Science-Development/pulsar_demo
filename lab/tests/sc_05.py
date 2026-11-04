"""SC-05 - Queue saturation and memory bound.

For max_queue_rows in 1k..1M, a fresh process hosts one Pulse whose storage request is replaced by a fake one absorbing
WRITE_RATE rows/s (so the queue is the only variable); a burst at 10x that rate is captured. The fake request is
injected at the Target of the Pulse's 0.10 Channel, after the real line-protocol encoding: replacing
storage_engine.write_frame (as up to 0.9) is no longer called by the 0.10 write path, and every row was then dropped
against the unreachable URL. Measured: RSS ceiling
against the bound implied by max_queue_rows, peak queued rows, and exact drop accounting
(captured = written + dropped + queued, written = rows the fake storage received). A last case submits one batch
larger than max_queue_rows.
"""

import json
import os
import subprocess
import sys
import threading
import time

LEVELS = [1_000, 10_000, 100_000, 1_000_000]
WRITE_RATE = 10_000  # rows/s absorbed by the fake storage
BURST_FACTOR = 10
BURST_BATCH = 500
PARAMS = {"levels": LEVELS, "write_rate_rows_s": WRITE_RATE, "burst_factor": BURST_FACTOR, "burst_batch_rows": BURST_BATCH,
          "features": 20, "storage": "fake request in process (no network) at the Channel's Target, real encoding"}
# writes_measured_all: the fake storage received rows at every level, otherwise the other criteria are empty (0.10
# write path not hooked: every row dropped, as in the 2026-11-02 rerun); oversize_*: m20, a single batch above
# max_queue_rows keeps its newest max_queue_rows rows and the rest is dropped and counted
CRITERIA = {"accounting_exact_all": ("==", True), "rss_within_bound_all": ("==", True),
            "writes_measured_all": ("==", True), "oversize_dropped_exact": ("==", True),
            "oversize_write_within_bound": ("==", True)}


def child(q: int, oversize: bool) -> dict:
    import psutil

    from pulsar_data_collection.models import CaptureOptions
    from lab.tests import _platform as C

    proc = psutil.Process()
    received = [0, 0, 0]  # rows received, largest single write, requests

    opts = CaptureOptions(max_queue_rows=q, max_batch_rows=min(5_000, q), flush_interval_s=0.2)
    env = {"url": "http://127.0.0.1:9", "token": "none", "org": "lab"}  # unreachable on purpose: no real storage
    pulse = C.make_pulse("sc05", "lab_none", opts, env)
    C.fake_storage(pulse, WRITE_RATE, received)
    pool = C.frames_pool(BURST_BATCH, 20, pool=8)
    if oversize:
        pool = C.frames_pool(q * 5, 20, pool=1)
    import gc

    gc.collect()
    base = proc.memory_info().rss
    peak = [base, 0]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            peak[1] = max(peak[1], pulse.stats()["queued_rows"])
            time.sleep(0.02)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    burst_rows = 1 if oversize else max(3 * q, 200_000)
    rate = WRITE_RATE * BURST_FACTOR
    t0 = time.perf_counter()
    sent = 0
    lat = []
    k = 0
    while sent < burst_rows:
        x, y = pool[k % len(pool)]
        x, y = x.copy(), y.copy()  # fresh frames, as in a real server (pandas copy-on-write would share the pool)
        s = time.perf_counter()
        C.capture(pulse, x, y)
        lat.append(time.perf_counter() - s)
        sent += len(x)
        k += 1
        ahead = sent / rate - (time.perf_counter() - t0)
        if ahead > 0:
            time.sleep(ahead)
    burst_s = time.perf_counter() - t0
    time.sleep(0.5)
    stats_mid = pulse.stats()
    pulse.flush(600)
    stop.set()
    th.join()
    st = pulse.stats()
    pulse.close(5)
    return {"q": q, "oversize": oversize, "rows_sent": sent, "burst_s": burst_s, "burst_rows_per_s": sent / burst_s,
            "base_rss_mb": base / 2**20, "peak_rss_mb": peak[0] / 2**20, "delta_rss_mb": (peak[0] - base) / 2**20,
            "peak_queued_rows": peak[1], "captured": st["captured_rows"], "written": st["written_rows"],
            "dropped": st["dropped_rows"], "queued_end": st["queued_rows"], "received_by_storage": received[0],
            "max_write_rows": received[1], "requests": received[2], "write_errors": st["write_errors"],
            "queued_mid": stats_mid["queued_rows"], "capture_p99_ms": sorted(lat)[int(0.99 * (len(lat) - 1))] * 1000,
            "bytes_per_row_estimate": 21 * 8 + 8}


def _spawn(q, oversize=False):
    out = subprocess.run([sys.executable, "-m", "lab.tests.sc_05", "child", str(q), str(int(oversize))],
                         capture_output=True, text=True, timeout=1800)
    lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if not lines:
        raise RuntimeError(f"child failed for q={q}: {out.stderr[-1500:]}")
    return json.loads(lines[-1])


def run(ctx):
    levels = [int(v) for v in os.environ.get("LAB_LEVELS", ",".join(map(str, LEVELS))).split(",")]
    rows = [_spawn(q) for q in levels]
    rows.append(_spawn(1_000, oversize=True))
    (ctx.workdir / "levels.json").write_text(json.dumps(rows, indent=1))
    exact_all, within_all = True, True
    for r in rows:
        r["accounting_exact"] = (r["captured"] == r["written"] + r["dropped"] + r["queued_end"]
                                 and r["written"] == r["received_by_storage"])
        # bound: queued rows x raw row size, x3 for pandas frames and the batch being written, + 20 MB slack
        r["rss_bound_mb"] = 3 * r["q"] * r["bytes_per_row_estimate"] / 2**20 + 20
        r["rss_within_bound"] = r["delta_rss_mb"] <= r["rss_bound_mb"]
        r["bytes_per_queued_row"] = r["delta_rss_mb"] * 2**20 / max(r["peak_queued_rows"], 1)
        exact_all &= r["accounting_exact"]
        within_all &= r["rss_within_bound"] or r["oversize"]
        if not r["accounting_exact"]:
            ctx.limit(f"Drop accounting not exact at max_queue_rows={r['q']}",
                      f"captured {r['captured']} vs written {r['written']} + dropped {r['dropped']} + queued {r['queued_end']}; "
                      f"storage received {r['received_by_storage']}",
                      "Count drops and writes under the same lock as the queue", "critical")
        if not r["oversize"] and r["peak_queued_rows"] > r["q"] + BURST_BATCH:
            ctx.limit(f"Queue exceeded max_queue_rows={r['q']}: peak {r['peak_queued_rows']} rows",
                      json.dumps({k: r[k] for k in ('q', 'peak_queued_rows', 'delta_rss_mb')}),
                      "Enforce the bound before appending", "major")
        if not r["oversize"] and not r["rss_within_bound"]:
            ctx.limit(f"RSS grew {r['delta_rss_mb']:.0f} MB at max_queue_rows={r['q']}, above the bound "
                      f"{r['rss_bound_mb']:.0f} MB", f"{r['bytes_per_queued_row']:.0f} B per queued row",
                      "Express the bound in bytes (max_queue_bytes) and document the per-row memory", "major")
    over = rows[-1]
    if over["dropped"] == 0 and over["max_write_rows"] > over["q"]:
        ctx.limit(f"A single batch larger than max_queue_rows is never dropped: {over['rows_sent']} rows accepted and "
                  f"written in one write of {over['max_write_rows']} rows with max_queue_rows={over['q']} "
                  f"(max_batch_rows={over['q']})",
                  f"one capture of {over['rows_sent']} rows; RSS +{over['delta_rss_mb']:.1f} MB; BackgroundWriter.submit "
                  "drops only while len(queue) > 1",
                  "Reject or split a batch above max_queue_rows (count it as dropped) so the memory bound holds for any "
                  "batch size; document it in CaptureOptions", "minor")
    big = [r for r in rows if not r["oversize"]]
    measured = all(r["received_by_storage"] > 0 for r in rows)
    if not measured:
        ctx.note("the fake storage received no row at some level: the write path was not hooked, the run is not a "
                 "measurement (" + ", ".join(f"q={r['q']}: {r['received_by_storage']}" for r in rows) + ")")
    metrics = {"accounting_exact_all": exact_all, "rss_within_bound_all": within_all, "writes_measured_all": measured,
               "oversize_dropped_exact": over["dropped"] == over["rows_sent"] - over["q"],
               "oversize_write_within_bound": 0 < over["max_write_rows"] <= over["q"],
               "levels": {str(r["q"]): {k: r[k] for k in ("peak_rss_mb", "delta_rss_mb", "rss_bound_mb",
                                                          "peak_queued_rows", "captured", "written", "dropped",
                                                          "bytes_per_queued_row", "burst_rows_per_s", "capture_p99_ms")}
                          for r in big},
               "oversize_case": {k: over[k] for k in ("q", "rows_sent", "max_write_rows", "delta_rss_mb", "dropped",
                                                      "received_by_storage")}}
    bpr = [r["bytes_per_queued_row"] for r in big if r["peak_queued_rows"] >= 10_000]
    if bpr:
        metrics["bytes_per_queued_row_median"] = sorted(bpr)[len(bpr) // 2]
        if metrics["bytes_per_queued_row_median"] > 2 * (21 * 8 + 8):
            ctx.limit(f"Each queued row costs {metrics['bytes_per_queued_row_median']:.0f} B of RSS, "
                      f"{metrics['bytes_per_queued_row_median'] / 176:.1f}x its raw size (20 float features + time)",
                      json.dumps({str(r['q']): round(r['bytes_per_queued_row']) for r in big}),
                      "Document memory per row for sizing max_queue_rows, or add max_queue_bytes", "minor")
    return metrics


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "child":
    print(json.dumps(child(int(sys.argv[2]), bool(int(sys.argv[3])))), flush=True)
