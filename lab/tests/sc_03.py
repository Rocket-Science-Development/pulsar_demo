"""SC-03 - Storage outage under load, with and without spool.

10 models share 200k rows/min (batches of 50 rows, rows carry a unique int `seq`) into the platform InfluxDB
(lab-db:8087, dedicated container) through a local TCP proxy. The outage cuts the proxy (connection refused, as a
stopped InfluxDB) for 60 s, 10 min and 30 min, with spool_dir and without; one extra case black-holes the network
(connections accepted, never answered) for 60 s with spool. Measured: capture latency during the outage, spooled /
dropped rows, spool disk usage, replay duration, losses and duplicates (points vs distinct seq queried back).

Rows spooled or dropped "after the recovery" are counted by their capture timestamp (rows captured after the storage
was back), not by the instant they reached the spool: after a black hole, the batches captured during the hang still
wait behind the write timeout and reach the spool after the recovery.
"""

import json
import os
import shutil
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from lab.tests import _platform as C

OUTAGES = [int(v) for v in os.environ.get("LAB_OUTAGES", "60,600,1800").split(",")]
MODELS, BATCH, ROWS_PER_MIN = 10, 50, 200_000
PRE_S, POST_S = float(os.environ.get("LAB_PRE_S", 60)), float(os.environ.get("LAB_POST_S", 60))
PARAMS = {"outages_s": OUTAGES, "models": MODELS, "batch_rows": BATCH, "rows_per_min": ROWS_PER_MIN, "pre_s": PRE_S,
          "post_s": POST_S, "outage": "local TCP proxy closed (refused) / black hole (hang)",
          "storage": "lab-db InfluxDB 2.6.1 :8087 (dedicated)", "write": C.versions(),
          "tuned_spool": "spool_compression=gzip (with spool)"}
CRITERIA = {"spool_lost_total": ("==", 0), "spool_duplicates_total": ("==", 0), "spool_outage_p99_ms_max": ("<", 2.0),
            "nospool_drops_exact_all": ("==", True)}


def _distinct_and_points(bucket, measurement, env):
    q = f'''from(bucket: "{bucket}") |> range(start: 0, stop: {C.FAR_FUTURE})
  |> filter(fn: (r) => r._measurement == "{measurement}" and r._field == "seq")'''
    with C.client(env, 1200) as c:
        api = c.query_api()
        pts = api.query(q + " |> group() |> count()")
        dis = api.query(q + " |> group() |> distinct() |> count()")
    val = lambda t: int(sum(r.get_value() for x in t for r in x.records))  # noqa: E731
    return val(pts), val(dis)


def _spool_bytes(root):
    """Size of the spool files (.json, .json.gz); a file replayed and removed while it is measured counts 0"""
    total = 0
    for f in root.rglob("*.json*"):
        try:
            total += f.stat().st_size
        except FileNotFoundError:
            pass
    return total


class _FallbackByCapture:
    """Rows sent to the spool or dropped (the writer's fallback: storage unavailable, failed or timed-out write, close),
    split by the phase in which they were captured (their `_time`, the capture timestamp)"""

    def __init__(self, writers):
        self.rows = {"pre": 0, "outage": 0, "post": 0}
        self.marks = {}  # phase -> start (UTC Timestamp)
        self.hooked = False
        self._lock = threading.Lock()
        for w in writers:
            if hasattr(w, "_fallback"):
                w._fallback = self._wrap(w._fallback)
                self.hooked = True

    def mark(self, phase):
        self.marks[phase] = pd.Timestamp.now(tz="UTC")

    def _wrap(self, orig):
        def fallback(unit, *a, **kw):
            if not unit.replay and "_time" in unit.frame:  # a replayed batch stays in the spool: counted once
                times = pd.to_datetime(unit.frame["_time"], utc=True)
                post = int((times >= self.marks["post"]).sum()) if "post" in self.marks else 0
                outage = int((times >= self.marks["outage"]).sum()) - post if "outage" in self.marks else 0
                with self._lock:
                    self.rows["post"] += post
                    self.rows["outage"] += outage
                    self.rows["pre"] += len(times) - post - outage
            return orig(unit, *a, **kw)

        return fallback


def _case(ctx, env, proxy, outage_s, spool, mode="refuse"):
    name = f"{mode}_{outage_s}s_{'spool' if spool else 'nospool'}"
    bucket = C.ensure_bucket(f"lab_sc03_{datetime.now(timezone.utc):%m%d%H%M}_{name}", env)
    spool_root = ctx.workdir / "spool" / name
    if spool_root.exists():
        shutil.rmtree(spool_root)
    proxy.up()
    proxy.hang = False
    penv = {**env, "url": proxy.url}
    writer = C.shared_writer()
    pulses = [C.make_pulse(f"o{i}", bucket,
                           C.capture_options(tuned={"spool_compression": "gzip"} if spool else None,
                                             spool_dir=(spool_root / f"m{i}") if spool else None),
                           penv, writer=writer)
              for i in range(MODELS)]
    fallbacks = _FallbackByCapture(C.writers_of(pulses))
    pool = C.frames_pool(BATCH, 20, pool=32)
    seq = [i * 100_000_000 for i in range(MODELS)]
    seq0 = list(seq)
    phase = {"name": "pre"}
    lat = {"pre": [], "outage": [], "post": []}
    stop = threading.Event()
    trace = []

    def generator():
        step = BATCH / (ROWS_PER_MIN / 60)
        t0 = time.perf_counter()
        k = 0
        while not stop.is_set():
            i = k % MODELS
            x, y = pool[k % len(pool)]
            s = time.perf_counter()
            C.capture(pulses[i], x, y, seq_start=seq[i])
            lat[phase["name"]].append(time.perf_counter() - s)
            seq[i] += BATCH
            k += 1
            ahead = k * step - (time.perf_counter() - t0)
            if ahead > 0:
                time.sleep(ahead)

    def sampler():
        t0 = time.time()
        while not stop.wait(5):
            st = [p.stats() for p in pulses]
            du = _spool_bytes(spool_root) if spool else 0
            trace.append({"t": time.time() - t0, "phase": phase["name"], "spool_mb": du / 2**20,
                          "spool_files": sum(s["spool_files"] for s in st), "queued": sum(s["queued_rows"] for s in st),
                          "spooled": sum(s["spooled_rows"] for s in st), "dropped": sum(s["dropped_rows"] for s in st),
                          "replayed": sum(s["replayed_rows"] for s in st), "written": sum(s["written_rows"] for s in st),
                          "rss_mb": C.proc_snapshot()["rss_mb"]})

    g = threading.Thread(target=generator, daemon=True)
    sm = threading.Thread(target=sampler, daemon=True)
    g.start()
    sm.start()
    time.sleep(PRE_S)
    phase["name"] = "outage"
    fallbacks.mark("outage")
    if mode == "hang":
        proxy.down()
        proxy.hang = True
        proxy.up()
    else:
        proxy.down()
    time.sleep(outage_s)
    st_out = [p.stats() for p in pulses]
    proxy.hang = False
    proxy.down()
    proxy.up()
    t_back = time.time()
    fallbacks.mark("post")
    phase["name"] = "post"
    replay_s = None
    if spool:
        while time.time() - t_back < max(POST_S, 7200):
            if sum(p.stats()["spool_files"] for p in pulses) == 0:
                replay_s = time.time() - t_back
                break
            time.sleep(1)
    rest = POST_S - (time.time() - t_back)
    if rest > 0:
        time.sleep(rest)
    stop.set()
    g.join()
    flushed = all(p.flush(300) for p in pulses)
    sm.join()
    stats = [p.stats() for p in pulses]
    for p in pulses:
        p.close()
    if writer is not None:
        writer.close()
    points = distinct = 0
    for i in range(MODELS):
        pt, ds = _distinct_and_points(bucket, f"o{i}_1_input_data", env)
        points += pt
        distinct += ds
    captured = sum(s["captured_rows"] for s in stats)
    submitted = sum(seq[i] - seq0[i] for i in range(MODELS))
    dropped = sum(s["dropped_rows"] for s in stats)
    row = {"case": name, "outage_s": outage_s, "spool": spool, "mode": mode, "captured": captured, "submitted": submitted,
           "stored_points": points, "distinct": distinct, "lost": submitted - distinct, "duplicates": points - distinct,
           "dropped": dropped, "uncertain": sum(s.get("uncertain_rows", 0) for s in stats),
           "spooled": sum(s["spooled_rows"] for s in stats),
           "replayed": sum(s["replayed_rows"] for s in stats), "written": sum(s["written_rows"] for s in stats),
           "write_errors": sum(s["write_errors"] for s in stats), "flushed": flushed,
           "spooled_during_outage": sum(s["spooled_rows"] for s in st_out),
           "dropped_during_outage": sum(s["dropped_rows"] for s in st_out),
           "spool_mb_max": max([t["spool_mb"] for t in trace], default=0),
           "spool_files_max": max([t["spool_files"] for t in trace], default=0),
           "queued_max_outage": max([t["queued"] for t in trace if t["phase"] == "outage"], default=0),
           "rss_max_mb": max([t["rss_mb"] for t in trace], default=0), "replay_s": replay_s,
           "spool_files_end": sum(s["spool_files"] for s in stats),
           "fallback_hooked": fallbacks.hooked,
           **{f"fallback_rows_captured_{ph}": n for ph, n in fallbacks.rows.items()}}
    for ph in ("pre", "outage", "post"):
        row[f"p50_ms_{ph}"] = (C.pct(lat[ph], 50) or 0) * 1000
        row[f"p99_ms_{ph}"] = (C.pct(lat[ph], 99) or 0) * 1000
        row[f"max_ms_{ph}"] = max(lat[ph], default=0) * 1000
    row["spool_bytes_per_row"] = row["spool_mb_max"] * 2**20 / max(row["spooled_during_outage"], 1)
    row["drops_exact"] = (captured - points == dropped) if not spool else None
    (ctx.workdir / f"trace_{name}.json").write_text(json.dumps(trace))
    return row


def run(ctx):
    os.environ["LAB_INFLUX_ENV"] = C.platform_env_file()
    env = C.influx_env()
    proxy = C.OutageProxy(env["url"], port=18103)
    cases = [(o, s, "refuse") for o in OUTAGES for s in (True, False)]
    if os.environ.get("LAB_HANG", "1") == "1":
        cases.append((60, True, "hang"))
    rows = []
    for o, s, mode in cases:
        try:
            rows.append(_case(ctx, env, proxy, o, s, mode))
        except Exception as exc:  # noqa: BLE001 - one failed case does not stop the others
            rows.append({"case": f"{mode}_{o}s_{s}", "error": f"{type(exc).__name__}: {exc}"[:400]})
        (ctx.workdir / "cases.json").write_text(json.dumps(rows, indent=1, default=str))
    proxy.down()
    ok = [r for r in rows if "error" not in r]
    sp = [r for r in ok if r["spool"]]
    ns = [r for r in ok if not r["spool"]]
    metrics = {"cases": {r["case"]: r for r in rows},
               "spool_lost_total": sum(r["lost"] for r in sp), "spool_duplicates_total": sum(r["duplicates"] for r in sp),
               "spool_outage_p99_ms_max": max([r["p99_ms_outage"] for r in sp], default=None),
               "nospool_drops_exact_all": all(r["drops_exact"] for r in ns) if ns else None}
    for r in rows:
        if "error" in r:
            ctx.note(f"case {r['case']} failed: {r['error']}")
    for r in sp:
        if r["lost"] or r["duplicates"]:
            ctx.limit(f"[{r['case']}] with spool: {r['lost']} rows lost, {r['duplicates']} duplicates",
                      json.dumps({k: r[k] for k in ("submitted", "distinct", "stored_points", "spooled", "replayed",
                                                    "dropped", "spool_files_end")}),
                      "Reconcile spool replay with captured rows", "critical")
        if r["p99_ms_outage"] >= 2:
            ctx.limit(f"[{r['case']}] capture p99 {r['p99_ms_outage']:.2f} ms during the outage (target < 2 ms; "
                      f"{r['p99_ms_pre']:.2f} ms before)",
                      f"p50 {r['p50_ms_outage']:.2f} ms, max {r['max_ms_outage']:.0f} ms; spool writes "
                      f"{r['spool_files_max']} files, {r['spool_mb_max']:.0f} MB",
                      "Spool writes happen in the writer thread but hold the queue lock (_fallback under _condition): "
                      "write spool files outside the lock", "major")
        if r["replay_s"] is not None and r["replay_s"] > max(60, r["outage_s"] / 2):
            ctx.limit(f"[{r['case']}] replay of a {r['outage_s']} s outage takes {r['replay_s']:.0f} s",
                      f"{r['spool_files_max']} spool files ({r['spool_mb_max']:.0f} MB, {r['spool_bytes_per_row']:.0f} "
                      "B/row); replay is bounded to 10 files per writer iteration (about 1/s per Pulse when idle)",
                      "Replay in larger chunks (merge spool files) and adapt to storage throughput", "minor")
        if r["spool_bytes_per_row"] > 400:
            ctx.limit(f"[{r['case']}] the spool stores {r['spool_bytes_per_row']:.0f} B per row (JSON); "
                      f"{r['spool_mb_max']:.0f} MB for {r['spooled_during_outage']} rows",
                      "20 float features + seq + time per row", "Spool in Parquet/Arrow (5-10x smaller) and add "
                      "max_spool_bytes with oldest-first eviction, counted as dropped", "minor")
    for r in ok:
        # by counter: rows that reached the spool (or were dropped) after the recovery, whenever they were captured
        by_counter = ((r["dropped"] - r["dropped_during_outage"]) if not r["spool"]
                      else (r["spooled"] - r["spooled_during_outage"]))
        r["fallback_rows_after_recovery_by_counter"] = by_counter
        # by capture timestamp: rows captured after the recovery that still went to the spool or were dropped
        lost_after = r["fallback_rows_captured_post"] if r["fallback_hooked"] else by_counter
        r["fallback_rows_after_recovery"] = lost_after
        rate = ROWS_PER_MIN / 60
        if lost_after > 2 * rate:
            ctx.limit(f"[{r['case']}] {lost_after} rows {'dropped' if not r['spool'] else 'spooled'} after the storage was "
                      f"back (about {lost_after / rate:.0f} s of traffic): the exponential retry delay keeps the storage "
                      "marked unavailable",
                      f"rows captured after the recovery and {'dropped' if not r['spool'] else 'spooled'}: {lost_after}"
                      f" (captured during the outage: {r['fallback_rows_captured_outage']}); reaching the "
                      f"{'drop' if not r['spool'] else 'spool'} after the recovery, whatever their capture: "
                      f"{by_counter}; "
                      "max_retry_delay_s=60 (default)",
                      "Probe the storage (cheap /health or /ping) while in back-off instead of failing every batch until "
                      "the delay ends; cap the delay at a few seconds by default", "major" if not r["spool"] else "minor")
        if r["p99_ms_post"] >= 2 and r["p99_ms_post"] > 3 * r["p99_ms_pre"]:
            ctx.limit(f"[{r['case']}] capture p99 rises to {r['p99_ms_post']:.1f} ms after the outage (before: "
                      f"{r['p99_ms_pre']:.2f} ms), max {r['max_ms_post']:.0f} ms",
                      f"replay of {r['spool_files_max']} spool files in {r['replay_s']} s; writer and caller share "
                      "one lock and the GIL", "Throttle replay and keep file I/O and JSON parsing outside the queue lock",
                      "minor")
    for r in ns:
        if not r["drops_exact"]:
            over = r["dropped"] - (r["captured"] - r["stored_points"])
            what = (f"{over} rows counted as dropped were stored (a write cut by the outage has an unknown outcome)"
                    if over > 0 else f"{-over} lost rows not counted as dropped")
            ctx.limit(f"[{r['case']}] without spool: drops not counted exactly: {what}",
                      f"captured {r['captured']}, stored {r['stored_points']}, dropped {r['dropped']}, "
                      f"uncertain {r['uncertain']}, write errors {r['write_errors']}",
                      "Count a write whose connection broke after the request was sent as uncertain (like a timeout), "
                      "not as dropped" if over > 0 else "Count every lost row in dropped_rows", "major")
    return metrics
