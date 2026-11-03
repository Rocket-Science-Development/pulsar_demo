"""SC-02 - High-volume sustained ingestion: ramp 10k -> 1M rows/min, one model then 100 models.

Each step lasts STEP_S seconds against the platform InfluxDB (lab-db:8087, dedicated container), into its own
bucket. Per step: achieved rate, capture latency, queue depth, drops, write errors, write batch sizes (mean rows per
write), ingestion lag (time to flush at the end of the step), InfluxDB heap/requests, then an end-to-end
reconciliation (rows queried back = rows captured - dropped) and how far row timestamps ran ahead of the wall clock.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone

from lab.tests import _platform as C

RATES = [int(v) for v in os.environ.get("LAB_RATES", "10000,100000,500000,1000000,2000000,4000000").split(",")]
MODEL_SETS = [int(v) for v in os.environ.get("LAB_MODEL_SETS", "1,100").split(",")]
STEP_S = float(os.environ.get("LAB_STEP_S", 180))
BATCH = 50
PARAMS = {"rates_rows_per_min": RATES, "model_sets": MODEL_SETS, "step_s": STEP_S, "batch_rows": BATCH, "features": 20,
          "storage": "lab-db InfluxDB 2.6.1 :8087 (dedicated container, r7i.2xlarge)",
          "deviation": "3 min per step instead of 15 (night budget); saturation located on the same ramp",
          "write": C.versions()}
CRITERIA = {"loss_below_saturation": ("==", 0), "drops_counted_exactly": ("==", True)}


def _step(env, n_models, rate):
    bucket = C.ensure_bucket(f"lab_sc02_{datetime.now(timezone.utc):%m%d%H%M}_m{n_models}_r{rate}", env)
    writer = C.shared_writer()
    pulses = [C.make_pulse(f"v{i:03d}", bucket, C.capture_options(), env, writer=writer) for i in range(n_models)]
    writes = []
    # Rows per write call: wraps the instance's write_frame (private writers). A SharedWriter sends line protocol
    # without write_frame; rows per write then come from the InfluxDB write requests.
    for p in pulses if writer is None else []:
        orig = p.storage_engine.write_frame

        def wrapped(*a, _orig=orig, **kw):
            writes.append(len(kw.get("frame", a[1] if len(a) > 1 else [])))
            return _orig(*a, **kw)

        p.storage_engine.write_frame = wrapped
    pool = C.frames_pool(BATCH, 20, pool=64)
    lat, trace = [], []
    stop = threading.Event()
    m0 = C.remote_influx_stats(env)

    def sampler():
        while not stop.wait(5):
            st = [p.stats() for p in pulses]
            trace.append({"queued": sum(s["queued_rows"] for s in st), "dropped": sum(s["dropped_rows"] for s in st),
                          "written": sum(s["written_rows"] for s in st), "errors": sum(s["write_errors"] for s in st),
                          **{f"influx_{k}": v for k, v in C.remote_influx_stats(env).items() if k != "error"}})

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    step = BATCH / (rate / 60)
    t0 = time.perf_counter()
    wall0 = datetime.now(timezone.utc)
    k = 0
    while time.perf_counter() - t0 < STEP_S:
        x, y = pool[k % len(pool)]
        s = time.perf_counter()
        C.capture(pulses[k % n_models], x, y)
        lat.append(time.perf_counter() - s)
        k += 1
        ahead = k * step - (time.perf_counter() - t0)
        if ahead > 0:
            time.sleep(ahead)
    elapsed = time.perf_counter() - t0
    st_end = [p.stats() for p in pulses]
    t_f = time.perf_counter()
    flushed = all(p.flush(900) for p in pulses)
    lag_s = time.perf_counter() - t_f
    stop.set()
    th.join()
    stats = [p.stats() for p in pulses]
    for p in pulses:
        p.close()
    if writer is not None:
        writer.close()
    m1 = C.remote_influx_stats(env)
    wall_end = datetime.now(timezone.utc)
    stored = C.count_rows(bucket, "f0", None, env, timeout_s=1800)
    q = f'''from(bucket: "{bucket}") |> range(start: 0, stop: {C.FAR_FUTURE}) |> filter(fn: (r) => r._field == "f0")
  |> group() |> sort(columns: ["_time"], desc: true) |> limit(n: 1)'''
    with C.client(env, 900) as c:
        tables = c.query_api().query(q)
    last_ts = max((r.get_time() for t in tables for r in t.records), default=None)
    ahead_s = (last_ts - wall_end).total_seconds() if last_ts else None
    captured = sum(s["captured_rows"] for s in stats)
    dropped = sum(s["dropped_rows"] for s in stats)
    requests = m1.get("write_requests", 0) - m0.get("write_requests", 0)
    n_writes = len(writes) if writer is None else int(requests)
    if writer is None:
        rows_per_write = sum(writes) / len(writes) if writes else None
    else:
        rows_per_write = stored / requests if requests else None
    return {"models": n_models, "target_rows_per_min": rate, "achieved_rows_per_min": k * BATCH / elapsed * 60,
            "captured": captured, "dropped": dropped, "dropped_at_step_end": sum(s["dropped_rows"] for s in st_end),
            "rejected": sum(s["rejected_rows"] for s in stats), "write_errors": sum(s["write_errors"] for s in stats),
            "stored": stored, "lost_unaccounted": captured - dropped - stored, "flushed": flushed,
            "uncertain": sum(s.get("uncertain_rows", 0) for s in stats),
            "ingestion_lag_s": lag_s, "queued_at_step_end": sum(s["queued_rows"] for s in st_end),
            "queued_max": max([t["queued"] for t in trace], default=0),
            "written_rows_per_s": (stored) / (elapsed + lag_s),
            "writes": n_writes, "rows_per_write_mean": rows_per_write,
            "capture_p50_ms": C.pct(lat, 50) * 1000, "capture_p99_ms": C.pct(lat, 99) * 1000,
            "influx_write_requests": requests,
            "influx_heap_mb_max": max([t.get("influx_go_memstats_alloc_bytes", 0) for t in trace], default=0) / 2**20,
            "last_row_ahead_of_wall_clock_s": ahead_s, "wall_start": wall0.isoformat()}


def run(ctx):
    os.environ["LAB_INFLUX_ENV"] = C.platform_env_file()
    env = C.influx_env()
    rows = []
    for n in MODEL_SETS:
        for rate in RATES:
            try:
                r = _step(env, n, rate)
            except Exception as exc:  # noqa: BLE001
                r = {"models": n, "target_rows_per_min": rate, "error": f"{type(exc).__name__}: {exc}"[:400]}
            rows.append(r)
            (ctx.workdir / "steps.json").write_text(json.dumps(rows, indent=1, default=str))
            ctx.note(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()
                                 if k in ("models", "target_rows_per_min", "achieved_rows_per_min", "dropped", "stored",
                                          "lost_unaccounted", "ingestion_lag_s", "queued_max", "error")}))
    ok = [r for r in rows if "error" not in r]
    sat = {}
    for n in MODEL_SETS:
        rs = [r for r in ok if r["models"] == n]
        bad = [r for r in rs if r["dropped"] > 0 or r["ingestion_lag_s"] > 30 or r["achieved_rows_per_min"] < 0.95 * r["target_rows_per_min"]]
        sat[n] = min((r["target_rows_per_min"] for r in bad), default=None)
        good = [r for r in rs if r not in bad]
        best = max(good, key=lambda r: r["target_rows_per_min"]) if good else None
        if sat[n] is not None:
            first = min(bad, key=lambda r: r["target_rows_per_min"])
            ctx.limit(f"Saturation with {n} model(s) at {sat[n]:,} rows/min (last clean step "
                      f"{best['target_rows_per_min']:,} rows/min)" if best else f"Saturation with {n} model(s) from the first step",
                      f"at {first['target_rows_per_min']:,}: achieved {first['achieved_rows_per_min']:,.0f} rows/min, "
                      f"dropped {first['dropped']:,}, queue max {first['queued_max']:,}, flush lag {first['ingestion_lag_s']:.0f} s, "
                      f"{first['rows_per_write_mean']} rows/write, written {first['written_rows_per_s']:,.0f} rows/s",
                      "One synchronous writer thread per Pulse caps throughput: write with several in-flight requests "
                      "(async write_api or a pool), gzip, and larger batches when the queue grows; document the per-Pulse "
                      "ceiling", "major")
    # A row of a timed-out write (uncertain_rows, pulsar_data_collection >= 0.10) may or may not be stored: rows are
    # unaccounted only beyond the uncertain ones
    for r in ok:
        gap = r["lost_unaccounted"]
        r["unexplained"] = max(0, gap - r.get("uncertain", 0)) if gap > 0 else gap
    lost = sum(abs(r["unexplained"]) for r in ok)  # 0 when every lost row is counted
    for r in ok:
        if r["unexplained"] > 0:
            ctx.limit(f"{r['unexplained']} rows neither stored nor counted as dropped or uncertain ({r['models']} models, "
                      f"{r['target_rows_per_min']:,} rows/min)",
                      json.dumps({k: r[k] for k in ("captured", "dropped", "uncertain", "rejected", "stored", "write_errors")}),
                      "Reconcile counters: count timeouts with unknown outcome separately", "critical")
        elif r["lost_unaccounted"] < 0:
            ctx.limit(f"{-r['lost_unaccounted']} rows counted as dropped were stored ({r['models']} models, "
                      f"{r['target_rows_per_min']:,} rows/min): write timeouts with unknown outcome",
                      json.dumps({k: r[k] for k in ("captured", "dropped", "stored", "write_errors")}),
                      "Count timed-out writes as unknown_rows, not dropped; make the counters reconcile with the store",
                      "major")
        if r["write_errors"] and r["dropped"]:
            ctx.limit(f"Write timeouts at {r['target_rows_per_min']:,} rows/min with {r['models']} models: "
                      f"{r['write_errors']} failed writes, {r['dropped']:,} rows dropped "
                      f"({r['dropped'] / max(r['captured'], 1):.1%}) without spool",
                      f"queue max {r['queued_max']:,}, {r['rows_per_write_mean']} rows/write, {r['writes']} writes; "
                      "default write_timeout_s=5",
                      "Spool on timeout by default for servers; fewer, larger writes across models (shared writer)", "major")
    ahead = [r for r in ok if (r["last_row_ahead_of_wall_clock_s"] or 0) > 60]
    if ahead:
        r = max(ahead, key=lambda r: r["last_row_ahead_of_wall_clock_s"])
        ctx.limit(f"Row timestamps run ahead of the wall clock: {r['last_row_ahead_of_wall_clock_s']:,.0f} s in the future "
                  f"after {STEP_S:.0f} s at {r['target_rows_per_min'] / r['models']:,.0f} rows/min per model",
                  "prepare_frame spaces rows by 1 ms and _unique_times pushes each batch after the previous one: above "
                  "1,000 rows/s per model, timestamps drift into the future; "
                  + json.dumps({f"{x['models']}m@{x['target_rows_per_min']}": round(x['last_row_ahead_of_wall_clock_s'] or 0)
                                for x in ok}),
                  "Space rows by 1 us (or ns) instead of 1 ms, or keep the capture timestamp and add a row index tag/field; "
                  "queries with range(stop: now()) miss these rows", "major")
    below = [r for r in ok if sat.get(r["models"]) is None or r["target_rows_per_min"] < sat[r["models"]]]
    return {"steps": rows, "saturation_rows_per_min": {str(k): v for k, v in sat.items()},
            "loss_below_saturation": sum(r["dropped"] + abs(r["unexplained"]) for r in below),
            "drops_counted_exactly": lost == 0}
