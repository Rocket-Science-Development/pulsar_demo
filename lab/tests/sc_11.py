"""SC-11 - Soak, 8 hours steady state: 100 models at 50k rows/min + periodic analysis job + 1 agent at 1 run/s.

One process hosts everything (the harder case: capture threads, analysis and agent share the interpreter). Every
minute: RSS, threads, descriptors, sockets, queue, spool, InfluxDB RSS/disk from /metrics, to samples.csv.
At the end: RSS slope after warm-up, thread/descriptor slopes, and an exact reconciliation per measurement.
"""

import json
import os
import random
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from lab.tests import _platform as C

DURATION_MIN = float(os.environ.get("LAB_DURATION_MIN", 480))
WARMUP_MIN = min(30.0, DURATION_MIN / 4)
N_MODELS = int(os.environ.get("LAB_MODELS", 100))
ROWS_PER_MIN = int(os.environ.get("LAB_ROWS_PER_MIN", 50_000))
BATCH = 50
ANALYSIS_EVERY_S = float(os.environ.get("LAB_ANALYSIS_EVERY_S", 900))

PARAMS = {"duration_min": DURATION_MIN, "warmup_min": WARMUP_MIN, "models": N_MODELS, "rows_per_min": ROWS_PER_MIN,
          "batch_rows": BATCH, "features": 20, "analysis_every_s": ANALYSIS_EVERY_S, "agent_runs_per_s": 1}
CRITERIA = {"rss_growth_pct_per_h": ("<=", 5.0), "thread_growth_per_h": ("<=", 0.5), "fd_growth_per_h": ("<=", 0.5),
            "lost_rows": ("==", 0)}


def _slope_per_h(t, v):
    if len(t) < 3:
        return None
    return float(np.polyfit(np.asarray(t) / 3600.0, np.asarray(v, dtype=float), 1)[0])


def run(ctx):
    from pulsar_data_collection.models import CaptureOptions
    from pulsar_metrics.analyzers.base import Analyzer
    from pulsar_metrics.calibration import AACalibrator
    from pulsar_metrics.multiple_testing import fdr_correct

    env = C.influx_env()
    bucket = C.ensure_bucket(f"lab_sc11_{datetime.now(timezone.utc):%m%d%H%M}", env)
    ctx.params["bucket"] = bucket
    spool_root = ctx.workdir / "spool"
    pulses = [C.make_pulse(f"soak{i:03d}", bucket, CaptureOptions(spool_dir=spool_root / f"m{i:03d}"), env)
              for i in range(N_MODELS)]
    agent = C.make_pulse("soakagent", bucket, CaptureOptions(spool_dir=spool_root / "agent"), env)
    pool = C.frames_pool(BATCH, 20, pool=32)
    reference = pd.concat([x for x, _ in pool[:16]], ignore_index=True)

    stop = threading.Event()
    interval = N_MODELS * BATCH / (ROWS_PER_MIN / 60.0)  # seconds between two batches of one model
    lat, late = [], [0]
    analysis_log, agent_runs = [], [0]
    seq = {i: 0 for i in range(N_MODELS)}

    def generator():
        step = interval / N_MODELS
        nxt = time.monotonic()
        k = 0
        while not stop.is_set():
            i = k % N_MODELS
            x, y = pool[k % len(pool)]
            t0 = time.perf_counter()
            C.capture(pulses[i], x, y, seq_start=seq[i])
            lat.append(time.perf_counter() - t0)
            seq[i] += BATCH
            k += 1
            nxt += step
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -1:
                late[0] += 1

    def agent_loop():
        rng = random.Random(1)
        while not stop.wait(1.0):
            row = {"steps": rng.randint(1, 9), "tool_calls": rng.randint(0, 5), "latency_ms": rng.random() * 900,
                   "answer": rng.choice(["approved", "denied", "needs review"]), "run": agent_runs[0]}
            if rng.random() < 0.7:  # optional field, as in agent records
                row["task_success"] = rng.random() < 0.8
            C.capture(agent, pd.DataFrame([row]), None)
            agent_runs[0] += 1

    def analysis_loop():
        q_api = C.client(env, 300).query_api()
        while not stop.wait(ANALYSIS_EVERY_S):
            t0, s0 = time.perf_counter(), C.proc_snapshot()["rss_mb"]
            models = random.Random(len(analysis_log)).sample(range(N_MODELS), 5)
            try:
                for i in models:
                    m = f"soak{i:03d}_1_input_data"
                    q = f'''from(bucket: "{bucket}") |> range(start: -{int(ANALYSIS_EVERY_S)}s, stop: {C.FAR_FUTURE})
  |> filter(fn: (r) => r._measurement == "{m}") |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'''
                    cur = q_api.query_data_frame(q)
                    if isinstance(cur, list):
                        cur = pd.concat(cur)
                    feats = [f"f{j}" for j in range(20)]
                    cur = cur.assign(model_id=f"soak{i:03d}", model_version="1", pred_timestamp=cur["_time"])
                    ref = reference.assign(model_id=f"soak{i:03d}", model_version="1",
                                           pred_timestamp=datetime.now(timezone.utc))
                    an = Analyzer(name="soak", model_id=f"soak{i:03d}", model_version="1")
                    an.add_drift_metrics(metrics_list=["ks_2samp"], features_list=feats)
                    cal = AACalibrator(window_size=200, n_draws=50, contiguous=False).fit(ref, an._metrics_list)
                    an.run(current=cur, reference=ref, options=cal.options())
                    res = an.results_to_pandas()
                    res["p_empirical"] = cal.empirical_pvalues(res)
                    fdr_correct(res, "p_empirical")
                analysis_log.append({"t": time.time(), "seconds": time.perf_counter() - t0, "rows": int(len(cur)),
                                     "rss_delta_mb": C.proc_snapshot()["rss_mb"] - s0})
            except Exception as exc:  # noqa: BLE001 - recorded, the soak continues
                analysis_log.append({"t": time.time(), "error": f"{type(exc).__name__}: {exc}"[:300]})

    threads = [threading.Thread(target=f, daemon=True) for f in (generator, agent_loop, analysis_loop)]
    t_start = time.time()
    for th in threads:
        th.start()
    samples = []
    csv = ctx.workdir / "samples.csv"
    end = t_start + DURATION_MIN * 60
    while time.time() < end:
        time.sleep(min(60, max(0, end - time.time())))
        snap = C.proc_snapshot()
        stats = [p.stats() for p in pulses]
        snap.update({"t": time.time() - t_start,
                     "captured": sum(s["captured_rows"] for s in stats), "written": sum(s["written_rows"] for s in stats),
                     "dropped": sum(s["dropped_rows"] for s in stats), "spooled": sum(s["spooled_rows"] for s in stats),
                     "queued": sum(s["queued_rows"] for s in stats), "spool_files": sum(s["spool_files"] for s in stats),
                     "write_errors": sum(s["write_errors"] for s in stats), "late_ticks": late[0],
                     "capture_p99_ms_last": C.pct(lat[-20000:], 99) * 1000 if lat else None})
        snap.update({f"influx_{k}": v for k, v in C.remote_influx_stats(env).items() if k != "error"})
        samples.append(snap)
        pd.DataFrame(samples).to_csv(csv, index=False)
    stop.set()
    for th in threads:
        th.join(timeout=ANALYSIS_EVERY_S + 60)
    for p in pulses + [agent]:
        p.flush(120)
    stats = [p.stats() for p in pulses]
    agent_stats = agent.stats()
    for p in pulses + [agent]:
        p.close()

    # reconciliation: rows queried back per measurement vs captured
    stored = C.count_by_measurement(bucket, "f0", env)
    lost = 0
    per_model_loss = {}
    for i, s in enumerate(stats):
        got = stored.get(f"soak{i:03d}_1_input_data", 0)
        if got != s["captured_rows"]:
            per_model_loss[i] = s["captured_rows"] - got
        lost += s["captured_rows"] - got
    agent_stored = C.count_rows(bucket, "steps", "soakagent", env)

    df = pd.DataFrame(samples)
    warm = df[df["t"] >= WARMUP_MIN * 60]
    rss_slope = _slope_per_h(warm["t"], warm["rss_mb"]) if len(warm) else None
    rss_after_warm = float(warm["rss_mb"].iloc[0]) if len(warm) else None
    metrics = {
        "duration_h": (time.time() - t_start) / 3600, "samples": len(df),
        "captured_rows": sum(s["captured_rows"] for s in stats), "written_rows": sum(s["written_rows"] for s in stats),
        "replayed_rows": sum(s["replayed_rows"] for s in stats), "dropped_rows": sum(s["dropped_rows"] for s in stats),
        "rejected_rows": sum(s["rejected_rows"] for s in stats), "write_errors": sum(s["write_errors"] for s in stats),
        "stored_rows": sum(stored.get(f"soak{i:03d}_1_input_data", 0) for i in range(N_MODELS)),
        "lost_rows": lost, "models_with_loss": len(per_model_loss),
        "agent_captured": agent_stats["captured_rows"], "agent_stored": agent_stored,
        "agent_rejected": agent_stats["rejected_rows"], "agent_runs": agent_runs[0],
        "achieved_rows_per_min": sum(s["captured_rows"] for s in stats) / max((time.time() - t_start) / 60, 1e-9),
        "capture_p50_ms": C.pct(lat, 50) * 1000, "capture_p99_ms": C.pct(lat, 99) * 1000,
        "capture_max_ms": max(lat) * 1000, "late_ticks": late[0],
        "rss_start_mb": float(df["rss_mb"].iloc[0]), "rss_after_warmup_mb": rss_after_warm,
        "rss_end_mb": float(df["rss_mb"].iloc[-1]),
        "rss_growth_mb_per_h": rss_slope,
        "rss_growth_pct_per_h": (100 * rss_slope / rss_after_warm) if rss_slope is not None else None,
        "thread_growth_per_h": _slope_per_h(warm["t"], warm["threads"]),
        "fd_growth_per_h": _slope_per_h(warm["t"], warm["fds"]),
        "threads_max": int(df["threads"].max()), "fds_max": int(df["fds"].max()), "tcp_max": int(df["tcp"].max()),
        "analysis_runs": len([a for a in analysis_log if "seconds" in a]),
        "analysis_errors": len([a for a in analysis_log if "error" in a]),
        "analysis_seconds_max": max([a["seconds"] for a in analysis_log if "seconds" in a], default=None),
        "influx_heap_start_mb": df.get("influx_go_memstats_alloc_bytes", pd.Series([np.nan])).iloc[0] / 2**20,
        "influx_heap_end_mb": df.get("influx_go_memstats_alloc_bytes", pd.Series([np.nan])).iloc[-1] / 2**20,
    }
    (ctx.workdir / "analysis_log.json").write_text(json.dumps(analysis_log, indent=1))
    if analysis_log and any("error" in a for a in analysis_log):
        ctx.note("analysis errors: " + "; ".join(a["error"] for a in analysis_log if "error" in a)[:800])

    # limits
    if metrics["rss_growth_pct_per_h"] is not None and metrics["rss_growth_pct_per_h"] > 5:
        ctx.limit(f"Capture process RSS grows {metrics['rss_growth_pct_per_h']:.1f} %/h after warm-up (criterion 5 %/h)",
                  f"RSS {rss_after_warm:.0f} MB after {WARMUP_MIN:.0f} min -> {metrics['rss_end_mb']:.0f} MB at "
                  f"{metrics['duration_h']:.1f} h; slope {rss_slope:.1f} MB/h; samples.csv",
                  "Profile with tracemalloc on the writer and InfluxDB client paths (write_api per write, pandas frames "
                  "kept by the queue); reuse one write_api per Pulse", "major")
    for key, label in (("thread_growth_per_h", "threads"), ("fd_growth_per_h", "file descriptors")):
        if metrics[key] is not None and metrics[key] > 0.5:
            ctx.limit(f"{label} grow {metrics[key]:.2f}/h during the soak", f"max {label} {df['threads' if key.startswith('thread') else 'fds'].max()}; samples.csv",
                      f"Find the {label} leak (client sessions, write_api objects) and close them per write", "major")
    if lost:
        ctx.limit(f"{lost} rows captured but not found in InfluxDB after 8 h ({len(per_model_loss)} models)",
                  f"captured {metrics['captured_rows']}, stored {metrics['stored_rows']}, dropped {metrics['dropped_rows']}, "
                  f"rejected {metrics['rejected_rows']}; per-model sample {dict(list(per_model_loss.items())[:5])}",
                  "Reconcile counters with the store; investigate timestamp collisions or silent partial writes", "critical")
    if metrics["achieved_rows_per_min"] < 0.95 * ROWS_PER_MIN:
        ctx.limit(f"Generator reached {metrics['achieved_rows_per_min']:.0f} rows/min instead of {ROWS_PER_MIN}",
                  f"late ticks {late[0]}; capture p99 {metrics['capture_p99_ms']:.2f} ms",
                  "Capture cost in the caller thread limits the rate on one interpreter", "minor")
    if metrics["capture_p99_ms"] > 2:
        ctx.limit(f"Capture p99 {metrics['capture_p99_ms']:.2f} ms over the soak (target < 2 ms)",
                  f"p50 {metrics['capture_p50_ms']:.2f} ms, max {metrics['capture_max_ms']:.1f} ms; analysis job in the "
                  "same process", "Run analysis out of process; reduce per-capture pandas work", "minor")
    if metrics["agent_stored"] != metrics["agent_captured"]:
        ctx.limit(f"Agent records: {metrics['agent_captured']} captured, {metrics['agent_stored']} stored",
                  f"rejected {metrics['agent_rejected']}; optional field task_success on 70 % of runs",
                  "Check grouping by columns of agent records with optional fields", "major")
    return metrics
