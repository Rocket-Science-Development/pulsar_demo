"""SC-01 - Many models per instance: one process hosts N Pulse (N = 1..500), each capturing 10 batches/s of
50 rows x 20 features into the platform InfluxDB (lab-db:8087, dedicated container).

Per N: startup time, caller-side capture latency p50/p99, achieved vs target rate, rows captured / written / dropped /
stored (queried back), queue depth, RSS, threads, descriptors, TCP sockets, CPU; then close() and check that threads
and sockets are released.

Server side: the InfluxDB descriptors are measured (process_open_fds / process_max_fds from its /metrics, and with
LAB_INFLUX_CONTAINER set on the InfluxDB host, /proc/1/fd and /proc/1/limits of the container and "too many open
files" in its logs; InfluxDB 2.6 exports no process_open_fds, so run SC-01 where Docker reaches the container).
A descriptor limit is recorded only with that evidence (peak >= 90 % of the limit, or the log line), never assumed.
"""

import gc
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from lab.tests import _platform as C

LEVELS = [int(v) for v in os.environ.get("LAB_LEVELS", "1,10,50,100,300,500").split(",")]
STEP_S = float(os.environ.get("LAB_STEP_S", 60))
BATCH, FEATURES, BATCHES_PER_S = 50, 20, 10
PARAMS = {"levels": LEVELS, "step_s": STEP_S, "batch_rows": BATCH, "features": FEATURES, "batches_per_s_per_model": 10,
          "generator": "one thread, round robin, pre-built frames", "storage": "lab-db InfluxDB 2.6.1 :8087 (dedicated)"}
CRITERIA = {"capture_p99_ms_max": ("<", 2.0), "lost_rows_upto_300": ("==", 0), "rss_mb_per_model_max": ("<", 30.0)}


def run(ctx):
    from pulsar_data_collection.models import CaptureOptions

    os.environ["LAB_INFLUX_ENV"] = str(Path.home() / ".pulsar-lab" / "influx2.env")
    env = C.influx_env()
    proc = psutil.Process()
    pool = C.frames_pool(BATCH, FEATURES, pool=64)
    steps = []
    for n in LEVELS:
        gc.collect()
        bucket = C.ensure_bucket(f"lab_sc01_{datetime.now(timezone.utc):%m%d%H%M}_n{n}", env)
        before = C.proc_snapshot(proc)
        t0 = time.perf_counter()
        pulses = [C.make_pulse(f"m{i:03d}", bucket, CaptureOptions(), env) for i in range(n)]
        startup_s = time.perf_counter() - t0
        created = C.proc_snapshot(proc)
        lat = []
        target = n * BATCHES_PER_S
        stop = threading.Event()
        peak = {"queued": 0, "rss_mb": 0, "threads": 0, "fds": 0, "tcp": 0}
        server = {"open_fds_peak": None, "max_fds": None, "container_fds_peak": None, "samples": 0}
        server_before = C.influx_fds(env)
        server["max_fds"] = server_before.get("process_max_fds", server_before.get("container_max_fds"))

        def sampler():
            tick = 0
            while not stop.wait(2):
                s = C.proc_snapshot(proc)
                q = sum(p.stats()["queued_rows"] for p in pulses)
                peak.update({"queued": max(peak["queued"], q), **{k: max(peak[k], s[k]) for k in ("rss_mb", "threads", "fds", "tcp")}})
                tick += 1
                if tick % 5 == 1:  # server descriptors every 10 s
                    f = C.influx_fds(env)
                    server["samples"] += 1
                    for src, dst in (("process_open_fds", "open_fds_peak"), ("container_fds", "container_fds_peak")):
                        if src in f:
                            server[dst] = max(server[dst] or 0, f[src])
                    server["max_fds"] = f.get("process_max_fds", f.get("container_max_fds", server["max_fds"]))

        th = threading.Thread(target=sampler, daemon=True)
        th.start()
        cpu0 = sum(proc.cpu_times()[:2])
        start = time.perf_counter()
        k = 0
        step = 1.0 / target
        while time.perf_counter() - start < STEP_S:
            x, y = pool[k % len(pool)]
            s = time.perf_counter()
            C.capture(pulses[k % n], x, y)
            lat.append(time.perf_counter() - s)
            k += 1
            ahead = k * step - (time.perf_counter() - start)
            if ahead > 0:
                time.sleep(ahead)
        elapsed = time.perf_counter() - start
        cpu_s = sum(proc.cpu_times()[:2]) - cpu0
        steady = C.proc_snapshot(proc)
        t_flush = time.perf_counter()
        flushed = all(p.flush(300) for p in pulses)
        flush_s = time.perf_counter() - t_flush
        stop.set()
        th.join()
        server_after = C.influx_fds(env)
        stats = [p.stats() for p in pulses]
        for p in pulses:
            p.close()
        del pulses
        gc.collect()
        time.sleep(2)
        closed = C.proc_snapshot(proc)
        stored = sum(C.count_by_measurement(bucket, "f0", env).values())
        captured = sum(s["captured_rows"] for s in stats)
        row = {"n": n, "startup_s": startup_s, "startup_ms_per_model": 1000 * startup_s / n,
               "target_batches_per_s": target, "achieved_batches_per_s": k / elapsed,
               "rate_ratio": (k / elapsed) / target,
               "capture_p50_ms": C.pct(lat, 50) * 1000, "capture_p99_ms": C.pct(lat, 99) * 1000,
               "capture_max_ms": max(lat) * 1000, "captured": captured,
               "written": sum(s["written_rows"] for s in stats), "dropped": sum(s["dropped_rows"] for s in stats),
               "write_errors": sum(s["write_errors"] for s in stats), "stored": stored, "lost": captured - stored,
               "rows_written_per_s": stored / (elapsed + flush_s), "flush_s": flush_s, "flushed": flushed,
               "peak_queued_rows": peak["queued"], "cpu_pct_of_one_core": 100 * cpu_s / elapsed,
               "rss_before_mb": before["rss_mb"], "rss_created_mb": created["rss_mb"], "rss_steady_mb": steady["rss_mb"],
               "rss_peak_mb": peak["rss_mb"], "rss_mb_per_model": (max(peak["rss_mb"], steady["rss_mb"]) - before["rss_mb"]) / n,
               "threads_before": before["threads"], "threads_steady": steady["threads"], "threads_peak": peak["threads"],
               "fds_steady": steady["fds"], "tcp_steady": steady["tcp"], "tcp_peak": peak["tcp"],
               "threads_after_close": closed["threads"], "tcp_after_close": closed["tcp"], "fds_after_close": closed["fds"],
               "rss_after_close_mb": closed["rss_mb"],
               "influx_open_fds_before": server_before.get("process_open_fds"),
               "influx_open_fds_peak": server["open_fds_peak"], "influx_max_fds": server["max_fds"],
               "influx_container_fds_peak": server["container_fds_peak"],
               "influx_too_many_open_files_lines": (server_after.get("too_many_open_files_log_lines", 0)
                                                    - server_before.get("too_many_open_files_log_lines", 0))
               if "too_many_open_files_log_lines" in server_after else None,
               "influx_fd_samples": server["samples"]}
        steps.append(row)
        (ctx.workdir / "steps.json").write_text(json.dumps(steps, indent=1))
        ctx.note(f"N={n}: p99 {row['capture_p99_ms']:.2f} ms, rate {row['rate_ratio']:.2f}, lost {row['lost']}, "
                 f"threads {row['threads_steady']}, tcp {row['tcp_steady']}, RSS/model {row['rss_mb_per_model']:.2f} MB")

    metrics = {"steps": {str(r["n"]): {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in steps},
               "capture_p99_ms_max": max(r["capture_p99_ms"] for r in steps),
               "lost_rows_upto_300": sum(r["lost"] for r in steps if r["n"] <= 300),
               "rss_mb_per_model_max": max(r["rss_mb_per_model"] for r in steps if r["n"] >= 10) if any(r["n"] >= 10 for r in steps) else None}
    for r in steps:
        if r["capture_p99_ms"] >= 2:
            ctx.limit(f"Capture p99 {r['capture_p99_ms']:.2f} ms at N={r['n']} models in one process (target < 2 ms)",
                      f"p50 {r['capture_p50_ms']:.2f} ms, max {r['capture_max_ms']:.1f} ms; {r['threads_steady']} threads, "
                      f"CPU {r['cpu_pct_of_one_core']:.0f} % of one core",
                      "Share one writer thread and one HTTP client across the Pulse of a process (a capture hub keyed by "
                      "model), so caller threads do not compete with N writer threads for the GIL", "major")
        if r["rate_ratio"] < 0.95:
            ctx.limit(f"One process cannot sustain {r['target_batches_per_s']} captures/s at N={r['n']}: "
                      f"{r['achieved_batches_per_s']:.0f}/s ({r['rate_ratio']:.0%})",
                      f"capture p50 {r['capture_p50_ms']:.2f} ms is CPU in the caller (prepare_frame: pd.concat + "
                      f"date_range + Pydantic model) -> about {1000 / max(r['capture_p50_ms'], 1e-3):.0f} captures/s per core",
                      "Make capture_data cheaper (append raw arrays to the queue, build the DataFrame in the writer thread); "
                      "document the per-process ceiling and scale out with processes", "major")
        if r["lost"] and r["n"] <= 300:
            ctx.limit(f"{r['lost']} rows lost at N={r['n']}",
                      f"captured {r['captured']}, written {r['written']}, dropped {r['dropped']}, stored {r['stored']}, "
                      f"write errors {r['write_errors']}", "Raise writer throughput or bound per-process models", "critical")
        overcount = r["dropped"] - (r["captured"] - r["stored"])
        r["dropped_but_stored"] = overcount
        if overcount > 0:
            ctx.limit(f"At N={r['n']}, {overcount} rows counted as dropped were in fact stored: write timeouts with "
                      "unknown outcome are counted as failures",
                      f"captured {r['captured']}, written {r['written']}, dropped {r['dropped']}, stored {r['stored']}, "
                      f"{r['write_errors']} write errors (read timeout 5 s)",
                      "Count timeouts separately (unknown_rows) and make writes idempotent so they can be retried or "
                      "spooled; the counters must not claim a loss that did not happen", "major")
        if r["lost"] > 0 and r["n"] > 300:
            ctx.limit(f"{r['lost']} rows lost at N={r['n']} models ({r['lost'] / r['captured']:.1%}), "
                      f"{r['write_errors']} write timeouts",
                      f"captured {r['captured']}, stored {r['stored']}, dropped {r['dropped']}; {r['threads_steady']} "
                      f"threads, process CPU {r['cpu_pct_of_one_core']:.0f} % of one core: writer threads starve on the GIL "
                      "and InfluxDB sees one connection per model",
                      "Shared writer per process; spool on timeout (enable spool_dir by default for servers)", "major")
        if r["threads_after_close"] > r["threads_before"] + 2 or r["tcp_after_close"] > 2:
            ctx.limit(f"close() does not release resources at N={r['n']}: {r['threads_after_close']} threads "
                      f"(before {r['threads_before']}), {r['tcp_after_close']} TCP sockets after close",
                      json.dumps({k: r[k] for k in ("threads_before", "threads_steady", "threads_after_close",
                                                    "tcp_steady", "tcp_after_close")}),
                      "Close the InfluxDB client pool and join the writer thread in Pulse.close()", "minor")
    # InfluxDB descriptor exhaustion: only with measured evidence (refuted at the night's run for lack of it)
    measured = [r for r in steps if r["influx_open_fds_peak"] is not None or r["influx_container_fds_peak"] is not None]
    if not measured:
        ctx.note("InfluxDB descriptors not measured (no process_open_fds in /metrics and no LAB_INFLUX_CONTAINER): "
                 "no conclusion on server-side descriptor exhaustion")
    for r in measured:
        fds = max(v for v in (r["influx_open_fds_peak"], r["influx_container_fds_peak"]) if v is not None)
        near = r["influx_max_fds"] and fds >= 0.9 * r["influx_max_fds"]
        if near or (r["influx_too_many_open_files_lines"] or 0) > 0:
            ctx.limit(f"InfluxDB runs out of file descriptors at N={r['n']} Pulse (one TCP connection each): "
                      f"{fds:.0f} open of a {r['influx_max_fds']} limit",
                      json.dumps({k: r[k] for k in ("influx_open_fds_before", "influx_open_fds_peak", "influx_max_fds",
                                                    "influx_container_fds_peak", "influx_too_many_open_files_lines",
                                                    "tcp_steady")}),
                      "Document `ulimits: nofile` for InfluxDB in the compose files; a shared writer per process "
                      "(one connection pool for all the Pulse) keeps the server-side count independent of N", "major")
    big = [r for r in steps if r["n"] >= 100]
    if big:
        r = max(big, key=lambda r: r["n"])
    if big and r["threads_steady"] >= r["n"]:
        ctx.limit(f"One writer thread and one HTTP connection pool per Pulse: {r['threads_steady']} threads and "
                  f"{r['tcp_steady']} TCP sockets at N={r['n']}, startup {r['startup_s']:.1f} s "
                  f"({r['startup_ms_per_model']:.0f} ms/model for the bucket check)",
                  json.dumps({str(x['n']): [x['threads_steady'], x['tcp_steady'], round(x['startup_s'], 2)] for x in steps}),
                  "Offer a shared writer (PulseHub) for multi-model servers: one thread, one client, batching across "
                  "models; check the bucket once per process", "minor")
    return metrics
