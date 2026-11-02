"""SC-12 - Hardware and software health telemetry across the platform tests.

1) every SC result has its health trace (harness HealthSampler: CPU, RSS, threads, descriptors, disk);
2) host traces (whole host, docker stats, network) of lab-scale, lab-soak and lab-db exist (lab.tests._hostmon);
3) everything goes to the lab InfluxDB bucket `lab_health` (measurements lab_host, lab_test_health) and Pulse counters
   through Pulsar's own ResultsSink / StatsReporter;
4) StatsReporter cost at 100 Pulse (threads, write errors);
5) alert rules evaluated: CPU > 90 % (3 consecutive samples), disk > 80 %, RSS growth > 5 %/h, InfluxDB container memory.
"""

import csv
import glob
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from lab.tests import _platform as C

PARAMS = {"bucket": "lab_health", "alert_rules": {"cpu_pct": 90, "cpu_consecutive": 3, "disk_pct": 80,
                                                  "rss_growth_pct_per_h": 5}}
CRITERIA = {"tests_with_health_trace_share": ("==", 1.0), "hosts_with_trace": (">=", 3), "points_written": (">", 0)}


def _read_csv(path):
    rows = []
    with open(path) as fh:
        header = None
        for rec in csv.reader(fh):
            if rec and rec[0] == "t":
                header = rec
                continue
            if header and len(rec) == len(header):
                row = {}
                for k, v in zip(header, rec):
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = None
                rows.append(row)
    return rows


def _own_trace(workdir: Path) -> int:
    """Tests that sample themselves (SC-11 samples.csv) count as traced"""
    f = workdir / "samples.csv"
    return max(0, sum(1 for _ in open(f)) - 1) if f.exists() else 0


def run(ctx):
    from influxdb_client import Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS

    from pulsar_data_collection.models import CaptureOptions
    from pulsar_data_collection.results import ResultsSink, StatsReporter

    results_dir = ctx.workdir.parent
    env = C.influx_env()  # main lab InfluxDB (lab-db:8086)
    bucket = C.ensure_bucket("lab_health", env)

    # 1) per-test health traces
    tests = {}
    for f in sorted(glob.glob(str(results_dir / "SC-*.json"))):
        r = json.loads(Path(f).read_text())
        if r.get("id") == "SC-12":
            continue
        h = r.get("health") or {}
        tests[r["id"]] = {"status": r.get("status"), "seconds": r.get("seconds"), "host": r.get("host"),
                          "samples": h.get("samples", 0) or _own_trace(results_dir / r["id"]), **{k: h.get(k) for k in ("cpu_max", "mem_max", "rss_mb_max",
                                                                                  "threads_max", "fds_max", "disk_max",
                                                                                  "rss_growth_mb_per_h")}}
    expected = ["SC-01", "SC-02", "SC-03", "SC-05", "SC-06", "SC-11", "SC-13", "SC-14"]
    missing = [t for t in expected if t not in tests]
    with_trace = [t for t, v in tests.items() if v["samples"]]

    # 2) host traces
    hosts = {}
    for f in sorted(glob.glob(str(ctx.workdir / "host_*.csv"))):
        hosts[Path(f).stem.replace("host_", "")] = _read_csv(f)

    # 3) write to lab_health
    points = 0
    with C.client(env) as c:
        w = c.write_api(write_options=SYNCHRONOUS)
        for host, rows in hosts.items():
            batch = []
            for row in rows:
                p = Point("lab_host").tag("host", host).time(int(row["t"] * 1e9), WritePrecision.NS)
                for k, v in row.items():
                    if k != "t" and v is not None:
                        p.field(k, float(v))
                batch.append(p)
                if len(batch) >= 5000:
                    w.write(bucket=bucket, record=batch)
                    points += len(batch)
                    batch = []
            if batch:
                w.write(bucket=bucket, record=batch)
                points += len(batch)
        now = datetime.now(timezone.utc)
        for tid, v in tests.items():
            p = Point("lab_test_health").tag("test", tid).tag("status", v["status"] or "").time(now)
            for k, val in v.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool) and val is not None:
                    p.field(k, float(val))
            w.write(bucket=bucket, record=p)
            points += 1

    # 4) StatsReporter at 100 Pulse (storage replaced by a no-op so only the reporting path is measured)
    proc = psutil.Process()
    base_threads = proc.num_threads()
    unreachable = {"url": "http://127.0.0.1:9", "token": "none", "org": "lab"}
    pulses = [C.make_pulse(f"hb{i:03d}", "none", CaptureOptions(), unreachable) for i in range(100)]
    for p in pulses:
        p.storage_engine.write_frame = lambda *a, **k: None
    t_pulses = proc.num_threads()
    stat_points0 = C.count_rows(bucket, "captured_rows", None, env)
    sink = ResultsSink(url=env["url"], token=env["token"], org=env["org"], bucket=bucket)
    cpu0 = sum(proc.cpu_times()[:2])
    reporters = [StatsReporter(p, sink, interval_s=1.0, tags={"campaign": "lab-sc12"}).start() for p in pulses]
    t_rep = proc.num_threads()
    pool = C.frames_pool(50, 20, pool=8)
    t0 = time.time()
    k = 0
    while time.time() - t0 < 30:
        x, y = pool[k % len(pool)]
        C.capture(pulses[k % 100], x, y)
        k += 1
        time.sleep(0.002)
    for r in reporters:
        r.stop()
    rep_cpu = sum(proc.cpu_times()[:2]) - cpu0
    rep_errors = sum(r.errors for r in reporters)
    sink.close()
    for p in pulses:
        p.close(5)
    stat_points = C.count_rows(bucket, "captured_rows", None, env) - stat_points0
    stats_keys = sorted(pulses[0].stats().keys())

    # 5) alert rules
    rules = PARAMS["alert_rules"]
    alerts = []
    for host, rows in hosts.items():
        streak = 0
        for row in rows:
            streak = streak + 1 if (row.get("cpu_pct") or 0) > rules["cpu_pct"] else 0
            if streak == rules["cpu_consecutive"]:
                alerts.append({"host": host, "rule": "cpu>90% x3", "t": row["t"], "value": row["cpu_pct"]})
            if (row.get("disk_pct") or 0) > rules["disk_pct"]:
                alerts.append({"host": host, "rule": "disk>80%", "t": row["t"], "value": row["disk_pct"]})
                break
        mem = [(row["t"], row[k]) for row in rows for k in row if k.startswith("docker_influxdb") and k.endswith("_mem_mb") and row[k]]
        if mem:
            ctx.note(f"{host}: InfluxDB container memory {min(v for _, v in mem):.0f} -> {max(v for _, v in mem):.0f} MB")
    for tid, v in tests.items():
        if v.get("rss_growth_mb_per_h") and v.get("rss_mb_max") and v["seconds"] and v["seconds"] > 1800:
            pct = 100 * v["rss_growth_mb_per_h"] / v["rss_mb_max"]
            if pct > rules["rss_growth_pct_per_h"]:
                alerts.append({"test": tid, "rule": "rss growth > 5 %/h", "value": pct})
    (ctx.workdir / "alerts.json").write_text(json.dumps(alerts, indent=1))
    (ctx.workdir / "tests_health.json").write_text(json.dumps(tests, indent=1))

    metrics = {"tests_found": sorted(tests), "tests_missing": missing,
               "tests_with_health_trace_share": len(with_trace) / max(len(tests), 1),
               "hosts_with_trace": len(hosts), "host_samples": {h: len(r) for h, r in hosts.items()},
               "points_written": points, "stats_points_written": stat_points, "alerts": len(alerts),
               "alerts_by_rule": {r: sum(1 for a in alerts if a["rule"] == r) for r in {a["rule"] for a in alerts}},
               "statsreporter_threads_added": t_rep - t_pulses, "pulse_threads_added": t_pulses - base_threads,
               "statsreporter_cpu_s_30s": rep_cpu, "statsreporter_errors": rep_errors, "pulse_stats_keys": stats_keys}
    if t_rep - t_pulses >= 100:
        ctx.limit(f"StatsReporter starts one thread per Pulse: +{t_rep - t_pulses} threads for 100 Pulse (on top of "
                  f"+{t_pulses - base_threads} writer threads), one point per Pulse per interval",
                  f"{stat_points} stats points in 30 s at interval 1 s; {rep_errors} report errors; CPU {rep_cpu:.1f} s",
                  "One reporter for all the Pulse of a process (registry), writing one batch of points per interval",
                  "minor")
    host_keys = {"rss", "cpu", "threads", "fds", "memory"}
    if not any(any(h in k for h in host_keys) for k in stats_keys):
        ctx.limit("Pulsar exports no process health: Pulse.stats() / StatsReporter carry only capture counters, so CPU, "
                  "RSS, threads and descriptors of the capturing process must come from outside (the lab used psutil)",
                  f"stats keys: {stats_keys}",
                  "Add optional process gauges (RSS, threads, open files, writer lag, last write latency) to "
                  "StatsReporter, and a 'Capture health' Grafana dashboard with alerts", "minor")
    if missing:
        ctx.note(f"tests without a result file at SC-12 time: {missing}")
    ctx.note("harness HealthSampler samples the harness process only; child processes (SC-05, SC-06, SC-13) are covered "
             "by the host traces")
    return metrics
