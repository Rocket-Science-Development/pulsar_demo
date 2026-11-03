"""SC-13 - Process crash and restart: kill -9 the capturing process mid-burst (spool on), restart, replay.

A child process captures a burst (rows carry a unique int `seq`) through a local TCP proxy to the platform InfluxDB
(lab-db:8087, dedicated container). It writes its progress (last seq submitted, counters) to a file after every
batch. The parent kills it with SIGKILL, restarts a child on the same spool directory that replays and exits, then
reads every `seq` back: losses = submitted - distinct stored, duplicates = stored points - distinct seq.

Scenarios: (1) storage up; (2) storage down (spooling) at the kill, back for the restart; (3) kill during replay.
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lab.tests import _platform as C

ENV_FILE = Path(C.platform_env_file())
BATCH = 500
RATE = 50_000  # rows/s during the burst
PARAMS = {"batch_rows": BATCH, "burst_rows_per_s": RATE, "kill": "SIGKILL", "spool": True,
          "storage": "lab-db InfluxDB 2.6.1 dedicated container :8087 through a local TCP proxy",
          "repetitions": int(os.environ.get("LAB_REPS", 3)), "write": C.versions(),
          "tuned": 'durability="wal" (write-ahead copy of each batch in the spool)'}
CRITERIA = {"duplicates_total": ("==", 0), "loss_within_queue_all": ("==", True)}


def child(url, bucket, spool, seq0, seconds, progress, replay_only):
    os.environ["LAB_INFLUX_URL_OVERRIDE"] = url
    env = C.influx_env()
    p = C.make_pulse("sc13", bucket, C.capture_options(tuned={"durability": "wal"}, spool_dir=spool, max_retry_delay_s=2.0),
                     env)
    pool = C.frames_pool(BATCH, 20, pool=8)
    fd = os.open(progress, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    seq = seq0
    t0 = time.perf_counter()
    k = 0
    if not replay_only:
        while time.perf_counter() - t0 < seconds:
            x, y = pool[k % len(pool)]
            C.capture(p, x, y, seq_start=seq)
            seq += BATCH
            k += 1
            st = p.stats()
            os.write(fd, (json.dumps({"seq_next": seq, **{k2: st[k2] for k2 in ("captured_rows", "written_rows",
                     "spooled_rows", "replayed_rows", "dropped_rows", "queued_rows")}}) + "\n").encode())
            ahead = (seq - seq0) / RATE - (time.perf_counter() - t0)
            if ahead > 0:
                time.sleep(ahead)
    else:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(0.2)
            st = p.stats()
            os.write(fd, (json.dumps({"seq_next": seq, **{k2: st[k2] for k2 in ("captured_rows", "written_rows",
                     "spooled_rows", "replayed_rows", "dropped_rows", "queued_rows", "spool_files")}}) + "\n").encode())
            if st["spool_files"] == 0:
                break
    p.flush(60)
    st = p.stats()
    p.close()
    os.write(fd, (json.dumps({"final": True, "seq_next": seq, **st}, default=str) + "\n").encode())
    os.close(fd)


def _spool_files(spool):
    """Spooled batches and WAL copies still on disk (.json, .json.gz, .wal, .wal.gz)"""
    if not spool.exists():
        return 0
    return sum(1 for f in spool.iterdir() if ".json" in f.name or ".wal" in f.name)


def _last(progress):
    lines = [l for l in Path(progress).read_text().splitlines() if l.strip()]
    for l in reversed(lines):
        try:
            return json.loads(l)
        except ValueError:
            continue
    return {}


def _spawn(url, bucket, spool, seq0, seconds, progress, replay_only=False):
    return subprocess.Popen([sys.executable, "-m", "lab.tests.sc_13", "child", url, bucket, str(spool), str(seq0),
                             str(seconds), str(progress), str(int(replay_only))], stdout=subprocess.DEVNULL,
                            stderr=open(str(progress) + ".err", "w"))


def run(ctx):
    os.environ["LAB_INFLUX_ENV"] = str(ENV_FILE)
    env = C.influx_env()
    proxy = C.OutageProxy(env["url"], port=18113)
    reps = PARAMS["repetitions"]
    rows = []
    seq0 = 0
    for scenario in ("storage_up", "storage_down", "kill_during_replay"):
        for rep in range(reps):
            name = f"{scenario}_{rep}"
            bucket = C.ensure_bucket(f"lab_sc13_{datetime.now(timezone.utc):%m%d%H%M%S}_{name}", env)
            work = ctx.workdir / name
            work.mkdir(parents=True, exist_ok=True)
            spool = work / "spool"
            prog1 = work / "run1.jsonl"
            kill_after = 4.0 + 3 * rep
            proxy.up()
            if scenario != "storage_up":
                proxy.down()
            p1 = _spawn(proxy.url, bucket, spool, seq0, 60, prog1)
            if scenario != "storage_up":
                time.sleep(2.0)
            time.sleep(kill_after)
            p1.send_signal(signal.SIGKILL)
            p1.wait()
            at_kill = _last(prog1)
            submitted = at_kill.get("seq_next", seq0) - seq0
            spool_files_after_kill = _spool_files(spool)
            tmp_orphans = len(list(spool.glob("*.tmp"))) if spool.exists() else 0
            proxy.up()
            replay_runs = []
            if scenario == "kill_during_replay":
                prog2 = work / "run2.jsonl"
                p2 = _spawn(proxy.url, bucket, spool, seq0 + submitted, 120, prog2, replay_only=True)
                time.sleep(2.5 + rep)  # replay is in progress
                p2.send_signal(signal.SIGKILL)
                p2.wait()
                replay_runs.append(_last(prog2))
            prog3 = work / "run3.jsonl"
            p3 = _spawn(proxy.url, bucket, spool, seq0 + submitted, 300, prog3, replay_only=True)
            p3.wait(timeout=400)
            replay_runs.append(_last(prog3))
            seqs = C.seq_values(bucket, "sc13_1_input_data", env)
            distinct = len(np.unique(seqs))
            expected = np.arange(seq0, seq0 + submitted)
            present = np.isin(expected, seqs)
            lost = int((~present).sum())
            replayed_total = sum(r.get("replayed_rows", 0) for r in replay_runs)
            row = {"scenario": scenario, "rep": rep, "kill_after_s": kill_after, "submitted": int(submitted),
                   "stored_points": int(len(seqs)), "distinct_seq": distinct, "lost": lost,
                   "duplicates": int(len(seqs) - distinct),
                   # the batch captured between the last progress line and the kill (durable under WAL) is not extra
                   "captured_after_progress": int(((seqs >= seq0 + submitted) & (seqs < seq0 + submitted + BATCH)).sum()),
                   "extra_seq": int((~np.isin(seqs, np.arange(seq0, seq0 + submitted + BATCH))).sum()),
                   "queued_at_kill": at_kill.get("queued_rows"), "written_at_kill": at_kill.get("written_rows"),
                   "spooled_at_kill": at_kill.get("spooled_rows"), "spool_files_after_kill": spool_files_after_kill,
                   "tmp_orphans": tmp_orphans, "tmp_left_at_end": len(list(spool.glob("*.tmp"))) if spool.exists() else 0,
                   "replayed_rows_reported": replayed_total,
                   "spool_files_end": _spool_files(spool),
                   "lost_contiguous_tail": bool(lost == 0 or (~present)[-lost:].all())}
            # in-memory queue at the kill = queued rows + the batch being written (at most max_batch_rows) + last batches
            # submitted after the last progress line (none: progress is written after every capture)
            row["dropped_before_kill"] = at_kill.get("dropped_rows") or 0  # queue overflow, counted by Pulse
            row["lost_by_crash"] = lost - row["dropped_before_kill"]
            row["loss_bound"] = (row["queued_at_kill"] or 0) + 5_000 + BATCH
            row["loss_within_queue"] = row["lost_by_crash"] <= row["loss_bound"]
            row["replay_overcount"] = int(replayed_total - max(0, distinct - (row["written_at_kill"] or 0)))
            rows.append(row)
            seq0 += 10_000_000
    proxy.down()
    (ctx.workdir / "scenarios.json").write_text(json.dumps(rows, indent=1))

    def agg(sc, key, fn=max):
        vals = [r[key] for r in rows if r["scenario"] == sc]
        return fn(vals) if vals else None

    metrics = {"duplicates_total": sum(r["duplicates"] for r in rows),
               "loss_within_queue_all": all(r["loss_within_queue"] for r in rows),
               "lost_rows_total": sum(r["lost"] for r in rows),
               "lost_by_crash_total": sum(r["lost_by_crash"] for r in rows), "submitted_total": sum(r["submitted"] for r in rows),
               "tmp_orphans_total": sum(r["tmp_left_at_end"] for r in rows),
               "per_scenario": {sc: {"lost_max": agg(sc, "lost"), "lost_mean": agg(sc, "lost", lambda v: sum(v) / len(v)),
                                     "queued_at_kill_max": agg(sc, "queued_at_kill"),
                                     "duplicates_max": agg(sc, "duplicates"),
                                     "replay_overcount_max": agg(sc, "replay_overcount")}
                                for sc in ("storage_up", "storage_down", "kill_during_replay")}}
    for sc in ("storage_up", "storage_down", "kill_during_replay"):
        rs = [r for r in rows if r["scenario"] == sc]
        if any(r["lost_by_crash"] for r in rs):
            ctx.limit(f"[{sc}] kill -9 loses up to {max(r['lost_by_crash'] for r in rs)} rows "
                      f"({max(r['lost_by_crash'] / max(r['submitted'], 1) for r in rs):.1%} of the burst) held in memory",
                      json.dumps([{k: r[k] for k in ('submitted', 'lost', 'dropped_before_kill', 'lost_by_crash',
                                                     'queued_at_kill', 'spooled_at_kill', 'lost_contiguous_tail')}
                                  for r in rs]),
                      "Document the loss window (max_queue_rows + one batch); offer an optional write-ahead spool "
                      "(append every batch to the spool before queueing, delete after write) for no-loss deployments",
                      "major" if sc != "storage_down" else "minor")
        if any(r["dropped_before_kill"] for r in rs):
            ctx.limit(f"[{sc}] at {RATE:,} rows/s one Pulse overflows its queue before the crash: up to "
                      f"{max(r['dropped_before_kill'] for r in rs)} rows dropped (counted) in "
                      f"{max(r['kill_after_s'] for r in rs):.0f} s",
                      json.dumps([{k: r[k] for k in ('kill_after_s', 'submitted', 'written_at_kill', 'queued_at_kill',
                                                     'dropped_before_kill')} for r in rs]),
                      "Writer throughput of one Pulse is below 50k rows/s: see SC-02 (parallel writes)", "minor")
        if any(not r["loss_within_queue"] for r in rs):
            ctx.limit(f"[{sc}] losses exceed the in-memory queue at the kill", json.dumps(rs)[:600],
                      "Investigate rows lost outside the queue (spool files or writes lost)", "critical")
        if any(r["duplicates"] or r["extra_seq"] for r in rs):
            ctx.limit(f"[{sc}] duplicates after restart", json.dumps(rs)[:600],
                      "Make replay idempotent (unique timestamps are kept in the spool)", "critical")
        if any(r["replay_overcount"] > 0 for r in rs):
            ctx.limit(f"[{sc}] replayed_rows counts a spool batch twice when the process dies between the write and "
                      f"the unlink (overcount up to {max(r['replay_overcount'] for r in rs)} rows); InfluxDB absorbs it "
                      "because timestamps are kept, so no duplicate is stored",
                      json.dumps([{k: r[k] for k in ('replayed_rows_reported', 'distinct_seq', 'written_at_kill')}
                                  for r in rs]),
                      "Rename the spool file to .inflight before the write, and reconcile counters on startup", "minor")
    if metrics["tmp_orphans_total"]:
        ctx.limit(f"{metrics['tmp_orphans_total']} half-written spool .tmp files left after kill -9 and never cleaned",
                  json.dumps([r["tmp_left_at_end"] for r in rows]),
                  "Spool: remove or recover *.tmp files at startup", "minor")
    return metrics


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "child":
    a = sys.argv
    child(a[2], a[3], a[4], int(a[5]), float(a[6]), a[7], bool(int(a[8])))
