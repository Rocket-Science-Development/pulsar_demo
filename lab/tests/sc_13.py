"""SC-13 - Process crash and restart: kill -9 the capturing process mid-burst (spool on), restart, replay.

A child process captures a burst (rows carry a unique int `seq`) through a local TCP proxy to the platform InfluxDB
(lab-db:8087, dedicated container). It writes its progress (last seq submitted, counters) to a file after every
batch. The parent kills it with SIGKILL, restarts a child on the same spool directory that replays and exits, then
reads every `seq` back: losses = submitted - distinct stored, duplicates = stored points - distinct seq.

Scenarios: (1) storage up; (2) storage down (spooling) at the kill, back for the restart; (3) kill during replay.

Each loss is attributed to the kill that caused it: after every kill, the rows still recoverable (stored, or in a spool
file) are read back, so that a row missing after the first kill (storage down, in scenario 3) is not blamed on the
kill during replay.
"""

import gzip
import json
import os
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


def _spool_seqs(spool):
    """`seq` of every row held by the spool directory (spooled batches, write-ahead copies, complete .tmp files), read
    without opening a Spool (which would recover the leftovers); unreadable files are not recoverable"""
    from pulsar_data_collection.spool import frame_from_json

    out = []
    if spool.exists():
        for f in spool.iterdir():
            if not any(s in f.name for s in (".json", ".wal", ".tmp")):
                continue
            try:
                data = f.read_bytes()
                frame, _ = frame_from_json((gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data).decode("utf-8"))
            except Exception:  # noqa: BLE001 - half-written file
                continue
            if "seq" in frame:
                out.append(frame["seq"].to_numpy(dtype="int64"))
    return np.concatenate(out) if out else np.array([], dtype="int64")


def _recoverable(bucket, spool, env, expected):
    """Rows of `expected` stored or still in the spool, right after a kill"""
    time.sleep(1.0)  # a request fully received before the kill may still be committed
    held = np.concatenate([C.seq_values(bucket, "sc13_1_input_data", env), _spool_seqs(spool)])
    return np.isin(expected, held)


def _last(progress):
    if not Path(progress).exists():  # killed before its first progress line (slow start)
        return {}
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
            C.wait_until(prog1.exists, 60, 0.05)  # timed from the first batch: process start-up varies with the host
            if scenario != "storage_up":
                time.sleep(2.0)
            time.sleep(kill_after)
            p1.kill()  # SIGKILL (TerminateProcess on Windows)
            p1.wait()
            at_kill = _last(prog1)
            submitted = at_kill.get("seq_next", seq0) - seq0
            spool_files_after_kill = _spool_files(spool)
            tmp_orphans = len(list(spool.glob("*.tmp"))) if spool.exists() else 0
            expected = np.arange(seq0, seq0 + submitted)
            after_kill1 = _recoverable(bucket, spool, env, expected)
            proxy.up()
            replay_runs = []
            after_kill2 = None
            if scenario == "kill_during_replay":
                prog2 = work / "run2.jsonl"
                p2 = _spawn(proxy.url, bucket, spool, seq0 + submitted, 120, prog2, replay_only=True)
                C.wait_until(prog2.exists, 60, 0.05)
                time.sleep(2.5 + rep)  # replay is in progress
                p2.kill()
                p2.wait()
                replay_runs.append(_last(prog2))
                after_kill2 = _recoverable(bucket, spool, env, expected)
            prog3 = work / "run3.jsonl"
            p3 = _spawn(proxy.url, bucket, spool, seq0 + submitted, 300, prog3, replay_only=True)
            p3.wait(timeout=400)
            replay_runs.append(_last(prog3))
            seqs = C.seq_values(bucket, "sc13_1_input_data", env)
            distinct = len(np.unique(seqs))
            present = np.isin(expected, seqs)
            lost = int((~present).sum())
            before_restart = after_kill1 if after_kill2 is None else after_kill2
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
            # attribution of every row missing at the end: not recoverable right after the first kill, recoverable
            # after it but not after the kill during replay, or recoverable before the last restart (the three add up
            # to `lost`); a request received before a kill can still be committed later (stored_late, not lost)
            row["lost_at_first_kill"] = int((~after_kill1 & ~present).sum())
            row["lost_by_replay_kill"] = (int((after_kill1 & ~after_kill2 & ~present).sum())
                                          if after_kill2 is not None else None)
            row["lost_after_restart"] = int((before_restart & ~present).sum())
            row["stored_late"] = int((~before_restart & present).sum())
            row["first_kill_storage"] = "up" if scenario == "storage_up" else "down"
            row["lost_by_crash"] = row["lost_at_first_kill"] - row["dropped_before_kill"]  # first kill
            row["loss_bound"] = (row["queued_at_kill"] or 0) + 5_000 + BATCH
            # spooled rows must survive any later kill and the restart
            row["loss_within_queue"] = (row["lost_by_crash"] <= row["loss_bound"] and not row["lost_by_replay_kill"]
                                        and not row["lost_after_restart"])
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
               "lost_by_crash_total": sum(r["lost_by_crash"] for r in rows),
               "lost_by_replay_kill_total": sum(r["lost_by_replay_kill"] or 0 for r in rows),
               "lost_after_restart_total": sum(r["lost_after_restart"] for r in rows),
               "submitted_total": sum(r["submitted"] for r in rows),
               "tmp_orphans_total": sum(r["tmp_left_at_end"] for r in rows),
               "per_scenario": {sc: {"lost_max": agg(sc, "lost"), "lost_mean": agg(sc, "lost", lambda v: sum(v) / len(v)),
                                     "lost_at_first_kill_max": agg(sc, "lost_at_first_kill"),
                                     "lost_by_replay_kill_max": agg(sc, "lost_by_replay_kill",
                                                                    lambda v: max(v) if None not in v else None),
                                     "queued_at_kill_max": agg(sc, "queued_at_kill"),
                                     "duplicates_max": agg(sc, "duplicates"),
                                     "replay_overcount_max": agg(sc, "replay_overcount")}
                                for sc in ("storage_up", "storage_down", "kill_during_replay")}}
    # losses by the kill that caused them: the first kill of storage_down and kill_during_replay happens with the
    # storage down; only the second kill of kill_during_replay happens during the replay
    for storage in ("up", "down"):
        rs = [r for r in rows if r["first_kill_storage"] == storage]
        if any(r["lost_by_crash"] for r in rs):
            ctx.limit(f"[kill with storage {storage}] kill -9 loses up to {max(r['lost_by_crash'] for r in rs)} rows "
                      f"({max(r['lost_by_crash'] / max(r['submitted'], 1) for r in rs):.1%} of the burst) held in memory",
                      json.dumps([{k: r[k] for k in ('scenario', 'rep', 'submitted', 'lost', 'dropped_before_kill',
                                                     'lost_by_crash', 'queued_at_kill', 'spooled_at_kill',
                                                     'lost_contiguous_tail')} for r in rs]),
                      "Document the loss window (max_queue_rows + one batch); offer an optional write-ahead spool "
                      "(append every batch to the spool before queueing, delete after write) for no-loss deployments",
                      "major" if storage == "up" else "minor")
    rs = [r for r in rows if r["lost_by_replay_kill"]]
    if rs:
        ctx.limit(f"[kill during replay] kill -9 during the replay loses up to "
                  f"{max(r['lost_by_replay_kill'] for r in rs)} spooled rows",
                  json.dumps([{k: r[k] for k in ('rep', 'submitted', 'lost_at_first_kill', 'lost_by_replay_kill',
                                                 'lost_after_restart', 'spool_files_end')} for r in rs]),
                  "Delete a spool file only once its rows are acknowledged by the storage", "critical")
    for sc in ("storage_up", "storage_down", "kill_during_replay"):
        rs = [r for r in rows if r["scenario"] == sc]
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
