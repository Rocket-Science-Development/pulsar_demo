"""SC-04 - Degraded network between Pulse and InfluxDB (latency, loss, bandwidth, flapping link), through Toxiproxy.

10 models share 120k rows/min (batches of 50 rows, rows carry a unique int `seq`) into an InfluxDB 2.6.1 container
through a Toxiproxy 2.x proxy (both containers dedicated to this test). Every case runs PRE_S clean, TOXIC_S under one
network fault, then POST_S clean, then waits for the queues (and the spool) to drain; it is repeated with spool_dir and
without, in the write profile of LAB_PROFILE (default: one private writer per Pulse; tuned: one SharedWriter(concurrency=4)).

Network faults (Toxiproxy has no packet loss: it is a TCP proxy, so the faults are the ones TCP shows to the
application):
- lat200 / lat1000: `latency` toxic on the answers (downstream), i.e. +200 ms / +1 s per request round trip;
- loss5 / loss20: lossy link as TCP sees it. TCP retransmits lost segments, so the application sees stalls, not lost
  bytes: every 250 ms slot is stalled with probability p (`timeout` toxic, timeout=0, both directions: data held, then
  released when the slot ends; a run of k lost slots stalls k x 250 ms, without TCP's doubling of the retransmission
  timeout), and each new TCP connection is reset by the peer with probability p (`reset_peer`, toxicity=p, after 1 s);
- bw1mbit: `bandwidth` toxic, both directions. Toxiproxy limits each TCP connection, not the link: the rate per
  connection is 125 KB/s (1 Mbit/s) divided by the connections the profile opens (10 private writers, or the
  SharedWriter's concurrency), so that the aggregate is about 1 Mbit/s; the effective rate is measured (rows written
  during the fault x bytes per row);
- bw1mbit_t30: the same cap with write_timeout_s=30 instead of 5 (mitigation: a 5,000-row request of about 560 B per
  row needs about 22 s at 1 Mbit/s);
- flap: the proxy is disabled for FLAP_S seconds every 2 x FLAP_S seconds (connections cut, then refused).

Measured: caller capture latency per phase (p50, p99, max; captures >= 200 ms count as blocked), RSS against the bound
implied by max_queue_rows and by the requests in flight, queued and in-flight rows, back-off entries (breaker opening)
and sampled state transitions, written / dropped / uncertain / spooled / replayed rows, and the reconciliation against
the rows read back from InfluxDB (directly, not through the proxy):

    captured = written + dropped + uncertain + rejected + spooled + queued_end      (counters, 0 unexplained)
    written + replayed <= distinct seq stored <= written + replayed + uncertain     (storage)

Locally (registry env aws-scale): reduced size, on a workstation shared with other tests. Environment variables:
LAB_TOXIPROXY_URL (API, default http://127.0.0.1:8475), LAB_TOXI_LISTEN (proxy listen address inside the container,
default 0.0.0.0:8098), LAB_TOXI_PROXY_URL (the proxy as seen by Pulse, default http://127.0.0.1:8098),
LAB_TOXI_UPSTREAM (InfluxDB as seen by Toxiproxy, default sc04-influxdb:8086), LAB_PLATFORM_INFLUX_ENV (InfluxDB env
file, URL of the direct access), LAB_TOXICS, LAB_SPOOL (both | spool | nospool), LAB_PRE_S, LAB_TOXIC_S, LAB_POST_S.
"""

import gc
import json
import os
import random
import shutil
import threading
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone

from lab.tests import _platform as C

TOXICS = os.environ.get("LAB_TOXICS", "none,lat200,lat1000,loss5,loss20,bw1mbit,bw1mbit_t30,flap").split(",")
SPOOL = os.environ.get("LAB_SPOOL", "both")
MODELS, BATCH = 10, 50
ROWS_PER_S = float(os.environ.get("LAB_RATE", 2000))
QUEUE_ROWS = int(os.environ.get("LAB_QUEUE_ROWS", 5000))
PRE_S, TOXIC_S, POST_S = (float(os.environ.get(k, d)) for k, d in (("LAB_PRE_S", 10), ("LAB_TOXIC_S", 45),
                                                                   ("LAB_POST_S", 10)))
DRAIN_S = float(os.environ.get("LAB_DRAIN_S", 240))
FLAP_S = float(os.environ.get("LAB_FLAP_S", 5))
SLOT_S = 0.25
BLOCKED_MS = 200.0
TOXI_API = os.environ.get("LAB_TOXIPROXY_URL", "http://127.0.0.1:8475")
TOXI_LISTEN = os.environ.get("LAB_TOXI_LISTEN", "0.0.0.0:8098")
TOXI_PROXY_URL = os.environ.get("LAB_TOXI_PROXY_URL", "http://127.0.0.1:8098")
TOXI_UPSTREAM = os.environ.get("LAB_TOXI_UPSTREAM", "sc04-influxdb:8086")
PROXY = "sc04_influx"
BW_CONNECTIONS = C.WRITE_CONCURRENCY if C.PROFILE == "tuned" else MODELS
# memory: a queued row is a pandas row of 20 float features + y_pred + seq + _time (23 x 8 B); a row in flight also has
# its line protocol (about 560 B, kept on the unit) and the joined request body (about 560 B)
QUEUED_ROW_B, INFLIGHT_ROW_B, WRITER_BATCH = 23 * 8, 23 * 8 + 2 * 600, 5_000
PARAMS = {"models": MODELS, "batch_rows": BATCH, "rows_per_s": ROWS_PER_S, "max_queue_rows": QUEUE_ROWS,
          "pre_s": PRE_S, "toxic_s": TOXIC_S, "post_s": POST_S, "drain_timeout_s": DRAIN_S, "toxics": TOXICS,
          "write_timeout_s": 5.0, "spool": SPOOL, "flap_period_s": 2 * FLAP_S, "loss_slot_s": SLOT_S, "blocked_ms": BLOCKED_MS,
          "proxy": "Toxiproxy 2.12.0 (container sc04-toxiproxy)", "storage": "InfluxDB 2.6.1 (container sc04-influxdb)",
          "env_note": "local, reduced size (registry env aws-scale); workstation shared with SC-07 (Flux query timings)",
          "write": C.versions(), "tuned_spool": "spool_compression=gzip (with spool)"}
# blocked_captures_total: captures >= 200 ms (the caller waited for the network); the others are exact accounting
CRITERIA = {"blocked_captures_total": ("==", 0), "unexplained_total": ("==", 0), "unexplained_loss_total": ("==", 0),
            "unexplained_stored_total": ("==", 0), "duplicates_total": ("==", 0), "rss_within_bound_all": ("==", True),
            "queue_within_bound_all": ("==", True), "spool_lost_total": ("==", 0),
            "drops_only_beyond_bound_all": ("==", True)}


# ---------------------------------------------------------------------------------------------------------- Toxiproxy
class Toxi:
    """Minimal Toxiproxy HTTP API client for the one proxy of this test"""

    def __init__(self, api=TOXI_API, name=PROXY):
        self.api, self.name, self.calls, self.errors = api.rstrip("/"), name, 0, 0

    def _req(self, method, path, body=None, ok_missing=False):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.api + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        self.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                text = resp.read().decode()
                return json.loads(text) if text else None
        except urllib.error.HTTPError as exc:
            if ok_missing and exc.code == 404:
                return None
            self.errors += 1
            raise

    def version(self):
        with urllib.request.urlopen(self.api + "/version", timeout=10) as resp:
            return resp.read().decode().strip()

    def create(self):
        self._req("DELETE", f"/proxies/{self.name}", ok_missing=True)
        return self._req("POST", "/proxies", {"name": self.name, "listen": TOXI_LISTEN, "upstream": TOXI_UPSTREAM,
                                              "enabled": True})

    def enabled(self, on: bool):
        self._req("POST", f"/proxies/{self.name}", {"enabled": on})

    def add(self, name, kind, stream="downstream", toxicity=1.0, **attributes):
        self._req("POST", f"/proxies/{self.name}/toxics", {"name": name, "type": kind, "stream": stream,
                                                           "toxicity": toxicity, "attributes": attributes})

    def remove(self, name):
        self._req("DELETE", f"/proxies/{self.name}/toxics/{name}", ok_missing=True)

    def toxics(self):
        return self._req("GET", f"/proxies/{self.name}/toxics") or []

    def clear(self):
        for t in self.toxics():
            self.remove(t["name"])
        self.enabled(True)

    def delete(self):
        self._req("DELETE", f"/proxies/{self.name}", ok_missing=True)


class Fault:
    """One network fault applied for the toxic phase; dynamic faults (loss, flap) run a controller thread"""

    def __init__(self, toxi: Toxi, kind: str, seed: int = 0):
        self.toxi, self.kind, self.rng = toxi, kind, random.Random(seed)
        self.stop = threading.Event()
        self.thread = None
        self.events = Counter()  # stalled / clear slots, flaps down / up
        self.stalled_s = 0.0
        self.down_s = 0.0

    def start(self):
        t = self.toxi
        if self.kind == "lat200":
            t.add("latency", "latency", "downstream", latency=200, jitter=0)
        elif self.kind == "lat1000":
            t.add("latency", "latency", "downstream", latency=1000, jitter=0)
        elif self.kind.startswith("bw1mbit"):
            rate = max(1, round(125 / BW_CONNECTIONS))
            t.add("bw_up", "bandwidth", "upstream", rate=rate)
            t.add("bw_down", "bandwidth", "downstream", rate=rate)
            self.events["rate_kb_s_per_connection"] = rate
        elif self.kind.startswith("loss"):
            p = int(self.kind[4:]) / 100
            t.add("reset", "reset_peer", "upstream", toxicity=p, timeout=1000)
            self.thread = threading.Thread(target=self._loss, args=(p,), daemon=True)
        elif self.kind == "flap":
            self.thread = threading.Thread(target=self._flap, daemon=True)
        elif self.kind != "none":
            raise ValueError(f"unknown fault {self.kind}")
        if self.thread:
            self.thread.start()

    def _loss(self, p):
        stalled = False
        while not self.stop.is_set():
            want = self.rng.random() < p
            if want and not stalled:
                self.toxi.add("stall_up", "timeout", "upstream", timeout=0)
                self.toxi.add("stall_down", "timeout", "downstream", timeout=0)
            elif stalled and not want:
                self.toxi.remove("stall_up")
                self.toxi.remove("stall_down")
            stalled = want
            self.events["stalled_slots" if want else "clear_slots"] += 1
            self.stalled_s += SLOT_S if want else 0.0
            self.stop.wait(SLOT_S)
        if stalled:
            self.toxi.remove("stall_up")
            self.toxi.remove("stall_down")

    def _flap(self):
        while not self.stop.is_set():
            self.toxi.enabled(False)
            self.events["down"] += 1
            t0 = time.monotonic()
            self.stop.wait(FLAP_S)
            self.down_s += time.monotonic() - t0
            self.toxi.enabled(True)
            self.events["up"] += 1
            self.stop.wait(FLAP_S)

    def end(self):
        self.stop.set()
        if self.thread:
            self.thread.join(30)
        self.toxi.clear()
        return {"events": dict(self.events), "stalled_s": round(self.stalled_s, 2), "down_s": round(self.down_s, 2)}


# ---------------------------------------------------------------------------------------------------------- probes
class _Breaker:
    """Back-off entries of the writers (every failed write opens or extends the breaker), with error kinds"""

    def __init__(self, writers):
        self.entries, self.kinds, self.max_delay_s = 0, Counter(), 0.0
        self.times = []
        self._lock = threading.Lock()
        for w in writers:
            w._record_failure = self._wrap(w, w._record_failure)

    def _wrap(self, writer, orig):
        def record(units, error, uncertain):
            orig(units, error, uncertain)
            endpoint = units[0].channel.endpoint
            with self._lock:
                self.entries += 1
                self.kinds[f"{type(error).__name__}{' (uncertain)' if uncertain else ''}"] += 1
                self.max_delay_s = max(self.max_delay_s, min(2 ** (endpoint.failures - 1), writer.max_retry_delay_s))
                self.times.append(time.time())

        return record


def _endpoints(writers):
    out = {}
    for w in writers:
        with w._condition:
            for endpoint, _ in w._destinations.values():
                out[id(endpoint)] = endpoint
    return list(out.values())


def _distinct_and_points(bucket, measurement, env):
    q = f'''from(bucket: "{bucket}") |> range(start: 0, stop: {C.FAR_FUTURE})
  |> filter(fn: (r) => r._measurement == "{measurement}" and r._field == "seq")'''
    with C.client(env, 1200) as c:
        api = c.query_api()
        pts = api.query(q + " |> group() |> count()")
        dis = api.query(q + " |> group() |> distinct() |> count()")
    val = lambda t: int(sum(r.get_value() for x in t for r in x.records))  # noqa: E731
    return val(pts), val(dis)


def _line_bytes(pulse, sample) -> float:
    """Mean size of one row in line protocol (request body), measured on a sample batch"""
    x, y = sample
    x = x.copy()
    x["seq"] = range(len(x))
    frame = pulse._prepare(x, y, datetime.now(timezone.utc))
    lines = C.channel_of(pulse).target.encode(frame, pulse._write_params)
    return round(sum(len(line) + 1 for line in lines) / len(lines), 1)


def _rss_bound_mb(writer_profile_tuned: bool, queue_only: bool = False) -> float:
    """Bound implied by max_queue_rows (SC-05 formula: x3 for pandas frames and copies, + 20 MB) plus the requests in
    flight (max_batch_rows x concurrency per writer, with their line protocol and body)"""
    queued = 3 * MODELS * QUEUE_ROWS * QUEUED_ROW_B
    inflight_rows = (C.WRITE_CONCURRENCY if writer_profile_tuned else MODELS) * WRITER_BATCH
    inflight = 0 if queue_only else 2 * inflight_rows * INFLIGHT_ROW_B
    return (queued + inflight) / 2**20 + 20


# ---------------------------------------------------------------------------------------------------------- one case
def _case(ctx, env, toxi, kind, spool, seed):
    name = f"{kind}_{'spool' if spool else 'nospool'}"
    tuned = C.PROFILE == "tuned"
    bucket = C.ensure_bucket(f"lab_sc04_{C.PROFILE}_{datetime.now(timezone.utc):%m%d%H%M%S}_{name}", env)
    spool_root = ctx.workdir / "spool" / name
    if spool_root.exists():
        shutil.rmtree(spool_root)
    toxi.clear()
    penv = {**env, "url": TOXI_PROXY_URL}
    writer = C.shared_writer()
    pulses = [C.make_pulse(f"n{i}", bucket,
                           C.capture_options(tuned={"spool_compression": "gzip"} if spool else None,
                                             max_queue_rows=QUEUE_ROWS,
                                             write_timeout_s=30.0 if kind.endswith("_t30") else 5.0,
                                             spool_dir=(spool_root / f"m{i}") if spool else None),
                           penv, writer=writer)
              for i in range(MODELS)]
    writers = C.writers_of(pulses)
    channels = [C.channel_of(p) for p in pulses]
    breaker = _Breaker(writers)
    pool = C.frames_pool(BATCH, 20, pool=32, seed=seed)
    seq = [i * 100_000_000 for i in range(MODELS)]
    seq0 = list(seq)
    phase = {"name": "pre"}
    lat = {"pre": [], "toxic": [], "post": [], "drain": []}
    stop = threading.Event()
    trace = []
    peak = {"queued": 0, "inflight": 0, "rss_mb": 0.0, "per_channel": [0] * MODELS, "spool_mb": 0.0}
    transitions = Counter()
    gc.collect()
    base_rss = C.proc_snapshot()["rss_mb"]

    def generator():
        step = BATCH / ROWS_PER_S
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
        last_state = {}
        n = 0
        while not stop.wait(0.25):
            now = time.monotonic()
            for e in _endpoints(writers):
                state = "open" if now < e.retry_at else "closed"
                before = last_state.get(id(e))
                if before is not None and before != state:
                    transitions[f"{before}->{state}"] += 1
                last_state[id(e)] = state
            queued = [c.queued_rows for c in channels]
            inflight = sum(c.pending_rows for c in channels)
            rss = C.proc_snapshot()["rss_mb"]
            peak["queued"] = max(peak["queued"], sum(queued))
            peak["inflight"] = max(peak["inflight"], inflight)
            peak["rss_mb"] = max(peak["rss_mb"], rss)
            peak["per_channel"] = [max(a, b) for a, b in zip(peak["per_channel"], queued)]
            n += 1
            if n % 4 == 0:  # one trace point per second
                st = [p.stats() for p in pulses]
                spool_mb = sum(s.get("spool_bytes", 0) for s in st) / 2**20
                peak["spool_mb"] = max(peak["spool_mb"], spool_mb)
                trace.append({"t": round(time.time() - t0, 2), "phase": phase["name"], "queued": sum(queued),
                              "inflight": inflight, "rss_mb": round(rss, 1),
                              "open_endpoints": sum(1 for v in last_state.values() if v == "open"),
                              **{k: sum(s[k] for s in st) for k in ("written_rows", "dropped_rows", "uncertain_rows",
                                                                     "spooled_rows", "replayed_rows", "write_errors",
                                                                     "write_timeouts", "spool_files")},
                              "spool_mb": round(spool_mb, 2)})

    g = threading.Thread(target=generator, daemon=True)
    sm = threading.Thread(target=sampler, daemon=True)
    g.start()
    sm.start()
    time.sleep(PRE_S)
    st_start = [p.stats() for p in pulses]
    fault = Fault(toxi, kind, seed)
    phase["name"] = "toxic"
    t_toxic = time.time()
    fault.start()
    time.sleep(TOXIC_S)
    st_toxic = [p.stats() for p in pulses]
    fault_info = fault.end()
    t_clean = time.time()
    phase["name"] = "post"
    time.sleep(POST_S)
    phase["name"] = "drain"
    stop.set()
    g.join()
    t_drain = time.time()
    flushed = all(p.flush(DRAIN_S) for p in pulses)
    replay_s = None
    if spool:
        while time.time() - t_drain < DRAIN_S:
            if sum(p.stats()["spool_files"] for p in pulses) == 0:
                break
            time.sleep(0.5)
        replay_s = time.time() - t_drain
    drain_s = time.time() - t_drain
    sm.join()
    for p in pulses:
        p.close()
    if writer is not None:
        writer.close()
    stats = [p.stats() for p in pulses]
    points = distinct = 0
    for i in range(MODELS):
        pt, ds = _distinct_and_points(bucket, f"n{i}_1_input_data", env)
        points += pt
        distinct += ds
    tot = lambda key, s=stats: sum(x.get(key, 0) or 0 for x in s)  # noqa: E731
    captured, written, dropped = tot("captured_rows"), tot("written_rows"), tot("dropped_rows")
    uncertain, rejected, spooled = tot("uncertain_rows"), tot("rejected_rows"), tot("spooled_rows")
    replayed, queued_end = tot("replayed_rows"), tot("queued_rows")
    submitted = sum(seq[i] - seq0[i] for i in range(MODELS))
    spool_left = spooled - replayed
    row = {"case": name, "toxic": kind, "spool": spool, "profile": C.PROFILE, "captured": captured,
           "submitted": submitted, "written": written, "dropped": dropped, "uncertain": uncertain,
           "rejected": rejected, "spooled": spooled, "replayed": replayed, "queued_end": queued_end,
           "spool_left_rows": spool_left, "spool_files_end": tot("spool_files"), "write_errors": tot("write_errors"),
           "write_timeouts": tot("write_timeouts"), "stored_points": points, "distinct": distinct,
           "duplicates": points - distinct, "lost": submitted - distinct, "flushed": flushed,
           "drain_s": round(drain_s, 1), "replay_s": None if replay_s is None else round(replay_s, 1),
           "written_during_toxic": sum(b["written_rows"] - a["written_rows"] for a, b in zip(st_start, st_toxic)),
           "line_bytes_per_row": _line_bytes(pulses[0], pool[0]),
           "dropped_during_toxic": sum(s["dropped_rows"] for s in st_toxic),
           "uncertain_during_toxic": sum(s.get("uncertain_rows", 0) for s in st_toxic),
           "spooled_during_toxic": sum(s["spooled_rows"] for s in st_toxic),
           "queued_max": peak["queued"], "inflight_max": peak["inflight"], "queued_max_per_channel": max(peak["per_channel"]),
           "rss_base_mb": round(base_rss, 1), "rss_max_mb": round(peak["rss_mb"], 1),
           "rss_delta_mb": round(peak["rss_mb"] - base_rss, 1), "rss_bound_mb": round(_rss_bound_mb(tuned), 1),
           "rss_bound_queue_only_mb": round(_rss_bound_mb(tuned, queue_only=True), 1),
           "spool_mb_max": round(peak["spool_mb"], 2),
           "breaker_entries": breaker.entries, "breaker_max_delay_s": breaker.max_delay_s,
           "breaker_error_kinds": dict(breaker.kinds), "breaker_entries_during_toxic":
               sum(1 for t in breaker.times if t_toxic <= t < t_clean),
           "breaker_transitions_sampled": dict(transitions), "endpoints": len(_endpoints(writers)) or None,
           "fault": fault_info, "toxiproxy_calls": toxi.calls}
    row["written_rows_per_s_toxic"] = round(row["written_during_toxic"] / TOXIC_S, 1)
    row["effective_mbit_s_toxic"] = round(row["written_rows_per_s_toxic"] * row["line_bytes_per_row"] * 8 / 1e6, 3)
    for ph in ("pre", "toxic", "post"):
        row[f"captures_{ph}"] = len(lat[ph])
        row[f"p50_ms_{ph}"] = round((C.pct(lat[ph], 50) or 0) * 1000, 3)
        row[f"p99_ms_{ph}"] = round((C.pct(lat[ph], 99) or 0) * 1000, 3)
        row[f"max_ms_{ph}"] = round(max(lat[ph], default=0) * 1000, 1)
    all_lat = lat["pre"] + lat["toxic"] + lat["post"]
    row["blocked_captures"] = sum(1 for v in all_lat if v * 1000 >= BLOCKED_MS)
    row["unexplained"] = captured - (written + dropped + uncertain + rejected + spooled + queued_end)
    row["unexplained_loss"] = max(0, written + replayed - distinct)
    row["unexplained_stored"] = max(0, distinct - (written + replayed + uncertain))
    row["uncertain_stored"] = max(0, distinct - written - replayed)  # uncertain rows InfluxDB did store
    row["uncertain_lost"] = uncertain - row["uncertain_stored"]
    row["rss_within_bound"] = row["rss_delta_mb"] <= row["rss_bound_mb"]
    row["queue_within_bound"] = row["queued_max_per_channel"] <= QUEUE_ROWS + BATCH
    # drops beyond the queue bound only: a Pulse dropped rows only if its queue reached max_queue_rows (sampled every
    # 250 ms, so within two batches of it)
    row["drops_only_beyond_bound"] = all(
        s["dropped_rows"] == 0 or pk >= QUEUE_ROWS - 2 * BATCH for s, pk in zip(stats, peak["per_channel"]))
    (ctx.workdir / f"trace_{C.PROFILE}_{name}.json").write_text(json.dumps(trace))
    return row


def run(ctx):
    os.environ["LAB_INFLUX_ENV"] = C.platform_env_file()
    env = C.influx_env()
    toxi = Toxi()
    ctx.params["toxiproxy_version"] = toxi.version()
    toxi.create()
    spools = {"both": (False, True), "spool": (True,), "nospool": (False,)}[SPOOL]
    cases = [(k, s) for k in TOXICS for s in ((False,) if k == "none" else spools)]
    ctx.note(f"profile {C.PROFILE}; {len(cases)} cases; local reduced-size run (registry env aws-scale), workstation "
             "shared with SC-07 (Flux query timings) running at the same time")
    rows = []
    for n, (kind, spool) in enumerate(cases):
        try:
            rows.append(_case(ctx, env, toxi, kind, spool, seed=n))
        except Exception as exc:  # noqa: BLE001 - one failed case does not stop the others
            rows.append({"case": f"{kind}_{'spool' if spool else 'nospool'}", "error": f"{type(exc).__name__}: {exc}"[:400]})
            try:
                toxi.clear()
            except Exception:  # noqa: BLE001
                pass
        (ctx.workdir / f"cases_{C.PROFILE}.json").write_text(json.dumps(rows, indent=1, default=str))
        print(json.dumps({k: rows[-1].get(k) for k in ("case", "p99_ms_toxic", "max_ms_toxic", "rss_delta_mb", "dropped",
                                                        "uncertain", "spooled", "lost", "unexplained", "error")}),
              flush=True)
    toxi.delete()
    return evaluate(ctx, rows)


def evaluate(ctx, rows):
    """Metrics and limits from the case rows (also used by `reevaluate` on a recorded result)"""
    ok = [r for r in rows if "error" not in r]
    base = next((r for r in ok if r["toxic"] == "none"), None)
    metrics = {"cases": {r["case"]: r for r in rows},
               "blocked_captures_total": sum(r["blocked_captures"] for r in ok),
               "unexplained_total": sum(abs(r["unexplained"]) for r in ok),
               "unexplained_loss_total": sum(r["unexplained_loss"] for r in ok),
               "unexplained_stored_total": sum(r["unexplained_stored"] for r in ok),
               "duplicates_total": sum(r["duplicates"] for r in ok),
               "rss_within_bound_all": all(r["rss_within_bound"] for r in ok) if ok else None,
               "queue_within_bound_all": all(r["queue_within_bound"] for r in ok) if ok else None,
               # with spool, rows missing beyond the counted queue drops (the spool only takes the rows of an
               # unavailable storage: a slow link that keeps answering still overflows the queue)
               "spool_lost_total": sum(r["lost"] - r["dropped"] for r in ok if r["spool"]),
               "spool_dropped_total": sum(r["dropped"] for r in ok if r["spool"]),
               "drops_only_beyond_bound_all": all(r["drops_only_beyond_bound"] for r in ok) if ok else None,
               "p99_ms_toxic_max": max([r["p99_ms_toxic"] for r in ok], default=None),
               "baseline_p99_ms": base["p99_ms_toxic"] if base else None,
               "errors": len(rows) - len(ok)}
    for r in rows:
        if "error" in r:
            ctx.note(f"case {r['case']} failed: {r['error']}")
    for r in ok:
        tag = f"[{r.get('profile', C.PROFILE)} {r['case']}]"
        if r["blocked_captures"]:
            ctx.limit(f"{tag} {r['blocked_captures']} captures took >= {BLOCKED_MS:.0f} ms (max {r['max_ms_toxic']} ms "
                      "under the fault)", json.dumps({k: r[k] for k in ("p50_ms_toxic", "p99_ms_toxic", "max_ms_toxic",
                                                                         "max_ms_pre", "max_ms_post")}),
                      "Find what the caller waits for (lock held during a network call?)", "critical")
        if r["unexplained"] or r["unexplained_loss"] or r["unexplained_stored"] or r["duplicates"]:
            ctx.limit(f"{tag} accounting not exact: {r['unexplained']} unexplained by the counters, "
                      f"{r['unexplained_loss']} written rows missing, {r['unexplained_stored']} stored rows not counted, "
                      f"{r['duplicates']} duplicates",
                      json.dumps({k: r[k] for k in ("captured", "written", "dropped", "uncertain", "rejected", "spooled",
                                                     "replayed", "queued_end", "distinct", "stored_points")}),
                      "Reconcile the counters of the write path under this network fault", "critical")
        if not r["rss_within_bound"]:
            ctx.limit(f"{tag} RSS +{r['rss_delta_mb']} MB above the bound {r['rss_bound_mb']} MB",
                      json.dumps({k: r[k] for k in ("rss_base_mb", "rss_max_mb", "queued_max", "inflight_max")}),
                      "Bound the memory held by requests in flight", "major")
        if not r["queue_within_bound"]:
            ctx.limit(f"{tag} a queue held {r['queued_max_per_channel']} rows (max_queue_rows={QUEUE_ROWS})",
                      json.dumps({k: r[k] for k in ("queued_max", "queued_max_per_channel")}),
                      "Enforce max_queue_rows on requeued rows", "major")
        if not r["drops_only_beyond_bound"]:
            ctx.limit(f"{tag} rows dropped while the queues had room: {r['dropped']} dropped, peak queue "
                      f"{r['queued_max_per_channel']} of {QUEUE_ROWS}", json.dumps({k: r[k] for k in (
                          "dropped", "dropped_during_toxic", "uncertain", "breaker_error_kinds")}),
                      "Keep certainly-unsent rows in the queue (requeue) under this fault", "major")
        if r["spool"] and r["dropped"]:
            ctx.limit(f"{tag} with spool, {r['dropped']} rows dropped beyond max_queue_rows (counted): the link kept "
                      "answering, slower than the traffic, so the storage was never unavailable and the spool did not "
                      "take the overflow",
                      json.dumps({k: r[k] for k in ("written", "dropped", "spooled", "queued_max", "inflight_max",
                                                     "write_timeouts")}),
                      "With spool_dir, spool the oldest batches beyond max_queue_rows instead of dropping them "
                      "(bounded by spool_max_bytes)", "minor")
        if r["spool"] and r["lost"] - r["dropped"]:
            ctx.limit(f"{tag} with spool: {r['lost'] - r['dropped']} rows lost beyond the counted drops", json.dumps({k: r[k] for k in (
                "submitted", "distinct", "spooled", "replayed", "spool_files_end", "dropped", "uncertain")}),
                      "Reconcile spool replay with captured rows", "critical")
        if not r["spool"] and r["uncertain_lost"] > 0:
            ctx.limit(f"{tag} without spool, {r['uncertain_lost']} rows lost while the queues had room: writes that "
                      f"timed out or were cut ({r['uncertain']} uncertain rows, {r['uncertain_stored']} of them stored "
                      "by InfluxDB anyway); counted, never retried",
                      json.dumps({k: r[k] for k in ("uncertain", "uncertain_stored", "write_timeouts", "write_errors",
                                                     "breaker_error_kinds", "queued_max")}),
                      "Set spool_dir on links that time out (uncertain writes are then replayed, idempotent); or retry "
                      "uncertain writes in memory once, since InfluxDB writes are idempotent", "minor")
        if r["dropped"] and not r["spool"]:
            ctx.limit(f"{tag} without spool, {r['dropped']} rows dropped beyond max_queue_rows (counted): the link "
                      f"wrote less than the {ROWS_PER_S:.0f} rows/s offered",
                      json.dumps({k: r[k] for k in ("written", "dropped", "queued_max", "write_timeouts",
                                                     "breaker_entries")}),
                      "Expected behaviour of the bound; size max_queue_rows for the longest degraded period, or set "
                      "spool_dir", "minor")
        if r["spool"] and r["flushed"] and r["replay_s"] is not None and r["spool_files_end"]:
            ctx.limit(f"{tag} spool not empty {DRAIN_S:.0f} s after the link came back ({r['spool_files_end']} files)",
                      json.dumps({k: r[k] for k in ("spooled", "replayed", "spool_left_rows", "replay_s")}),
                      "Replay faster after a degraded period", "major")
    return metrics


def reevaluate(path):
    """Recompute metrics, checks, limits and status of a recorded SC-04 result from its case rows (same data), e.g.
    after a correction of a criterion; the previous status is kept in `reevaluated_from`"""
    from pathlib import Path

    from lab import harness

    f = Path(path)
    result = json.loads(f.read_text(encoding="utf-8"))
    ctx = harness.Context(result["id"], result["env"], f.with_suffix(""), result["params"])
    rows = list(result["metrics"]["cases"].values())
    metrics = evaluate(ctx, rows)
    checks = harness.check(metrics, CRITERIA)
    status = "error" if result.get("error") else ("limit-found" if ctx.limits else
                                                  ("passed" if all(checks.values()) else "failed"))
    from dataclasses import asdict

    result["reevaluated_from"] = {"status": result["status"], "limits": len(result["limits"]),
                                  "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    result.update(metrics=metrics, checks=checks, status=status, limits=[asdict(x) for x in ctx.limits],
                  criteria={k: list(v) for k, v in CRITERIA.items()})
    f.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"{f.name}: {result['reevaluated_from']['status']} -> {status}; {len(ctx.limits)} limit(s)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "reevaluate":
        for arg in sys.argv[2:]:
            reevaluate(arg)
