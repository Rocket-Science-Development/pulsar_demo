"""Shared helpers of the platform tests: InfluxDB access from the lab env file, Pulse factory, synthetic batches,
reconciliation queries, process sampling.

Credentials are read from ~/.pulsar-lab/influx.env (copied from /etc/pulsar-lab/influx.env on lab-db) and are never
printed nor written to the results.
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ENV_FILE = Path(os.environ.get("LAB_INFLUX_ENV", Path.home() / ".pulsar-lab" / "influx.env"))
FAR_FUTURE = "2100-01-01T00:00:00Z"


def influx_env() -> dict:
    values = {}
    env_file = Path(os.environ.get("LAB_INFLUX_ENV", ENV_FILE))
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"')
    url = os.environ.get("LAB_INFLUX_URL_OVERRIDE") or values.get("LAB_INFLUX_URL") or values["INFLUX_URL"]
    return {"url": url, "token": values["INFLUX_TOKEN"], "org": values["INFLUX_ORG"]}


def client(env=None, timeout_s=120):
    from influxdb_client import InfluxDBClient

    env = env or influx_env()
    return InfluxDBClient(url=env["url"], token=env["token"], org=env["org"], timeout=int(timeout_s * 1000))


def ensure_bucket(name: str, env=None) -> str:
    """Create the bucket if needed (infinite retention); lab buckets are prefixed lab_"""
    from influxdb_client import BucketsApi

    env = env or influx_env()
    with client(env) as c:
        from influxdb_client.rest import ApiException

        api = BucketsApi(c)
        try:
            found = api.find_buckets(name=name, org=env["org"]).buckets
        except ApiException as exc:
            if exc.status != 404:
                raise
            found = []
        if not found:
            api.create_bucket(bucket_name=name, org=env["org"])
    return name


def make_pulse(model_id: str, bucket: str, options=None, env=None, version="1", tags=None):
    from pulsar_data_collection.models import CaptureOptions, PulseParameters
    from pulsar_data_collection.pulse import Pulse

    env = env or influx_env()
    params = PulseParameters(
        model_id=model_id, model_version=version, data_id="lab", reference_data_storage="none", target_name="y",
        storage_engine="influxdb", login={**env, "bucket_name": bucket}, timestamp_column_name="_time",
        additional_tags=tags,
    )
    return Pulse(params, options or CaptureOptions())


def frames_pool(n_rows=50, n_features=20, pool=64, seed=0, with_seq=False):
    """Pre-built (data_points, predictions) pairs so that generation cost stays out of the capture measurement"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(pool):
        x = pd.DataFrame(rng.normal(size=(n_rows, n_features)), columns=[f"f{i}" for i in range(n_features)])
        y = pd.DataFrame({"y_pred": rng.integers(0, 2, n_rows)})
        out.append((x, y))
    return out


def capture(pulse, x, y, ts=None, seq_start=None):
    from pulsar_data_collection.models import DataWithPrediction

    if seq_start is not None:
        x = x.copy()
        x["seq"] = np.arange(seq_start, seq_start + len(x), dtype="int64")
    pulse.capture_data(DataWithPrediction(timestamp=ts or datetime.now(timezone.utc), predictions=y, data_points=x,
                                          features_names=list(x.columns)))


def count_rows(bucket: str, field: str = "f0", measurement_prefix: str = None, env=None, timeout_s=600) -> int:
    """Rows stored = points of one field (InfluxDB keeps one point per measurement, tags and timestamp)"""
    flt = f'r._field == "{field}"'
    if measurement_prefix:
        flt += f' and strings.hasPrefix(v: r._measurement, prefix: "{measurement_prefix}")'
    q = f'''import "strings"
from(bucket: "{bucket}") |> range(start: 0, stop: {FAR_FUTURE}) |> filter(fn: (r) => {flt})
  |> group() |> count()'''
    with client(env, timeout_s) as c:
        tables = c.query_api().query(q)
    return int(sum(rec.get_value() for t in tables for rec in t.records))


def count_by_measurement(bucket: str, field: str = "f0", env=None, timeout_s=600) -> dict:
    q = f'''from(bucket: "{bucket}") |> range(start: 0, stop: {FAR_FUTURE}) |> filter(fn: (r) => r._field == "{field}")
  |> group(columns: ["_measurement"]) |> count()'''
    with client(env, timeout_s) as c:
        tables = c.query_api().query(q)
    return {rec.values["_measurement"]: int(rec.get_value()) for t in tables for rec in t.records}


def seq_values(bucket: str, measurement: str, env=None, timeout_s=900) -> np.ndarray:
    q = f'''from(bucket: "{bucket}") |> range(start: 0, stop: {FAR_FUTURE})
  |> filter(fn: (r) => r._measurement == "{measurement}" and r._field == "seq") |> keep(columns: ["_value"])'''
    with client(env, timeout_s) as c:
        frame = c.query_api().query_data_frame(q)
    if isinstance(frame, list):
        frame = pd.concat(frame) if frame else pd.DataFrame({"_value": []})
    return frame["_value"].to_numpy(dtype="int64") if len(frame) else np.array([], dtype="int64")


def proc_snapshot(proc=None) -> dict:
    import psutil

    p = proc or psutil.Process()
    try:
        conns = len(p.net_connections(kind="tcp"))
    except Exception:  # noqa: BLE001
        conns = -1
    return {"rss_mb": p.memory_info().rss / 2**20, "threads": p.num_threads(), "fds": p.num_fds(), "tcp": conns,
            "cpu_s": sum(p.cpu_times()[:2])}


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def remote_influx_stats(env=None) -> dict:
    """A few InfluxDB /metrics gauges (process RSS, CPU seconds, write requests)"""
    import urllib.request

    env = env or influx_env()
    try:
        text = urllib.request.urlopen(env["url"] + "/metrics", timeout=10).read().decode()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    out = {"go_memstats_alloc_bytes": 0.0, "write_requests": 0.0, "write_errors_5xx_4xx": 0.0}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        if line.startswith("go_memstats_alloc_bytes "):
            out["go_memstats_alloc_bytes"] = value
        elif line.startswith("http_api_requests_total") and 'path="/api/v2/write"' in line:
            out["write_requests"] += value
            if 'status="2XX"' not in line:
                out["write_errors_5xx_4xx"] += value
    return out


def influx_fds(env=None, container: str = None) -> dict:
    """File descriptors of the InfluxDB server: open and limit from its /metrics (process_open_fds, process_max_fds),
    and, when `container` (or LAB_INFLUX_CONTAINER) names a local Docker container, the count of /proc/1/fd inside it
    and its soft limit (/proc/1/limits), and the number of "too many open files" lines in its logs. InfluxDB 2.6 does
    not export process_open_fds, so the container source is the one that works there. Missing sources are left out,
    never guessed."""
    import shutil
    import subprocess
    import urllib.request

    env = env or influx_env()
    out = {}
    try:
        text = urllib.request.urlopen(env["url"] + "/metrics", timeout=10).read().decode()
        for line in text.splitlines():
            for key in ("process_open_fds", "process_max_fds"):
                if line.startswith(key + " "):
                    out[key] = float(line.rsplit(" ", 1)[1])
    except Exception as exc:  # noqa: BLE001
        out["metrics_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    container = container or os.environ.get("LAB_INFLUX_CONTAINER")
    if container and shutil.which("docker"):
        try:
            r = subprocess.run(["docker", "exec", container, "sh", "-c", "ls /proc/1/fd | wc -l"],
                               capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and r.stdout.strip().isdigit():
                out["container_fds"] = int(r.stdout.strip())
            lim = subprocess.run(["docker", "exec", container, "sh", "-c", "grep 'Max open files' /proc/1/limits"],
                                 capture_output=True, text=True, timeout=20).stdout.split()
            if len(lim) >= 4 and lim[3].isdigit():  # Max open files <soft> <hard> files
                out["container_max_fds"] = int(lim[3])
            logs = subprocess.run(["docker", "logs", "--since", "24h", container], capture_output=True, text=True,
                                  timeout=60)
            out["too_many_open_files_log_lines"] = sum(
                "too many open files" in line.lower() for line in (logs.stdout + logs.stderr).splitlines())
        except Exception as exc:  # noqa: BLE001
            out["docker_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return out


def wait_until(predicate, timeout_s, step_s=0.5):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step_s)
    return predicate()


class OutageProxy:
    """TCP proxy 127.0.0.1:port -> InfluxDB, used to cut the storage without touching the InfluxDB container.

    down(): closes the listening socket and every open connection (clients get "connection refused", as with a stopped
    server). up(): listens again. hang=True instead accepts connections and never answers (timeouts).
    """

    def __init__(self, target_url: str, port: int = 18086):
        from urllib.parse import urlparse

        u = urlparse(target_url)
        self.target = (u.hostname, u.port or 8086)
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.hang = False
        self._srv = None
        self._conns = set()
        self._lock = __import__("threading").Lock()
        self.up()

    def _pipe(self, a, b):
        try:
            while True:
                data = a.recv(65536)
                if not data:
                    break
                b.sendall(data)
        except OSError:
            pass
        finally:
            for s in (a, b):
                try:
                    s.close()
                except OSError:
                    pass

    def _serve(self, srv):
        import socket
        import threading

        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            if self.hang:
                with self._lock:
                    self._conns.add(c)
                continue
            try:
                t = socket.create_connection(self.target, timeout=10)
                t.settimeout(None)
            except OSError:
                c.close()
                continue
            with self._lock:
                self._conns.update((c, t))
            threading.Thread(target=self._pipe, args=(c, t), daemon=True).start()
            threading.Thread(target=self._pipe, args=(t, c), daemon=True).start()

    def up(self):
        import socket
        import threading

        if self._srv:
            return
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", self.port))
        srv.listen(512)
        self._srv = srv
        threading.Thread(target=self._serve, args=(srv,), daemon=True).start()

    def down(self):
        import socket

        if self._srv:
            try:
                self._srv.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._srv.close()
            self._srv = None
        with self._lock:
            for s in list(self._conns):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                    s.close()
                except OSError:
                    pass
            self._conns.clear()
