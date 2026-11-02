"""Shared harness of the Pulsar.ml lab: run one registry test, sample host health, check criteria, record limits.

Every test is a function `run(ctx) -> dict of metrics`. The harness times it, samples the host every few seconds
(CPU, RSS, threads, open files, disk, GPU when present), checks the pass criteria, and writes one JSON result per test
in `lab/results/<id>.json`. A test that finds a limit records it with `ctx.limit(finding, evidence, proposal)`: the
limits and the improvement proposals are the main output of the lab, not a side note.

    python -m lab.harness SC-05            # run one test registered in lab/tests/
    python -m lab.harness --self-check     # check that the health sampler records on this host
"""

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

RESULTS = Path(__file__).parent / "results"


@dataclass
class Limit:
    finding: str  # what broke or degraded, with the threshold where it happened
    evidence: str  # numbers, logs or files that show it
    proposal: str  # the improvement to make (code, configuration, documentation)
    severity: str = "major"  # critical | major | minor


@dataclass
class Context:
    test_id: str
    env: str
    workdir: Path
    params: dict = field(default_factory=dict)
    limits: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def limit(self, finding: str, evidence: str, proposal: str, severity: str = "major") -> None:
        self.limits.append(Limit(finding, evidence, proposal, severity))

    def note(self, text: str) -> None:
        self.notes.append(text)


class HealthSampler:
    """Background sampling of the host while a test runs (psutil; nvidia-smi when available)"""

    def __init__(self, interval_s: float = 5.0):
        import psutil

        self.psutil, self.interval_s, self.samples = psutil, interval_s, []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._proc = psutil.Process()
        self.errors, self.last_error = 0, None

    def _gpu(self) -> dict:
        if not shutil.which("nvidia-smi"):
            return {}
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().split(",")
            return {"gpu_util": float(out[0]), "gpu_mem_mib": float(out[1])}
        except Exception:  # noqa: BLE001 - health sampling never fails a test
            return {}

    def _open_files(self) -> int:
        # num_fds exists on POSIX only, num_handles on Windows only: test before calling. The former
        # getattr(proc, "num_fds", proc.num_handles) evaluated proc.num_handles eagerly and raised on Linux.
        return self._proc.num_fds() if hasattr(self._proc, "num_fds") else self._proc.num_handles()

    def sample(self) -> dict:
        return {
            "t": time.time(), "cpu": self.psutil.cpu_percent(), "mem": self.psutil.virtual_memory().percent,
            "rss_mb": self._proc.memory_info().rss / 2**20, "threads": self._proc.num_threads(),
            "fds": self._open_files(), "disk": self.psutil.disk_usage(str(RESULTS.anchor or "/")).percent, **self._gpu(),
        }

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self.sample())
            except Exception as exc:  # noqa: BLE001 - health sampling never fails a test, but the cause is kept
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)

    def summary(self) -> dict:
        if not self.samples:
            return {"samples": 0, "errors": self.errors, "last_error": self.last_error}
        keys = [k for k in self.samples[0] if k != "t"]
        out = {f"{k}_max": max(s.get(k, 0) for s in self.samples) for k in keys}
        rss = [s["rss_mb"] for s in self.samples]
        hours = max((self.samples[-1]["t"] - self.samples[0]["t"]) / 3600, 1e-9)
        out["rss_growth_mb_per_h"] = (rss[-1] - rss[0]) / hours
        out["samples"] = len(self.samples)
        if self.errors:
            out["errors"], out["last_error"] = self.errors, self.last_error
        return out


def self_check() -> int:
    """Quick check that the health sampler records on this host (it recorded nothing on Linux before the fix)"""
    with HealthSampler(interval_s=0.2) as health:
        time.sleep(1.0)
    summary = health.summary()
    expected = {"cpu_max", "mem_max", "rss_mb_max", "threads_max", "fds_max", "disk_max"}
    missing = expected - set(summary)
    ok = summary.get("samples", 0) >= 2 and not missing and summary["fds_max"] > 0
    print(f"health sampler on {platform.system()}: {'OK' if ok else 'FAILED'} {json.dumps(summary, default=str)}")
    return 0 if ok else 1


def check(metrics: dict, criteria: dict) -> dict:
    """criteria: {"metric": (">=" | "<=" | "==", value)}; returns {metric: passed}"""
    ops = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, "==": lambda a, b: a == b,
           "<": lambda a, b: a < b, ">": lambda a, b: a > b}
    return {m: (m in metrics and metrics[m] is not None and ops[op](metrics[m], v)) for m, (op, v) in criteria.items()}


def run_test(test_id: str, fn: Callable[[Context], dict], criteria: Optional[dict] = None, env: str = "local",
             params: Optional[dict] = None) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    # LAB_RUN_TAG keeps several runs of one test apart: results/<id>.<tag>.json and results/<id>.<tag>/
    tag = os.environ.get("LAB_RUN_TAG")
    name = f"{test_id}.{tag}" if tag else test_id
    workdir = RESULTS / name
    workdir.mkdir(exist_ok=True)
    ctx = Context(test_id, env, workdir, params or {})
    started = datetime.now(timezone.utc)
    error, metrics = None, {}
    with HealthSampler() as health:
        t0 = time.perf_counter()
        try:
            metrics = fn(ctx) or {}
        except Exception as exc:  # noqa: BLE001 - a crashing test is a result too
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}"
        seconds = time.perf_counter() - t0
    checks = check(metrics, criteria or {})
    status = "error" if error else ("limit-found" if ctx.limits else ("passed" if all(checks.values()) else "failed"))
    result = {
        "id": test_id, "run_tag": tag, "status": status, "env": env, "host": platform.node(), "python": sys.version.split()[0],
        "started": started.isoformat(timespec="seconds"), "seconds": round(seconds, 1), "params": ctx.params,
        "metrics": metrics, "criteria": {k: list(v) for k, v in (criteria or {}).items()}, "checks": checks,
        "limits": [asdict(x) for x in ctx.limits], "notes": ctx.notes, "health": health.summary(), "error": error,
    }
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"{name}: {status} in {seconds:.0f} s; {len(ctx.limits)} limit(s)", flush=True)
    return result


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        sys.exit(self_check())
    test = sys.argv[1]
    module = importlib.import_module(f"lab.tests.{test.lower().replace('-', '_')}")
    run_test(test, module.run, getattr(module, "CRITERIA", None), env=os.environ.get("LAB_ENV", "local"),
             params=getattr(module, "PARAMS", None))
