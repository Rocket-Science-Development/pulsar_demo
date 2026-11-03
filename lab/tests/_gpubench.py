"""Shared helpers of SC-09 / SC-10 (platform family, GPU host): GPU sampling with nvidia-smi, percentiles, and a
lab-owned Ollama server started under a systemd memory limit.

Nothing here touches another host or the shared Ollama service configuration: the lab server listens on its own
localhost port, keeps OLLAMA_MAX_LOADED_MODELS=1, and is stopped when the test ends.
"""

import json
import shutil
import subprocess
import threading
import time
import urllib.request

import numpy as np


def pct(values, q):
    values = [v for v in values if v is not None]
    return round(float(np.percentile(values, q)), 4) if values else None


class GpuSampler:
    """nvidia-smi every `interval_s` seconds: utilisation (%), memory used (MiB); `window(t0, t1)` summarises"""

    def __init__(self, interval_s: float = 1.0):
        self.interval_s, self.samples = interval_s, []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.available = bool(shutil.which("nvidia-smi"))

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().split(",")
                self.samples.append((time.time(), float(out[0]), float(out[1]), float(out[2])))
            except Exception:  # noqa: BLE001 - sampling never fails a test
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        if self.available:
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self.available:
            self._thread.join(timeout=10)

    def window(self, t0: float, t1: float) -> dict:
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return {"gpu_util_mean": None, "gpu_util_max": None, "gpu_mem_max_mib": None, "gpu_samples": 0}
        return {"gpu_util_mean": round(float(np.mean([x[1] for x in s])), 1), "gpu_util_max": max(x[1] for x in s),
                "gpu_mem_max_mib": max(x[2] for x in s), "gpu_mem_total_mib": s[0][3], "gpu_samples": len(s)}


def http_json(url: str, payload=None, timeout: float = 30.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "null")


def loaded_models(base: str) -> list:
    """/api/ps of an Ollama server: name, size, size_vram, context_length"""
    try:
        return [{"name": m.get("name"), "size": m.get("size"), "size_vram": m.get("size_vram"),
                 "context_length": m.get("context_length")} for m in http_json(f"{base}/api/ps").get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def unload_all(base: str) -> list:
    """Unload every model of an Ollama server (keep_alive 0); returns the names unloaded"""
    names = [m["name"] for m in loaded_models(base)]
    for name in names:
        try:
            http_json(f"{base}/api/generate", {"model": name, "keep_alive": 0}, timeout=60)
        except Exception:  # noqa: BLE001
            pass
    return names


class LabOllama:
    """A lab-owned `ollama serve` on its own port, in a transient systemd unit with MemoryMax (RAM limit), one loaded
    model at most; OLLAMA_NUM_PARALLEL is the variable under test"""

    def __init__(self, port: int, num_parallel: int, models_dir: str, binary: str, memory_max: str = "13G",
                 unit: str = "pulsar-lab-sc10-ollama", context_length: int = 4096):
        self.port, self.num_parallel, self.models_dir, self.binary = port, num_parallel, models_dir, binary
        self.memory_max, self.unit, self.context_length = memory_max, unit, context_length
        self.base = f"http://127.0.0.1:{port}"

    def start(self, timeout_s: float = 60.0):
        self.stop()
        cmd = ["sudo", "-n", "systemd-run", f"--unit={self.unit}", "--uid=ubuntu", "--collect",
               "-p", f"MemoryMax={self.memory_max}",
               "-E", f"OLLAMA_HOST=127.0.0.1:{self.port}", "-E", f"OLLAMA_MODELS={self.models_dir}",
               "-E", "OLLAMA_MAX_LOADED_MODELS=1", "-E", f"OLLAMA_NUM_PARALLEL={self.num_parallel}",
               "-E", f"OLLAMA_CONTEXT_LENGTH={self.context_length}", "-E", "OLLAMA_KEEP_ALIVE=10m",
               "-E", "HOME=/home/ubuntu", self.binary, "serve"]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                http_json(f"{self.base}/api/version", timeout=3)
                return self
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        raise RuntimeError(f"lab Ollama did not answer on port {self.port} within {timeout_s} s")

    def stop(self):
        subprocess.run(["sudo", "-n", "systemctl", "stop", self.unit], capture_output=True, text=True, timeout=60)
        subprocess.run(["sudo", "-n", "systemctl", "reset-failed", self.unit], capture_output=True, text=True, timeout=30)

    def memory_peak_mb(self):
        r = subprocess.run(["systemctl", "show", self.unit, "-p", "MemoryPeak", "-p", "MemoryCurrent"],
                           capture_output=True, text=True, timeout=10).stdout
        out = {}
        for line in r.splitlines():
            k, _, v = line.partition("=")
            out[k] = round(int(v) / 2**20, 1) if v.isdigit() else None
        return out
