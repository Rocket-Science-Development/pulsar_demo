"""Host health sampler for SC-12: whole-host psutil (CPU, memory, disk, network), docker stats and nvidia-smi when
present, one CSV line every INTERVAL seconds until the deadline.

    python -m lab.tests._hostmon <out.csv> <interval_s> <deadline_epoch>
"""

import csv
import shutil
import subprocess
import sys
import time

import psutil


def docker_stats() -> dict:
    if not shutil.which("docker"):
        return {}
    try:
        out = subprocess.run(["sudo", "-n", "docker", "stats", "--no-stream", "--format",
                              "{{.Name}};{{.CPUPerc}};{{.MemUsage}}"], capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        return {}
    res = {}
    for line in out.splitlines():
        try:
            name, cpu, mem = line.split(";")
            used = mem.split("/")[0].strip()
            unit = {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024}
            num = next(float(used[: -len(u)]) * f for u, f in unit.items() if used.endswith(u))
            res[f"docker_{name}_cpu_pct"] = float(cpu.rstrip("%"))
            res[f"docker_{name}_mem_mb"] = num
        except Exception:  # noqa: BLE001
            continue
    return res


def main(path, interval, deadline):
    fields = None
    net0 = psutil.net_io_counters()
    t0 = time.time()
    with open(path, "a", newline="") as fh:
        while time.time() < deadline:
            net = psutil.net_io_counters()
            dt = max(time.time() - t0, 1e-9)
            row = {"t": round(time.time(), 1), "cpu_pct": psutil.cpu_percent(interval=1),
                   "mem_pct": psutil.virtual_memory().percent, "disk_pct": psutil.disk_usage("/").percent,
                   "load1": psutil.getloadavg()[0], "net_tx_mb_s": (net.bytes_sent - net0.bytes_sent) / dt / 2**20,
                   "net_rx_mb_s": (net.bytes_recv - net0.bytes_recv) / dt / 2**20,
                   "procs": len(psutil.pids()), **docker_stats()}
            net0, t0 = net, time.time()
            if fields is None or set(row) - set(fields):
                fields = list(dict.fromkeys((fields or []) + list(row)))
                w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
            w.writerow(row)
            fh.flush()
            time.sleep(max(0, interval - 1))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
