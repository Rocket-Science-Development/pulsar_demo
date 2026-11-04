"""Exact environment of a validation run: hardware, OS, services, package and model versions, source commits.

Every number cited as evidence must say where it was measured. This module collects that once per run, in one JSON
document, so the reports never have to reconstruct it from memory:

- hardware: CPU model, logical / physical cores, RAM, GPU (nvidia-smi), cloud instance type (EC2 IMDSv2, 1 s timeout);
- software: OS, Python, the versions of the packages the lab uses, Docker and the containers running;
- services: InfluxDB and Grafana versions (health endpoints), Ollama models with their digests;
- sources: git commit, branch and dirty flag of the pulsar_* checkouts the Python imports come from.

Each probe is independent and bounded in time: a missing tool or service gives `null` and a reason, never an error.
No secret is read or written (no token, no environment dump).

    python -m lab.environment                 # print the JSON
    python -m lab.environment --out env.json  # write it
    python -m lab.environment --markdown      # a table for a report
"""

import argparse
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

PACKAGES = (
    "pulsar_data_collection", "pulsar_metrics", "influxdb-client", "pandas", "numpy", "scipy", "scikit-learn",
    "pydantic", "opentelemetry-sdk", "torch", "transformers", "gliner", "sentence-transformers", "psutil",
)
SERVICES = {
    "influxdb": os.environ.get("LAB_INFLUX_URL", "http://localhost:8087") + "/health",
    "grafana": os.environ.get("LAB_GRAFANA_URL", "http://localhost:3000") + "/api/health",
    "ollama": os.environ.get("LAB_OLLAMA_URL", "http://localhost:11434") + "/api/tags",
}
TIMEOUT_S = 2.0


def _run(cmd: list, timeout: float = 5.0) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _get_json(url: str, headers: dict | None = None, method: str = "GET", timeout: float = TIMEOUT_S):
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else None


def cpu_model() -> str | None:
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        return out.splitlines()[0].strip() if out else platform.processor() or None
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return platform.processor() or None


def hardware() -> dict:
    out = {"cpu_model": cpu_model(), "logical_cpus": os.cpu_count(), "physical_cpus": None, "ram_gib": None}
    try:
        import psutil

        out["physical_cpus"] = psutil.cpu_count(logical=False)
        out["ram_gib"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        out["note"] = "psutil not installed: physical cores and RAM unknown"
    gpu = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    out["gpus"] = [g.strip() for g in gpu.splitlines()] if gpu else []
    out["cloud"] = ec2_instance()
    return out


def ec2_instance() -> dict | None:
    """Instance type, region and AMI from the EC2 metadata service (IMDSv2); None outside EC2"""
    base = "http://169.254.169.254/latest"
    try:
        req = urllib.request.Request(f"{base}/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            token = resp.read().decode()
        doc = _get_json(f"{base}/dynamic/instance-identity/document", {"X-aws-ec2-metadata-token": token}, timeout=1.0)
    except Exception:  # noqa: BLE001 - not on EC2, or metadata disabled
        return None
    return {k: doc.get(k) for k in ("instanceType", "region", "availabilityZone", "imageId", "architecture")}


def software() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    # An editable install keeps the dist-info of the version it was installed at: the source's __version__ (or its
    # pyproject.toml) is what really runs, so it wins when it differs.
    for name in ("pulsar_data_collection", "pulsar_metrics"):
        source = _source_version(name)
        if source and source != versions.get(name):
            versions[name] = source if versions.get(name) is None else f"{source} (dist-info says {versions[name]})"
    return {
        "os": platform.platform(), "python": sys.version.split()[0], "python_executable": sys.executable,
        "packages": versions, "docker": _run(["docker", "version", "--format", "{{.Server.Version}}"]),
        "containers": _containers(),
    }


def _source_version(module: str) -> str | None:
    try:
        path = Path(importlib_file(module)).resolve().parent
    except ImportError:
        return None
    for candidate in (path.parent / "pyproject.toml", path / "pyproject.toml"):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("version") and "=" in line:
                    return line.split("=", 1)[1].strip().strip("\"'")
    return None


def _containers() -> list:
    out = _run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}"])
    return [dict(zip(("name", "image"), line.split("\t"))) for line in out.splitlines()] if out else []


def services() -> dict:
    out = {}
    for name, url in SERVICES.items():
        try:
            body = _get_json(url)
        except Exception as exc:  # noqa: BLE001 - a service that is down is a fact, not an error
            out[name] = {"url": url, "reachable": False, "reason": type(exc).__name__}
            continue
        entry = {"url": url, "reachable": True}
        if name == "ollama":
            entry["models"] = [{"name": m.get("name"), "digest": m.get("digest"),
                                "size_gb": round(m.get("size", 0) / 1e9, 2),
                                "quantization": (m.get("details") or {}).get("quantization_level")}
                               for m in (body or {}).get("models", [])]
        else:
            entry["version"] = (body or {}).get("version")
        out[name] = entry
    return out


def sources() -> dict:
    """Commit of the checkout each pulsar package is imported from (editable installs point at a git tree)"""
    out = {}
    for module in ("pulsar_data_collection", "pulsar_metrics"):
        try:
            path = Path(importlib_file(module)).resolve().parent
        except ImportError:
            out[module] = None
            continue
        out[module] = _git(path)
    out["pulsar_demo"] = _git(Path(__file__).resolve().parent)
    return out


def importlib_file(module: str) -> str:
    import importlib.util

    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise ImportError(module)
    return spec.origin


def _git(path: Path) -> dict:
    top = _run(["git", "-C", str(path), "rev-parse", "--show-toplevel"])
    if not top:
        return {"path": str(path), "git": None}
    return {
        "path": top, "commit": _run(["git", "-C", top, "rev-parse", "HEAD"]),
        "branch": _run(["git", "-C", top, "rev-parse", "--abbrev-ref", "HEAD"]),
        "tag": _run(["git", "-C", top, "describe", "--tags", "--exact-match"]),
        "dirty": bool(_run(["git", "-C", top, "status", "--porcelain", "--untracked-files=no"])),
    }


def collect(with_services: bool = True) -> dict:
    return {
        "collected_at": datetime.now(UTC).isoformat(timespec="seconds"), "host": platform.node(),
        "hardware": hardware(), "software": software(), "services": services() if with_services else None,
        "sources": sources(),
    }


def markdown(env: dict) -> str:
    hw, sw = env["hardware"], env["software"]
    rows = [
        ("Collected", env["collected_at"]), ("Host", env["host"]),
        ("CPU", f"{hw['cpu_model']} ({hw['physical_cpus']} cores / {hw['logical_cpus']} threads)"),
        ("RAM", f"{hw['ram_gib']} GiB"), ("GPU", ", ".join(hw["gpus"]) or "none"),
        ("Cloud", json.dumps(hw["cloud"]) if hw["cloud"] else "not EC2"), ("OS", sw["os"]),
        ("Python", sw["python"]), ("Docker", sw["docker"] or "n/a"),
    ]
    rows += [(f"`{k}`", v) for k, v in sw["packages"].items() if v]
    for name, s in (env.get("services") or {}).items():
        if not s.get("reachable"):
            rows.append((name, f"unreachable ({s.get('reason')})"))
        elif name == "ollama":
            rows += [(f"ollama `{m['name']}`", f"digest {str(m['digest'])[:12]}, {m['quantization']}") for m in s["models"]]
        else:
            rows.append((name, s.get("version")))
    for repo, g in env["sources"].items():
        if g and g.get("commit"):
            rows.append((repo, f"{g['commit'][:10]} ({g.get('tag') or g.get('branch')}{', dirty' if g['dirty'] else ''})"))
    return "| Item | Value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--no-services", action="store_true")
    args = parser.parse_args()
    env = collect(with_services=not args.no_services)
    text = markdown(env) if args.markdown else json.dumps(env, indent=2, default=str)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
