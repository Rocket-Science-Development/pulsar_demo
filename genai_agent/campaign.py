"""Drift campaign: reference traffic, stable hold-out traffic, then injected drifts, in one tagged campaign.

Each phase calls run_agent.py with its own scenario label and seed; all records carry the `campaign` tag, so that
analyze_drift.py can rebuild the windows. Phases run in order; a failing phase is logged and the campaign goes on.

Usage:
    python campaign.py --campaign s2-cpu-1                      # full campaign (about 630 runs)
    python campaign.py --campaign test --scale 0.1              # 10 % of the runs, for a quick check
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent

# (label, runs, extra run_agent.py options) - reference first, then a stable hold-out, then one drift per phase
PHASES = [
    ("reference", 300, []),
    ("holdout", 60, []),
    ("tool_fault", 90, ["--timeout-rate", "0.3"]),
    ("model_change", 90, ["--model", "granite3.3:2b"]),
    ("question_shift", 90, ["--file-share", "0.6"]),
]


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (released automatically at exit; settings unchanged)"""
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--scale", type=float, default=1.0, help="multiply the number of runs of every phase")
    parser.add_argument("--only", nargs="*", help="run only these phases")
    args = parser.parse_args(argv)
    keep_awake()

    log_path = HERE / f"campaign_{args.campaign}.jsonl"
    print(f"campaign {args.campaign}: log {log_path}", flush=True)
    for i, (label, runs, options) in enumerate(PHASES):
        if args.only and label not in args.only:
            continue
        n = max(1, round(runs * args.scale))
        command = [
            sys.executable,
            str(HERE / "run_agent.py"),
            "--runs",
            str(n),
            "--label",
            label,
            "--campaign",
            args.campaign,
            "--seed",
            str(1000 + i),
            *options,
        ]
        started = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        print(f"\n=== phase {label}: {n} runs {' '.join(options)}", flush=True)
        result = subprocess.run(command, cwd=HERE)
        entry = {
            "campaign": args.campaign,
            "phase": label,
            "runs": n,
            "options": options,
            "started": started.isoformat(timespec="seconds"),
            "duration_s": round(time.perf_counter() - t0, 1),
            "returncode": result.returncode,
        }
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(entry) + "\n")
        print(f"=== phase {label} done in {entry['duration_s']} s (code {result.returncode})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
