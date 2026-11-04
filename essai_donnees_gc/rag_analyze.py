"""Part 2 analysis: which Pulsar signals see each provoked change of the register agent, and after how many windows.

Signals, each phase cut into consecutive windows of `--window` runs and compared with the reference phase:

| Signal | Source | Test |
|---|---|---|
| trajectory and scores | `agent_run` records read back from InfluxDB: steps, tool calls, tokens, latency, best retrieval score, empty retrievals, faithfulness (NLI), first tool, termination, task success | the S2 analysis of `genai_agent/analyze_drift.py`: chi2 / ks_2samp calibrated by A/A draws on the reference, Benjamini-Hochberg, confirmation over 2 windows |
| request embeddings | `embeddings.npy` (nomic-embed-text) | MMD permutation test (pulsar_metrics.genai.mmd_test), confirmation over 2 windows |
| model identity | canary set (CanaryProbe) | model equality test (pulsar_metrics.genai.model_equality_test) |
| accuracy | agreement with the register (known answers; an evaluation, not a production signal) | Clopper-Pearson 95 % intervals |

The `det_eval_correct` and `det_target_retrieved` fields need the known answers: they are reported, never monitored.

    python rag_analyze.py --env-file ../../validation_e2e/.env --campaign registre-gc-1 --window 8
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "genai_agent"))

import analyze_drift  # noqa: E402
from pulsar_data_collection.genai import CanaryProbe  # noqa: E402
from pulsar_metrics.genai import mmd_test, model_equality_test  # noqa: E402

from influx_env import load_env, query  # noqa: E402
from rag_campaign import MODEL_ID, MODEL_VERSION  # noqa: E402

NUMERIC = ["steps", "tool_calls", "input_tokens", "output_tokens", "latency_s", "det_retrieval_top",
           "det_retrieval_empty", "det_faithfulness"]
CATEGORICAL = ["first_tool", "termination", "task_success"]


def exact_ci(k: int, n: int) -> list:
    lo = beta.ppf(0.025, k, n - k + 1) if k else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return [round(float(lo), 3), round(float(hi), 3)]


def load_records(login: dict, campaign: str) -> pd.DataFrame:
    flux = f"""
    from(bucket: "{login['bucket_name']}")
      |> range(start: -90d, stop: 1m)
      |> filter(fn: (r) => r["_measurement"] == "{MODEL_ID}_{MODEL_VERSION}_input_data")
      |> filter(fn: (r) => r["record_type"] == "agent_run" and r["campaign"] == "{campaign}")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    frame = query(login, flux).rename(columns={"_time": "pred_timestamp"})
    return analyze_drift.prepare(frame)


def first_streak(flags: list, length: int = 2):
    """1-based index of the window where `length` consecutive flags are first reached"""
    run = 0
    for k, flag in enumerate(flags, start=1):
        run = run + 1 if flag else 0
        if run >= length:
            return k
    return None


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--results", default=None, help="campaign folder (default results/<campaign>)")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)
    results = Path(args.results or HERE / "results" / args.campaign)
    login = load_env(args.env_file)

    # 1. trajectories and scores from InfluxDB, through the S2 analysis of the agent demo
    analyze_drift.MODEL_ID, analyze_drift.MODEL_VERSION = MODEL_ID, MODEL_VERSION
    analyze_drift.NUMERIC[:], analyze_drift.CATEGORICAL[:] = NUMERIC, CATEGORICAL
    records = load_records(login, args.campaign)
    drift = analyze_drift.analyze(records, window=args.window, alpha=args.alpha, persistence=2)

    # 2. request embeddings: MMD of every window against the reference
    runs = [json.loads(line) for line in open(results / "runs.jsonl", encoding="utf-8")]
    embeddings = np.load(results / "embeddings.npy")
    reference_idx = [r["index"] for r in runs if r["phase"] == "reference"]
    mmd = {}
    for phase in dict.fromkeys(r["phase"] for r in runs):
        if phase == "reference":
            continue
        idx = sorted(r["index"] for r in runs if r["phase"] == phase)
        windows = [idx[k:k + args.window] for k in range(0, len(idx) - args.window + 1, args.window)]
        p_values = [mmd_test(embeddings[reference_idx], embeddings[w], n_permutations=500, seed=0).p_value for w in windows]
        flags = [p < args.alpha for p in p_values]
        mmd[phase] = {"p_values": [round(float(p), 4) for p in p_values], "windows_flagged": sum(flags),
                      "first_flag": next((k for k, f in enumerate(flags, 1) if f), None),
                      "first_confirmed": first_streak(flags)}

    # 3. canary set
    canary = {}
    canary_dir = results / "canary"
    if (canary_dir / "reference.json").exists():
        ref = CanaryProbe.load(canary_dir / "reference.json")
        for name in ("same", "granite"):
            if (canary_dir / f"{name}.json").exists():
                test = model_equality_test(ref, CanaryProbe.load(canary_dir / f"{name}.json"), n_permutations=1000, seed=0)
                canary[name] = {"p": float(test.p_value), "statistic": float(test.statistic)}

    # 4. accuracy against the register, per phase and per kind of question
    accuracy = {}
    for phase in dict.fromkeys(r["phase"] for r in runs):
        rows = [r for r in runs if r["phase"] == phase]
        k = sum(r.get("eval_correct", 0.0) == 1.0 for r in rows)
        entry = {"runs": len(rows), "correct": k, "accuracy": round(k / len(rows), 3), "ci95": exact_ci(k, len(rows)),
                 "target_retrieved": round(float(np.mean([r.get("target_retrieved", 0.0) for r in rows])), 3),
                 "errors": sum(r["termination"] == "error" for r in rows),
                 "mean_seconds": round(float(np.mean([r["seconds"] for r in rows])), 1), "by_kind": {}}
        for kind in sorted({r["kind"] for r in rows}):
            sub = [r for r in rows if r["kind"] == kind]
            kk = sum(r.get("eval_correct", 0.0) == 1.0 for r in sub)
            entry["by_kind"][kind] = {"runs": len(sub), "accuracy": round(kk / len(sub), 3), "ci95": exact_ci(kk, len(sub))}
        accuracy[phase] = entry

    report = {"campaign": args.campaign, "window": args.window, "alpha": args.alpha, "records_read": len(records),
              "runs_logged": len(runs), "drift": drift, "mmd": mmd, "canary": canary, "accuracy": accuracy}
    (results / "analysis.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")

    lines = [analyze_drift.markdown(drift), "", "MMD of the request embeddings (windows flagged, first flag, first confirmed):", "",
             "| Phase | Windows flagged | First flag | First confirmed |", "|---|---|---|---|"]
    lines += [f"| {p} | {v['windows_flagged']} / {len(v['p_values'])} | {v['first_flag'] or '-'} | {v['first_confirmed'] or '-'} |"
              for p, v in mmd.items()]
    lines += ["", "Canary model equality: " + ", ".join(f"{k} p = {v['p']:.3f}" for k, v in canary.items()), "",
              "| Phase | Runs | Accuracy [95 % CI] | Target retrieved | Mean s/run |", "|---|---|---|---|---|"]
    lines += [f"| {p} | {v['runs']} | {v['accuracy']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] | "
              f"{v['target_retrieved']:.2f} | {v['mean_seconds']} |" for p, v in accuracy.items()]
    text = "\n".join(lines) + "\n"
    (results / "analysis.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
