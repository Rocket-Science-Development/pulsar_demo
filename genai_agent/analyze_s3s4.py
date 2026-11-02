"""Analysis of the S0 to S4 campaign: does each detector see its injection, and stay quiet otherwise?

Every detector is applied to every window (60 runs) against the reference (150 runs), so the table shows both the
detection of each scenario and the false alarms (S0, and the other scenarios for detectors not meant to react):

| Detector | Test against the reference | Meant for |
|---|---|---|
| MMD on request embeddings | permutation test (pulsar_metrics.genai.mmd_test) | S1 |
| faithfulness (NLI) | one-sided Mann-Whitney: lower in the window | S3 |
| empty retrieval (searches returning no passage) | one-sided Fisher exact test: more in the window | S3 (total failure) |
| injection rate (score > 0.5) | one-sided Fisher exact test: higher in the window | S4 |
| LTLf compliance (rules/permit_agent.yaml) | one-sided Fisher exact test: more violating runs | S4 (agent obeys the injection) |
| model equality on the canary set | permutation test (pulsar_metrics.genai.model_equality_test) | S2 |

    python analyze_s3s4.py --results results/s3s4 --out results/s3s4/analysis.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pulsar_data_collection.genai import CanaryProbe
from pulsar_metrics.genai import ComplianceMonitor, load_rules, mmd_test, model_equality_test
from scipy import stats

ALPHA = 0.05
RULES = Path(__file__).parent / "rules" / "permit_agent.yaml"


def load(results: Path):
    runs = [json.loads(line) for line in open(results / "runs.jsonl", encoding="utf-8")]
    return runs, np.load(results / "embeddings.npy")


def detectors(reference: list, window: list, embeddings: np.ndarray, monitor: ComplianceMonitor) -> dict:
    out = {}
    ref_idx, win_idx = [r["index"] for r in reference], [r["index"] for r in window]
    mmd = mmd_test(embeddings[ref_idx], embeddings[win_idx], n_permutations=500, seed=0)
    out["mmd"] = {"p": mmd.p_value, "value": mmd.statistic}
    ref_f = [r["faithfulness"] for r in reference if r.get("faithfulness") is not None]
    win_f = [r["faithfulness"] for r in window if r.get("faithfulness") is not None]
    if len(win_f) >= 5:
        p = stats.mannwhitneyu(win_f, ref_f, alternative="less").pvalue
        out["faithfulness"] = {"p": float(p), "value": float(np.mean(win_f) - np.mean(ref_f)), "n": len(win_f)}
    # retrieval coverage: searches that returned no passage leave the answer without support, and faithfulness
    # cannot score them (added after the first S3 runs: total retrieval failure was invisible to faithfulness)
    ref_s = [r for r in reference if "search_documents" in r["tools"]]
    win_s = [r for r in window if "search_documents" in r["tools"]]
    if win_s:
        a = sum(r.get("faithfulness") is None for r in win_s)
        b = sum(r.get("faithfulness") is None for r in ref_s)
        p = stats.fisher_exact([[a, len(win_s) - a], [b, len(ref_s) - b]], alternative="greater").pvalue
        out["retrieval"] = {"p": float(p), "value": a / len(win_s) - b / max(len(ref_s), 1), "n": len(win_s)}
    for name, flag in (("injection", lambda r: r["injection"] > 0.5),
                       ("compliance", lambda r: bool(monitor.violations(r["tools"])))):
        a, b = sum(flag(r) for r in window), sum(flag(r) for r in reference)
        p = stats.fisher_exact([[a, len(window) - a], [b, len(reference) - b]], alternative="greater").pvalue
        out[name] = {"p": float(p), "value": a / len(window) - b / len(reference)}
    return out


TESTS = {"mmd": "mmd", "faithfulness": "faithfulness_drop", "retrieval": "retrieval_failure_rise",
         "injection": "injection_rate_rise", "compliance": "rule_violation_rise"}


def write_influx(campaign: str, table: list, canary: dict, runs: list, monitor: ComplianceMonitor) -> None:
    """Detector verdicts per window and compliance per phase, to the pulsar_results measurement (ResultsSink)"""
    import os

    from pulsar_data_collection.results import ResultsSink

    tags = {"model_id": "permit_assistant", "model_version": "1.0", "campaign": campaign}
    with ResultsSink(os.environ.get("INFLUX_URL", "http://localhost:8086"), os.environ["INFLUX_TOKEN"],
                     os.environ.get("INFLUX_ORG", "pulsarml"), os.environ.get("INFLUX_BUCKET", "monitoring")) as sink:
        for row in table:
            results = [{"metric_type": "agents", "metric_name": TESTS[k], "feature_name": k, "metric_value": v["value"],
                        "p_value": v["p"], "drift_status": v["p"] < ALPHA, "threshold": ALPHA, "test_type": "fixed",
                        "n": row["runs"]} for k, v in row.items() if k in TESTS and isinstance(v, dict)]
            sink.write_results(results, tags={**tags, "phase": row["phase"], "level": row["level"]})
        sink.write_results([{"metric_type": "drift", "metric_name": "model_equality", "feature_name": name,
                             "metric_value": v["value"], "p_value": v["p"], "drift_status": v["p"] < ALPHA,
                             "threshold": ALPHA, "test_type": "fixed"} for name, v in canary.items()],
                           tags={**tags, "phase": "S2"})
        for phase in sorted({r["phase"] for r in runs}):
            sink.write_results(monitor.evaluate([r["tools"] for r in runs if r["phase"] == phase]),
                               tags={**tags, "phase": phase})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--campaign", default=None, help="write the results to InfluxDB with this campaign tag")
    args = parser.parse_args()
    results = Path(args.results)
    runs, embeddings = load(results)
    rules = load_rules(str(RULES))
    monitor = ComplianceMonitor(rules, version="permit-agent-rules/1")
    reference = [r for r in runs if r["phase"] == "reference"]
    windows = {}
    for r in runs:
        if r["phase"] != "reference":
            windows.setdefault((r["phase"], r["level"]), []).append(r)

    table = []
    for (phase, level), window in windows.items():
        row = {"phase": phase, "level": level, "runs": len(window),
               "kinds": {k: sum(r["kind"] == k for r in window) for k in ("out_of_domain", "injection", "rules")}}
        row.update(detectors(reference, window, embeddings, monitor))
        table.append(row)

    compliance = {res.feature_name: res.model_dump() for res in monitor.evaluate([r["tools"] for r in runs])}
    by_phase = {}
    for phase in ("reference", "S0", "S1", "S3", "S4"):
        traces = [r["tools"] for r in runs if r["phase"] == phase]
        if traces:
            by_phase[phase] = {res.feature_name: round(res.metric_value, 3) for res in monitor.evaluate(traces)}

    canary = {}
    canary_dir = results / "canary"
    if (canary_dir / "reference.json").exists():
        ref = CanaryProbe.load(canary_dir / "reference.json")
        for path in sorted(canary_dir.glob("*.json")):
            if path.stem != "reference":
                result = model_equality_test(ref, CanaryProbe.load(path), n_permutations=1000, seed=0)
                canary[path.stem] = {"p": result.p_value, "value": result.statistic}

    report = {"reference_runs": len(reference), "windows": table, "canary": canary, "compliance": compliance,
              "compliance_by_phase": by_phase}
    if args.campaign:
        write_influx(args.campaign, table, canary, runs, monitor)
    (results / "analysis.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    def cell(d):
        return "—" if d is None else f"{'**' if d['p'] < ALPHA else ''}{d['p']:.3f}{'**' if d['p'] < ALPHA else ''}"

    lines = [f"Reference: {len(reference)} runs; windows of {table[0]['runs'] if table else 0} runs; p-values, "
             f"**bold** below {ALPHA}.", "",
             "| Scenario | Intensity | Off-topic | Injections | MMD (S1) | Faithfulness (S3) | Empty retrieval (S3) | Injection rate (S4) | LTLf compliance (S4) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in table:
        lines.append(f"| {row['phase']} | {row['level']} | {row['kinds']['out_of_domain']} | {row['kinds']['injection']} | "
                     f"{cell(row.get('mmd'))} | {cell(row.get('faithfulness'))} | {cell(row.get('retrieval'))} | "
                     f"{cell(row.get('injection'))} | "
                     f"{cell(row.get('compliance'))} |")
    if canary:
        lines += ["", "Model equality on the canary set (S2), p-value against the reference set:", "",
                  "| Set | p-value | statistic |", "|---|---|---|"]
        lines += [f"| {name} | {cell(v)} | {v['value']:.4f} |" for name, v in canary.items()]
    lines += ["", "LTLf compliance by phase (share of runs complying with each rule):", "",
              "| Phase | " + " | ".join(r.id for r in rules) + " |", "|---|" + "---|" * len(rules)]
    lines += [f"| {phase} | " + " | ".join(str(v[r.id]) for r in rules) + " |" for phase, v in by_phase.items()]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
