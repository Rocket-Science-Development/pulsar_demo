"""Analysis of the consistency campaign: window metrics, drift against the reference, A/A false alarms, evidence.

Reference: llama3.2:3b at temperature 0.7, answers 0-2 of each question. A/A holdout: answers 3-5 of the same run.
Changed conditions: granite3.3:2b at 0.7, llama3.2:3b and granite3.3:2b at 1.5. The A/A false-alarm rate is also measured on
random splits of the 6 llama answers of each question into two halves of 3.

    python analyze.py --campaign coherence-1 [--splits 500] [--embeddings granite-embedding:278m]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pulsar_data_collection.genai import ConsistencyProbe, EmbeddingSimilarity
from pulsar_metrics.evidence import dumps, to_evidence
from pulsar_metrics.genai import consistency, consistency_drift
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from questions import questions  # noqa: E402

SYSTEM_ID = "demo-coherence-llm"
CHANGED = ("llama_t07_holdout", "granite_t07", "llama_t15", "granite_t15")


def load(out_dir: Path, name: str) -> tuple:
    data = json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))
    return data, pd.DataFrame(data["answers"])


def bootstrap_mean(values, n_boot=5000, seed=0) -> list:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return [float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summary(name: str, frame: pd.DataFrame, answers: pd.DataFrame, model: str) -> dict:
    agreement, full = consistency(frame, model=model, n_boot=5000)
    return {
        "condition": name,
        "questions": int(agreement.n),
        "answers": int(len(answers)),
        "errors": int((answers["answer_status"] == "error").sum()),
        "unparsed_share": float((answers["answer_status"] == "unparsed").mean()),
        "refusal_share": float((answers["answer_status"] == "refusal").mean()),
        "agreement": [agreement.metric_value, agreement.ci_low, agreement.ci_high],
        "full_agreement": [full.metric_value, full.ci_low, full.ci_high],
        "accuracy": bootstrap_mean(frame["correct_share"].astype(float)),
        "latency_median_s": float(answers["latency_s"].median()),
    }


def drift_row(label: str, result) -> dict:
    return {
        "comparison": label,
        "metric": result.metric_name,
        "difference": result.metric_value,
        "ci": [result.ci_low, result.ci_high],
        "p_value": result.p_value,
        "alert": result.drift_status,
        "reliable": result.reliable is not False,
    }


def wilson(k: int, n: int) -> list:
    ci = stats.binomtest(k, n).proportion_ci(method="wilson")
    return [k / n, float(ci.low), float(ci.high)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", default="coherence-1")
    parser.add_argument("--splits", type=int, default=500)
    parser.add_argument("--embeddings", default=None, help="embedding model for the free-text view (e.g. granite-embedding:278m)")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    args = parser.parse_args()
    out_dir = HERE / "results" / args.campaign
    probe = ConsistencyProbe(questions(), repeats=3)

    meta_l, llama = load(out_dir, "llama_t07")
    meta_g, granite = load(out_dir, "granite_t07")
    meta_h, hot = load(out_dir, "llama_t15")
    meta_gh, granite_hot = load(out_dir, "granite_t15")
    halves = {"llama_t07_ref": llama[llama["repeat"] < 3], "llama_t07_holdout": llama[llama["repeat"] >= 3]}
    answers = {**halves, "granite_t07": granite, "llama_t15": hot, "granite_t15": granite_hot}
    models = {"llama_t07_ref": "llama3.2:3b", "llama_t07_holdout": "llama3.2:3b", "granite_t07": "granite3.3:2b"}
    models["llama_t15"], models["granite_t15"] = "llama3.2:3b", "granite3.3:2b"
    frames = {name: probe.score(a) for name, a in answers.items()}

    report = {
        "campaign": args.campaign,
        "elapsed_s": {m["condition"]: m["elapsed_s"] for m in (meta_l, meta_g, meta_h, meta_gh)},
        "settings": {
            m["condition"]: {k: m[k] for k in ("model", "temperature", "repeats", "max_tokens", "workers")}
            for m in (meta_l, meta_g, meta_h, meta_gh)
        },
        "windows": [summary(name, frames[name], answers[name], models[name]) for name in answers],
        "drift": [],
    }
    ref = frames["llama_t07_ref"]
    evidence = [(r, "llama_t07_ref") for r in consistency(ref, model="llama3.2:3b")]  # (result, window id)
    for name in CHANGED:
        for statistic in ("agreement", "full_agreement"):
            result = consistency_drift(ref, frames[name], model=models[name], statistic=statistic, n_permutations=9999)
            report["drift"].append(drift_row(f"{name} vs llama_t07_ref", result))
            evidence.append((result, name))
        evidence += [(r, name) for r in consistency(frames[name], model=models[name])]

    # A/A: random splits of the 6 answers of each question into 3 + 3
    rng = np.random.default_rng(0)
    ok = llama[llama["answer_status"] != "error"]
    alarms = {"agreement": 0, "full_agreement": 0}
    for _ in range(args.splits):
        flags = ok.groupby("question_id")["repeat"].transform(lambda r: rng.permutation(len(r)) < len(r) // 2)
        a, b = probe.score(ok[flags.astype(bool)]), probe.score(ok[~flags.astype(bool)])
        for statistic in alarms:
            alarms[statistic] += bool(consistency_drift(a, b, statistic=statistic, n_permutations=999, n_boot=200).drift_status)
    report["aa_splits"] = {"splits": args.splits, **{s: wilson(k, args.splits) for s, k in alarms.items()}}

    if args.embeddings:
        sim = EmbeddingSimilarity(args.base_url, model=args.embeddings)
        free = ConsistencyProbe([{**q, "answer_type": "free", "labels": ()} for q in questions()], repeats=3, similarity=sim)
        report["embedding_view"] = {"model": args.embeddings}
        for name, a in answers.items():
            frame = free.score(a)
            report["embedding_view"][name] = bootstrap_mean(frame["agreement"].dropna())
        e_ref = free.score(answers["llama_t07_ref"])
        for name in CHANGED:
            result = consistency_drift(e_ref, free.score(answers[name]), n_permutations=9999)
            report["embedding_view"][f"drift {name}"] = drift_row(name, result)

    reference = {"id": f"{args.campaign}/llama_t07_ref"}
    records = [to_evidence(r, SYSTEM_ID, window={"id": f"{args.campaign}/{w}"}, reference=reference)[0] for r, w in evidence]
    (out_dir / "evidence.json").write_text(dumps(records), encoding="utf-8")
    report["evidence"] = [
        {
            "name": r.metric.name,
            "feature": r.metric.feature,
            "family": r.metric.family,
            "status": r.metric.status,
            "direction": r.metric.direction,
            "basis": r.metric.decision_basis,
            "reliable": r.reliability.reliable,
            "reasons": r.reliability.reasons,
            "notes": r.reliability.notes,
        }
        for r in records
    ]
    for name, frame in frames.items():
        frame.drop(columns=["pair_scores"]).to_csv(out_dir / f"questions_{name}.csv", index=False)
    (out_dir / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("windows", "drift", "aa_splits")}, indent=1))
    if "embedding_view" in report:
        print(json.dumps(report["embedding_view"], indent=1))


if __name__ == "__main__":
    main()
