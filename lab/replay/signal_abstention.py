"""Offline replay: abstentions and answers without support as drift signals of a RAG agent (LR-03 and the GC register).

No agent call: the answers recorded in lab/results/LR-03/runs.jsonl (1,800 runs, PIAF corpus, French answers) and in
essai_donnees_gc/results/registre-gc-1/runs.jsonl (240 runs on the Government of Canada AI Register) are judged again
with pulsar_data_collection.genai.abstention, then tested per window with pulsar_metrics `rate_increase` (one-sided
Fisher exact test) and the protocol of lab/replay/declencheur_derive.py, whose code is reused.

1. Abstention detector against hand labels (lab/replay/abstention_labels.json: 102 answers drawn at random within each
   group of both runs, labelled by reading the answer): precision and recall with Wilson intervals. With --nli, the same
   with the NLI fallback (FaithfulnessScorer.entailment, CPU) on the answers the rules leave undecided.
2. Per run: abstained (the detector), unsupported (answered although the retrieval was empty, the best retrieval score
   under its threshold or the faithfulness under its threshold). The thresholds are the 5th percentiles of the
   reference of each draw (calibrate_support_thresholds), never of the window.
3. Signals per window, against the reference:
       abstention         rate_increase on the abstentions (one-sided Fisher exact, window higher)
       unsupported        rate_increase on the answers without support
       answer_support     abstention or unsupported: smallest p-value x 2 (Bonferroni)
       trigger_score      the recommended trigger (faithfulness, empty_retrieval, relevance_bm25), x 3, as in
                          declencheur_derive.py
       trigger_support    trigger_score plus abstention and unsupported: smallest p-value x 5
       trigger_unsupported  trigger_score plus unsupported: smallest p-value x 4
       unsupported_entity, trigger_entity   register only, what-if: an answer also counts as without support when
                          the record of the system named in the question is not among the records returned
                          (`retrieval_relevant=False`). Measured here with target_retrieved, which matches record ids;
                          names are unique in the question pool, so an application can make the same check on the
                          name, without known answers
4. LR-03: the draws of declencheur_derive.py, same seeds (reference 300, 3 A/A windows of 100 from the 700 runs at 0 %;
   400 draws per corpus-loss level), A/A false-trigger rate and delay with DriftPersistence(k).
   Register: the same design scaled to the run, windows of 8 (reference 80 drawn from the 120 runs without change,
   3 A/A windows of 8 from the 40 others; 400 draws of a window of 8 from the 40 corpus-loss runs); the A/A draws reuse
   the same 120 runs, so their Wilson interval is only indicative. Plus the campaign order itself, as in
   rag_analyze.py: reference phase against each later phase cut into consecutive windows of 8 (and 20).

Run from pulsar_demo with pulsar_data_collection and pulsar_metrics (with rate_increase) importable:

    python -m lab.replay.signal_abstention                     # about 2 min on one CPU
    python -m lab.replay.signal_abstention --draws 200 --level-draws 50   # quick check
    python -m lab.replay.signal_abstention --nli               # adds the NLI fallback on the labelled sample
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import stats

from lab.replay.declencheur_derive import (
    ALPHAS,
    PERSISTENCE,
    aa_summary,
    delay_summary,
    load_runs,
    wilson,
)
from lab.replay.declencheur_derive import window_pvalues as base_pvalues

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT.parent
LR03 = ROOT / "results" / "LR-03" / "runs.jsonl"
REGISTER = DEMO / "essai_donnees_gc" / "results" / "registre-gc-1"
LABELS = Path(__file__).resolve().parent / "abstention_labels.json"
OUT = ROOT / "rapports" / "signal_abstention.json"
QUANTILE = 0.05  # thresholds of the support criteria: 5th percentile of the reference
COMPOSITES = {
    "trigger_score": ("faithfulness", "empty_retrieval", "relevance_bm25"),
    "answer_support": ("abstention", "unsupported"),
    "trigger_unsupported": ("faithfulness", "empty_retrieval", "relevance_bm25", "unsupported"),
    "trigger_support": ("faithfulness", "empty_retrieval", "relevance_bm25", "abstention", "unsupported"),
    # register only, what-if: the application checks that the system named in the question is in the records returned
    "trigger_entity": ("faithfulness", "empty_retrieval", "relevance_bm25", "unsupported_entity"),
}


# ------------------------------------------------------------------------------------------------ runs
def lr03_rows(runs: list, detector) -> list:
    """LR-03 runs with the fields of the window tests"""
    out = []
    for r in runs:
        verdict = detector.detect(r.get("answer"))
        out.append(
            {
                **r,
                "empty": bool(r["empty"]),
                "abstained": None if verdict is None else verdict.abstained,
                "kind": None if verdict is None else verdict.kind,
            }
        )
    return out


def register_rows(runs: list, detector, removed: set) -> list:
    """Register runs mapped to the LR-03 field names; no search counts as an empty retrieval"""
    out = []
    for r in sorted(runs, key=lambda r: r["index"]):
        verdict = detector.detect(r.get("answer"))
        out.append(
            {
                "phase": r["phase"],
                "index": r["index"],
                "affected": r["system"] in removed,
                "empty": bool(r.get("retrieval_empty")),
                "no_search": not r.get("tools"),
                "faithfulness": r.get("faithfulness"),
                "top1_bm25": r.get("retrieval_top") if r.get("tools") else None,
                "qp_cos": None,
                "retrieved": [],
                "correct": r.get("eval_correct") == 1.0,
                "relevant": r.get("target_retrieved") == 1.0 if r.get("tools") else None,
                "abstained": None if verdict is None else verdict.abstained,
                "kind": None if verdict is None else verdict.kind,
            }
        )
    return out


def support_flags(ref: list, win: list, relevance: bool = False) -> tuple:
    """unsupported (True / False / None) of the reference and window runs, thresholds fitted on the reference only;
    with `relevance`, the run's own relevance check (`relevant`) is added"""
    from pulsar_data_collection.genai import answer_support, calibrate_support_thresholds

    def searched(r):
        return not (r["empty"] or r["no_search"])

    thresholds = calibrate_support_thresholds(
        [r["top1_bm25"] for r in ref if searched(r)],
        [r["faithfulness"] for r in ref if searched(r) and r["abstained"] is False],
        quantile=QUANTILE,
    )

    def flag(r):
        return answer_support(
            r["abstained"],
            retrieval_empty=not searched(r),
            retrieval_top=r["top1_bm25"],
            faithfulness=r["faithfulness"],
            thresholds=thresholds,
            retrieval_relevant=r.get("relevant") if relevance else None,
        ).unsupported

    return [flag(r) for r in ref], [flag(r) for r in win]


# ------------------------------------------------------------------------------------------------ window tests
def _rate_p(win: list, ref: list):
    from pulsar_metrics.metrics.utils import rate_increase_test

    w, r = [float(v) for v in win if v is not None], [float(v) for v in ref if v is not None]
    return float(rate_increase_test(w, r).pvalue) if w and r else None


def _mw_lower(win, ref, minimum):
    w = [v for v in win if v is not None and not (isinstance(v, float) and math.isnan(v))]
    r = [v for v in ref if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(w) < minimum or len(r) < minimum:
        return None
    return float(stats.mannwhitneyu(w, r, alternative="less").pvalue)


def window_pvalues(ref: list, win: list, minimum: int = 5) -> dict:
    """p-values of a window: the base signals of declencheur_derive.py (no MMD) and the two new ones"""
    if minimum == 5:
        out = base_pvalues(ref, win, {}, 0, 0, mmd=False)
    else:  # small windows of the register: Mann-Whitney from 3 scored runs, as rag_analyze.py
        out = base_pvalues(ref, win, {}, 0, 0, mmd=False)
        out["faithfulness"] = _mw_lower([r["faithfulness"] for r in win], [r["faithfulness"] for r in ref], minimum)
        out["relevance_bm25"] = _mw_lower([r["top1_bm25"] for r in win], [r["top1_bm25"] for r in ref], minimum)
    out.pop("relevance_cos", None)
    ref_flags, win_flags = support_flags(ref, win)
    out["abstention"] = _rate_p([r["abstained"] for r in win], [r["abstained"] for r in ref])
    out["unsupported"] = _rate_p(win_flags, ref_flags)
    if "relevant" in win[0]:
        ref_flags, win_flags = support_flags(ref, win, relevance=True)
        out["unsupported_entity"] = _rate_p(win_flags, ref_flags)
    return add_composites(out)


def add_composites(pvalues: dict) -> dict:
    out = dict(pvalues)
    for name, members in COMPOSITES.items():
        if any(m not in pvalues for m in members):
            continue  # a signal this run does not have
        values = [pvalues.get(m) for m in members if pvalues.get(m) is not None]
        out[name] = min(1.0, len(members) * min(values)) if values else None
    return out


# ------------------------------------------------------------------------------------------------ draws
def aa_draws(zero: list, n_draws: int, reference: int, window: int, minimum: int) -> list:
    """declencheur_derive._aa_draw: a reference and 3 disjoint windows per draw, same seeds"""
    out = []
    for seed in range(n_draws):
        perm = np.random.default_rng(seed).permutation(len(zero))
        ref = [zero[i] for i in perm[:reference]]
        out.append(
            [
                window_pvalues(ref, [zero[i] for i in perm[reference + j * window : reference + (j + 1) * window]], minimum)
                for j in range(3)
            ]
        )
    return out


def level_draws(zero: list, pool: list, n_draws: int, reference: int, window: int, minimum: int) -> list:
    """declencheur_derive._level_draw: a reference from the runs without change, a window from the level, same seeds"""
    out = []
    for seed in range(n_draws):
        rng = np.random.default_rng(10_000 + seed)
        ref = [zero[i] for i in rng.choice(len(zero), reference, replace=False)]
        win = [pool[i] for i in rng.choice(len(pool), min(window, len(pool)), replace=False)]
        out.append(window_pvalues(ref, win, minimum))
    return out


def summarize(aa: list, levels: dict, signals: list) -> dict:
    return {
        "aa": {str(a): aa_summary(aa, signals, a) for a in ALPHAS},
        "levels": {
            str(a): {lv: {s: delay_summary(d, s, a, seed=i) for i, s in enumerate(signals)} for lv, d in levels.items()}
            for a in ALPHAS
        },
    }


def first_streak(flags: list, length: int = 2):
    run = 0
    for k, flag in enumerate(flags, start=1):
        run = run + 1 if flag else 0
        if run >= length:
            return k
    return None


def campaign_order(rows: list, window: int, alpha: float = 0.05) -> dict:
    """The register campaign as it ran: reference phase against consecutive windows of each later phase"""
    ref = [r for r in rows if r["phase"] == "reference"]
    out = {}
    for phase in [p for p in dict.fromkeys(r["phase"] for r in rows) if p != "reference"]:
        runs = [r for r in rows if r["phase"] == phase]
        windows = [runs[k : k + window] for k in range(0, len(runs) - window + 1, window)]
        pvalues = [window_pvalues(ref, w, minimum=3) for w in windows]
        out[phase] = {}
        for signal in pvalues[0]:
            flags = [p[signal] is not None and p[signal] < alpha for p in pvalues]
            out[phase][signal] = {
                "p_values": [None if p[signal] is None else round(p[signal], 4) for p in pvalues],
                "windows_flagged": sum(flags),
                "first_flag": next((k for k, f in enumerate(flags, 1) if f), None),
                "first_confirmed": first_streak(flags),
            }
    return out


# ------------------------------------------------------------------------------------------------ labels, rates
def label_check(detector, labels: list) -> dict:
    """Precision and recall of the detector on the hand labels, with Wilson intervals, overall and per source"""

    def score(rows):
        predicted = [bool(detector.detect(r["answer"]).abstained) for r in rows]
        truth = [r["abstention"] for r in rows]
        tp = sum(p and t for p, t in zip(predicted, truth))
        fp = sum(p and not t for p, t in zip(predicted, truth))
        fn = sum(t and not p for p, t in zip(predicted, truth))
        return {
            "n": len(rows),
            "positives": sum(truth),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "precision_ci": list(wilson(tp, tp + fp)),
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "recall_ci": list(wilson(tp, tp + fn)),
            "errors": [r["id"] for r, p, t in zip(rows, predicted, truth) if p != t],
        }

    return {
        "all": score(labels),
        **{src: score([r for r in labels if r["source"] == src]) for src in sorted({r["source"] for r in labels})},
    }


def run_rates(rows: list, group_key) -> dict:
    """Per group: runs, abstention rate, unsupported rate (thresholds from the runs without change), and among the
    runs whose answer was removed from the corpus (affected): abstained, unsupported, answered and not flagged"""
    zero = [r for r in rows if group_key(r) in ("0", "reference", "holdout")]
    out = {}
    groups = list(dict.fromkeys(group_key(r) for r in rows))
    for g in groups:
        sub = [r for r in rows if group_key(r) == g]
        _, flags = support_flags(zero, sub)
        judged = [(r, f) for r, f in zip(sub, flags) if r["abstained"] is not None]
        affected = [(r, f) for r, f in judged if r.get("affected")]
        n, k_a, k_u = len(judged), sum(r["abstained"] for r, _ in judged), sum(bool(f) for _, f in judged)
        out[g] = {
            "runs": n,
            "abstention_rate": round(k_a / n, 4),
            "abstention_ci": list(wilson(k_a, n)),
            "unsupported_rate": round(k_u / n, 4),
            "unsupported_ci": list(wilson(k_u, n)),
            "hedged": sum(r["kind"] == "hedged" for r, _ in judged),
            "affected": len(affected),
            "affected_abstained": sum(r["abstained"] for r, _ in affected),
            "affected_unsupported": sum(bool(f) for _, f in affected),
            "affected_answered_unflagged": sum(not r["abstained"] and not f for r, f in affected),
            "affected_answered_unflagged_correct": sum(not r["abstained"] and not f and r["correct"] for r, f in affected),
        }
        if "relevant" in sub[0]:
            _, entity = support_flags(zero, sub, relevance=True)
            judged_e = [(r, f) for r, f in zip(sub, entity) if r["abstained"] is not None]
            k_e = sum(bool(f) for _, f in judged_e)
            out[g]["unsupported_entity_rate"] = round(k_e / n, 4)
            out[g]["unsupported_entity_ci"] = list(wilson(k_e, n))
            out[g]["affected_unsupported_entity"] = sum(bool(f) for r, f in judged_e if r.get("affected"))
    return out


def nli_check(labels: list) -> dict:
    """Rules + NLI fallback (FaithfulnessScorer.entailment) on the labelled sample, threshold 0.5"""
    from pulsar_data_collection.genai import AbstentionDetector, FaithfulnessScorer, NliAbstentionFallback

    t0 = time.perf_counter()
    detector = AbstentionDetector(fallback=NliAbstentionFallback.from_scorer(FaithfulnessScorer()))
    out = label_check(detector, labels)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lr03", type=Path, default=LR03)
    parser.add_argument("--register", type=Path, default=REGISTER)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--draws", type=int, default=2000, help="A/A draws (3 windows each)")
    parser.add_argument("--level-draws", type=int, default=400, help="draws per degradation level")
    parser.add_argument("--nli", action="store_true", help="also measure the NLI fallback on the labelled sample")
    args = parser.parse_args(argv)
    from pulsar_data_collection.genai import AbstentionDetector

    t0 = time.perf_counter()
    detector = AbstentionDetector()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))["labels"]
    result = {
        "design": {
            "quantile": QUANTILE,
            "composites": COMPOSITES,
            "alphas": list(ALPHAS),
            "persistence": list(PERSISTENCE),
            "aa_draws": args.draws,
            "level_draws": args.level_draws,
        },
        "labels": label_check(detector, labels),
    }
    if args.nli:
        result["labels_nli"] = nli_check(labels)

    # LR-03: 1,800 runs, windows of 100
    lr = lr03_rows(load_runs(args.lr03), detector)
    lr_zero = [r for r in lr if r["label"] in ("reference", "aa")]

    def lr_key(r):
        return "total" if r["label"] == "total_failure" else str(round(100 * r["level"]))

    lr_aa = aa_draws(lr_zero, args.draws, 300, 100, 5)
    lr_levels = {
        lv: level_draws(lr_zero, [r for r in lr if lr_key(r) == lv], args.level_draws, 300, 100, 5)
        for lv in ("10", "20", "40", "60", "80", "total")
    }
    lr_signals = sorted(lr_aa[0][0])
    result["lr03"] = {"rates": run_rates(lr, lr_key), **summarize(lr_aa, lr_levels, lr_signals)}

    # register: 240 runs, windows of 8
    removed = set(json.loads((args.register / "setup.json").read_text(encoding="utf-8"))["corpus_loss_removed"])
    rg = register_rows(load_runs(args.register / "runs.jsonl"), detector, removed)
    rg_zero = [r for r in rg if r["phase"] in ("reference", "holdout")]
    rg_aa = aa_draws(rg_zero, args.draws, 80, 8, 3)
    rg_levels = {"corpus_loss": level_draws(rg_zero, [r for r in rg if r["phase"] == "corpus_loss"], args.level_draws, 80, 8, 3)}
    result["register"] = {
        "rates": run_rates(rg, lambda r: r["phase"]),
        **summarize(rg_aa, rg_levels, sorted(rg_aa[0][0])),
        "campaign_w8": campaign_order(rg, 8),
        "campaign_w20": campaign_order(rg, 20),
    }
    result["seconds"] = round(time.perf_counter() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    lab = result["labels"]["all"]
    print(f"labels: precision {lab['precision']} {lab['precision_ci']}  recall {lab['recall']} {lab['recall_ci']}")
    for name, block in (("LR-03", result["lr03"]), ("register", result["register"])):
        for s in (
            "abstention",
            "unsupported",
            "answer_support",
            "trigger_score",
            "trigger_unsupported",
            "trigger_support",
            "unsupported_entity",
            "trigger_entity",
        ):
            if s not in block["aa"]["0.05"]:
                continue
            aa = block["aa"]["0.05"][s]
            line = f"{name:8s} {s:20s} A/A k1 {aa['k1']['rate']} {aa['k1']['ci']} k2 {aa['k2']['rate']} {aa['k2']['ci']}"
            for lv, d in block["levels"]["0.05"].items():
                line += f" | {lv}: power {d[s]['power']} k2 median {d[s]['k2']['median_delay']}"
            print(line)
    print(f"{args.out} written in {result['seconds']} s")
    return result


if __name__ == "__main__":
    main()
