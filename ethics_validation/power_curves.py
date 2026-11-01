"""Power and false-alarm curves of the Pulsar.ml ethics and bias modules (validation E5).

For each module, a bias of known size is injected and the share of runs that establish it is measured, by sample
size. The row "bias 0" gives the false-alarm rate. Experiments:

- fixed:    FairnessAnalyzer, demographic parity difference (zone [-0.1, 0.1]), semi-synthetic eligibility decisions;
            power by size of the minority group (reference group: 2,000 rows).
- probe:    counterfactual_analysis of paired probe rows (rate of an action, zone [-0.1, 0.1]; excess flip rate over
            placebo, tolerance 0.05); power by number of sets.
- explain:  AttributionDrift on ACSIncome (Folktables) TreeSHAP attributions; the current window mixes rows explained by
            a model retrained on labels biased against women; power by window size. Needs folktables, shap and
            --data-dir.
- sequential: no simulation here, the curves come from the bias-injection bench of pulsar_metrics
            (benchmarks/fairness_drift.py), passed with --sequential-json.

    python power_curves.py --experiments fixed probe --out results
    python power_curves.py --experiments explain --data-dir C:/dev/pulsar-ml/data/folktables --out results
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pulsar_metrics.fairness import FairnessAnalyzer, FairnessPolicy, counterfactual_analysis

from synthetic_eligibility import eligibility_decisions


def rates(statuses):
    statuses = list(statuses)
    n = len(statuses)
    return {
        "established": sum(s is True for s in statuses) / n,
        "within": sum(s is False for s in statuses) / n,
        "inconclusive": sum(s is None for s in statuses) / n,
    }


def fixed(runs, seed):
    sizes, deltas = [30, 50, 100, 200, 500, 1000], [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    rng = np.random.default_rng(seed)
    cells = []
    for size in sizes:
        for delta in deltas:
            statuses = []
            for _ in range(runs):
                data = eligibility_decisions(
                    0,
                    bias={"language": {"other": delta}},
                    group_sizes={"language": {"fr": 2000, "other": size}},
                    seed=int(rng.integers(1 << 31)),
                )
                analyzer = FairnessAnalyzer(
                    ["language"],
                    FairnessPolicy(criteria=["demographic_parity_difference"], min_group_size=30, n_bootstrap=200),
                    reference_groups={"language": "fr"},
                    seed=int(rng.integers(1 << 31)),
                )
                analyzer.run(data)
                result = [r for r in analyzer.get_result() if r.group == "language=other"][0]
                statuses.append(result.drift_status)
            cells.append({"size": size, "delta": delta, **rates(statuses)})
            print(f"fixed size {size:5d} delta {delta:.2f} {cells[-1]}", flush=True)
    return {"zone": [-0.1, 0.1], "reference_size": 2000, "base_rate": 0.5, "cells": cells}


def probe_rows(n_sets, base, delta, noise, rng):
    """Paired rows: the request calls for the action with probability `base`; the probed group gets `delta` more;
    each answer flips at random with probability `noise` (the placebo sets show that noise floor)"""
    u = rng.random(n_sets)
    rows = []
    for variant, p in (("ref", base), ("group", base + delta)):
        act = (u < p) != (rng.random(n_sets) < noise)
        rows.append(pd.DataFrame({"set_id": [f"cf-{i}" for i in range(n_sets)], "variant": variant, "control": "none",
                                  "attribute": "name", "outcome": np.where(act, "act", "other")}))
    u = rng.random(n_sets)
    for variant in ("pa", "pb"):
        act = (u < base) != (rng.random(n_sets) < noise)
        rows.append(pd.DataFrame({"set_id": [f"pl-{i}" for i in range(n_sets)], "variant": variant, "control": "placebo",
                                  "attribute": "reference", "outcome": np.where(act, "act", "other")}))
    frame = pd.concat(rows, ignore_index=True)
    frame["error"] = None
    return frame


def probe(runs, seed, base=0.05, noise=0.02):
    sizes, deltas = [30, 60, 100, 200, 400], [0.0, 0.05, 0.1, 0.2, 0.33]
    rng = np.random.default_rng(seed)
    cells = []
    for n_sets in sizes:
        for delta in deltas:
            action, excess = [], []
            for _ in range(runs):
                results = counterfactual_analysis(
                    probe_rows(n_sets, base, delta, noise, rng), "ref", action="act", n_resamples=500,
                    seed=int(rng.integers(1 << 31)),
                )
                action.append(next(r for r in results if r.metric_name == "counterfactual_action_rate_difference").drift_status)
                excess.append(next(r for r in results if r.metric_name == "counterfactual_excess_flip_rate").drift_status)
            cells.append({"sets": n_sets, "delta": delta, "action": rates(action), "excess_flip": rates(excess)})
            print(f"probe sets {n_sets:4d} delta {delta:.2f} action {cells[-1]['action']} excess {cells[-1]['excess_flip']}",
                  flush=True)
    return {"base_rate": base, "noise": noise, "zone": [-0.1, 0.1], "flip_tolerance": 0.05, "cells": cells}


def explain(runs, seed, data_dir):
    from folktables import ACSDataSource, ACSIncome
    from pulsar_metrics.explain import AttributionDrift, importance, tree_attributions
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person", root_dir=data_dir)
    X, y, _ = ACSIncome.df_to_pandas(source.get_data(states=["MA"], download=False))
    y = y.iloc[:, 0].astype(int)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    v1 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0).fit(X_train, y_train)
    rng0 = np.random.default_rng(0)
    biased = y_train.copy()
    biased[(X_train["SEX"] == 2) & (biased == 1) & (rng0.random(len(biased)) < 0.3)] = 0
    v2 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0).fit(X_train, biased)
    phi1, phi2 = tree_attributions(v1, X_test), tree_attributions(v2, X_test)
    shift = float(importance(phi2)["SEX"] - importance(phi1)["SEX"])

    sizes, mixes = [100, 250, 500, 1000, 2000], [0.0, 0.25, 0.5, 1.0]
    rng = np.random.default_rng(seed)
    cells = []
    for n in sizes:
        for mix in mixes:
            sex, any_alarm = [], []
            for _ in range(runs):
                order = rng.permutation(len(phi1))
                reference = phi1.iloc[order[:n]]
                current_rows = order[n : 2 * n]
                from_v2 = rng.random(n) < mix
                current = pd.concat([phi1.iloc[current_rows[~from_v2]], phi2.iloc[current_rows[from_v2]]])
                drift = AttributionDrift(k=5, n_bootstrap=100, seed=int(rng.integers(1 << 31))).fit(reference)
                results = drift.run(current)
                sex.append(next(r for r in results if r.feature_name == "SEX").drift_status)
                any_alarm.append(True if any(r.drift_status is True for r in results) else False)
            cells.append({"window": n, "mix": mix, "sex_share": rates(sex),
                          "any_alarm": sum(any_alarm) / len(any_alarm)})
            print(f"explain window {n:5d} mix {mix:.2f} {cells[-1]}", flush=True)
    return {"sex_share_shift_full_mix": shift, "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", default=["fixed", "probe"], choices=["fixed", "probe", "explain"])
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--explain-runs", type=int, default=50)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.experiments:
        started = time.perf_counter()
        if name == "fixed":
            result = fixed(args.runs, args.seed)
        elif name == "probe":
            result = probe(args.runs, args.seed)
        else:
            result = explain(args.explain_runs, args.seed, args.data_dir)
        result["runs"] = args.explain_runs if name == "explain" else args.runs
        result["seconds"] = round(time.perf_counter() - started, 1)
        (out / f"power_{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{name}: {result['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
