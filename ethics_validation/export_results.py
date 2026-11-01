"""Export the results of the ethics and bias modules as one JSON Lines file, for the governance layer.

Each line is a `MetricResults` of pulsar_metrics (fields with no value are left out) plus its provenance:
`scenario`, `module`, `dataset`, `pulsar_metrics_version`, `generated_at`. The governance layer maps them to the
clauses of the Directive on Automated Decision-Making and of the AIA; nothing here depends on it.

Scenarios (all reproducible, no real personal data):
- fixed:      FairnessAnalyzer on semi-synthetic eligibility decisions, approval lowered by 0.20 for "other"
              language, with intersections language x region;
- sequential: FairnessMonitor on a stream of the same decisions, bias injected after 4,000 decisions;
- explain:    AttributionDrift and importance_by_group on synthetic attributions (a feature gains weight);
- probe:      counterfactual_analysis and comparative_analysis of the rows saved by probe_agent.py (--probe-rows).

    python export_results.py --out metric_results.jsonl --probe-rows probe_biased_rows.csv
"""

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from pulsar_metrics.explain import AttributionDrift, importance_by_group
from pulsar_metrics.fairness import (
    FairnessAnalyzer,
    FairnessMonitor,
    FairnessPolicy,
    comparative_analysis,
    counterfactual_analysis,
)

from synthetic_eligibility import eligibility_decisions


def fixed():
    data = eligibility_decisions(20000, bias={"language": {"other": 0.2}}, seed=11)
    analyzer = FairnessAnalyzer(
        ["language", "region"],
        FairnessPolicy(criteria=["disparate_impact_ratio", "demographic_parity_difference"], min_group_size=30),
        reference_groups={"language": "fr"},
        intersections=True,
        seed=0,
    )
    analyzer.run(data)
    return analyzer.get_result(), "semi-synthetic eligibility, 20,000 rows, approval -0.20 for language=other"


def sequential():
    monitor = FairnessMonitor("language", "fr", ["en", "other"], tolerance=0.05)
    for k in range(60):
        bias = {"language": {"other": 0.2}} if k >= 8 else None
        results = monitor.update(eligibility_decisions(500, bias=bias, seed=100 + k))
    return results, "semi-synthetic stream, 60 batches of 500, approval -0.20 for language=other after 4,000 decisions"


def explain():
    rng = np.random.default_rng(5)
    columns = ["income", "hours", "age", "education", "region", "tenure"]
    reference = pd.DataFrame(rng.normal(0, 1, (3000, 6)) * [1.0, 0.7, 0.5, 0.3, 0.15, 0.05], columns=columns)
    current = pd.DataFrame(rng.normal(0, 1, (3000, 6)) * [1.0, 0.7, 0.5, 0.3, 0.9, 0.05], columns=columns)
    drift = AttributionDrift(k=3).fit(reference)
    results = drift.run(current)
    groups = pd.Series(np.where(rng.random(3000) < 0.3, "other", "fr"), name="language")
    results += importance_by_group(current, groups, reference_group="fr")
    return results, "synthetic attributions, region gains weight in the current window"


def probe(rows_path, comparative_path):
    rows = pd.read_csv(rows_path)
    rows = rows.astype({"error": object}).where(rows.notna(), None)
    results = counterfactual_analysis(rows, "fr_qc", action="authenticate")
    if comparative_path and Path(comparative_path).exists():
        cmp_rows = pd.read_csv(comparative_path)
        cmp_rows = cmp_rows.astype({"error": object}).where(cmp_rows.notna(), None)
        results += comparative_analysis(cmp_rows)
    return results, f"counterfactual probe of the demo agent ({Path(rows_path).name})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="metric_results.jsonl")
    parser.add_argument("--probe-rows", default=None)
    parser.add_argument("--probe-comparative-rows", default=None)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    common = {"pulsar_metrics_version": version("pulsar-metrics"), "generated_at": stamp}
    scenarios = [("fixed", "fairness", fixed), ("sequential", "fairness.sequential", sequential), ("explain", "explain", explain)]
    if args.probe_rows:
        scenarios.append(("probe", "fairness.paired", lambda: probe(args.probe_rows, args.probe_comparative_rows)))
    count = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for scenario, module, run in scenarios:
            results, dataset = run()
            for r in results:
                record = {k: v for k, v in r.model_dump().items() if v is not None}
                record.update(scenario=scenario, module=module, dataset=dataset, **common)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            print(f"{scenario}: {len(results)} results")
    print(f"{count} results written to {args.out}")


if __name__ == "__main__":
    main()
