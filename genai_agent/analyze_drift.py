"""Analyze a drift campaign: calibrated drift detection on agent trajectories, window by window.

For the campaign written by campaign.py:
1. reference windows (scenario "reference") calibrate the thresholds of every (metric, feature) on A/A draws;
2. every later phase is cut into consecutive windows of `--window` agent runs, each compared with the reference;
3. per window, empirical p-values are corrected with Benjamini-Hochberg across all features;
4. a drift is confirmed when it persists over `--persistence` consecutive windows (DriftPersistence).

Reported: false alarms on the stable hold-out phase, and for each injected drift the first window with an alert, the
first confirmed window and the features involved.

Usage:
    python analyze_drift.py --campaign s2-cpu-1 [--window 30] [--out results]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.analyzers.persistence import DriftPersistence
from pulsar_metrics.calibration import AACalibrator
from pulsar_metrics.multiple_testing import fdr_correct

MODEL_ID, MODEL_VERSION = "permit_assistant", "1.0"
NUMERIC = ["steps", "tool_calls", "tool_error_rate", "input_tokens", "output_tokens", "latency_s"]
CATEGORICAL = ["first_tool", "termination", "task_success"]
METADATA = ["request_model", "prompt_version"]  # reported as configuration changes, not tested


def load_runs(campaign: str) -> pd.DataFrame:
    """agent_run records of a campaign, in time order"""
    query = f"""
    from(bucket: "{os.environ.get('INFLUX_BUCKET', 'monitoring')}")
      |> range(start: -30d, stop: 1m)
      |> filter(fn: (r) => r["_measurement"] == "{MODEL_ID}_{MODEL_VERSION}_input_data")
      |> filter(fn: (r) => r["record_type"] == "agent_run" and r["campaign"] == "{campaign}")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    with InfluxDBClient(
        url=os.environ.get("INFLUX_URL", "http://localhost:8086"),
        token=os.environ["INFLUX_TOKEN"],
        org=os.environ.get("INFLUX_ORG", "pulsarml"),
        timeout=120_000,
    ) as client:
        frame = client.query_api().query_data_frame(query=query)
    if isinstance(frame, list):
        frame = pd.concat(frame, ignore_index=True)
    return prepare(frame.rename(columns={"_time": "pred_timestamp"}))


def prepare(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.sort_values("pred_timestamp").reset_index(drop=True)
    for column in NUMERIC:
        runs[column] = pd.to_numeric(runs.get(column, 0), errors="coerce").fillna(0).astype(float)
    for column in CATEGORICAL:
        runs[column] = runs.get(column, "none").astype(str)
    return runs


def build_metrics(analyzer: Analyzer):
    analyzer.add_drift_metrics(metrics_list=["chi2"], features_list=CATEGORICAL)
    analyzer.add_drift_metrics(metrics_list=["ks_2samp"], features_list=NUMERIC)


def windows_of(frame: pd.DataFrame, size: int):
    for start in range(0, len(frame) - size + 1, size):
        yield frame.iloc[start : start + size]


def analyze(runs: pd.DataFrame, window: int = 30, alpha: float = 0.05, persistence: int = 2, n_draws: int = 200) -> dict:
    reference = runs[runs["scenario"] == "reference"]
    if len(reference) < 2 * window:
        raise ValueError(f"the reference phase holds {len(reference)} runs; at least {2 * window} are needed")
    template = Analyzer(name="calibration", model_id=MODEL_ID, model_version=MODEL_VERSION)
    build_metrics(template)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        calibrator = AACalibrator(window_size=window, n_draws=n_draws).fit(reference, template._metrics_list)
    caught = [w for w in caught if "reference" in str(w.message)]  # keep the calibration warnings only
    options = calibrator.options()

    phases = [p for p in dict.fromkeys(runs["scenario"]) if p != "reference"]
    report = {
        "window": window,
        "alpha": alpha,
        "persistence": persistence,
        "reference_runs": len(reference),
        "calibration_warnings": [str(w.message) for w in caught],
        "nominal_false_alarm_rate": {
            f"{m}|{f}": calibrator.false_alarm_rate(m, f) for (m, f), null in calibrator.nulls.items() if null.kind == "pvalue"
        },
        "reference_models": (
            sorted(reference["request_model"].dropna().astype(str).unique()) if "request_model" in reference else []
        ),
        "phases": {},
    }
    tracker = DriftPersistence(windows=persistence)
    for phase in phases:
        tracker.reset()
        frame = runs[runs["scenario"] == phase]
        rows = []
        for k, current in enumerate(windows_of(frame, window), start=1):
            analyzer = Analyzer(name=phase, model_id=MODEL_ID, model_version=MODEL_VERSION)
            build_metrics(analyzer)
            analyzer.run(current=current, reference=reference, options=options)
            drift = analyzer.results_to_pandas().query("metric_type == 'drift'").copy()
            drift["p_empirical"] = calibrator.empirical_pvalues(drift)
            drift = fdr_correct(drift, "p_empirical", alpha=alpha)
            raw = sorted(drift.loc[drift["drift_fdr"] == True, "feature_name"])  # noqa: E712
            drift["drift_status"] = drift["drift_fdr"].astype(bool)
            confirmed = sorted(v.feature_name for v in tracker.update(drift) if v.confirmed_drift)
            config = {
                c: sorted(set(current[c].dropna().astype(str)) - set(reference[c].dropna().astype(str)))
                for c in METADATA
                if c in current
            }
            rows.append(
                {
                    "window": k,
                    "runs": len(current),
                    "features_flagged": raw,
                    "features_confirmed": confirmed,
                    "min_p_adjusted": float(np.nanmin(drift["p_adjusted"])),
                    "new_configuration": {c: v for c, v in config.items() if v},
                    "task_success_rate": float((current["task_success"] == "True").mean()),
                    "mean_steps": float(current["steps"].mean()),
                    "tool_error_rate": float(current["tool_error_rate"].mean()),
                }
            )
        first_raw = next((r["window"] for r in rows if r["features_flagged"]), None)
        first_confirmed = next((r["window"] for r in rows if r["features_confirmed"]), None)
        report["phases"][phase] = {
            "runs": len(frame),
            "windows": len(rows),
            "windows_with_alert": sum(bool(r["features_flagged"]) for r in rows),
            "windows_confirmed": sum(bool(r["features_confirmed"]) for r in rows),
            "first_alert_window": first_raw,
            "first_confirmed_window": first_confirmed,
            "features": sorted({f for r in rows for f in r["features_flagged"]}),
            "features_confirmed": sorted({f for r in rows for f in r["features_confirmed"]}),
            "detail": rows,
        }
    return report


def markdown(report: dict) -> str:
    lines = [
        f"Windows of {report['window']} agent runs; reference: {report['reference_runs']} runs; "
        f"FDR {report['alpha']}; confirmation over {report['persistence']} windows.",
        "",
        "| Phase | Runs | Windows | With alert | Confirmed | First alert | First confirmed | Confirmed features | Any alert |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for phase, p in report["phases"].items():
        lines.append(
            f"| {phase} | {p['runs']} | {p['windows']} | {p['windows_with_alert']} | {p['windows_confirmed']} | "
            f"{p['first_alert_window'] or '-'} | {p['first_confirmed_window'] or '-'} | "
            f"{', '.join(p['features_confirmed']) or '-'} | {', '.join(p['features']) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    # summary statistics on constant columns (e.g. no tool error at all) emit numerical warnings without consequence
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Precision loss occurred in moment calculation")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--persistence", type=int, default=2)
    parser.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = parser.parse_args(argv)

    runs = load_runs(args.campaign)
    print(
        f"{len(runs)} agent runs in campaign {args.campaign}: "
        + ", ".join(f"{k}={v}" for k, v in runs["scenario"].value_counts(sort=False).items())
    )
    report = analyze(runs, args.window, args.alpha, args.persistence)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"drift_{args.campaign}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (out / f"drift_{args.campaign}.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
