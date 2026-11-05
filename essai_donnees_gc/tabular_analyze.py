"""Part 1 analysis: does Pulsar see the COVID-19 lockdown in the NO2 of Montréal, and how often does it alarm before?

The replayed weeks are read back from InfluxDB and compared one by one with the out-of-fold reference of 2016-2018.
Three groups of signals, each with its own Benjamini-Hochberg correction and a confirmation over 2 consecutive weeks
(DriftPersistence):

| Group | Features and metrics | Expected |
|---|---|---|
| error | residual (observed - predicted): ks_2samp, wasserstein | change from March 2020 |
| performance | weekly MAE above the 95th percentile of the reference weeks of the same period | change from March 2020 |
| inputs | weather (temp, dew, rh, wspd, wd_s, wd_c, press) and the prediction: ks_2samp | none (negative control: a lockdown does not change the weather) |

Thresholds are calibrated by A/A draws (AACalibrator, compare_to="reference", contiguous windows of one week). Two
references are compared: `period`, the same weeks of 2016-2018 +- 4 weeks (period_reference, the setting recommended
for seasonal data), and `annual`, the whole of 2016-2018.

Known change: Québec declared a public health emergency on 13 March 2020 and closed schools from 16 March, then
non-essential businesses from 25 March. Weeks starting on or before 2 March 2020 are the stable period (false alarms);
the week of 9 March is a transition, left out; weeks from 16 March 2020 are the changed period.

    python tabular_analyze.py --data data --env-file ../../validation_e2e/.env --campaign no2-20261104
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.analyzers.persistence import DriftPersistence
from pulsar_metrics.calibration import AACalibrator, CalibrationCache, period_reference
from pulsar_metrics.multiple_testing import fdr_correct

from influx_env import load_env, query
from no2_data import MONITORED
from tabular_replay import MODEL_ID, MODEL_VERSION

HERE = Path(__file__).parent
CHANGE = pd.Timestamp("2020-03-16")
TRANSITION = pd.Timestamp("2020-03-09")
GROUPS = {
    "error": (["ks_2samp", "wasserstein"], ["residual"]),
    "inputs": (["ks_2samp"], MONITORED + ["prediction"]),
}


def read_replay(login: dict, campaign: str) -> pd.DataFrame:
    flux = f"""
    from(bucket: "{login['bucket_name']}")
      |> range(start: -90d, stop: 1m)
      |> filter(fn: (r) => r["_measurement"] == "{MODEL_ID}_{MODEL_VERSION}_input_data" and r["campaign"] == "{campaign}")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    frame = query(login, flux)
    frame["pred_timestamp"] = pd.to_datetime(frame["obs_ts"], unit="s")
    return prepare(frame.drop(columns=["result", "table", "_start", "_stop", "_measurement", "_time"]))


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("pred_timestamp").reset_index(drop=True)
    frame["model_id"], frame["model_version"] = MODEL_ID, MODEL_VERSION
    frame["residual"] = frame["no2"] - frame["prediction"]
    return frame


def read_reference(data: Path) -> pd.DataFrame:
    frame = pd.read_csv(data / "no2_reference.csv.gz", parse_dates=["obs_time"])
    return prepare(frame.rename(columns={"obs_time": "pred_timestamp"}))


def analyzer_for(group: str) -> Analyzer:
    metrics, features = GROUPS[group]
    analyzer = Analyzer(name=group, model_id=MODEL_ID, model_version=MODEL_VERSION)
    analyzer.add_drift_metrics(metrics_list=metrics, features_list=features)
    return analyzer


def weekly_mae(frame: pd.DataFrame) -> pd.Series:
    weeks = frame["pred_timestamp"].dt.to_period("W-SUN").dt.start_time
    return frame["residual"].abs().groupby(weeks).mean()


def run(replay: pd.DataFrame, reference: pd.DataFrame, mode: str, n_draws: int) -> list:
    cache = CalibrationCache()
    trackers = {g: DriftPersistence(windows=2) for g in [*GROUPS, "performance"]}
    annual = {}
    if mode == "annual":
        for group in GROUPS:
            annual[group] = AACalibrator(window_size=168, n_draws=n_draws).fit(reference, analyzer_for(group)._metrics_list)
    rows = []
    weeks = replay["pred_timestamp"].dt.to_period("W-SUN").dt.start_time
    for week, current in replay.groupby(weeks):
        ref = period_reference(reference, current, weeks=4) if mode == "period" else reference
        row = {"week": str(week.date()), "rows": len(current), "reference_rows": len(ref),
               "mean_residual": float(current["residual"].mean()), "mae": float(current["residual"].abs().mean())}
        for group in GROUPS:
            analyzer = analyzer_for(group)
            if mode == "period":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)  # short period references are flagged reliable=False
                    calibrator = AACalibrator(window_size=168, n_draws=n_draws, allow_short=True).fit(
                        ref, analyzer._metrics_list, cache=cache)
            else:
                calibrator = annual[group]
            analyzer.run(current=current, reference=ref, options=calibrator.options())
            drift = analyzer.results_to_pandas().query("metric_type == 'drift'").copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                drift["p_empirical"] = calibrator.empirical_pvalues(drift)
            drift = fdr_correct(drift, "p_empirical", alpha=0.05)
            drift["drift_status"] = drift["drift_fdr"].astype(bool)
            confirmed = [v for v in trackers[group].update(drift) if v.confirmed_drift]
            row[group] = {
                "flagged": sorted(f"{m}|{f}" for m, f in drift.loc[drift["drift_status"], ["metric_name", "feature_name"]].values),
                "confirmed": sorted({f"{v.metric_name}|{v.feature_name}" for v in confirmed}),
                "min_p_adjusted": float(np.nanmin(drift["p_adjusted"])),
                "reliable": bool(drift["reliable"].fillna(True).all()) if "reliable" in drift else None,
            }
        # performance: weekly MAE against the reference weeks of the same period (or of the whole reference)
        threshold = float(np.quantile(weekly_mae(ref), 0.95))
        flagged = row["mae"] > threshold
        verdict = pd.DataFrame([{"metric_type": "performance", "metric_name": "mae", "feature_name": "prediction",
                                 "drift_status": flagged}])
        confirmed = trackers["performance"].update(verdict)[0].confirmed_drift
        row["performance"] = {"flagged": ["mae"] if flagged else [], "confirmed": ["mae"] if confirmed else [],
                              "threshold": threshold}
        rows.append(row)
    return rows


def summarize(rows: list) -> dict:
    out = {}
    stable = [r for r in rows if pd.Timestamp(r["week"]) < TRANSITION]
    changed = [r for r in rows if pd.Timestamp(r["week"]) >= CHANGE]
    for group in [*GROUPS, "performance"]:
        def first(key):
            week = next((r["week"] for r in changed if r[group][key]), None)
            return None if week is None else {"week": week, "weeks_after_change": (pd.Timestamp(week) - CHANGE).days // 7}

        def exact_ci(k, n):  # Clopper-Pearson 95 % interval
            from scipy.stats import beta
            lo = beta.ppf(0.025, k, n - k + 1) if k else 0.0
            hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
            return [round(float(lo), 3), round(float(hi), 3)]

        fa_raw = sum(bool(r[group]["flagged"]) for r in stable)
        fa_conf = sum(bool(r[group]["confirmed"]) for r in stable)
        episodes = sum(1 for a, b in zip([None] + stable, stable) if b[group]["confirmed"] and not (a and a[group]["confirmed"]))
        det_raw = sum(bool(r[group]["flagged"]) for r in changed)
        det_conf = sum(bool(r[group]["confirmed"]) for r in changed)
        out[group] = {
            "stable_weeks": len(stable), "changed_weeks": len(changed),
            "stable_weeks_flagged": fa_raw, "stable_weeks_flagged_ci95": exact_ci(fa_raw, len(stable)),
            "stable_weeks_confirmed": fa_conf, "stable_confirmed_episodes": episodes,
            "stable_weeks_confirmed_ci95": exact_ci(fa_conf, len(stable)),
            "changed_weeks_flagged": det_raw, "changed_weeks_flagged_ci95": exact_ci(det_raw, len(changed)),
            "changed_weeks_confirmed": det_conf, "changed_weeks_confirmed_ci95": exact_ci(det_conf, len(changed)),
            "first_flag_after_change": first("flagged"), "first_confirmed_after_change": first("confirmed"),
            "features_confirmed_after_change": sorted({f for r in changed for f in r[group]["confirmed"]}),
            "features_confirmed_stable": sorted({f for r in stable for f in r[group]["confirmed"]}),
        }
    return out


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(HERE / "data"))
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--modes", nargs="*", default=["period", "annual"])
    parser.add_argument("--n-draws", type=int, default=200)
    args = parser.parse_args(argv)
    login = load_env(args.env_file)
    replay, reference = read_replay(login, args.campaign), read_reference(Path(args.data))
    print(f"{len(replay)} replayed rows, {len(reference)} reference rows")
    report = {"campaign": args.campaign, "change_start": str(CHANGE.date()), "transition_week": str(TRANSITION.date()),
              "replay_rows": len(replay), "reference_rows": len(reference), "modes": {}}
    for mode in args.modes:
        started = time.perf_counter()
        rows = run(replay, reference, mode, args.n_draws)
        report["modes"][mode] = {"summary": summarize(rows), "seconds": round(time.perf_counter() - started, 1),
                                 "weeks": rows}
        print(mode, json.dumps(report["modes"][mode]["summary"], indent=1), flush=True)
    out = HERE / "results" / f"tabular_analysis_{args.campaign}.json"
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
