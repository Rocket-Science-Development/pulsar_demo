"""Analyses run on each window and written to InfluxDB for the dashboards (shared by seed_demo.py and analysis_job.py).

Each function returns a list of result dicts (the fields of pulsar_metrics' MetricResults, plus `p_adjusted` and
`confirmed` for drift); `ResultsSink.write_results` stores them in the `pulsar_results` measurement. The hourly
quantiles of the numeric features (`hourly_quantiles`, `write_hourly_quantiles`) go to their own compact measurement,
`pulsar_feature_quantiles`, read by the drift dashboard instead of the raw captures.
"""

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.analyzers.persistence import DriftPersistence
from pulsar_metrics.calibration import AACalibrator
from pulsar_metrics.explain import AttributionDrift, importance, importance_by_group
from pulsar_metrics.fairness import FairnessAnalyzer, FairnessPolicy
from pulsar_metrics.multiple_testing import fdr_correct

TABULAR_NUMERIC = ["income", "age", "household_size"]
TABULAR_CATEGORICAL = ["region", "language"]
AGENT_NUMERIC = ["steps", "tool_calls", "tool_error_rate", "input_tokens", "output_tokens", "latency_s"]
AGENT_CATEGORICAL = ["first_tool", "termination", "task_success"]


def _records(frame: pd.DataFrame) -> list:
    keep = ["metric_type", "metric_name", "feature_name", "metric_value", "drift_status", "threshold", "p_adjusted",
            "confirmed"]
    out = []
    for row in frame.to_dict("records"):
        record = {k: row.get(k) for k in keep if k in row}
        record["test_type"] = "fixed"
        out.append(record)
    return out


class DriftWatch:
    """Calibrated drift tests on successive windows: A/A thresholds, Benjamini-Hochberg, confirmation over 2 windows"""

    def __init__(self, reference: pd.DataFrame, numeric: list, categorical: list, window: int, model_id: str,
                 model_version: str, alpha: float = 0.05, persistence: int = 2, n_draws: int = 200, contiguous: bool = True):
        """`contiguous=False` draws the A/A windows as random rows: right for independent rows; keep True when
        successive rows are correlated (the calibration then needs a reference of 10 windows or more)"""
        self.reference, self.numeric, self.categorical = reference, numeric, categorical
        self.model_id, self.model_version, self.alpha = model_id, model_version, alpha
        template = self._analyzer()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            self.calibrator = AACalibrator(window_size=window, n_draws=n_draws, contiguous=contiguous).fit(
                reference, template._metrics_list)
        self.warnings = [str(w.message) for w in caught if "reference" in str(w.message)]
        for message in self.warnings:
            print(f"calibration warning ({model_id}): {message}")
        self.tracker = DriftPersistence(windows=persistence)

    def _analyzer(self) -> Analyzer:
        analyzer = Analyzer(name="window", model_id=self.model_id, model_version=self.model_version)
        if self.categorical:
            analyzer.add_drift_metrics(metrics_list=["chi2"], features_list=self.categorical)
        if self.numeric:
            analyzer.add_drift_metrics(metrics_list=["ks_2samp"], features_list=self.numeric)
        return analyzer

    def run(self, current: pd.DataFrame) -> list:
        analyzer = self._analyzer()
        analyzer.run(current=current, reference=self.reference, options=self.calibrator.options())
        drift = analyzer.results_to_pandas().query("metric_type == 'drift'").copy()
        drift["p_empirical"] = self.calibrator.empirical_pvalues(drift)
        drift = fdr_correct(drift, "p_empirical", alpha=self.alpha)
        drift["drift_status"] = drift["drift_fdr"].astype(bool)
        confirmed = {v.feature_name: v.confirmed_drift for v in self.tracker.update(drift)}
        drift["confirmed"] = drift["feature_name"].map(confirmed).fillna(False).astype(bool)
        return _records(drift)


def performance_results(current: pd.DataFrame, reference_accuracy: float, tolerance: float = 0.05) -> list:
    """Accuracy, precision, recall, F1 on the rows whose label has arrived; accuracy flagged below reference - tolerance"""
    labelled = current.dropna(subset=["y_true"]) if "y_true" in current else current.iloc[0:0]
    coverage = {"metric_type": "performance", "metric_name": "label_coverage", "metric_value": len(labelled) / len(current),
                "n": len(current)}
    if len(labelled) < 30:
        return [coverage]  # labels not arrived yet: only the coverage is known
    y, p = labelled["y_true"].astype(int).to_numpy(), labelled["y_pred"].astype(int).to_numpy()
    tp, fp, fn = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
    accuracy = float((y == p).mean())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    low = reference_accuracy - tolerance
    return [
        {"metric_type": "performance", "metric_name": "accuracy", "metric_value": accuracy, "n": len(labelled),
         "threshold": low, "drift_status": accuracy < low},
        {"metric_type": "performance", "metric_name": "precision", "metric_value": precision, "n": len(labelled)},
        {"metric_type": "performance", "metric_name": "recall", "metric_value": recall, "n": len(labelled)},
        {"metric_type": "performance", "metric_name": "f1", "metric_value": f1, "n": len(labelled)},
        coverage,
    ]


def fairness_results(current: pd.DataFrame, seed: int = 0) -> list:
    analyzer = FairnessAnalyzer(
        ["language", "region"],
        FairnessPolicy(criteria=["disparate_impact_ratio", "demographic_parity_difference"], min_group_size=30,
                       n_bootstrap=200),
        reference_groups={"language": "fr", "region": "Montréal"},
        intersections=True,
        seed=seed,
    )
    analyzer.run(current)
    return analyzer.get_result()


def explain_results(drift: AttributionDrift, attributions: pd.DataFrame, groups: pd.Series) -> list:
    results = [r.model_dump() for r in drift.run(attributions)]
    shares = importance(attributions)
    results += [
        {"metric_type": "explainability", "metric_name": "importance_share", "feature_name": f, "metric_value": float(v),
         "n": len(attributions)}
        for f, v in shares.items()
    ]
    results += [r.model_dump() for r in importance_by_group(attributions, groups, reference_group="fr", n_bootstrap=100)]
    return results


def agent_windows(runs: pd.DataFrame, window: int = 30, model_id: str = "permit_assistant", model_version: str = "1.0"):
    """(timestamp, results) per window of agent runs after the reference phase, drift calibrated on the reference"""
    reference = runs[runs["scenario"] == "reference"]
    watch = DriftWatch(reference, AGENT_NUMERIC, AGENT_CATEGORICAL, window, model_id, model_version)
    for phase in [p for p in dict.fromkeys(runs["scenario"]) if p != "reference"]:
        frame = runs[runs["scenario"] == phase]
        for start in range(0, len(frame) - window + 1, window):
            current = frame.iloc[start : start + window]
            yield current["pred_timestamp"].iloc[-1], phase, watch.run(current)


QUANTILE_MEASUREMENT = "pulsar_feature_quantiles"
QUANTILES = {"p10": 0.1, "median": 0.5, "p90": 0.9}


def hourly_quantiles(frame: pd.DataFrame, features: list, time_column: str = "pred_timestamp") -> list:
    """Exact p10, median and p90 (linear interpolation, as numpy) and row count of each numeric feature, per hour.

    Returns (hour end, feature, {"p10", "median", "p90", "n"}) sorted by time then feature. The point of an hour is
    stamped at its end, as `aggregateWindow` does. Non-numeric (categorical, boolean) and missing columns are skipped,
    so are missing values; an hour where a feature has no value gives no point. Only complete hours should be passed:
    the rows of a partial hour would overwrite the point of the full hour written earlier.
    """
    if frame.empty or time_column not in frame:
        return []
    numeric = [f for f in features if f in frame and pd.api.types.is_numeric_dtype(frame[f])
               and not pd.api.types.is_bool_dtype(frame[f])]
    if not numeric:
        return []
    times = pd.to_datetime(frame[time_column], utc=True)
    hours = times.dt.floor("h")
    values = frame[numeric].to_numpy(dtype=float)
    values[~np.isfinite(values)] = np.nan
    keys = hours.to_numpy()
    order = np.argsort(keys, kind="stable")
    keys, values = keys[order], values[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    out = []
    for i, first in enumerate(starts):  # one hour at a time, every feature at once (same result as np.quantile)
        chunk = values[first: starts[i + 1] if i + 1 < len(starts) else len(values)]
        counts = np.isfinite(chunk).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a feature without value in this hour: no point
            q = np.nanquantile(chunk, list(QUANTILES.values()), axis=0)
        end = pd.Timestamp(keys[first])
        end = (end.tz_localize("UTC") if end.tzinfo is None else end).to_pydatetime() + timedelta(hours=1)
        for j, feature in enumerate(numeric):
            if counts[j]:
                out.append((end, feature, {"p10": float(q[0, j]), "median": float(q[1, j]), "p90": float(q[2, j]),
                                           "n": float(counts[j])}))
    return out


def write_hourly_quantiles(sink, frame: pd.DataFrame, features: list, tags: dict,
                           time_column: str = "pred_timestamp") -> dict:
    """Write `hourly_quantiles` to `pulsar_feature_quantiles` through `ResultsSink.write_stats_many`, one request per
    hour, tags `model_id`, `model_version` (from `tags`) and `feature_name`. Rewriting an hour overwrites its points
    (same series, same timestamp), so the hours can be recomputed when late rows arrive.

    Returns {"points", "requests", "hours", "compute_s", "write_s"}."""
    import time

    t0 = time.perf_counter()
    rows = hourly_quantiles(frame, features, time_column)
    compute_s = time.perf_counter() - t0
    by_hour = {}
    for end, feature, fields in rows:
        by_hour.setdefault(end, []).append((fields, {**tags, "feature_name": feature}))
    t1 = time.perf_counter()
    points = sum(sink.write_stats_many(items, timestamp=end, measurement=QUANTILE_MEASUREMENT)
                 for end, items in by_hour.items())
    return {"points": points, "requests": len(by_hour), "hours": len(by_hour), "compute_s": compute_s,
            "write_s": time.perf_counter() - t1}


def quantile_gaps(raw: dict, aggregated: dict) -> list:
    """Hours to (re)aggregate: hour end -> count of raw values of the features (`raw`) against hour end -> sum of the
    `n` written for that hour (`aggregated`). An hour is a gap when it has more raw values than its aggregates counted:
    never aggregated, a feature missing, or rows that arrived after it was aggregated. Sorted, oldest first.

    `n` counts the finite values of each numeric feature, as InfluxDB counts the stored values of the field (InfluxDB
    stores no NaN), so an hour aggregated from all its rows is never a gap: a backfill run on it writes nothing."""
    return sorted(h for h, count in raw.items() if count > aggregated.get(h, 0))


def backfill_ranges(hours: list, max_hours: int = None, chunk_hours: int = 24) -> list:
    """[start, stop) ranges to aggregate for the given hour ends: the `max_hours` oldest (all if None), consecutive
    hours merged into one range, each range at most `chunk_hours` long (so one read stays bounded in memory)."""
    hours = sorted(hours)[:max_hours] if max_hours is not None else sorted(hours)
    one = timedelta(hours=1)
    ranges = []
    for end in hours:
        start = end - one
        if ranges and ranges[-1][1] == start and ranges[-1][1] - ranges[-1][0] < timedelta(hours=chunk_hours):
            ranges[-1][1] = end
        else:
            ranges.append([start, end])
    return [tuple(r) for r in ranges]


def to_numeric(frame: pd.DataFrame, columns: list) -> pd.DataFrame:
    for column in columns:
        frame[column] = pd.to_numeric(frame.get(column, 0), errors="coerce").fillna(0).astype(float)
    return frame


def windows(frame: pd.DataFrame, freq: str, origin=None):
    """(end time, rows) per window of `freq`, aligned on `origin` (default: the first row)"""
    origin = frame["pred_timestamp"].min() if origin is None else origin
    for _, chunk in frame.groupby(pd.Grouper(key="pred_timestamp", freq=freq, origin=origin)):
        if len(chunk):
            yield chunk["pred_timestamp"].max(), chunk

