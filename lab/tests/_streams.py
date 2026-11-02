"""Shared helpers of the "ml-streams" family (TS-*, TC-*): data location, windowed analysis through the public
Pulsar.ml API (Analyzer + AACalibrator + Benjamini-Hochberg + DriftPersistence), and small statistics.

Nothing here re-implements a Pulsar.ml statistic: every verdict comes from pulsar_metrics; this module only feeds
windows and counts outcomes.
"""

import contextlib
import io
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.analyzers.persistence import DriftPersistence
from pulsar_metrics.calibration import AACalibrator
from pulsar_metrics.multiple_testing import fdr_correct

logging.getLogger("pulsar_metrics").setLevel(logging.ERROR)

DATA = Path(os.environ.get("LAB_DATA", "/data/lab" if Path("/data/lab").exists() else "C:/dev/pulsar-ml/data/lab"))
DATASETS = DATA / "datasets"
CACHE = DATA / "cache" / "ml-streams"
CACHE.mkdir(parents=True, exist_ok=True)

DRY = os.environ.get("LAB_DRY", "0") == "1"
MODEL_ID, MODEL_VERSION = "lab", "1"


def tag(df: pd.DataFrame, start=None) -> pd.DataFrame:
    """Add the metadata columns the Analyzer filters on (model_id, model_version, pred_timestamp)"""
    out = df.copy()
    out["model_id"], out["model_version"] = MODEL_ID, MODEL_VERSION
    if "pred_timestamp" not in out:
        out["pred_timestamp"] = pd.Timestamp(start or "2020-01-01")
    return out


def make_analyzer(features, drift_metrics=("ks_2samp", "wasserstein", "psi"), perf_metrics=(), nan_policy=None,
                  **perf_kwargs):
    """nan_policy: None keeps the product default ("omit" since the calibration fix); "propagate" is the old behaviour"""
    a = Analyzer(name="lab", model_id=MODEL_ID, model_version=MODEL_VERSION)
    extra = {"nan_policy": nan_policy} if nan_policy is not None else {}
    a.add_drift_metrics(metrics_list=list(drift_metrics), features_list=list(features), **extra)
    if perf_metrics:
        a.add_performance_metrics(metrics_list=list(perf_metrics), **perf_kwargs)
    return a


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def calibrate(analyzer, reference: pd.DataFrame, window: int, n_draws: int = 200, seed: int = 0, **options):
    """AACalibrator on the reference; returns (calibrator, warnings raised)

    `options` go to AACalibrator (post-fix API): allow_short=True for references shorter than 10 windows (refused
    otherwise), auto_draws=False to keep n_draws fixed and reproduce the old p-value floor 1/(n_draws+1).
    """
    cal = AACalibrator(window_size=window, n_draws=n_draws, contiguous=True, seed=seed, **options)
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        cal.fit(reference, analyzer._metrics_list)
    return cal, [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def analyze(analyzer, current, reference, cal=None, options=None, alpha=0.05) -> pd.DataFrame:
    """One window through Analyzer.run; adds p_empirical / drift_fdr (BH over all drift rows) when calibrated"""
    opts = options if options is not None else (cal.options() if cal is not None else {})
    quiet(analyzer.run, current=tag(current), reference=tag(reference), options=opts)
    res = analyzer.results_to_pandas()
    res = res[res.metric_type.isin(["drift", "performance"])].reset_index(drop=True)
    if cal is not None:
        drift = res.metric_type == "drift"
        res.loc[drift, "p_empirical"] = cal.empirical_pvalues(res[drift]).to_numpy()
        corrected = fdr_correct(res[drift], "p_empirical", alpha=alpha)
        res.loc[drift, "drift_fdr"] = corrected["drift_fdr"].to_numpy()
    return res


def persistence_frame(res: pd.DataFrame, column: str) -> pd.DataFrame:
    """Result rows with drift_status replaced by `column` (e.g. drift_fdr), as fed to DriftPersistence"""
    out = res.copy()
    if column != "drift_status":
        out["drift_status"] = out[column].where(out[column].notna(), None)
    return out


def window_flag(res: pd.DataFrame, column="drift_status", metric_type="drift") -> bool:
    sub = res[res.metric_type == metric_type][column]
    return bool(sub.fillna(False).astype(bool).any())


def any_confirmed(verdicts: pd.DataFrame, metric_type="drift") -> bool:
    if verdicts.empty:
        return False
    return bool(verdicts[verdicts.metric_type == metric_type].confirmed_drift.any())


def first_index(flags, start=0):
    for i, f in enumerate(flags):
        if i >= start and f:
            return i - start
    return None


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 4), round((c + h) / d, 4))


def fmt_rate(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / max(n, 1):.1%} (95 % CI {lo:.1%}-{hi:.1%})"
