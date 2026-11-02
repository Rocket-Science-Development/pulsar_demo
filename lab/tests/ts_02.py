"""TS-02 - Documented sensor drift (UCI Air Quality, CC BY 4.0)

Hourly multi-sensor device, Mar 2004 - Apr 2005, sensors known to drift; missing values coded -200.
Features: 5 metal-oxide sensors + T, RH, AH. Target: CO(GT) from the reference analyser (regression).
Reference = first 10 weeks; windows = one week (168 rows); ks_2samp, wasserstein, psi calibrated by AACalibrator,
BH over the window, DriftPersistence(2). Labels arrive with a delay of LABEL_LAG weeks.

Three versions of the same stream are sent to Pulsar.ml:
  raw     - -200 left as a value (what an unaware integration sends)
  nan     - -200 replaced by NaN (what a careful integration sends)
  dropped - rows with any missing feature removed before the analysis (user-side workaround)
"""

import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from lab.tests import _streams as S

SENSORS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
WEATHER = ["T", "RH", "AH"]
FEATURES = SENSORS + WEATHER
TARGET = "CO(GT)"
W = 168
REF_WEEKS = 10
LABEL_LAG = 2
CRITERIA = {"sensors_confirmed_dropped": (">=", 1), "rmse_degradation_flagged": ("==", True),
            "sentinel_induced_alert_windows_raw": ("==", 0)}
PARAMS = {"window_rows": W, "reference_weeks": REF_WEEKS, "label_lag_weeks": LABEL_LAG, "persistence": 2,
          "metrics": ["ks_2samp", "wasserstein", "psi"], "rmse_threshold_factor": 1.3}


def load() -> pd.DataFrame:
    with zipfile.ZipFile(S.DATASETS / "uci_air_quality_360" / "air+quality.zip") as z, z.open("AirQualityUCI.csv") as f:
        df = pd.read_csv(f, sep=";", decimal=",")
    df = df.dropna(how="all", axis=1).dropna(subset=["Date"])
    df["pred_timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S")
    return df.sort_values("pred_timestamp").reset_index(drop=True)[["pred_timestamp"] + FEATURES + [TARGET]]


def run(ctx):
    raw = load()
    sentinel_share = {c: float((raw[c] == -200).mean()) for c in FEATURES + [TARGET]}
    nan = raw.copy()
    nan[FEATURES + [TARGET]] = nan[FEATURES + [TARGET]].where(nan[FEATURES + [TARGET]] != -200)

    ref_n = REF_WEEKS * W
    # model: trained on the reference period, valid labels only; handles NaN features natively
    tr = nan.iloc[:ref_n].dropna(subset=[TARGET])
    model = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(tr[FEATURES], tr[TARGET])
    pred = model.predict(nan[FEATURES])
    # reference RMSE: out-of-sample on the last 2 reference weeks with a model fitted on the first 8
    cut = (REF_WEEKS - 2) * W
    m8 = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(
        nan.iloc[:cut].dropna(subset=[TARGET])[FEATURES], nan.iloc[:cut].dropna(subset=[TARGET])[TARGET])
    hold = nan.iloc[cut:ref_n].dropna(subset=[TARGET])
    rmse_ref = float(np.sqrt(np.mean((m8.predict(hold[FEATURES]) - hold[TARGET]) ** 2)))
    mse_threshold = float((PARAMS["rmse_threshold_factor"] * rmse_ref) ** 2)

    versions = {}
    for name, frame in (("raw", raw), ("nan", nan)):
        f = frame.copy()
        f["y_pred"] = pred
        f["y_true"] = frame[TARGET]
        versions[name] = f
    dropped = versions["nan"].dropna(subset=FEATURES).copy()
    versions["dropped"] = dropped

    n_weeks = (len(raw) - ref_n) // W
    if S.DRY:
        n_weeks = 6
    metrics = {"rows": len(raw), "sentinel_share": sentinel_share, "rmse_ref": rmse_ref, "mse_threshold": mse_threshold,
               "weeks_streamed": n_weeks}

    timelines = {}
    for name, f in versions.items():
        # windows by time so the three versions compare the same calendar weeks
        t_ref_end = raw.pred_timestamp.iloc[ref_n]
        ref = f[f.pred_timestamp < t_ref_end]
        analyzer = S.make_analyzer(FEATURES, PARAMS["metrics"])
        cw = W if name != "dropped" else int(W * len(f) / len(raw))
        nd = 40 if S.DRY else 200
        try:
            cal, warns = S.calibrate(analyzer, ref.reset_index(drop=True), cw, n_draws=nd)
            metrics[f"{name}_calibration_error"] = None
        except Exception as exc:  # noqa: BLE001 - AACalibrator behaviour on this input, recorded
            metrics[f"{name}_calibration_error"] = f"{type(exc).__name__}: {exc}"[:300]
            usable = []
            for m in PARAMS["metrics"]:
                try:
                    S.calibrate(S.make_analyzer(FEATURES, [m]), ref.reset_index(drop=True), cw, n_draws=20)
                    usable.append(m)
                except Exception:  # noqa: BLE001
                    pass
            metrics[f"{name}_calibration_usable_metrics"] = usable
            cal, warns = S.calibrate(S.make_analyzer(FEATURES, usable), ref.reset_index(drop=True), cw, n_draws=nd)
        nan_nulls = {f"{k[0]}|{k[1]}": float(np.isnan(v.values).mean()) for k, v in cal.nulls.items() if np.isnan(v.values).any()}
        pers = S.DriftPersistence(windows=2)
        rows = []
        for i in range(n_weeks):
            t0 = t_ref_end + pd.Timedelta(hours=W * i)
            cur = f[(f.pred_timestamp >= t0) & (f.pred_timestamp < t0 + pd.Timedelta(hours=W))]
            if len(cur) < 10:
                continue
            res = S.analyze(analyzer, cur.reset_index(drop=True), ref.reset_index(drop=True), cal=cal)
            drift = res[res.metric_type == "drift"]
            ver = pers.update_to_pandas(S.persistence_frame(drift, "drift_fdr"))
            confirmed = set(ver[ver.confirmed_drift].feature_name) if not ver.empty else set()
            rows.append({
                "week": i, "start": str(t0), "n": len(cur),
                "bh_features": sorted(set(drift[drift.drift_fdr.fillna(False).astype(bool)].feature_name)),
                "confirmed": sorted(confirmed),
                "nan_values": {m: int(drift[(drift.metric_name == m)].metric_value.isna().sum()) for m in PARAMS["metrics"]},
                "no_verdict": int(drift.drift_fdr.isna().sum()),
                "nan_value_status": {str(k): int(v) for k, v in drift[drift.metric_value.isna()].drift_status.astype(object)
                                     .where(drift[drift.metric_value.isna()].drift_status.notna(), "None").value_counts().items()},
                "errors": int(drift.error.notna().sum()) if "error" in drift else 0,
                "sentinel_share": {c: float((cur[c] == -200).mean()) for c in SENSORS} if name == "raw" else None,
            })
        timelines[name] = rows
        metrics[f"{name}_calibration_warning"] = bool(warns)
        metrics[f"{name}_null_with_nan"] = nan_nulls
        metrics[f"{name}_no_verdict_tests"] = sum(r["no_verdict"] for r in rows)
        metrics[f"{name}_tests"] = len(rows) * len(FEATURES) * 3
        first = {}
        share = {}
        for c in FEATURES:
            flags = [c in r["confirmed"] for r in rows]
            first[c] = S.first_index(flags)
            share[c] = float(np.mean(flags)) if flags else None
        metrics[f"{name}_first_confirmed_week"] = first
        metrics[f"{name}_confirmed_share"] = share

    metrics["sensors_confirmed_dropped"] = sum(1 for c in SENSORS if metrics["dropped_first_confirmed_week"][c] is not None)
    metrics["weather_confirmed_dropped"] = sum(1 for c in WEATHER if metrics["dropped_first_confirmed_week"][c] is not None)

    # sentinel-induced alerts: (week, feature) bh alert in raw but not in dropped, when the week holds -200 on it
    raw_rows = {r["week"]: r for r in timelines["raw"]}
    drop_rows = {r["week"]: r for r in timelines["dropped"]}
    induced, induced_windows = [], set()
    for wk, r in raw_rows.items():
        d = drop_rows.get(wk)
        if not d:
            continue
        for c in SENSORS:
            if c in r["bh_features"] and c not in d["bh_features"] and r["sentinel_share"][c] > 0:
                induced.append((wk, c, round(r["sentinel_share"][c], 3)))
                induced_windows.add(wk)
    metrics["sentinel_induced_alerts_raw"] = len(induced)
    metrics["sentinel_induced_alert_windows_raw"] = len(induced_windows)
    metrics["sentinel_induced_examples"] = induced[:15]
    metrics["weeks_with_sentinel_in_some_sensor"] = sum(1 for r in timelines["raw"] if any(v > 0 for v in r["sentinel_share"].values()))

    # performance with delayed labels: verdict of week t is available at t + LABEL_LAG
    perf = {}
    versions["labelled"] = versions["nan"].dropna(subset=["y_true"]).copy()
    for name in ("raw", "nan", "labelled"):
        f = versions[name]
        t_ref_end = raw.pred_timestamp.iloc[ref_n]
        ref = f[f.pred_timestamp < t_ref_end]
        a = S.make_analyzer(["PT08.S1(CO)"], ["wasserstein"], perf_metrics=["mse"])
        pers = S.DriftPersistence(windows=2)
        out = []
        for i in range(n_weeks):
            t0 = t_ref_end + pd.Timedelta(hours=W * i)
            cur = f[(f.pred_timestamp >= t0) & (f.pred_timestamp < t0 + pd.Timedelta(hours=W))]
            res = S.analyze(a, cur.reset_index(drop=True), ref.reset_index(drop=True), options={"mse": {"threshold": mse_threshold}})
            p = res[res.metric_type == "performance"].iloc[0]
            ver = pers.update_to_pandas(res[res.metric_type == "performance"])
            valid = cur.dropna(subset=["y_true"]) if name != "raw" else cur
            true_rmse = float(np.sqrt(np.mean((valid.y_pred - valid.y_true) ** 2))) if len(valid) else None
            out.append({"week": i, "mse": None if pd.isna(p.metric_value) else float(p.metric_value),
                        "status": None if pd.isna(p.drift_status) else bool(p.drift_status),
                        "error": None if pd.isna(p.get("error")) else str(p.get("error"))[:120],
                        "confirmed": bool(ver.confirmed_drift.any()) if not ver.empty else False,
                        "rmse_valid_labels": true_rmse,
                        "missing_label_share": float(cur.y_true.isna().mean() if name != "raw" else (cur.y_true == -200).mean())})
        perf[name] = out
    nan_perf = perf["nan"]
    metrics["perf_nan_windows_failed"] = sum(1 for r in nan_perf if r["error"])
    metrics["perf_nan_windows"] = len(nan_perf)
    metrics["perf_nan_weeks_with_partial_labels"] = sum(1 for r in nan_perf if 0 < r["missing_label_share"] < 1)
    metrics["perf_raw_weeks_flagged"] = sum(1 for r in perf["raw"] if r["status"])
    # what a correct handling (labels missing ignored) would give
    degraded = [r for r in nan_perf if r["rmse_valid_labels"] and r["rmse_valid_labels"] ** 2 > mse_threshold]
    metrics["weeks_truly_degraded"] = len(degraded)
    metrics["rmse_by_week_valid_labels"] = [round(r["rmse_valid_labels"], 3) if r["rmse_valid_labels"] else None for r in nan_perf]
    metrics["rmse_degradation_flagged"] = any(r["confirmed"] and not r["error"] for r in nan_perf)
    first_true = S.first_index([r["rmse_valid_labels"] is not None and r["rmse_valid_labels"] ** 2 > mse_threshold for r in nan_perf])
    first_conf = S.first_index([r["confirmed"] for r in nan_perf])
    metrics["first_degraded_week"] = first_true
    metrics["first_confirmed_perf_week_nan"] = first_conf
    metrics["perf_alert_available_week_nan"] = None if first_conf is None else first_conf + LABEL_LAG
    lab = perf["labelled"]
    metrics["perf_labelled_only_weeks_confirmed"] = sum(r["confirmed"] for r in lab)
    metrics["perf_labelled_only_first_confirmed_week"] = S.first_index([r["confirmed"] for r in lab])
    metrics["perf_labelled_only_alert_available_week"] = (None if metrics["perf_labelled_only_first_confirmed_week"] is None
                                                          else metrics["perf_labelled_only_first_confirmed_week"] + LABEL_LAG)
    metrics["rmse_degradation_flagged_after_user_filtering"] = any(r["confirmed"] for r in lab)
    nan_status = {}
    errs = 0
    for r in timelines["nan"]:
        errs += r["errors"]
        for k, v in r["nan_value_status"].items():
            nan_status[k] = nan_status.get(k, 0) + v
    metrics["nan_version_status_of_nan_valued_tests"] = nan_status
    metrics["nan_version_metric_errors"] = errs
    ctx.params["timelines"] = {k: v[:60] for k, v in timelines.items()}
    ctx.params["perf"] = perf

    # ------------------------------------------------------------------ limits
    if metrics["sentinel_induced_alerts_raw"]:
        ctx.limit(
            f"The -200 missing-value sentinel is analysed as a measurement: {metrics['sentinel_induced_alerts_raw']} "
            f"(week, sensor) drift alerts over {metrics['sentinel_induced_alert_windows_raw']} weeks exist only because of "
            f"-200 values (absent when the same rows are cleaned)",
            f"sentinel share per column {{{', '.join(f'{k}: {v:.1%}' for k, v in sentinel_share.items())}}}; examples "
            f"(week, sensor, share of -200 in the week) {metrics['sentinel_induced_examples'][:8]}",
            "Add a missing_values / sentinel option to Analyzer.add_drift_metrics (per feature, e.g. {-200, -999, ''}) "
            "that maps sentinels to NaN, and report the missing share as its own drift metric (missing_rate drift) "
            "rather than mixing it into the distribution tests.",
            "major")
    if metrics.get("nan_calibration_error"):
        ctx.limit(
            "AACalibrator.fit aborts the whole calibration when the reference contains NaN: one metric that cannot "
            "handle NaN (psi) raises and no threshold is produced for any (metric, feature)",
            f"error: {metrics['nan_calibration_error']}; metrics that calibrate alone on the same reference: "
            f"{metrics.get('nan_calibration_usable_metrics')}",
            "In AACalibrator.fit, catch MetricEvaluationError per (metric, feature), keep the others, and list the "
            "failures in calibrator.errors; make population_stability_index / kl_divergence drop NaN before binning "
            "(np.histogram_bin_edges fails on NaN).",
            "major")
    nn = metrics["nan_no_verdict_tests"]
    if nn:
        nanv = {m: sum(r["nan_values"][m] for r in timelines["nan"]) for m in PARAMS["metrics"]}
        ctx.limit(
            f"With NaN instead of -200 (a single missing row in the reference suffices), {nn} of {metrics['nan_tests']} "
            f"drift tests give no usable verdict: metric_value NaN, yet drift_status is set to True or False instead "
            f"of None",
            f"NaN metric values by metric over {len(timelines['nan'])} weeks: {nanv}; drift_status of the NaN-valued "
            f"tests: {metrics['nan_version_status_of_nan_valued_tests']} (errors recorded: "
            f"{metrics['nan_version_metric_errors']}); null distributions containing NaN "
            f"in calibration: {metrics['nan_null_with_nan']}",
            "In DriftMetric/DriftTestMetric.evaluate, drop NaN from both samples before the test (nan_policy='omit' "
            "for scipy tests), record the dropped share in the result, and set drift_status=None with an error when a "
            "statistic is NaN instead of False.",
            "major")
    if metrics["perf_nan_windows_failed"]:
        ctx.limit(
            f"Performance monitoring is impossible on real delayed/partial labels: the Analyzer refuses the whole "
            f"window when any y_true is missing ({metrics['perf_nan_windows_failed']} of {metrics['perf_nan_windows']} "
            f"weeks fail; {metrics['perf_nan_weeks_with_partial_labels']} weeks have only part of their labels)",
            f"error 'Dataset has no ground truth for performance assessment'; computed on the available labels, "
            f"{metrics['weeks_truly_degraded']} weeks exceed the RMSE threshold ({PARAMS['rmse_threshold_factor']} x "
            f"reference RMSE {rmse_ref:.3f}); first degraded week {first_true}; when the user removes unlabelled rows "
            f"first, Pulsar.ml confirms degradation in {metrics['perf_labelled_only_weeks_confirmed']} weeks, first at "
            f"week {metrics['perf_labelled_only_first_confirmed_week']} (alert available at week "
            f"{metrics['perf_labelled_only_alert_available_week']} with a {LABEL_LAG}-week label lag)",
            "In Analyzer.run, evaluate performance metrics on the rows with ground truth, add n_labelled and "
            "label_coverage to MetricResults, and refuse only below a configurable minimum coverage (e.g. 30 rows).",
            "critical")
    if metrics["perf_raw_weeks_flagged"]:
        ctx.limit(
            f"With the sentinel left in y_true, {metrics['perf_raw_weeks_flagged']} of {len(perf['raw'])} weeks are "
            f"flagged as performance degradation because -200 is taken as a true value",
            f"MSE by week (raw) first values {[r['mse'] and round(r['mse'], 1) for r in perf['raw'][:8]]} vs threshold "
            f"{mse_threshold:.2f}",
            "Same sentinel option for y_name / pred_name in add_performance_metrics.",
            "major")
    if metrics["weather_confirmed_dropped"]:
        ctx.limit(
            f"Seasonal weather covariates confirm drift ({metrics['weather_confirmed_dropped']} of 3) with the same "
            f"verdict type as the drifting sensors, so sensor ageing cannot be told apart from the seasonal cycle",
            f"first confirmed week (cleaned stream): {metrics['dropped_first_confirmed_week']}; share of weeks confirmed: "
            f"{ {k: round(v, 2) if v is not None else None for k, v in metrics['dropped_confirmed_share'].items()} }",
            "Offer conditional drift (drift of a sensor given T/RH/AH, e.g. residual of a reference regression of the "
            "sensor on the covariates) so that physical sensor drift is separated from covariate seasonality.",
            "minor")
    return metrics
