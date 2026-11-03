"""TS-04 - Year-on-year growth (UCI Bike Sharing, hourly, CC BY 4.0)

Capital Bikeshare (Washington DC), 2011-2012, hourly counts (cnt). Ridership grows by about 64 % from 2011 to 2012,
on top of a strong seasonal cycle. Model: gradient boosting on calendar + weather features (no `yr`), trained on 2011.
The 2011 reference carries out-of-fold predictions (leave-one-month-out), so its error is a real error.

Monitoring: weekly windows (7 calendar days, about 168 rows) through 2012; labels arrive 48 h late, so a window
evaluated at its end has its last 48 rows unlabelled (performance on the labelled rows, label_coverage recorded); the
same window is evaluated again once all labels are in ("backfilled").

Two reference schemes, same metrics (ks_2samp, wasserstein, psi on temp, hum, windspeed, weathersit, hr, the target
cnt and the prediction; AACalibrator, BH, DriftPersistence(2)):
  - static: the whole of 2011 (what Analyzer.run is given in practice); its 2011 weeks are also streamed against it,
    which measures the seasonal confounding (in-sample weeks alerting only because of the season);
  - year-on-year: each 2012 week against the same calendar weeks of 2011 (+-4 weeks), calibrated on those weeks.
Performance: MAE with threshold mean + 2 sd of the weekly out-of-fold MAE of 2011, DriftPersistence(2).
"""

import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from lab.tests import _mlstreams_p2 as P
from lab.tests import _streams as S

MODEL_FEATURES = ["season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum",
                  "windspeed"]
DRIFT_FEATURES = ["temp", "hum", "windspeed", "weathersit", "hr", "y_true", "y_pred"]
WEATHER = ["temp", "hum", "windspeed", "weathersit"]
W = 168
DELAY_H = 48
YOY_HALF_WEEKS = 4
CRITERIA = {"perf_detected_2012": ("==", True), "growth_detected_yoy_cnt_share": (">=", 0.5),
            "crash_free": ("==", True)}
PARAMS = {"window": "7 days (~168 rows)", "label_delay_hours": DELAY_H, "metrics": ["ks_2samp", "wasserstein", "psi"],
          "drift_features": DRIFT_FEATURES, "persistence": 2, "yoy_reference_weeks": 2 * YOY_HALF_WEEKS + 1,
          "perf_threshold": "mean + 2 sd of weekly out-of-fold MAE on 2011"}


def load() -> pd.DataFrame:
    with zipfile.ZipFile(S.DATASETS / "uci_bike_275" / "bike.zip") as z:
        with z.open("hour.csv") as f:
            df = pd.read_csv(f, parse_dates=["dteday"])
    df["ts"] = df.dteday + pd.to_timedelta(df.hr, unit="h")
    df["day"] = (df.dteday - df.dteday.min()).dt.days
    df["week"] = df.day // 7
    df["y_true"] = df.cnt.astype(float)
    df["pred_timestamp"] = df.ts
    return df.sort_values("ts").reset_index(drop=True)


def fit(X, y):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0).fit(X, y)


def run(ctx):
    df = load()
    y11 = df[df.yr == 0].copy()
    y12 = df[df.yr == 1].copy()
    # out-of-fold predictions on 2011 (leave-one-month-out)
    oof = np.zeros(len(y11))
    for m in range(1, 13):
        mask = (y11.mnth == m).to_numpy()
        oof[mask] = fit(y11.loc[~mask, MODEL_FEATURES], y11.loc[~mask, "cnt"]).predict(y11.loc[mask, MODEL_FEATURES])
    y11["y_pred"] = np.clip(oof, 0, None)
    model = fit(y11[MODEL_FEATURES], y11.cnt)
    y12["y_pred"] = np.clip(model.predict(y12[MODEL_FEATURES]), 0, None)

    ref = y11[DRIFT_FEATURES + ["pred_timestamp"]].reset_index(drop=True)
    wk11 = y11.groupby("week")
    mae11 = [float(np.mean(np.abs(g.y_true - g.y_pred))) for _, g in wk11 if len(g) >= 100]
    thr = float(np.mean(mae11) + 2 * np.std(mae11, ddof=1))

    analyzer = S.make_analyzer(DRIFT_FEATURES, PARAMS["metrics"], perf_metrics=["mae"])
    drift_an = S.make_analyzer(DRIFT_FEATURES, PARAMS["metrics"])
    cal, warns = S.calibrate(drift_an, ref, W, n_draws=40 if S.DRY else 200)
    opts = cal.options()
    opts["mae"] = {"threshold": thr}

    def delayed(g):
        g = g.copy()
        cutoff = g.ts.max() - pd.Timedelta(hours=DELAY_H)
        g.loc[g.ts > cutoff, "y_true"] = np.nan
        return g

    weeks12 = [g for _, g in y12.groupby("week") if len(g) >= 24]
    weeks11 = [g for _, g in y11.groupby("week") if len(g) >= 24]
    if S.DRY:
        weeks12, weeks11 = weeks12[:8], weeks11[:8]
    cols = DRIFT_FEATURES + ["pred_timestamp"]

    # static reference: 2011 in-sample weeks (seasonal confounding) then 2012, labels delayed
    st11 = P.WindowStream(analyzer, ref, cal, opts)
    for g in weeks11:
        st11.step(delayed(g)[cols], label=str(g.dteday.min().date()))
    st12 = P.WindowStream(analyzer, ref, cal, opts)
    bf12 = P.WindowStream(analyzer, ref, cal, opts)
    bias12 = []
    for g in weeks12:
        st12.step(delayed(g)[cols], label=str(g.dteday.min().date()))
        bf12.step(g[cols], label=str(g.dteday.min().date()))
        bias12.append(round(float((g.y_pred - g.y_true).mean()), 1))

    # year-on-year reference: the same calendar weeks of 2011
    yoy_rows, yoy_errors = [], []
    yoy_metrics = ["ks_2samp", "psi"]
    yoy_an = S.make_analyzer(DRIFT_FEATURES, yoy_metrics)
    w11_index = {int(g.week.iloc[0]): g for g in weeks11} if S.DRY else {int(w): g for w, g in wk11}
    yoy_fp = S.DriftPersistence(2)
    for g in weeks12:
        w = int(g.week.iloc[0]) - 52  # 52 weeks earlier, same weekday alignment
        sel = [w11_index[k] for k in range(w - YOY_HALF_WEEKS, w + YOY_HALF_WEEKS + 1) if k in w11_index]
        if len(sel) < 4:
            yoy_errors.append(f"week {w}: {len(sel)} reference weeks")
            continue
        r = pd.concat(sel)[cols].reset_index(drop=True)
        try:
            c, _ = S.calibrate(yoy_an, r, W, n_draws=40 if S.DRY else 200, allow_short=True)
            res = S.analyze(yoy_an, delayed(g)[cols].reset_index(drop=True), r, cal=c)
        except Exception as exc:  # noqa: BLE001 - recorded, the stream goes on
            yoy_errors.append(f"week {w}: {type(exc).__name__}: {exc}")
            continue
        drift = res[res.metric_type == "drift"]
        fv = yoy_fp.update_to_pandas(S.persistence_frame(drift, "drift_fdr"))
        yoy_rows.append({"label": str(g.dteday.min().date()),
                         "bh": sorted(set(drift[P.truthy(drift.drift_fdr)].feature_name)),
                         "confirmed": sorted(set(fv[fv.confirmed_drift].feature_name)) if not fv.empty else []})

    def conf_share(rows, feat, key="confirmed"):
        return round(P.share(rows, lambda r: feat in r[key]), 3) if rows else None

    by_feat = lambda rows, key="confirmed": {f: conf_share(rows, f, key) for f in DRIFT_FEATURES}  # noqa: E731
    first_perf = P.first(st12.rows, lambda r: r.get("perf_confirmed"))
    first_perf_bf = P.first(bf12.rows, lambda r: r.get("perf_confirmed"))
    cov = [r.get("label_coverage") for r in st12.rows]
    metrics = {
        "rows_2011": len(y11), "rows_2012": len(y12), "weeks_2011": len(weeks11), "weeks_2012": len(weeks12),
        "cnt_mean_2011": round(float(y11.cnt.mean()), 1), "cnt_mean_2012": round(float(y12.cnt.mean()), 1),
        "growth_pct": round(100 * (y12.cnt.mean() / y11.cnt.mean() - 1), 1),
        "oof_weekly_mae_2011_mean": round(float(np.mean(mae11)), 1), "mae_threshold": round(thr, 1),
        "mae_by_week_2012_delayed": [None if r.get("perf_mae") is None else round(r["perf_mae"], 1) for r in st12.rows],
        "bias_by_week_2012": bias12,
        "label_coverage_delayed_mean": round(float(np.nanmean([c for c in cov if c is not None])), 3) if cov else None,
        "perf_detected_2012": first_perf is not None,
        "perf_first_confirmed_week_2012_delayed": first_perf,
        "perf_first_confirmed_week_2012_backfilled": first_perf_bf,
        "perf_confirmed_share_2012": round(P.share(st12.rows, lambda r: r.get("perf_confirmed")), 3),
        "perf_confirmed_share_2011_oof": round(P.share(st11.rows, lambda r: r.get("perf_confirmed")), 3),
        "static_confirmed_share_2011_in_sample": by_feat(st11.rows),
        "static_confirmed_share_2012": by_feat(st12.rows),
        "static_any_weather_confirmed_2011_in_sample": round(P.share(st11.rows, lambda r: set(r["confirmed"]) & set(WEATHER)), 3),
        "yoy_weeks": len(yoy_rows), "yoy_errors": yoy_errors[:10],
        "yoy_confirmed_share_2012": by_feat(yoy_rows),
        "yoy_bh_share_2012": by_feat(yoy_rows, "bh"),
        "growth_detected_yoy_cnt_share": conf_share(yoy_rows, "y_true"),
        "seconds_per_window_static": round(float(np.mean(st12.seconds)), 3),
        "calibration_warnings": warns[:3], "calibration_draws": cal.n_draws_used,
        "crash_free": not yoy_errors,
    }
    sc11 = metrics["static_confirmed_share_2011_in_sample"]
    sc12 = metrics["static_confirmed_share_2012"]
    yc = metrics["yoy_confirmed_share_2012"]
    metrics["seasonal_confounding_static_temp_2011"] = sc11["temp"]

    # ------------------------------------------------------------------ limits
    if sc11["temp"] is not None and sc11["temp"] >= 0.3:
        ctx.limit(
            f"With the only reference Analyzer.run accepts (a static frame: all of 2011), the in-sample 2011 weeks "
            f"themselves are confirmed drifting on temp in {sc11['temp']:.0%} of weeks: season is reported as drift, "
            f"and growth on cnt cannot be told apart from season (cnt confirmed in {sc11['y_true']:.0%} of 2011 weeks, "
            f"{sc12['y_true']:.0%} of 2012 weeks)",
            f"static reference, confirmed share by feature, 2011 in-sample weeks {sc11}; 2012 weeks {sc12}; "
            f"year-on-year reference (same weeks of 2011 +-{YOY_HALF_WEEKS}) 2012 weeks {yc}; growth "
            f"{metrics['growth_pct']} %",
            "Add a reference selector to the Analyzer / scheduler: 'same period last year' (+- k weeks) and rolling "
            "references, with the calibration cached per reference slice; document that a static all-year reference "
            "conflates seasonality and drift on seasonal data.",
            "major")
    thr_cnt = {k: v for k, v in opts.items() if k.endswith("|y_true")}
    metrics["static_calibrated_thresholds_cnt"] = thr_cnt
    if sc12["y_true"] is not None and yc["y_true"] is not None and yc["y_true"] - sc12["y_true"] >= 0.3:
        ctx.limit(
            f"The static all-2011 reference is nearly blind to the +{metrics['growth_pct']} % growth: contiguous A/A "
            f"draws on a seasonal year compare winter with summer weeks, so the calibrated thresholds absorb the season "
            f"and cnt is confirmed drifting in {sc12['y_true']:.0%} of 2012 weeks, against {yc['y_true']:.0%} with a "
            f"year-on-year reference",
            f"calibrated thresholds on cnt (static) {thr_cnt}; static 2012 confirmed share {sc12}; year-on-year "
            f"{yc}; cnt mean 2011 {metrics['cnt_mean_2011']} vs 2012 {metrics['cnt_mean_2012']}",
            "Calibrate per season (A/A draws restricted to the same period of the reference, e.g. same month +- k weeks) "
            "or offer a year-on-year reference; warn when the A/A null of a feature is dominated by between-period "
            "variance (ratio of between-window to within-window variance).",
            "major")
    if yc["y_true"] is not None and yc["y_true"] < 0.5:
        ctx.limit(
            f"Even with a year-on-year reference, the +{metrics['growth_pct']} % growth of cnt is confirmed in only "
            f"{yc['y_true']:.0%} of 2012 weeks",
            f"year-on-year confirmed share {yc}; BH share {metrics['yoy_bh_share_2012']}",
            "Short year-on-year references (9 weeks) give wide calibrated thresholds; add a level-shift test on the "
            "target/prediction mean (ratio to last year with a CI) for growth monitoring.",
            "minor")
    if first_perf is not None and first_perf_bf is not None and first_perf > first_perf_bf:
        ctx.note(f"48 h label delay postpones the performance confirmation from week {first_perf_bf} (backfilled) to "
                 f"week {first_perf} (as-of evaluation)")
    if first_perf is None:
        ctx.limit(
            f"Under-prediction from growth (mean bias {np.mean(bias12):.0f} rides/h in 2012) is never confirmed by the "
            f"MAE threshold of {thr:.1f}",
            f"MAE by week {metrics['mae_by_week_2012_delayed']}; bias by week {bias12}",
            "Add a signed error metric (mean error / bias) to PerformanceMetricsFuncs and a relative threshold mode.",
            "major")
    ctx.note(f"seasonal confounding of the performance threshold: weekly OOF MAE in 2011 ranges "
             f"{min(mae11):.1f}-{max(mae11):.1f}; threshold {thr:.1f}")
    return metrics
