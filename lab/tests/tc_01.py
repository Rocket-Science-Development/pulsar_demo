"""TC-01 - Concept drift on ELEC2 (OpenML 151, electricity-normalized)

NSW electricity market 1996-1998, 30-min periods, class = price UP/DOWN vs the last 24 h average.
Model trained on the first TRAIN_WEEKS weeks; the rest is streamed in weekly windows (336 rows).
Per window: feature drift (ks_2samp, wasserstein, psi calibrated by AACalibrator on the training period, BH,
DriftPersistence(2)) and accuracy with labels (threshold = mean - 2 sd of leave-one-week-out accuracy on the training
period, DriftPersistence(2)).

Also: a controlled pure concept drift (labels flipped where nswprice > its median for 6 weeks, X unchanged) to
measure what feature drift and performance monitoring see on the same windows.
"""

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import HistGradientBoostingClassifier

from lab.tests import _streams as S

FEATURES = ["day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
NSW = ["day", "period", "nswprice", "nswdemand"]
W = 336
TRAIN_WEEKS = 12
CRITERIA = {"accuracy_decay_detected": ("==", True), "performance_only_windows": (">=", 1),
            "injected_concept_perf_confirmed_within_2": ("==", True)}
PARAMS = {"window_rows": W, "train_weeks": TRAIN_WEEKS, "persistence": 2, "metrics": ["ks_2samp", "wasserstein", "psi"],
          "injection": "labels flipped where nswprice > median, 6 weeks"}


def load() -> pd.DataFrame:
    data, _ = arff.loadarff(S.DATASETS / "openml_elec2_151" / "electricity-normalized.arff")
    df = pd.DataFrame(data)
    df["day"] = df["day"].str.decode("utf-8").astype(int)
    df["y_true"] = (df["class"].str.decode("utf-8") == "UP").astype(int)
    return df.drop(columns=["class"]).reset_index(drop=True)


def model_fit(X, y):
    return HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X, y)


def run(ctx):
    df = load()
    n_train = TRAIN_WEEKS * W
    train = df.iloc[:n_train]
    model = model_fit(train[FEATURES], train.y_true)
    # accuracy threshold from leave-one-week-out on the training period
    accs = []
    for k in range(TRAIN_WEEKS):
        mask = np.zeros(n_train, bool)
        mask[k * W:(k + 1) * W] = True
        m = model_fit(train[FEATURES][~mask], train.y_true[~mask])
        accs.append(float((m.predict(train[FEATURES][mask]) == train.y_true[mask]).mean()))
    acc_threshold = float(np.mean(accs) - 2 * np.std(accs, ddof=1))
    df["y_pred"] = model.predict(df[FEATURES])

    constant_in_ref = [c for c in FEATURES if train[c].nunique() == 1]
    first_var = {c: int(np.argmax(df[c].to_numpy() != df[c].iloc[0])) for c in constant_in_ref}

    analyzer = S.make_analyzer(FEATURES, PARAMS["metrics"], perf_metrics=["accuracy"])
    drift_an = S.make_analyzer(FEATURES, PARAMS["metrics"])
    cal, warns = S.calibrate(drift_an, train.reset_index(drop=True), W, n_draws=40 if S.DRY else 200)
    opts = cal.options()
    opts["accuracy"] = {"threshold": acc_threshold}

    def stream(frame, n_weeks, start):
        fp, pp = S.DriftPersistence(2), S.DriftPersistence(2)
        rows = []
        for i in range(n_weeks):
            cur = frame.iloc[start + i * W:start + (i + 1) * W].reset_index(drop=True)
            if len(cur) < W // 2:
                break
            res = S.analyze(analyzer, cur, train.reset_index(drop=True), cal=cal, options=opts)
            drift = res[res.metric_type == "drift"]
            perf = res[res.metric_type == "performance"]
            fv = fp.update_to_pandas(S.persistence_frame(drift, "drift_fdr"))
            pv = pp.update_to_pandas(perf)
            conf = set(fv[fv.confirmed_drift].feature_name) if not fv.empty else set()
            rows.append({"week": i, "acc": float(perf.metric_value.iloc[0]), "perf_raw": bool(perf.drift_status.iloc[0]),
                         "perf_confirmed": bool(pv.confirmed_drift.any()), "feat_confirmed": sorted(conf),
                         "feat_any": bool(conf), "feat_nsw": bool(conf & set(NSW)),
                         "bh_features": sorted(set(drift[drift.drift_fdr.fillna(False).astype(bool)].feature_name))})
        return rows

    n_weeks = (len(df) - n_train) // W
    if S.DRY:
        n_weeks = 10
    rows = stream(df, n_weeks, n_train)

    def table(rows, key):
        t = {"both": 0, "features_only": 0, "performance_only": 0, "none": 0}
        for r in rows:
            f, p = r[key], r["perf_confirmed"]
            t["both" if f and p else "features_only" if f else "performance_only" if p else "none"] += 1
        return t

    t_all, t_nsw = table(rows, "feat_any"), table(rows, "feat_nsw")
    perf_only_weeks = [r["week"] for r in rows if r["perf_confirmed"] and not r["feat_any"]]
    metrics = {
        "rows": len(df), "weeks_streamed": len(rows), "train_accuracy_loo_weeks": accs, "accuracy_threshold": acc_threshold,
        "accuracy_by_week": [round(r["acc"], 3) for r in rows],
        "accuracy_decay_detected": any(r["perf_confirmed"] for r in rows),
        "perf_confirmed_share": float(np.mean([r["perf_confirmed"] for r in rows])),
        "feature_confirmed_share": float(np.mean([r["feat_any"] for r in rows])),
        "feature_confirmed_share_nsw_only": float(np.mean([r["feat_nsw"] for r in rows])),
        "contingency_all_features": t_all, "contingency_nsw_features": t_nsw,
        "performance_only_windows": t_all["performance_only"], "performance_only_weeks": perf_only_weeks[:40],
        "features_constant_in_reference": constant_in_ref, "row_where_constant_feature_starts_varying": first_var,
        "calibrated_thresholds_constant_features": {k: v for k, v in opts.items() if any(c in k for c in constant_in_ref)},
        "confirmed_share_by_feature": {c: float(np.mean([c in r["feat_confirmed"] for r in rows])) for c in FEATURES},
        "calibration_warning": bool(warns),
    }
    if constant_in_ref:
        vary_week = (min(first_var.values()) - n_train) // W
        after_rows = [r for r in rows if r["week"] > vary_week]
        metrics["constant_features_vary_from_week"] = int(vary_week)
        metrics["constant_features_confirmed_share_after_vary"] = {
            c: (float(np.mean([c in r["feat_confirmed"] for r in after_rows])) if after_rows else None) for c in constant_in_ref}
        metrics["constant_features_confirmed_share_before_vary"] = {
            c: float(np.mean([c in r["feat_confirmed"] for r in rows if r["week"] < vary_week] or [0])) for c in constant_in_ref}

    # controlled pure concept drift on 6 weeks where the baseline is quiet on performance
    quiet_start = None
    for i in range(len(rows) - 8):
        if not any(r["perf_confirmed"] or r["perf_raw"] for r in rows[i:i + 8]):
            quiet_start = i
            break
    if quiet_start is None:
        quiet_start = 0
        ctx.note("no 8-week window without performance alert in the baseline; injection placed at week 0")
    inj = df.copy()
    med = float(train.nswprice.median())
    a, b = n_train + (quiet_start + 2) * W, n_train + (quiet_start + 8) * W
    seg = inj.index[(inj.index >= a) & (inj.index < b) & (inj.nswprice > med)]
    inj.loc[seg, "y_true"] = 1 - inj.loc[seg, "y_true"]
    base = rows[quiet_start:quiet_start + 8]
    after = stream(inj, 8, n_train + quiet_start * W)
    same_feature_verdicts = all(x["bh_features"] == y["bh_features"] for x, y in zip(base, after))
    delay = S.first_index([r["perf_confirmed"] for r in after], 2)
    metrics.update({
        "injected_start_week": quiet_start + 2, "injected_rows_flipped": int(len(seg)),
        "injected_accuracy": [round(r["acc"], 3) for r in after], "baseline_accuracy": [round(r["acc"], 3) for r in base],
        "injected_feature_verdicts_identical": same_feature_verdicts,
        "injected_feature_alerts_during": sum(r["feat_any"] for r in after[2:]),
        "injected_perf_confirm_delay_windows": delay,
        "injected_concept_perf_confirmed_within_2": delay is not None and delay <= 1,
    })

    # ------------------------------------------------------------------ limits
    if same_feature_verdicts:
        ctx.limit(
            "Feature drift tests cannot see a pure concept drift: on 6 weeks where P(y|x) was changed (labels flipped for "
            f"nswprice > median, {len(seg)} rows) the feature verdicts are identical to the baseline; only performance "
            "monitoring with labels detects it",
            f"feature verdicts identical = {same_feature_verdicts}; accuracy baseline {metrics['baseline_accuracy']} vs "
            f"injected {metrics['injected_accuracy']}; performance confirmation delay after onset = {delay} windows "
            f"(+ label latency in production)",
            "Document that feature drift is a proxy only; add a prediction-drift metric (distribution of y_pred and of "
            "predicted probabilities) and a label-free performance estimator (confidence-based, e.g. CBPE) to the Analyzer "
            "so that concept drift with unchanged inputs has a chance before labels arrive.",
            "major")
    if delay is None or delay > 1:
        ctx.limit(
            f"An injected concept drift is not confirmed within 2 windows: accuracy falls from "
            f"{metrics['baseline_accuracy'][2]:.3f} to {metrics['injected_accuracy'][2]:.3f} in the first drifted week, "
            f"above the threshold {acc_threshold:.3f}, so confirmation comes {delay} windows after onset",
            f"injected accuracy {metrics['injected_accuracy']}; threshold = mean - 2 sd of 12 leave-one-week-out "
            f"accuracies {[round(a, 3) for a in accs]}; DriftPersistence(2)",
            "Add a sequential test on the performance stream (CUSUM / Page-Hinkley on per-row correctness, or a "
            "bootstrap CI of the window accuracy vs the reference CI) so a drop of 0.12 is flagged in the first window; "
            "a fixed threshold from weekly accuracies is dominated by week-to-week variance.",
            "minor")
    vary_after = metrics.get("constant_features_confirmed_share_after_vary") or {}
    if constant_in_ref and vary_after and all(v is not None and v >= 0.8 for v in vary_after.values()):
        ctx.limit(
            f"Features constant in the reference ({', '.join(constant_in_ref)}: imputed values in ELEC2 before May 1997) "
            f"get degenerate calibrated thresholds and then alert in every window once they vary",
            f"confirmed share after they start varying (week {metrics["constant_features_vary_from_week"]}) {vary_after}, before "
            f"{metrics['constant_features_confirmed_share_before_vary']}; thresholds "
            f"{metrics['calibrated_thresholds_constant_features']}; confirmed share by feature "
            f"{ {k: round(v, 2) for k, v in metrics['confirmed_share_by_feature'].items()} }; starts varying at rows "
            f"{first_var}",
            "Detect constant / imputed features at reference time (nunique == 1 or a single value > 50 % of rows) and "
            "warn, exclude them from the window verdict, or report them as 'reference not informative'.",
            "major")
    fs = metrics["feature_confirmed_share"]
    if fs >= 0.5:
        ctx.limit(
            f"Feature drift is confirmed in {fs:.0%} of weekly windows over two years "
            f"({metrics['feature_confirmed_share_nsw_only']:.0%} on the NSW features alone), while accuracy is confirmed "
            f"degraded in {metrics['perf_confirmed_share']:.0%}: the feature alarm carries little information on the "
            f"model's health",
            f"contingency (features x performance confirmed) all features {t_all}; NSW only {t_nsw}; accuracy threshold "
            f"{acc_threshold:.3f}",
            "Rank feature alerts by their impact on predictions (drift weighted by feature importance, or drift of the "
            "prediction distribution) and expose the static-reference age; propose reference refresh after a "
            "confirmed, accepted drift.",
            "major")
    if t_all["performance_only"]:
        ctx.note(f"documented performance-only windows (concept moved, features quiet): weeks {perf_only_weeks[:20]}")
    return metrics
