"""TC-03 - Injected covariate shift and explanation drift (UCI Covertype, CC BY 4.0)

Binary task: Lodgepole Pine (Cover_Type 2) vs the rest; gradient boosting trained on wilderness area 1 (Rawah).
Dose-response: each current window (N rows) mixes a share p of rows from another population into rows of area 1:
  - "wilderness": rows of wilderness area 3 (Comanche Peak)
  - "elevation":  rows of area 1 above the 80th percentile of Elevation (selection shift inside the same area)
Feature drift: ks_2samp, wasserstein, psi on the 10 numeric features, calibrated by AACalibrator on area-1 rows,
BH over the window; confirmation = DriftPersistence(2) over consecutive replicate windows at the same p.
Explanation drift: TreeSHAP (pulsar_metrics.explain.tree_attributions) on the window, AttributionDrift against the
reference window; importance_by_group by wilderness area on the mixed window.

AttributionDrift applies a documented share tolerance (0.05 by default): an alert exactly at the crossing of that
tolerance is the documented behaviour, not a lag. The test therefore varies `share_tolerance` (SHARE_TOLERANCES) and
reports the dose-response per tolerance, with the A/A alert rate at 0 % for each; a limit is recorded only when a
share difference well beyond the tolerance in force (> 2 x tolerance) gets no alert.
"""

import gzip
import io
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from lab.tests import _streams as S
from pulsar_metrics.explain import AttributionDrift, importance, importance_by_group, tree_attributions

NUMERIC = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
           "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
           "Horizontal_Distance_To_Fire_Points"]
WILD = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL = [f"Soil_Type{i}" for i in range(1, 41)]
COLUMNS = NUMERIC + WILD + SOIL + ["Cover_Type"]
N = 2000
DOSES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
SHARE_TOLERANCES = [0.01, 0.02, 0.05]  # 0.05 = AttributionDrift default
CRITERIA = {"far_p0_wilderness": ("<=", 0.05), "far_p0_elevation": ("<=", 0.05), "attribution_alert_p0": ("<=", 0.05),
            "attribution_moves_with_shift": ("==", True)}
PARAMS = {"window_rows": N, "doses": DOSES, "replicates_drift": 30, "replicates_shap": 6, "model": "GradientBoosting 150x3",
          "metrics": ["ks_2samp", "wasserstein", "psi"], "share_tolerances": SHARE_TOLERANCES}


def load() -> pd.DataFrame:
    cache = S.CACHE / "covtype.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    with zipfile.ZipFile(S.DATASETS / "uci_covertype_31" / "covertype.zip") as z:
        raw = gzip.decompress(z.read("covtype.data.gz"))
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COLUMNS)
    df.to_pickle(cache)
    return df


def run(ctx):
    df = load()
    rng = np.random.default_rng(0)
    df["y_true"] = (df.Cover_Type == 2).astype(int)
    feats = NUMERIC + WILD + SOIL
    a1 = df[df.Wilderness_Area1 == 1].sample(frac=1, random_state=0).reset_index(drop=True)
    a3 = df[df.Wilderness_Area3 == 1].sample(frac=1, random_state=1).reset_index(drop=True)
    train, cal_pool, ref_win, pool = a1.iloc[:30000], a1.iloc[30000:50000], a1.iloc[50000:50000 + N], a1.iloc[50000 + N:]
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0).fit(train[feats], train.y_true)
    elev_cut = float(a1.Elevation.quantile(0.8))
    pool_low = pool  # area-1 rows of any elevation
    pool_high = pool[pool.Elevation > elev_cut]

    reps = 6 if S.DRY else PARAMS["replicates_drift"]
    reps_shap = 2 if S.DRY else PARAMS["replicates_shap"]
    doses = [0.0, 0.1, 0.5, 1.0] if S.DRY else DOSES

    analyzer = S.make_analyzer(NUMERIC, PARAMS["metrics"], perf_metrics=["accuracy"])
    drift_an = S.make_analyzer(NUMERIC, PARAMS["metrics"])
    cal, warns = S.calibrate(drift_an, cal_pool[NUMERIC].reset_index(drop=True), N, n_draws=60 if S.DRY else 300)
    ctx.note(f"calibration warnings: {warns}")

    ref_attr = tree_attributions(model, ref_win[feats])
    ad = AttributionDrift(k=5, n_bootstrap=100 if S.DRY else 200, seed=0).fit(ref_attr)
    ads = {tol: AttributionDrift(k=5, n_bootstrap=100 if S.DRY else 200, seed=0, share_tolerance=tol).fit(ref_attr)
           for tol in SHARE_TOLERANCES}
    ref_imp = importance(ref_attr)
    metrics = {"reference_attribution_stability": ad.stability, "elevation_cut_p80": elev_cut,
               "reference_top5_importance": ref_imp.sort_values(ascending=False).head(5).round(4).to_dict(),
               "model_accuracy_reference": float((model.predict(ref_win[feats]) == ref_win.y_true).mean())}

    def window(kind, p):
        k = int(round(p * N))
        base = pool_low.sample(N - k, random_state=int(rng.integers(1 << 31)))
        other = (a3 if kind == "wilderness" else pool_high).sample(k, random_state=int(rng.integers(1 << 31)))
        w = pd.concat([base, other]).reset_index(drop=True)
        w["y_pred"] = model.predict(w[feats])
        return w

    curves = {}
    for kind in ("wilderness", "elevation"):
        curve = []
        for p in doses:
            pers = S.DriftPersistence(2)
            raw_flags, conf_flags, accs, n_rej = [], [], [], []
            attr = []
            for r in range(reps):
                w = window(kind, p)
                res = S.analyze(analyzer, w[NUMERIC + ["y_true", "y_pred"]], ref_win[NUMERIC + ["y_true"]].assign(
                    y_pred=model.predict(ref_win[feats])), cal=cal)
                drift = res[res.metric_type == "drift"]
                raw_flags.append(S.window_flag(drift, "drift_fdr"))
                n_rej.append(int(drift.drift_fdr.fillna(False).astype(bool).sum()))
                v = pers.update_to_pandas(S.persistence_frame(drift, "drift_fdr"))
                conf_flags.append(S.any_confirmed(v))
                accs.append(float(res[res.metric_type == "performance"].metric_value.iloc[0]))
                if r < reps_shap:
                    cur_attr = tree_attributions(model, w[feats])
                    out = pd.DataFrame([x.model_dump() for x in ad.run(cur_attr)])
                    nd = out[out.metric_name == "attribution_ndcg_at_k"].iloc[0]
                    jc = out[out.metric_name == "attribution_top_k_jaccard"].iloc[0]
                    sh = out[out.metric_name == "importance_share_difference"]
                    cur_imp = importance(cur_attr)
                    tol_alerts = {}
                    for tol, adt in ads.items():
                        o = pd.DataFrame([x.model_dump() for x in adt.run(cur_attr)])
                        tol_alerts[tol] = int((o[o.metric_name == "importance_share_difference"].drift_status == True).sum())  # noqa: E712
                    g, gdiff = None, None
                    if kind == "wilderness" and 0 < p < 1:
                        grp = pd.Series(np.where(w.Wilderness_Area3 == 1, "area3", "area1"), index=w.index, name="area")
                        gres = pd.DataFrame([x.model_dump() for x in importance_by_group(cur_attr, grp, reference_group="area1",
                                                                                          n_bootstrap=100)])
                        g = int((gres.drift_status == True).sum())  # noqa: E712
                        m3, m1 = (grp == "area3").to_numpy(), (grp == "area1").to_numpy()
                        gdiff = float((importance(cur_attr[m3]) - importance(cur_attr[m1])).abs().max())
                    attr.append({
                        "ndcg": float(nd.metric_value), "ndcg_status": nd.drift_status, "jaccard": float(jc.metric_value),
                        "jaccard_status": jc.drift_status,
                        "share_alerts": int((sh.drift_status == True).sum()),  # noqa: E712
                        "share_inconclusive": int(sh.drift_status.isna().sum()),
                        "elev_share": float(cur_imp["Elevation"]), "wild_share": float(cur_imp[WILD].sum()),
                        "soil_share": float(cur_imp[SOIL].sum()),
                        "max_abs_share_diff": float(sh.metric_value.abs().max()),
                        "group_alerts": g, "group_max_point_diff": gdiff,
                        **{f"share_alerts_tol{tol}": n for tol, n in tol_alerts.items()},
                    })
            row = {"p": p, "detect_raw": float(np.mean(raw_flags)), "detect_confirmed": float(np.mean(conf_flags[1:])) if reps > 1 else None,
                   "mean_bh_rejections": float(np.mean(n_rej)), "accuracy": float(np.mean(accs)), "replicates": reps}
            if attr:
                A = pd.DataFrame(attr)
                row.update({
                    "attr_ndcg": float(A.ndcg.mean()), "attr_jaccard": float(A.jaccard.mean()),
                    "attr_any_alert_share": float(((A.ndcg_status == True) | (A.jaccard_status == True) | (A.share_alerts > 0)).mean()),  # noqa: E712
                    "attr_share_alerts_mean": float(A.share_alerts.mean()),
                    "attr_inconclusive_mean": float(A.share_inconclusive.mean()),
                    "elev_share": float(A.elev_share.mean()), "wild_share": float(A.wild_share.mean()),
                    "soil_share": float(A.soil_share.mean()), "max_abs_share_diff": float(A.max_abs_share_diff.mean()),
                    "group_alerts_mean": None if A.group_alerts.isna().all() else float(A.group_alerts.mean()),
                    "group_max_point_diff": None if A.group_max_point_diff.isna().all() else float(A.group_max_point_diff.mean()),
                    **{f"share_alert_share_tol{tol}": float((A[f"share_alerts_tol{tol}"] > 0).mean()) for tol in SHARE_TOLERANCES},
                })
            curve.append(row)
        curves[kind] = curve
        c0 = curve[0]
        metrics[f"far_p0_{kind}"] = c0["detect_raw"]
        metrics[f"attr_alert_p0_{kind}"] = c0.get("attr_any_alert_share")
        # smallest dose with >= 80 % raw detection
        mdd = next((c["p"] for c in curve if c["detect_raw"] >= 0.8), None)
        metrics[f"min_dose_80pct_power_{kind}"] = mdd
        amd = next((c["p"] for c in curve if c.get("attr_any_alert_share", 0) >= 0.8), None)
        metrics[f"min_dose_80pct_attribution_alert_{kind}"] = amd
        for tol in SHARE_TOLERANCES:
            metrics[f"attr_share_alert_p0_{kind}_tol{tol}"] = c0.get(f"share_alert_share_tol{tol}")
            metrics[f"min_dose_80pct_share_alert_{kind}_tol{tol}"] = next(
                (c["p"] for c in curve if c.get(f"share_alert_share_tol{tol}", 0) >= 0.8), None)
    metrics["curves"] = curves
    metrics["attribution_alert_p0"] = max(metrics["attr_alert_p0_wilderness"] or 0, metrics["attr_alert_p0_elevation"] or 0)
    cw = curves["wilderness"]
    metrics["attribution_moves_with_shift"] = bool(
        abs(cw[-1].get("elev_share", 0) - cw[0].get("elev_share", 0)) > 0.02 or cw[-1].get("attr_ndcg", 1) < cw[0].get("attr_ndcg", 1) - 0.02)

    # ------------------------------------------------------------------ limits
    for kind in ("wilderness", "elevation"):
        far = metrics[f"far_p0_{kind}"]
        if S.wilson(int(round(far * reps)), reps)[0] > 0.05:  # only when the excess over 5 % is significant
            ctx.limit(
                f"False alarms at 0 % shift ({kind} run): {far:.0%} of A/A windows alerted after calibration + BH",
                f"{S.fmt_rate(int(round(far * reps)), reps)}; windows of {N} rows, calibration pool 20 000 rows, 10 "
                f"numeric features x 3 metrics",
                "Check the calibration draw count and the BH family; report the empirical FWER of the calibrated "
                "pipeline in AACalibrator (simulate whole windows, not one test at a time).",
                "major")
    for kind in ("wilderness", "elevation"):
        cur = curves[kind]
        mdd_f, mdd_a = metrics[f"min_dose_80pct_power_{kind}"], metrics[f"min_dose_80pct_attribution_alert_{kind}"]
        sweep = {tol: metrics[f"min_dose_80pct_share_alert_{kind}_tol{tol}"] for tol in SHARE_TOLERANCES}
        ctx.note(f"{kind}: feature drift 80 % at p = {mdd_f}, AttributionDrift (default tolerance) at p = {mdd_a}; share "
                 f"alerts 80 % by share_tolerance {sweep}; A/A share alert rate by tolerance "
                 f"{ {tol: metrics[f'attr_share_alert_p0_{kind}_tol{tol}'] for tol in SHARE_TOLERANCES} }. An alert at "
                 f"the crossing of the documented tolerance is the expected behaviour, not a lag.")
        # a limit only when a share difference well beyond the tolerance in force gets no alert
        for tol in SHARE_TOLERANCES:
            missed = [c for c in cur if "max_abs_share_diff" in c and c["max_abs_share_diff"] > 2 * tol
                      and c.get(f"share_alert_share_tol{tol}", 1) < 0.5]
            if missed:
                ctx.limit(
                    f"AttributionDrift misses share differences beyond twice its tolerance ({kind} shift, "
                    f"share_tolerance {tol}): {len(missed)} dose(s) with a mean max share difference > {2 * tol:.2f} "
                    f"alerted in fewer than half of the windows",
                    "; ".join(f"p={c['p']:.0%}: max share diff {c['max_abs_share_diff']:.3f}, alert share "
                              f"{c[f'share_alert_share_tol{tol}']:.0%}, inconclusive "
                              f"{c.get('attr_inconclusive_mean', float('nan')):.1f}/54" for c in missed)[:900],
                    "Report a per-feature attribution-shift p-value from A/A draws at the window size (like "
                    "AACalibrator) next to the tolerance verdict, and offer grouping of one-hot columns (Soil_Type*, "
                    "Wilderness_Area*) into one feature before the comparison.",
                    "minor")
                break
    ws = [c for c in cw if c.get("group_alerts_mean") is not None]
    missed = [c for c in ws if c["group_alerts_mean"] == 0 and (c["group_max_point_diff"] or 0) > 0.10]
    if missed:
        ctx.limit(
            f"importance_by_group gives no verdict of difference between wilderness areas in {len(missed)} of {len(ws)} "
            f"mixed doses although the point estimate of a feature's share differs by more than twice the 5 % tolerance",
            "; ".join(f"p={c['p']:.0%}: max point difference {c['group_max_point_diff']:.3f}, group alerts "
                      f"{c['group_alerts_mean']}" for c in ws),
            "Report inconclusive counts and the minimum group size needed for a verdict; small groups at low p have "
            "very wide simultaneous intervals (Bonferroni over 54 features).",
            "minor")
    ctx.note("wilderness one-hot columns get zero attribution: the model is trained on area 1 only, where they are "
             "constant, so a wilderness shift is visible in attributions only through the other features")
    return metrics
