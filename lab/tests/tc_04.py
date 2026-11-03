"""TC-04 - Natural temporal drift across the 2008 crisis (UCI Bank Marketing, bank-additional-full, CC BY 4.0)

41 188 phone-marketing contacts of a Portuguese bank, in date order May 2008 - Nov 2010; the year is not in the file
and is inferred from the month sequence (a new year starts when the month goes back). Macro features (euribor3m,
emp.var.rate, nr.employed, cons.price.idx, cons.conf.idx) move with the crisis; the subscription rate goes from 4 % to
about 50 %. `duration` is excluded (known only after the call: target leak).

Reference: May-June 2008 (12 137 rows, before Lehman, Sept 2008). Model: gradient boosting trained on the reference
(out-of-fold scores on the reference for the performance thresholds). Windows: consecutive chunks of 1000 rows in
date order (the calibration window), each labelled by its months; plus true calendar-month windows (10 to 7 763 rows)
to check robustness on very small windows.
Drift: ks_2samp + wasserstein + psi on numeric features, chi2 + psi on categoricals, calibrated by AACalibrator on the reference,
BH, DriftPersistence(2). Performance: auc (threshold mean - 2 sd of the out-of-fold chunk AUC) and brier (mean + 2 sd).
"""

import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from lab.tests import _mlstreams_p2 as P
from lab.tests import _streams as S
from pulsar_metrics.analyzers.base import Analyzer

NUMERIC = ["age", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m",
           "nr.employed"]
CATEG = ["job", "marital", "education", "default", "housing", "loan", "contact", "day_of_week", "poutcome"]
MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
W = 1000
CRITERIA = {"crisis_drift_confirmed": ("==", True), "macro_leads": ("==", True), "monthly_windows_crash_free": ("==", True)}
PARAMS = {"reference": "May-June 2008", "window_rows": W, "numeric_metrics": ["ks_2samp", "wasserstein", "psi"],
          "categorical_metrics": ["chi2", "psi"], "persistence": 2, "perf_metrics": ["auc", "brier"],
          "excluded": ["duration (target leak)", "month (time itself)"]}


def load() -> pd.DataFrame:
    # bank+marketing.zip (UCI 222) holds bank-additional.zip, extracted under uci_bank_222/
    with zipfile.ZipFile(S.DATASETS / "uci_bank_222" / "bank-additional.zip") as z:
        with z.open("bank-additional/bank-additional-full.csv") as f:
            df = pd.read_csv(f, sep=";")
    m = df.month.map(MONTHS.index)
    year = 2008 + (m.diff().fillna(0) < 0).cumsum()
    df["ym"] = [f"{y}-{mm + 1:02d}" for y, mm in zip(year, m)]
    df["pred_timestamp"] = pd.to_datetime(df.ym + "-01")
    df["y_true"] = (df.y == "yes").astype(int)
    for c in CATEG:
        df[c] = df[c].astype(str)
    return df.reset_index(drop=True)


def encode(frame, cats):
    X = frame[NUMERIC + CATEG].copy()
    for c in CATEG:
        X[c] = pd.Categorical(X[c], categories=cats[c]).codes
    return X


def fit(X, y):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, categorical_features=[X.columns.get_loc(c) for c in CATEG],
                                          random_state=0).fit(X, y)


def make_analyzer(perf=True):
    a = Analyzer(name="lab", model_id=S.MODEL_ID, model_version=S.MODEL_VERSION)
    a.add_drift_metrics(metrics_list=PARAMS["numeric_metrics"], features_list=NUMERIC + ["y_pred"])
    a.add_drift_metrics(metrics_list=PARAMS["categorical_metrics"], features_list=CATEG)
    if perf:
        a.add_performance_metrics(metrics_list=PARAMS["perf_metrics"])
    return a


def run(ctx):
    df = load()
    ref_mask = df.ym.isin(["2008-05", "2008-06"])
    ref = df[ref_mask].reset_index(drop=True)
    rest = df[~ref_mask].reset_index(drop=True)
    cats = {c: sorted(df[c].unique()) for c in CATEG}
    Xr = encode(ref, cats)
    # out-of-fold scores on the reference (contiguous folds) -> performance thresholds
    oof = np.zeros(len(ref))
    for tr, te in KFold(5, shuffle=False).split(Xr):
        oof[te] = fit(Xr.iloc[tr], ref.y_true.iloc[tr]).predict_proba(Xr.iloc[te])[:, 1]
    ref["y_pred"] = oof
    model = fit(Xr, ref.y_true)
    rest["y_pred"] = model.predict_proba(encode(rest, cats))[:, 1]
    from sklearn.metrics import brier_score_loss, roc_auc_score
    chunk_auc, chunk_brier = [], []
    for k in range(len(ref) // W):
        s = ref.iloc[k * W:(k + 1) * W]
        if s.y_true.nunique() == 2:
            chunk_auc.append(roc_auc_score(s.y_true, s.y_pred))
            chunk_brier.append(brier_score_loss(s.y_true, s.y_pred))
    auc_thr = float(np.mean(chunk_auc) - 2 * np.std(chunk_auc, ddof=1))
    brier_thr = float(np.mean(chunk_brier) + 2 * np.std(chunk_brier, ddof=1))

    cols = NUMERIC + CATEG + ["y_pred", "y_true", "pred_timestamp"]
    analyzer = make_analyzer()
    cal, warns = S.calibrate(make_analyzer(perf=False), ref[cols], W, n_draws=40 if S.DRY else 200)
    summ = cal.summary()
    opts = cal.options()
    opts["auc"], opts["brier"] = {"threshold": auc_thr}, {"threshold": brier_thr}

    # fixed-size chunks in date order
    n_chunks = len(rest) // W + (1 if len(rest) % W >= W // 2 else 0)
    if S.DRY:
        n_chunks = 8
    stream = P.WindowStream(analyzer, ref[cols], cal, opts)
    spans = []
    for k in range(n_chunks):
        c = rest.iloc[k * W:(k + 1) * W]
        span = f"{c.ym.iloc[0]}..{c.ym.iloc[-1]}"
        spans.append(span)
        stream.step(c[cols], label=span, pos_rate=round(float(c.y_true.mean()), 3),
                    euribor=round(float(c.euribor3m.mean()), 2))
    rows = stream.rows
    precrisis = [r for r in rows if r["label"].split("..")[-1] <= "2008-08"]
    crisis = [r for r in rows if r["label"].split("..")[0] >= "2008-10"]

    # first confirmation per feature (chunk index in the stream) and "lead" of macro features
    first_conf = {f: P.first(rows, lambda r, f=f: f in r["confirmed"]) for f in NUMERIC + CATEG + ["y_pred"]}
    first_perf = P.first(rows, lambda r: r.get("perf_confirmed"))
    first_crisis_idx = rows.index(crisis[0]) if crisis else None
    macro_first = min([v for f, v in first_conf.items() if f in MACRO and v is not None], default=None)
    other_first = min([v for f, v in first_conf.items() if f not in MACRO and f != "y_pred" and v is not None], default=None)
    # within crisis chunks only (what drifts first once the crisis starts)
    first_conf_crisis = {f: P.first(crisis, lambda r, f=f: f in r["confirmed"]) for f in NUMERIC + CATEG + ["y_pred"]}

    # calendar-month windows (10..7763 rows): robustness, not calibrated for their size
    month_rows, month_errors = [], []
    for ym, g in rest.groupby("ym", sort=False):
        try:
            res = S.analyze(analyzer, g[cols].reset_index(drop=True), ref[cols], cal=cal, options=opts)
            perf = res[res.metric_type == "performance"].set_index("metric_name")
            drift = res[res.metric_type == "drift"]
            month_rows.append({"month": ym, "n": len(g), "pos_rate": round(float(g.y_true.mean()), 3),
                               "bh_features": sorted(set(drift[P.truthy(drift.drift_fdr)].feature_name)),
                               "auc": None if pd.isna(perf.metric_value.get("auc")) else round(float(perf.metric_value["auc"]), 3),
                               "auc_error": perf.error.get("auc") if isinstance(perf.error.get("auc"), str) else None,
                               "drift_no_verdict": int(drift.drift_status.isna().sum())})
        except Exception as exc:  # noqa: BLE001 - recorded
            month_errors.append(f"{ym} ({len(g)} rows): {type(exc).__name__}: {exc}")
        if S.DRY and len(month_rows) >= 6:
            break

    def share(sub, f):
        return round(P.share(sub, lambda r: f in r["confirmed"]), 3) if sub else None

    statuses = summ.groupby("status").size().to_dict() if "status" in summ else {}
    non_ok = summ[summ.status != "ok"][["metric_name", "feature_name", "status", "threshold"]].to_dict("records") \
        if "status" in summ else []
    metrics = {
        "rows": len(df), "reference_rows": len(ref), "chunks": len(rows), "chunk_spans": spans,
        "reference_pos_rate": round(float(ref.y_true.mean()), 3),
        "pos_rate_by_chunk": [r["pos_rate"] for r in rows], "euribor_by_chunk": [r["euribor"] for r in rows],
        "auc_threshold": round(auc_thr, 3), "brier_threshold": round(brier_thr, 4),
        "auc_by_chunk": [None if r.get("perf_auc") is None else round(r["perf_auc"], 3) for r in rows],
        "brier_by_chunk": [None if r.get("perf_brier") is None else round(r["perf_brier"], 4) for r in rows],
        "precrisis_chunks": len(precrisis), "crisis_chunks": len(crisis),
        "precrisis_confirmed_share": {f: share(precrisis, f) for f in NUMERIC + CATEG + ["y_pred"]},
        "crisis_confirmed_share": {f: share(crisis, f) for f in NUMERIC + CATEG + ["y_pred"]},
        "precrisis_any_confirmed_share": round(P.share(precrisis, lambda r: r["confirmed"]), 3) if precrisis else None,
        "crisis_drift_confirmed": bool(crisis) and P.share(crisis, lambda r: set(r["confirmed"]) & set(MACRO)) >= 0.5,
        "first_confirmed_chunk": first_conf, "first_confirmed_chunk_in_crisis": first_conf_crisis,
        "first_perf_confirmed_chunk": first_perf, "first_crisis_chunk": first_crisis_idx,
        "macro_first": macro_first, "other_first": other_first,
        "macro_leads": macro_first is not None and (other_first is None or macro_first <= other_first)
        and (first_perf is None or macro_first <= first_perf),
        "calibration_status_counts": statuses, "calibration_non_ok": non_ok[:30],
        "calibration_warnings": warns[:6], "calibration_draws": cal.n_draws_used,
        "seconds_per_chunk": round(float(np.mean(stream.seconds)), 3),
        "monthly_windows": month_rows, "monthly_errors": month_errors,
        "monthly_windows_crash_free": not month_errors,
    }

    # saturation of the A/A nulls: ks p-values that underflow to 0.0, psi at its epsilon cap
    sat = {}
    for (mname, feat), null in cal.nulls.items():
        if feat in MACRO or feat in ("age", "campaign"):
            v = null.values
            edge = (v <= 0.0) if null.kind == "pvalue" else (v >= 0.999 * v.max())
            sat[f"{mname}|{feat}"] = {"share_at_extreme": round(float(edge.mean()), 3), "extreme": float(v.min() if null.kind == "pvalue" else v.max()),
                                      "threshold": round(float(null.threshold), 6)}
    pair_share = {f"{m}|{f}": round(P.share(crisis, lambda r, k=f"{m}|{f}": k in r["bh_pairs"]), 3) if crisis else None
                  for m in PARAMS["numeric_metrics"] for f in MACRO}
    metrics["null_saturation"] = sat
    metrics["crisis_bh_share_by_metric_macro"] = pair_share
    metrics["reference_distinct_values_macro"] = {f: int(ref[f].nunique()) for f in MACRO}

    # ------------------------------------------------------------------ limits
    blind = [f for f in MACRO if (pair_share.get(f"ks_2samp|{f}") or 0) < 0.2 and (pair_share.get(f"psi|{f}") or 0) < 0.2
             and (pair_share.get(f"wasserstein|{f}") or 0) >= 0.8]
    if blind:
        ctx.limit(
            f"Calibrated ks_2samp and psi are blind to the collapse of the macro features (euribor3m "
            f"{ref.euribor3m.mean():.2f} -> {rest.euribor3m.iloc[-W:].mean():.2f}) because their A/A nulls saturate: "
            f"the reference mixes two monthly regimes (May / June 2008), so most A/A draws already give a ks p-value "
            f"that underflows to 0.0 and a psi at its epsilon cap; the crisis windows tie with the null. Wasserstein, "
            f"unbounded, flags them: {', '.join(blind)}",
            f"null saturation {sat}; BH share over the {len(crisis)} crisis chunks by metric {pair_share}; reference "
            f"distinct values {metrics['reference_distinct_values_macro']}",
            "Calibrate ks on the D statistic (distance) instead of its p-value, which underflows to 0.0; compute the "
            "empirical p-value with a strict inequality on ties at the bound; flag a null as status='saturated' when "
            "more than 5 % of A/A draws sit at the bound (p = 0, psi = cap) and tell the user to prefer wasserstein or "
            "to split the reference by regime.",
            "major")
    pre_macro = {f: metrics["precrisis_confirmed_share"][f] for f in MACRO}
    if precrisis and max(v or 0 for v in pre_macro.values()) >= 0.5:
        ctx.limit(
            f"Macro features with 1-3 distinct values in the reference (one value per month) are confirmed drifting "
            f"in most pre-crisis chunks (Jul-Aug 2008, before Lehman): the 'lead' of macro features is partly an "
            f"artefact of near-constant references",
            f"pre-crisis confirmed share {pre_macro}; reference distinct values "
            f"{ {f: int(ref[f].nunique()) for f in MACRO} }; calibration status counts {statuses}; non-ok "
            f"{[x for x in non_ok if x['feature_name'] in MACRO]}",
            "Flag features whose reference holds fewer than k distinct values (k ~ 5) as 'step features' and test them "
            "as categoricals (chi2 on the levels, with a 'new level' alert), not with ks/psi on sturges bins; report "
            "status=degenerate_null on the result row, not only as a calibration warning.",
            "major")
    small = [m for m in month_rows if m["n"] < 100]
    small_auc = [m for m in small if m["auc"] is not None]
    if small_auc:
        ctx.limit(
            f"Calendar-month windows of {min(m['n'] for m in small)}-{max(m['n'] for m in small)} rows get an AUC and "
            f"BH drift verdicts with thresholds calibrated for {W}-row windows, with no warning on the window size",
            f"small months {[{k: m[k] for k in ('month', 'n', 'auc', 'bh_features')} for m in small]}",
            "Make AACalibrator thresholds size-aware (calibrate per window-size bucket or refuse / flag "
            "reliable=False when len(current) differs from window_size by more than 2x) and add a minimum-row rule to "
            "performance metrics (min_labelled default > 1, e.g. 30, with positives and negatives for auc).",
            "minor")
    if month_errors:
        ctx.limit(f"{len(month_errors)} calendar-month windows crash the analysis",
                  f"{month_errors[:5]}", "Return a result with an error instead of raising.", "major")
    if first_perf is not None and macro_first is not None and first_perf < macro_first:
        ctx.note("performance confirmed before any macro feature")
    ctx.note(f"subscription rate by chunk {metrics['pos_rate_by_chunk']}")
    return metrics
