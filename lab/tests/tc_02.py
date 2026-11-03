"""TC-02 - Large stream with high-cardinality categoricals (OpenML 1169 airlines, licence "Public" on OpenML)

539 383 US flights (2009 Data Expo / BTS on-time data, ~4 weeks), ordered by day then scheduled departure time.
Categoricals: Airline (18 levels), AirportFrom and AirportTo (293 levels each); numeric: Time, Length. Target: Delay.
The day is derived from the DayOfWeek sequence. Windows: one day (~20 000 rows); reference: the first 11 days (Saturdays are ~14k rows).
Model: gradient boosting on the reference days (categoricals as integer codes).

Measured:
  1. sparse levels per window: levels, share of contingency cells with an expected count < 5 (Cochran rule), levels
     absent from the reference;
  2. nominal chi2 false-alarm rate (p < 0.05) on A/A draws of iid rows from the reference, for several window sizes,
     raw levels vs levels grouped (top-K + "other"), all through pulsar_metrics' chi2;
  3. the calibrated stream (chi2 + psi on categoricals, ks_2samp + psi on numerics, AACalibrator, BH, persistence 2)
     over all days, with the runtime per window, of chi2 alone, of the calibration, and of the whole stream.
"""

import time

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import HistGradientBoostingClassifier

from lab.tests import _mlstreams_p2 as P
from lab.tests import _streams as S
from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.metrics.drift import DriftTestMetric
from pulsar_metrics.metrics.utils import get_population_counts

CATEG = ["Airline", "AirportFrom", "AirportTo"]
NUMERIC = ["Time", "Length"]
REF_DAYS = 11  # Saturdays hold ~14k rows: 11 days are needed for >= 10 windows of the median day size
AA_SIZES = [500, 2000, 5000, 20000]
TOP_K = 30
SHIFT = 0.05
CRITERIA = {"crash_free": ("==", True), "seconds_per_window_max": ("<=", 60.0)}
PARAMS = {"window": "1 day (~20k rows)", "reference_days": REF_DAYS, "categorical_metrics": ["chi2", "psi"],
          "numeric_metrics": ["ks_2samp", "psi"], "aa_window_sizes": AA_SIZES, "aa_draws": 200, "grouping_top_k": TOP_K, "power_shift_share": SHIFT,
          "persistence": 2}


def load() -> pd.DataFrame:
    cache = S.CACHE / "airlines_1169.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    data, _ = arff.loadarff(S.DATASETS / "openml_airlines_1169" / "airlines.arff")
    df = pd.DataFrame(data)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.decode("utf-8")
    df["DayOfWeek"] = df.DayOfWeek.astype(int)
    df["day"] = (df.DayOfWeek != df.DayOfWeek.shift()).cumsum() - 1
    df["y_true"] = df.Delay.astype(int)
    df["pred_timestamp"] = pd.Timestamp("2008-01-01") + pd.to_timedelta(df.day, unit="D") + pd.to_timedelta(df.Time, unit="m")
    df = df.drop(columns=["Delay"])
    df.to_pickle(cache)
    return df


def group_levels(series: pd.Series, keep) -> pd.Series:
    return series.where(series.isin(keep), "other")


def chi2_p(new: pd.Series, ref: pd.Series) -> float:
    m = DriftTestMetric("chi2", "x")
    r = m.evaluate(pd.DataFrame({"x": new.reset_index(drop=True)}), pd.DataFrame({"x": ref.reset_index(drop=True)}))
    return r.metric_value


def sparsity(new: pd.Series, ref: pd.Series) -> dict:
    counts = get_population_counts(new, ref)
    tot = counts.to_numpy().sum()
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / tot
    return {"levels": int(len(counts)), "cells_expected_lt5": round(float((expected < 5).mean()), 3),
            "levels_absent_in_ref": int((counts.ref == 0).sum()), "levels_absent_in_window": int((counts.new == 0).sum())}


def run(ctx):
    t_load = time.perf_counter()
    df = load()
    t_load = time.perf_counter() - t_load
    n_days = int(df.day.max()) + 1
    ref = df[df.day < REF_DAYS].reset_index(drop=True)
    cur_days = list(range(REF_DAYS, n_days))
    if S.DRY:
        ref = df[df.day < 5].reset_index(drop=True)
        cur_days = list(range(5, 8))
    # codes by frequency rank, capped at 254: HistGradientBoosting accepts at most 255 categories (model only)
    codes = {c: {v: min(i, 254) for i, v in enumerate(df[c].value_counts().index)} for c in CATEG}

    def X(frame):
        out = frame[CATEG + NUMERIC + ["DayOfWeek"]].copy()
        for c in CATEG:
            out[c] = out[c].map(codes[c])
        return out

    model = HistGradientBoostingClassifier(max_iter=150, random_state=0, categorical_features=[0, 1, 2],
                                           max_bins=255).fit(X(ref), ref.y_true)
    df["y_pred"] = model.predict(X(df))
    ref = df[df.day < (5 if S.DRY else REF_DAYS)].reset_index(drop=True)
    day_size = int(ref.groupby("day").size().median())

    # 1. sparsity of the categoricals per day window vs the reference
    sp = {c: [sparsity(df[df.day == d][c], ref[c]) for d in cur_days] for c in CATEG}
    sp_summary = {c: {"levels_max": max(x["levels"] for x in v), "cells_expected_lt5_mean": round(float(np.mean([x["cells_expected_lt5"] for x in v])), 3),
                      "levels_absent_in_ref_max": max(x["levels_absent_in_ref"] for x in v)} for c, v in sp.items()}

    # 2. nominal chi2 false-alarm rate on iid A/A draws from the reference: raw vs grouped levels
    rng = np.random.default_rng(0)
    n_aa = 40 if S.DRY else PARAMS["aa_draws"]
    top = {c: ref[c].value_counts().index[:TOP_K] for c in CATEG}
    aa = {}
    chi2_seconds = []
    for c in ["AirportFrom", "Airline"]:
        aa[c] = {}
        for n in AA_SIZES:
            if 2 * n > len(ref):
                continue
            raw, grp, lt5 = [], [], []
            pw = {"top_raw": [], "top_grp": [], "rare_raw": [], "rare_grp": []}
            rare_levels = ref[c].value_counts().index[TOP_K:].to_numpy()
            for _ in range(n_aa):
                idx = rng.choice(len(ref), 2 * n, replace=False)
                a, b = ref[c].iloc[idx[:n]], ref[c].iloc[idx[n:]]
                t0 = time.perf_counter()
                raw.append(chi2_p(a, b))
                chi2_seconds.append(time.perf_counter() - t0)
                grp.append(chi2_p(group_levels(a, top[c]), group_levels(b, top[c])))
                if len(lt5) < 10:
                    lt5.append(sparsity(a, b)["cells_expected_lt5"])
                # power: SHIFT of the rows moved to the 5 largest levels ("top") or to random levels outside the top
                # K ("rare"), same window size
                k = int(SHIFT * n)
                for kind, pool in (("top", top[c][:5].to_numpy()), ("rare", rare_levels)):
                    if len(pool) == 0:
                        continue
                    s_ = a.to_numpy().copy()
                    pos = rng.choice(n, k, replace=False)
                    s_[pos] = rng.choice(pool, k)
                    s_ = pd.Series(s_)
                    pw[f"{kind}_raw"].append(chi2_p(s_, b) < 0.05)
                    pw[f"{kind}_grp"].append(chi2_p(group_levels(s_, top[c]), group_levels(b, top[c])) < 0.05)
            raw, grp = np.array(raw, float), np.array(grp, float)
            aa[c][n] = {"far_raw": round(float((raw < 0.05).mean()), 3), "far_grouped": round(float((grp < 0.05).mean()), 3),
                        "far_raw_ci": S.wilson(int((raw < 0.05).sum()), len(raw)),
                        "pvalue_sd_raw": round(float(raw.std()), 3), "pvalue_mean_raw": round(float(raw.mean()), 3),
                        "cells_expected_lt5": round(float(np.mean(lt5)), 3),
                        "power": {k_: round(float(np.mean(v_)), 3) for k_, v_ in pw.items() if v_}}

    # 3. calibrated stream over the days
    def analyzer(perf=True):
        a = Analyzer(name="lab", model_id=S.MODEL_ID, model_version=S.MODEL_VERSION)
        a.add_drift_metrics(metrics_list=PARAMS["numeric_metrics"], features_list=NUMERIC)
        a.add_drift_metrics(metrics_list=PARAMS["categorical_metrics"], features_list=CATEG)
        if perf:
            a.add_performance_metrics(metrics_list=["accuracy"])
        return a

    cols = CATEG + NUMERIC + ["y_true", "y_pred", "pred_timestamp"]
    t0 = time.perf_counter()
    cal, warns = S.calibrate(analyzer(False), ref[cols], day_size, n_draws=40 if S.DRY else 200,
                             allow_short=S.DRY)
    cal_seconds = time.perf_counter() - t0
    accs = [float((g.y_pred == g.y_true).mean()) for _, g in ref.groupby("day")]
    opts = cal.options()
    opts["accuracy"] = {"threshold": float(np.mean(accs) - 2 * np.std(accs, ddof=1))}
    stream = P.WindowStream(analyzer(), ref[cols], cal, opts)
    errors = []
    t0 = time.perf_counter()
    for d in cur_days:
        g = df[df.day == d]
        try:
            stream.step(g[cols], label=int(d), dow=int(g.DayOfWeek.iloc[0]))
        except Exception as exc:  # noqa: BLE001 - recorded
            errors.append(f"day {d}: {type(exc).__name__}: {exc}")
    stream_seconds = time.perf_counter() - t0
    rows = stream.rows
    # raw (uncalibrated, nominal p < 0.05) chi2 verdicts on the same day windows: share flagged
    nominal = {c: [] for c in CATEG}
    pvals = {c: [] for c in CATEG}
    for d in cur_days:
        g = df[df.day == d]
        for c in CATEG:
            pvals[c].append(chi2_p(g[c], ref[c]))
            nominal[c].append(pvals[c][-1] < 0.05)
    # resolution of the calibrated chi2 null: day-to-day flight schedules repeat, so A/A p-values sit at 1.0
    chi2_null = {f: {"share_exactly_1": round(float((n.values == 1.0).mean()), 3), "distinct_values": int(len(np.unique(n.values))),
                     "alpha": float(n.threshold)} for (m, f), n in cal.nulls.items() if m == "chi2"}
    near_one_flagged = [{"day": r["label"], "chi2_p": pvals["AirportFrom"][i]} for i, r in enumerate(rows)
                        if "chi2|AirportFrom" in r["bh_pairs"] and pvals["AirportFrom"][i] > 0.9999]
    summ = cal.summary()
    thr = summ[summ.feature_name.isin(CATEG)][["metric_name", "feature_name", "threshold", "status", "reliable"]]

    metrics = {
        "rows": len(df), "days": n_days, "reference_rows": len(ref), "day_size_median": day_size, "load_seconds": round(t_load, 1),
        "sparsity_by_feature": sp_summary, "sparsity_first_days": {c: v[:3] for c, v in sp.items()},
        "aa_nominal_chi2": aa,
        "chi2_seconds_mean": round(float(np.mean(chi2_seconds)), 4) if chi2_seconds else None,
        "calibration_seconds": round(cal_seconds, 1), "calibration_draws": cal.n_draws_used,
        "calibration_warnings": warns[:5], "calibrated_thresholds_categorical": thr.to_dict("records"),
        "stream_windows": len(rows), "stream_seconds": round(stream_seconds, 1),
        "seconds_per_window_mean": round(float(np.mean(stream.seconds)), 2) if stream.seconds else None,
        "seconds_per_window_max": round(float(np.max(stream.seconds)), 2) if stream.seconds else None,
        "rows_per_second": round(sum(r["n"] for r in rows) / max(stream_seconds, 1e-9)),
        "confirmed_share": {f: round(P.share(rows, lambda r, f=f: f in r["confirmed"]), 3) for f in CATEG + NUMERIC} if rows else {},
        "bh_share": {f: round(P.share(rows, lambda r, f=f: f in r["bh_features"]), 3) for f in CATEG + NUMERIC} if rows else {},
        "nominal_chi2_flag_share_day_vs_ref": {c: round(float(np.mean(v)), 3) for c, v in nominal.items()},
        "accuracy_by_day": [None if r.get("perf_accuracy") is None else round(r["perf_accuracy"], 3) for r in rows],
        "perf_confirmed_share": round(P.share(rows, lambda r: r.get("perf_confirmed")), 3) if rows else None,
        "chi2_null_resolution": chi2_null, "chi2_p_by_day_AirportFrom": [float(x) for x in pvals["AirportFrom"]],
        "bh_flagged_days_with_chi2_p_above_0_9999": near_one_flagged,
        "no_verdict_rows": sum(r["no_verdict"] for r in rows), "errors": errors[:5],
        "crash_free": not errors,
    }

    # ------------------------------------------------------------------ limits
    ap = aa.get("AirportFrom", {})
    bad = {n: v for n, v in ap.items() if v["far_raw"] > 0.08 and v["far_grouped"] < v["far_raw"]}
    weak = {n: v for n, v in ap.items() if v["cells_expected_lt5"] >= 0.2 and v.get("power")
            and v["power"].get("top_grp", 0) - v["power"].get("top_raw", 0) >= 0.2}
    if weak:
        ctx.limit(
            f"chi2 on AirportFrom (293 levels) loses power when many contingency cells are sparse: a {SHIFT:.0%} shift "
            f"of rows onto the 5 largest airports is detected far less often on raw levels than after grouping to the "
            f"top {TOP_K} + 'other' (degrees of freedom spent on empty levels)",
            f"iid A/A + injected shift, {n_aa} draws, by window size (far raw / grouped, power raw / grouped, share of "
            f"cells with expected < 5): "
            f"{ {n: (v['far_raw'], v['far_grouped'], v['power'], v['cells_expected_lt5']) for n, v in ap.items()} }",
            "Add level grouping to chi2_test / psi (max_levels or min_expected: merge levels with expected count < 5 "
            "into 'other', computed on the reference) and report the share of low-expected cells on the result.",
            "major")
    if bad:
        ctx.limit(
            f"Nominal chi2 on AirportFrom (293 levels) gives a false-alarm rate above 5 % on iid A/A draws of the same "
            f"reference when many cells have an expected count < 5; grouping to the top {TOP_K} levels + 'other' "
            f"brings it back",
            f"A/A (iid, {n_aa} draws) by window size: { {n: (v['far_raw'], v['far_grouped'], v['cells_expected_lt5']) for n, v in ap.items()} } "
            f"(far raw, far grouped, share of cells with expected < 5)",
            "In chi2_test: group levels with an expected count < 5 into 'other' (or use a Monte-Carlo / G-test p-value) "
            "and return the share of low-expected cells on the result; expose max_levels / min_expected options.",
            "major")
    cn = chi2_null.get("AirportFrom", {})
    if cn.get("share_exactly_1", 0) >= 0.5 and near_one_flagged:
        ctx.limit(
            f"The calibrated chi2 on AirportFrom loses its resolution at p = 1: {cn['share_exactly_1']:.0%} of the A/A "
            f"p-values between reference days are exactly 1.0 in float (flight schedules repeat, the counts are "
            f"under-dispersed), the null holds {cn['distinct_values']} distinct values, the calibrated alpha is "
            f"{cn['alpha']:.13f}, and {len(near_one_flagged)} day(s) with a chi2 p-value above 0.9999 are flagged as "
            f"drift after BH: the verdict rests on p-value differences of 1e-6 to 1e-12 and ties hide the ranking",
            f"null {cn}; flagged days {near_one_flagged[:8]}; chi2 p by day {[round(x, 12) for x in pvals['AirportFrom']][:20]}",
            "Calibrate chi2 (and ks) on the test statistic, not on the p-value, so the null keeps its resolution at both "
            "ends; flag a null as 'saturated' when more than 5 % of its draws are tied at a bound.",
            "major")
    sparse_cells = sp_summary["AirportFrom"]["cells_expected_lt5_mean"]
    if sparse_cells >= 0.2:
        ctx.limit(
            f"On day windows, {sparse_cells:.0%} of the chi2 contingency cells of AirportFrom have an expected count < 5 "
            f"(Cochran rule violated), and the result carries no warning or reliability flag",
            f"sparsity per feature {sp_summary}; first days {sp['AirportFrom'][:3]}",
            "Compute the low-expected share in chi2_test and set reliable=False / a status 'sparse_levels' on the "
            "MetricResults above 20 %; document a level-grouping preprocessing for high-cardinality features.",
            "minor")
    nom = metrics["nominal_chi2_flag_share_day_vs_ref"]
    if nom["AirportFrom"] >= 0.5 and metrics["confirmed_share"].get("AirportFrom", 0) < nom["AirportFrom"]:
        ctx.note(f"day vs reference: nominal chi2 flags AirportFrom in {nom['AirportFrom']:.0%} of days, calibrated + BH "
                 f"+ persistence in {metrics['confirmed_share']['AirportFrom']:.0%}: the n = 20k test detects "
                 f"operationally negligible schedule differences; calibration absorbs them")
    if metrics["seconds_per_window_max"] and metrics["seconds_per_window_max"] > 60:
        ctx.limit(f"Analysis of one 20k-row window takes up to {metrics['seconds_per_window_max']} s",
                  f"per-window seconds {stream.seconds[:10]}", "Profile Analyzer.run (summary statistics on all columns).",
                  "major")
    ctx.note(f"the model lumps airports beyond the 254 most frequent into one code (sklearn limit of 255 categories (model only, not "
             f"Pulsar)")
    return metrics
