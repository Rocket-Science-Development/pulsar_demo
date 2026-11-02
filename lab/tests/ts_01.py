"""TS-01 - Seasonal electricity load: A/A calibration across seasons (UCI ElectricityLoadDiagrams20112014, CC BY 4.0)

370 clients, 15-min steps, 2011-2014. Features = load of a fixed panel of clients present over the four years.
Window = one week (672 rows). Drift metrics: ks_2samp, wasserstein, psi, each calibrated by AACalibrator on the
reference (contiguous A/A draws), empirical p-values, Benjamini-Hochberg over all (metric, feature) of the window,
then DriftPersistence(2).

Measured:
- same-season windows (winter vs winter reference): false alarm rate, nominal / calibrated / calibrated+BH / confirmed;
- opposite-season windows (summer vs winter reference): confirmation delay in windows;
- a static winter reference streamed over a full year: share of summer weeks alerted;
- seasonal references (same month of the previous year): false alarm rate, to quantify the proposal;
- clients joining late (zero load in the reference): onset detection delay.
"""

import zipfile

import numpy as np
import pandas as pd

from lab.tests import _streams as S

CRITERIA = {"same_season_far_bh_monthref": ("<=", 0.05), "opposite_confirmed_within_2_share": (">=", 1.0)}
PARAMS = {"window_rows": 672, "n_clients": 24, "metrics": ["ks_2samp", "wasserstein", "psi"], "n_draws": 200,
          "persistence": 2, "alpha_fdr": 0.05}
W = 672


def load() -> pd.DataFrame:
    cache = S.CACHE / "ld2011_2014.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    with zipfile.ZipFile(S.DATASETS / "uci_electricity_321" / "electricityloaddiagrams20112014.zip") as z:
        with z.open("LD2011_2014.txt") as f:
            df = pd.read_csv(f, sep=";", decimal=",", index_col=0, parse_dates=True, low_memory=False)
    df = df.astype("float32")
    df.to_pickle(cache)
    return df


def month(df, ym: str) -> pd.DataFrame:
    return df.loc[ym]


def weeks(df_month: pd.DataFrame, n: int = 4):
    return [df_month.iloc[i * W:(i + 1) * W] for i in range(n) if (i + 1) * W <= len(df_month)]


def stream(analyzer, cal, reference, windows):
    """Run windows in order with a fresh persistence; returns list of dicts per window"""
    pers = S.DriftPersistence(windows=PARAMS["persistence"])
    out = []
    for w in windows:
        res = S.analyze(analyzer, w, reference, cal=cal)
        nominal = S.analyze(analyzer, w, reference, options={})
        drift = res[res.metric_type == "drift"]
        verdicts = pers.update_to_pandas(S.persistence_frame(drift, "drift_fdr"))
        out.append({
            "nominal_ks_any": S.window_flag(nominal[nominal.metric_name == "ks_2samp"]),
            "nominal_ks_tests": int(nominal[nominal.metric_name == "ks_2samp"].drift_status.fillna(False).sum()),
            "n_tests": int(len(drift)),
            "calibrated_tests": int(drift.drift_status.fillna(False).astype(bool).sum()),
            "bh_any": S.window_flag(drift, "drift_fdr"),
            "bh_tests": int(drift.drift_fdr.fillna(False).astype(bool).sum()),
            "confirmed": S.any_confirmed(verdicts),
            "errors": int(drift.error.notna().sum()) if "error" in drift else 0,
        })
    return out


def summarise(rows):
    n = len(rows)
    tests = sum(r["n_tests"] for r in rows)
    return {
        "windows": n,
        "far_nominal_ks_window": sum(r["nominal_ks_any"] for r in rows) / max(n, 1),
        "far_nominal_ks_test": sum(r["nominal_ks_tests"] for r in rows) / max(tests / 3, 1),
        "far_calibrated_test": sum(r["calibrated_tests"] for r in rows) / max(tests, 1),
        "far_bh_window": sum(r["bh_any"] for r in rows) / max(n, 1),
        "far_confirmed_window": sum(r["confirmed"] for r in rows) / max(n, 1),
        "k_bh": sum(r["bh_any"] for r in rows), "k_confirmed": sum(r["confirmed"] for r in rows),
        "errors": sum(r["errors"] for r in rows),
    }


def run(ctx):
    df = load()
    dry = S.DRY
    # panel of clients with load in every month of 2011-2014 (no all-zero month)
    monthly = (df > 0).groupby(df.index.to_period("M")).any()
    stable = [c for c in df.columns if monthly[c].all()]
    n_clients = 6 if dry else PARAMS["n_clients"]
    clients = [stable[i] for i in np.linspace(0, len(stable) - 1, n_clients).astype(int)]
    ctx.note(f"{len(stable)} of {df.shape[1]} clients have load in every month; panel = {clients}")
    data = df[clients]
    analyzer = S.make_analyzer(clients, PARAMS["metrics"])
    n_draws = 40 if dry else PARAMS["n_draws"]
    metrics = {"n_clients_total": df.shape[1], "n_clients_stable": len(stable), "panel_size": len(clients)}

    winter_targets = ["2012-02", "2012-12", "2013-01", "2013-02", "2013-12", "2014-01", "2014-02"]
    summer_targets = ["2012-06", "2012-07", "2012-08", "2013-06", "2013-07", "2013-08"]
    if dry:
        winter_targets, summer_targets = winter_targets[:2], summer_targets[:1]

    refs = {
        "monthref": month(data, "2012-01"),
        "quarterref": pd.concat([month(data, "2011-12"), month(data, "2012-01"), month(data, "2012-02")]),
    }
    detail = {}
    for name, ref in refs.items():
        cal, warns = S.calibrate(analyzer, ref, W, n_draws=n_draws)
        metrics[f"{name}_rows"] = len(ref)
        metrics[f"{name}_calibration_warning"] = bool(warns)
        if warns:
            ctx.note(f"{name}: AACalibrator warning: {warns[0][:200]}")
        same = []
        for ym in winter_targets:
            if name == "quarterref" and ym == "2012-02":
                continue
            same += stream(analyzer, cal, ref, weeks(month(data, ym)))
        s = summarise(same)
        for k, v in s.items():
            metrics[f"same_season_{k}_{name}"] = v
        metrics[f"same_season_far_bh_{name}"] = s["far_bh_window"]
        delays = []
        opp = []
        for ym in summer_targets:
            rows = stream(analyzer, cal, ref, weeks(month(data, ym)))
            opp += rows
            delays.append(S.first_index([r["confirmed"] for r in rows]))
        metrics[f"opposite_confirm_delay_windows_{name}"] = delays
        metrics[f"opposite_far_bh_{name}"] = summarise(opp)["far_bh_window"]
        if name == "monthref":
            metrics["opposite_confirmed_within_2_share"] = float(np.mean([d is not None and d <= 1 for d in delays]))
        detail[name] = {"same": same, "opposite": opp}

    # static reference over a full year: weekly windows Mar 2012 - Feb 2013 against January 2012
    ref = refs["monthref"]
    cal, _ = S.calibrate(analyzer, ref, W, n_draws=n_draws)
    year = data.loc["2012-03-01":"2013-02-28"]
    yweeks = [year.iloc[i * W:(i + 1) * W] for i in range(len(year) // W)]
    if dry:
        yweeks = yweeks[::8]
    yrows = stream(analyzer, cal, ref, yweeks)
    starts = [w.index[0] for w in yweeks]
    summer = [r for r, t in zip(yrows, starts) if t.month in (6, 7, 8)]
    winter = [r for r, t in zip(yrows, starts) if t.month in (12, 1, 2)]
    metrics["static_ref_summer_weeks_confirmed_share"] = sum(r["confirmed"] for r in summer) / max(len(summer), 1)
    metrics["static_ref_winter_weeks_confirmed_share"] = sum(r["confirmed"] for r in winter) / max(len(winter), 1)
    metrics["static_ref_all_weeks_confirmed_share"] = sum(r["confirmed"] for r in yrows) / max(len(yrows), 1)
    metrics["static_ref_weeks"] = len(yrows)
    metrics["static_ref_mean_bh_tests_per_week_summer"] = float(np.mean([r["bh_tests"] for r in summer])) if summer else None

    # seasonal reference: same month of the previous year
    pairs = [("2012-01", "2013-01"), ("2012-02", "2013-02"), ("2012-07", "2013-07"), ("2012-08", "2013-08"),
             ("2013-01", "2014-01"), ("2013-07", "2014-07")]
    if dry:
        pairs = pairs[:1]
    seas = []
    for r_ym, c_ym in pairs:
        r = month(data, r_ym)
        c, _ = S.calibrate(analyzer, r, W, n_draws=n_draws)
        seas += stream(analyzer, c, r, weeks(month(data, c_ym)))
    s = summarise(seas)
    metrics["seasonal_ref_far_bh_window"] = s["far_bh_window"]
    metrics["seasonal_ref_far_confirmed_window"] = s["far_confirmed_window"]
    metrics["seasonal_ref_far_calibrated_test"] = s["far_calibrated_test"]
    metrics["seasonal_ref_windows"] = s["windows"]

    # clients joining late: zero load in 2011, first load during 2012-2013
    first = (df > 0).idxmax()
    late = [c for c in df.columns if (df[c] > 0).any() and pd.Timestamp("2012-03-01") <= first[c] <= pd.Timestamp("2013-10-01")]
    late = late[: (2 if dry else 6)]
    onset = []
    for c in late:
        t0 = first[c]
        ref = df.loc[t0 - pd.Timedelta(days=60):t0 - pd.Timedelta(minutes=15), [c]]
        a1 = S.make_analyzer([c], PARAMS["metrics"])
        try:
            cal1, _ = S.calibrate(a1, ref, W, n_draws=n_draws)
            seq = df.loc[t0 - pd.Timedelta(days=14):t0 + pd.Timedelta(days=56), [c]]
            ws = [seq.iloc[i * W:(i + 1) * W] for i in range(len(seq) // W)]
            rows = stream(a1, cal1, ref, ws)
            onset_idx = 2  # two pre-onset weeks
            onset.append({"client": c, "first_load": str(t0), "pre_onset_alerts": sum(r["bh_any"] for r in rows[:onset_idx]),
                          "delay_raw": S.first_index([r["bh_any"] for r in rows], onset_idx),
                          "delay_confirmed": S.first_index([r["confirmed"] for r in rows], onset_idx),
                          "null_thresholds": {k: v for k, v in cal1.options().items()}})
        except Exception as exc:  # noqa: BLE001 - recorded, it is a behaviour of the calibrator
            onset.append({"client": c, "first_load": str(t0), "error": f"{type(exc).__name__}: {exc}"[:300]})
    metrics["late_joiners"] = onset
    metrics["late_joiners_found"] = len(late)

    # pipeline control: i.i.d. data through the same pipeline must give ~5 % per test and <= 5 % per window after BH
    rng = np.random.default_rng(1)
    n_ctrl = 10 if dry else 40
    iid = pd.DataFrame(rng.normal(size=(W * (10 + n_ctrl), len(clients))), columns=clients)
    cref = iid.iloc[:10 * W].reset_index(drop=True)
    ccal, _ = S.calibrate(analyzer, cref, W, n_draws=n_draws)
    crow = stream(analyzer, ccal, cref, [iid.iloc[(10 + i) * W:(11 + i) * W].reset_index(drop=True) for i in range(n_ctrl)])
    cs = summarise(crow)
    metrics["control_iid_far_calibrated_test"] = cs["far_calibrated_test"]
    metrics["control_iid_far_bh_window"] = cs["far_bh_window"]
    metrics["control_iid_far_confirmed_window"] = cs["far_confirmed_window"]
    metrics["control_iid_windows"] = cs["windows"]
    ctx.note(f"pipeline control on i.i.d. data: per-test {cs['far_calibrated_test']:.1%}, window BH {cs['far_bh_window']:.1%}, "
             f"confirmed {cs['far_confirmed_window']:.1%} over {cs['windows']} windows")

    # ------------------------------------------------------------------ limits
    far_m = metrics["same_season_far_bh_monthref"]
    far_q = metrics["same_season_far_bh_quarterref"]
    nm = metrics["same_season_windows_monthref"]
    nq = metrics["same_season_windows_quarterref"]
    if far_m > 0.05 or far_q > 0.05:
        ctx.limit(
            f"Same-season false alarms stay above 5 % after A/A calibration + BH: {far_m:.0%} of winter weeks alerted "
            f"with a one-month winter reference, {far_q:.0%} with a three-month winter reference",
            f"monthref: {S.fmt_rate(metrics['same_season_k_bh_monthref'], nm)}; quarterref: "
            f"{S.fmt_rate(metrics['same_season_k_bh_quarterref'], nq)}; per-test calibrated rate "
            f"{metrics['same_season_far_calibrated_test_monthref']:.1%} (target 5 %); nominal KS any-feature rate "
            f"{metrics['same_season_far_nominal_ks_window_monthref']:.0%}; confirmed (persistence 2) "
            f"{metrics['same_season_far_confirmed_window_monthref']:.0%}; {len(clients)} clients x 3 metrics, weekly windows; "
            f"same pipeline on i.i.d. control data: per-test {metrics['control_iid_far_calibrated_test']:.1%}, window "
            f"{metrics['control_iid_far_bh_window']:.0%} ({metrics['control_iid_windows']} windows)",
            "A/A draws inside one reference period only capture within-period noise, not year-to-year level changes of the "
            "same season (load growth, client churn). Add a between-period calibration mode to AACalibrator "
            "(fit(references=[Jan 2012, Jan 2013, ...]) drawing A and B from different same-season periods), and document "
            "that a calibration is valid only for the season it was fitted on.",
            "major")
    if metrics["monthref_calibration_warning"]:
        ctx.limit(
            "A one-month reference is too short for AACalibrator with weekly windows: the calibrator warns that the "
            "thresholds are unreliable, yet the registry method (reference = a month) is the natural setting",
            f"January 2012 = {metrics['monthref_rows']} rows = {metrics['monthref_rows'] / W:.1f} windows of {W} "
            f"(< 10 required); thresholds are still returned and used",
            "Make AACalibrator refuse (or switch to contiguous=False with block bootstrap) below 10 windows unless "
            "allow_short=True; expose calibrator.reliable in options() so the Analyzer can tag verdicts as provisional.",
            "minor")
    if metrics["static_ref_summer_weeks_confirmed_share"] >= 0.5:
        ctx.limit(
            f"A single static winter reference alerts {metrics['static_ref_summer_weeks_confirmed_share']:.0%} of summer "
            f"weeks (confirmed drift) and {metrics['static_ref_all_weeks_confirmed_share']:.0%} of all weeks of the year",
            f"{metrics['static_ref_weeks']} weekly windows Mar 2012-Feb 2013 vs January 2012; mean "
            f"{metrics['static_ref_mean_bh_tests_per_week_summer']:.1f} BH rejections per summer week out of "
            f"{3 * len(clients)} tests; same-month-previous-year references give "
            f"{metrics['seasonal_ref_far_bh_window']:.0%} window alarms over {metrics['seasonal_ref_windows']} windows",
            "Add seasonal references to the analysis scheduler: reference = same calendar period of the previous cycle "
            "(or a reference per season / per day-of-week profile), selected automatically from pred_timestamp; "
            "optionally de-seasonalise features (ratio to the seasonal profile) before the tests.",
            "major")
    errs = [o for o in onset if "error" in o]
    if errs:
        ctx.limit(
            f"Calibration fails or misbehaves for a feature that is constant (zero) in the reference: {len(errs)} of "
            f"{len(onset)} late-joining clients",
            "; ".join(f"{o['client']}: {o['error']}" for o in errs)[:600],
            "Detect constant features in AACalibrator.fit and Analyzer.run: report a dedicated 'new_activity' / "
            "'constant_reference' status instead of a statistical test on a degenerate distribution.",
            "major")
    degenerate = [o for o in onset if "error" not in o and any(
        v.get("alpha") == 1.0 or v.get("threshold") == 0.0 for v in o["null_thresholds"].values())]
    if degenerate:
        ctx.limit(
            f"A feature constant in the reference gets degenerate calibrated thresholds ({len(degenerate)} of "
            f"{len(onset)} late joiners): ks alpha = 1.0 and wasserstein / psi threshold = 0.0, so a single non-zero "
            f"reading alerts; the calibration gives no warning",
            "; ".join(f"{o['client']}: {o['null_thresholds']}" for o in degenerate)[:600],
            "In AACalibrator.fit, flag null distributions with zero variance (all draws equal) and return no threshold "
            "(or a dedicated 'constant_reference' status) with a warning; in the Analyzer, report such features as "
            "'inactive in reference' rather than as a statistical drift.",
            "minor")
    ok = [o for o in onset if "error" not in o]
    slow = [o for o in ok if o["delay_confirmed"] is None or o["delay_confirmed"] > 1]
    if ok and slow:
        ctx.limit(
            f"Natural onsets (client starts consuming) are not confirmed within 2 windows for {len(slow)} of {len(ok)} "
            f"late joiners",
            "; ".join(f"{o['client']} first load {o['first_load'][:10]}: raw delay {o['delay_raw']}, confirmed delay "
                      f"{o['delay_confirmed']}" for o in slow)[:600],
            "Treat a zero-variance reference as a special case (any non-zero activity = new regime) and consider "
            "persistence 1 for onsets of features that were inactive.",
            "minor")
    dq = metrics["opposite_confirm_delay_windows_quarterref"]
    missed = sum(d is None or d > 1 for d in dq)
    if missed:
        ctx.limit(
            f"With a three-month winter reference, {missed} of {len(dq)} summer months are not confirmed within 2 "
            f"weekly windows (opposite season), while the one-month reference confirms "
            f"{metrics['opposite_confirmed_within_2_share']:.0%}",
            f"confirmation delays (windows) quarterref {dq} vs monthref {metrics['opposite_confirm_delay_windows_monthref']}; "
            f"summer window alarm rate quarterref {metrics['opposite_far_bh_quarterref']:.0%}",
            "A longer reference widens the A/A null (it contains the within-winter level changes), which costs power: "
            "report the calibrated minimum detectable shift per feature next to the thresholds, and let users pick the "
            "reference length on that trade-off.",
            "major")
    ctx.params["detail_counts"] = {k: {kk: summarise(vv) for kk, vv in v.items()} for k, v in detail.items()}
    return metrics
