"""TC-05 - Ground-truth drift points: detection delay per drift type

Registry dataset: INSECTS (Souza et al. 2020), licence unverified, internal use only. The provisioning could not fetch
it (Google Sites/Drive links, no stable URL) and it is not fetched here: the licence is unverified, so a manual,
licence-checked download is required. When CSV files are present in <data>/datasets/insects_souza2020/ the test
reports it but still runs the substitute below; the INSECTS run is left to a follow-up once the licence is verified.

Substitute with exact ground truth (same structure as INSECTS: abrupt, gradual, incremental, recurring drifts):
10 Gaussian features, windows of 500 rows, static reference = 10 pre-drift windows, drift at window 20 on 3 features
(mean shift of DELTA sd). Pipeline: ks_2samp, wasserstein, psi calibrated by AACalibrator, BH over the window,
DriftPersistence(2). Measured per type: false alarms before the drift, delay to raw and to confirmed alert, and for
recurring drift the re-alerts with a static reference vs a reference updated after each confirmed drift.

Two controls make the delays interpretable:
- a baseline comparison detector on the same windows, outside Pulsar.ml: two-sample KS per feature (scipy), Bonferroni
  over the features, the same persistence of 2 windows;
- a run with m < 30 tests per window (SMALL_D features x 3 metrics), so that the delays are not confounded with the
  empirical p-value floor x BH limit (report C1, which bites from m = 30).
The gradual / incremental delay is recorded as a limit only when Pulsar.ml is slower than the baseline at m < 30. The
return of an old concept after a reference update is reported as a feature request (concept library), not a limit:
the update policy is written by the test and the alert is correct against the reference in force.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from lab.tests import _streams as S

D, W, N_REF_WIN, ONSET, N_WIN = 10, 500, 10, 20, 60
DRIFTING = ["x0", "x1", "x2"]
DELTA = 0.25
CRITERIA = {"abrupt_confirmed_delay": ("<=", 1), "recurring_realerts_static": (">=", 2), "pre_drift_far_window": ("<=", 0.05)}
PARAMS = {"features": D, "window_rows": W, "reference_windows": N_REF_WIN, "onset_window": ONSET, "delta_sd": DELTA,
          "gradual_windows": 10, "incremental_windows": 15, "recurring": "on 20-29, off 30-39, on 40-49, off 50-59",
          "replicates": 10, "dataset": "synthetic substitute (INSECTS unavailable, licence unverified)"}
COLS = [f"x{i}" for i in range(D)]
SMALL_D = 6  # 6 features x 3 metrics = 18 tests per window, below the m = 30 floor of report C1
SMALL_COLS = COLS[:SMALL_D]
PARAMS.update({"small_features": SMALL_D, "baseline": "scipy KS per feature, Bonferroni over features, persistence 2"})


def shift_profile(kind: str, w: int) -> float:
    """share of the new concept (gradual) or fraction of DELTA (others) in window w"""
    if w < ONSET:
        return 0.0
    if kind == "abrupt":
        return 1.0
    if kind == "gradual":
        return min(1.0, (w - ONSET + 1) / PARAMS["gradual_windows"])
    if kind == "incremental":
        return min(1.0, (w - ONSET + 1) / PARAMS["incremental_windows"])
    if kind == "recurring":
        return 1.0 if (w - ONSET) // 10 % 2 == 0 else 0.0
    raise ValueError(kind)


def make_window(kind, w, rng):
    x = rng.normal(size=(W, D))
    s = shift_profile(kind, w)
    if kind == "gradual":
        new = rng.random(W) < s  # rows from the new concept
        x[new, :len(DRIFTING)] += DELTA
    else:
        x[:, :len(DRIFTING)] += DELTA * s
    return pd.DataFrame(x, columns=COLS)


def baseline_flag(cur: pd.DataFrame, ref: pd.DataFrame, alpha: float = 0.05) -> bool:
    """Comparison detector outside Pulsar.ml: KS per feature, Bonferroni over the features"""
    cols = list(cur.columns)
    return any(ks_2samp(cur[c], ref[c]).pvalue < alpha / len(cols) for c in cols)


def run_stream(kind, rng, analyzer, cal, reference, policy="static", cols=None):
    pers = S.DriftPersistence(2)
    ref, c = reference, cal
    recent = []
    out = []
    update_at = None
    base_run = 0
    for w in range(N_REF_WIN, N_WIN):
        cur = make_window(kind, w, rng)
        if cols is not None:
            cur = cur[cols]
        b = baseline_flag(cur, ref)
        base_run = base_run + 1 if b else 0
        res = S.analyze(analyzer, cur, ref, cal=c)
        drift = res[res.metric_type == "drift"]
        v = pers.update_to_pandas(S.persistence_frame(drift, "drift_fdr"))
        conf = S.any_confirmed(v)
        changed_on = bool(((v.changed) & (v.confirmed_drift)).any()) if not v.empty else False
        out.append({"w": w, "raw": S.window_flag(drift, "drift_fdr"), "confirmed": conf, "new_confirmation": changed_on,
                    "drifting_hit": bool(drift[drift.feature_name.isin(DRIFTING)].drift_fdr.fillna(False).astype(bool).any()),
                    "stable_hit": bool(drift[~drift.feature_name.isin(DRIFTING)].drift_fdr.fillna(False).astype(bool).any()),
                    "R": int(drift.drift_fdr.fillna(False).astype(bool).sum()),
                    "V": int(drift[~drift.feature_name.isin(DRIFTING)].drift_fdr.fillna(False).astype(bool).sum()),
                    "baseline_raw": b, "baseline_confirmed": base_run >= 2})
        recent = (recent + [cur])[-N_REF_WIN:]
        if policy == "update" and changed_on:
            update_at = w + 2  # accept the new concept once 4 windows of it are available
        if policy == "update" and update_at == w:
            ref = pd.concat(recent[-4:]).reset_index(drop=True)
            c, _ = S.calibrate(analyzer, ref, W, n_draws=100, allow_short=True)
            pers.reset()
            update_at = None
    return out


def episodes(flags):
    """number of off->on transitions"""
    n, prev = 0, False
    for f in flags:
        if f and not prev:
            n += 1
        prev = f
    return n


def run(ctx):
    insects = S.DATASETS / "insects_souza2020"
    metrics = {"insects_available": insects.exists() and any(insects.glob("*.csv")),
               "dataset_used": PARAMS["dataset"]}
    ctx.note("INSECTS not available (provisioning: no stable URL, licence unverified); synthetic ground-truth substitute used")
    reps = 3 if S.DRY else PARAMS["replicates"]
    rng = np.random.default_rng(0)
    analyzer = S.make_analyzer(COLS, ["ks_2samp", "wasserstein", "psi"])
    summary = {}
    ks_alphas = []
    small_an = S.make_analyzer(SMALL_COLS, ["ks_2samp", "wasserstein", "psi"])
    for kind in ("abrupt", "gradual", "incremental", "recurring"):
        delays_raw, delays_conf, pre_fa, pre_n, realerts_static, realerts_update, clear_delays = [], [], 0, 0, [], [], []
        stable_hits, fdr_R, fdr_V, fdr_win = 0, 0, 0, 0
        base_delays, small_delays, small_base_delays, base_pre_fa = [], [], [], 0
        for r in range(reps):
            reference = pd.concat([make_window(kind, w, rng) for w in range(N_REF_WIN)]).reset_index(drop=True)
            cal, _ = S.calibrate(analyzer, reference, W, n_draws=200, seed=r)
            ks_alphas.extend(v["alpha"] for k, v in cal.options().items() if k.startswith("ks_2samp|"))
            rows = run_stream(kind, rng, analyzer, cal, reference)
            pre = [x for x in rows if x["w"] < ONSET]
            pre_fa += sum(x["raw"] for x in pre)
            pre_n += len(pre)
            post = [x for x in rows if x["w"] >= ONSET]
            delays_raw.append(S.first_index([x["drifting_hit"] for x in post]))
            delays_conf.append(S.first_index([x["confirmed"] for x in post]))
            base_delays.append(S.first_index([x["baseline_confirmed"] for x in post]))
            base_pre_fa += sum(x["baseline_raw"] for x in pre)
            # same stream type with m = 18 tests per window (below the C1 floor), Pulsar.ml and the baseline
            sref = reference[SMALL_COLS]
            scal, _ = S.calibrate(small_an, sref, W, n_draws=200, seed=r)
            srows = run_stream(kind, rng, small_an, scal, sref, cols=SMALL_COLS)
            spost = [x for x in srows if x["w"] >= ONSET]
            small_delays.append(S.first_index([x["confirmed"] for x in spost]))
            small_base_delays.append(S.first_index([x["baseline_confirmed"] for x in spost]))
            stable_hits += sum(x["stable_hit"] for x in post)
            fdr_R += sum(x["R"] for x in post if shift_profile(kind, x["w"]) > 0)
            fdr_V += sum(x["V"] for x in post if shift_profile(kind, x["w"]) > 0)
            fdr_win += sum(1 for x in post if shift_profile(kind, x["w"]) > 0)
            if kind == "recurring":
                realerts_static.append(episodes([x["confirmed"] for x in post]))
                off = [x for x in rows if 30 <= x["w"] < 40]
                clear_delays.append(S.first_index([not x["confirmed"] for x in off]))
                rows_u = run_stream(kind, rng, analyzer, cal, reference, policy="update")
                realerts_update.append({"episodes": episodes([x["confirmed"] for x in rows_u if x["w"] >= ONSET]),
                                        "confirmed_in_off_phase": sum(x["confirmed"] for x in rows_u if 30 <= x["w"] < 40 or x["w"] >= 50),
                                        "confirmed_in_on_phase": sum(x["confirmed"] for x in rows_u if 20 <= x["w"] < 30 or 40 <= x["w"] < 50)})
        d = [x for x in delays_conf if x is not None]
        summary[kind] = {
            "delay_raw_windows": delays_raw, "delay_confirmed_windows": delays_conf,
            "delay_confirmed_median": float(np.median(d)) if d else None, "missed": sum(x is None for x in delays_conf),
            "pre_drift_far_window": pre_fa / max(pre_n, 1), "pre_drift_windows": pre_n,
            "stable_feature_alert_windows_post": stable_hits,
            "discoveries_in_drift_windows": fdr_R, "false_discoveries_in_drift_windows": fdr_V, "drift_windows": fdr_win,
            "observed_fdr": fdr_V / fdr_R if fdr_R else None,
            "baseline_delay_confirmed_windows": base_delays, "baseline_pre_drift_far_window": base_pre_fa / max(pre_n, 1),
            "small_m_delay_confirmed_windows": small_delays, "small_m_baseline_delay_confirmed_windows": small_base_delays,
        }
        for key in ("baseline_delay_confirmed_windows", "small_m_delay_confirmed_windows",
                    "small_m_baseline_delay_confirmed_windows"):
            vals = [x for x in summary[kind][key] if x is not None]
            summary[kind][key.replace("_windows", "_median")] = float(np.median(vals)) if vals else None
        if kind == "recurring":
            summary[kind].update({"episodes_static": realerts_static, "clear_delay_windows": clear_delays,
                                  "update_policy": realerts_update})
    metrics["by_type"] = summary
    ka = np.array(ks_alphas)
    metrics["calibrated_ks_alpha_iid"] = {"n": int(ka.size), "mean": float(ka.mean()), "sd": float(ka.std()),
                                          "min": float(ka.min()), "p10": float(np.quantile(ka, 0.1)),
                                          "p90": float(np.quantile(ka, 0.9)), "max": float(ka.max())}

    # empirical p-value floor x BH: one feature with a large shift, growing number of features.
    # "fixed" keeps n_draws=200 (auto_draws=False: the pre-fix behaviour, evidence of C1); "auto" is the post-fix
    # default, which raises n_draws until 1/(n_draws+1) <= alpha/(2m).
    floors = {}
    for mode in ("fixed", "auto"):
        floor = {}
        for d in ([3, 20] if S.DRY else [3, 10, 11, 20, 40]):
            cols = [f"x{i}" for i in range(d)]
            an = S.make_analyzer(cols, ["ks_2samp", "wasserstein", "psi"])
            hits = []
            for r in range(2 if S.DRY else 5):
                ref = pd.DataFrame(rng.normal(size=(W * N_REF_WIN, d)), columns=cols)
                c, _ = S.calibrate(an, ref, W, n_draws=200, seed=r, auto_draws=(mode == "auto"))
                cur = pd.DataFrame(rng.normal(size=(W, d)), columns=cols)
                cur["x0"] += 2.0
                res = S.analyze(an, cur, ref, cal=c)
                drift = res[res.metric_type == "drift"]
                hits.append({"bh_any": S.window_flag(drift, "drift_fdr"),
                             "min_p_empirical": float(drift.p_empirical.min()),
                             "n_draws_used": getattr(c, "n_draws_used", None),
                             "x0_raw_calibrated": int(drift[drift.feature_name == "x0"].drift_status.fillna(False).astype(bool).sum())})
            floor[d] = {"tests": 3 * d, "detect_bh": float(np.mean([h["bh_any"] for h in hits])),
                        "min_p_empirical": min(h["min_p_empirical"] for h in hits),
                        "n_draws_used": max((h["n_draws_used"] or 0) for h in hits) or None,
                        "bh_first_rank_bound": 0.05 / (3 * d), "x0_tests_flagged_before_bh_mean": float(np.mean([h["x0_raw_calibrated"] for h in hits]))}
        floors[mode] = floor
    floor = floors["fixed"]
    metrics["empirical_floor_vs_bh"] = floor
    metrics["empirical_floor_vs_bh_auto_draws"] = floors["auto"]

    # AACalibrator contiguous draws on a reference barely longer than 2 windows
    import time
    from pulsar_metrics.calibration import AACalibrator
    from pulsar_metrics.metrics.drift import DriftTestMetric
    short = {}
    w = 200
    for mult in (2.0, 2.05, 2.5, 4.0):
        ref = pd.DataFrame({"x0": rng.normal(size=int(mult * w))})
        t0 = time.perf_counter()
        try:  # post-fix API: contiguous draws below 4 windows are refused, even with allow_short=True
            c = S.quiet(AACalibrator(window_size=w, n_draws=30, seed=0, allow_short=True).fit, ref,
                        [DriftTestMetric("ks_2samp", "x0")])
        except Exception as exc:  # noqa: BLE001 - the refusal is the fixed behaviour, recorded
            short[mult] = {"rows": len(ref), "refused": f"{type(exc).__name__}: {exc}"[:200]}
            continue
        null = c.nulls[("ks_2samp", "x0")].values
        short[mult] = {"rows": len(ref), "fit_seconds": round(time.perf_counter() - t0, 2),
                       "distinct_null_values": int(np.unique(null).size), "n_draws": 30, "refused": None,
                       "calibrated_alpha": float(c.options()["ks_2samp|x0"]["alpha"])}
    metrics["short_reference_calibration"] = short
    metrics["abrupt_confirmed_delay"] = max((x for x in summary["abrupt"]["delay_confirmed_windows"] if x is not None), default=None)
    if summary["abrupt"]["missed"]:
        metrics["abrupt_confirmed_delay"] = None
    metrics["recurring_realerts_static"] = min(summary["recurring"]["episodes_static"])
    metrics["pre_drift_far_window"] = float(np.mean([summary[k]["pre_drift_far_window"] for k in summary]))

    # ------------------------------------------------------------------ limits
    sr = metrics["short_reference_calibration"]
    if sr[2.0].get("refused"):
        ctx.note(f"AACalibrator refuses a contiguous reference of 2 windows (post-fix API): {sr[2.0]['refused']}")
    elif sr[2.0]["distinct_null_values"] <= 2 or sr[2.0]["fit_seconds"] > 10 * sr[4.0].get("fit_seconds", 0):
        ctx.limit(
            "AACalibrator with contiguous=True on a reference of about 2 windows returns a degenerate null "
            "distribution (the only disjoint pair is drawn every time) after a slow rejection loop, with only a warning",
            f"window 200, n_draws 30: {sr}",
            "Enumerate the admissible disjoint pairs when they are few (or require n >= 4 windows and raise below), "
            "and warn when the number of distinct A/A draws is far below n_draws.",
            "minor")
    blind = {d: v for d, v in floor.items() if v["detect_bh"] == 0}
    blind_auto = {d: v for d, v in floors["auto"].items() if v["detect_bh"] == 0}
    if blind and not blind_auto:
        ctx.note(f"C1 reproduced with auto_draws=False (n_draws fixed at 200): blind at {sorted(3 * d for d in blind)} "
                 f"tests; with the post-fix default auto_draws=True every case is detected: {floors['auto']}")
    if blind and blind_auto:
        ctx.limit(
            "A drift on a single feature, however large, can never pass Benjamini-Hochberg on AACalibrator empirical "
            f"p-values once a window holds more than about 30 (metric, feature) tests (10 features x 3 metrics): "
            f"detection 0 % for a 2-sd shift with {', '.join(str(3 * d) for d in blind)} tests"
            + (f", {floor[11]['detect_bh']:.0%} with 33 tests" if 11 in floor else ""),
            f"empirical p-value floor = 1/(n_draws+1) = {1 / 201:.5f} with the default n_draws=200; 3 tests (ks, wasserstein, "
            f"psi) of the shifted feature reach it; BH needs p_(k) <= 0.05 k / m, i.e. 3 x 0.05 / m >= 0.004975 only for "
            f"m <= 30; measured {floor}",
            "Use the parametric p-value of test metrics when it is smaller than the empirical one, or fit a tail model "
            "(e.g. generalised Pareto on the A/A distances) to get p-values below 1/(n_draws+1); at minimum, warn when "
            "n_draws + 1 < m / alpha and set n_draws >= m / alpha automatically in AACalibrator.",
            "critical")
    if metrics["calibrated_ks_alpha_iid"]["p90"] > 2 * metrics["calibrated_ks_alpha_iid"]["p10"]:
        ka = metrics["calibrated_ks_alpha_iid"]
        ctx.limit(
            f"Calibrated thresholds are unstable at the recommended minimum reference of 10 windows: on i.i.d. data the "
            f"calibrated KS alpha ranges from {ka['min']:.3f} to {ka['max']:.3f} (p10 {ka['p10']:.3f}, p90 {ka['p90']:.3f}; "
            f"nominal 0.05)",
            f"{ka['n']} (feature, replicate) calibrations, reference {N_REF_WIN} x {W} rows, contiguous A/A, n_draws 200",
            "The contiguous A/A draws overlap heavily in a 10-window reference, so the 200 draws are far from "
            "independent; report a bootstrap CI of each calibrated threshold, shrink thresholds of test metrics toward "
            "the nominal alpha when the reference is short, and raise the recommended minimum to 30+ windows.",
            "major")
    g, a, inc = summary["gradual"], summary["abrupt"], summary["incremental"]
    ctx.note("confirmation delay medians (Pulsar m=30 / baseline / Pulsar m=18 / baseline m=18): " + "; ".join(
        f"{k} {v['delay_confirmed_median']} / {v['baseline_delay_confirmed_median']} / {v['small_m_delay_confirmed_median']}"
        f" / {v['small_m_baseline_delay_confirmed_median']}" for k, v in summary.items()))
    slow = []
    for name, v in (("gradual", g), ("incremental", inc)):
        mine, ref_d = v["small_m_delay_confirmed_median"], v["small_m_baseline_delay_confirmed_median"]
        if ref_d is not None and (mine is None or mine > ref_d + 1):
            slow.append(f"{name}: Pulsar.ml {mine} vs baseline {ref_d} windows")
    if slow:
        ctx.limit(
            f"Gradual / incremental drifts are confirmed later than by a plain KS + Bonferroni baseline with the same "
            f"persistence, at m = {3 * SMALL_D} tests per window (below the C1 floor): {'; '.join(slow)}",
            f"{reps} replicates, shift {DELTA} sd on 3 features, windows of {W}; delays (Pulsar m=18 / baseline m=18): "
            f"gradual {g['small_m_delay_confirmed_windows']} / {g['small_m_baseline_delay_confirmed_windows']}; "
            f"incremental {inc['small_m_delay_confirmed_windows']} / {inc['small_m_baseline_delay_confirmed_windows']}; "
            f"abrupt {a['small_m_delay_confirmed_windows']} / {a['small_m_baseline_delay_confirmed_windows']}",
            "Add a cumulative detector on the per-window statistics (CUSUM / Page-Hinkley on the calibrated empirical "
            "p-values or on wasserstein) next to DriftPersistence, and an expanding-window comparison for slow drifts.",
            "major")
    rec = summary["recurring"]
    upd = rec["update_policy"]
    off_alerts = [u["confirmed_in_off_phase"] for u in upd]
    if any(x > 0 for x in off_alerts):
        # correct against the reference in force (the update policy is the test's): a feature request, not a limit
        ctx.note(
            f"feature request (concept library): with a reference updated after a confirmed drift, the return of the old "
            f"concept is reported as a new drift ({off_alerts} confirmed windows in the off phases over {reps} "
            f"replicates; episodes update {[u['episodes'] for u in upd]} vs static {rec['episodes_static']}). Keeping "
            f"past references and testing the window against each would report 'known concept k is back'.")
    if (rec["clear_delay_windows"] and any(x is None or x > 1 for x in rec["clear_delay_windows"])):
        ctx.limit(
            "After a drift ends, the confirmed state takes more than 2 windows to clear",
            f"clear delays (windows after the old concept returns): {rec['clear_delay_windows']}",
            "Document the hysteresis cost and expose windows_to_clear separately from windows_to_confirm.",
            "minor")
    fd = {k: (v["observed_fdr"], v["false_discoveries_in_drift_windows"], v["discoveries_in_drift_windows"]) for k, v in summary.items()}
    worst = max((v[0] or 0) for v in fd.values())
    if worst > 0.05:
        ctx.limit(
            f"Which features drifted is unreliable during a drift: the observed false discovery rate among flagged "
            f"(metric, feature) tests reaches {worst:.1%} (target 5 %), i.e. stable features are reported as drifting",
            f"(observed FDR, false, total discoveries) by type in drift windows: {fd}",
            "Empirical p-values are discrete (step 1/(n_draws+1)) and dependent across the 3 metrics of a feature: run "
            "BH on one p-value per feature (min over metrics with a Bonferroni/Simes combination) and raise n_draws.",
            "minor")
    if metrics["pre_drift_far_window"] > 0.05:
        ctx.limit(
            f"Window false alarm rate before the drift is {metrics['pre_drift_far_window']:.1%} (> 5 %) on i.i.d. data",
            f"{ {k: round(v['pre_drift_far_window'], 3) for k, v in summary.items()} }",
            "Check AACalibrator quantile and draw count; report the whole-pipeline FWER.",
            "major")
    return metrics
