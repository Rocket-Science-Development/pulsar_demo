"""FA-03: aggregate-level fairness (UCI Communities and Crime, CC BY 4.0).

Data: the unnormalised release of the same study (UCI 211 "Communities and Crime Unnormalized", CC BY 4.0, checked
on its source page 2026-11-02), because the normalised release (UCI 183) min-max scales population and shares: a
community of 10,005 people maps to population 0, so population weights computed from it are meaningless (dry run
p2: 57 % of the "population" in the top Black-share tercile, Black exposure 0.90). Rows with a missing
ViolentCrimesPerPop are dropped (221 of 2,215), which leaves the 1,994 communities of UCI 183.

Each row is a community (US communities, 1990 census + 1995 FBI UCR), not a person: race is only known as
shares of the population. A classifier flags "high violent crime" communities (top 30 % of ViolentCrimesPerPop),
with out-of-fold predictions on the dataset's 10 folds. Parts:

A. Community-level analysis: groups from race-share terciles (racepctblack, racePctWhite, racePctHisp), reference
   the lowest-share tercile; DIR, parity, equal opportunity, FPR difference. Null control with permuted terciles.
B. Passing the share itself (the real float column) as the protected attribute, what a user does when the attribute
   is not categorical: number of groups, suppression, coverage, or Pulsar's refusal (AllCellsSuppressed), which is
   then the measured behaviour.
C. What the decision means for people: exposure of Black / White / Asian residents to a "flagged" community,
   population-weighted from the aggregates (exact for a community-level decision), with a cluster bootstrap over
   communities - compared with the community-level tercile gap. Pulsar has no row weights: not computable natively.
D. Workaround: expand each community into pseudo-individual rows by race (K rows per unit of population x share) and
   run FairnessAnalyzer on them. The point estimate matches C, but the intervals assume independent persons: their
   width is compared with the cluster bootstrap at several K.
E. attributes_data join with a composition table (one community, several race rows).

No individual data, no join
with any other source; only aggregates are written.

    LAB_MODE=dry python -m lab.harness FA-03
    LAB_MODE=full python -m lab.harness FA-03
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

MODE = os.environ.get("LAB_MODE", "dry")
DATA = os.environ.get("LAB_DATA_CC", "/data/lab/datasets/communities_crime_unnorm" if os.path.isdir("/data/lab")
                      else "C:/dev/pulsar-ml/data/lab/communities_crime_unnorm")

if MODE == "full":
    PARAMS = {"mode": "full", "n_bootstrap": 1000, "null_reps": 400, "cluster_bootstrap": 5000,
              "expansion_rows": [2_000, 20_000, 200_000, 2_000_000], "high_crime_quantile": 0.7, "seed": 20261102}
else:
    PARAMS = {"mode": "dry", "n_bootstrap": 200, "null_reps": 20, "cluster_bootstrap": 500,
              "expansion_rows": [2_000, 200_000], "high_crime_quantile": 0.7, "seed": 20261102}
PARAMS["data_dir"] = DATA

CRITERIA = {
    "communities": (">=", 1900),
    "null_false_alarm_rate": ("<=", 0.05),
    "expansion_point_max_abs_diff": ("<=", 0.01),
}

FAIR = ["disparate_impact_ratio", "demographic_parity_difference", "equal_opportunity_difference",
        "false_positive_rate_difference"]
SHARES = {"BLACK_T": "racepctblack", "WHITE_T": "racePctWhite", "HISP_T": "racePctHisp"}
RACES = {"Black": "racepctblack", "White": "racePctWhite", "Asian": "racePctAsian"}


# ---------------------------------------------------------------- data and model
# column layout of CommViolPredUnnormalizedData.txt (UCI 211): 4 identifiers, fold, 124 predictors, 18 crime targets
ID_COLS, FOLD, PREDICTORS, TARGET = [0, 1, 2, 3], 4, list(range(5, 129)), 145
NAMED = {5: "population", 7: "racepctblack", 8: "racePctWhite", 9: "racePctAsian", 10: "racePctHisp"}


def load() -> pd.DataFrame:
    raw = pd.read_csv(Path(DATA) / "CommViolPredUnnormalizedData.txt", header=None, na_values="?")
    raw = raw[raw[TARGET].notna()].reset_index(drop=True)
    df = raw[PREDICTORS].copy()
    df.columns = [NAMED.get(c, f"x{c}") for c in PREDICTORS]
    for c in ("racepctblack", "racePctWhite", "racePctAsian", "racePctHisp"):
        df[c] = df[c] / 100.0  # percentages -> shares
    df["fold"] = raw[FOLD].astype(int)
    df["ViolentCrimesPerPop"] = raw[TARGET]
    return df


def predict(df: pd.DataFrame, q: float, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    y = (df["ViolentCrimesPerPop"] >= df["ViolentCrimesPerPop"].quantile(q)).astype(int).to_numpy()
    drop = ["fold", "ViolentCrimesPerPop"]
    X = df.drop(columns=drop)
    X = X.loc[:, X.isna().mean() < 0.5].to_numpy(dtype=float)  # police (LEMAS) columns are ~84 % missing
    proba = np.empty(len(df))
    for f in sorted(df["fold"].unique()):
        test = (df["fold"] == f).to_numpy()
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)
        proba[test] = model.fit(X[~test], y[~test]).predict_proba(X[test])[:, 1]
    return y, proba, X.shape[1]


def terciles(values: pd.Series) -> pd.Series:
    """T1 < T2 < T3 by rank (ties on the 0.01 grid broken by order), so that the three groups have equal sizes"""
    return pd.Series(pd.qcut(values.rank(method="first"), 3, labels=["T1", "T2", "T3"]).astype(str), index=values.index)


def _rows(results):
    return [{"criterion": r.metric_name, "attribute": r.feature_name, "group": r.group, "n": r.n,
             "suppressed": bool(r.suppressed), "value": r.metric_value, "ci_low": r.ci_low, "ci_high": r.ci_high,
             "verdict": r.drift_status, "method": r.method} for r in results]


# ---------------------------------------------------------------- part C: population-weighted exposure
def exposure(pop, shares: dict, flag, y=None):
    """Share of the residents of each race living in a flagged community; with y, among high-crime / low-crime ones"""
    out = {}
    for race, s in shares.items():
        w = pop * s
        if y is None:
            out[race] = float((w * flag).sum() / w.sum())
        else:
            out[race] = {"tpr": float((w * flag * y).sum() / (w * y).sum()),
                         "fpr": float((w * flag * (1 - y)).sum() / (w * (1 - y)).sum())}
    return out


def cluster_bootstrap(pop, shares, flag, b, rng, reference="White"):
    """Percentile 95 % interval of the exposure difference race - reference, resampling communities"""
    n = len(pop)
    idx = rng.integers(0, n, (b, n))
    counts = np.apply_along_axis(np.bincount, 1, idx, minlength=n).astype(float)  # b x n multiplicities
    rates = {}
    for race, s in shares.items():
        w = pop * s
        rates[race] = (counts @ (w * flag)) / (counts @ w)
    out = {}
    for race in shares:
        if race == reference:
            continue
        d = rates[race] - rates[reference]
        r = rates[race] / rates[reference]
        out[race] = {"dp_ci": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
                     "dir_ci": [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))]}
    return out


# ---------------------------------------------------------------- part D: pseudo-individual expansion
def expand(pop, shares: dict, flag, y, total_rows: int):
    """Rows per (community, race) proportional to population x share, about total_rows in all"""
    mass = sum((pop * s).sum() for s in shares.values())
    k = total_rows / mass
    parts = []
    for race, s in shares.items():
        n = np.round(k * pop * s).astype(int)
        parts.append(pd.DataFrame({"RACE": np.repeat(race, n.sum()), "y_pred": np.repeat(flag, n),
                                   "y_true": np.repeat(y, n), "community": np.repeat(np.arange(len(pop)), n)}))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- run
def run(ctx):
    from pulsar_metrics.fairness import FairnessAnalyzer, FairnessPolicy

    p = ctx.params
    rng = np.random.default_rng(p["seed"])
    df = load()
    t0 = time.perf_counter()
    y, proba, n_features = predict(df, p["high_crime_quantile"], p["seed"])
    flag = (proba >= 0.5).astype(int)
    m = {"communities": int(len(df)), "features_used": n_features, "train_oof_s": round(time.perf_counter() - t0, 1),
         "accuracy_oof": float((flag == y).mean()), "base_rate": float(y.mean()), "flag_rate": float(flag.mean())}
    frame = pd.DataFrame({"y_true": y, "y_pred": flag, "y_pred_proba": proba})
    for col, src in SHARES.items():
        frame[col] = terciles(df[src])
    ctx.note("Each row is a community: group sizes n count communities, not people (the communities total "
             f"{len(df)} rows for roughly 1990 census populations of 10k+ each).")

    # ------------------------------------------------ A. community-level terciles
    policy = FairnessPolicy(criteria=FAIR, n_bootstrap=p["n_bootstrap"])
    t0 = time.perf_counter()
    analyzer = FairnessAnalyzer(list(SHARES), policy, reference_groups={c: "T1" for c in SHARES}, seed=p["seed"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = analyzer.run(frame)
    m["A_seconds"] = round(time.perf_counter() - t0, 2)
    m["A_results"] = _rows(res)
    m["A_warnings"] = [str(w.message) for w in caught]
    m["A_tercile_cut_points"] = {c: [float(df[s].quantile(1 / 3)), float(df[s].quantile(2 / 3))] for c, s in SHARES.items()}
    a = {(r.feature_name, r.group, r.metric_name): r for r in res}
    gap_dp_black = a[("BLACK_T", "BLACK_T=T3", "demographic_parity_difference")]
    m["A_black_T3_vs_T1"] = {k: {"value": a[("BLACK_T", "BLACK_T=T3", k)].metric_value,
                                 "ci": [a[("BLACK_T", "BLACK_T=T3", k)].ci_low, a[("BLACK_T", "BLACK_T=T3", k)].ci_high],
                                 "verdict": a[("BLACK_T", "BLACK_T=T3", k)].drift_status} for k in FAIR}
    # mean share of Black residents in each tercile (normalised scale): terciles are not races
    m["A_mean_black_share_by_black_tercile"] = {t: float(df["racepctblack"][frame["BLACK_T"] == t].mean())
                                                for t in ("T1", "T2", "T3")}
    m["A_mean_white_share_by_black_tercile"] = {t: float(df["racePctWhite"][frame["BLACK_T"] == t].mean())
                                                for t in ("T1", "T2", "T3")}

    # null control: permuted tercile labels
    alarms, t0 = 0, time.perf_counter()
    for i in range(p["null_reps"]):
        perm = frame[["y_true", "y_pred", "y_pred_proba"]].copy()
        perm["G"] = rng.permutation(frame["BLACK_T"].to_numpy())
        r = FairnessAnalyzer(["G"], FairnessPolicy(criteria=["demographic_parity_difference"], n_bootstrap=p["n_bootstrap"]),
                             reference_groups={"G": "T1"}, seed=i).run(perm)
        alarms += any(x.drift_status is True for x in r)
    m["null_reps"] = p["null_reps"]
    m["null_false_alarm_rate"] = alarms / p["null_reps"]
    m["null_seconds"] = round(time.perf_counter() - t0, 1)

    # ------------------------------------------------ B. the share itself as attribute
    # the float column as a user has it (not converted to strings on a grid: that made it categorical on purpose)
    from pulsar_metrics.fairness import analyzer as fairness_analyzer

    refusals = tuple(c for c in (getattr(fairness_analyzer, "AllCellsSuppressed", None),) if c is not None)
    raw = frame[["y_true", "y_pred", "y_pred_proba"]].copy()
    raw["BLACK_SHARE"] = df["racepctblack"].to_numpy(dtype=float)
    m["B_attribute_dtype"] = str(raw["BLACK_SHARE"].dtype)
    m["B_groups"] = int(raw["BLACK_SHARE"].nunique())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        an_b = FairnessAnalyzer(["BLACK_SHARE"], FairnessPolicy(criteria=["demographic_parity_difference"],
                                                                n_bootstrap=p["n_bootstrap"]), seed=0)
        try:
            rb = an_b.run(raw)
        except refusals as exc:  # every distinct share is a group below min_group_size: the refusal is the measure
            rb = None
            m["B_outcome"] = f"refused: {type(exc).__name__}: {str(exc)[:200]}"
    m["B_warnings"] = [str(w.message) for w in caught]
    if rb is not None:
        kept = [r for r in rb if not r.suppressed]
        cov = an_b.coverage()
        m["B_outcome"] = "accepted"
        m["B_kept_groups"] = len(kept)
        m["B_suppressed_groups"] = len(rb) - len(kept)
        m["B_reference"] = rb[0].reference_group if rb else None
        m["B_no_verdict_share"] = cov[0].metric_value if cov else None
        m["B_established"] = sum(r.drift_status is True for r in kept)
    # does Pulsar tell the user to bin the numeric attribute (in the error or a warning)?
    m["B_binning_hint"] = any(w in text.lower() for text in [m["B_outcome"], *m["B_warnings"]]
                              for w in ("bin", "numeric", "continuous", "quantile"))

    # ------------------------------------------------ C. population-weighted exposure by race
    pop = df["population"].to_numpy(dtype=float)
    shares = {r: df[c].to_numpy(dtype=float) for r, c in RACES.items()}
    exp = exposure(pop, shares, flag)
    exp_lab = exposure(pop, shares, flag, y)
    m["C_exposure_flagged"] = exp
    m["C_exposure_tpr_fpr"] = exp_lab
    m["C_unweighted_flag_rate_by_black_tercile"] = {t: float(flag[frame["BLACK_T"].to_numpy() == t].mean())
                                                    for t in ("T1", "T2", "T3")}
    t0 = time.perf_counter()
    m["C_cluster_ci"] = cluster_bootstrap(pop, shares, flag, p["cluster_bootstrap"], rng)
    m["C_cluster_seconds"] = round(time.perf_counter() - t0, 1)
    m["C_dp_black_minus_white"] = exp["Black"] - exp["White"]
    m["C_dir_black_over_white"] = exp["Black"] / exp["White"]
    m["C_dp_asian_minus_white"] = exp["Asian"] - exp["White"]
    m["C_community_dp_T3_vs_T1"] = gap_dp_black.metric_value
    m["C_pop_share_in_black_T3"] = float(pop[frame["BLACK_T"].to_numpy() == "T3"].sum() / pop.sum())

    # ------------------------------------------------ D. pseudo-individual expansion
    def zone_verdict(lo, hi, zone=(-0.1, 0.1)):
        if hi < zone[0] or lo > zone[1]:
            return True
        if zone[0] <= lo and hi <= zone[1]:
            return False
        return None

    cluster = {race: {"ci": m["C_cluster_ci"][race]["dp_ci"],
                      "width": m["C_cluster_ci"][race]["dp_ci"][1] - m["C_cluster_ci"][race]["dp_ci"][0],
                      "verdict": zone_verdict(*m["C_cluster_ci"][race]["dp_ci"]),
                      "point": exp[race] - exp["White"]} for race in ("Black", "Asian")}
    m["C_cluster_dp"] = cluster
    exp_rows, diffs = [], []
    for total in p["expansion_rows"]:
        big = expand(pop, shares, flag, y, total)
        t0 = time.perf_counter()
        an = FairnessAnalyzer(["RACE"], FairnessPolicy(criteria=["demographic_parity_difference", "disparate_impact_ratio"],
                                                       n_bootstrap=p["n_bootstrap"]),
                              reference_groups={"RACE": "White"}, seed=0)
        rd = an.run(big)
        sec = time.perf_counter() - t0
        row = {"rows": int(len(big)), "rows_by_race": big["RACE"].value_counts().to_dict(), "seconds": round(sec, 2)}
        for r in rd:
            race = r.group.split("=", 1)[1]
            if race not in cluster:
                continue
            if r.metric_name == "demographic_parity_difference":
                row[race] = {"dp": r.metric_value, "dp_ci": [r.ci_low, r.ci_high], "dp_verdict": r.drift_status,
                             "dp_ci_width": r.ci_high - r.ci_low,
                             "width_over_cluster": (r.ci_high - r.ci_low) / cluster[race]["width"], "method": r.method}
                if race == "Black":
                    diffs.append(abs(r.metric_value - cluster[race]["point"]))
            elif r.metric_name == "disparate_impact_ratio":
                row.setdefault(race, {}).update({"dir": r.metric_value, "dir_ci": [r.ci_low, r.ci_high],
                                                 "dir_verdict": r.drift_status})
        exp_rows.append(row)
        del big
    m["D_expansion"] = exp_rows
    m["expansion_point_max_abs_diff"] = diffs[-1] if diffs else None  # largest expansion (rounding vanishes)
    m["expansion_point_diffs"] = diffs
    # only expansions whose point estimate matches the weighted exposure (small expansions round small groups away)
    flips = [(race, r["rows"], r[race]["dp_verdict"]) for r in exp_rows for race in cluster
             if race in r and abs(r[race]["dp"] - cluster[race]["point"]) <= 0.01
             and r[race]["dp_verdict"] != cluster[race]["verdict"]]
    m["D_verdicts_differing_from_cluster"] = [{"race": a, "rows": b, "pulsar_verdict": c,
                                               "cluster_verdict": cluster[a]["verdict"]} for a, b, c in flips]

    # ------------------------------------------------ E. composition table through attributes_data
    comp = pd.DataFrame([{"cid": i, "RACE": race} for i in range(len(df)) for race in RACES])
    preds = pd.DataFrame({"cid": np.arange(len(df)), "y_pred": flag, "y_true": y})
    try:
        FairnessAnalyzer(["RACE"], FairnessPolicy(n_bootstrap=50)).run(preds, attributes_data=comp, on="cid")
        m["E_composition_join"] = "accepted"
    except Exception as exc:  # noqa: BLE001 - the refusal is the observation
        m["E_composition_join"] = f"refused: {type(exc).__name__}: {str(exc)[:200]}"

    # ------------------------------------------------ limits
    ctx.limit(
        "FairnessAnalyzer assumes one row = one person with an individual protected attribute: with aggregate data "
        "(race known only as community shares) it can only compare communities grouped by share tercile; it has no "
        "row weights and no fractional group membership, so the impact on people of each race is not computable",
        f"{len(df)} communities. Community-level parity gap T3 vs T1 of racepctblack = {gap_dp_black.metric_value:+.3f} "
        f"(verdict {gap_dp_black.drift_status}); population-weighted share of residents living in a flagged community, "
        f"computed outside Pulsar: Black {exp['Black']:.3f}, White {exp['White']:.3f}, Asian {exp['Asian']:.3f} "
        f"(Black - White {m['C_dp_black_minus_white']:+.3f}, cluster bootstrap 95 % {cluster['Black']['ci'][0]:+.3f} to "
        f"{cluster['Black']['ci'][1]:+.3f}; "
        f"ratio {m['C_dir_black_over_white']:.2f}). Mean Black share of the communities in T1/T2/T3: "
        + "/".join(f"{v:.3f}" for v in m["A_mean_black_share_by_black_tercile"].values())
        + f"; the top tercile holds {m['C_pop_share_in_black_T3']:.0%} of the population: a tercile is a mix of residents "
        "of all races, not a race group, and its n counts communities, not people.",
        "Add `weights=` (frequency / population weights) to FairnessAnalyzer and a composition input "
        "(unit id, group, count or share) that computes group rates as weighted exposures; label results with "
        "`unit='community'` and state in the report that aggregate attributes give ecological, not individual, results.",
        "major")

    if exp_rows and "Black" in exp_rows[-1]:
        def describe(race):
            c = cluster[race]
            return (f"{race} - White gap {c['point']:+.3f}: cluster bootstrap over {len(df)} communities "
                    f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] (width {c['width']:.4f}, verdict {c['verdict']}); Pulsar on "
                    + ", ".join(f"{r['rows']:,} rows: width {r[race]['dp_ci_width']:.4f} "
                                f"({r[race]['width_over_cluster']:.2f}x, verdict {r[race]['dp_verdict']})" for r in exp_rows))
        flip_text = (f"; the verdict differs from the cluster verdict in {len(flips)} case(s): "
                     + ", ".join(f"{a} at {b:,} rows ({c} vs {cluster[a]['verdict']})" for a, b, c in flips)) if flips else             "; no verdict differs from the cluster verdict at these effect sizes"
        ctx.limit(
            "Workaround by pseudo-individual expansion gives the right point estimate but intervals that treat "
            "replicated rows as independent people: their width shrinks as 1/sqrt(expansion factor), an arbitrary "
            "choice, while the information (number of communities) does not change"
            + (", and the verdict changes with it" if flips else ""),
            describe("Black") + "; " + describe("Asian")
            + f"; point vs weighted exposure at the largest expansion {diffs[-1]:.1e}" + flip_text,
            "Support a cluster / unit id (`cluster=`) in the bootstrap (resample units, not rows) and warn when many rows "
            "repeat identical (unit, decision) tuples; document that the effective sample is the number of units.",
            "major")

    if "refused" in m["E_composition_join"]:
        ctx.limit(
            "attributes_data join requires one attribute row per prediction (validate='one_to_one'): a composition "
            "table (one community, several groups with shares) is refused",
            m["E_composition_join"],
            "Accept a many-to-one composition table with a share / count column and turn it into weights.",
            "minor")
    elif m["E_composition_join"] == "accepted":
        ctx.note("composition join accepted (unexpected): check the semantics of the result")

    if m["B_outcome"].startswith("refused") and not m["B_binning_hint"]:
        ctx.limit(
            f"A continuous share passed as protected attribute is treated as {m['B_groups']} categories and refused, "
            "without any hint that the attribute is numeric and must be binned",
            f"racepctblack as {m['B_attribute_dtype']} ({m['B_groups']} distinct values over {len(df)} rows): "
            f"{m['B_outcome']}",
            "Detect numeric attributes with many distinct values and require explicit binning "
            "(e.g. `bins={'share': 3}` by quantile), recording the cut points in the result; say so in the error.",
            "minor")
    elif m["B_outcome"] == "accepted" and m["B_kept_groups"] > 3:
        ctx.limit(
            "A continuous share passed as protected attribute is treated as categorical: every distinct value is a "
            "group, most are suppressed or inconclusive, without any warning that the attribute is numeric",
            f"racepctblack as {m['B_attribute_dtype']}: {m['B_groups']} groups, {m['B_kept_groups']} kept, "
            f"{m['B_suppressed_groups']} suppressed, no-verdict share {m['B_no_verdict_share']:.1%}, "
            f"reference {m['B_reference']}, {m['B_established']} 'established'",
            "Detect numeric attributes with many distinct values and require explicit binning (e.g. `bins={'share': 3}` "
            "by quantile), recording the cut points in the result.",
            "minor")
    return m
