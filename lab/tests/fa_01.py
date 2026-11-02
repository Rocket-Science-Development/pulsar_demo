"""FA-01: fairness across states and years (Folktables ACSIncome / ACSEmployment).

Models trained on California 2014, then applied to 10 states x 2014-2018. Four parts:

A. FairnessAnalyzer on every state-year (SEX, RAC1P; DIR, parity, equal opportunity, FPR difference), with intervals,
   and agreement of the point estimates with Fairlearn MetricFrame.
B. Intersections: SEX x RAC1P x age band x education x marital status on CA and GA 2018 - groups, suppression,
   runtime, cost of simultaneous intervals.
C. FairnessMonitor fed year after year (2014 -> 2018) per state; alarms compared with the yearly gaps; null control
   with permuted group labels.

Data: ACS PUMS via Folktables, kept outside any repository (LAB_DATA, default /data/lab/datasets/folktables). Only
group-level aggregates of groups with >= 30 rows are written. No row, no identifier, no join with any other source.
This product uses the Census Bureau Data API / PUMS but is not endorsed or certified by the Census Bureau.

    LAB_MODE=dry python -m lab.harness FA-01     # 2 states x 2 years, small bootstrap
    LAB_MODE=full python -m lab.harness FA-01
"""

import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

MODE = os.environ.get("LAB_MODE", "dry")
DATA = os.environ.get("LAB_DATA", "/data/lab/datasets/folktables" if os.path.isdir("/data/lab") else "C:/dev/pulsar-ml/data/lab/folktables")

if MODE == "full":
    PARAMS = {"mode": "full", "states": ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"],
              "years": [2014, 2015, 2016, 2017, 2018], "n_bootstrap": 500, "intersection_bootstrap": 500,
              "intersection_states": ["CA", "GA"], "max_depth": 5, "level_budget_s": 900, "null_reps": 20,
              "batch": 2000, "tolerance": 0.05, "alpha": 0.05, "workers": 6}
else:
    PARAMS = {"mode": "dry", "states": ["CA", "GA"], "years": [2014, 2018], "n_bootstrap": 50,
              "intersection_bootstrap": 50, "intersection_states": ["GA"], "max_depth": 3, "level_budget_s": 120,
              "null_reps": 2, "batch": 2000, "tolerance": 0.05, "alpha": 0.05, "workers": 4}
PARAMS["data_dir"] = DATA

CRITERIA = {
    "fairlearn_max_abs_diff": ("<=", 1e-9),
    "share_kept_results_with_interval": ("==", 1.0),
    "false_alarms_on_stable_within_gaps": ("==", 0),
    "null_false_alarm_rate": ("<=", 0.05),
}

FAIR_CRITERIA = ["disparate_impact_ratio", "demographic_parity_difference", "equal_opportunity_difference",
                 "false_positive_rate_difference"]
TASKS = ("income", "employment")


# ---------------------------------------------------------------- data
def _task(name):
    from folktables import ACSEmployment, ACSIncome

    return ACSIncome if name == "income" else ACSEmployment


def _bands(frame: pd.DataFrame) -> pd.DataFrame:
    """Coarse protected / intersection attributes as small integers (group labels only)"""
    out = pd.DataFrame(index=frame.index)
    out["SEX"] = frame["SEX"].astype(int)
    out["RAC1P"] = frame["RAC1P"].astype(int)
    out["AGEG"] = pd.cut(frame["AGEP"], [-1, 24, 34, 44, 54, 64, 200], labels=False).astype(int)
    out["SCHLG"] = pd.cut(frame["SCHL"].fillna(0), [-1, 15, 17, 20, 21, 30], labels=False).astype(int)
    out["MARG"] = frame["MAR"].astype(int)
    return out


def _raw(state, year):
    from folktables import ACSDataSource

    source = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person", root_dir=DATA)
    return source.get_data(states=[state], download=False)


def _compact(raw, task, model, test_index=None):
    X, y, _ = _task(task).df_to_pandas(raw)
    if test_index is not None:
        X, y = X.loc[test_index], y.loc[test_index]
    score = model.predict_proba(X.to_numpy())[:, 1]
    out = _bands(X)
    out["y_true"] = y.iloc[:, 0].astype(int).to_numpy()
    out["y_pred"] = (score >= 0.5).astype(int)
    out["y_pred_proba"] = score
    return out.reset_index(drop=True)


def train(seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    raw = _raw("CA", 2014)
    models, test_index, accuracy = {}, {}, {}
    for task in TASKS:
        X, y, _ = _task(task).df_to_pandas(raw)
        y = y.iloc[:, 0].astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
        model = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(X_tr.to_numpy(), y_tr.to_numpy())
        models[task], test_index[task] = model, X_te.index
        accuracy[task] = float(model.score(X_te.to_numpy(), y_te.to_numpy()))
    return models, test_index, accuracy


def _limit_threads():
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(2)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------- part A
def _fairlearn_gaps(frame, attr, reference, groups):
    from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate, true_positive_rate

    mf = MetricFrame(metrics={"sr": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate},
                     y_true=frame["y_true"], y_pred=frame["y_pred"], sensitive_features=frame[attr].astype(str))
    bg = mf.by_group
    ref = bg.loc[reference]
    out = {}
    for g in groups:
        row = bg.loc[g]
        out[g] = {"disparate_impact_ratio": row["sr"] / ref["sr"] if ref["sr"] else float("nan"),
                  "demographic_parity_difference": row["sr"] - ref["sr"],
                  "equal_opportunity_difference": row["tpr"] - ref["tpr"],
                  "false_positive_rate_difference": row["fpr"] - ref["fpr"]}
    return out


def analyse_state_year(args):
    """Load one state-year, apply both models, run FairnessAnalyzer and Fairlearn; return aggregates only"""
    state, year, models, test_index, n_bootstrap = args
    _limit_threads()
    from pulsar_metrics.fairness import FairnessAnalyzer, FairnessPolicy

    t0 = time.perf_counter()
    raw = _raw(state, year)
    load_s = time.perf_counter() - t0
    out = {"state": state, "year": year, "load_s": round(load_s, 1), "tasks": {}}
    frames = {}
    for task in TASKS:
        idx = test_index[task] if (state, year) == ("CA", 2014) else None
        frame = _compact(raw, task, models[task], idx)
        frames[task] = frame
        t1 = time.perf_counter()
        analyzer = FairnessAnalyzer(["SEX", "RAC1P"], FairnessPolicy(criteria=FAIR_CRITERIA, n_bootstrap=n_bootstrap),
                                    reference_groups={"SEX": "1", "RAC1P": "1"}, seed=0)
        results = analyzer.run(frame)
        pulsar_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        kept = [r for r in results if not r.suppressed and r.error is None]
        diffs = []
        for attr in ("SEX", "RAC1P"):
            groups = sorted({r.group.split("=", 1)[1] for r in kept if r.feature_name == attr})
            fl = _fairlearn_gaps(frame, attr, "1", groups)
            for r in kept:
                if r.feature_name == attr:
                    a, b = r.metric_value, fl[r.group.split("=", 1)[1]][r.metric_name]
                    if not (np.isnan(a) and np.isnan(b)):
                        diffs.append(abs(a - b))
        fairlearn_s = time.perf_counter() - t2
        rows = []
        for r in results:
            rows.append({"criterion": r.metric_name, "attribute": r.feature_name, "group": r.group,
                         "suppressed": bool(r.suppressed), "n": None if r.suppressed else r.n,
                         "value": r.metric_value, "ci_low": r.ci_low, "ci_high": r.ci_high,
                         "verdict": r.drift_status, "error": r.error})
        out["tasks"][task] = {
            "rows": int(len(frame)), "accuracy": float((frame["y_true"] == frame["y_pred"]).mean()),
            "pulsar_s": round(pulsar_s, 2), "fairlearn_s": round(fairlearn_s, 2),
            "fairlearn_max_abs_diff": float(max(diffs)) if diffs else None, "fairlearn_comparisons": len(diffs),
            "results": rows,
        }
    # yearly per-group streams for part C (income decisions only, group labels + decision)
    out["_stream"] = frames["income"][["SEX", "RAC1P", "y_true", "y_pred"]].astype(np.int8)
    return out


# ---------------------------------------------------------------- part B
def intersections(frame, n_bootstrap, max_depth, budget_s):
    from pulsar_metrics.fairness import FairnessAnalyzer, FairnessPolicy

    ladder = ["SEX", "RAC1P", "AGEG", "SCHLG", "MARG"][:max_depth]
    levels = []
    for k in range(1, len(ladder) + 1):
        attrs = ladder[:k]
        t0 = time.perf_counter()
        analyzer = FairnessAnalyzer(attrs, FairnessPolicy(criteria=["demographic_parity_difference",
                                                                    "equal_opportunity_difference"],
                                                          n_bootstrap=n_bootstrap), intersections=True, seed=0)
        results = analyzer.run(frame)
        seconds = time.perf_counter() - t0
        dp = [r for r in results if r.metric_name == "demographic_parity_difference"]
        deepest = [r for r in dp if r.feature_name == "|".join(attrs)]
        kept = [r for r in deepest if not r.suppressed]
        supp = [r for r in deepest if r.suppressed]
        levels.append({
            "depth": k, "attributes": attrs, "attribute_sets": len({r.feature_name for r in dp}),
            "groups_all_sets": len(dp) + len({r.feature_name for r in dp}),
            "deepest_groups": len(deepest) + 1, "deepest_kept": len(kept) + 1, "deepest_suppressed": len(supp),
            "deepest_suppressed_share_groups": len(supp) / (len(deepest) + 1),
            "deepest_established": sum(r.drift_status is True for r in kept),
            "deepest_inconclusive": sum(r.drift_status is None for r in kept),
            "deepest_within": sum(r.drift_status is False for r in kept),
            "median_halfwidth": float(np.median([(r.ci_high - r.ci_low) / 2 for r in kept])) if kept else None,
            "seconds": round(seconds, 1),
        })
        sizes = frame.groupby(attrs).size()
        levels[-1]["deepest_rows_in_suppressed_cells_share"] = round(float(sizes[sizes < 30].sum() / len(frame)), 4)
        growth = seconds / max(levels[-2]["seconds"], 0.1) if len(levels) > 1 else 5.0
        if seconds > budget_s or (k < len(ladder) and seconds * growth > budget_s):
            levels[-1]["stopped"] = (f"level took {seconds:.0f} s; next level projected at {seconds * growth:.0f} s "
                                     f"> budget {budget_s} s; deeper levels not run")
            break
    # cost of simultaneous intervals on the deepest level reached: one cell column, simultaneous vs pointwise
    attrs = levels[-1]["attributes"]
    cell = frame[attrs].astype(str).agg("|".join, axis=1)
    data = frame.assign(cell=cell)
    widths = {}
    for simultaneous in (True, False):
        analyzer = FairnessAnalyzer(["cell"], FairnessPolicy(criteria=["demographic_parity_difference"],
                                                             n_bootstrap=n_bootstrap, simultaneous=simultaneous), seed=0)
        res = [r for r in analyzer.run(data) if not r.suppressed]
        widths[simultaneous] = {"median_halfwidth": float(np.median([(r.ci_high - r.ci_low) / 2 for r in res])),
                                "established": sum(r.drift_status is True for r in res),
                                "within": sum(r.drift_status is False for r in res),
                                "inconclusive": sum(r.drift_status is None for r in res), "groups": len(res)}
    return {"levels": levels, "simultaneous": widths[True], "pointwise": widths[False],
            "width_ratio": widths[True]["median_halfwidth"] / widths[False]["median_halfwidth"]}


def intersections_job(args):
    state, models, n_bootstrap, max_depth, budget_s = args
    _limit_threads()
    raw = _raw(state, 2018)
    frame = _compact(raw, "income", models["income"])
    out = intersections(frame, n_bootstrap, max_depth, budget_s)
    out["state"], out["rows"] = state, int(len(frame))
    return out


# ---------------------------------------------------------------- part C
def monitor_stream(streams, attr, groups, criterion, tolerance, alpha, batch, seed, permute=False):
    """Feed years in order, shuffled within year, by batches; return alarm (year, pair) per group"""
    from pulsar_metrics.fairness import FairnessMonitor

    rng = np.random.default_rng(seed)
    monitor = FairnessMonitor(attr, "1", groups, criterion=criterion, tolerance=tolerance, alpha=alpha)
    alarm_year = {g: None for g in monitor.groups}
    for year, frame in streams:
        frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
        if permute:  # null control: group labels carry no information on decisions
            frame[attr] = rng.permutation(frame[attr].to_numpy())
        for start in range(0, len(frame), batch):
            res = monitor.update(frame.iloc[start:start + batch])
            for r in res:
                g = r.group.split("=", 1)[1]
                if r.drift_status and alarm_year[g] is None:
                    alarm_year[g] = year
    return {g: {"alarm_year": alarm_year[g], "pairs": t.n, "alarm_pair": t.rejected_at, "mean_difference": t.mean_difference,
                "e_value": t.evidence} for g, t in monitor.tests.items()}


def monitor_job(args):
    state, streams, groups_by_attr, tolerance, alpha, batch, null_reps = args
    out = {"state": state, "real": {}, "null": {}}
    for attr, groups in groups_by_attr.items():
        for criterion in ("demographic_parity", "equal_opportunity"):
            out["real"][f"{attr}:{criterion}"] = monitor_stream(streams, attr, groups, criterion, tolerance, alpha, batch, 0)
        alarms = 0
        for rep in range(null_reps):
            res = monitor_stream(streams, attr, groups, "demographic_parity", tolerance, alpha, batch, 1000 + rep, True)
            alarms += any(v["alarm_year"] is not None for v in res.values())
        out["null"][attr] = {"runs": null_reps, "runs_with_alarm": alarms}
    return out


# ---------------------------------------------------------------- run
def _yearly_gap(analysis, state, attr, group, criterion):
    """(year, value, ci_low, ci_high) of the fixed analysis, income task"""
    out = []
    for a in analysis:
        if a["state"] != state:
            continue
        for r in a["tasks"]["income"]["results"]:
            if r["attribute"] == attr and r["group"] == f"{attr}={group}" and r["criterion"] == criterion and not r["suppressed"]:
                out.append((a["year"], r["value"], r["ci_low"], r["ci_high"]))
    return sorted(out)


def run(ctx):
    p = ctx.params
    ctx.note("Folktables / ACS PUMS: not endorsed or certified by the Census Bureau. Aggregates only, groups >= 30 rows.")
    t0 = time.perf_counter()
    models, test_index, accuracy = train()
    train_s = time.perf_counter() - t0
    jobs = [(s, y, models, test_index, p["n_bootstrap"]) for s in p["states"] for y in p["years"]]

    # ------------------------------------------------ A
    analysis = []
    with ProcessPoolExecutor(max_workers=p["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
        for a in pool.map(analyse_state_year, jobs):
            analysis.append(a)
            print(f"A {a['state']} {a['year']} income {a['tasks']['income']['pulsar_s']} s", flush=True)
    streams = {}
    for a in analysis:
        streams.setdefault(a["state"], []).append((a["year"], a.pop("_stream")))

    all_rows = [(a, t, r) for a in analysis for t in TASKS for r in a["tasks"][t]["results"]]
    kept = [r for _, _, r in all_rows if not r["suppressed"] and r["error"] is None]
    with_ci = [r for r in kept if r["ci_low"] is not None and r["ci_high"] is not None]
    fl_diffs = [a["tasks"][t]["fairlearn_max_abs_diff"] for a in analysis for t in TASKS
                if a["tasks"][t]["fairlearn_max_abs_diff"] is not None]
    verdicts = {}
    for _, t, r in all_rows:
        if r["suppressed"] or r["error"]:
            continue
        key = f"{t}:{r['attribute']}:{r['criterion']}"
        v = verdicts.setdefault(key, {"established": 0, "within": 0, "inconclusive": 0})
        v["established" if r["verdict"] is True else "within" if r["verdict"] is False else "inconclusive"] += 1
    suppressed_groups = {}
    for a, t, r in all_rows:
        if r["suppressed"] and t == "income" and r["criterion"] == FAIR_CRITERIA[0]:
            suppressed_groups.setdefault(r["group"], set()).add((a["state"], a["year"]))
    # inconclusive share by group size band (income, parity)
    bands = {}
    for r in kept:
        if r["criterion"] != "demographic_parity_difference":
            continue
        band = "<300" if r["n"] < 300 else "300-1000" if r["n"] < 1000 else "1000-5000" if r["n"] < 5000 else ">=5000"
        b = bands.setdefault(band, {"results": 0, "inconclusive": 0, "halfwidths": []})
        b["results"] += 1
        b["inconclusive"] += r["verdict"] is None
        b["halfwidths"].append((r["ci_high"] - r["ci_low"]) / 2)
    for b in bands.values():
        b["median_halfwidth"] = float(np.median(b.pop("halfwidths")))
    pulsar_s = [a["tasks"][t]["pulsar_s"] for a in analysis for t in TASKS]
    fairlearn_s = [a["tasks"][t]["fairlearn_s"] for a in analysis for t in TASKS]
    acc_by_state = {f"{a['state']}-{a['year']}": round(a["tasks"]["income"]["accuracy"], 4) for a in analysis}

    # ------------------------------------------------ B and C
    groups_by_state = {}
    for s in p["states"]:
        race = [g for g in ["2", "3", "4", "5", "6", "7", "8", "9"]
                if len(_yearly_gap(analysis, s, "RAC1P", g, "demographic_parity_difference")) == len(p["years"])]
        groups_by_state[s] = {"SEX": ["2"], "RAC1P": race}
    b_jobs = [(s, models, p["intersection_bootstrap"], p["max_depth"], p["level_budget_s"]) for s in p["intersection_states"]]
    c_jobs = [(s, sorted(streams[s]), groups_by_state[s], p["tolerance"], p["alpha"], p["batch"], p["null_reps"])
              for s in p["states"]]
    with ProcessPoolExecutor(max_workers=p["workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
        b_fut = [pool.submit(intersections_job, j) for j in b_jobs]
        c_res = list(pool.map(monitor_job, c_jobs))
        b_res = [f.result() for f in b_fut]

    # classify monitored (state, attribute, group) from the yearly fixed gaps
    tol = p["tolerance"]
    cases = []
    for c in c_res:
        s = c["state"]
        for attr, groups in groups_by_state[s].items():
            for g in groups:
                gaps = _yearly_gap(analysis, s, attr, g, "demographic_parity_difference")
                values = [v for _, v, _, _ in gaps]
                spread = max(values) - min(values)
                within = all(-tol <= lo and hi <= tol for _, _, lo, hi in gaps)
                beyond = all(hi < -tol or lo > tol for _, _, lo, hi in gaps)
                alarm = c["real"][f"{attr}:demographic_parity"][g]
                cases.append({"state": s, "attribute": attr, "group": g, "gap_min": round(min(values), 4),
                              "gap_max": round(max(values), 4), "spread": round(spread, 4),
                              "stable": spread <= 0.02, "within_tolerance_all_years": within,
                              "beyond_tolerance_all_years": beyond, "alarm_year": alarm["alarm_year"],
                              "alarm_pair": alarm["alarm_pair"]})
    stable_within = [x for x in cases if x["stable"] and x["within_tolerance_all_years"]]
    stable_beyond = [x for x in cases if x["stable"] and x["beyond_tolerance_all_years"]]
    false_alarms = [x for x in stable_within if x["alarm_year"] is not None]
    beyond_first_year = [x for x in stable_beyond if x["alarm_year"] == p["years"][0]]
    null_runs = sum(c["null"][a]["runs"] for c in c_res for a in c["null"])
    null_alarms = sum(c["null"][a]["runs_with_alarm"] for c in c_res for a in c["null"])

    metrics = {
        "train_s": round(train_s, 1), "accuracy_ca2014_test": accuracy, "state_years": len(analysis),
        "rows_income": sum(a["tasks"]["income"]["rows"] for a in analysis),
        "rows_employment": sum(a["tasks"]["employment"]["rows"] for a in analysis),
        "kept_results": len(kept), "share_kept_results_with_interval": len(with_ci) / len(kept) if kept else 0,
        "fairlearn_max_abs_diff": max(fl_diffs) if fl_diffs else None,
        "fairlearn_comparisons": sum(a["tasks"][t]["fairlearn_comparisons"] for a in analysis for t in TASKS),
        "pulsar_seconds_per_state_year_task": {"median": float(np.median(pulsar_s)), "max": float(max(pulsar_s))},
        "fairlearn_seconds_per_state_year_task": {"median": float(np.median(fairlearn_s)), "max": float(max(fairlearn_s))},
        "verdicts": verdicts, "parity_by_group_size": bands,
        "suppressed_groups_income": {g: len(v) for g, v in sorted(suppressed_groups.items())},
        "income_accuracy_by_state_year": acc_by_state,
        "intersections": b_res,
        "monitor_cases": cases, "stable_within_cases": len(stable_within), "stable_beyond_cases": len(stable_beyond),
        "false_alarms_on_stable_within_gaps": len(false_alarms),
        "stable_beyond_alarmed_first_year": len(beyond_first_year),
        "null_runs": null_runs, "null_runs_with_alarm": null_alarms,
        "null_false_alarm_rate": null_alarms / null_runs if null_runs else None,
        "monitor_real": {c["state"]: c["real"] for c in c_res},
    }
    (ctx.workdir / "analysis_aggregates.json").write_text(json.dumps(analysis, indent=1, default=str), encoding="utf-8")

    # ------------------------------------------------ limits (from the numbers above)
    if fl_diffs and max(fl_diffs) > 1e-9:
        ctx.limit(f"FairnessAnalyzer point estimates differ from Fairlearn MetricFrame by up to {max(fl_diffs):.2e}",
                  f"{metrics['fairlearn_comparisons']} comparisons over {len(analysis)} state-years x 2 tasks",
                  "Investigate the statistic or the grouping (string keys of float codes); add a Fairlearn parity test.",
                  "major")
    for lv_state in b_res:
        levels = lv_state["levels"]
        deep = levels[-1]
        if deep["deepest_suppressed_share_groups"] >= 0.3 or deep["seconds"] > 10 * max(levels[0]["seconds"], 0.1):
            ctx.limit(
                f"Intersections explode on {lv_state['state']} 2018 ({lv_state['rows']} rows): depth {deep['depth']} gives "
                f"{deep['deepest_groups']} cells, {deep['deepest_suppressed']} suppressed "
                f"({deep['deepest_suppressed_share_groups']:.0%}, holding {deep['deepest_rows_in_suppressed_cells_share']:.1%} "
                f"of rows, left without verdict); runtime {levels[0]['seconds']} s at depth 1 -> "
                f"{deep['seconds']} s at depth {deep['depth']} ({deep['attribute_sets']} attribute sets)"
                + (f"; {deep['stopped']}" if deep.get("stopped") else ""),
                "; ".join(f"depth {x['depth']}: {x['deepest_groups']} cells, {x['deepest_suppressed']} suppressed, "
                          f"{x['deepest_inconclusive']} inconclusive of {x['deepest_kept'] - 1}, {x['seconds']} s"
                          for x in levels),
                "Cap the depth (max_intersection_depth, default 2) and the number of cells; vectorise the bootstrap "
                "(one multinomial weight matrix per group instead of a Python loop per replicate); report the share "
                "of the population left in suppressed cells; offer a hierarchical / shrinkage estimate for small cells.",
                "major")
        if lv_state["width_ratio"] > 1.2:
            sim, pw = lv_state["simultaneous"], lv_state["pointwise"]
            ctx.limit(
                f"Simultaneous intervals over {sim['groups']} intersection cells ({lv_state['state']} 2018) are "
                f"{lv_state['width_ratio']:.2f}x wider than pointwise ones",
                f"median half-width {sim['median_halfwidth']:.4f} vs {pw['median_halfwidth']:.4f}; established "
                f"{sim['established']} vs {pw['established']}, within {sim['within']} vs {pw['within']}, inconclusive "
                f"{sim['inconclusive']} vs {pw['inconclusive']}",
                "Document the trade-off; offer FDR-controlling (Benjamini-Yekutieli) selective intervals for many cells "
                "and a pre-declared list of cells of interest, so that the band is paid only over declared cells.",
                "minor")
    if false_alarms:
        ctx.limit(f"FairnessMonitor alarmed on {len(false_alarms)} of {len(stable_within)} stable gaps within tolerance",
                  str(false_alarms[:5]), "Investigate pairing across batches of different years.", "major")
    if beyond_first_year:
        ctx.limit(
            f"FairnessMonitor tests the level of the gap, not its change: {len(beyond_first_year)} of "
            f"{len(stable_beyond)} stable gaps beyond tolerance alarm in the first year and stay alarmed for 5 years",
            "; ".join(f"{x['state']} {x['attribute']}={x['group']}: gap {x['gap_min']}..{x['gap_max']}, alarm at pair "
                      f"{x['alarm_pair']}" for x in beyond_first_year[:8]),
            "Add a change mode to FairnessMonitor (H0: |gap - baseline gap| <= tolerance, baseline from a reference "
            "period) and a reset / re-arm after an acknowledged alarm, so that a known historical gap does not mask a "
            "new degradation.",
            "major")
    if null_runs and null_alarms / null_runs > p["alpha"]:
        ctx.limit(f"Null control: {null_alarms}/{null_runs} permuted streams alarmed (> alpha {p['alpha']})",
                  "group labels permuted within each year", "Check the e-process validity under batch pairing.",
                  "critical")
    sup = {g: n for g, n in metrics["suppressed_groups_income"].items()}
    if sup:
        worst = max(sup.items(), key=lambda kv: kv[1])
        ctx.limit(
            f"Small groups get no verdict at all: {len(sup)} RAC1P/SEX groups suppressed in at least one state-year; "
            f"{worst[0]} suppressed in {worst[1]} of {len(analysis)} state-years (income task)",
            str(sup) + f"; parity results inconclusive by size: "
            + ", ".join(f"{k}: {v['inconclusive']}/{v['results']} (half-width {v['median_halfwidth']:.3f})" for k, v in bands.items()),
            "Offer multi-year or multi-region pooling for suppressed groups (with the pooling stated in the result) and "
            "say in the report which population share is left without a verdict.",
            "minor")
    return metrics
