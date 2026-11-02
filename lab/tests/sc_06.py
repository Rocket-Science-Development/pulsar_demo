"""SC-06 - Analysis at scale: hundreds of models per window.

Per model and 6-hour window (synthetic, 20 features, current 2,000 rows, reference 5,000 rows):
A/A calibration (AACalibrator, 200 draws) of KS on 20 features, Analyzer run with calibrated options, empirical p-values
+ Benjamini-Hochberg, DriftPersistence; FairnessAnalyzer (2 attributes + intersection, bootstrap 500);
AttributionDrift (20 features, bootstrap 200). Timed per stage.

1) cost per model and per stage on a sample, one process; 2) fixed workload (32 models) on 1-32 processes (scaling);
3) the full window of 300 models on 16 processes (criterion: < 10 min).
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

N_MODELS = int(os.environ.get("LAB_MODELS", 300))
SCALING = [int(v) for v in os.environ.get("LAB_PROCS", "1,2,4,8,16,32").split(",")]
SCALING_WORK = int(os.environ.get("LAB_SCALING_WORK", 32))
FULL_PROCS = int(os.environ.get("LAB_FULL_PROCS", 16))
N_CUR, N_REF, FEATURES = 2000, 5000, 20
PARAMS = {"models": N_MODELS, "features": FEATURES, "current_rows": N_CUR, "reference_rows": N_REF,
          "aa_draws": 200, "fairness_bootstrap": 500, "attribution_bootstrap": 200, "processes": SCALING,
          "scaling_workload_models": SCALING_WORK, "full_window_processes": FULL_PROCS, "host": "c7i.8xlarge (32 vCPU / 16 cores)"}
CRITERIA = {"full_window_minutes": ("<", 10.0), "efficiency_at_16": (">=", 0.7)}


def analyse_model(i: int) -> dict:
    import contextlib
    import io
    import warnings

    import resource

    from pulsar_metrics.analyzers.base import Analyzer
    from pulsar_metrics.analyzers.persistence import DriftPersistence
    from pulsar_metrics.calibration import AACalibrator
    from pulsar_metrics.explain.drift import AttributionDrift
    from pulsar_metrics.fairness.analyzer import FairnessAnalyzer, FairnessPolicy
    from pulsar_metrics.multiple_testing import fdr_correct

    warnings.simplefilter("ignore")
    rng = np.random.default_rng(i)
    feats = [f"f{j}" for j in range(FEATURES)]
    mid = f"model{i:03d}"
    ref = pd.DataFrame(rng.normal(size=(N_REF, FEATURES)), columns=feats)
    cur = pd.DataFrame(rng.normal(size=(N_CUR, FEATURES)), columns=feats)
    if i % 3 == 0:
        cur["f0"] += 0.3  # a third of the models drift on one feature
    now = pd.Timestamp.now(tz="UTC")
    meta = {"model_id": mid, "model_version": "1"}
    ref = ref.assign(**meta, pred_timestamp=now)
    cur = cur.assign(**meta, pred_timestamp=pd.date_range(now, periods=N_CUR, freq="10s"))
    t = {}
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        s = time.perf_counter()
        an = Analyzer(name="w", model_id=mid, model_version="1")
        an.add_drift_metrics(metrics_list=["ks_2samp"], features_list=feats)
        cal = AACalibrator(window_size=N_CUR, n_draws=200, contiguous=False).fit(ref, an._metrics_list)
        t["calibration"] = time.perf_counter() - s
        s = time.perf_counter()
        an.run(current=cur, reference=ref, options=cal.options())
        res = an.results_to_pandas().query("metric_type == 'drift'").copy()
        res["p_empirical"] = cal.empirical_pvalues(res)
        res = fdr_correct(res, "p_empirical")
        res["drift_status"] = res["drift_fdr"].astype(bool)
        tracker = DriftPersistence(windows=2)
        tracker.update(res)
        confirmed = tracker.update(res)
        t["drift_bh_persistence"] = time.perf_counter() - s
        s = time.perf_counter()
        fair = pd.DataFrame({"y_pred": rng.integers(0, 2, N_CUR), "y_true": rng.integers(0, 2, N_CUR),
                             "y_pred_proba": rng.random(N_CUR), "sex": rng.choice(["f", "m"], N_CUR),
                             "region": rng.choice(["a", "b", "c", "d", "e"], N_CUR)})
        fa = FairnessAnalyzer(["sex", "region"], FairnessPolicy(n_bootstrap=500), intersections=True, seed=i)
        fres = fa.run(fair)
        t["fairness"] = time.perf_counter() - s
        s = time.perf_counter()
        scale = np.linspace(1, 0.05, FEATURES)
        ad = AttributionDrift(k=5, n_bootstrap=200, seed=i).fit(
            pd.DataFrame(rng.normal(size=(N_REF, FEATURES)) * scale, columns=feats))
        ares = ad.run(pd.DataFrame(rng.normal(size=(N_CUR, FEATURES)) * scale, columns=feats))
        t["attribution"] = time.perf_counter() - s
    return {"i": i, "seconds": t, "total": sum(t.values()), "results": len(res) + len(fres) + len(ares),
            "drift_flagged": int(res["drift_status"].sum()), "confirmed": sum(1 for v in confirmed if v.confirmed_drift),
            "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def _batch(models, procs):
    t0 = time.perf_counter()
    if procs == 1:
        out = [analyse_model(i) for i in models]
    else:
        with ProcessPoolExecutor(max_workers=procs) as ex:
            out = list(ex.map(analyse_model, models, chunksize=1))
    return time.perf_counter() - t0, out


def run(ctx):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    # 1) cost per model and stage
    wall, sample = _batch(list(range(8)), 1)
    stages = {k: float(np.mean([r["seconds"][k] for r in sample])) for k in sample[0]["seconds"]}
    per_model = float(np.mean([r["total"] for r in sample]))
    # 2) scaling on a fixed workload
    scaling = {}
    for p in SCALING:
        w, out = _batch(list(range(1000, 1000 + SCALING_WORK)), p)
        scaling[p] = {"wall_s": w, "models_per_min": SCALING_WORK / w * 60,
                      "maxrss_mb_per_worker": max(r["maxrss_mb"] for r in out)}
    base = scaling.get(1, {}).get("wall_s")
    for p, v in scaling.items():
        v["speedup"] = base / v["wall_s"] if base else None
        v["efficiency"] = v["speedup"] / p if base else None
    # 3) full window
    wf, full = _batch(list(range(N_MODELS)), FULL_PROCS)
    results_written = sum(r["results"] for r in full)
    metrics = {"per_model_s_one_process": per_model, "stage_seconds_mean": stages,
               "stage_share": {k: v / per_model for k, v in stages.items()},
               "scaling": {str(k): v for k, v in scaling.items()},
               "efficiency_at_16": scaling.get(16, {}).get("efficiency"),
               "full_window_minutes": wf / 60, "full_window_models": N_MODELS, "full_window_procs": FULL_PROCS,
               "results_per_s": results_written / wf, "results_per_window": results_written,
               "cpu_seconds_per_model": float(np.mean([r["total"] for r in full])),
               "maxrss_mb_per_worker": max(r["maxrss_mb"] for r in full),
               "models_drifting": sum(1 for i in range(N_MODELS) if i % 3 == 0),
               "models_flagged": sum(1 for r in full if r["drift_flagged"]),
               "projected_sequential_minutes_300": per_model * 300 / 60}
    (ctx.workdir / "full.json").write_text(json.dumps(full))
    top = max(stages, key=stages.get)
    if metrics["full_window_minutes"] >= 10:
        ctx.limit(f"A 6-hour window of {N_MODELS} models takes {metrics['full_window_minutes']:.1f} min on {FULL_PROCS} "
                  f"processes (target < 10 min)",
                  f"{per_model:.1f} s per model in one process; stage shares "
                  + json.dumps({k: round(v, 2) for k, v in metrics['stage_share'].items()}),
                  f"The dominant stage is {top}: reuse the A/A null distributions across windows (recalibrate only "
                  "when the reference changes), vectorise KS over draws, cap bootstrap replicates adaptively", "major")
    if metrics["stage_share"][top] > 0.5:
        ctx.limit(f"{top} dominates the analysis cost: {metrics['stage_share'][top]:.0%} of {per_model:.1f} s per model",
                  json.dumps({k: round(v, 2) for k, v in stages.items()}),
                  "Cache the calibration per (model, reference version) - it is recomputed every window in this "
                  "workflow because AACalibrator has no persistence; add save/load of null distributions", "minor")
    for p, v in scaling.items():
        if p <= 16 and v["efficiency"] is not None and v["efficiency"] < 0.7:
            ctx.limit(f"Parallel efficiency {v['efficiency']:.0%} at {p} processes (speedup {v['speedup']:.1f}x)",
                      json.dumps({str(k): round(x["speedup"], 2) for k, x in scaling.items()}),
                      "Check BLAS threads per worker and memory bandwidth; batch models per worker", "minor")
    p_min, bh_first = 1 / (200 + 1), 0.05 / FEATURES
    metrics["empirical_p_min"], metrics["bh_first_threshold"] = p_min, bh_first
    if metrics["models_flagged"] < metrics["models_drifting"] * 0.9 and p_min > bh_first:
        ctx.limit(f"Calibrated drift + BH misses single-feature drift: {metrics['models_flagged']} of "
                  f"{metrics['models_drifting']} drifting models flagged (f0 shifted by 0.3 SD, n = {N_CUR})",
                  f"empirical p-values cannot go below 1/(n_draws+1) = {p_min:.4f} with 200 A/A draws, while BH over "
                  f"{FEATURES} features needs p <= {bh_first:.4f} for a single rejection; a model is flagged only when a "
                  "second feature also beats every A/A draw by chance",
                  "Require n_draws >= m/alpha (400 for 20 features at 5 %), or fit a parametric tail to the null "
                  "distribution (generalised Pareto) for p-values below 1/(n_draws+1); warn in AACalibrator/fdr_correct "
                  "when 1/(n_draws+1) > alpha/m", "major")
    elif metrics["models_flagged"] < metrics["models_drifting"] * 0.9:
        ctx.note(f"drift flagged on {metrics['models_flagged']} models of {metrics['models_drifting']} drifting")
    return metrics
