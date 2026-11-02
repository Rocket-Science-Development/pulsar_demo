"""FA-04: bias-injection power curves at 1,000 simulations per cell (rerun of validation E5 at overnight scale).

Same simulators as E5, called unchanged:
- fixed:      pulsar_demo/ethics_validation/power_curves.fixed  (FairnessAnalyzer, parity difference, zone [-0.1, 0.1])
- probe:      pulsar_demo/ethics_validation/power_curves.probe  (counterfactual_analysis, action rate and excess flips)
- sequential: pulsar_metrics/benchmarks/fairness_drift.run_once (FairnessMonitor vs naive repeated z-test, synthetic)

Runs are split in chunks with independent seeds over worker processes, then pooled. Every pooled rate is compared with
the E5 rate (100 runs): Wilson 95 % interval of E5 and a two-proportion test with Benjamini-Hochberg over all cells.
The E5 "explain" curve and the Folktables sequential bench need ACS Massachusetts 2018, which is not in the lab data:
they are not rerun. Synthetic data only (no real person).

    LAB_MODE=dry python -m lab.harness FA-04     # 20 runs per cell
    LAB_MODE=full python -m lab.harness FA-04    # 1,000 runs per cell
"""

import contextlib
import io
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MODE = os.environ.get("LAB_MODE", "dry")
ROOT = Path(__file__).resolve().parents[3]  # .../pulsar-ml
ETHICS = ROOT / "pulsar_demo" / "ethics_validation"
BENCH = ROOT / "pulsar_metrics" / "benchmarks"

PARAMS = {"mode": MODE, "runs": 1000 if MODE == "full" else 20, "chunks": 20 if MODE == "full" else 4,
          "workers": 6, "seed": 20261102, "experiments": ["fixed", "probe", "sequential"],
          "sequential": {"deltas": [0.0, 0.05, 0.1, 0.15, 0.2], "shares": [0.05, 0.2, 0.5], "warmup": 2000,
                         "horizon": 20000, "batch": 500, "tolerance": 0.05, "alpha": 0.05}}

CRITERIA = {"false_alarm_max": ("<=", 0.05), "bh_significant_differences": ("==", 0)}


def _paths():
    for p in (str(ETHICS), str(BENCH)):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------- workers
def fixed_chunk(args):
    runs, seed = args
    _paths()
    import power_curves

    with contextlib.redirect_stdout(io.StringIO()):
        return power_curves.fixed(runs, seed)["cells"]


def probe_chunk(args):
    runs, seed = args
    _paths()
    import power_curves

    with contextlib.redirect_stdout(io.StringIO()):
        return power_curves.probe(runs, seed)["cells"]


def sequential_chunk(args):
    runs, seed, s = args
    _paths()
    import fairness_drift

    sample = fairness_drift.Synthetic()
    rng = np.random.default_rng(seed)
    out = []
    for share in s["shares"]:
        for delta in s["deltas"]:
            firsts = [fairness_drift.run_once(sample, delta, share, rng, s["warmup"], s["horizon"], s["batch"],
                                              s["tolerance"], s["alpha"]) for _ in range(runs)]
            out.append({"share": share, "delta": delta, "firsts": firsts})
    return out


# ---------------------------------------------------------------- pooling and statistics
def _pool_rates(chunks, keys, nested=None):
    """Average the rates of equally sized chunks, cell by cell"""
    cells = []
    for i, cell in enumerate(chunks[0]):
        merged = {k: cell[k] for k in keys}
        for part in nested or [None]:
            src = [c[i][part] if part else c[i] for c in chunks]
            pooled = {r: float(np.mean([x[r] for x in src])) for r in ("established", "within", "inconclusive")}
            if part:
                merged[part] = pooled
            else:
                merged.update(pooled)
        cells.append(merged)
    return cells


def wilson(p, n, z=1.959964):
    if n == 0:
        return (0.0, 1.0)
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_prop_p(p1, n1, p2, n2):
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs(p1 - p2) / se
    return math.erfc(z / math.sqrt(2))


def bh(pvalues, q=0.05):
    m = len(pvalues)
    order = np.argsort(pvalues)
    passed = np.zeros(m, dtype=bool)
    k_max = 0
    for rank, idx in enumerate(order, 1):
        if pvalues[idx] <= q * rank / m:
            k_max = rank
    passed[order[:k_max]] = True
    return passed


def compare(new_cells, e5_cells, key_fields, rate_paths, n_new, n_e5):
    rows = []
    index = {tuple(c[k] for k in key_fields): c for c in e5_cells}
    for c in new_cells:
        ref = index.get(tuple(c[k] for k in key_fields))
        if ref is None:
            continue
        for path in rate_paths:
            a, b = c, ref
            for part in path:
                a, b = a[part], b[part]
            lo, hi = wilson(b, n_e5)
            rows.append({"cell": {k: c[k] for k in key_fields}, "rate": ".".join(path), "new": round(a, 4),
                         "e5": round(b, 4), "e5_wilson": [round(lo, 4), round(hi, 4)], "inside": lo <= a <= hi,
                         "p": two_prop_p(a, n_new, b, n_e5)})
    return rows


# ---------------------------------------------------------------- run
def run(ctx):
    p = ctx.params
    runs, chunks = p["runs"], p["chunks"]
    per_chunk = runs // chunks
    runs = per_chunk * chunks
    e5 = {name: json.loads((ETHICS / "results" / f).read_text(encoding="utf-8"))
          for name, f in (("fixed", "power_fixed.json"), ("probe", "power_probe.json"),
                          ("sequential", "bench_synthetic.json"))}
    ctx.note("E5 explain curve and Folktables sequential bench not rerun: they need ACS MA 2018, absent from lab data.")
    metrics, seconds, comparisons = {"runs_per_cell": runs}, {}, []
    ctx_mp = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=p["workers"], mp_context=ctx_mp) as pool:
        base = p["seed"]
        t0 = time.perf_counter()
        fixed_parts = list(pool.map(fixed_chunk, [(per_chunk, base + i) for i in range(chunks)]))
        seconds["fixed"] = time.perf_counter() - t0
        print(f"fixed done in {seconds['fixed']:.0f} s", flush=True)
        t0 = time.perf_counter()
        probe_parts = list(pool.map(probe_chunk, [(per_chunk, base + 100 + i) for i in range(chunks)]))
        seconds["probe"] = time.perf_counter() - t0
        print(f"probe done in {seconds['probe']:.0f} s", flush=True)
        t0 = time.perf_counter()
        seq_parts = list(pool.map(sequential_chunk, [(per_chunk, base + 200 + i, p["sequential"]) for i in range(chunks)]))
        seconds["sequential"] = time.perf_counter() - t0
        print(f"sequential done in {seconds['sequential']:.0f} s", flush=True)

    _paths()
    import fairness_drift

    fixed_cells = _pool_rates(fixed_parts, ["size", "delta"])
    probe_cells = _pool_rates(probe_parts, ["sets", "delta"], nested=["action", "excess_flip"])
    seq_cells = []
    for i, cell in enumerate(seq_parts[0]):
        firsts = [f for part in seq_parts for f in part[i]["firsts"]]
        seq_cells.append({"share": cell["share"], "delta": cell["delta"],
                          **fairness_drift.summarise(firsts, p["sequential"]["warmup"])})

    comparisons += [dict(r, experiment="fixed") for r in compare(
        fixed_cells, e5["fixed"]["cells"], ["size", "delta"], [("established",), ("within",)], runs, e5["fixed"]["runs"])]
    comparisons += [dict(r, experiment="probe") for r in compare(
        probe_cells, e5["probe"]["cells"], ["sets", "delta"],
        [("action", "established"), ("action", "within"), ("excess_flip", "established"), ("excess_flip", "within")],
        runs, e5["probe"]["runs"])]
    comparisons += [dict(r, experiment="sequential") for r in compare(
        seq_cells, e5["sequential"]["cells"], ["share", "delta"],
        [("monitor", "alarm_rate"), ("monitor", "power"), ("naive", "alarm_rate"), ("naive", "power")],
        runs, e5["sequential"]["settings"]["runs"])]
    significant = bh([c["p"] for c in comparisons]) if comparisons else []
    for c, s in zip(comparisons, significant):
        c["bh_significant"] = bool(s)

    # false alarms of Pulsar modules (bias 0, or delta at the tolerance of the sequential monitor)
    fa = []
    for c in fixed_cells:
        if c["delta"] == 0.0:
            fa.append(("fixed", f"size {c['size']}", c["established"]))
    for c in probe_cells:
        if c["delta"] == 0.0:
            fa.append(("probe action", f"sets {c['sets']}", c["action"]["established"]))
            fa.append(("probe excess_flip", f"sets {c['sets']}", c["excess_flip"]["established"]))
    tol = p["sequential"]["tolerance"]
    for c in seq_cells:
        if c["delta"] <= tol + 1e-12:
            fa.append(("sequential monitor", f"share {c['share']} delta {c['delta']}", c["monitor"]["alarm_rate"]))
        fa.append(("sequential monitor before injection", f"share {c['share']} delta {c['delta']}",
                   c["monitor"]["alarm_before_injection"]))
    boundary = [("fixed at zone bound (delta 0.1)", f"size {c['size']}", c["established"])
                for c in fixed_cells if abs(c["delta"] - 0.1) < 1e-12]
    naive_fa = [("naive repeated z-test", f"share {c['share']} delta {c['delta']}", c["naive"]["alarm_rate"])
                for c in seq_cells if c["delta"] <= tol + 1e-12]

    metrics.update({
        "seconds": {k: round(v, 1) for k, v in seconds.items()},
        "false_alarm_max": max(x[2] for x in fa),
        "false_alarms": [{"module": m, "cell": c, "rate": round(r, 4), "wilson95": [round(v, 4) for v in wilson(r, runs)]}
                         for m, c, r in fa],
        "boundary_false_alarms": [{"module": m, "cell": c, "rate": round(r, 4)} for m, c, r in boundary],
        "naive_false_alarms": [{"cell": c, "rate": round(r, 4)} for _, c, r in naive_fa],
        "comparisons": len(comparisons),
        "share_within_e5_wilson": sum(c["inside"] for c in comparisons) / len(comparisons) if comparisons else None,
        "bh_significant_differences": int(sum(significant)) if comparisons else None,
        "fixed_cells": fixed_cells, "probe_cells": probe_cells, "sequential_cells": seq_cells,
    })
    (ctx.workdir / "comparisons_e5.json").write_text(json.dumps(comparisons, indent=1), encoding="utf-8")

    # ------------------------------------------------ limits from the numbers
    worst = max(fa, key=lambda x: x[2])
    if worst[2] > 0.05:
        ctx.limit(f"False alarm rate above 5 %: {worst[0]} {worst[1]} raises {worst[2]:.1%} at no bias",
                  f"{runs} runs; Wilson 95 % {wilson(worst[2], runs)}", "Recalibrate the interval / e-process.",
                  "critical")
    diffs = [c for c in comparisons if c["bh_significant"]]
    if diffs:
        ctx.limit(f"{len(diffs)} of {len(comparisons)} curve points differ from E5 beyond sampling noise (BH 5 %)",
                  "; ".join(f"{d['experiment']} {d['cell']} {d['rate']}: {d['new']} vs E5 {d['e5']}" for d in diffs[:8]),
                  "Re-baseline E5 at 1,000 runs and pin seeds and package versions in its results.", "major")
    # at the zone bound the true disparity equals the bound: a 95 % two-sided interval should exclude it <= 2.5 %
    over = [(c, r) for _, c, r in boundary if wilson(r, runs)[0] > 0.025]
    if over:
        ctx.limit(
            f"Percentile bootstrap is anti-conservative for small groups: at a true disparity equal to the zone bound "
            f"(0.1), 'established' is raised in up to {max(r for _, r in over):.1%} of runs (nominal 2.5 %)",
            "; ".join(f"{c}: {r:.1%} (Wilson 95 % {wilson(r, runs)[0]:.1%}-{wilson(r, runs)[1]:.1%})" for c, r in over)
            + f"; all sizes: " + ", ".join(f"{c}: {r:.1%}" for _, c, r in boundary),
            "Use a BCa or studentised bootstrap (or a Newcombe / Wilson difference interval for rates) for groups "
            "under ~200 rows, and raise the default n_bootstrap; add this bound test to the fairness test suite.",
            "minor")
    # certifying fairness: "within" verdict at no bias, by group size
    within0 = {c["size"]: c["within"] for c in fixed_cells if c["delta"] == 0.0}
    small = [s for s, w in within0.items() if w < 0.5]
    if small:
        ctx.limit(
            f"With no bias at all, FairnessAnalyzer concludes 'within the zone' in under half the runs for groups of "
            f"{min(small)}-{max(small)} rows (reference 2,000 rows, zone +/-0.1): the result stays 'inconclusive'",
            "within rate at bias 0 by group size: " + ", ".join(f"{s}: {w:.1%}" for s, w in sorted(within0.items())),
            "Publish the minimum group size needed to certify (equivalence) for the chosen zone next to each verdict; "
            "offer a TOST-style equivalence test with an explicit margin and a power hint in the result.",
            "minor")
    probe0 = {c["sets"]: c["action"]["within"] for c in probe_cells if c["delta"] == 0.0}
    small_p = [s for s, w in probe0.items() if w < 0.5]
    if small_p:
        ctx.limit(
            f"Counterfactual probe: with no bias, the action-rate difference is concluded 'within' in under half the "
            f"runs up to {max(small_p)} probe sets",
            "within rate at bias 0 by sets: " + ", ".join(f"{s}: {w:.1%}" for s, w in sorted(probe0.items())),
            "Recommend a probe budget (sets) per tolerance in the probe documentation and return it with the result.",
            "minor")
    for c in seq_cells:
        m, n = c["monitor"], c["naive"]
        if c["delta"] > tol and m["power"] < 0.5 and n["power"] >= 0.8:
            ctx.limit(
                f"Anytime-valid monitor: power {m['power']:.1%} vs {n['power']:.1%} for the naive repeated test, "
                f"for a drop of {c['delta']} on a group at {c['share']:.0%} of the stream over "
                f"{p['sequential']['horizon']} decisions",
                f"monitor median delay {m['median_delay']}, naive {n['median_delay']}; naive false alarms at no bias "
                f"up to {max(x[2] for x in naive_fa):.1%} vs monitor {max(x[2] for x in fa if x[0] == 'sequential monitor'):.1%}",
                "Expected price of validity; document horizon / share needed per delta, and offer a tuned betting "
                "strategy (mixture or aGRAPA bets, tolerance-aware prior) to recover power for small groups.",
                "minor")
    return metrics
