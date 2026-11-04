"""Offline replay of LR-03: false-trigger rate and detection delay of the "measured drift" trigger.

The governance layer re-evaluates a compliance control when Pulsar reports a measured drift of the RAG agent. Each
false alarm flips a control and sends a reviewer a bogus alert, so this script measures, from the 1,799 recorded runs
of LR-03 (lab/results/LR-03/runs.jsonl, no agent call), for each candidate signal and persistence k (DriftPersistence,
an alert after k consecutive windows):

- the A/A false-trigger rate per window, with its Wilson interval, and per week assuming one window per hour (168);
- the detection delay, in windows, at each level of corpus removal (10 to 80 %, total failure).

Design (runs are the exchangeable unit: every question of LR-03 was asked once, in random order)
- A/A: each draw splits the 700 runs at 0 % into a reference of 300 and 3 disjoint windows of 100 from the other
  400, i.e. what a deployment sees: a fixed reference and successive windows. Trigger at persistence k = the first k
  windows all reject. The draws are independent, so the Wilson interval is exact for the per-window rate.
- Degradation: each draw takes a reference of 300 runs at 0 % and a window of 100 runs from the level's pool (200 runs,
  100 for total failure): per-window power, Wilson interval. The delay feeds DriftPersistence with sequences of window
  verdicts drawn independently from those draws (the pools are too small for long disjoint sequences); it counts from
  the first full window after the change.

Signals (window of 100 runs against the reference)
    faithfulness       one-sided Mann-Whitney, window lower (as LR-03)
    empty_retrieval    one-sided Fisher exact test, window higher (empty search or no search)
    relevance_cos      one-sided Mann-Whitney on the cosine question / top passage (granite-embedding), window lower
    relevance_bm25     same on the retriever's own score of the top passage (BM25; its scale also moves with the
                       index statistics, so it reacts to any change of the corpus)
    mmd_rows_<e>       pulsar_metrics mmd_test, one row per retrieved passage permuted as independent rows (LR-03)
    mmd_groups_<e>     same rows, whole runs permuted (mmd_test(groups=...), pulsar_metrics >= this change)
    mmd_top1_<e>       one row per run (its top passage)
    trigger_cos        faithfulness, empty_retrieval or relevance_cos: smallest p-value x 3 (Bonferroni)
    trigger_score      faithfulness, empty_retrieval or relevance_bm25, same
    retrieval_score    empty_retrieval or relevance_bm25, smallest p-value x 2

Every summary is given at alpha 0.05 and 0.01. The raw p-values of every window are kept in
lab/results/fiabilite_declencheur_pvalues.json: --from-pvalues recomputes the summaries without any test.

Embeddings of the 761 PIAF passages come from the local Ollama (nomic-embed-text, granite-embedding:278m), cached in
lab/results/LR-03/passage_embeddings.npz; the corpus (AgentPublic/piaf, MIT) is needed only to build the cache.
Run from pulsar_demo, with pulsar_metrics (with mmd_test(groups=)) importable:

    python -m lab.replay.declencheur_derive                          # full replay, about 55 min on 16 CPUs
    python -m lab.replay.declencheur_derive --draws 64 --level-draws 32 --levels 60   # quick check, 2 min
    python -m lab.replay.declencheur_derive --from-pvalues           # summaries only, from the saved p-values
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one process per CPU, one thread each

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "LR-03" / "runs.jsonl"
CACHE = ROOT / "results" / "LR-03" / "passage_embeddings.npz"
OUT = ROOT / "results" / "fiabilite_declencheur.json"
PVALUES = ROOT / "results" / "fiabilite_declencheur_pvalues.json"
CORPUS = (
    Path(os.environ.get("LAB_DATA", "/data/lab/datasets"))
    / "AgentPublic"
    / "piaf"
    / "plain_text"
    / "train-00000-of-00001.parquet"
)
EMBEDDERS = ["nomic-embed-text", "granite-embedding:278m"]
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
ALPHAS = (0.05, 0.01)
COMPOSITES = {
    "trigger_cos": ("faithfulness", "empty_retrieval", "relevance_cos"),
    "trigger_score": ("faithfulness", "empty_retrieval", "relevance_bm25"),
    "retrieval_score": ("empty_retrieval", "relevance_bm25"),
}
REFERENCE, WINDOW, AA_WINDOWS = 300, 100, 3
WINDOWS_PER_WEEK = 168
PERSISTENCE = (1, 2, 3)
MAX_DELAY = 24  # windows simulated after the change
SEQUENCES = 1000  # sequences simulated per level for the delay


# ------------------------------------------------------------------------------------------------ data
def load_runs(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def passage_embeddings(cache: Path, corpus: Path) -> dict:
    """{embedder: {passage id: unit vector}}; the ids are those of LR-03 (P0000.. over the sorted unique contexts)"""
    if not cache.exists():
        import pandas as pd
        import requests

        contexts = sorted(pd.read_parquet(corpus)["context"].unique())
        arrays = {}
        for model in EMBEDDERS:
            out = []
            for i in range(0, len(contexts), 64):
                r = requests.post(f"{OLLAMA}/api/embed", json={"model": model, "input": contexts[i : i + 64]}, timeout=900)
                r.raise_for_status()
                out += r.json()["embeddings"]
            e = np.asarray(out, dtype=float)
            arrays[model.split(":")[0]] = (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float32)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, pids=np.array([f"P{i:04d}" for i in range(len(contexts))]), **arrays)
    data = np.load(cache)
    pids = [str(p) for p in data["pids"]]
    return {name: dict(zip(pids, data[name].astype(float))) for name in data.files if name != "pids"}


# ------------------------------------------------------------------------------------------------ window tests
def _clean(values) -> list:
    return [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]


def _mw_lower(win, ref):
    w, r = _clean(win), _clean(ref)
    if len(w) < 5 or len(r) < 5:
        return None
    return float(stats.mannwhitneyu(w, r, alternative="less").pvalue)


def _fisher_higher(win, ref):
    a, b = sum(bool(x) for x in win), sum(bool(x) for x in ref)
    return float(stats.fisher_exact([[a, len(win) - a], [b, len(ref) - b]], alternative="greater").pvalue)


def _mmd(x, y, perms, seed, groups=None):
    from pulsar_metrics.genai import mmd_test

    if len(x) < 5 or len(y) < 5:
        return None
    return float(mmd_test(np.asarray(x), np.asarray(y), n_permutations=perms, seed=seed, groups=groups).p_value)


def window_pvalues(ref, win, emb, perms, seed, mmd=True) -> dict:
    empty = [r["empty"] or r["no_search"] for r in win], [r["empty"] or r["no_search"] for r in ref]
    out = {
        "faithfulness": _mw_lower([r["faithfulness"] for r in win], [r["faithfulness"] for r in ref]),
        "empty_retrieval": _fisher_higher(*empty),
        "relevance_cos": _mw_lower([r.get("qp_cos") for r in win], [r.get("qp_cos") for r in ref]),
        "relevance_bm25": _mw_lower([r.get("top1_bm25") for r in win], [r.get("top1_bm25") for r in ref]),
    }
    if mmd:
        for name, table in emb.items():
            rx = [r for r in ref if r["retrieved"]]
            ry = [r for r in win if r["retrieved"]]
            x = [table[p] for r in rx for p in r["retrieved"]]
            y = [table[p] for r in ry for p in r["retrieved"]]
            gx = [i for i, r in enumerate(rx) for _ in r["retrieved"]]
            gy = [i for i, r in enumerate(ry) for _ in r["retrieved"]]
            out[f"mmd_rows_{name}"] = _mmd(x, y, perms, seed)
            out[f"mmd_groups_{name}"] = _mmd(x, y, perms, seed, groups=(gx, gy)) if len(rx) > 1 and len(ry) > 1 else None
            out[f"mmd_top1_{name}"] = _mmd(
                [table[r["retrieved"][0]] for r in rx], [table[r["retrieved"][0]] for r in ry], perms, seed
            )
    return out


def add_composites(pvalues: dict) -> dict:
    """Composite triggers: the smallest p-value of their signals, Bonferroni-corrected (None when none is computable)"""
    out = dict(pvalues)
    for name, members in COMPOSITES.items():
        values = [pvalues.get(m) for m in members if pvalues.get(m) is not None]
        out[name] = min(1.0, len(members) * min(values)) if values else None
    return out


# ------------------------------------------------------------------------------------------------ draws
_STATE: dict = {}


def _init(runs, emb, perms):
    _STATE.update(runs=runs, emb=emb, perms=perms)


def _aa_draw(seed: int) -> list:
    """Reference of 300 and 3 disjoint windows of 100 from the 700 runs at 0 %"""
    runs = _STATE["runs"]
    zero = [r for r in runs if r["label"] in ("reference", "aa")]
    perm = np.random.default_rng(seed).permutation(len(zero))
    ref = [zero[i] for i in perm[:REFERENCE]]
    out = []
    for j in range(AA_WINDOWS):
        win = [zero[i] for i in perm[REFERENCE + j * WINDOW : REFERENCE + (j + 1) * WINDOW]]
        out.append(window_pvalues(ref, win, _STATE["emb"], _STATE["perms"], seed * 10 + j))
    return out


def _level_draw(args) -> dict:
    level, seed = args
    runs = _STATE["runs"]
    zero = [r for r in runs if r["label"] in ("reference", "aa")]
    pool = [r for r in runs if _level_key(r) == level]
    rng = np.random.default_rng(10_000 + seed)
    ref = [zero[i] for i in rng.choice(len(zero), REFERENCE, replace=False)]
    win = [pool[i] for i in rng.choice(len(pool), min(WINDOW, len(pool)), replace=False)]
    return window_pvalues(ref, win, _STATE["emb"], _STATE["perms"], seed)


def _level_key(row) -> str:
    return "total" if row["label"] == "total_failure" else str(round(100 * row["level"]))


# ------------------------------------------------------------------------------------------------ summaries
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def aa_summary(draws: list, signals: list, alpha: float) -> dict:
    """Per signal and persistence k: share of draws whose first k windows all reject, Wilson, and per week"""
    out = {}
    for s in signals:
        out[s] = {}
        for k in PERSISTENCE:
            flags = [all(w[s] < alpha for w in d[:k]) for d in draws if all(w.get(s) is not None for w in d[:k])]
            n, hits = len(flags), int(sum(flags))
            lo, hi = wilson(hits, n)
            rate = hits / n if n else None
            out[s][f"k{k}"] = {
                "rate": round(rate, 4) if rate is not None else None,
                "ci": [lo, hi],
                "n": n,
                "hits": hits,
                # upper bound of new alerts per week: every window ending a streak of k rejections counts
                "per_week": round(WINDOWS_PER_WEEK * rate, 2) if rate is not None else None,
                "per_week_ci": [round(WINDOWS_PER_WEEK * lo, 2), round(WINDOWS_PER_WEEK * hi, 2)] if n else None,
            }
    return out


def delay_summary(draws: list, signal: str, alpha: float, seed: int) -> dict:
    """Per-window power with Wilson, and the delay (in windows) of DriftPersistence(k) on sequences of independent
    window verdicts drawn from the measured ones"""
    from pulsar_metrics.analyzers.persistence import DriftPersistence

    verdicts = np.array([d[signal] < alpha for d in draws if d.get(signal) is not None])
    n, hits = len(verdicts), int(verdicts.sum())
    out = {"power": round(hits / n, 4) if n else None, "power_ci": list(wilson(hits, n)), "n": n}
    rng = np.random.default_rng(seed)
    for k in PERSISTENCE:
        delays = []
        for _ in range(SEQUENCES if n else 0):
            tracker = DriftPersistence(windows=k)
            delay = None
            for t, flag in enumerate(rng.choice(verdicts, MAX_DELAY), start=1):
                record = {"metric_type": "drift", "metric_name": signal, "feature_name": None, "drift_status": bool(flag)}
                if tracker.update([record])[0].confirmed_drift:
                    delay = t
                    break
            delays.append(delay)
        found = [d for d in delays if d is not None]
        out[f"k{k}"] = {
            "median_delay": float(np.median(found)) if len(found) > len(delays) / 2 else None,
            "p90_delay": float(np.quantile(found, 0.9)) if len(found) > 0.9 * len(delays) else None,
            "detected_within_6": round(sum(d is not None and d <= 6 for d in delays) / len(delays), 4) if delays else None,
            "detected_within_24": round(len(found) / len(delays), 4) if delays else None,
        }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--pvalues", type=Path, default=PVALUES, help="raw p-values of every window (written, or read back)")
    parser.add_argument("--from-pvalues", action="store_true", help="recompute the summaries from --pvalues, no test")
    parser.add_argument("--draws", type=int, default=2000, help="A/A draws (3 windows each)")
    parser.add_argument("--level-draws", type=int, default=400, help="draws per degradation level")
    parser.add_argument("--levels", nargs="*", default=["10", "20", "40", "60", "80", "total"])
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    if args.from_pvalues:
        raw = json.loads(args.pvalues.read_text(encoding="utf-8"))
    else:
        runs = load_runs(args.runs)
        emb = passage_embeddings(args.cache, args.corpus)
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(runs, emb, args.permutations)) as pool:
            aa = list(pool.map(_aa_draw, range(args.draws), chunksize=8))
            levels = {
                lv: list(pool.map(_level_draw, [(lv, s) for s in range(args.level_draws)], chunksize=8)) for lv in args.levels
            }
        design = {
            "source": str(args.runs.name),
            "reference_runs": REFERENCE,
            "window_runs": WINDOW,
            "aa_draws": args.draws,
            "aa_windows_per_draw": AA_WINDOWS,
            "level_draws": args.level_draws,
            "permutations": args.permutations,
            "windows_per_week": WINDOWS_PER_WEEK,
            "embedders": sorted(emb),
        }
        raw = {"design": design, "aa": aa, "levels": levels}
        args.pvalues.parent.mkdir(parents=True, exist_ok=True)
        args.pvalues.write_text(json.dumps(raw), encoding="utf-8")
    aa = [[add_composites(w) for w in d] for d in raw["aa"]]
    levels = {lv: [add_composites(w) for w in d] for lv, d in raw["levels"].items()}
    signals = sorted(aa[0][0])
    result = {
        "design": {**raw["design"], "alphas": list(ALPHAS), "composites": COMPOSITES},
        "aa": {str(a): aa_summary(aa, signals, a) for a in ALPHAS},
        "levels": {
            str(a): {lv: {s: delay_summary(d, s, a, seed=i) for i, s in enumerate(signals)} for lv, d in levels.items()}
            for a in ALPHAS
        },
        "seconds": round(time.perf_counter() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{args.out} written in {result['seconds']} s")
    for a in ALPHAS:
        for s in signals:
            row = result["aa"][str(a)][s]
            print(
                f"alpha {a} {s:28s} A/A k1 {row['k1']['rate']} {row['k1']['ci']}  k2 {row['k2']['rate']}  k3 {row['k3']['rate']}"
            )
    return result


if __name__ == "__main__":
    main()
