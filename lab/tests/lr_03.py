"""LR-03 - RAG degradation on a real corpus (agent): faithfulness, empty retrieval and MMD on retrieved passages.

Dataset: AgentPublic/piaf (MIT), read from /data/lab/datasets/AgentPublic/piaf (outside the repositories): 3,835
French questions written on 761 Wikipedia paragraphs. The 761 paragraphs are the agent's corpus; each question has one
relevant paragraph (the one its answer was written on). Public encyclopedic text: no personal data.

Method
- Agent: the lab agent of the demo (`LabAgent`, llama3.2:3b through Ollama, temperature 0, at most 6 steps, output
  capped) with one tool, `search_documents(query)`, over the PIAF corpus. The retriever is BM25 (k1 1.5, b 0.75,
  French stop words) returning the top 2 paragraphs that share at least one term with the query, like the demo's
  `search_documents` (word overlap). System prompt: search, then answer briefly from the passages only.
- Corpus degradation: at level p, a random p of the 761 paragraphs is removed from the index (nested sets, so the
  paragraphs removed at 20 % include those removed at 10 %). Questions are drawn uniformly, so about p of the runs
  lose their relevant paragraph. Levels 0 (A/A), 10, 20, 40, 60, 80 %, and a total failure (the index answers nothing).
  Every question is asked once (sampling without replacement), 4 concurrent workers.
- Reference: 300 runs at 0 %. Pools: 400 more runs at 0 % (A/A), 200 runs per level, 100 runs of total failure.
- Detectors, window of 100 runs against the reference, alpha 0.05:
    faithfulness   `FaithfulnessScorer.score(answer, retrieved passages)` (product defaults, NLI on GPU); one-sided
                   Mann-Whitney, window lower (the test of genai_agent/analyze_s3s4.py);
    empty retrieval  runs whose searches returned no passage; one-sided Fisher exact test, window higher;
    MMD            `pulsar_metrics.genai.mmd_test` on the embeddings of the retrieved passages (one row per passage
                   retrieved in the window), nomic-embed-text (demo default) and granite-embedding:278m (multilingual).
  Ground truth (not a detector): the gold answer appears in the agent's answer (normalised), and whether the relevant
  paragraph was retrieved. Diagnostics (not product detectors, evidence for proposals): BM25 score of the top passage,
  cosine query / top passage (granite-embedding).
- Rejection rates: the fixed reference against the disjoint windows of each pool (what a deployment sees), and a
  resampling estimate (reference of 300 drawn from the 700 runs at 0 %, window of 100 drawn from the rest or from
  the level's pool; 200 draws, 100 for MMD).

    LAB_DRY=1 LAB_RUN_TAG=dry python -m lab.harness LR-03
    python -m lab.harness LR-03
"""

import json
import math
import os
import random
import re
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import stats

from lab.tests import _llm_common as L

L.offline()

from lab.tests import _agentlab as A  # noqa: E402

DRY = os.environ.get("LAB_DRY") == "1"
PARQUET = L.DATA / "AgentPublic" / "piaf" / "plain_text" / "train-00000-of-00001.parquet"
NLI_MODEL = os.environ.get("LAB_NLI_MODEL", str(L.MODELS / "MoritzLaurer" / "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"))
LEVELS = [0.1, 0.2, 0.4, 0.6, 0.8]
SEED = 31
PARAMS = {
    "dataset": "AgentPublic/piaf (MIT)", "model": A.MODEL, "retriever": "BM25 top-2, score > 0", "dry": DRY,
    "reference_runs": 30 if DRY else 300, "aa_pool_runs": 20 if DRY else 400, "level_runs": 10 if DRY else 200,
    "total_failure_runs": 10 if DRY else 100, "window": 10 if DRY else 100, "levels": LEVELS,
    "workers": int(os.environ.get("LAB_WORKERS", "4")), "draws": 20 if DRY else 200, "mmd_draws": 10 if DRY else 100,
    "mmd_permutations": 200, "alpha": 0.05, "embedders": ["nomic-embed-text", "granite-embedding:278m"], "seed": SEED,
}
CRITERIA = {"aa_level_max": ("<=", 0.05), "total_failure_caught_by_empty_retrieval": ("==", True),
            "faithfulness_dose_spearman": ("<=", -0.9)}

PROMPT_KEY = "lab_rag"
A.SYSTEM_PROMPT[PROMPT_KEY] = (  # in-process only: the demo's prompts are unchanged
    "You are an assistant that answers questions about a corpus of encyclopedia articles. Always call "
    "search_documents first, then answer only from the passages returned, in one or two sentences. If the passages do "
    "not contain the answer, say so. Answer in the language of the user."
)
SCHEMA = A.T._schema(
    "search_documents",
    "Search the encyclopedia corpus. Returns the most relevant passages; base the answer on them.",
    {"query": {"type": "string"}}, ["query"],
)

# ------------------------------------------------------------------------------------------------ retriever
_STOP = set("""les des une est que qui dans pour par sur avec son ses aux du de la le un et en au ce cette ces il elle
ils elles ont été sont était avait fait leur leurs plus pas mais ou où se sa ne ni nous vous lui comme dont tout tous
aussi entre sous après avant lors très peu bien quel quelle quels quelles combien comment pourquoi quand quoi qu
est-ce quoi être avoir fut furent sera seront the and of""".split())
_TOKEN = re.compile(r"\w+")


def tokens(text: str) -> list:
    text = unicodedata.normalize("NFKC", text or "").lower().replace("’", "'")
    return [w for w in _TOKEN.findall(text) if len(w) > 2 and w not in _STOP and not w.isdigit() or (w.isdigit() and len(w) >= 2)]


class BM25:
    def __init__(self, docs: dict, k1: float = 1.5, b: float = 0.75):
        self.ids = list(docs)
        self.tf = [Counter(tokens(docs[i])) for i in self.ids]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=float)
        self.avg = float(self.len.mean()) if len(self.len) else 1.0
        df = Counter(w for t in self.tf for w in t)
        n = len(self.ids)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 2) -> list:
        q = set(tokens(query))
        out = []
        for j, tf in enumerate(self.tf):
            s = 0.0
            for w in q & tf.keys():
                f = tf[w]
                s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[j] / self.avg))
            if s > 0:
                out.append((s, self.ids[j]))
        out.sort(reverse=True)
        return out[:k]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"(?<=\d)[\s \xa0.](?=\d{3}\b)", "", text)  # 100 000 -> 100000
    return " ".join(re.findall(r"\w+", text))


def correct(answer: str, golds: list) -> bool:
    """Gold answer in the agent's answer: verbatim (normalised), or >= 80 % of its content words present"""
    a = f" {_norm(answer)} "
    words = set(a.split())
    for g in golds:
        ng = _norm(g)
        if not ng:
            continue
        if f" {ng} " in a:
            return True
        content = [w for w in ng.split() if w not in _STOP and len(w) > 1]
        if len(content) >= 2 and sum(w in words for w in content) / len(content) >= 0.8:
            return True
    return False


# ------------------------------------------------------------------------------------------------ runs
def run_pool(label, level, questions, index, passages, workers, log):
    """One block of agent runs against one state of the index; returns one row per run"""
    A.collector()
    agent = A.LabAgent(base_url=A.BASE_URL, model=A.MODEL, temperature=0.0, prompt_version=PROMPT_KEY)
    out = [None] * len(questions)
    done = [0]
    lock = threading.Lock()

    def one(i):
        q = questions[i]
        calls = []

        def search_documents(query: str = "", **_):
            hits = index.search(query) if index is not None else []
            calls.append({"query": query, "hits": hits})
            return [{"id": pid, "text": passages[pid]} for _, pid in hits]

        t = time.perf_counter()
        row = {"label": label, "level": level, "qid": q["id"], "pid": q["pid"], "affected": q["pid"] not in index.ids_set
               if index is not None else True}
        try:
            res = agent.run_with(q["question"], expected_tool="search_documents", schemas=[SCHEMA],
                                 functions={"search_documents": search_documents})
            retrieved = list(dict.fromkeys(pid for c in calls for _, pid in c["hits"]))
            row.update(answer=res.answer, termination=res.termination, steps=res.steps, n_search=len(calls),
                       retrieved=retrieved, empty=bool(calls) and not retrieved, no_search=not calls,
                       gold_retrieved=q["pid"] in retrieved, correct=correct(res.answer, q["answers"]),
                       top1_bm25=calls[0]["hits"][0][0] if calls and calls[0]["hits"] else None,
                       first_query=calls[0]["query"] if calls else None, error=None)
        except Exception as e:  # noqa: BLE001 - a failed run is a row too
            row.update(answer=None, error=f"{type(e).__name__}: {e}"[:300], retrieved=[], empty=None, no_search=None,
                       gold_retrieved=False, correct=False, n_search=len(calls))
        row["wall_s"] = round(time.perf_counter() - t, 3)
        out[i] = row
        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                log(f"{label}: {done[0]}/{len(questions)}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, range(len(questions))))
    return out


def unload(model):
    import requests

    base = A.BASE_URL.rsplit("/v1", 1)[0]
    try:
        requests.post(f"{base}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------------------------------------ window tests
def _mw_lower(win, ref):
    w = [x for x in win if x is not None and not (isinstance(x, float) and math.isnan(x))]
    r = [x for x in ref if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(w) < 5 or len(r) < 5:
        return None
    return float(stats.mannwhitneyu(w, r, alternative="less").pvalue)


def _fisher_higher(win, ref):
    a, b = sum(bool(x) for x in win), sum(bool(x) for x in ref)
    return float(stats.fisher_exact([[a, len(win) - a], [b, len(ref) - b]], alternative="greater").pvalue)


def _fisher_lower(win, ref):
    a, b = sum(bool(x) for x in win), sum(bool(x) for x in ref)
    return float(stats.fisher_exact([[a, len(win) - a], [b, len(ref) - b]], alternative="less").pvalue)


def window_pvalues(ref, win, emb, mmd=True, perms=200, seed=0):
    from pulsar_metrics.genai import mmd_test

    out = {
        "faithfulness": _mw_lower([r["faithfulness"] for r in win], [r["faithfulness"] for r in ref]),
        "empty_retrieval": _fisher_higher([r["empty"] or r["no_search"] for r in win],
                                          [r["empty"] or r["no_search"] for r in ref]),
        "diag_top1_bm25": _mw_lower([r.get("top1_bm25") for r in win], [r.get("top1_bm25") for r in ref]),
        "diag_query_passage_cos": _mw_lower([r.get("qp_cos") for r in win], [r.get("qp_cos") for r in ref]),
        "truth_correct": _fisher_lower([r["correct"] for r in win], [r["correct"] for r in ref]),
    }
    if mmd:
        for name, table in emb.items():
            x = [table[p] for r in ref for p in r["retrieved"]]
            y = [table[p] for r in win for p in r["retrieved"]]
            out[f"mmd_{name}"] = (mmd_test(np.asarray(x), np.asarray(y), n_permutations=perms, seed=seed).p_value
                                  if len(x) >= 5 and len(y) >= 5 else None)
            # one row per run (its top passage): the exchangeable unit, since a run's passages are not independent
            x1 = [table[r["retrieved"][0]] for r in ref if r["retrieved"]]
            y1 = [table[r["retrieved"][0]] for r in win if r["retrieved"]]
            out[f"mmd_top1_{name}"] = (mmd_test(np.asarray(x1), np.asarray(y1), n_permutations=perms, seed=seed).p_value
                                       if len(x1) >= 5 and len(y1) >= 5 else None)
    return out


def rates(pvals: list, alpha: float) -> dict:
    keys = sorted({k for p in pvals for k in p})
    out = {}
    for k in keys:
        vals = [p[k] for p in pvals if p.get(k) is not None]
        n_none = sum(p.get(k) is None for p in pvals)
        out[k] = {"rate": round(float(np.mean([v < alpha for v in vals])), 4) if vals else None, "n": len(vals),
                  "not_computable": n_none}
    return out


def describe(rows):
    f = [r["faithfulness"] for r in rows if r.get("faithfulness") is not None]
    n = len(rows)
    return {
        "runs": n, "errors": sum(r["error"] is not None for r in rows),
        "affected_share": round(np.mean([r["affected"] for r in rows]), 4),
        "gold_retrieved": round(np.mean([r["gold_retrieved"] for r in rows]), 4),
        "correct": round(np.mean([r["correct"] for r in rows]), 4),
        "correct_ci": L.wilson(sum(r["correct"] for r in rows), n),
        "empty_retrieval": round(np.mean([bool(r.get("empty")) for r in rows]), 4),
        "no_search": round(np.mean([bool(r.get("no_search")) for r in rows]), 4),
        "faithfulness_mean": round(float(np.mean(f)), 4) if f else None,
        "faithfulness_scored_share": round(len(f) / n, 4) if n else None,
        "top1_bm25_mean": round(float(np.nanmean([r["top1_bm25"] for r in rows if r.get("top1_bm25") is not None])), 3)
        if any(r.get("top1_bm25") is not None for r in rows) else None,
        "wall_s_p50": round(float(np.median([r["wall_s"] for r in rows])), 2),
    }


# ------------------------------------------------------------------------------------------------ test
def run(ctx):
    import pandas as pd
    import torch
    from pulsar_data_collection.genai.detectors import FaithfulnessScorer

    p = ctx.params
    def log(m):
        print(m, flush=True)
    rng = random.Random(SEED)
    df = pd.read_parquet(PARQUET)
    contexts = sorted(df["context"].unique())
    pid_of = {c: f"P{i:04d}" for i, c in enumerate(contexts)}
    passages = {pid_of[c]: c for c in contexts}
    questions = [{"id": r.id, "question": r.question, "pid": pid_of[r.context], "answers": [str(a) for a in r.answers["text"]]}
                 for r in df.itertuples() if len(r.answers["text"])]
    rng.shuffle(questions)
    sizes = [("reference", 0.0, p["reference_runs"]), ("aa", 0.0, p["aa_pool_runs"])]
    sizes += [(f"level_{int(lv * 100)}", lv, p["level_runs"]) for lv in LEVELS]
    sizes += [("total_failure", 1.0, p["total_failure_runs"])]
    if sum(s for *_, s in sizes) > len(questions):
        raise ValueError("not enough questions for sampling without replacement")
    order = list(passages)
    rng.shuffle(order)  # nested removal: the first p of this order is removed at level p

    def index_at(level):
        if level >= 1.0:
            return None
        kept = order[int(round(level * len(order))):]
        idx = BM25({k: passages[k] for k in kept})
        idx.ids_set = set(kept)
        return idx

    rows, pos = [], 0
    t0 = time.perf_counter()
    for label, level, n in sizes:
        block = questions[pos:pos + n]
        pos += n
        t1 = time.perf_counter()
        rows += run_pool(label, level, block, index_at(level), passages, p["workers"], log)
        log(f"{label} done: {n} runs in {time.perf_counter() - t1:.0f} s")
    agent_s = time.perf_counter() - t0
    unload(A.MODEL)

    # faithfulness, scored once the generation model is unloaded
    scorer = FaithfulnessScorer(model=NLI_MODEL, device=0 if torch.cuda.is_available() else None)
    t1 = time.perf_counter()
    for r in rows:
        ctx_list = [passages[pid] for pid in r["retrieved"]]
        r["faithfulness"] = scorer.score(r["answer"] or "", ctx_list) if r["error"] is None else None
    faith_s = time.perf_counter() - t1
    del scorer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # embeddings of the corpus passages and of the questions
    t1 = time.perf_counter()
    pids = list(passages)
    emb = {}
    for name in p["embedders"]:
        e = A.embed([passages[k] for k in pids], model=name)
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        emb[name.split(":")[0]] = dict(zip(pids, e))
        unload(name)
    qtext = {q["id"]: q["question"] for q in questions}
    qe = A.embed([qtext[r["qid"]] for r in rows], model="granite-embedding:278m")
    qe = qe / np.linalg.norm(qe, axis=1, keepdims=True)
    for r, v in zip(rows, qe):
        r["qp_cos"] = float(v @ emb["granite-embedding"][r["retrieved"][0]]) if r["retrieved"] else None
    unload("granite-embedding:278m")
    emb_s = time.perf_counter() - t1

    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    ref, aa = by["reference"], by["aa"]
    zero = ref + aa
    W, alpha = p["window"], p["alpha"]
    pools = {"0": aa, **{str(int(lv * 100)): by[f"level_{int(lv * 100)}"] for lv in LEVELS}, "total": by["total_failure"]}

    # fixed reference vs disjoint windows
    disjoint = {}
    for key, pool in pools.items():
        wins = [pool[i:i + W] for i in range(0, len(pool) - W + 1, W)]
        disjoint[key] = [window_pvalues(ref, w, emb, perms=p["mmd_permutations"], seed=k) for k, w in enumerate(wins)]
    # resampling: reference of len(ref) from the 0 % runs, window of W from the rest (0 %) or from the level's pool
    rs = np.random.default_rng(SEED)
    resampled = {}
    for key, pool in pools.items():
        pv = []
        for d in range(p["draws"]):
            if key == "0":
                perm = rs.permutation(len(zero))
                r_ = [zero[i] for i in perm[:len(ref)]]
                w_ = [zero[i] for i in perm[len(ref):len(ref) + W]]
            else:
                r_ = [zero[i] for i in rs.choice(len(zero), len(ref), replace=False)]
                w_ = [pool[i] for i in rs.choice(len(pool), min(W, len(pool)), replace=False)]
            pv.append(window_pvalues(r_, w_, emb, mmd=d < p["mmd_draws"], perms=p["mmd_permutations"], seed=d))
        resampled[key] = rates(pv, alpha)
    disjoint_rates = {k: rates(v, alpha) for k, v in disjoint.items()}

    curve = {"0 (reference)": describe(ref), "0 (A/A pool)": describe(aa)}
    curve.update({f"{int(lv * 100)} %": describe(by[f"level_{int(lv * 100)}"]) for lv in LEVELS})
    curve["total failure"] = describe(by["total_failure"])

    # faithfulness vs ground truth, run level (all runs with a score)
    scored = [r for r in rows if r.get("faithfulness") is not None]
    auroc_affected = L.auroc([int(r["affected"]) for r in scored], [1 - r["faithfulness"] for r in scored])
    auroc_wrong = L.auroc([int(not r["correct"]) for r in scored], [1 - r["faithfulness"] for r in scored])
    aff = [r for r in rows if r["affected"] and r["label"] != "total_failure"]
    unaff = [r for r in rows if not r["affected"]]
    affected_view = {
        "affected_runs": len(aff),
        "affected_answered_correctly": round(np.mean([r["correct"] for r in aff]), 4) if aff else None,
        "unaffected_answered_correctly": round(np.mean([r["correct"] for r in unaff]), 4) if unaff else None,
        "affected_faithfulness_mean": round(float(np.mean([r["faithfulness"] for r in aff if r["faithfulness"] is not None])), 4)
        if aff else None,
        "unaffected_faithfulness_mean": round(float(np.mean([r["faithfulness"] for r in unaff if r["faithfulness"] is not None])), 4),
        "affected_empty_retrieval": round(np.mean([bool(r["empty"]) for r in aff]), 4) if aff else None,
        "auroc_faithfulness_affected": round(auroc_affected, 4) if auroc_affected is not None else None,
        "auroc_faithfulness_wrong_answer": round(auroc_wrong, 4) if auroc_wrong is not None else None,
    }
    # same-article fallback: when the relevant paragraph is gone, the retriever often returns a paragraph of the
    # same article (PIAF has ~4 paragraphs per article), which keeps the passage distribution unchanged
    title_of = {pid_of[c]: t for c, t in zip(df["context"], df["title"])}
    same_title = [title_of[r["retrieved"][0]] == title_of[r["pid"]] for r in aff if r["retrieved"]]
    affected_view["affected_top1_same_article"] = round(float(np.mean(same_title)), 4) if same_title else None

    levels_x = [0.0] + LEVELS
    fmeans = [describe(zero)["faithfulness_mean"]] + [curve[f"{int(lv * 100)} %"]["faithfulness_mean"] for lv in LEVELS]
    rho = stats.spearmanr(levels_x, fmeans).statistic if None not in fmeans else None
    product = ["faithfulness", "empty_retrieval", "mmd_nomic-embed-text", "mmd_granite-embedding",
               "mmd_top1_nomic-embed-text", "mmd_top1_granite-embedding"]
    aa_rates = {k: resampled["0"][k]["rate"] for k in product if resampled["0"].get(k, {}).get("rate") is not None}
    power = {k: {lv: resampled[lv].get(k, {}).get("rate") for lv in list(pools)[1:]} for k in
             product + ["diag_top1_bm25", "diag_query_passage_cos", "truth_correct"]}

    def first_level(k, target=0.8):
        for lv in [str(int(x * 100)) for x in LEVELS]:
            v = power[k].get(lv)
            if v is not None and v >= target:
                return int(lv)
        return None

    total_empty = resampled["total"]["empty_retrieval"]["rate"]
    metrics = {
        "n_questions": len(questions), "n_passages": len(passages), "n_runs": len(rows),
        "errors": sum(r["error"] is not None for r in rows), "agent_seconds": round(agent_s, 1),
        "faithfulness_seconds": round(faith_s, 1), "embedding_seconds": round(emb_s, 1),
        "curve": curve, "aa_level": aa_rates, "aa_level_max": max(aa_rates.values()) if aa_rates else None,
        "power_resampled": power, "rates_resampled": resampled, "rates_disjoint": disjoint_rates,
        "first_level_80pct_power": {k: first_level(k) for k in power},
        "faithfulness_dose_spearman": round(float(rho), 4) if rho is not None else None,
        "total_failure_empty_rate": total_empty,
        "total_failure_caught_by_empty_retrieval": bool(total_empty is not None and total_empty >= 0.95),
        "total_failure_faithfulness_not_computable": resampled["total"]["faithfulness"]["not_computable"],
        "affected": affected_view,
    }

    # ---------------------------------------------------------------- limits
    n_dis = {k: len(v) for k, v in disjoint.items()}
    if aa_rates and max(aa_rates.values()) > 0.05:
        for k, v in aa_rates.items():
            if v > 0.05:
                lo, hi = L.wilson(round(v * p["draws"]), p["draws"])
                ctx.limit(f"A/A false-alarm rate of {k} on windows of {W} runs is {v:.1%} (> 5 %)",
                          f"resampled windows at 0 % removal ({p['draws']} draws, Wilson {lo}-{hi}); disjoint windows: "
                          f"{disjoint_rates['0'].get(k)}",
                          ("mmd_test permutes rows as if independent, but the passages of one run (top 2 of the same "
                           "query, often the same article) are correlated: add a `groups=` argument to mmd_test / "
                           "embedding_drift that permutes whole runs, and document one row per run for retrieval."
                           if k.startswith("mmd_") and not k.startswith("mmd_top1") else
                           "Calibrate the window test on an A/A reference of the application (empirical null, as "
                           "AACalibrator does for tabular drift) instead of the nominal p-value."), "major")
    fl = first_level("faithfulness")
    pf = power["faithfulness"]
    if fl is None or fl > 20:
        ctx.limit(
            f"Faithfulness sees partial corpus loss late: power >= 80 % only from {fl if fl else '>80'} % of relevant "
            f"passages removed (windows of {W} runs)",
            "power by level (resampled): " + ", ".join(f"{k} %: {v}" for k, v in pf.items() if k != "total")
            + f"; faithfulness mean {fmeans} at levels {[int(x * 100) for x in levels_x]}; correct answers "
            + str([curve[c]["correct"] for c in curve]) + f"; run-level AUROC (relevant passage missing) "
            f"{affected_view['auroc_faithfulness_affected']}",
            "Add a retrieval-relevance signal next to faithfulness (top retrieval score or query/passage similarity, "
            "tested like faithfulness) and an answer-level 'not found in the passages' rate; faithfulness alone needs "
            "a large share of degraded answers.", "major")
    pe = power["empty_retrieval"]
    if all((v or 0) < 0.5 for k, v in pe.items() if k != "total"):
        ctx.limit(
            "Empty retrieval does not see partial corpus loss: a top-k retriever still returns the next-best passages, "
            "so the empty-retrieval rate stays flat until the index is completely gone",
            "empty-retrieval rate by level: " + ", ".join(f"{c}: {curve[c]['empty_retrieval']}" for c in curve)
            + "; detection rate by level: " + ", ".join(f"{k}: {v}" for k, v in pe.items())
            + f"; runs that lost their relevant passage with an empty retrieval: {affected_view['affected_empty_retrieval']}",
            "Document empty retrieval as a total-failure alarm only; record the retrieval score of the top passage on "
            "the tool span (e.g. pulsar.retrieval.top_score) and test its window distribution, and/or a relevance "
            "threshold in the retriever so that low-relevance results count as empty.", "major")
    for name in ("mmd_top1_nomic-embed-text", "mmd_top1_granite-embedding"):
        pm = power.get(name, {})
        if pm and all((v or 0) < 0.5 for k, v in pm.items() if k not in ("total",)):
            ctx.limit(
                f"MMD on the retrieved passages ({name[9:]}, one row per run) barely reacts to corpus loss: the replacement passages come "
                f"from the same corpus (often the same article), so their distribution does not move",
                "detection rate by level: " + ", ".join(f"{k}: {v}" for k, v in pm.items())
                + f"; top passage from the same article as the removed one in "
                f"{affected_view['affected_top1_same_article']} of affected runs",
                "Run MMD on (query, passage) pairs or on the query/passage similarity rather than on the passages "
                "alone; keep passage MMD for a corpus swap (new documents), not for missing documents.", "minor")
    if metrics["total_failure_faithfulness_not_computable"] and not metrics["total_failure_caught_by_empty_retrieval"]:
        ctx.limit("Total retrieval failure is invisible: faithfulness cannot score it and empty retrieval does not fire",
                  f"total-failure windows: {resampled['total']}", "Count searches that return nothing per window.",
                  "critical")

    keep = ["label", "level", "qid", "pid", "affected", "retrieved", "gold_retrieved", "empty", "no_search", "n_search",
            "correct", "faithfulness", "top1_bm25", "qp_cos", "termination", "wall_s", "error", "first_query", "answer"]
    (ctx.workdir / "runs.jsonl").write_text("\n".join(json.dumps({k: r.get(k) for k in keep}, ensure_ascii=False,
                                                                 default=str) for r in rows), encoding="utf-8")
    (ctx.workdir / "disjoint_pvalues.json").write_text(json.dumps(disjoint, indent=1), encoding="utf-8")
    return metrics
