"""LR-01 - FaithfulnessScorer vs human hallucination labels (RAGTruth, English).

Dataset: RAGTruth (ParticleMedia, MIT), read from /data/lab/datasets/ragtruth (outside the repositories): 17,790
responses of 6 LLMs to QA (MS MARCO passages), Summary (CNN/DM, Recent News) and Data2txt (Yelp records) prompts, with
human span labels (Evident / Subtle Conflict, Evident / Subtle Baseless Info).

Method
- The scorer under test is the product class `pulsar_data_collection.genai.detectors.FaithfulnessScorer` with its
  defaults (mDeBERTa-v3 XNLI, truncation at 512 tokens, at most 6 sentences of >= 3 words, passages joined).
- Every sentence of a response is scored with the scorer's own NLI pipeline (same splitting rule), so that the
  response-level score (mean of the first 6) and the sentence-level analysis come from one pass. A sample of responses
  is re-scored through `FaithfulnessScorer.score` itself to check that the replication is exact.
- Response level: AUROC of (1 - faithfulness) against "has at least one hallucination span", overall, per task, per
  generator model; calibration curve (10 bins) and ECE; alert threshold chosen on train (F1 and 5 % false alarm),
  evaluated on test.
- Span level: AUROC of (1 - entailment) per sentence against "the sentence overlaps a labelled span".
- Limits probed: the 512-token NLI window (share of pairs truncated, AUROC truncated vs not, chunked-premise variant on
  a sample), the 6-sentence cap (hallucinated spans that fall after sentence 6 are never seen).

    LAB_DRY=1 python -m lab.harness LR-01    # short dry run (300 responses)
    python -m lab.harness LR-01
"""

import json
import os
import random
import re
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("LAB_DATA", "/data/lab/datasets"))
ZIP = DATA / "ragtruth" / "RAGTruth-main.zip"
DRY = os.environ.get("LAB_DRY") == "1"
MODEL = os.environ.get("LAB_NLI_MODEL", "/data/lab/models/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")

N_TEST = 300 if DRY else None  # None = the whole test split (2,700)
N_TRAIN = 300 if DRY else 3000  # train sample used only to choose the alert threshold
N_CHECK = 20 if DRY else 100  # responses re-scored through FaithfulnessScorer.score
N_CHUNK = 60 if DRY else 400  # truncated responses re-scored with a chunked premise
SEED = 7

PARAMS = {"dataset": "RAGTruth (MIT)", "model": "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "dry": DRY,
          "n_test": N_TEST or "all", "n_train_threshold": N_TRAIN, "max_sentences": 6, "max_length": 512}
CRITERIA = {"auroc_response_test": (">=", 0.75)}

_SENTENCES = re.compile(r"(?<=[.!?])\s+")  # the scorer's rule


def _load():
    with zipfile.ZipFile(ZIP) as z:
        resp = [json.loads(x) for x in z.read("RAGTruth-main/dataset/response.jsonl").decode("utf-8").splitlines() if x]
        src = [json.loads(x) for x in z.read("RAGTruth-main/dataset/source_info.jsonl").decode("utf-8").splitlines() if x]
    return resp, {s["source_id"]: s for s in src}


def _passages(source: dict) -> list:
    """What a RAG application would hand to the scorer as the retrieved context"""
    info, task = source["source_info"], source["task_type"]
    if task == "QA":
        return [p.strip() for p in re.split(r"\n\s*\n", info["passages"]) if p.strip()]
    if task == "Data2txt":
        return [json.dumps(info, ensure_ascii=False)]
    return [info]


def _sentences_with_offsets(answer: str):
    """Same split as the scorer, with character offsets in the original response (labels are offsets)"""
    stripped = answer.strip()
    base = answer.find(stripped) if stripped else 0
    out, pos = [], 0
    for s in _SENTENCES.split(stripped):
        start = stripped.find(s, pos)
        pos = start + len(s)
        if len(s.split()) >= 3:
            out.append((s, base + start, base + pos))
    return out


def _auroc(y, s):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def _calibration(y_faithful, score, bins=10):
    y_faithful, score = np.asarray(y_faithful, float), np.asarray(score, float)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for i in range(bins):
        m = (score >= edges[i]) & ((score < edges[i + 1]) if i < bins - 1 else (score <= 1))
        if m.sum():
            rows.append({"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "n": int(m.sum()),
                         "mean_score": round(float(score[m].mean()), 3), "share_faithful": round(float(y_faithful[m].mean()), 3)})
            ece += m.sum() / len(score) * abs(score[m].mean() - y_faithful[m].mean())
    return rows, float(ece)


def run(ctx):
    import torch
    from pulsar_data_collection.genai.detectors import FaithfulnessScorer

    device = 0 if torch.cuda.is_available() else None
    rng = random.Random(SEED)
    resp, sources = _load()
    test = [r for r in resp if r["split"] == "test"]
    train = [r for r in resp if r["split"] == "train"]
    if N_TEST:
        test = rng.sample(test, N_TEST)
    train = rng.sample(train, N_TRAIN)

    t_load = time.perf_counter()
    scorer = FaithfulnessScorer(model=MODEL, device=device)
    load_s = time.perf_counter() - t_load
    nli, tok = scorer._nli, scorer._nli.tokenizer

    def entail_batch(pairs):
        outs = nli([{"text": p, "text_pair": h} for p, h in pairs], batch_size=16)
        if outs and isinstance(outs[0], dict):
            outs = [outs]
        return [float(next((x["score"] for x in o if x["label"].lower().startswith("entail")), 0.0)) for o in outs]

    def score_split(rows, keep_sentences: bool):
        records = []
        for r in rows:
            src = sources[r["source_id"]]
            passages = _passages(src)
            premise = " ".join(p for p in passages if p)
            sents = _sentences_with_offsets(r["response"])
            labels = r["labels"]
            rec = {"id": r["id"], "task": src["task_type"], "model": r["model"], "y": int(bool(labels)),
                   "types": sorted({l["label_type"] for l in labels}), "n_sent": len(sents)}
            if not sents or not premise:
                rec["score"] = None
                records.append(rec)
                continue
            premise_tokens = len(tok(premise, add_special_tokens=False)["input_ids"])
            t0 = time.perf_counter()
            ent = entail_batch([(premise, s) for s, _, _ in sents])
            if device is not None:
                torch.cuda.synchronize()
            rec["latency_all_s"] = time.perf_counter() - t0
            rec["score"] = float(np.mean(ent[:6]))  # FaithfulnessScorer.score: mean over the first 6 sentences
            rec["premise_tokens"] = premise_tokens
            rec["truncated"] = int(premise_tokens + 40 > 512)  # hypothesis + specials push past 512 tokens
            # hallucinated spans: which sentence they fall in (index), to measure the 6-sentence cap
            span_sent = []
            for lab in labels:
                idx = next((i for i, (_, a, b) in enumerate(sents) if lab["start"] < b and lab["end"] > a), None)
                span_sent.append(idx)
            rec["span_sentence_idx"] = span_sent
            if keep_sentences:
                rec["sent"] = [{"e": e, "y": int(any(l["start"] < b and l["end"] > a for l in labels)), "i": i}
                               for i, (e, (_, a, b)) in enumerate(zip(ent, sents))]
                # for the chunked variant (truncation limit)
                rec["_premise"], rec["_sents"] = premise, [s for s, _, _ in sents[:6]]
            records.append(rec)
        return records

    t0 = time.perf_counter()
    test_rec = score_split(test, keep_sentences=True)
    test_s = time.perf_counter() - t0
    train_rec = score_split(train, keep_sentences=False)

    # ---------------------------------------------------------------- replication check against the product method
    checked = [r for r in test_rec if r.get("score") is not None][:N_CHECK]
    diffs, prod_lat = [], []
    by_id = {x["id"]: x for x in test}
    for r in checked:
        raw = by_id[r["id"]]
        t1 = time.perf_counter()
        s = scorer.score(raw["response"], _passages(sources[raw["source_id"]]))
        prod_lat.append(time.perf_counter() - t1)
        diffs.append(abs(s - r["score"]))
    max_diff = float(max(diffs)) if diffs else None
    if max_diff is not None and max_diff > 1e-3:
        ctx.note(f"replication differs from FaithfulnessScorer.score by up to {max_diff:.4f} (batching effects)")

    scored = [r for r in test_rec if r.get("score") is not None]
    none_share = 1 - len(scored) / len(test_rec)
    y = np.array([r["y"] for r in scored])
    s = np.array([r["score"] for r in scored])
    auroc = _auroc(y, 1 - s)

    by_task = {}
    for task in sorted({r["task"] for r in scored}):
        m = np.array([r["task"] == task for r in scored])
        by_task[task] = {"n": int(m.sum()), "hallucinated_share": round(float(y[m].mean()), 3),
                         "auroc": _auroc(y[m], 1 - s[m])}
    by_model = {}
    for model in sorted({r["model"] for r in scored}):
        m = np.array([r["model"] == model for r in scored])
        by_model[model] = {"n": int(m.sum()), "auroc": _auroc(y[m], 1 - s[m])}
    by_type = {}
    clean = np.array([r["y"] == 0 for r in scored])
    for t in ["Evident Conflict", "Subtle Conflict", "Evident Baseless Info", "Subtle Baseless Info"]:
        m = np.array([t in r["types"] for r in scored])
        if m.sum() >= 10:
            sel = m | clean
            by_type[t] = {"n": int(m.sum()), "auroc_vs_clean": _auroc(y[sel], 1 - s[sel])}

    calib, ece = _calibration(1 - y, s)

    # ---------------------------------------------------------------- threshold chosen on train, applied on test
    tr = [r for r in train_rec if r.get("score") is not None]
    ytr, str_ = np.array([r["y"] for r in tr]), np.array([r["score"] for r in tr])
    grid = np.linspace(0.02, 0.98, 49)

    def rates(yv, sv, thr):
        alert = sv < thr
        tp, fp = int((alert & (yv == 1)).sum()), int((alert & (yv == 0)).sum())
        fn = int((~alert & (yv == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / max(int((yv == 0).sum()), 1)
        return prec, rec, fpr, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)

    thr_f1 = float(max(grid, key=lambda t: rates(ytr, str_, t)[3]))
    thr_fa5 = float(np.quantile(str_[ytr == 0], 0.05))  # 5 % of clean train responses alert
    thresholds = {}
    for name, thr in [("max_f1_train", thr_f1), ("false_alarm_5pct_train", thr_fa5), ("default_0.5", 0.5)]:
        p, r_, f, f1 = rates(y, s, thr)
        thresholds[name] = {"threshold": round(thr, 3), "precision": round(p, 3), "recall": round(r_, 3),
                            "false_alarm": round(f, 3), "f1": round(f1, 3)}

    # ---------------------------------------------------------------- span (sentence) level
    sent = [x for r in scored for x in r["sent"]]
    ys, es = np.array([x["y"] for x in sent]), np.array([x["e"] for x in sent])
    auroc_sentence = _auroc(ys, 1 - es)

    # ---------------------------------------------------------------- 6-sentence cap
    spans = [i for r in scored for i in r["span_sentence_idx"]]
    spans_after6 = sum(1 for i in spans if i is not None and i >= 6)
    spans_unseen = sum(1 for i in spans if i is None)  # inside a sentence of < 3 words or not aligned
    resp_only_after6 = [r for r in scored if r["y"] == 1 and r["span_sentence_idx"]
                        and all(i is not None and i >= 6 for i in r["span_sentence_idx"])]
    long_resp = sum(1 for r in scored if r["n_sent"] > 6)
    halluc_after_auroc = None
    if resp_only_after6:
        # how the response score ranks these responses: AUROC vs clean
        sel = [r for r in scored if r["y"] == 0] + resp_only_after6
        halluc_after_auroc = _auroc([r["y"] for r in sel], [1 - r["score"] for r in sel])
    # what a scorer that read every sentence would have scored (mean over all) and min over sentences
    s_all = np.array([np.mean([x["e"] for x in r["sent"]]) for r in scored])
    s_min = np.array([np.min([x["e"] for x in r["sent"]]) for r in scored])
    auroc_all_sent, auroc_min_sent = _auroc(y, 1 - s_all), _auroc(y, 1 - s_min)

    # ---------------------------------------------------------------- 512-token window
    trunc = np.array([r["truncated"] for r in scored]) == 1
    toks = np.array([r["premise_tokens"] for r in scored])
    auroc_trunc, auroc_fit = _auroc(y[trunc], 1 - s[trunc]), _auroc(y[~trunc], 1 - s[~trunc])
    trunc_by_task = {t: round(float(trunc[np.array([r["task"] == t for r in scored])].mean()), 3) for t in by_task}
    # faithful sentences that a truncated premise cannot support: mean entailment of clean responses
    clean_trunc = float(s[trunc & (y == 0)].mean()) if (trunc & (y == 0)).any() else None
    clean_fit = float(s[~trunc & (y == 0)].mean()) if (~trunc & (y == 0)).any() else None

    # chunked-premise variant on a sample of truncated responses: max entailment over 400-token windows (stride 300)
    chunk_eval = None
    trunc_rows = [r for r in scored if r["truncated"]]
    sample = rng.sample(trunc_rows, min(N_CHUNK, len(trunc_rows))) if trunc_rows else []
    if sample:
        chunk_scores, chunk_lat = [], []
        for r in sample:
            ids = tok(r["_premise"], add_special_tokens=False)["input_ids"]
            windows = [tok.decode(ids[i:i + 400]) for i in range(0, max(len(ids) - 100, 1), 300)]
            t1 = time.perf_counter()
            ent = entail_batch([(w, h) for h in r["_sents"] for w in windows])
            chunk_lat.append(time.perf_counter() - t1)
            k = len(windows)
            chunk_scores.append(float(np.mean([max(ent[j * k:(j + 1) * k]) for j in range(len(r["_sents"]))])))
        ys_ = np.array([r["y"] for r in sample])
        base = np.array([r["score"] for r in sample])
        cs = np.array(chunk_scores)
        chunk_eval = {"n": len(sample), "auroc_truncated_default": _auroc(ys_, 1 - base),
                      "auroc_truncated_chunked": _auroc(ys_, 1 - cs),
                      "mean_score_clean_default": float(base[ys_ == 0].mean()) if (ys_ == 0).any() else None,
                      "mean_score_clean_chunked": float(cs[ys_ == 0].mean()) if (ys_ == 0).any() else None,
                      "latency_p50_s_chunked": float(np.median(chunk_lat)),
                      "windows_mean": None}

    lat = np.array([r["latency_all_s"] for r in scored])
    metrics = {
        "n_test_scored": len(scored), "n_test": len(test_rec), "share_none": round(none_share, 4),
        "hallucinated_share_test": round(float(y.mean()), 3),
        "auroc_response_test": round(auroc, 4) if auroc is not None else None,
        "auroc_by_task": by_task, "auroc_by_generator": by_model, "auroc_by_label_type": by_type,
        "calibration": calib, "ece": round(ece, 4), "thresholds": thresholds,
        "auroc_sentence_level": round(auroc_sentence, 4) if auroc_sentence is not None else None,
        "n_sentences": int(len(sent)),
        "auroc_mean_all_sentences": round(auroc_all_sent, 4), "auroc_min_sentence": round(auroc_min_sent, 4),
        "responses_over_6_sentences_share": round(long_resp / len(scored), 4),
        "spans_total": len(spans), "spans_after_sentence_6": spans_after6, "spans_unaligned": spans_unseen,
        "responses_hallucinated_only_after_6": len(resp_only_after6),
        "auroc_responses_hallucinated_only_after_6": halluc_after_auroc,
        "truncated_share": round(float(trunc.mean()), 4), "truncated_share_by_task": trunc_by_task,
        "premise_tokens_p50": int(np.median(toks)), "premise_tokens_p95": int(np.quantile(toks, 0.95)),
        "auroc_truncated": auroc_trunc, "auroc_fits_512": auroc_fit,
        "mean_score_clean_truncated": clean_trunc, "mean_score_clean_fits": clean_fit,
        "chunked_variant": chunk_eval,
        "latency_per_response_p50_s": round(float(np.median(lat)), 4),
        "latency_per_response_p95_s": round(float(np.quantile(lat, 0.95)), 4),
        "latency_product_score_p50_s": round(float(np.median(prod_lat)), 4) if prod_lat else None,
        "replication_max_abs_diff": max_diff, "model_load_s": round(load_s, 1), "test_scoring_s": round(test_s, 1),
        "device": "cuda" if device is not None else "cpu",
    }

    # ---------------------------------------------------------------- limits
    if auroc is not None and auroc < 0.75:
        worst = min((t for t in by_task if by_task[t]["auroc"] is not None), key=lambda t: by_task[t]["auroc"])
        ctx.limit(
            f"Response-level AUROC {auroc:.3f} on RAGTruth test is below the 0.75 target",
            f"n={len(scored)}, hallucinated share {y.mean():.2f}; by task: "
            + ", ".join(f"{t} {v['auroc']:.3f}" for t, v in by_task.items() if v['auroc'] is not None)
            + f"; worst {worst}",
            "Score each sentence against chunked passages (max entailment over 400-token windows) and alert on the "
            "minimum sentence entailment rather than the mean; consider a RAG-specific checker (e.g. an NLI model "
            "fine-tuned on RAGTruth-like data) as an optional model behind the same interface.",
            "major")
    if trunc.mean() > 0.10:
        ev = (f"{trunc.mean():.1%} of test responses have a premise longer than the 512-token window "
              f"(p50 {int(np.median(toks))} tokens, p95 {int(np.quantile(toks, 0.95))}); by task {trunc_by_task}; "
              f"AUROC truncated {auroc_trunc if auroc_trunc is None else round(auroc_trunc, 3)} vs fits "
              f"{auroc_fit if auroc_fit is None else round(auroc_fit, 3)}; mean score of clean responses "
              f"{clean_trunc if clean_trunc is None else round(clean_trunc, 3)} truncated vs "
              f"{clean_fit if clean_fit is None else round(clean_fit, 3)} fits")
        if chunk_eval:
            ev += (f"; chunked premise on {chunk_eval['n']} truncated responses: AUROC "
                   f"{chunk_eval['auroc_truncated_default']} -> {chunk_eval['auroc_truncated_chunked']}, clean mean "
                   f"{chunk_eval['mean_score_clean_default']} -> {chunk_eval['mean_score_clean_chunked']}")
        ctx.limit(
            "FaithfulnessScorer silently truncates the joined passages at 512 tokens: evidence beyond the window "
            "cannot support a sentence, so faithful answers on long contexts score low",
            ev,
            "In FaithfulnessScorer.score, split the premise into overlapping windows that fit the model (premise "
            "budget = 512 - hypothesis tokens), take the max entailment per sentence over windows (or score each "
            "passage separately), and expose `truncated` / `windows` counts as span attributes.",
            "major")
    if long_resp / len(scored) > 0.05:
        ctx.limit(
            "The 6-sentence cap ignores the end of long answers: hallucinations placed after sentence 6 do not "
            "move the score",
            f"{long_resp / len(scored):.1%} of test responses have more than 6 scorable sentences; "
            f"{spans_after6} of {len(spans)} labelled spans ({spans_after6 / max(len(spans), 1):.1%}) fall after "
            f"sentence 6; {len(resp_only_after6)} hallucinated responses have all their spans there "
            f"(AUROC vs clean {halluc_after_auroc}); mean over all sentences AUROC {auroc_all_sent:.3f}, "
            f"min sentence AUROC {auroc_min_sent:.3f} vs default {auroc:.3f}",
            "Score every sentence (batching keeps the cost linear) or make max_sentences a constructor argument with "
            "a documented default, and add an aggregate option `min` (weakest sentence) for alerting.",
            "minor" if spans_after6 / max(len(spans), 1) < 0.1 else "major")
    if ece > 0.15:
        ctx.limit(
            "The faithfulness score is not a calibrated probability of a faithful answer",
            f"ECE {ece:.3f} over 10 bins on {len(scored)} test responses; bins: "
            + "; ".join(f"{c['bin']} score {c['mean_score']} vs faithful {c['share_faithful']} (n={c['n']})" for c in calib),
            "Document the score as a ranking signal; ship a per-application threshold calibration helper (A/A on "
            "known-good answers, like AACalibrator) instead of a fixed 0.5 alert threshold.",
            "minor")
    d05 = thresholds["default_0.5"]
    # Not a limit (refuted by the night's verifier): Pulsar.ml ships no fixed 0.5 faithfulness threshold; the product
    # alert is a Mann-Whitney test of the window against the reference, and the score calibration is covered by M19.
    ctx.note(f"Fixed 0.5 cut-off, for reference only (not a product setting): false alarm {d05['false_alarm']:.1%}, "
             f"recall {d05['recall']:.1%}, precision {d05['precision']:.1%}; train-chosen 5 % false-alarm threshold "
             f"{thresholds['false_alarm_5pct_train']}")
    for r in test_rec:
        r.pop("_premise", None), r.pop("_sents", None), r.pop("sent", None)
    (ctx.workdir / "scores_test.jsonl").write_text("\n".join(json.dumps(r) for r in test_rec), encoding="utf-8")
    return metrics
