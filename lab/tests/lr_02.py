"""LR-02 - FaithfulnessScorer in French: gold answers vs synthetic perturbations of PIAF.

Dataset: AgentPublic/piaf (MIT), 3,835 French questions with an extractive answer in a Wikipedia paragraph, read from
/data/lab/datasets/AgentPublic/piaf (outside the repositories).

Method
- Context = the PIAF paragraph (one retrieved passage). Gold answer = the sentence of the paragraph that holds the
  answer span (extractive, supported by construction), and a templated declarative answer
  ("La réponse à la question « q » est a.").
- Perturbations of the gold sentence, one at a time:
  number   - a number of the sentence (the answer itself when numeric) changed: small (+/-1 to 10 %) or large (x2 to x5;
             years +/-3 to 15);
  entity   - the answer span replaced by another capitalised entity of the same paragraph (a plausible distractor that
             is in the context, the hard case) or by the answer of another question (out of context);
  negation - the first finite verb negated (est -> n'est pas, a -> n'a pas, sont -> ne sont pas, ...).
- Score = `FaithfulnessScorer.score(answer, [paragraph])` with the product defaults. AUROC of (1 - score) per
  perturbation type, gold vs perturbed on the same items; detection rate at the threshold that gives 5 % false alarm
  on gold answers.

    LAB_DRY=1 python -m lab.harness LR-02
    python -m lab.harness LR-02
"""

import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("LAB_DATA", "/data/lab/datasets"))
PARQUET = DATA / "AgentPublic" / "piaf" / "plain_text" / "train-00000-of-00001.parquet"
MODEL = os.environ.get("LAB_NLI_MODEL", "/data/lab/models/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
DRY = os.environ.get("LAB_DRY") == "1"
N_ITEMS = 200 if DRY else None  # None = all 3,835
SEED = 11

PARAMS = {"dataset": "AgentPublic/piaf (MIT)", "dry": DRY, "n_items": N_ITEMS or "all", "seed": SEED}
CRITERIA = {"auroc_min_type": (">=", 0.80), "auroc_number": (">=", 0.80)}

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý«\"(])")
_NUM = re.compile(r"(?<![\w,.])(\d{1,3}(?:[  .]\d{3})+|\d+(?:,\d+)?)(?![\w])")
_ENT = re.compile(r"(?<![\w'’])([A-ZÀ-ÖØ-Ý][\wÀ-ÿ'’-]+(?:\s+(?:de |du |des |la |le |d'|d’)?[A-ZÀ-ÖØ-Ý][\wÀ-ÿ'’-]+)*)")
_NEG = [(r"\best\b", "n'est pas"), (r"\bsont\b", "ne sont pas"), (r"\bétait\b", "n'était pas"),
        (r"\bétaient\b", "n'étaient pas"), (r"\bfut\b", "ne fut pas"), (r"\bfurent\b", "ne furent pas"),
        (r"\ba\b", "n'a pas"), (r"\bont\b", "n'ont pas"), (r"\bavait\b", "n'avait pas"), (r"\bpeut\b", "ne peut pas"),
        (r"\bdevient\b", "ne devient pas"), (r"\breprésente\b", "ne représente pas"), (r"\bcompte\b", "ne compte pas")]
_STOP_ENT = {"Le", "La", "Les", "Il", "Elle", "Ils", "Elles", "En", "Un", "Une", "Des", "Ce", "Cette", "Ces", "Dans",
             "Au", "Aux", "Par", "Pour", "Sur", "Selon", "De", "Du", "Son", "Sa", "Ses", "Leur", "Après", "Avant", "On",
             "Mais", "Et", "Si", "Lors", "Depuis", "Parmi", "L'", "À", "Á", "Cependant", "Ainsi", "C'est"}


def _is_entity(text: str) -> bool:
    """A named entity (proper noun phrase): capitalised first word that is not a function word, 1-5 words, no digit"""
    words = text.split()
    return (0 < len(words) <= 5 and text[:1].isupper() and words[0].strip("'’") not in _STOP_ENT
            and not re.search(r"\d", text) and len(text) > 2 and not re.match(r"^(L|D)['’]", text))


def _to_int(s: str):
    try:
        return int(re.sub(r"[  .]", "", s)) if "," not in s else None
    except ValueError:
        return None


def _fmt_like(orig: str, value: int) -> str:
    if re.search(r"[  .]\d{3}", orig):
        sep = next(c for c in orig if c in "  .")
        return f"{value:,}".replace(",", sep)
    return str(value)


def _change_number(tok: str, rng: random.Random, size: str):
    v = _to_int(tok)
    if v is None:  # decimals like 3,5
        a, b = tok.split(",")
        f = float(f"{a}.{b}")
        g = f * (rng.uniform(1.01, 1.10) if size == "small" else rng.uniform(2, 5))
        return f"{g:.{len(b)}f}".replace(".", ",")
    if 1000 <= v <= 2100 and not re.search(r"[  .]", tok):  # a year
        d = rng.randint(1, 2) if size == "small" else rng.randint(3, 15)
        return str(v + rng.choice([-1, 1]) * d)
    if size == "small":
        if v < 10:
            new = v + rng.choice([-1, 1]) if v > 1 else v + 1
        else:
            new = int(round(v * rng.choice([-1, 1]) * rng.uniform(0.01, 0.10))) + v
            new = new if new != v else v + 1
    else:
        new = int(v * rng.uniform(2, 5)) if v > 0 else rng.randint(2, 9)
    return _fmt_like(tok, new)


def _gold_sentence(context: str, start: int):
    pos = 0
    for s in _SENT.split(context):
        i = context.find(s, pos)
        if i <= start < i + len(s):
            return s, start - i
        pos = i + len(s)
    return None, None


def _items(df, rng):
    entity_answers = sorted({str(a["text"][0]) for a in df["answers"] if len(a["text"]) and _is_entity(str(a["text"][0]))})
    items = []
    for row in df.itertuples():
        if not len(row.answers["text"]):
            continue
        ans, start = str(row.answers["text"][0]), int(row.answers["answer_start"][0])
        sent, off = _gold_sentence(row.context, start)
        if not sent or sent[off:off + len(ans)] != ans or len(sent.split()) < 5:
            continue
        it = {"id": row.id, "context": row.context, "question": row.question, "answer": ans, "gold": sent,
              "template": f"La réponse à la question « {row.question.strip().rstrip('?').strip()} » est {ans}.",
              "perturbed": {}}
        # number
        nums = list(_NUM.finditer(sent))
        target = next((m for m in nums if off <= m.start() < off + len(ans)), nums[0] if nums else None)
        if target:
            for size in ("small", "large"):
                new = _change_number(target.group(0), rng, size)
                if new != target.group(0):
                    it["perturbed"][f"number_{size}"] = sent[:target.start()] + new + sent[target.end():]
            it["number_in_answer"] = bool(off <= target.start() < off + len(ans))
        # entity swap: another capitalised entity of the paragraph, absent from the gold sentence
        if _is_entity(ans):
            cands = sorted({m.group(1) for m in _ENT.finditer(row.context)} - {ans})
            cands = [c for c in cands if _is_entity(c) and c not in sent and ans not in c and c not in ans]
            if cands:
                it["perturbed"]["entity_in_context"] = sent[:off] + rng.choice(cands) + sent[off + len(ans):]
            others = [o for o in rng.sample(entity_answers, min(30, len(entity_answers)))
                      if o != ans and o not in row.context]
            if others:
                it["perturbed"]["entity_out_of_context"] = sent[:off] + others[0] + sent[off + len(ans):]
        # negation
        for pat, rep in _NEG:
            m = re.search(pat, sent)
            if m:
                neg = sent[:m.start()] + rep + sent[m.end():]
                neg = re.sub(r"\bne ([aeéèêiouyhAEÉIOUYH])", r"n'\1", neg)
                it["perturbed"]["negation"] = neg
                break
        if it["perturbed"]:
            items.append(it)
    return items


def _auroc(gold, pert):
    from sklearn.metrics import roc_auc_score

    g, p = np.asarray(gold, float), np.asarray(pert, float)
    if len(g) == 0 or len(p) == 0:
        return None
    return float(roc_auc_score(np.r_[np.zeros(len(g)), np.ones(len(p))], np.r_[1 - g, 1 - p]))


def _paired_auroc(scores, kind):
    """AUROC on the items that have this perturbation: gold vs perturbed, same items"""
    sel = [s for s in scores if kind in s["p"] and s["gold"] is not None and s["p"][kind] is not None]
    return _auroc([s["gold"] for s in sel], [s["p"][kind] for s in sel]), sel


def run(ctx):
    import pandas as pd
    import torch
    from pulsar_data_collection.genai.detectors import FaithfulnessScorer

    rng = random.Random(SEED)
    df = pd.read_parquet(PARQUET)
    if N_ITEMS:
        df = df.sample(N_ITEMS, random_state=SEED)
    items = _items(df, rng)
    scorer = FaithfulnessScorer(model=MODEL, device=0 if torch.cuda.is_available() else None)
    tok = scorer._nli.tokenizer

    scores, lat = [], []
    t0 = time.perf_counter()
    for it in items:
        ctx_list = [it["context"]]
        t1 = time.perf_counter()
        rec = {"id": it["id"], "gold": scorer.score(it["gold"], ctx_list), "template": scorer.score(it["template"], ctx_list),
               "p": {k: scorer.score(v, ctx_list) for k, v in it["perturbed"].items()},
               "num_in_answer": it.get("number_in_answer"), "gold_words": len(it["gold"].split()),
               "ctx_tokens": len(tok(it["context"], add_special_tokens=False)["input_ids"])}
        lat.append((time.perf_counter() - t1) / (2 + len(it["perturbed"])))
        scores.append(rec)
    total_s = time.perf_counter() - t0

    gold = np.array([s["gold"] for s in scores if s["gold"] is not None])
    template = np.array([s["template"] for s in scores if s["template"] is not None])
    thr = float(np.quantile(gold, 0.05))  # 5 % false alarm on gold answers
    kinds = ["number_small", "number_large", "entity_in_context", "entity_out_of_context", "negation"]
    per_type = {}
    for k in kinds:
        auc, sel = _paired_auroc(scores, k)
        if not sel:
            continue
        pv = np.array([s["p"][k] for s in sel])
        drop = np.array([s["gold"] - s["p"][k] for s in sel])
        per_type[k] = {"n": len(sel), "auroc": round(auc, 4) if auc is not None else None,
                       "detected_at_5pct_fa": round(float((pv < thr).mean()), 4),
                       "detected_at_0.5": round(float((pv < 0.5).mean()), 4),
                       "mean_score_perturbed": round(float(pv.mean()), 4), "mean_drop": round(float(drop.mean()), 4),
                       "share_score_not_lower": round(float((drop <= 0).mean()), 4)}
    # number swaps where the number is the answer itself (what matters for program rules)
    sel = [s for s in scores if "number_large" in s["p"] and s["num_in_answer"] and s["gold"] is not None]
    num_ans = {}
    for k in ("number_small", "number_large"):
        sk = [s for s in sel if k in s["p"]]
        if sk:
            num_ans[k] = {"n": len(sk), "auroc": _auroc([s["gold"] for s in sk], [s["p"][k] for s in sk]),
                          "detected_at_5pct_fa": float(np.mean([s["p"][k] < thr for s in sk]))}
    # template answers (the answer re-worded around the question)
    auroc_template_vs_numlarge = _auroc(template, [s["p"]["number_large"] for s in scores if "number_large" in s["p"]])
    ct = np.array([s["ctx_tokens"] for s in scores])

    # gold (verbatim) score by sentence length: NLI models tend to call long hypotheses "neutral"
    by_len = {}
    for name, lo, hi in [("<=15 words", 0, 15), ("16-25 words", 16, 25), (">25 words", 26, 10_000)]:
        sel_l = [s for s in scores if s["gold"] is not None and lo <= s["gold_words"] <= hi]
        if sel_l:
            g = np.array([s["gold"] for s in sel_l])
            by_len[name] = {"n": len(sel_l), "gold_mean": round(float(g.mean()), 4),
                            "gold_share_below_0.5": round(float((g < 0.5).mean()), 4)}
            for k in kinds:
                sk = [s for s in sel_l if k in s["p"]]
                if len(sk) >= 10:
                    by_len[name][f"auroc_{k}"] = round(_auroc([s["gold"] for s in sk], [s["p"][k] for s in sk]), 4)
    aucs = [v["auroc"] for v in per_type.values() if v["auroc"] is not None]
    num_aucs = [per_type[k]["auroc"] for k in ("number_small", "number_large") if k in per_type]
    metrics = {
        "n_items": len(items), "n_rows": int(len(df)), "per_type": per_type, "number_is_answer": num_ans,
        "auroc_min_type": round(min(aucs), 4) if aucs else None,
        "auroc_number": round(min(num_aucs), 4) if num_aucs else None,
        "gold_mean": round(float(gold.mean()), 4), "gold_p05_threshold": round(thr, 4),
        "gold_share_below_0.5": round(float((gold < 0.5).mean()), 4), "gold_by_length": by_len,
        "template_mean": round(float(template.mean()), 4), "template_share_below_0.5": round(float((template < 0.5).mean()), 4),
        "template_share_below_gold_threshold": round(float((template < thr).mean()), 4),
        "auroc_template_gold_vs_number_large": auroc_template_vs_numlarge,
        "context_tokens_p50": int(np.median(ct)), "context_tokens_p95": int(np.quantile(ct, 0.95)),
        "context_over_470_share": round(float((ct > 470).mean()), 4),
        "latency_per_answer_p50_s": round(float(np.median(lat)), 4), "total_scoring_s": round(total_s, 1),
    }

    # ---------------------------------------------------------------- limits
    for k, v in per_type.items():
        if v["auroc"] is not None and v["auroc"] < 0.80:
            label = {"number_small": "small number changes (1-10 %, years +/-1-2)",
                     "number_large": "large number changes (x2-x5, years +/-3-15)",
                     "entity_in_context": "an entity swapped for another entity of the same passage",
                     "entity_out_of_context": "an entity swapped for an entity absent from the passage",
                     "negation": "a negated verb"}[k]
            ctx.limit(
                f"French faithfulness misses {label}: AUROC {v['auroc']:.3f} < 0.80",
                f"n={v['n']} PIAF items; mean score gold {gold.mean():.3f} vs perturbed {v['mean_score_perturbed']:.3f}; "
                f"perturbed score not lower than gold in {v['share_score_not_lower']:.1%} of items; detected at the "
                f"5 % false-alarm threshold ({thr:.3f}) in {v['detected_at_5pct_fa']:.1%}",
                {"number_small": "Add a deterministic numeric-consistency check next to NLI: extract numbers, dates and "
                                 "amounts from the answer and require each to appear (normalised) in the passages; "
                                 "report `unsupported_numbers` as its own detector score.",
                 "number_large": "Add a deterministic numeric-consistency check next to NLI (numbers, dates, amounts "
                                 "in the answer must appear in the passages, after normalisation).",
                 "entity_in_context": "Complement NLI with an entity-relation check (the swapped entity exists in the "
                                      "passage, so lexical overlap hides it); test a larger NLI model or a French "
                                      "fine-tune; document the blind spot.",
                 "entity_out_of_context": "Add an unsupported-entity check (capitalised entities of the answer absent "
                                          "from the passages) as a cheap complement to NLI.",
                 "negation": "Document the negation blind spot; evaluate a larger multilingual NLI model on French "
                             "negation (XNLI-fr) before making it the default."}[k],
                "critical" if k.startswith("number") else "major")
    if (gold < 0.5).mean() > 0.2:
        ctx.limit(
            "Answers copied verbatim from the passage are scored as unsupported: the NLI model calls long "
            "hypotheses 'neutral', so the faithfulness score falls with answer length, not with faithfulness",
            f"gold sentences (verbatim from the PIAF passage, supported by construction): mean score {gold.mean():.3f}, "
            f"{(gold < 0.5).mean():.1%} below 0.5; by length: "
            + "; ".join(f"{k} mean {v['gold_mean']} ({v['gold_share_below_0.5']:.0%} < 0.5, n={v['n']})" for k, v in by_len.items())
            + f"; templated short answers mean {template.mean():.3f}",
            "Split answers into clauses (commas, conjunctions) before NLI, or cap hypothesis length; add a lexical "
            "support shortcut (a sentence whose content words are all in one passage sentence is supported); "
            "re-test a larger NLI model and keep the alert on a window mean relative to an A/A reference.",
            "major")
    if (template < thr).mean() > 0.2:
        ctx.limit(
            "Correct answers worded differently from the passage score low: the threshold learnt on extractive "
            "answers flags paraphrased correct answers",
            f"templated correct answers: mean {template.mean():.3f}, {(template < thr).mean():.1%} below the gold 5 % "
            f"threshold {thr:.3f}, {(template < 0.5).mean():.1%} below 0.5 (extractive gold mean {gold.mean():.3f})",
            "Calibrate the alert threshold on the application's own answers (A/A window), not on extractive "
            "references; drop the question restatement before scoring (score only answer clauses).",
            "minor")
    (ctx.workdir / "scores.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in scores), encoding="utf-8")
    # 20 examples per perturbation type for the verification step (PIAF is public, MIT; no personal data)
    ex = {k: [{"gold": it["gold"], "perturbed": it["perturbed"][k]} for it in items if k in it["perturbed"]][:20] for k in kinds}
    (ctx.workdir / "examples.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1), encoding="utf-8")
    return metrics
