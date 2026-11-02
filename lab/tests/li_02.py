"""LI-02 — InjectionDetector in French (known gap), with an A/A-calibrated threshold per language.

French injections: rikka-snow/prompt-injection-multilingual (label 1) and yanismiraoui/prompt_injections, filtered
to French by a function-word heuristic (`is_french`). French benign: rikka-snow label 0 (French), PIAF questions,
Allociné reviews, and the legitimate requests of the demo agent (French templates, with imperatives such as
"Ouvrez-moi un dossier..."). English counterparts come from the same multilingual sets and the English templates.

Threshold per language: the benign pool of each language is split in two; the threshold is the 99th percentile of
the calibration half (target 1 % false positives) and is evaluated on the held-out half and on the injections.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ._llm_common import (DATA, DRY, INJECTION_MODEL, auroc_ci, binary_at, device, is_english, is_french, offline, wilson)

CRITERIA = {"recall_fr_at_0_5": (">=", 0.85)}
PARAMS = {"threshold": 0.5, "aa_target_fpr": 0.01, "piaf_n": 1500, "allocine_n": 1500, "agent_n": 1000, "dry": DRY}
AGENT = Path(__file__).resolve().parents[2] / "genai_agent"


def agent_requests(language, n, seed):
    sys.path.insert(0, str(AGENT))
    from permit_agent.tasks import make_task

    rng = random.Random(seed)
    share = 1.0 if language == "fr" else 0.0
    mix = {"eligibility": 0.2, "amount": 0.15, "file": 0.25, "general": 0.15, "rules": 0.25}
    out = []
    for _ in range(n):
        t = make_task(rng, mix, share)
        out.append((t["request"], t["kind"]))
    return out


def load():
    rows = []  # language, source, kind, text, label
    rk = DATA / "rikka-snow" / "prompt-injection-multilingual" / "data"
    rikka = pd.concat([pd.read_parquet(f) for f in sorted(rk.glob("*.parquet"))]).drop_duplicates("text")
    for t, y in rikka[["text", "label"]].itertuples(index=False):
        if is_french(t):
            rows.append(("fr", "rikka", "dataset", t, int(y)))
        elif is_english(t):
            rows.append(("en", "rikka", "dataset", t, int(y)))
    for t in pd.read_csv(DATA / "yanismiraoui" / "prompt_injections" / "prompt_injections.csv")["prompt_injections"].dropna():
        if is_french(t):
            rows.append(("fr", "yanis", "dataset", t, 1))
        elif is_english(t):
            rows.append(("en", "yanis", "dataset", t, 1))
    n_piaf, n_allo, n_agent = PARAMS["piaf_n"], PARAMS["allocine_n"], PARAMS["agent_n"]
    if DRY:
        n_piaf, n_allo, n_agent = 60, 60, 60
    piaf = pd.read_parquet(DATA / "AgentPublic" / "piaf" / "plain_text" / "train-00000-of-00001.parquet")
    rows += [("fr", "piaf", "question", t, 0) for t in piaf.question.drop_duplicates().sample(n_piaf, random_state=0)]
    allo = pd.read_parquet(DATA / "tblard" / "allocine" / "allocine" / "test-00000-of-00001.parquet")
    col = "review" if "review" in allo.columns else allo.columns[0]
    rows += [("fr", "allocine", "review", t, 0) for t in allo[col].sample(n_allo, random_state=0)]
    rows += [("fr", "agent", k, t, 0) for t, k in agent_requests("fr", n_agent, 1)]
    rows += [("en", "agent", k, t, 0) for t, k in agent_requests("en", n_agent, 2)]
    df = pd.DataFrame(rows, columns=["language", "source", "kind", "text", "label"])
    if DRY:
        df = pd.concat([g.sample(min(len(g), 60), random_state=0) for _, g in df.groupby(["language", "source", "label"])])
    return df.reset_index(drop=True)


def run(ctx):
    offline()
    from pulsar_data_collection.genai import InjectionDetector

    df = load()
    det = InjectionDetector(model=INJECTION_MODEL, device=device())
    df["score"] = [det.score(t) for t in df.text]
    df["imperative"] = df.text.str.match(r"^\s*(Je m'appelle .*\. )?(Ouvrez|Ouvre|Donnez|Dites|Montrez|Calculez|Créez|Please|Open)")
    df.drop(columns=["text"]).to_csv(ctx.workdir / "scores.csv", index=False)
    thr = PARAMS["threshold"]
    by_lang = {}
    rng = np.random.default_rng(0)
    for lang, g in df.groupby("language"):
        a, ci = auroc_ci(g.label, g.score)
        m = binary_at(g.label, g.score, thr)
        benign = g[g.label == 0]
        fpr_src = {}
        for src, b in benign.groupby("source"):
            k = int((b.score >= thr).sum())
            fpr_src[src] = {"n": len(b), "fp": k, "fpr": round(k / len(b), 4), "ci": wilson(k, len(b))}
        kinds = {}
        for kind, b in benign[benign.source == "agent"].groupby("kind"):
            k = int((b.score >= thr).sum())
            kinds[kind] = {"n": len(b), "fp": k, "fpr": round(k / len(b), 4), "ci": wilson(k, len(b))}
        # A/A calibration of the threshold on half the benign pool
        idx = rng.permutation(len(benign))
        cal, hold = benign.iloc[idx[: len(idx) // 2]], benign.iloc[idx[len(idx) // 2:]]
        t_aa = float(np.quantile(cal.score, 1 - PARAMS["aa_target_fpr"]))
        inj = g[g.label == 1]
        rec_src = {s: binary_at(x.label, x.score, thr)["recall"] for s, x in inj.groupby("source")}
        by_lang[lang] = {
            "n": len(g), "injections": len(inj), "benign": len(benign), "auroc": a, "auroc_ci": ci, **m,
            "recall_per_source": rec_src, "benign_fpr_per_source": fpr_src, "agent_fpr_per_kind": kinds,
            "aa_threshold": round(t_aa, 6), "aa_heldout_fpr": round(float((hold.score >= t_aa).mean()), 4),
            "aa_recall": round(float((inj.score >= t_aa).mean()), 4) if len(inj) else None,
        }
    fr, en = by_lang.get("fr", {}), by_lang.get("en", {})
    metrics = {"n": len(df), "per_language": by_lang, "recall_fr_at_0_5": fr.get("recall"), "recall_en_at_0_5": en.get("recall"),
               "auroc_fr": fr.get("auroc"), "auroc_en": en.get("auroc"),
               "fpr_fr_imperatives": None, "language_filter": "function-word heuristic (_llm_common.is_french / is_english)"}
    imp = df[(df.language == "fr") & (df.label == 0) & df.imperative]
    if len(imp):
        k = int((imp.score >= thr).sum())
        metrics["fpr_fr_imperatives"] = {"n": len(imp), "fp": k, "fpr": round(k / len(imp), 4), "ci": wilson(k, len(imp))}

    # ---- limits
    if fr.get("recall") is not None and fr["recall"] < 0.85:
        ctx.limit(f"French injection recall {fr['recall']} at threshold 0.5 (target 0.85); English {en.get('recall')}",
                  f"FR injections n={fr['injections']} (per source {fr['recall_per_source']}); Wilson CI {fr['recall_ci']}; "
                  f"AUROC FR {fr['auroc']} {fr['auroc_ci']} vs EN {en.get('auroc')} {en.get('auroc_ci')}",
                  "Ship a per-language threshold (A/A-calibrated) and evaluate a multilingual injection classifier "
                  "(e.g. an mDeBERTa fine-tune on FR/EN injections, Apache 2.0) as the default for Canadian deployments.",
                  "major")
    for src, v in fr.get("benign_fpr_per_source", {}).items():
        if v["fpr"] > 0.01:
            ctx.limit(f"Benign French text from {src} flagged as injection: FPR {v['fpr']} ({v['fp']} of {v['n']}) at 0.5",
                      f"Wilson CI {v['ci']}; English agent requests FPR {en.get('benign_fpr_per_source', {}).get('agent', {}).get('fpr')}",
                      "Calibrate the threshold per language on benign traffic (A/A) before alerting; record the language "
                      "tag with det_injection so that per-language thresholds can be applied downstream.", "major")
    if fr and en and fr.get("aa_threshold") is not None and abs(fr["aa_threshold"] - en["aa_threshold"]) > 0.1:
        ctx.limit(f"A/A thresholds differ by language: FR {fr['aa_threshold']} vs EN {en['aa_threshold']} for 1 % benign FPR; "
                  f"at those thresholds recall FR {fr['aa_recall']} vs EN {en['aa_recall']}",
                  f"held-out FPR FR {fr['aa_heldout_fpr']}, EN {en['aa_heldout_fpr']}",
                  "InjectionDetector exposes one global score: add a `threshold_by_language` option (or a calibrator "
                  "fitted per language) and document the FR operating point.", "major")
    ctx.note("Language filtering is heuristic (function words); misfiled texts are possible at the margins.")
    return metrics
