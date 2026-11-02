"""Shared helpers of the "llm-security" family: data locations, metrics with intervals, language heuristic.

Nothing here writes a text: tests keep scores, labels, sources and counts only.
"""

import math
import os
import random
import re
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("LAB_DATA", "/data/lab/datasets"))
MODELS = Path(os.environ.get("LAB_MODELS", "/data/lab/models"))
DRY = os.environ.get("LAB_DRY", "") not in ("", "0")

INJECTION_MODEL = str(MODELS / "protectai" / "deberta-v3-base-prompt-injection-v2")
GLINER_MODEL = str(MODELS / "urchade" / "gliner_multi_pii-v1")


def offline():
    """Models and datasets are already on disk: never reach the hub during a test"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def device():
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


def sample(items, n, seed=0):
    items = list(items)
    if n is None or len(items) <= n:
        return items
    return random.Random(seed).sample(items, n)


# ------------------------------------------------------------------------------------------------ metrics
def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def auroc(y, s):
    from sklearn.metrics import roc_auc_score

    y, s = np.asarray(y), np.asarray(s)
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, s))


def auroc_ci(y, s, b=300, seed=0):
    """AUROC with a percentile bootstrap interval (stratified by label)"""
    y, s = np.asarray(y), np.asarray(s)
    a = auroc(y, s)
    if a is None:
        return None, (None, None)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(b):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(auroc(y[idx], s[idx]))
    return round(a, 4), (round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4))


def ece(y, s, bins=10):
    """Expected calibration error of a probability score"""
    y, s = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total, out = len(y), 0.0
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & ((s < hi) if hi < 1 else (s <= hi))
        if m.sum():
            out += m.sum() / total * abs(y[m].mean() - s[m].mean())
            table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_score": round(float(s[m].mean()), 3),
                          "positive_rate": round(float(y[m].mean()), 3)})
    return round(float(out), 4), table


def binary_at(y, s, threshold=0.5):
    y, s = np.asarray(y), np.asarray(s)
    p = s >= threshold
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec and round(prec, 4), "recall": rec and round(rec, 4),
            "f1": f1 and round(f1, 4), "recall_ci": wilson(tp, tp + fn), "fpr": round(fp / (fp + tn), 4) if fp + tn else None,
            "fpr_ci": wilson(fp, fp + tn)}


# ------------------------------------------------------------------------------------------------ language
_WORD = re.compile(r"[a-zàâäçéèêëîïôöùûüœ']+", re.IGNORECASE)
FR_WORDS = set("""les des est vous tu je nous pas une pour avec dans sur mais qui cette tes mes moi toi ton ta votre vos
sont été être avez fais faire tout tous toutes au aux du ne plus leur leurs ces ses mon ma très comme où aussi même
quel quelle quels quelles oublie oubliez ignore ignorez désormais maintenant instructions précédentes dis donne
montre révèle ton tes rien elle il ils elles on été cela ça avoir peux pouvez dois doit""".split())
OTHER_WORDS = set("""the and you your is are of to with what how ignore previous now all please el los las una por del
con para pero como esto esta muy il di che gli della sono per non anche questo der die das und ist nicht ein eine
ich du sie mit auf für zu den dem het een van zijn och att är som jag""".split()) - FR_WORDS


def is_french(text: str) -> bool:
    """Lexical heuristic (function words): at least 3 French words and more French than other-language words"""
    words = [w.lower() for w in _WORD.findall(text or "")]
    fr = sum(w in FR_WORDS for w in words)
    other = sum(w in OTHER_WORDS for w in words)
    accents = len(re.findall(r"[éèêàçùœ]", text or ""))
    return fr >= 3 and fr + min(accents, 3) > 1.5 * other or (fr >= 2 and other == 0 and accents >= 1)


def is_english(text: str) -> bool:
    words = [w.lower() for w in _WORD.findall(text or "")]
    en = set("the and you your is are of to with what how please this that for it be not".split())
    k = sum(w in en for w in words)
    return k >= 2 and k >= 0.15 * max(len(words), 1) and not re.search(r"[^\x00-\x7f’‘“”—–…]", text or "")
