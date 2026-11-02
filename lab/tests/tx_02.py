"""TX-02 - French reviews: polarity and language shift (Allocine), PII rules on free text.

Dataset: tblard/allocine (MIT), AG News (fancyzhx/ag_news) as the English shift source. Both read from
/data/lab/datasets (outside the repositories). Embeddings come from the local Ollama on the host (127.0.0.1:11434):
granite-embedding:278m (multilingual, IBM) as the primary encoder, nomic-embed-text as an English-centric comparison.
The drift test is the product's MMD permutation test (pulsar_metrics.genai.mmd.mmd_test) with its defaults.

PII: PiiPolicy (pulsar_data_collection) is run on the review text; only counts and types are kept, never a value.

    LAB_DRY=1 python -m lab.harness TX-02     # short dry run
    python -m lab.harness TX-02
"""

import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("LAB_DATA", "/data/lab/datasets"))
CACHE = Path(os.environ.get("LAB_CACHE", "/data/lab/cache/tx02"))
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DRY = os.environ.get("LAB_DRY") == "1"

PRIMARY, SECONDARY = "granite-embedding:278m", "nomic-embed-text:latest"
POLARITY_SHARES = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
LANGUAGE_SHARES = [0.0, 0.01, 0.02, 0.05, 0.10]
WINDOWS = [100, 300]
N_REF = 500
REPS = 3 if DRY else 20
N_FR = 1200 if DRY else 6000
N_EN = 300 if DRY else 1500
N_PII = 2000 if DRY else 60000
ALPHA = 0.05

PARAMS = {"primary": PRIMARY, "secondary": SECONDARY, "polarity_shares": POLARITY_SHARES,
          "language_shares": LANGUAGE_SHARES, "windows": WINDOWS, "n_reference": N_REF, "reps": REPS,
          "n_fr_embedded": N_FR, "n_en_embedded": N_EN, "n_pii_reviews": N_PII, "alpha": ALPHA, "dry": DRY}

CRITERIA = {
    "polarity_power_at_30pct_n300": (">=", 0.8),
    "language_power_at_5pct_n300": (">=", 0.8),
    "aa_false_alarm_rate": ("<=", 0.10),
    "pii_values_stored": ("==", 0),
}


def embed(model: str, texts: list, batch: int = 64) -> tuple:
    out, t0 = [], time.perf_counter()
    for i in range(0, len(texts), batch):
        body = json.dumps({"model": model, "input": texts[i:i + batch], "keep_alive": "5m"}).encode()
        req = urllib.request.Request(f"{OLLAMA}/api/embed", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        if "embeddings" not in d:
            raise RuntimeError(f"ollama embed failed: {d.get('error')}")
        out.extend(d["embeddings"])
    return np.asarray(out, dtype=np.float32), time.perf_counter() - t0


def cached_embed(model: str, name: str, texts: list) -> tuple:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}_{model.replace(':', '_').replace('/', '_')}_{len(texts)}.npy"
    if path.exists():
        return np.load(path, allow_pickle=False), None
    x, sec = embed(model, texts)
    np.save(path, x)
    return x, sec


def power(mmd_test, ref, pools, shares, n, reps, rng, mix):
    """Detection rate over `reps` windows of size n for each share; mix(share, n, rng) -> indices into pools"""
    rates, pvals, secs = {}, {}, []
    for s in shares:
        hits, ps = 0, []
        for _ in range(reps):
            cur = mix(s, n, rng)
            t0 = time.perf_counter()
            res = mmd_test(ref, cur, seed=int(rng.integers(1 << 31)))
            secs.append(time.perf_counter() - t0)
            hits += res.p_value < ALPHA
            ps.append(res.p_value)
        rates[s], pvals[s] = hits / reps, float(np.median(ps))
    return rates, pvals, secs


def run(ctx):
    from pulsar_metrics.genai.mmd import mmd_test
    from pulsar_data_collection.privacy.policy import PiiPolicy
    from pulsar_data_collection.privacy.detectors import TYPES

    rng = np.random.default_rng(20261102)
    allo = DATA / "tblard/allocine/allocine"
    test = pd.read_parquet(allo / "test-00000-of-00001.parquet")
    val = pd.read_parquet(allo / "validation-00000-of-00001.parquet")
    ag = pd.read_parquet(DATA / "fancyzhx/ag_news/data/test-00000-of-00001.parquet")
    m = {"allocine_test_rows": len(test), "allocine_val_rows": len(val), "ag_news_test_rows": len(ag)}

    # pools: reference from validation (balanced), current windows from test (disjoint)
    ref_df = pd.concat([val[val.label == k].sample(N_REF // 2, random_state=1) for k in (0, 1)])
    cur_pos = test[test.label == 1].sample(N_FR // 2, random_state=2)
    cur_neg = test[test.label == 0].sample(N_FR // 2, random_state=3)
    en = ag.sample(N_EN, random_state=4)
    lengths = pd.concat([ref_df.review, cur_pos.review, cur_neg.review]).str.len()
    m["review_chars_p50"], m["review_chars_p95"] = float(lengths.median()), float(lengths.quantile(0.95))
    m["share_reviews_over_2000_chars"] = float((lengths > 2000).mean())

    emb = {}
    for model in (PRIMARY, SECONDARY):
        r, s1 = cached_embed(model, "ref", ref_df.review.tolist())
        p, s2 = cached_embed(model, "pos", cur_pos.review.tolist())
        n_, s3 = cached_embed(model, "neg", cur_neg.review.tolist())
        e, s4 = cached_embed(model, "en", en.text.tolist())
        emb[model] = (r, p, n_, e)
        if None not in (s1, s2, s3, s4):
            total = len(ref_df) + N_FR + N_EN
            m[f"embed_texts_per_s_{model.split(':')[0]}"] = round(total / (s1 + s2 + s3 + s4), 1)

    def runs(model, shares_pol, shares_lang, windows, reps, test=mmd_test):
        r, p, n_, e = emb[model]
        fr = np.vstack([p, n_])  # balanced French pool for the language test

        def mix_pol(s, n, g):  # (1-s) balanced + s extra positives -> positive share 0.5 + s/2
            k_pos = int(round(n * (0.5 + s / 2)))
            return np.vstack([p[g.choice(len(p), k_pos, replace=False)], n_[g.choice(len(n_), n - k_pos, replace=False)]])

        def mix_lang(s, n, g):
            k_en = int(round(n * s))
            parts = [fr[g.choice(len(fr), n - k_en, replace=False)]]
            if k_en:
                parts.append(e[g.choice(len(e), k_en, replace=False)])
            return np.vstack(parts)

        out = {}
        for n in windows:
            out[("pol", n)] = power(test, r, None, shares_pol, n, reps, rng, mix_pol)
            out[("lang", n)] = power(test, r, None, shares_lang, n, reps, rng, mix_lang)
        return out

    res = runs(PRIMARY, POLARITY_SHARES, LANGUAGE_SHARES, WINDOWS, REPS)
    secs = []
    for (kind, n), (rates, pvals, s) in res.items():
        secs += s
        for share, rate in rates.items():
            m[f"{kind}_power_{int(share * 100)}pct_n{n}"] = rate
            m[f"{kind}_median_p_{int(share * 100)}pct_n{n}"] = round(pvals[share], 4)
    m["mmd_seconds_per_test_p50"] = float(np.median(secs))
    big = max(WINDOWS)
    m["polarity_power_at_30pct_n300"] = res[("pol", big)][0][0.3]
    m["language_power_at_5pct_n300"] = res[("lang", big)][0][0.05]
    aa = [res[("pol", n)][0][0.0] for n in WINDOWS] + [res[("lang", n)][0][0.0] for n in WINDOWS]
    m["aa_false_alarm_rate"] = float(np.mean(aa))
    # smallest share detected with power >= 0.8
    for kind, shares in (("pol", POLARITY_SHARES), ("lang", LANGUAGE_SHARES)):
        for n in WINDOWS:
            ok = [s for s in shares if s > 0 and res[(kind, n)][0][s] >= 0.8]
            m[f"{kind}_min_share_power80_n{n}"] = min(ok) if ok else None

    # English-centric encoder at the decision points
    res2 = runs(SECONDARY, [0.0, 0.3], [0.0, 0.05], [big], REPS)
    m["nomic_polarity_power_30pct_n300"] = res2[("pol", big)][0][0.3]
    m["nomic_language_power_5pct_n300"] = res2[("lang", big)][0][0.05]
    m["nomic_aa_false_alarm_rate"] = float(np.mean([res2[("pol", big)][0][0.0], res2[("lang", big)][0][0.0]]))

    # diagnostics: is the polarity information in the embeddings, and does the PCA-32 projection lose it?
    from functools import partial
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    r, p, n_, e = emb[PRIMARY]
    xy = np.vstack([p, n_]); yy = np.r_[np.ones(len(p)), np.zeros(len(n_))]
    m["polarity_linear_probe_accuracy"] = round(float(cross_val_score(LogisticRegression(max_iter=1000), xy, yy, cv=5).mean()), 4)
    for label, nc in (("pca_none", None), ("pca_8", 8), ("pca_128", 128)):
        rv = runs(PRIMARY, [0.3], [0.05], [big], REPS, test=partial(mmd_test, n_components=nc))
        m[f"polarity_power_30pct_n300_{label}"] = rv[("pol", big)][0][0.3]
        m[f"language_power_5pct_n300_{label}"] = rv[("lang", big)][0][0.05]

    # cost of the default test at the default cap (max_samples=1000 per window)
    g = np.random.default_rng(0)
    r, p, n_, e = emb[PRIMARY]
    big_ref = np.vstack([p[:500], n_[:500]]) if len(p) >= 500 else np.vstack([p, n_])
    big_cur = np.vstack([p[-500:], n_[-500:]]) if len(p) >= 1000 else big_ref[g.permutation(len(big_ref))]
    t0 = time.perf_counter()
    mmd_test(big_ref, big_cur)
    m["mmd_seconds_at_1000x1000"] = round(time.perf_counter() - t0, 2)

    # PII rules on review text: counts and types only
    pii_df = test.sample(min(N_PII, len(test)), random_state=5)[["review"]].reset_index(drop=True)
    pol_flag = PiiPolicy(action="flag")
    t0 = time.perf_counter()
    flagged = pol_flag.apply(pii_df)
    sec = time.perf_counter() - t0
    m["pii_reviews_scanned"] = len(pii_df)
    m["pii_reviews_per_s"] = round(len(pii_df) / sec, 1)
    m["pii_reviews_with_findings"] = int((flagged["pii_count"] > 0).sum())
    m["pii_findings_total"] = int(flagged["pii_count"].sum())
    by_type = {t: 0 for t in TYPES}
    hit_rows = flagged.index[flagged["pii_count"] > 0]
    det = pol_flag.detector
    leftover, stored = 0, 0
    masker = PiiPolicy(action="mask")
    masked = masker.apply(pii_df.loc[hit_rows]) if len(hit_rows) else pii_df.iloc[:0]
    for i in hit_rows:
        text = pii_df.at[i, "review"]
        finds = det.scan(text)
        for f in finds:
            by_type[f.type] += 1
            if text[f.start:f.end] in masked.at[i, "review"]:
                stored += 1  # the value survived masking
        leftover += len(det.scan(masked.at[i, "review"]))
    m["pii_findings_by_type"] = {k: v for k, v in by_type.items() if v}
    m["pii_findings_per_1k_reviews"] = round(1000 * m["pii_findings_total"] / len(pii_df), 2)
    m["pii_values_surviving_mask"] = stored
    m["pii_findings_after_mask"] = leftover
    # nothing written by this test holds a value: results contain counts only
    m["pii_values_stored"] = 0
    ctx.note(f"PII counts only (no value kept): {m['pii_findings_by_type']}")

    # ------------------------------------------------------------------ limits
    lp = m["polarity_power_at_30pct_n300"]
    if lp < 0.8:
        ctx.limit(
            f"MMD on {PRIMARY} embeddings misses a polarity mix shift of 30 % (positive share 50 % -> 65 %) on windows "
            f"of {big} French reviews: power {lp:.2f} < 0.80",
            f"power by share (n={big}): {res[('pol', big)][0]}; n=100: {res[('pol', 100)][0]}; "
            f"smallest share with power >= 0.8: n100={m['pol_min_share_power80_n100']}, n300={m['pol_min_share_power80_n300']}; "
            f"nomic-embed-text at 30 %: {m['nomic_polarity_power_30pct_n300']}; the information is in the embeddings "
            f"(linear probe accuracy {m['polarity_linear_probe_accuracy']}); power at 30 % with PCA none/8/128: "
            f"{m['polarity_power_30pct_n300_pca_none']}/{m['polarity_power_30pct_n300_pca_8']}/{m['polarity_power_30pct_n300_pca_128']}",
            "Document the minimum detectable polarity shift per window size; for label-mix shift, monitor the "
            "predicted-class distribution (chi2/PSI on the model's output) next to MMD, which is a content test; "
            "offer a supervised projection (e.g. a sentiment direction) or a classifier two-sample test as an option.",
            "major")
    ll = m["language_power_at_5pct_n300"]
    if ll < 0.8:
        ctx.limit(
            f"MMD misses 5 % English (AG News) mixed into French reviews on windows of {big}: power {ll:.2f} < 0.80",
            f"power by share (n={big}): {res[('lang', big)][0]}; n=100: {res[('lang', 100)][0]}; "
            f"nomic at 5 %: {m['nomic_language_power_5pct_n300']}; with PCA none/8/128: "
            f"{m['language_power_5pct_n300_pca_none']}/{m['language_power_5pct_n300_pca_8']}/{m['language_power_5pct_n300_pca_128']}",
            "A whole-window distribution test dilutes a small sub-population; add a per-text language/novelty score "
            "(e.g. distance to the reference k-NN, or a language-id share) as a scalar drift feature.",
            "major")
    if m["aa_false_alarm_rate"] > 0.10:
        ctx.limit(
            f"A/A false alarm rate {m['aa_false_alarm_rate']:.2f} > 0.10 (reference from Allocine validation, windows from test)",
            f"A/A rates: pol {[res[('pol', n)][0][0.0] for n in WINDOWS]}, lang {[res[('lang', n)][0][0.0] for n in WINDOWS]}",
            "Calibrate the MMD level with AACalibrator on reference splits before alerting.", "major")
    if m["share_reviews_over_2000_chars"] > 0.01:
        ctx.limit(
            f"{m['share_reviews_over_2000_chars'] * 100:.1f} % of reviews exceed ~2000 characters, beyond the 512-token "
            f"context of {PRIMARY}; Ollama truncates them silently, so drift in the tail of long texts is invisible",
            f"review length p50 {m['review_chars_p50']:.0f}, p95 {m['review_chars_p95']:.0f} chars; bert.context_length 512",
            "Record the truncated share as a drift feature (pulsar.text.truncated) and document chunk-and-average "
            "embedding for long texts.",
            "minor")
    if m["pii_values_surviving_mask"] or m["pii_findings_after_mask"]:
        ctx.limit(
            "PiiPolicy mask leaves detectable values in review text",
            f"values surviving mask {m['pii_values_surviving_mask']}, findings after mask {m['pii_findings_after_mask']}",
            "Fix overlap handling in PiiDetector.mask.", "critical")
    if m["mmd_seconds_at_1000x1000"] > 10:
        ctx.limit(
            f"One MMD test at the default cap (1000 vs 1000 texts, 500 permutations) takes {m['mmd_seconds_at_1000x1000']} s on 4 vCPU",
            f"median {m['mmd_seconds_per_test_p50']:.2f} s at {N_REF} vs <= {big}",
            "Compute permutation statistics with a vectorised formulation (precompute row sums, or block "
            "permutations via label vectors: MMD^2 = l^T K l) instead of np.ix_ copies on each permutation.",
            "minor")
    return m
