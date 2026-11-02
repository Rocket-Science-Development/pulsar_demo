"""IM-02 - Cheap image drift: MNIST-C, 15 corruption types vs clean.

Dataset: MNIST-C (Mu & Gilmer 2019, CC BY 4.0; MNIST itself MIT-style), zip in /data/lab/datasets/mnist_c (outside the
repositories). Only the test_images/test_labels arrays of each corruption and the clean (identity) train/test arrays
are read; they are extracted to /data/lab/cache/im02 and loaded with allow_pickle=False.

Pulsar ships no image encoder, so the representation is the cheapest one a user would have: raw pixels (784), reduced
by the PCA that mmd_test fits on both windows pooled (n_components=32, its default). Each corruption is "explained" by
the accuracy drop of a clean-trained classifier: a corruption that MMD misses but that hurts accuracy is a limit.

    LAB_DRY=1 python -m lab.harness IM-02
    python -m lab.harness IM-02
"""

import os
import time
import zipfile
from pathlib import Path

import numpy as np

ZIP = Path(os.environ.get("LAB_DATA", "/data/lab/datasets")) / "mnist_c/mnist_c.zip"
CACHE = Path(os.environ.get("LAB_CACHE_IM", "/data/lab/cache/im02"))
DRY = os.environ.get("LAB_DRY") == "1"
WINDOWS = [20, 50, 100, 200]
N_REF = 500
REPS = 3 if DRY else 20
ALPHA = 0.05
PARAMS = {"windows": WINDOWS, "n_reference": N_REF, "reps": REPS, "representation": "raw pixels, mmd_test PCA 32",
          "alpha": ALPHA, "dry": DRY}
CRITERIA = {"corruptions_detected_or_explained": ("==", 15), "aa_false_alarm_rate_max": ("<=", 0.10)}


def extract():
    CACHE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if n.endswith(".npy") and ("/test_" in n or n.startswith("mnist_c/identity/"))]
        for n in names:
            target = CACHE / n.replace("mnist_c/", "").replace("/", "__")
            if not target.exists():
                with z.open(n) as src, open(target, "wb") as dst:
                    dst.write(src.read())
    return sorted({p.name.split("__")[0] for p in CACHE.glob("*__test_images.npy")})


def load(kind, split):
    x = np.load(CACHE / f"{kind}__{split}_images.npy", allow_pickle=False).reshape(-1, 784).astype(np.float32) / 255.0
    y = np.load(CACHE / f"{kind}__{split}_labels.npy", allow_pickle=False).astype(int)
    return x, y


def run(ctx):
    from pulsar_metrics.genai.mmd import mmd_test
    from sklearn.linear_model import LogisticRegression

    kinds = extract()
    corruptions = [k for k in kinds if k != "identity"]
    m = {"n_corruptions": len(corruptions), "corruptions": corruptions}
    rng = np.random.default_rng(20261102)
    xtr, ytr = load("identity", "train")
    xte, yte = load("identity", "test")

    ref_idx = rng.choice(len(xtr), N_REF, replace=False)
    ref = xtr[ref_idx]
    # classifier trained on clean train rows not used as the reference
    pool = np.setdiff1d(np.arange(len(xtr)), ref_idx)
    fit_idx = rng.choice(pool, 3000 if DRY else 20000, replace=False)
    clf = LogisticRegression(max_iter=300).fit(xtr[fit_idx], ytr[fit_idx])
    clean_acc = float(clf.score(xte, yte))
    m["clean_accuracy"] = round(clean_acc, 4)

    def rate(x, n):
        hits, ps, secs = 0, [], []
        for _ in range(REPS):
            cur = x[rng.choice(len(x), n, replace=False)]
            t0 = time.perf_counter()
            r = mmd_test(ref, cur, seed=int(rng.integers(1 << 31)))
            secs.append(time.perf_counter() - t0)
            hits += r.p_value < ALPHA
            ps.append(r.p_value)
        return hits / REPS, float(np.median(ps)), secs

    allsecs, aa = [], []
    for n in WINDOWS:
        r, p, s = rate(xte, n)
        aa.append(r)
        allsecs += s
        m[f"aa_power_n{n}"] = r
    m["aa_false_alarm_rate"] = float(np.mean(aa))

    table, missed_harmful, detected_or_explained = {}, [], 0
    for c in corruptions:
        x, y = load(c, "test")
        acc = float(clf.score(x, y))
        row = {"accuracy": round(acc, 4), "accuracy_drop": round(clean_acc - acc, 4),
               "mean_pixel_shift": round(float(x.mean() - xte.mean()), 4)}
        for n in WINDOWS:
            r, p, s = rate(x, n)
            allsecs += s
            row[f"power_n{n}"], row[f"median_p_n{n}"] = r, round(p, 4)
        table[c] = row
        detected = row[f"power_n{max(WINDOWS)}"] >= 0.8
        harmful = row["accuracy_drop"] >= 0.02
        # explained: missed and harmless (accuracy drop < 2 points) is acceptable silence
        if detected or not harmful:
            detected_or_explained += 1
        if not detected and harmful:
            missed_harmful.append(c)
    m["per_corruption"] = table
    m["corruptions_detected_or_explained"] = detected_or_explained
    m["mmd_seconds_per_test_p50"] = float(np.median(allsecs))
    derive(ctx, m)
    ctx.note("Representation: raw pixels; Pulsar ships no image encoder (IM-01 expected limit) - embedding is the user's.")
    return m


def derive(ctx, m, windows=WINDOWS):
    """Summary metrics and limits from the measured table (also used to re-derive limits from a saved result)"""
    table, big = m["per_corruption"], max(windows)
    for n in windows:
        m[f"corruptions_detected_n{n}"] = sum(t[f"power_n{n}"] >= 0.8 for t in table.values())
    m["aa_false_alarm_rate_max"] = max(m[f"aa_power_n{n}"] for n in windows)
    m["min_window_power80"] = {c: next((n for n in windows if t[f"power_n{n}"] >= 0.8), None) for c, t in table.items()}
    missed_harmful = [c for c, t in table.items() if t[f"power_n{big}"] < 0.8 and t["accuracy_drop"] >= 0.02]
    if missed_harmful:
        ctx.limit(
            f"Raw-pixel MMD (mmd_test, PCA 32) misses {len(missed_harmful)} of {len(table)} MNIST-C corruptions that "
            f"cut a clean classifier's accuracy by >= 2 points, on windows of {big} images",
            "; ".join(f"{c}: power {table[c][f'power_n{big}']}, accuracy drop {table[c]['accuracy_drop']}" for c in missed_harmful),
            "Ship an optional image embedding step or document it as the user's responsibility; pair MMD with "
            "performance/confidence monitoring (predicted-class mix, mean max-probability).",
            "major")
    weak = [c for c, w in m["min_window_power80"].items() if w is not None and w > 50]
    if weak:
        ctx.limit(
            f"{len(weak)} of {len(table)} MNIST-C corruptions need windows of more than 50 images for power >= 0.8 with "
            f"raw-pixel MMD, although they cut a clean classifier's accuracy by "
            f"{100 * min(table[c]['accuracy_drop'] for c in weak):.0f}-{100 * max(table[c]['accuracy_drop'] for c in weak):.0f} points; at 20 images "
            f"{len(table) - m['corruptions_detected_n20']} of {len(table)} are missed",
            "; ".join(f"{c} (accuracy drop {table[c]['accuracy_drop']}): " +
                      ", ".join(f"n{n} {table[c][f'power_n{n}']:.2f}" for n in windows) for c in weak),
            "Document a minimum window of 200 images for pixel-space MMD; offer an image encoder option (local, "
            "texture-sensitive features) since sparse local corruptions (spatter, dotted_line) barely move the "
            "PCA-32 pixel projection; monitor model confidence alongside.",
            "minor")
    if m["aa_false_alarm_rate_max"] > 0.10:
        ctx.limit(
            f"A/A false alarm rate {m['aa_false_alarm_rate_max']:.2f} at window {big} on clean MNIST (reference = 500 train "
            f"images, windows = clean test images), above the nominal 0.05",
            f"A/A rate by window: " + ", ".join(f"n{n} {m[f'aa_power_n{n}']:.2f}" for n in windows) +
            f" ({REPS} windows each); plausibly the MNIST train/test writer difference, a real split difference that "
            "the uncalibrated permutation test reports as drift once windows are large",
            "Calibrate MMD with AACalibrator on reference-vs-holdout splits before alerting, and build the reference "
            "from the same source as production windows; report the A/A level with each MMD verdict.",
            "minor")
