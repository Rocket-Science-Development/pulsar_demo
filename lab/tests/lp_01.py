"""LP-01 — Personal information spans in FR/EN finance documents: PiiDetector rules alone vs rules + GLiNER.

Dataset: gretelai/synthetic_pii_finance_multilingual (synthetic documents, Apache 2.0), English and French
("France") documents, gold spans with labels. Gold labels are mapped to Pulsar types; a gold span is found when a
finding of the mapped type overlaps it. Precision counts findings that overlap a gold span of a compatible label
(gold annotations may be incomplete: precision is a lower bound). Labels without a Pulsar type are reported as
coverage (any finding overlapping). Only counts are kept: no value, no text, is ever written.

ModelPiiDetector loads GLiNER on CPU (no device option); latency is measured as shipped on CPU, then the model is
moved to the GPU by the test to score the rest in reasonable time.
"""

import json
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from ._llm_common import DATA, DRY, GLINER_MODEL, offline, wilson

GLINER_LOCAL = "/data/lab/work/gliner_multi_pii-v1-local"  # copy whose backbone points to the local mdeberta-v3-base
CRITERIA = {"names_recall_gain": (">", 0.0), "addresses_recall_gain": (">", 0.0), "values_written": ("==", 0)}
PARAMS = {"docs_per_language": 800, "cpu_latency_docs": 60, "gliner_threshold": 0.5, "seed": 0, "dry": DRY}

MAP = {  # gold label -> Pulsar type
    "email": "email", "phone_number": "phone", "credit_card_number": "credit_card", "ssn": "sin",
    "passport_number": "passport", "driver_license_number": "drivers_licence_qc", "date_of_birth": "date_of_birth",
    "name": "person_name", "first_name": "person_name", "last_name": "person_name", "street_address": "street_address",
}
RULE_TYPES = {"email", "phone", "credit_card", "sin", "passport", "drivers_licence_qc", "date_of_birth", "ramq", "postal_code"}
COMPATIBLE = defaultdict(set)
for g, t in MAP.items():
    COMPATIBLE[t].add(g)
COMPATIBLE["person_name"] |= {"user_name"}
COMPATIBLE["date_of_birth"] |= {"date"}
COMPATIBLE["postal_code"] |= {"street_address"}
COMPATIBLE["credit_card"] |= {"bban", "iban"}
_WORDS = re.compile(r"\w+(?:[-_]\w+)*|\S")


def load():
    d = DATA / "gretelai" / "synthetic_pii_finance_multilingual" / "data"
    frames = [pd.read_parquet(f) for f in (d / "combined_test-00000-of-00001.parquet", d / "combined_train-00000-of-00001.parquet")]
    df = pd.concat(frames)
    out = []
    n = 40 if DRY else PARAMS["docs_per_language"]
    for lang, key in (("en", "English"), ("fr", "France")):
        g = df[df.language == key].drop_duplicates("generated_text")
        out.append(g.sample(min(n, len(g)), random_state=PARAMS["seed"]).assign(lang=lang))
    return pd.concat(out).reset_index(drop=True)


def overlap(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


def evaluate(docs, findings_per_doc):
    """Recall per gold label and precision per predicted type"""
    rec, prec, cover = Counter(), Counter(), Counter()
    wrong = defaultdict(Counter)  # findings overlapping a gold span of an incompatible label
    gold_n, pred_n = Counter(), Counter()
    pos = defaultdict(Counter)  # person/address recall by word position (GLiNER window = 384 words)
    for (text, spans), finds in zip(docs, findings_per_doc):
        for s in spans:
            lab = s["label"]
            gold_n[lab] += 1
            any_hit = any(overlap(f.start, f.end, s["start"], s["end"]) for f in finds)
            cover[lab] += any_hit
            if lab in MAP:
                hit = any(f.type == MAP[lab] and overlap(f.start, f.end, s["start"], s["end"]) for f in finds)
                rec[lab] += hit
                if MAP[lab] in ("person_name", "street_address"):
                    w = len(_WORDS.findall(text[: s["start"]]))
                    bucket = "le_384_words" if w <= 384 else "gt_384_words"
                    pos[bucket]["n"] += 1
                    pos[bucket]["hit"] += hit
        for f in finds:
            pred_n[f.type] += 1
            ok = any(s["label"] in COMPATIBLE[f.type] and overlap(f.start, f.end, s["start"], s["end"]) for s in spans)
            prec[f.type] += ok
            if not ok:
                labs = [s["label"] for s in spans if overlap(f.start, f.end, s["start"], s["end"])]
                wrong[f.type][labs[0] if labs else "<unannotated>"] += 1
    recall = {l: {"n": gold_n[l], "recall": round(rec[l] / gold_n[l], 4), "ci": wilson(rec[l], gold_n[l])}
              for l in sorted(gold_n) if l in MAP}
    precision = {t: {"n": pred_n[t], "precision": round(prec[t] / pred_n[t], 4), "ci": wilson(prec[t], pred_n[t]),
                     "unannotated": wrong[t]["<unannotated>"],
                     "wrong_label": {l: c for l, c in wrong[t].most_common(5) if l != "<unannotated>"},
                     "wrong_label_share": round(sum(c for l, c in wrong[t].items() if l != "<unannotated>") / pred_n[t], 4)}
                 for t in sorted(pred_n)}
    coverage = {l: {"n": gold_n[l], "any_finding": round(cover[l] / gold_n[l], 4)} for l in sorted(gold_n) if l not in MAP}
    position = {k: {"n": v["n"], "recall": round(v["hit"] / v["n"], 4) if v["n"] else None} for k, v in pos.items()}
    grouped = {}
    for name, labels in (("names", ("name", "first_name", "last_name")), ("addresses", ("street_address",))):
        n = sum(gold_n[l] for l in labels)
        k = sum(rec[l] for l in labels)
        grouped[name] = {"n": n, "recall": round(k / n, 4) if n else None, "ci": wilson(k, n)}
    unmapped = sum(gold_n[l] for l in gold_n if l not in MAP)
    return {"recall": recall, "precision": precision, "coverage_unmapped": coverage, "grouped": grouped,
            "by_position": position, "gold_spans": sum(gold_n.values()), "unmapped_share": round(unmapped / max(sum(gold_n.values()), 1), 4)}


def run(ctx):
    offline()
    from pulsar_data_collection.genai import ModelPiiDetector
    from pulsar_data_collection.privacy import PiiDetector

    df = load()
    docs = {lang: [(t, json.loads(s)) for t, s in g[["generated_text", "pii_spans"]].itertuples(index=False)]
            for lang, g in df.groupby("lang")}
    rules = PiiDetector()
    t0 = time.perf_counter()
    model = ModelPiiDetector(model=GLINER_LOCAL, threshold=PARAMS["gliner_threshold"], rules=rules)
    load_s = time.perf_counter() - t0
    out = {"model_load_s": round(load_s, 1), "gliner_device_as_shipped": str(next(model._model.parameters()).device)}

    # latency as shipped (CPU), on the first docs of each language
    cpu_lat, rule_lat = [], []
    k = min(PARAMS["cpu_latency_docs"], 10 if DRY else PARAMS["cpu_latency_docs"])
    for lang in docs:
        for text, _ in docs[lang][:k]:
            t = time.perf_counter()
            model.scan(text)
            cpu_lat.append(time.perf_counter() - t)
    try:
        import torch

        if torch.cuda.is_available():
            model._model.to("cuda")  # test-side: ModelPiiDetector has no device option
            out["gliner_device_for_bulk"] = "cuda"
    except Exception as exc:  # noqa: BLE001
        ctx.note(f"GLiNER could not be moved to the GPU: {exc}")

    results = {}
    gpu_lat = []
    for lang, items in docs.items():
        f_rules, f_both = [], []
        for text, _ in items:
            t = time.perf_counter()
            f_rules.append(rules.scan(text))
            rule_lat.append(time.perf_counter() - t)
            t = time.perf_counter()
            f_both.append(model.scan(text))
            gpu_lat.append(time.perf_counter() - t)
        results[lang] = {"docs": len(items), "rules": evaluate(items, f_rules), "rules_plus_gliner": evaluate(items, f_both),
                         "chars_p50": int(np.median([len(t) for t, _ in items])),
                         "docs_over_384_words": int(sum(len(_WORDS.findall(t)) > 384 for t, _ in items))}

    def g(lang, mode, name):
        return results[lang][mode]["grouped"][name]["recall"] or 0.0

    gains = {name: round(min(g(l, "rules_plus_gliner", name) - g(l, "rules", name) for l in results), 4) for name in ("names", "addresses")}
    metrics = {
        **out, "per_language": results, "names_recall_gain": gains["names"], "addresses_recall_gain": gains["addresses"],
        "latency_ms_rules_p50": round(1000 * float(np.median(rule_lat)), 3),
        "latency_ms_gliner_cpu_p50": round(1000 * float(np.median(cpu_lat)), 1),
        "latency_ms_gliner_cpu_p95": round(1000 * float(np.percentile(cpu_lat, 95)), 1),
        "latency_ms_gliner_gpu_p50": round(1000 * float(np.median(gpu_lat)), 1),
        "values_written": 0,
    }

    # ---- limits
    for lang, r in results.items():
        both = r["rules_plus_gliner"]
        for name in ("names", "addresses"):
            v = both["grouped"][name]
            if v["recall"] is not None and v["recall"] < 0.8:
                ctx.limit(f"{lang.upper()} {name} recall with rules + GLiNER is {v['recall']} (rules alone "
                          f"{r['rules']['grouped'][name]['recall']})",
                          f"n={v['n']} gold spans, Wilson CI {v['ci']}; threshold {PARAMS['gliner_threshold']}",
                          "Lower the GLiNER threshold for names/addresses per language (calibrate on labelled FR/EN), "
                          "and add French address cues (rue, avenue, boul., app.) as a rule recognizer.", "major")
        weak = {l: v for l, v in both["recall"].items() if l not in ("name", "first_name", "last_name", "street_address")
                and v["n"] >= 20 and v["recall"] < 0.8}
        if weak:
            ctx.limit(f"{lang.upper()} rule types with recall < 0.8 on finance documents: " +
                      ", ".join(f"{l} {v['recall']} (n={v['n']})" for l, v in weak.items()),
                      "per-label recall in results/LP-01.json (per_language.*.rules_plus_gliner.recall)",
                      "The recognizers are Canada-specific (SIN Luhn, NANP phones, QC licence): add international "
                      "formats as optional recognizers (E.164 phones, US SSN, passport formats) behind a locale option.",
                      "major")
        not_personal = {"company", "date", "time", "date_time"}  # business data, not personal information
        big = {l: v for l, v in both["coverage_unmapped"].items() if l not in not_personal and v["n"] >= 20 and v["any_finding"] < 0.5}
        if big:
            n_big = sum(v["n"] for v in big.values())
            ctx.limit(f"{lang.upper()} sensitive finance identifiers without a Pulsar type are not detected ({n_big} gold spans): " +
                      ", ".join(f"{l} (n={v['n']}, any finding {v['any_finding']})" for l, v in sorted(big.items(), key=lambda x: -x[1]['n'])),
                      "coverage_unmapped in results/LP-01.json (company/date/time excluded: not personal information)",
                      "Add finance recognizers with validation (IBAN mod-97, SWIFT/BIC, account and routing numbers "
                      "with context, customer/employee ids with context, CVV next to a card) behind a 'finance' profile.", "major")
        low = {t: v for t, v in both["precision"].items() if v["n"] >= 20 and v["wrong_label_share"] > 0.1}
        if low:
            ctx.limit(f"{lang.upper()} findings typed wrongly (overlap a gold span of another label): " +
                      ", ".join(f"{t} {v['wrong_label_share']} of n={v['n']} ({v['wrong_label']})" for t, v in low.items()),
                      "findings on unannotated text are excluded (gretel gold is incomplete: e.g. company e-mails)",
                      "For GLiNER, add 'organization' as a negative label (predict it and drop it) so that companies "
                      "are not reported as person names; for rules, see the confused labels.", "minor")
        pos = both["by_position"]
        if pos.get("gt_384_words", {}).get("n", 0) >= 20 and pos["gt_384_words"]["recall"] < (pos.get("le_384_words", {}).get("recall") or 0) - 0.1:
            ctx.limit(f"{lang.upper()} names/addresses after the 384-word GLiNER window are missed: recall "
                      f"{pos['gt_384_words']['recall']} vs {pos['le_384_words']['recall']}",
                      f"n beyond window {pos['gt_384_words']['n']}", "Chunk long texts into windows of <= 384 words with "
                      "overlap in ModelPiiDetector.scan and shift the offsets back.", "major")
    if metrics["latency_ms_gliner_cpu_p50"] > 200:
        ctx.limit(f"ModelPiiDetector runs GLiNER on CPU only: p50 {metrics['latency_ms_gliner_cpu_p50']} ms / doc "
                  f"(p95 {metrics['latency_ms_gliner_cpu_p95']}), vs {metrics['latency_ms_gliner_gpu_p50']} ms on the T4 "
                  "after a manual move", f"{len(cpu_lat)} docs on 4 vCPU; rules alone p50 {metrics['latency_ms_rules_p50']} ms",
                  "Add a `device` argument to ModelPiiDetector (like InjectionDetector) and batch predict_entities.", "minor")
    # Offline load from the UNPATCHED local copy, actually attempted (the first run asserted the failure without trying):
    # once with the host's Hugging Face cache, once with an empty cache (true air-gapped host). Limit only on failure.
    offline_load = {"host_cache": offline_load_attempt(GLINER_MODEL, empty_cache=False),
                    "empty_cache": offline_load_attempt(GLINER_MODEL, empty_cache=True)}
    metrics["offline_load_unpatched"] = offline_load
    failed = {k: v for k, v in offline_load.items() if v["ok"] is False}
    if failed:
        ctx.limit("ModelPiiDetector cannot load offline from an unpatched local GLiNER copy (HF_HUB_OFFLINE=1): " +
                  "; ".join(f"{k}: {v['error']}" for k, v in failed.items()),
                  f"copy {GLINER_MODEL} as downloaded; subprocess load with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 "
                  f"({', '.join(failed)}); the patched copy {GLINER_LOCAL} loads in {load_s:.0f} s",
                  "Document air-gapped installation (download the backbone, rewrite model_name) or accept a "
                  "`backbone_path` argument; air-gapped deployments are common in the public sector.", "minor")
    else:
        ctx.note(f"Unpatched GLiNER copy loads offline: {offline_load}")
    return metrics


def offline_load_attempt(model_dir, empty_cache, timeout_s=900):
    """Load ModelPiiDetector from `model_dir` in a fresh process with the hub disabled; returns ok/error/seconds"""
    import os
    import subprocess
    import sys
    import tempfile

    if not os.path.isdir(model_dir):
        return {"ok": None, "error": f"copy not found: {model_dir}", "seconds": 0.0}
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    tmp = tempfile.TemporaryDirectory() if empty_cache else None
    if tmp:
        env["HF_HOME"] = tmp.name
        env.pop("HF_HUB_CACHE", None)
        env.pop("TRANSFORMERS_CACHE", None)
    code = ("import sys\n"
            "from pulsar_data_collection.genai import ModelPiiDetector\n"
            "try:\n"
            "    ModelPiiDetector(model=sys.argv[1])\n"
            "except Exception as e:\n"
            "    print('ERR', type(e).__name__, str(e).splitlines()[0][:300] if str(e) else '')\n"
            "    sys.exit(3)\n")
    t0 = time.perf_counter()
    try:
        r = subprocess.run([sys.executable, "-c", code, model_dir], env=env, capture_output=True, text=True, timeout=timeout_s)
        err = next((l[4:] for l in r.stdout.splitlines() if l.startswith("ERR ")), None)
        if r.returncode != 0 and err is None:
            err = (r.stderr.strip().splitlines() or ["exit code %d" % r.returncode])[-1][:300]
        return {"ok": r.returncode == 0, "error": err, "seconds": round(time.perf_counter() - t0, 1)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout_s} s", "seconds": float(timeout_s)}
    finally:
        if tmp:
            tmp.cleanup()
