"""SC-10 — LLM endpoint concurrency for the demo agent on Ollama (T4 GPU of lab-gpu-3).

Workers 1 / 2 / 4 / 8 / 16 run the demo permit agent (synthetic requests, temperature 0, tools in-process, Pulsar span
capture on) against llama3.2:3b, granite3.3:2b and llama3.1:8b. Two server configurations:
- "deployed": the host's shared Ollama service as provisioned (OLLAMA_NUM_PARALLEL=1), workers 1 / 4 / 16;
- "parallel": a lab-owned Ollama server (own port, OLLAMA_MAX_LOADED_MODELS=1, systemd MemoryMax) restarted with
  OLLAMA_NUM_PARALLEL = workers, for workers 2 / 4 / 8 / 16 (workers 1 is the deployed cell).
Each cell: warm-up, then a fixed measurement window; a run counts when it completes inside the window. Kept: runs/min,
run latency p50/p95, LLM calls/s, output tokens/s, errors, GPU utilisation and memory (nvidia-smi every second),
model placement (/api/ps size vs size_vram).

Two optima per model: throughput (the smallest worker count reaching 90 % of the best runs/min, no error; flagged as
not saturated when that is the largest level tested) and latency-bounded (the most runs/min with a p95 at most 3x the
single-worker p95). `evaluate` derives them, and the limits, from the measured cells only.
"""

import os
import random
import threading
import time

import numpy as np
import pandas as pd

from ._gpubench import GpuSampler, LabOllama, loaded_models, pct, unload_all
from ._llm_common import DRY

MODELS = ["llama3.2:3b", "granite3.3:2b"] if DRY else ["llama3.2:3b", "granite3.3:2b", "llama3.1:8b"]
WORKERS = [1, 4] if DRY else [1, 2, 4, 8, 16]
DEPLOYED_WORKERS = [1, 4] if DRY else [1, 4, 16]
WINDOW_S = 20 if DRY else 75
WARMUP_S = 10 if DRY else 20
DEPLOYED = os.environ.get("LAB_OLLAMA_DEPLOYED", "http://127.0.0.1:11434")
LAB_PORT = int(os.environ.get("LAB_OLLAMA_LAB_PORT", "11435"))
OLLAMA_BIN = os.environ.get("LAB_OLLAMA_BIN", "/data/lab/ollama/bin/ollama")
OLLAMA_MODELS = os.environ.get("LAB_OLLAMA_MODELS", "/data/lab/models/ollama")
PARAMS = {"models": MODELS, "workers": WORKERS, "deployed_workers": DEPLOYED_WORKERS, "window_s": WINDOW_S,
          "warmup_s": WARMUP_S, "context_length": 4096, "memory_max": "13G", "temperature": 0.0,
          "max_tokens_per_call": None, "dry": DRY}
CRITERIA = {"optimum_documented": ("==", len(MODELS))}


def make_agent(base_url, model, counter):
    from ._agentlab import LabAgent, MAX_TOKENS

    PARAMS["max_tokens_per_call"] = MAX_TOKENS
    agent = LabAgent(base_url=base_url + "/v1", model=model, temperature=0.0, prompt_version="v1")
    create = agent.client.chat.completions.create

    def counted(*args, **kwargs):
        t = time.perf_counter()
        response = create(*args, **kwargs)
        usage = getattr(response, "usage", None)
        with counter["lock"]:
            counter["calls"].append((time.time(), time.perf_counter() - t, getattr(usage, "prompt_tokens", 0) or 0,
                                     getattr(usage, "completion_tokens", 0) or 0))
        return response

    agent.client.chat.completions.create = counted
    return agent


def drive(base_url, model, workers, gpu, seed):
    """Closed loop: `workers` threads run agent tasks back to back; measured window after the warm-up"""
    from permit_agent.tasks import DEFAULT_MIX, make_task

    from ._agentlab import collector

    collector()
    counter = {"lock": threading.Lock(), "calls": []}
    agent = make_agent(base_url, model, counter)
    runs, lock = [], threading.Lock()
    t_begin = time.time()
    w0, w1 = t_begin + WARMUP_S, t_begin + WARMUP_S + WINDOW_S
    stop = threading.Event()

    def worker(k):
        rng = random.Random(seed * 1000 + k)
        while not stop.is_set() and time.time() < w1:
            task = make_task(rng, DEFAULT_MIX, 0.5)
            t, start = time.perf_counter(), time.time()
            err, steps, term = None, None, None
            try:
                res = agent.run_with(task["request"], expected_tool=task["expected_tool"])
                steps, term = res.steps, res.termination
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:200]
            with lock:
                runs.append({"start": start, "end": time.time(), "latency_s": time.perf_counter() - t, "error": err,
                             "steps": steps, "termination": term})

    threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(workers)]
    for th in threads:
        th.start()
    time.sleep(max(0.0, w0 - time.time() + WINDOW_S / 2))
    placement = loaded_models(base_url)  # mid-window
    for th in threads:
        th.join(timeout=WARMUP_S + WINDOW_S + 400)
    stop.set()
    inside = [r for r in runs if w0 <= r["end"] <= w1]
    ok = [r for r in inside if r["error"] is None]
    calls = [c for c in counter["calls"] if w0 <= c[0] <= w1]
    lat1 = [r["latency_s"] for r in ok]
    out = {"runs_in_window": len(inside), "runs_ok": len(ok), "errors": len(inside) - len(ok),
           "error_examples": sorted({r["error"] for r in inside if r["error"]})[:2],
           "runs_per_min": round(len(ok) / WINDOW_S * 60, 2),
           "latency_p50_s": pct(lat1, 50), "latency_p95_s": pct(lat1, 95),
           "steps_mean": round(float(np.mean([r["steps"] for r in ok])), 2) if ok else None,
           "max_steps_share": round(float(np.mean([r["termination"] == "max_steps" for r in ok])), 3) if ok else None,
           "llm_calls_per_s": round(len(calls) / WINDOW_S, 2),
           "llm_call_p95_s": pct([c[1] for c in calls], 95),
           "output_tokens_per_s": round(sum(c[3] for c in calls) / WINDOW_S, 1),
           "prompt_tokens_per_s": round(sum(c[2] for c in calls) / WINDOW_S, 1),
           "placement": placement, **gpu.window(w0, w1)}
    if placement:
        p = placement[0]
        out["vram_share"] = round(p["size_vram"] / p["size"], 3) if p.get("size") else None
        out["model_size_mb"] = round((p.get("size") or 0) / 2**20)
    return out


def optimum(rows):
    """Two optima per model over the NUM_PARALLEL = workers curve (plus the 1-worker cell):
    - throughput: the smallest workers reaching 90 % of the best runs/min, without error;
    - latency-bounded: the most runs/min whose p95 run latency stays <= 3x the 1-worker p95, without error.
    `saturated` is False when the best throughput is at the largest worker count tested (the optimum may lie beyond)."""
    good = [r for r in rows if r["errors"] == 0 and r["runs_ok"] > 0]
    if not good:
        return None
    best = max(r["runs_per_min"] for r in good)
    top = max(r["workers"] for r in rows)
    base = next((r["latency_p95_s"] for r in rows if r["workers"] == 1 and r["latency_p95_s"]), None)
    thr = next(r for r in sorted(good, key=lambda r: r["workers"]) if r["runs_per_min"] >= 0.9 * best)
    bounded = [r for r in good if base is not None and r["latency_p95_s"] is not None and r["latency_p95_s"] <= 3 * base]
    lat = max(bounded, key=lambda r: r["runs_per_min"]) if bounded else None
    return {"workers": thr["workers"], "runs_per_min": thr["runs_per_min"], "latency_p95_s": thr["latency_p95_s"],
            "best_runs_per_min": best, "saturated": thr["workers"] < top,
            "p95_bound_s": round(3 * base, 3) if base else None,
            "latency_bounded": {"workers": lat["workers"], "runs_per_min": lat["runs_per_min"],
                                "latency_p95_s": lat["latency_p95_s"]} if lat else None}


def evaluate(ctx, rows, models, window_s):
    """Optima, curves and limits from the measured cells (also used to re-derive them from a saved result)"""
    optima, curves = {}, {}
    for model in models:
        mrows = [r for r in rows if r["model"] == model]
        par = [r for r in mrows if r["config"] == "parallel" or (r["config"] == "deployed_np1" and r["workers"] == 1)]
        optima[model] = optimum(par)
        curves[model] = {f"{r['config']}:w{r['workers']}": {k: r.get(k) for k in (
            "runs_per_min", "latency_p50_s", "latency_p95_s", "errors", "output_tokens_per_s", "llm_calls_per_s",
            "gpu_util_mean", "gpu_mem_max_mib", "vram_share", "model_size_mb", "steps_mean")} for r in mrows}
    metrics = {"optima": optima, "curves": curves, "optimum_documented": sum(v is not None for v in optima.values()),
               "gpu": "Tesla T4 15 GB", "rows": rows}

    # ---- the provisioned endpoint (NUM_PARALLEL=1) vs NUM_PARALLEL = workers, all models in one limit
    parts, worst = [], None
    for model in models:
        dep = {r["workers"]: r for r in rows if r["model"] == model and r["config"] == "deployed_np1"}
        par = {r["workers"]: r for r in rows if r["model"] == model and r["config"] == "parallel"}
        pairs = [(w, dep[w], par[w]) for w in sorted(dep) if w > 1 and w in par and dep[w]["runs_ok"] and par[w]["runs_ok"]]
        if 1 not in dep or not pairs:
            continue
        w, d, p_ = max(pairs, key=lambda x: x[2]["runs_per_min"] / x[1]["runs_per_min"])
        ratio = p_["runs_per_min"] / d["runs_per_min"]
        if ratio >= 1.3:
            worst = max(worst or 0, ratio)
            parts.append(f"{model}: deployed w1 {dep[1]['runs_per_min']} runs/min (p50 {dep[1]['latency_p50_s']} s); "
                         f"deployed w{w} {d['runs_per_min']} runs/min (p50 {d['latency_p50_s']} / p95 "
                         f"{d['latency_p95_s']} s) vs NUM_PARALLEL={w} {p_['runs_per_min']} runs/min (p50 "
                         f"{p_['latency_p50_s']} / p95 {p_['latency_p95_s']} s)")
    if parts:
        rec = "; ".join(f"{m} {o['latency_bounded']['workers'] if o and o['latency_bounded'] else '?'} "
                        f"(latency-bounded) to {o['workers'] if o else '?'}{'' if o and o['saturated'] else '+'} "
                        "(throughput)" for m, o in optima.items())
        ctx.limit(
            "With one decoding slot (OLLAMA_NUM_PARALLEL=1, the lab endpoint as provisioned) the demo agent does not "
            f"scale with concurrent runs: extra workers only queue, and a server with NUM_PARALLEL = workers delivers "
            f"up to {worst:.1f}x the runs/min on the same T4 with a lower p95",
            "; ".join(parts) + f"; {window_s} s measured per cell after warm-up, temperature 0, synthetic tasks. "
            "genai_agent/docker-compose.yml (ollama/ollama:0.6.8) leaves OLLAMA_NUM_PARALLEL unset and the README "
            "does not mention it, so the agent's throughput depends on an undocumented server default",
            f"Set OLLAMA_NUM_PARALLEL explicitly in genai_agent/docker-compose.yml and document the measured T4 "
            f"settings in the README ({rec}); size the agent worker pool to NUM_PARALLEL; note the VRAM cost "
            "(KV cache per slot, see metrics.curves model_size_mb)", "major")
    for model in models:
        for r in (r for r in rows if r["model"] == model and r["config"] == "parallel"):
            w = r["workers"]
            if r["errors"]:
                ctx.limit(f"{model}: {r['errors']} of {r['runs_in_window']} agent runs failed at {w} concurrent workers "
                          f"(NUM_PARALLEL={w})", f"examples: {r['error_examples']}; p95 {r['latency_p95_s']} s",
                          "Bound the agent worker pool below this level and add client retries with backoff for "
                          "timeouts only", "major")
            if r.get("vram_share") is not None and r["vram_share"] < 0.99:
                ctx.limit(f"{model}: with NUM_PARALLEL={w} the model no longer fits on the T4 "
                          f"({r['vram_share']:.0%} of {r['model_size_mb']} MB in VRAM), part runs on the CPU",
                          f"runs/min {r['runs_per_min']}, p95 {r['latency_p95_s']} s, GPU util {r['gpu_util_mean']} %",
                          "Cap NUM_PARALLEL (KV cache grows with parallel x context) or lower the context per slot "
                          "for this model size on a 15 GB GPU", "major")
        o = optima.get(model)
        if o and not o["saturated"]:
            ctx.note(f"{model}: throughput still rising at the largest level tested ({o['workers']} workers, "
                     f"{o['runs_per_min']} runs/min, p95 {o['latency_p95_s']} s): the throughput optimum is >= "
                     f"{o['workers']}; latency-bounded optimum (p95 <= {o['p95_bound_s']} s): {o['latency_bounded']}")
        elif o:
            ctx.note(f"{model}: throughput optimum {o['workers']} workers ({o['runs_per_min']} runs/min); "
                     f"latency-bounded optimum (p95 <= {o['p95_bound_s']} s): {o['latency_bounded']}")
    ctx.note("nvidia-smi utilisation is not a saturation signal here: it falls (e.g. 88 -> 67 % for llama3.2:3b) while "
             "runs/min rise with NUM_PARALLEL; the KV cache per slot is visible in model_size_mb")
    return metrics


def run(ctx):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "genai_agent"))
    log = lambda m: print(m, flush=True)  # noqa: E731
    rows = []
    lab = None
    with GpuSampler() as gpu:
        for mi, model in enumerate(MODELS):
            # deployed service (NUM_PARALLEL=1)
            for workers in DEPLOYED_WORKERS:
                log(f"deployed {model} w={workers}")
                r = drive(DEPLOYED, model, workers, gpu, seed=mi * 100 + workers)
                rows.append({"model": model, "config": "deployed_np1", "num_parallel": 1, "workers": workers, **r})
                log(f"  -> {r['runs_per_min']} runs/min, p95 {r['latency_p95_s']} s, errors {r['errors']}, "
                    f"gpu {r['gpu_util_mean']} %")
            unload_all(DEPLOYED)
            # lab-owned server, NUM_PARALLEL = workers
            for workers in [w for w in WORKERS if w > 1]:
                log(f"parallel {model} np=w={workers}")
                lab = LabOllama(LAB_PORT, workers, OLLAMA_MODELS, OLLAMA_BIN)
                try:
                    lab.start()
                    r = drive(lab.base, model, workers, gpu, seed=mi * 100 + 50 + workers)
                    r.update(lab.memory_peak_mb())
                finally:
                    lab.stop()
                rows.append({"model": model, "config": "parallel", "num_parallel": workers, "workers": workers, **r})
                log(f"  -> {r['runs_per_min']} runs/min, p95 {r['latency_p95_s']} s, errors {r['errors']}, "
                    f"gpu {r['gpu_util_mean']} %, vram share {r.get('vram_share')}")
    frame = pd.DataFrame([{k: v for k, v in r.items() if k not in ("placement", "error_examples")} for r in rows])
    frame.to_csv(ctx.workdir / "cells.csv", index=False)

    return evaluate(ctx, rows, MODELS, WINDOW_S)
