"""Injection campaign S0 to S4 on the demo agent: known changes injected on purpose, detectors that must see them.

| Scenario | Injected | Detector |
|---|---|---|
| S0 A/A | nothing (5 repetitions) | false alarms of every detector |
| S1 inputs | 5 to 50 % of requests outside the domain of the assistant | MMD on the request embeddings |
| S2 model | the endpoint serves another model: quantized llama3.2 (q8), granite3.3 | model equality on the canary set |
| S3 RAG | the rules of 1 to 4 of the 4 programs removed from the corpus (both languages) | faithfulness of the answers (NLI) |
| S4 security | 5 to 50 % of prompt injections, some with synthetic personal information | injection detector, personal information, LTLf compliance |

S1, S3 and S4 run at five intensities (60 runs each) after a reference of 150 runs; S0 repeats the reference conditions
five times. Every run is scored in-process (injection, faithfulness, personal information by role); only the scores
reach InfluxDB. The request embeddings (nomic-embed-text) and a per-run log are kept locally: the requests are
synthetic. The canary answers are saved as JSON (they are not user data).

    python campaign_s3s4.py --campaign s3s4-20261102 --out results/s3s4

Environment: LLM_BASE_URL, INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET.
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pulsar_data_collection.genai import (
    CanaryProbe,
    FaithfulnessScorer,
    InjectionDetector,
    ModelPiiDetector,
    PulsarSpanExporter,
    PulseSink,
    record_score,
)
from pulsar_data_collection.models import CaptureOptions, PulseParameters
from pulsar_data_collection.privacy import PiiSession

from permit_agent import tools
from permit_agent.agent import PermitAgent
from permit_agent.tasks import make_task

MIX = {"eligibility": 0.2, "amount": 0.15, "file": 0.1, "general": 0.15, "rules": 0.4}
LEVELS = [0.05, 0.1, 0.2, 0.35, 0.5]
# S3 removes whole programs (both language versions of their rules): removing single documents leaves the other
# language version, which still holds the same figures (first run of the campaign: no information actually removed)
PROGRAMS = ["LOGE", "RENO", "GARDE", "EMPLOI"]
REMOVALS = [0.25, 0.5, 0.75, 1.0]  # share of the four programs whose rules are removed


def embed(base: str, texts: list) -> list:
    """Embeddings of a whole phase in one call (one model swap on the GPU instead of one per request)"""
    response = requests.post(f"{base}/api/embed", json={"model": "nomic-embed-text", "input": texts}, timeout=600)
    response.raise_for_status()
    return response.json()["embeddings"]


class Scorer:
    """Detectors run inside the agent span (on_finish): scores and personal-information counts only"""

    def __init__(self):
        self.injection = InjectionDetector()
        self.faithfulness = FaithfulnessScorer()
        self.names = ModelPiiDetector()
        self.last = {}

    def __call__(self, span, request, result):
        scores = {"injection": self.injection.score(request)}
        faithful = self.faithfulness.score(result.answer, result.context) if result.context and result.answer else None
        if faithful is not None:
            scores["faithfulness"] = faithful
        scores["names_input"] = float(sum(f.type == "person_name" for f in self.names.scan(request)))
        for name, value in scores.items():
            record_score(span, name, value)
        with PiiSession(self.names.rules) as session:
            session.record(span, "input", request)
            if result.context:
                session.record(span, "context", " ".join(result.context))
            leaks = session.record(span, "output", result.answer or "").leaks
        scores["pii_leaks"] = len(leaks)
        self.last = scores
        return scores


def run_phase(agent, scorer, rng, base, log, embeddings, phase, level, runs, out_of_domain=0.0, injection=0.0,
              workers=1):
    """Run a phase; with workers > 1 the requests are sent concurrently (the endpoint serves them in parallel)"""
    started = time.perf_counter()
    first = len(embeddings)
    tasks = [make_task(rng, MIX, 0.5, out_of_domain=out_of_domain, injection=injection) for _ in range(runs)]
    lock, done = threading.Lock(), [0]

    def one(i, task):
        t = time.perf_counter()
        scores = {}

        def finish(span, request, result):
            scores.update(scorer(span, request, result))

        result = agent.run(task["request"], expected_tool=task["expected_tool"], on_finish=finish)
        with lock:
            log.write(json.dumps({
                "phase": phase, "level": level, "index": first + i, "kind": task["kind"],
                "language": task["language"], "tools": result.tools, "termination": result.termination,
                "seconds": round(time.perf_counter() - t, 2), **scores,
            }) + "\n")
            log.flush()
            done[0] += 1
            if done[0] % 20 == 0:
                print(f"  {phase} {level}: {done[0]}/{runs} ({time.perf_counter() - started:.0f} s)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda args: one(*args), enumerate(tasks)))
    embeddings.extend(embed(base, [t["request"] for t in tasks]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", default="s3s4")
    parser.add_argument("--out", default="results/s3s4")
    parser.add_argument("--reference-runs", type=int, default=150)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--canary-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-canary", action="store_true")
    parser.add_argument("--phases", nargs="*", default=None, help="run only these phases (e.g. S3), appending to --out")
    parser.add_argument("--workers", type=int, default=1, help="concurrent requests (the endpoint serves them in parallel)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    base = llm.removesuffix("/v1")

    params = PulseParameters(
        model_id="permit_assistant", model_version="1.0", data_id="demo", reference_data_storage="synthetic://permit_assistant",
        target_name="task_success", storage_engine="influxdb", timestamp_column_name="_time",
        login={"url": os.environ.get("INFLUX_URL", "http://localhost:8086"), "token": os.environ["INFLUX_TOKEN"],
               "org": os.environ.get("INFLUX_ORG", "pulsarml"), "bucket_name": os.environ.get("INFLUX_BUCKET", "monitoring")},
        additional_tags={"campaign": args.campaign},
    )
    scorer = Scorer()
    rng = random.Random(args.seed)
    embeddings = list(np.load(out / "embeddings.npy")) if args.phases and (out / "embeddings.npy").exists() else []
    agent = PermitAgent(base_url=llm, model="llama3.2:3b")
    with open(out / "runs.jsonl", "a", encoding="utf-8") as log:
        phases = [("reference", 0.0, args.reference_runs, {})]
        phases += [("S0", k, args.window, {}) for k in range(1, 6)]
        phases += [("S1", level, args.window, {"out_of_domain": level}) for level in LEVELS]
        phases += [("S3", share, args.window, {"removed": share}) for share in REMOVALS]
        phases += [("S4", level, args.window, {"injection": level}) for level in LEVELS]
        if args.phases:
            phases = [p for p in phases if p[0] in args.phases]
        for phase, level, runs, options in phases:
            # one Pulse per phase, so that the records carry the phase as their scenario tag
            sink = PulseSink(params.model_copy(update={"additional_tags": {"campaign": args.campaign,
                                                                           "scenario": f"{phase}-{level}"}}),
                             CaptureOptions(spool_dir=out / ".spool"))
            provider = TracerProvider(resource=Resource.create({"service.name": "permit-assistant"}))
            provider.add_span_processor(BatchSpanProcessor(PulsarSpanExporter(sink, allowed_tools=tools.ALLOWED_TOOLS)))
            agent.tracer = provider.get_tracer("permit-agent")
            tools.FAULTS.clear()
            if "removed" in options:
                programs = rng.sample(PROGRAMS, round(options["removed"] * len(PROGRAMS)))
                removed = {d["id"] for d in tools.DOCUMENTS if d["id"].split("-")[1] in programs}
                tools.FAULTS["search_documents"] = {"removed": removed}
                print(f"  removed programs: {sorted(programs)}", flush=True)
            print(f"=== {phase} {level} ({runs} runs)", flush=True)
            run_phase(agent, scorer, rng, base, log, embeddings, phase, level, runs,
                      out_of_domain=options.get("out_of_domain", 0.0), injection=options.get("injection", 0.0),
                      workers=args.workers)
            provider.shutdown()
            np.save(out / "embeddings.npy", np.array(embeddings, dtype=np.float32))
    tools.FAULTS.clear()

    if not args.skip_canary:
        client = OpenAI(base_url=llm, api_key="not-needed-for-local-models")
        probe = CanaryProbe(samples=args.canary_samples)

        def asker(model):
            def ask(prompt):
                response = client.chat.completions.create(model=model, temperature=0.7, max_tokens=60,
                                                          messages=[{"role": "user", "content": prompt}])
                return response.choices[0].message.content or ""
            return ask

        sets = [("reference", "llama3.2:3b")]
        for k in range(1, 6):
            sets += [(f"same-{k}", "llama3.2:3b"), (f"q8-{k}", "llama3.2:3b-instruct-q8_0"), (f"granite-{k}", "granite3.3:2b")]
        for name, model in sets:
            answers = probe.run(asker(model))
            CanaryProbe.save(answers, out / "canary" / f"{name}.json", model=model, temperature=0.7, failures=probe.failures)
            print(f"canary {name} ({model}): {sum(len(v) for v in answers.values())} answers", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
