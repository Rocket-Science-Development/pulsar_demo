"""Part 2: the demo agent answers questions on the Government of Canada AI Register, with provoked changes.

The agent of `genai_agent/` (same loop, OpenTelemetry GenAI spans, PulsarSpanExporter -> Pulse -> InfluxDB) gets one
tool, `search_register` (BM25 over the register). Phases, in order:

| Phase | Questions | Corpus | Model |
|---|---|---|---|
| reference | French | whole register | llama3.2:3b |
| holdout | French | whole register | llama3.2:3b (no change: false alarms) |
| corpus_loss | French | half of the question systems removed | llama3.2:3b |
| language_en | English | whole register | llama3.2:3b |
| model_granite | French | whole register | granite3.3:2b |

Every run is scored in-process and the scores go to the agent span (`det_*` fields of `agent_run`): best retrieval
score, empty retrieval, faithfulness of the answer to the retrieved records (NLI), and agreement with the register
(`eval_correct`, a known answer: an evaluation, not something a production system would see). A local log keeps the
questions and the answers (public register content, no personal data), and the request embeddings (nomic-embed-text)
are saved for the MMD test. A canary set (CanaryProbe) is asked to llama3.2:3b twice and to granite3.3:2b once.

Run it on an otherwise idle machine: on a shared CPU, completions slowed down more than tenfold and timed out (300 s).

    python rag_campaign.py --data data --env-file ../../validation_e2e/.env --campaign registre-gc-1 --workers 2
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "genai_agent"))

from openai import OpenAI  # noqa: E402
from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: E402
from pulsar_data_collection.genai import CanaryProbe, FaithfulnessScorer, PulsarSpanExporter, PulseSink, record_score  # noqa: E402
from pulsar_data_collection.models import CaptureOptions, PulseParameters  # noqa: E402

from influx_env import ensure_bucket, load_env  # noqa: E402
from permit_agent import register  # noqa: E402
from permit_agent.agent import PermitAgent  # noqa: E402
from register_questions import make_question, pool, score  # noqa: E402

MODEL_ID, MODEL_VERSION = "registre_ia", "1.0"
PHASES = [  # name, runs, language, corpus loss, model
    ("reference", 80, "fr", False, "llama3.2:3b"),
    ("holdout", 40, "fr", False, "llama3.2:3b"),
    ("corpus_loss", 40, "fr", True, "llama3.2:3b"),
    ("language_en", 40, "en", False, "llama3.2:3b"),
    ("model_granite", 40, "fr", False, "granite3.3:2b"),
]


def keep_awake() -> None:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def embed(base: str, texts: list) -> list:
    response = requests.post(f"{base}/api/embed", json={"model": "nomic-embed-text", "input": texts}, timeout=600)
    response.raise_for_status()
    return response.json()["embeddings"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(HERE / "data"))
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", default=None, help="results folder (default results/<campaign>)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pool", type=int, default=150, help="systems the questions are drawn from")
    parser.add_argument("--scale", type=float, default=1.0, help="multiply the number of runs of every phase")
    parser.add_argument("--phases", nargs="*", default=None)
    parser.add_argument("--canary-samples", type=int, default=3)
    parser.add_argument("--skip-canary", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    keep_awake()
    out = Path(args.out or HERE / "results" / args.campaign)
    out.mkdir(parents=True, exist_ok=True)
    login = load_env(args.env_file)
    ensure_bucket(login)
    llm = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    base = llm.removesuffix("/v1")

    records = register.load_register(Path(args.data) / "registre_ia.csv")
    by_id = {r["id"]: r for r in records}
    systems = pool(records, args.pool, seed=args.seed)
    rng = random.Random(args.seed)
    lost = set(rng.sample([s["id"] for s in systems], len(systems) // 2))
    (out / "setup.json").write_text(json.dumps({
        "register_systems": len(records), "pool": [s["id"] for s in systems], "corpus_loss_removed": sorted(lost),
        "phases": PHASES, "workers": args.workers, "scale": args.scale, "seed": args.seed,
    }, indent=1), encoding="utf-8")
    faithfulness = FaithfulnessScorer()
    params = PulseParameters(
        model_id=MODEL_ID, model_version=MODEL_VERSION, data_id="trial", reference_data_storage="open.canada.ca AI register",
        target_name="task_success", storage_engine="influxdb", timestamp_column_name="_time", login=login,
        additional_tags={"campaign": args.campaign},
    )
    embeddings_path = out / "embeddings.npy"
    embeddings = list(np.load(embeddings_path)) if args.phases and embeddings_path.exists() else []
    log_lock = threading.Lock()
    client = OpenAI(base_url=llm, api_key="not-needed-for-local-models", timeout=300, max_retries=0)
    timings = {}
    with open(out / "runs.jsonl", "a", encoding="utf-8") as log:
        for phase, runs, language, corpus_loss, model in PHASES:
            if args.phases and phase not in args.phases:
                continue
            runs = max(1, round(runs * args.scale))
            sink = PulseSink(params.model_copy(update={"additional_tags": {"campaign": args.campaign, "scenario": phase}}),
                             CaptureOptions(spool_dir=out / ".spool"))
            provider = TracerProvider(resource=Resource.create({"service.name": "register-assistant"}))
            provider.add_span_processor(BatchSpanProcessor(PulsarSpanExporter(sink, allowed_tools={"search_register"})))
            tracer = provider.get_tracer("register-agent")
            register.REMOVED.clear()
            if corpus_loss:
                register.REMOVED.update(lost)
            questions = [make_question(rng, systems, language) for _ in range(runs)]
            first = len(embeddings)
            started, done = time.perf_counter(), [0]
            print(f"=== {phase}: {runs} runs, {language}, {model}, corpus loss {corpus_loss}", flush=True)

            def one(i, question, model=model, phase=phase):
                t = time.perf_counter()
                found = []

                def search(query: str) -> list:
                    result = register.search_register(query)
                    found.append(result)
                    return result

                agent = PermitAgent(base_url=llm, model=model, max_steps=4, system_prompt=register.SYSTEM_PROMPT,
                                    tool_schemas=[register.TOOL_SCHEMA], functions={"search_register": search},
                                    context_tools=("search_register",), max_tokens=200)
                agent.tracer = tracer
                agent.client = client  # one client, bounded wait: a stuck completion fails the run, not the campaign
                record = by_id[question["system"]]
                scores = {}

                def finish(span, request, result):
                    scores["retrieval_top"] = max((d["score"] for f in found for d in f), default=0.0)
                    scores["retrieval_empty"] = float(any(not f for f in found))
                    scores["target_retrieved"] = float(any(d["id"] == record["id"] for f in found for d in f))
                    faithful = faithfulness.score(result.answer, result.context) if result.context and result.answer else None
                    if faithful is not None:
                        scores["faithfulness"] = float(faithful)
                    scores["eval_correct"] = float(score(result.answer, question, record))
                    for name, value in scores.items():
                        record_score(span, name, value)

                try:
                    result = agent.run(question["request"], expected_tool="search_register", on_finish=finish)
                    outcome = {"answer": result.answer, "tools": result.tools, "steps": result.steps,
                               "termination": result.termination}
                except Exception as error:  # noqa: BLE001 - one failed run must not stop the campaign
                    outcome = {"answer": None, "tools": [], "steps": 0, "termination": "error",
                               "error": f"{type(error).__name__}: {error}"}
                with log_lock:
                    log.write(json.dumps({
                        "phase": phase, "index": first + i, **question, "model": model, **outcome,
                        "seconds": round(time.perf_counter() - t, 2), **scores,
                    }, ensure_ascii=False) + "\n")
                    log.flush()
                    done[0] += 1
                    if done[0] % 10 == 0:
                        print(f"  {phase}: {done[0]}/{runs} ({time.perf_counter() - started:.0f} s)", flush=True)

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                list(executor.map(lambda a: one(*a), enumerate(questions)))
            provider.shutdown()
            embeddings.extend(embed(base, [q["request"] for q in questions]))
            np.save(embeddings_path, np.array(embeddings, dtype=np.float32))
            timings[phase] = round(time.perf_counter() - started, 1)
            stats = sink.stats()
            print(f"=== {phase} done in {timings[phase]} s; written "
                  + ", ".join(f"{k}={v['written_rows']}/{v['captured_rows']}" for k, v in stats.items()), flush=True)
    register.REMOVED.clear()

    if not args.skip_canary:
        client = OpenAI(base_url=llm, api_key="not-needed-for-local-models")
        probe = CanaryProbe(samples=args.canary_samples)

        def asker(model):
            def ask(prompt):
                response = client.chat.completions.create(model=model, temperature=0.7, max_tokens=60,
                                                          messages=[{"role": "user", "content": prompt}])
                return response.choices[0].message.content or ""
            return ask

        for name, model in [("reference", "llama3.2:3b"), ("same", "llama3.2:3b"), ("granite", "granite3.3:2b")]:
            t = time.perf_counter()
            answers = probe.run(asker(model))
            CanaryProbe.save(answers, out / "canary" / f"{name}.json", model=model, temperature=0.7, failures=probe.failures)
            timings[f"canary_{name}"] = round(time.perf_counter() - t, 1)
            print(f"canary {name} ({model}): {sum(len(v) for v in answers.values())} answers", flush=True)
    timing_path = out / "timings.json"
    previous = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {}
    timing_path.write_text(json.dumps({**previous, **timings}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
