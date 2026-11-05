"""Consistency campaign: the same 30 questions asked several times to local models, under several conditions.

| Condition | Model | Temperature | Answers per question |
|---|---|---|---|
| `llama_t07` | llama3.2:3b | 0.7 | 6 (repeats 0-2: reference, 3-5: A/A holdout) |
| `granite_t07` | granite3.3:2b | 0.7 | 3 |
| `llama_t15` | llama3.2:3b | 1.5 | 3 |
| `granite_t15` | granite3.3:2b | 1.5 | 3 |

`ConsistencyProbe` (pulsar-data-collection) asks every question round after round through the OpenAI-compatible API of
Ollama, at most 2 calls at a time, with a cap on the answer length and a client timeout. The questions are synthetic
and public (`questions.py`): the answers are kept. A condition already saved is skipped (rerun with `--force`).

    python campaign.py --campaign coherence-1 [--base-url http://127.0.0.1:11434/v1] [--workers 2]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from pulsar_data_collection.genai import ConsistencyProbe, openai_chat_target

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from questions import questions  # noqa: E402

CONDITIONS = [  # name, model, temperature, repeats
    ("llama_t07", "llama3.2:3b", 0.7, 6),
    ("granite_t07", "granite3.3:2b", 0.7, 3),
    ("llama_t15", "llama3.2:3b", 1.5, 3),
    ("granite_t15", "granite3.3:2b", 1.5, 3),
]
SYSTEM = "Answer the question as briefly as possible, in the language of the question."


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", default="coherence-1")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--only", nargs="*", help="conditions to run (default: all)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out_dir = HERE / "results" / args.campaign
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (out_dir / "log.txt").open("a", encoding="utf-8")

    for name, model, temperature, repeats in CONDITIONS:
        path = out_dir / f"{name}.json"
        if (args.only and name not in args.only) or (path.exists() and not args.force):
            continue
        ask = openai_chat_target(
            args.base_url, model, temperature=temperature, max_tokens=args.max_tokens, timeout=args.timeout, system=SYSTEM
        )
        probe = ConsistencyProbe(questions(), repeats=repeats, seed=sum(map(ord, name)))
        started = time.time()

        def progress(i, n, row, name=name):
            if i % 10 == 0 or i == n:
                print(f"{name}: {i}/{n} ({time.time() - started:.0f} s)", flush=True)

        result = probe.run(ask, workers=args.workers, progress=progress)
        elapsed = time.time() - started
        payload = {
            "condition": name,
            "model": model,
            "temperature": temperature,
            "repeats": repeats,
            "max_tokens": args.max_tokens,
            "workers": args.workers,
            "elapsed_s": round(elapsed, 1),
            "answers": json.loads(result.answers.to_json(orient="records", force_ascii=False)),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        errors = int((result.answers["answer_status"] == "error").sum())
        median = result.answers["latency_s"].median()
        line = f"{name}: {len(result.answers)} calls, {errors} errors, {elapsed:.0f} s, median {median:.1f} s"
        print(line, flush=True)
        log.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
        log.flush()


if __name__ == "__main__":
    main()
