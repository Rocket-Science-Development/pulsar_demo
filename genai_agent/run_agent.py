"""Run the demo agent on synthetic requests and collect its traces with pulsar-data-collection.

Examples:
    python run_agent.py --runs 20
    python run_agent.py --runs 50 --model granite3.3:2b --label model_change
    python run_agent.py --runs 50 --timeout-rate 0.3 --label tool_fault

Environment: LLM_BASE_URL (default http://localhost:11434/v1), INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pulsar_data_collection.genai import PulsarSpanExporter, PulseSink
from pulsar_data_collection.models import CaptureOptions, PulseParameters

from permit_agent import tools
from permit_agent.agent import PermitAgent
from permit_agent.tasks import DEFAULT_MIX, make_task


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "llama3.2:3b"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-version", default="v1", choices=["v1", "v2"])
    parser.add_argument("--timeout-rate", type=float, default=0.0, help="share of check_eligibility calls that time out")
    parser.add_argument("--file-share", type=float, default=None, help="share of file-opening requests (question drift)")
    parser.add_argument("--french-share", type=float, default=0.5)
    parser.add_argument("--label", default="baseline", help="scenario label, written as the `scenario` tag")
    parser.add_argument("--campaign", default="adhoc", help="campaign identifier, written as the `campaign` tag")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spool-dir", default=str(Path(__file__).parent / ".spool"))
    args = parser.parse_args(argv)

    params = PulseParameters(
        model_id="permit_assistant",
        model_version="1.0",
        data_id="demo",
        reference_data_storage="synthetic://permit_assistant",
        target_name="task_success",
        storage_engine="influxdb",
        timestamp_column_name="_time",
        login={
            "url": os.environ.get("INFLUX_URL", "http://localhost:8086"),
            "token": os.environ["INFLUX_TOKEN"],
            "org": os.environ.get("INFLUX_ORG", "pulsarml"),
            "bucket_name": os.environ.get("INFLUX_BUCKET", "monitoring"),
        },
        additional_tags={"scenario": args.label, "campaign": args.campaign},
    )
    sink = PulseSink(params, CaptureOptions(spool_dir=Path(args.spool_dir)))
    provider = TracerProvider(resource=Resource.create({"service.name": "permit-assistant"}))
    provider.add_span_processor(BatchSpanProcessor(PulsarSpanExporter(sink, allowed_tools=tools.ALLOWED_TOOLS)))
    trace.set_tracer_provider(provider)

    tools.FAULTS.clear()
    if args.timeout_rate:
        tools.FAULTS["check_eligibility"] = {"timeout_rate": args.timeout_rate}
    mix = dict(DEFAULT_MIX)
    if args.file_share is not None:
        rest = (1 - args.file_share) / (sum(mix.values()) - mix["file"])
        mix = {k: (args.file_share if k == "file" else v * rest) for k, v in mix.items()}

    random.seed(args.seed)
    rng = random.Random(args.seed)
    agent = PermitAgent(
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
        model=args.model,
        temperature=args.temperature,
        prompt_version=args.prompt_version,
    )
    start = time.perf_counter()
    for i in range(args.runs):
        task = make_task(rng, mix, args.french_share)
        t = time.perf_counter()
        result = agent.run(task["request"], expected_tool=task["expected_tool"])
        print(
            f"[{i + 1}/{args.runs}] {task['kind']:<11} {task['language']} {time.perf_counter() - t:5.1f}s "
            f"steps={result.steps} tools={','.join(result.tools) or '-'} {result.termination}",
            flush=True,
        )
    provider.shutdown()
    stats = sink.stats()
    print(
        f"\n{args.runs} runs in {time.perf_counter() - start:.0f}s; written: "
        + ", ".join(f"{k}={v['written_rows']}/{v['captured_rows']}" for k, v in stats.items())
    )
    return 0 if all(v["dropped_rows"] == 0 for v in stats.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
