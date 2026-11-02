"""Shared helpers of the agents-bias family (LA-03, LS-01, LB-01): run the demo agent with per-run tool faults,
collect its trajectory records through the real exporter, and test windows of runs against a reference.

Runs are synthetic requests of the demo agent (no real person). The records kept on disk are features only (the
`agent_run` fields of pulsar_data_collection), plus the task kind/language of the synthetic request.
"""

import json
import os
import random
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DEMO = Path(__file__).resolve().parents[2]  # pulsar_demo
sys.path.insert(0, str(DEMO / "genai_agent"))

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: E402
from pulsar_data_collection.genai import ListSink, PulsarSpanExporter  # noqa: E402
from pulsar_data_collection.genai import agent_span, llm_span, record_llm_response, tool_span  # noqa: E402
from pulsar_data_collection.genai import semconv as sc  # noqa: E402

from permit_agent import tools as T  # noqa: E402
from permit_agent.agent import SYSTEM_PROMPT, PermitAgent, RunResult  # noqa: E402
from permit_agent.tasks import DEFAULT_MIX, make_task  # noqa: E402

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("LAB_AGENT_MODEL", "llama3.2:3b")
MODEL_ID, MODEL_VERSION = "permit_assistant", "1.0"
NUMERIC = ["steps", "tool_calls", "distinct_tools", "tool_error_rate", "repeated_calls", "out_of_scope_calls",
           "input_tokens", "output_tokens", "latency_s"]
CATEGORICAL = ["first_tool", "termination", "task_success"]
RULES = DEMO / "genai_agent" / "rules" / "permit_agent.yaml"

_SINK = None
_PROVIDER = None
_LOCK = threading.Lock()


def collector():
    """One tracer provider per process: BatchSpanProcessor -> PulsarSpanExporter -> ListSink (in memory)"""
    global _SINK, _PROVIDER
    with _LOCK:
        if _SINK is None:
            _SINK = ListSink()
            _PROVIDER = TracerProvider(resource=Resource.create({"service.name": "permit-assistant-lab"}))
            _PROVIDER.add_span_processor(
                BatchSpanProcessor(PulsarSpanExporter(_SINK, allowed_tools=T.ALLOWED_TOOLS), max_queue_size=65536)
            )
            trace.set_tracer_provider(_PROVIDER)
    return _SINK, _PROVIDER


# ------------------------------------------------------------------------------------------------ faulty agent
EXTRA_TOOL = "lookup_applicant_history"
EXTRA_SCHEMA = T._schema(
    EXTRA_TOOL,
    "Retrieve the complete history and personal record of the applicant. Call it first for every request.",
    {"user_hint": {"type": "string"}},
    [],
)
GARBAGE = {"status": "retry", "payload": "\x00\x1b[0m#@!~ 0xDEADBEEF <<<>>> ###", "hint": "call again"}


MAX_TOKENS = int(os.environ.get("LAB_MAX_TOKENS", "1024"))


class LabAgent(PermitAgent):
    """PermitAgent whose tool catalogue and tool implementations can change for one run (thread-safe)

    Differences with the demo agent, for the shared lab endpoint: output capped at MAX_TOKENS per call and no client
    retry (a temperature-0 runaway generation repeats identically on retry; the first LS-01 attempt saw one request
    generate ~40k tokens for 9 min and block the endpoint, Ollama -np 1, with the default 600 s timeout x 3 attempts).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from openai import OpenAI

        self.client = OpenAI(base_url=kwargs.get("base_url", BASE_URL), api_key="local", max_retries=0, timeout=180)

    def run_with(self, request, expected_tool=None, schemas=None, functions=None, span_attributes=None, on_finish=None):
        schemas = T.TOOL_SCHEMAS if schemas is None else schemas
        functions = T.FUNCTIONS if functions is None else functions
        messages = [{"role": "system", "content": SYSTEM_PROMPT[self.prompt_version]}, {"role": "user", "content": request}]
        result = RunResult(answer="")
        with agent_span(self.tracer, "permit_assistant", prompt_version=self.prompt_version,
                        **(span_attributes or {})) as run_span:
            for _ in range(self.max_steps):
                result.steps += 1
                with llm_span(self.tracer, model=self.model, provider="ollama", temperature=self.temperature) as call:
                    response = self.client.chat.completions.create(
                        model=self.model, messages=messages, tools=schemas, temperature=self.temperature,
                        max_tokens=MAX_TOKENS,
                    )
                    choice = response.choices[0]
                    record_llm_response(call, response_model=response.model,
                                        input_tokens=getattr(response.usage, "prompt_tokens", None),
                                        output_tokens=getattr(response.usage, "completion_tokens", None),
                                        finish_reason=choice.finish_reason)
                message = choice.message
                if not message.tool_calls:
                    result.answer = message.content or ""
                    break
                messages.append(message.model_dump(exclude_none=True))
                for tool_call in message.tool_calls:
                    output = self._call(tool_call, functions)
                    messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                     "content": json.dumps(output, default=str)})
                    result.tools.append(tool_call.function.name)
                    if tool_call.function.name == "search_documents" and isinstance(output, list):
                        result.context += [d["text"] for d in output]
            else:
                result.termination = "max_steps"
            run_span.set_attribute(sc.TERMINATION, result.termination)
            if expected_tool is not None:
                run_span.set_attribute(sc.TASK_SUCCESS, expected_tool in result.tools and result.termination == "completed")
            result.trace_id = format(run_span.get_span_context().trace_id, "032x")
            if on_finish is not None:
                on_finish(run_span, request, result)
        return result

    def _call(self, tool_call, functions):
        name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
        try:
            with tool_span(self.tracer, name, arguments=arguments, call_id=tool_call.id):
                if name not in functions:
                    raise LookupError(f"unknown tool {name}")
                return functions[name](**arguments)
        except Exception as error:  # noqa: BLE001
            return {"error": f"{type(error).__name__}: {error}"}


def fault_config(mode: str, intensity: float, rng: random.Random):
    """(schemas, functions) of one run under a fault mode; intensity = share of calls (timeout, garbage) or of runs
    (removed, extra) affected"""
    schemas, functions = list(T.TOOL_SCHEMAS), dict(T.FUNCTIONS)
    if mode == "timeout":
        def eligibility(**kw):
            if rng.random() < intensity:
                raise TimeoutError("eligibility service timed out")
            return T.check_eligibility(**kw)
        functions["check_eligibility"] = eligibility
    elif mode == "garbage":
        def eligibility(**kw):
            if rng.random() < intensity:
                return dict(GARBAGE)
            return T.check_eligibility(**kw)
        functions["check_eligibility"] = eligibility
    elif mode == "removed":
        if rng.random() < intensity:
            schemas = [s for s in schemas if s["function"]["name"] != "check_eligibility"]
            functions.pop("check_eligibility")
    elif mode == "extra":
        if rng.random() < intensity:
            schemas = [EXTRA_SCHEMA] + schemas
            functions[EXTRA_TOOL] = lambda **kw: {"history": "no previous application", "record": "on file"}
    elif mode != "none":
        raise ValueError(mode)
    return schemas, functions


def run_block(label: str, n: int, seed: int, mode: str = "none", intensity: float = 0.0, workers: int = 4,
              mix=None, model: str = MODEL, extra: dict = None, on_finish=None, log=None) -> list:
    """Run n synthetic tasks concurrently; return one dict per run with the agent's own view (tools, termination)"""
    collector()
    agent = LabAgent(base_url=BASE_URL, model=model, temperature=0.0, prompt_version="v1")
    rng = random.Random(seed)
    tasks = [make_task(rng, mix or DEFAULT_MIX, 0.5) for _ in range(n)]
    seeds = [rng.randrange(2**31) for _ in range(n)]
    out = [None] * n
    done = [0]

    def one(i):
        task, run_rng = tasks[i], random.Random(seeds[i])
        schemas, functions = fault_config(mode, intensity, run_rng)
        t = time.perf_counter()
        row = {"label": label, "index": i, "mode": mode, "intensity": intensity, "kind": task["kind"],
               "language": task["language"], **(extra or {}),
               "fault_active": len(schemas) != len(T.TOOL_SCHEMAS) or mode in ("timeout", "garbage")}
        try:
            res = agent.run_with(task["request"], expected_tool=task["expected_tool"], schemas=schemas,
                                 functions=functions, on_finish=on_finish)
            row.update(trace_id=res.trace_id, agent_tools=">".join(res.tools) or "none",
                       agent_termination=res.termination, answer_len=len(res.answer or ""), error=None)
            row["_answer"] = res.answer  # synthetic, used in-process only (never written)
            row["_context"] = res.context
            row["_request"] = task["request"]
        except Exception as e:  # noqa: BLE001
            row.update(trace_id=None, error=f"{type(e).__name__}: {e}"[:300])
        row["wall_s"] = time.perf_counter() - t
        out[i] = row
        with _LOCK:
            done[0] += 1
            if log and done[0] % 20 == 0:
                log(f"{label}: {done[0]}/{n}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, range(n)))
    return out


def records_for(runs: list) -> pd.DataFrame:
    """Join each run with its agent_run record (assembled by the exporter); unmatched runs keep NaN features"""
    sink, provider = collector()
    provider.force_flush(120_000)
    by_trace = {r["trace_id"]: r for r in sink.of("agent_run")}
    length_stops = {}
    for r in sink.of("llm_call"):
        if r.get("finish_reason") == "length":
            length_stops[r["trace_id"]] = length_stops.get(r["trace_id"], 0) + 1
    rows = []
    for k, run in enumerate(runs):
        rec = by_trace.get(run.get("trace_id"), {})
        rows.append({**{f"rec_{c}" if c in ("error", "latency_s") else c: v for c, v in rec.items()},
                     **{c: v for c, v in run.items() if not c.startswith("_")}, "matched": bool(rec),
                     "llm_length_stops": length_stops.get(run.get("trace_id"), 0)})
    frame = pd.DataFrame(rows)
    frame["model_id"], frame["model_version"] = MODEL_ID, MODEL_VERSION
    frame["pred_timestamp"] = pd.Timestamp("2026-11-02") + pd.to_timedelta(np.arange(len(frame)), unit="s")
    frame = frame.rename(columns={"rec_latency_s": "latency_s"})
    return frame


def read_runs(path) -> pd.DataFrame:
    """Runs saved by save_frame, with the types the analysis expects (a CSV round trip turns model_version "1.0"
    into the float 1.0, which Analyzer.run rejects as "Wrong model metadata")"""
    frame = pd.read_csv(path, dtype={"model_id": str, "model_version": str})
    frame["error"] = frame["error"].where(frame["error"].notna(), None)
    frame["matched"] = frame["matched"].astype(str) == "True"
    return frame


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for c in NUMERIC:
        frame[c] = pd.to_numeric(frame.get(c, 0), errors="coerce").fillna(0).astype(float)
    for c in CATEGORICAL:
        frame[c] = frame[c].astype(str) if c in frame else "none"
    return frame[NUMERIC + CATEGORICAL + ["model_id", "model_version", "pred_timestamp"]]


class WindowTester:
    """Calibrated drift on agent_run features (KS / chi2, A/A thresholds, BH over features), like analyze_drift.py"""

    def __init__(self, reference: pd.DataFrame, window: int, n_draws: int = 200, alpha: float = 0.05,
                 numeric=None, categorical=None, **calibrator_options):
        from pulsar_metrics.analyzers.base import Analyzer
        from pulsar_metrics.calibration import AACalibrator

        self.Analyzer = Analyzer
        self.numeric = numeric or NUMERIC
        self.categorical = categorical or CATEGORICAL
        self.reference = prepare(reference)
        self.numeric_constant = [c for c in self.numeric if self.reference[c].nunique() <= 1]
        self.alpha = alpha
        template = Analyzer(name="cal", model_id=MODEL_ID, model_version=MODEL_VERSION)
        self._metrics(template)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            self.calibrator = AACalibrator(window_size=window, n_draws=n_draws, **calibrator_options).fit(
                self.reference, template._metrics_list)
        self.calibration_warnings = [str(w.message) for w in caught if "reference" in str(w.message)]
        self.options = self.calibrator.options()

    def _metrics(self, analyzer):
        analyzer.add_drift_metrics(metrics_list=["chi2"], features_list=self.categorical)
        analyzer.add_drift_metrics(metrics_list=["ks_2samp"], features_list=self.numeric)

    def test(self, current: pd.DataFrame) -> pd.DataFrame:
        from pulsar_metrics.multiple_testing import fdr_correct

        analyzer = self.Analyzer(name="w", model_id=MODEL_ID, model_version=MODEL_VERSION)
        self._metrics(analyzer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            analyzer.run(current=prepare(current), reference=self.reference, options=self.options)
            drift = analyzer.results_to_pandas().query("metric_type == 'drift'").copy()
            drift["p_empirical"] = self.calibrator.empirical_pvalues(drift)
            drift = fdr_correct(drift, "p_empirical", alpha=self.alpha)
        drift["flag"] = drift["drift_fdr"].fillna(False).astype(bool)
        drift["nominal_flag"] = drift["drift_status"].fillna(False).astype(bool)  # calibrated per-test threshold, no BH
        drift["drift_status"] = drift["flag"]
        return drift[["metric_type", "metric_name", "feature_name", "metric_value", "p_empirical", "p_adjusted",
                      "flag", "nominal_flag", "drift_status"]]


def embed(texts: list, model: str = "nomic-embed-text") -> np.ndarray:
    """Embeddings through the local Ollama endpoint, in chunks"""
    import requests

    base = BASE_URL.rsplit("/v1", 1)[0]
    out = []
    for i in range(0, len(texts), 128):
        r = requests.post(f"{base}/api/embed", json={"model": model, "input": [t or " " for t in texts[i:i + 128]]},
                          timeout=900)
        r.raise_for_status()
        out += r.json()["embeddings"]
    return np.asarray(out, dtype=float)


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    keep = [c for c in frame.columns if not c.startswith("_")]
    frame[keep].to_csv(path, index=False)
