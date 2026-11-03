"""Shared helpers of the agents-bias family, part 2 (LA-01, LA-02, LB-02): an in-memory Pulsar collector per test,
dataset locations on the GPU host, Wilson intervals, and the LTLf witness of a violated rule.

Nothing here writes a prompt or an answer to disk: tests keep features, scores, labels and tool names only.
"""

import os
import threading
from pathlib import Path

DATA = Path(os.environ.get("LAB_DATA", "/data/lab")) / "datasets"
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
OLLAMA = BASE_URL.rsplit("/v1", 1)[0]
DRY = os.environ.get("LAB_DRY", "") not in ("", "0")


class Collector:
    """Own tracer provider (not the global one): BatchSpanProcessor -> PulsarSpanExporter -> ListSink"""

    def __init__(self, service: str, allowed_tools=None, **exporter_options):
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from pulsar_data_collection.genai import ListSink, PulsarSpanExporter

        self.sink = ListSink()
        self.exporter = PulsarSpanExporter(self.sink, allowed_tools=allowed_tools, **exporter_options)
        self.provider = TracerProvider(resource=Resource.create({"service.name": service}))
        self.provider.add_span_processor(BatchSpanProcessor(self.exporter, max_queue_size=65536))
        self.tracer = self.provider.get_tracer(service)
        self.lock = threading.Lock()

    def records(self, kind: str = "agent_run") -> list:
        self.provider.force_flush(120_000)
        return self.sink.of(kind)


def wilson(k: int, n: int, z: float = 1.96):
    import math

    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def first_violation(rule, trace: list):
    """Smallest prefix length whose trace already violates a safety-style rule (None when only the whole trace does,
    or when the rule holds). Pulsar's ComplianceMonitor returns rule ids only; this is the test's own witness."""
    if rule.check(trace):
        return None
    for k in range(1, len(trace) + 1):
        prefix = trace[:k]
        # a prefix that violates and whose every extension seen in the run also violates: report the first such k
        if not rule.check(prefix) and all(not rule.check(trace[:j]) for j in range(k, len(trace) + 1)):
            return k
    return len(trace)
