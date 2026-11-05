"""SC-07 probes: query variants of the two dashboard parts that scale badly, timed directly on InfluxDB.

- `quantiles`: the drift dashboard's "Hourly quantiles of ${feature}" (3 queries, t-digest) against 3 queries with
  exact quantiles and against a single query that reads the captures once (union), one user and 20 at once, 7 and 30 d;
- `quantiles-l1`: the raw panel ("current") against the pre-aggregated panel of this branch (`pulsar_feature_quantiles`,
  queries taken from `build_dashboards.drift`), same protocol;
- `modelvar <bucket>`: the model variable as provisioned before SC-07 (negation in the predicate) against four other
  ways to list the models.

    LAB_INFLUX_ENV=lab/results/SC-07/influx.env python -m lab.tests._sc07_probes quantiles lab/results/SC-07/bench_quantiles.json
    LAB_INFLUX_ENV=lab/results/SC-07/influx.env python -m lab.tests._sc07_probes quantiles-l1 lab/results/SC-07/bench_quantiles_l1.json
    LAB_INFLUX_ENV=lab/results/SC-07/influx.env python -m lab.tests._sc07_probes modelvar sc07_ladder lab/results/SC-07/bench_modelvar.json
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from lab.tests import _platform as C
from lab.tests.sc_07 import flux, flux_values, pctl

SRC = ('from(bucket: "monitoring") |> range(start: {start}, stop: {stop}) '
       '|> filter(fn: (r) => r._measurement == "{m}_1.0_input_data") |> filter(fn: (r) => r._field == "{f}") |> group()')
TAIL = ' |> group(columns: ["q"]) |> drop(fn: (column) => column == "_start" or column == "_stop")'
QUANTILES = ((0.9, "p90"), (0.5, "median"), (0.1, "p10"))


def _fn(q, method):
    if method is None and q == 0.5:
        return "median"
    meth = f', method: "{method}"' if method else ""
    return f"(column, tables=<-) => tables |> quantile(q: {q}, column: column{meth})"


def three(method=None) -> list:
    return [SRC + f' |> aggregateWindow(every: 1h, fn: {_fn(q, method)}, createEmpty: false) |> set(key: "q", value: "{n}")' + TAIL
            for q, n in QUANTILES]


def union(method=None) -> list:
    parts = [f'data |> aggregateWindow(every: 1h, fn: {_fn(q, method or "estimate_tdigest")}, createEmpty: false) '
             f'|> set(key: "q", value: "{n}")' for q, n in QUANTILES]
    return ["data = " + SRC + " union(tables: [" + ", ".join(parts) + "])" + TAIL]


def preaggregated() -> list:
    """The 3 queries of the drift panel as generated now (pre-aggregated hours, then the raw tail), as templates"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "observability"))
    from build_dashboards import drift

    panel = next(p for p in drift("monitoring").json([])["panels"] if p.get("title", "").startswith("Hourly quantiles"))
    out = []
    for t in panel["targets"]:
        q = t["query"].replace("${bucket}", "monitoring").replace("${version}", "1.0")
        q = q.replace("${model}", "{m}").replace("${feature}", "{f}")
        out.append(q.replace("v.timeRangeStart", "{start}").replace("v.timeRangeStop", "{stop}"))
    return out


def quantiles(env: dict, l1: bool = False) -> dict:
    variants = {"current (3 queries, tdigest)": three(), "3 queries, exact_mean": three("exact_mean"),
                "1 query (union), tdigest": union(), "1 query (union), exact_mean": union("exact_mean")}
    if l1:
        variants = {"current (3 queries, tdigest)": three(), "pre-aggregated (3 queries, pulsar_feature_quantiles)": preaggregated()}
    now = datetime.now(timezone.utc)

    def panel(qs, m, f, days):
        start = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        stop = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        with ThreadPoolExecutor(len(qs)) as pool:
            res = list(pool.map(lambda q: flux(q.format(start=start, stop=stop, m=m, f=f), env), qs))
        return max(r[0] for r in res), sum(r[2] for r in res), [r[3] for r in res if r[3]]

    out = {}
    for name, qs in variants.items():
        for days in (7, 30):
            single = [panel(qs, f"model{k:03d}", f"f0{k % 10}", days) for k in range(5)]
            with ThreadPoolExecutor(20) as pool:
                t0 = time.perf_counter()
                conc = list(pool.map(lambda v: panel(qs, f"model{v % 10:03d}", f"f0{v % 10}", days), range(20)))
                wall = time.perf_counter() - t0
            st, ct = sorted(s[0] for s in single), sorted(c[0] for c in conc)
            out[f"{name} @{days}d"] = {"single_p50_s": round(st[len(st) // 2], 3), "single_max_s": round(st[-1], 3),
                                       "rows": single[0][1], "errors": sum(len(s[2]) for s in single + conc),
                                       "20_concurrent_p50_s": round(ct[len(ct) // 2], 3),
                                       "20_concurrent_max_s": round(ct[-1], 3), "20_wall_s": round(wall, 2)}
            print(name, days, out[f"{name} @{days}d"], flush=True)
    return out


def modelvar(env: dict, bucket: str) -> dict:
    s = 'import "influxdata/influxdb/schema"\n'
    variants = {
        "before SC-07: tagValues, _measurement and model_id != permit_assistant":
            s + f'schema.tagValues(bucket: "{bucket}", tag: "model_id", predicate: (r) => r._measurement == "pulsar_results" and r.model_id != "permit_assistant", start: -30d)',
        "after SC-07: tagValues on _measurement, then filter on the values":
            s + f'schema.tagValues(bucket: "{bucket}", tag: "model_id", predicate: (r) => r._measurement == "pulsar_results", start: -30d) |> filter(fn: (r) => r._value != "permit_assistant")',
        "measurementTagValues":
            s + f'schema.measurementTagValues(bucket: "{bucket}", measurement: "pulsar_results", tag: "model_id")',
        "tagValues on one series per model (performance / label_coverage)":
            s + f'schema.tagValues(bucket: "{bucket}", tag: "model_id", predicate: (r) => r._measurement == "pulsar_results" and r.metric_type == "performance" and r.metric_name == "label_coverage", start: -30d)',
        "tagValues on pulsar_capture_stats":
            s + f'schema.tagValues(bucket: "{bucket}", tag: "model_id", predicate: (r) => r._measurement == "pulsar_capture_stats", start: -30d)',
    }
    out = {}
    for name, q in variants.items():
        t = [flux(q, env) for _ in range(5)]
        out[name] = {"values": len(flux_values(q, env)), "p50_s": pctl([x[0] for x in t], 50),
                     "max_s": round(max(x[0] for x in t), 3), "errors": sum(1 for x in t if x[3])}
        print(name, out[name], flush=True)
    return out


if __name__ == "__main__":
    env = C.influx_env()
    if sys.argv[1] in ("quantiles", "quantiles-l1"):
        result, path = quantiles(env, sys.argv[1] == "quantiles-l1"), sys.argv[2] if len(sys.argv) > 2 else None
    else:
        result, path = modelvar(env, sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None
    if path:
        open(path, "w", encoding="utf-8").write(json.dumps(result, indent=1))
