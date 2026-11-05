"""SC-07 - Results and dashboards at scale: 300 models, 30 days of results, every Flux query of the dashboards timed.

1) load: 30 days of analysis results for 300 models through the real write path (`ResultsSink.write_results`, one
   request per model and 6-hour window, as `analysis_job.py` does), with the shapes of the real analyses (drift on 20
   features with `p_adjusted` / `confirmed`, performance, fixed fairness on language, region and their intersection,
   anytime-valid monitor, explanations); capture counters of the 300 models every hour (`write_stats_many`, the
   single-reporter path of `StatsReporter`); captures of `CAPTURED` models through `Pulse` (150 rows/h, 20 features,
   personal information masked), so that the capture panels have their real volume;
2) cardinality of the bucket (`influxdb.cardinality`, distinct tag values) and its size on disk;
3) every templating variable and every panel query of the 8 provisioned dashboards (the real JSON of
   `observability/grafana/dashboards`), variables substituted as Grafana does, timed against InfluxDB directly
   (`/api/v2/query`, one user) at 7 d and 30 d, several models;
4) Grafana with 20 concurrent viewers (`/api/ds/query`, anonymous viewer, 6 requests in flight per viewer as a browser
   over HTTP/1.1), each loading overview, drift and fairness at 7 d then 30 d.

Local run (no AWS): own containers `sc07-influxdb` (:8107) and `sc07-grafana` (:3107); env files in
`lab/results/SC-07/` (never printed). `LAB_SC07_SKIP_LOAD=1` reuses a bucket already loaded.

    LAB_INFLUX_ENV=lab/results/SC-07/influx.env LAB_INFLUX_URL=http://localhost:8107 \\
    LAB_GRAFANA_URL=http://localhost:3107 python -m lab.harness SC-07
"""

import json
import os
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from lab.tests import _platform as C

DRY = os.environ.get("LAB_DRY") == "1"
N_MODELS = int(os.environ.get("LAB_MODELS", 30 if DRY else 300))
DAYS = int(os.environ.get("LAB_DAYS", 3 if DRY else 30))
CAPTURED = int(os.environ.get("LAB_CAPTURED", 2 if DRY else 10))
VIEWERS = int(os.environ.get("LAB_VIEWERS", 4 if DRY else 20))
REPEATS = int(os.environ.get("LAB_REPEATS", 2 if DRY else 5))
VIEWER_ROUNDS = int(os.environ.get("LAB_VIEWER_ROUNDS", 1 if DRY else 3))
LOAD_PROCS = int(os.environ.get("LAB_LOAD_PROCS", 8))
BUCKET = os.environ.get("LAB_SC07_BUCKET", "monitoring")
VERSION = "1.0"
WINDOW_H = 6
ROWS_PER_HOUR = 150
NUMERIC = [f"f{i:02d}" for i in range(16)]
CATEGORICAL = ["region", "language", "channel", "segment"]
FEATURES = NUMERIC + CATEGORICAL
REGIONS = ["Montréal", "Québec", "Est-du-Québec", "Ottawa-Gatineau"]
LANGUAGES = ["fr", "en", "other"]
DASHBOARDS = Path(__file__).resolve().parents[2] / "observability" / "grafana" / "dashboards"
KEY_DASHBOARDS = ["pulsar-overview", "pulsar-drift", "pulsar-fairness"]
RANGES = {"7d": 7, "30d": 30}
BUDGET_S = 3.0
TIMEOUT_S = 60.0  # timeout of the provisioned datasource (jsonData.timeout)
GRAFANA = os.environ.get("LAB_GRAFANA_URL", "http://127.0.0.1:3107")
INFLUX_CONTAINER = os.environ.get("LAB_INFLUX_CONTAINER", "sc07-influxdb")
PARAMS = {
    "env": "local (registry: aws-db)", "models": N_MODELS, "days": DAYS, "window_hours": WINDOW_H,
    "features": len(FEATURES), "captured_models": CAPTURED, "capture_rows_per_hour": ROWS_PER_HOUR,
    "viewers": VIEWERS, "viewer_rounds": VIEWER_ROUNDS, "direct_repeats": REPEATS, "budget_s": BUDGET_S,
    "influxdb": "2.6.1 (sc07-influxdb, :8107)", "grafana": "11.3.0 (sc07-grafana, :3107)", "bucket": BUCKET,
}
CRITERIA = {
    "worst_key_panel_p95_7d_direct_s": ("<", BUDGET_S),
    "worst_key_panel_p95_7d_20_viewers_s": ("<", BUDGET_S),
    "model_variable_values": ("==", N_MODELS),
    "flux_timeouts": ("==", 0),
    "query_errors": ("==", 0),
}


# ------------------------------------------------------------------------------------------- result shapes
def _model_profile(i: int) -> dict:
    """What happens to model i: a third drift on f00 from day DAYS-10, a sixth get a biased rule on language=other"""
    return {"drifts": i % 3 == 0, "biased": i % 6 == 1, "drift_feature": "f00"}


def _window_results(rng, i: int, k: int, n_windows: int) -> list:
    """The results of one model and one window, with the record shapes of observability/analysis.py (drift),
    performance_results, fairness_results (FairnessAnalyzer), FairnessMonitor.update and explain_results"""
    prof = _model_profile(i)
    late = k >= n_windows - 4 * 10  # last 10 days
    out = []
    n = 6 * ROWS_PER_HOUR
    p = rng.uniform(0.01, 1.0, len(FEATURES))
    for j, f in enumerate(FEATURES):
        drifting = prof["drifts"] and late and f == prof["drift_feature"]
        pa = float(rng.uniform(1e-4, 0.01)) if drifting else float(min(1.0, p[j] * 1.2))
        out.append({"metric_type": "drift", "metric_name": "chi2" if f in CATEGORICAL else "ks_2samp", "feature_name": f,
                    "metric_value": float(rng.uniform(0.2, 0.4) if drifting else rng.uniform(0.01, 0.06)),
                    "drift_status": pa < 0.05, "threshold": float(rng.uniform(0.03, 0.06)), "p_adjusted": pa,
                    "confirmed": bool(drifting and k > n_windows - 39), "test_type": "fixed"})
    acc = float(rng.normal(0.9, 0.01))
    labelled = k < n_windows - 8  # labels 48 hours late
    if labelled:
        out += [{"metric_type": "performance", "metric_name": "accuracy", "metric_value": acc, "n": n, "threshold": 0.85,
                 "drift_status": acc < 0.85}]
        out += [{"metric_type": "performance", "metric_name": m, "metric_value": float(rng.normal(0.9, 0.02)), "n": n}
                for m in ("precision", "recall", "f1")]
    out.append({"metric_type": "performance", "metric_name": "label_coverage", "metric_value": 1.0 if labelled else 0.0, "n": n})
    groups = [("language", f"language={g}", "language=fr") for g in LANGUAGES[1:]]
    groups += [("region", f"region={r}", "region=Montréal") for r in REGIONS[1:]]
    groups += [("language|region", f"language={g}|region={r}", "language=fr|region=Montréal")
               for g in LANGUAGES for r in REGIONS if (g, r) != ("fr", "Montréal")]
    for attr, group, ref in groups:
        size = {"language": 200, "region": 300, "language|region": 40}[attr] + int(rng.integers(-20, 20))
        suppressed = attr == "language|region" and "other" in group or size < 30
        biased = prof["biased"] and late and "language=other" in group
        for crit in ("disparate_impact_ratio", "demographic_parity_difference"):
            base = {"metric_type": "fairness", "metric_name": crit, "feature_name": attr, "group": group,
                    "reference_group": ref, "test_type": "fixed",
                    "threshold": [0.8, 1.25] if crit == "disparate_impact_ratio" else [-0.1, 0.1]}
            if suppressed:
                out.append({**base, "method": "suppressed: fewer than 30 rows", "suppressed": True})
                continue
            v = (0.6 if biased else float(rng.normal(1.0, 0.05))) if crit == "disparate_impact_ratio" else \
                (-0.2 if biased else float(rng.normal(0.0, 0.03)))
            half = float(rng.uniform(0.05, 0.15))
            status = True if biased else (None if rng.random() < 0.3 else False)
            out.append({**base, "metric_value": v, "n": size, "ci_low": v - half, "ci_high": v + half,
                        "drift_status": status,
                        "method": "Wilson score interval (MOVER, log scale), 0.975 (Bonferroni over 2 groups)"})
    for g in LANGUAGES[1:]:
        biased = prof["biased"] and late and g == "other"
        out.append({"metric_type": "fairness", "metric_name": "sequential_demographic_parity", "feature_name": "language",
                    "metric_value": -0.2 if biased else float(rng.normal(0, 0.02)), "drift_status": biased,
                    "threshold": [-0.05, 0.05], "group": f"language={g}", "reference_group": "language=fr", "n": 150 * (k + 1),
                    "method": "test by betting (ONS bets), two-sided", "test_type": "sequential",
                    "e_value": float(80.0 if biased else rng.uniform(0.05, 2.0))})
    for name, thr in (("attribution_ndcg_at_k", 0.9), ("attribution_top_k_jaccard", 0.6)):
        v = float(rng.uniform(0.85, 1.0))
        out.append({"metric_type": "explainability", "metric_name": name, "metric_value": v, "drift_status": v < thr,
                    "threshold": thr, "n": n, "ci_low": v - 0.05, "ci_high": min(1.0, v + 0.05),
                    "method": "NDCG of the top 3 features, bootstrap 100 on both windows", "test_type": "fixed"})
    shares = rng.dirichlet(np.ones(len(FEATURES)))
    for j, f in enumerate(FEATURES):
        d = float(rng.normal(0, 0.01))
        out.append({"metric_type": "explainability", "metric_name": "importance_share_difference", "feature_name": f,
                    "metric_value": d, "drift_status": False, "threshold": [-0.05, 0.05], "n": n, "ci_low": d - 0.02,
                    "ci_high": d + 0.02, "method": "current minus reference share of the total importance", "test_type": "fixed"})
        out.append({"metric_type": "explainability", "metric_name": "importance_share", "feature_name": f,
                    "metric_value": float(shares[j]), "n": n})
        for g in LANGUAGES[1:]:
            d = float(rng.normal(0, 0.01))
            out.append({"metric_type": "explainability", "metric_name": "importance_share_difference_by_group",
                        "feature_name": f, "metric_value": d, "drift_status": False, "threshold": [-0.05, 0.05],
                        "group": f"language={g}", "reference_group": "language=fr", "n": 200, "ci_low": d - 0.03,
                        "ci_high": d + 0.03, "method": "stratified bootstrap 100", "test_type": "fixed"})
    return out


def _end() -> datetime:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return now - timedelta(hours=now.hour % WINDOW_H)


def load_models(job: tuple) -> dict:
    """Worker: all the windows of a slice of models through ResultsSink.write_results (one request per window)"""
    from pulsar_data_collection.results import ResultsSink

    models, end_iso = job
    env = C.influx_env()
    end = datetime.fromisoformat(end_iso)
    n_windows = DAYS * 24 // WINDOW_H
    points, seconds, errors = 0, 0.0, 0
    with ResultsSink(env["url"], env["token"], env["org"], BUCKET, timeout_s=60) as sink:
        for i in models:
            rng = np.random.default_rng(i)
            tags = {"model_id": f"model{i:03d}", "model_version": VERSION}
            for k in range(n_windows):
                ts = end - timedelta(hours=WINDOW_H * (n_windows - 1 - k))
                results = _window_results(rng, i, k, n_windows)
                t0 = time.perf_counter()
                try:
                    points += sink.write_results(results, timestamp=ts, tags=tags)
                except Exception:  # noqa: BLE001 - counted, the load goes on
                    errors += 1
                seconds += time.perf_counter() - t0
    return {"points": points, "write_seconds": seconds, "errors": errors, "requests": len(models) * n_windows}


def load_stats(end: datetime, stats_template: dict) -> dict:
    """Capture counters of every model every hour: one request per hour for all models (single StatsReporter)"""
    from pulsar_data_collection.results import ResultsSink

    env = C.influx_env()
    hours = DAYS * 24
    points = 0
    with ResultsSink(env["url"], env["token"], env["org"], BUCKET, timeout_s=60) as sink:
        for h in range(hours):
            ts = end - timedelta(hours=hours - 1 - h)
            items = []
            for i in range(N_MODELS):
                rows = ROWS_PER_HOUR * (h + 1)
                s = {**stats_template, "captured_rows": rows, "written_rows": rows, "queued_rows": 0, "spooled_rows": 0,
                     "dropped_rows": 0, "write_errors": 0, "storage_available": True}
                items.append((s, {"model_id": f"model{i:03d}", "model_version": VERSION}))
            points += sink.write_stats_many(items, timestamp=ts)
    return {"points": points}


def load_captures(end: datetime) -> dict:
    """Captures of the first CAPTURED models through Pulse, 150 rows per hour, personal information masked"""
    import pandas as pd

    from pulsar_data_collection.models import DataWithPrediction
    from pulsar_data_collection.privacy import PiiPolicy

    env = C.influx_env()
    hours = DAYS * 24
    out = {"rows": 0, "dropped": 0, "stats": None}
    comments = ["Première demande.", "Aucune remarque.", "Rappeler au 514-555-0199.", "Courriel : client.demo@exemple.ca"]
    for i in range(CAPTURED):
        rng = np.random.default_rng(1000 + i)
        pulse = C.make_pulse(f"model{i:03d}", BUCKET, C.capture_options(pii=PiiPolicy(columns=["comment"]),
                                                                         max_queue_rows=500_000), env, version=VERSION)
        for h in range(hours):
            ts = end - timedelta(hours=hours - h)
            n = ROWS_PER_HOUR
            x = pd.DataFrame(rng.normal(size=(n, len(NUMERIC))), columns=NUMERIC)
            if _model_profile(i)["drifts"] and h >= hours - 240:
                x["f00"] += 0.5
            x["region"] = rng.choice(REGIONS, n)
            x["language"] = rng.choice(LANGUAGES, n, p=[0.7, 0.2, 0.1])
            x["channel"] = rng.choice(["web", "phone", "office"], n)
            x["segment"] = rng.choice(["a", "b", "c"], n)
            x["comment"] = rng.choice(comments, n, p=[0.48, 0.48, 0.02, 0.02])
            proba = rng.uniform(0, 1, n)
            x["y_true"] = (proba > 0.5).astype(float) if h < hours - 48 else np.nan
            y = pd.DataFrame({"y_pred": (proba >= 0.5).astype(int), "y_pred_proba": proba})
            pulse.capture_data(DataWithPrediction(timestamp=ts, predictions=y, data_points=x,
                                                  features_names=FEATURES + ["comment"]))
            if h % 24 == 23:
                pulse.flush(120)
        pulse.flush(300)
        pulse.close(60)
        st = pulse.stats()
        out["rows"] += st.get("written_rows", 0)
        out["dropped"] += st.get("dropped_rows", 0)
        out["stats"] = st
    return out


def load(ctx) -> dict:
    end = _end()
    t0 = time.perf_counter()
    slices = [list(range(N_MODELS))[k::LOAD_PROCS] for k in range(LOAD_PROCS)]
    with ProcessPoolExecutor(LOAD_PROCS) as pool:
        parts = list(pool.map(load_models, [(s, end.isoformat()) for s in slices]))
    results_s = time.perf_counter() - t0
    res = {k: sum(p[k] for p in parts) for k in ("points", "errors", "requests")}
    res["write_seconds_sum"] = sum(p["write_seconds"] for p in parts)
    t1 = time.perf_counter()
    cap = load_captures(end)
    captures_s = time.perf_counter() - t1
    template = {k: v for k, v in (cap["stats"] or {}).items() if isinstance(v, (int, float, bool))}
    t2 = time.perf_counter()
    stats = load_stats(end, template)
    stats_s = time.perf_counter() - t2
    return {
        "end": end.isoformat(), "results_points": res["points"], "results_requests": res["requests"],
        "results_write_errors": res["errors"], "results_seconds": round(results_s, 1),
        "results_points_per_s": round(res["points"] / results_s), "results_points_per_request": round(res["points"] / max(1, res["requests"]), 1),
        "results_request_ms_mean": round(1000 * res["write_seconds_sum"] / max(1, res["requests"]), 2),
        "capture_rows": cap["rows"], "capture_dropped": cap["dropped"], "captures_seconds": round(captures_s, 1),
        "stats_points": stats["points"], "stats_seconds": round(stats_s, 1),
    }


# ------------------------------------------------------------------------------------------- queries
def flux(q: str, env: dict, timeout: float = TIMEOUT_S) -> tuple:
    """(seconds, bytes, rows, error) of one Flux query on /api/v2/query, the whole CSV body read"""
    body = json.dumps({"query": q, "type": "flux", "dialect": {"annotations": ["datatype", "group", "default"]}}).encode()
    req = urllib.request.Request(f"{env['url']}/api/v2/query?org={urllib.request.quote(env['org'])}", data=body, method="POST",
                                 headers={"Authorization": f"Token {env['token']}", "Content-Type": "application/json",
                                          "Accept": "application/csv"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dt = time.perf_counter() - t0
        rows = sum(1 for line in data.decode("utf-8", "replace").splitlines() if line and not line.startswith("#")
                   and not line.startswith(",result"))
        return dt, len(data), rows, None
    except urllib.error.HTTPError as exc:
        return time.perf_counter() - t0, 0, 0, f"HTTP {exc.code}: {exc.read()[:300].decode('utf-8', 'replace')}"
    except Exception as exc:  # noqa: BLE001 - timeouts are results
        return time.perf_counter() - t0, 0, 0, f"{type(exc).__name__}: {str(exc)[:200]}"


def flux_values(q: str, env: dict, column: str = "_value") -> list:
    from influxdb_client import InfluxDBClient

    with InfluxDBClient(url=env["url"], token=env["token"], org=env["org"], timeout=int(TIMEOUT_S * 1000)) as c:
        tables = c.query_api().query(q)
    return [rec.values.get(column) for t in tables for rec in t.records]


def dashboards() -> dict:
    """uid -> {"title", "variables": [(name, query, default)], "panels": [(panel id, title, type, refId, query)]}"""
    out = {}
    for path in sorted(DASHBOARDS.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        panels = []

        def walk(items):
            for p in items:
                if p.get("type") == "row":
                    walk(p.get("panels", []))
                    continue
                for t in p.get("targets", []) or []:
                    if t.get("query"):
                        panels.append((p["id"], p.get("title", ""), p["type"], t["refId"], t["query"]))
        walk(d["panels"])
        variables = [(v["name"], v.get("query"), (v.get("current") or {}).get("value"), v["type"])
                     for v in d["templating"]["list"]]
        out[d["uid"]] = {"title": d["title"], "variables": variables, "panels": panels}
    return out


def substitute(q: str, values: dict) -> str:
    for k, v in values.items():
        q = q.replace("${" + k + "}", str(v))
    return q


def absolute(q: str, days: int, now: datetime) -> str:
    """The time macros as the Grafana Flux backend expands them, for a direct query"""
    start = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    stop = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    window = f"{max(1, int(days * 86400 / 1000))}s"
    return q.replace("v.timeRangeStart", start).replace("v.timeRangeStop", stop).replace("v.windowPeriod", window)


def resolve_variables(board: dict, env: dict, model: str, timings: list, pick: int = 0) -> dict:
    """Run the templating variables in order, as Grafana does on load; keep the chosen model and the first value
    value (the dashboard default when it exists, else the value at `pick`) of the others"""
    values = {}
    for name, query, default, kind in board["variables"]:
        if kind == "textbox":
            values[name] = BUCKET
            continue
        q = substitute(query, values)
        dt, _, _, err = flux(q, env)
        got = []
        if not err:
            try:
                got = [str(v) for v in flux_values(q, env)]
            except Exception as exc:  # noqa: BLE001
                err = str(exc)[:200]
        timings.append({"variable": name, "seconds": dt, "values": len(got), "error": err})
        if name == "model":
            values[name] = model
        elif default in got:
            values[name] = default
        elif got:
            values[name] = got[pick % len(got)]
        else:
            values[name] = default or ""
    return values


def pctl(values, q):
    return round(float(np.percentile(values, q)), 3) if values else None


def summarise(samples: list, keys=("dashboard", "panel_id", "ref", "title", "range")) -> list:
    groups = {}
    for s in samples:
        groups.setdefault(tuple(s[k] for k in keys), []).append(s)
    out = []
    for key, rows in groups.items():
        t = [r["seconds"] for r in rows]
        out.append({**dict(zip(keys, key)), "n": len(rows), "p50_s": pctl(t, 50), "p95_s": pctl(t, 95),
                    "max_s": round(max(t), 3), "errors": sum(1 for r in rows if r["error"]),
                    "timeouts": sum(1 for r in rows if r["error"] and "timed out" in r["error"].lower()),
                    "rows_max": max(r.get("rows", 0) for r in rows), "error_sample": next((r["error"] for r in rows if r["error"]), None)})
    return sorted(out, key=lambda r: -r["p95_s"])


def direct(env: dict, boards: dict, models: list, var_timings: list) -> list:
    samples = []
    now = datetime.now(timezone.utc)
    for uid, board in boards.items():
        for rep in range(REPEATS):
            model = models[rep % len(models)]
            values = resolve_variables(board, env, model, var_timings, pick=rep)
            for rng_name, days in RANGES.items():
                for pid, title, ptype, ref, query in board["panels"]:
                    q = absolute(substitute(query, values), days, now)
                    dt, size, rows, err = flux(q, env)
                    samples.append({"dashboard": uid, "panel_id": pid, "ref": ref, "title": title, "type": ptype,
                                    "range": rng_name, "model": model, "seconds": dt, "bytes": size, "rows": rows,
                                    "error": err})
    return samples


# ------------------------------------------------------------------------------------------- Grafana viewers
def grafana_query(query: str, days: int, timeout: float = TIMEOUT_S + 10) -> tuple:
    body = json.dumps({"queries": [{"refId": "A", "datasource": {"type": "influxdb", "uid": "pulsar-influx"}, "query": query,
                                    "maxDataPoints": 1500, "intervalMs": int(days * 86400 * 1000 / 1500)}],
                       "from": f"now-{days}d", "to": "now"}).encode()
    req = urllib.request.Request(f"{GRAFANA}/api/ds/query", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        dt = time.perf_counter() - t0
        r = (data.get("results") or {}).get("A") or {}
        err = r.get("error")
        frames = r.get("frames") or []
        rows = sum(len((f.get("data") or {}).get("values", [[]])[0] if (f.get("data") or {}).get("values") else [])
                   for f in frames)
        return dt, rows, err
    except urllib.error.HTTPError as exc:
        return time.perf_counter() - t0, 0, f"HTTP {exc.code}: {exc.read()[:300].decode('utf-8', 'replace')}"
    except Exception as exc:  # noqa: BLE001
        return time.perf_counter() - t0, 0, f"{type(exc).__name__}: {str(exc)[:200]}"


def viewers(env: dict, boards: dict, models: list, n_viewers: int) -> dict:
    """n viewers at once; each loads every key dashboard at 7 d then 30 d, VIEWER_ROUNDS times; within a dashboard,
    the variables first (in order), then all panels with 6 requests in flight"""
    samples, var_samples, lock = [], [], threading.Lock()
    start_gate = threading.Barrier(n_viewers)
    variants = {}
    for uid in KEY_DASHBOARDS:  # the non-model variables, resolved once: every model has the same features and groups
        variants[uid] = [resolve_variables(boards[uid], env, models[0], [], pick=k) for k in range(REPEATS)]

    def viewer(v: int):
        rng = random.Random(v)
        start_gate.wait()
        for rnd in range(VIEWER_ROUNDS):
            for rng_name, days in RANGES.items():
                for uid in KEY_DASHBOARDS:
                    board = boards[uid]
                    values = {**rng.choice(variants[uid]), "model": rng.choice(models)}
                    for name, query, _, kind in board["variables"]:
                        if kind == "textbox":
                            continue
                        dt, n, err = grafana_query(substitute(query, values), days)
                        with lock:
                            var_samples.append({"variable": f"{uid}:{name}", "seconds": dt, "error": err})
                    jobs = [(pid, title, ref, absolute_q) for pid, title, _, ref, absolute_q in board["panels"]]
                    t_board = time.perf_counter()
                    with ThreadPoolExecutor(6) as pool:
                        futs = {pool.submit(grafana_query, substitute(q, values), days): (pid, title, ref)
                                for pid, title, ref, q in jobs}
                        for f, (pid, title, ref) in futs.items():
                            dt, rows, err = f.result()
                            with lock:
                                samples.append({"dashboard": uid, "panel_id": pid, "ref": ref, "title": title,
                                                "range": rng_name, "seconds": dt, "rows": rows, "error": err,
                                                "viewer": v, "round": rnd})
                    with lock:
                        samples.append({"dashboard": uid, "panel_id": 0, "ref": "-", "title": "(whole dashboard)",
                                        "range": rng_name, "seconds": time.perf_counter() - t_board, "rows": 0,
                                        "error": None, "viewer": v, "round": rnd})

    t0 = time.perf_counter()
    cpu0 = _docker_cpu()
    with ThreadPoolExecutor(n_viewers) as pool:
        list(pool.map(viewer, range(n_viewers)))
    return {"samples": samples, "variables": var_samples, "seconds": time.perf_counter() - t0, "docker_after": cpu0}


def _docker_cpu() -> dict:
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}",
                          INFLUX_CONTAINER, "sc07-grafana"], capture_output=True, text=True, timeout=60).stdout
    return {line.split("\t")[0]: line.split("\t")[1:] for line in out.splitlines() if "\t" in line}


class DockerSampler:
    """docker stats of the two containers every few seconds (CPU %, memory) during the viewers' phase"""

    def __init__(self):
        self.samples, self._stop = [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                for name, (cpu, mem) in _docker_cpu().items():
                    self.samples.append({"t": time.time(), "name": name, "cpu_pct": float(cpu.rstrip("%")), "mem": mem})
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(2)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(10)

    def peak(self) -> dict:
        out = {}
        for s in self.samples:
            out[s["name"]] = max(out.get(s["name"], 0), s["cpu_pct"])
        return out


# ------------------------------------------------------------------------------------------- cardinality, disk
def cardinality(env: dict) -> dict:
    out = {}
    start = f"-{DAYS + 2}d"
    q = 'import "influxdata/influxdb"\ninfluxdb.cardinality(bucket: "{b}", start: {s}{p})'
    out["bucket_series"] = int(flux_values(q.format(b=BUCKET, s=start, p=""), env)[0])
    for m in ("pulsar_results", "pulsar_capture_stats"):
        out[f"{m}_series"] = int(flux_values(q.format(b=BUCKET, s=start, p=f', predicate: (r) => r._measurement == "{m}"'), env)[0])
    one = 'import "influxdata/influxdb"\ninfluxdb.cardinality(bucket: "{b}", start: {s}, predicate: (r) => r._measurement == "pulsar_results" and r.model_id == "model001")'
    out["pulsar_results_series_per_model"] = int(flux_values(one.format(b=BUCKET, s=start), env)[0])
    # the index counts measurement + tag set; TSM keeps one series per field of each tag set
    fs = (f'from(bucket: "{BUCKET}") |> range(start: {start}) |> filter(fn: (r) => r._measurement == "pulsar_results")'
          ' |> first() |> keep(columns: ["_field"]) |> group() |> count(column: "_field")')
    out["pulsar_results_field_series"] = int(flux_values(fs, env, "_field")[0])
    tags = {}
    for tag in ("model_id", "model_version", "metric_type", "metric_name", "feature_name", "group", "reference_group", "test_type"):
        vals = flux_values(f'import "influxdata/influxdb/schema"\nschema.tagValues(bucket: "{BUCKET}", tag: "{tag}", '
                           f'predicate: (r) => r._measurement == "pulsar_results", start: {start})', env)
        tags[tag] = len(vals)
    out["pulsar_results_distinct_tag_values"] = tags
    fields = flux_values(f'import "influxdata/influxdb/schema"\nschema.measurementFieldKeys(bucket: "{BUCKET}", '
                         f'measurement: "pulsar_results", start: {start})', env)
    out["pulsar_results_fields"] = sorted(str(f) for f in fields)
    measurements = flux_values(f'import "influxdata/influxdb/schema"\nschema.measurements(bucket: "{BUCKET}", start: {start})', env)
    out["measurements"] = len(measurements)
    return out


def disk(env: dict) -> dict:
    """Bytes of the bucket's TSM files, its WAL and the whole engine, from inside the InfluxDB container"""
    def du(path):
        r = subprocess.run(["docker", "exec", INFLUX_CONTAINER, "du", "-sb", path], capture_output=True, text=True, timeout=120)
        return int(r.stdout.split()[0]) if r.returncode == 0 and r.stdout.strip() else None

    from influxdb_client import BucketsApi

    with C.client(env) as c:
        bucket_id = BucketsApi(c).find_bucket_by_name(BUCKET).id
    files = subprocess.run(["docker", "exec", INFLUX_CONTAINER, "sh", "-c",
                            f"find /var/lib/influxdb2/engine/data/{bucket_id} -name '*.tsm' | wc -l"],
                           capture_output=True, text=True, timeout=120).stdout.strip()
    return {"bucket_tsm_bytes": du(f"/var/lib/influxdb2/engine/data/{bucket_id}"),
            "bucket_wal_bytes": du(f"/var/lib/influxdb2/engine/wal/{bucket_id}"),
            "engine_bytes": du("/var/lib/influxdb2/engine"), "tsm_files": int(files) if files.isdigit() else None,
            "shards": subprocess.run(["docker", "exec", INFLUX_CONTAINER, "sh", "-c",
                                      f"ls /var/lib/influxdb2/engine/data/{bucket_id}/autogen | wc -l"],
                                     capture_output=True, text=True, timeout=60).stdout.strip()}


# ------------------------------------------------------------------------------------------- cardinality ladder
LADDER = [int(v) for v in os.environ.get("LAB_SC07_LADDER", "").split(",") if v]
LADDER_BUCKET = "sc07_ladder"


def load_ladder(job: tuple) -> int:
    """Worker: one window of results per model (the last one) into the ladder bucket"""
    from pulsar_data_collection.results import ResultsSink

    models, end_iso = job
    env = C.influx_env()
    n_windows = DAYS * 24 // WINDOW_H
    points = 0
    with ResultsSink(env["url"], env["token"], env["org"], LADDER_BUCKET, timeout_s=60) as sink:
        for i in models:
            rng = np.random.default_rng(i)
            points += sink.write_results(_window_results(rng, i, n_windows - 20, n_windows), timestamp=datetime.fromisoformat(end_iso),
                                         tags={"model_id": f"model{i:05d}", "model_version": VERSION})
    return points


# the model variable as provisioned before SC-07 (negation inside the predicate), kept to compare at each step
MODEL_VAR_BEFORE = ('import "influxdata/influxdb/schema"\nschema.tagValues(bucket: "${bucket}", tag: "model_id", '
                    'predicate: (r) => r._measurement == "pulsar_results" and r.model_id != "permit_assistant", start: -30d)')


def _timed(q: str, env: dict, n: int = 5) -> dict:
    t = [flux(q, env) for _ in range(n)]
    try:
        values = len(flux_values(q, env))
    except Exception:  # noqa: BLE001 - a timeout is a result
        values = None
    return {"p50_s": pctl([x[0] for x in t], 50), "max_s": round(max(x[0] for x in t), 3), "values": values,
            "errors": sum(1 for x in t if x[3]), "error_sample": next((x[3][:160] for x in t if x[3]), None)}


def ladder(env: dict) -> list:
    """Series cardinality against the variable queries: N models with one window each in a separate bucket, the model
    variable (as provisioned, and as before SC-07) and four per-model panels timed at each step (direct, one user).
    A failing step is recorded and ends the ladder."""
    C.ensure_bucket(LADDER_BUCKET, env)
    end = _end()
    board = dashboards()["pulsar-drift"]
    overview = dashboards()["pulsar-overview"]
    probes = [p for p in board["panels"] if p[0] in (8, 2)] + [p for p in overview["panels"] if p[0] in (9, 3)]
    model_var = next(v for v in board["variables"] if v[0] == "model")[1]
    out, have = [], 0
    for n in LADDER:
        step = {"models": n}
        try:
            t0 = time.perf_counter()
            slices = [list(range(have, n))[k::LOAD_PROCS] for k in range(LOAD_PROCS)]
            with ProcessPoolExecutor(LOAD_PROCS) as pool:
                step["points_written"] = sum(pool.map(load_ladder, [(s, end.isoformat()) for s in slices]))
            step["write_s"] = round(time.perf_counter() - t0, 1)
            step["write_points_per_s"] = round(step["points_written"] / step["write_s"])
            have = n
            time.sleep(10)
            q = f'import "influxdata/influxdb"\ninfluxdb.cardinality(bucket: "{LADDER_BUCKET}", start: -40d)'
            dt, _, _, err = flux(q, env, timeout=300)
            step["cardinality_query_s"], step["cardinality_query_error"] = round(dt, 2), err
            step["index_series"] = 141 * n  # 141 tag sets per model (measured on the 300-model bucket and below)
            if not err:
                step["index_series"] = int(flux_values(q, env)[0]) if dt < 60 else step["index_series"]
            step["model_variable"] = _timed(substitute(model_var, {"bucket": LADDER_BUCKET}), env)
            step["model_variable_before_fix"] = _timed(substitute(MODEL_VAR_BEFORE, {"bucket": LADDER_BUCKET}), env)
            now = datetime.now(timezone.utc)
            values = {"bucket": LADDER_BUCKET, "model": "model00001", "version": VERSION, "feature": "f00"}
            step["panels_7d"] = {f"{pid}{ref} {title}": _timed(absolute(substitute(query, values), 7, now), env, 3)
                                 for pid, title, _, ref, query in probes}
            step["influx_mem"] = _docker_cpu().get(INFLUX_CONTAINER, [None, None])[1]
        except Exception as exc:  # noqa: BLE001 - the step that breaks is the result
            step["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            out.append(step)
            print(f"ladder {n}: {step}", flush=True)
            break
        out.append(step)
        print(f"ladder {n}: {step}", flush=True)
    return out


# ------------------------------------------------------------------------------------------- run
def run(ctx):
    env = C.influx_env()
    ctx.note("Run locally on the workstation (registry env: aws-db): own containers sc07-influxdb 2.6.1 and sc07-grafana "
             "11.3.0; the host was shared with SC-04 (network toxics) running at the same time.")
    metrics = {}
    C.ensure_bucket(BUCKET, env)
    if os.environ.get("LAB_SC07_SKIP_LOAD") != "1":
        metrics["load"] = load(ctx)
        (ctx.workdir / "load.json").write_text(json.dumps(metrics["load"], indent=2), encoding="utf-8")
    elif (ctx.workdir / "load.json").exists():
        metrics["load"] = json.loads((ctx.workdir / "load.json").read_text(encoding="utf-8"))
    time.sleep(0 if DRY else 30)  # let the cache snapshot of the last writes settle
    metrics["cardinality"] = cardinality(env)
    metrics["disk"] = disk(env)

    boards = dashboards()
    models = [f"model{i:03d}" for i in range(CAPTURED)]  # the models with captures: every panel has its data
    var_timings = []
    samples = direct(env, boards, models, var_timings)
    (ctx.workdir / "direct_samples.json").write_text(json.dumps(samples, indent=1, default=str), encoding="utf-8")
    per_panel = summarise(samples)
    (ctx.workdir / "direct_panels.json").write_text(json.dumps(per_panel, indent=1, default=str), encoding="utf-8")
    model_values = [t["values"] for t in var_timings if t["variable"] == "model"]

    # Grafana: the provisioned dashboards are the files measured
    with urllib.request.urlopen(f"{GRAFANA}/api/search?type=dash-db", timeout=10) as resp:
        provisioned = sorted(d["uid"] for d in json.loads(resp.read()))
    with DockerSampler() as dock:
        single = viewers(env, boards, models, 1)
        load20 = viewers(env, boards, models, VIEWERS)
    (ctx.workdir / "grafana_samples.json").write_text(json.dumps({"single": single, "viewers": load20}, default=str),
                                                     encoding="utf-8")
    g1, g20 = summarise(single["samples"]), summarise(load20["samples"])
    (ctx.workdir / "grafana_panels.json").write_text(json.dumps({"single": g1, "viewers": g20}, indent=1, default=str),
                                                    encoding="utf-8")

    def worst(rows, rng, boards_=KEY_DASHBOARDS):
        rows = [r for r in rows if r["range"] == rng and r["dashboard"] in boards_ and r["panel_id"] != 0]
        return max(rows, key=lambda r: r["p95_s"]) if rows else None

    by_board = {}
    for uid in boards:
        by_board[uid] = {}
        for rng in RANGES:
            rows = [r for r in per_panel if r["dashboard"] == uid and r["range"] == rng]
            w = max(rows, key=lambda r: r["p95_s"])
            by_board[uid][rng] = {"panels": len(rows), "over_budget": sum(r["p95_s"] >= BUDGET_S for r in rows),
                                  "worst_p95_s": w["p95_s"], "worst_max_s": w["max_s"], "worst_panel": f"{w['panel_id']}{w['ref']} {w['title']}",
                                  "errors": sum(r["errors"] for r in rows)}
    viewers_by_board = {}
    for uid in KEY_DASHBOARDS:
        viewers_by_board[uid] = {}
        for rng in RANGES:
            rows = [r for r in g20 if r["dashboard"] == uid and r["range"] == rng and r["panel_id"] != 0]
            whole = [r for r in g20 if r["dashboard"] == uid and r["range"] == rng and r["panel_id"] == 0]
            w = max(rows, key=lambda r: r["p95_s"])
            viewers_by_board[uid][rng] = {"worst_p95_s": w["p95_s"], "worst_max_s": w["max_s"],
                                          "worst_panel": f"{w['panel_id']}{w['ref']} {w['title']}",
                                          "over_budget": sum(r["p95_s"] >= BUDGET_S for r in rows),
                                          "whole_dashboard_p50_s": whole[0]["p50_s"] if whole else None,
                                          "whole_dashboard_p95_s": whole[0]["p95_s"] if whole else None,
                                          "errors": sum(r["errors"] for r in rows)}
    all_direct_errors = [s for s in samples if s["error"]]
    g_errors = [s for s in single["samples"] + load20["samples"] if s["error"]]
    timeouts = sum(1 for s in samples + single["samples"] + load20["samples"]
                   if s["error"] and ("timed out" in s["error"].lower() or "timeout" in s["error"].lower()))
    w7, w30 = worst(per_panel, "7d"), worst(per_panel, "30d")
    v7, v30 = worst(g20, "7d"), worst(g20, "30d")
    var_summary = {}
    for t in var_timings:
        var_summary.setdefault(t["variable"], []).append(t)
    metrics.update({
        "provisioned_dashboards": provisioned,
        "panels_queries": {uid: len(b["panels"]) for uid, b in boards.items()},
        "model_variable_values": min(model_values) if model_values else 0,
        "variables": {k: {"values_min": min(x["values"] for x in v), "values_max": max(x["values"] for x in v),
                          "p50_s": pctl([x["seconds"] for x in v], 50), "max_s": round(max(x["seconds"] for x in v), 3),
                          "errors": sum(1 for x in v if x["error"])} for k, v in var_summary.items()},
        "direct_by_dashboard": by_board,
        "viewers_by_dashboard": viewers_by_board,
        "worst_key_panel_p95_7d_direct_s": w7["p95_s"] if w7 else None, "worst_key_panel_7d_direct": w7,
        "worst_key_panel_p95_30d_direct_s": w30["p95_s"] if w30 else None, "worst_key_panel_30d_direct": w30,
        "worst_key_panel_p95_7d_20_viewers_s": v7["p95_s"] if v7 else None, "worst_key_panel_7d_20_viewers": v7,
        "worst_key_panel_p95_30d_20_viewers_s": v30["p95_s"] if v30 else None, "worst_key_panel_30d_20_viewers": v30,
        "grafana_single_viewer_worst_7d": worst(g1, "7d"), "grafana_single_viewer_worst_30d": worst(g1, "30d"),
        "viewers_phase_seconds": round(load20["seconds"], 1), "viewer_requests": len(load20["samples"]),
        "docker_cpu_peak_pct": dock.peak(),
        "flux_timeouts": timeouts, "query_errors": len(all_direct_errors) + len(g_errors),
        "error_samples": sorted({s["error"][:200] for s in all_direct_errors + g_errors})[:10],
        "versions": C.versions(),
    })
    over = [r for r in per_panel if r["range"] == "7d" and r["p95_s"] >= BUDGET_S]
    if over:
        ctx.limit(f"{len(over)} panel queries over {BUDGET_S} s at 7 d (direct, one user)",
                  "; ".join(f"{r['dashboard']} {r['panel_id']}{r['ref']} {r['title']}: p95 {r['p95_s']} s" for r in over[:8]),
                  "see the report: query shape of these panels")
    over20 = [r for r in g20 if r["range"] == "7d" and r["p95_s"] >= BUDGET_S and r["panel_id"] != 0]
    if over20:
        ctx.limit(f"{len(over20)} key panels over {BUDGET_S} s at 7 d with {VIEWERS} concurrent viewers",
                  "; ".join(f"{r['dashboard']} {r['panel_id']}{r['ref']} {r['title']}: p95 {r['p95_s']} s" for r in over20[:8]),
                  "see the report")
    if LADDER:
        try:
            metrics["cardinality_ladder"] = ladder(env)
        except Exception as exc:  # noqa: BLE001 - the main measurements are kept
            metrics["cardinality_ladder_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return metrics


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["load"]:
        print(json.dumps(load(None), indent=2))
    elif sys.argv[1:] == ["ladder"]:
        print(json.dumps(ladder(C.influx_env()), indent=2))
