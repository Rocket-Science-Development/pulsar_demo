"""Periodic analysis: every `--every` minutes, analyse the last closed window of a model's captures and write the results.

The reference is read once at start (a period known to be sound). Each run reads the window that just closed from
InfluxDB, then writes calibrated drift with confirmation, performance (when labels have arrived), fairness on a rolling
window and the anytime-valid monitor to `pulsar_results`, where the dashboards read them.

Each run also writes the hourly p10 / median / p90 of the numeric features to `pulsar_feature_quantiles` (read by the
drift dashboard's quantile band instead of the raw captures, lab SC-07 L1). The last `--quantile-hours` closed hours
are recomputed at every run, so rows that arrive late are counted; rewriting an hour overwrites its points.

Backfill: each run then looks, over the last `--backfill-days` (default 30), for the older closed hours that have raw
captures but no aggregate, or fewer aggregated values than captured ones (rows that arrived after the hour was
aggregated). One grouped Flux query finds them for every feature (and every model of `--models`); the oldest
`--backfill-max-hours` (default 168) of each model are aggregated, in reads of at most 24 hours, so that a run stays
bounded and the history fills up over the next runs. An hour aggregated from all its rows is not a gap: rerunning
writes nothing. `--backfill-only` fills the whole look-back at once, without the analyses (no reference needed), and
exits. The hour being captured is never aggregated (only closed hours are).

    python analysis_job.py --model admissibilite --version 1.0 --reference-start 2026-10-26T00:00:00Z \\
        --reference-hours 48 --window-hours 6 --every 360
    python analysis_job.py ... --once      # analyse the last closed window and exit
    python analysis_job.py --model admissibilite --version 1.0 --backfill-only --backfill-days 365  # history, once
    python analysis_job.py --models a:1.0,b:2.1 --backfill-only            # several models, gaps found in one query

Environment: INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG (default pulsarml), INFLUX_BUCKET (default monitoring).
"""

import argparse
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from influxdb_client import InfluxDBClient
from pulsar_data_collection.results import ResultsSink
from pulsar_metrics.fairness import FairnessMonitor

from analysis import (
    QUANTILE_MEASUREMENT,
    TABULAR_CATEGORICAL,
    TABULAR_NUMERIC,
    DriftWatch,
    backfill_ranges,
    fairness_results,
    performance_results,
    quantile_gaps,
    write_hourly_quantiles,
)
from seed_demo import connection

HOUR = timedelta(hours=1)


def captures(model: str, version: str) -> str:
    """Measurement of the captures of a model (written by Pulse)"""
    return f"{model}_{version}_input_data"


def _flux_time(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _flux_string(value) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _or(terms: list) -> str:
    """Disjunction as a balanced tree: Flux refuses a long flat chain of `or` ("Program is nested too deep" at 300
    models), and a balanced one stays a predicate pushed down to storage, unlike `contains()`"""
    if len(terms) == 1:
        return terms[0]
    half = len(terms) // 2
    return f"({_or(terms[:half])} or {_or(terms[half:])})"


def _any(column: str, values) -> str:
    return "(" + _or([f"r.{column} == {_flux_string(v)}" for v in values]) + ")"


def read(client, conn, model: str, version: str, start: datetime, stop: datetime, fields: list = None) -> pd.DataFrame:
    """Captures of [start, stop), one row per capture; with `fields`, only those fields (and the tags)"""
    only = f"\n      |> filter(fn: (r) => {_any('_field', fields)})" if fields else ""
    query = f"""
    from(bucket: "{conn['bucket']}")
      |> range(start: {start.isoformat()}, stop: {stop.isoformat()})
      |> filter(fn: (r) => r["_measurement"] == "{captures(model, version)}"){only}
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    frame = client.query_api().query_data_frame(query=query)
    if isinstance(frame, list):
        frame = pd.concat(frame, ignore_index=True)
    if frame.empty:
        return frame
    frame = frame.rename(columns={"_time": "pred_timestamp"}).sort_values("pred_timestamp").reset_index(drop=True)
    for column in TABULAR_NUMERIC + ["y_pred", "y_pred_proba", "y_true"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.assign(model_id=model, model_version=version)


def aggregate_quantiles(client, conn, sink, model: str, version: str, start: datetime, stop: datetime, features: list,
                        chunk_hours: int = 24) -> dict:
    """Hourly quantiles of the captures in [start, stop) written to `pulsar_feature_quantiles`, read in chunks of
    `chunk_hours` (only the feature fields). `start` and `stop` must be on the hour (complete hours only). Returns the
    counts and the seconds spent reading, computing and writing."""
    tags = {"model_id": model, "model_version": version}
    out = {"rows": 0, "points": 0, "requests": 0, "hours": 0, "read_s": 0.0, "compute_s": 0.0, "write_s": 0.0}
    at = start
    while at < stop:
        until = min(stop, at + timedelta(hours=chunk_hours))
        t0 = time.perf_counter()
        frame = read(client, conn, model, version, at, until, fields=features)
        out["read_s"] += time.perf_counter() - t0
        out["rows"] += len(frame)
        if not frame.empty:
            written = write_hourly_quantiles(sink, frame, features, tags)
            for key in ("points", "requests", "hours", "compute_s", "write_s"):
                out[key] += written[key]
        at = until
    return out


def gaps_flux(bucket: str, models: list, features: list, start: datetime, stop: datetime) -> str:
    """One query for every model and feature: per model and closed hour of [start, stop), the count of the raw values
    of the features (result "raw") and the sum of the `n` of their aggregates (result "agg"), both stamped at the start
    of the window. Both sides are first reduced in storage, per series and hour (aggregateWindow count / sum are
    pushed down), then summed per model: the answer is one row per model and hour, whatever the number of features."""
    by_name = _any("_measurement", [captures(m, v) for m, v in models])
    pairs = _or([f"(r.model_id == {_flux_string(m)} and r.model_version == {_flux_string(v)})" for m, v in models])
    return f"""
from(bucket: {_flux_string(bucket)})
  |> range(start: {_flux_time(start)}, stop: {_flux_time(stop)})
  |> filter(fn: (r) => {by_name} and {_any("_field", features)})
  |> aggregateWindow(every: 1h, fn: count, timeSrc: "_start", createEmpty: false)
  |> group(columns: ["_measurement"])
  |> aggregateWindow(every: 1h, fn: sum, timeSrc: "_start", createEmpty: false)
  |> keep(columns: ["_measurement", "_time", "_value"])
  |> yield(name: "raw")
from(bucket: {_flux_string(bucket)})
  |> range(start: {_flux_time(start + HOUR)}, stop: {_flux_time(stop + HOUR)})
  |> filter(fn: (r) => r._measurement == "{QUANTILE_MEASUREMENT}" and r._field == "n" and {pairs}
                       and {_any("feature_name", features)})
  |> aggregateWindow(every: 1h, fn: sum, timeSrc: "_start", createEmpty: false)
  |> group(columns: ["model_id", "model_version"])
  |> aggregateWindow(every: 1h, fn: sum, timeSrc: "_start", createEmpty: false)
  |> keep(columns: ["model_id", "model_version", "_time", "_value"])
  |> yield(name: "agg")
"""


def find_quantile_gaps(client, conn, models: list, features: list, start: datetime, stop: datetime) -> dict:
    """{(model, version): [hour ends to aggregate, oldest first]} over the closed hours of [start, stop) (on the hour),
    found in one query for all the models (see `gaps_flux` and `analysis.quantile_gaps`)."""
    models = [tuple(m) for m in models]
    raw = {m: {} for m in models}
    agg = {m: {} for m in models}
    model_of = {captures(m, v): (m, v) for m, v in models}
    for table in client.query_api().query(gaps_flux(conn["bucket"], models, features, start, stop)):
        for record in table.records:
            values = record.values
            if values.get("result") == "raw":  # stamped at the start of the hour: key by its end
                key, end = model_of.get(values.get("_measurement")), record.get_time() + HOUR
                counts = raw
            else:  # an aggregate is stamped at the end of its hour, which starts the window it falls in
                key, end = (values.get("model_id"), values.get("model_version")), record.get_time()
                counts = agg
            if key in counts:
                counts[key][end] = counts[key].get(end, 0) + float(values["_value"])
    return {m: quantile_gaps(raw[m], agg[m]) for m in models}


def backfill_quantiles(client, conn, sink, models: list, features: list, start: datetime, stop: datetime,
                       max_hours: int = None, chunk_hours: int = 24) -> dict:
    """Aggregate the hours of [start, stop) that have captures but no complete aggregate: the `max_hours` oldest of
    each model (all if None), in reads of at most `chunk_hours`. `start` and `stop` on the hour, `stop` at most the
    current hour (the hour being captured is never aggregated). Idempotent: an hour aggregated from all its rows is not
    a gap, so a second run finds nothing and writes nothing.

    Returns {"models", "gap_hours", "hours", "remaining", "rows", "points", "requests", "detect_s", "read_s",
    "compute_s", "write_s"}."""
    out = {"models": len(models), "gap_hours": 0, "hours": 0, "remaining": 0, "rows": 0, "points": 0, "requests": 0,
           "detect_s": 0.0, "read_s": 0.0, "compute_s": 0.0, "write_s": 0.0}
    if start >= stop or not models:
        return out
    t0 = time.perf_counter()
    gaps = find_quantile_gaps(client, conn, models, features, start, stop)
    out["detect_s"] = time.perf_counter() - t0
    for (model, version), hours in gaps.items():
        ranges = backfill_ranges(hours, max_hours, chunk_hours)
        done = sum(int((b - a) / HOUR) for a, b in ranges)
        out["gap_hours"] += len(hours)
        out["hours"] += done
        out["remaining"] += len(hours) - done
        for a, b in ranges:
            q = aggregate_quantiles(client, conn, sink, model, version, a, b, features, chunk_hours)
            for key in ("rows", "points", "requests", "read_s", "compute_s", "write_s"):
                out[key] += q[key]
    return out


def parse_models(text: str) -> list:
    """"a:1.0,b:2.1" -> [("a", "1.0"), ("b", "2.1")]"""
    pairs = [item.strip().rsplit(":", 1) for item in text.split(",") if item.strip()]
    if not pairs or any(len(p) != 2 or not all(p) for p in pairs):
        raise SystemExit("--models expects model:version pairs separated by commas")
    return [tuple(p) for p in pairs]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model")
    parser.add_argument("--version")
    parser.add_argument("--reference-start", help="ISO time, e.g. 2026-10-26T00:00:00Z (not needed with --backfill-only)")
    parser.add_argument("--reference-hours", type=float, default=48)
    parser.add_argument("--window-hours", type=float, default=6)
    parser.add_argument("--fairness-hours", type=float, default=24)
    parser.add_argument("--quantile-hours", type=int, default=24,
                        help="closed hours whose quantiles are recomputed at each run (late rows)")
    parser.add_argument("--quantile-features", default=",".join(TABULAR_NUMERIC),
                        help="numeric features whose hourly quantiles are written (comma-separated)")
    parser.add_argument("--backfill-days", type=float, default=30,
                        help="look-back of the backfill of the hourly quantiles, in days (0: no backfill)")
    parser.add_argument("--backfill-max-hours", type=int, default=168,
                        help="hours aggregated at most per run and model by the backfill, oldest first")
    parser.add_argument("--backfill-only", action="store_true",
                        help="aggregate every gap of the look-back at once, without the analyses, and exit")
    parser.add_argument("--models", help="with --backfill-only: model:version pairs, comma-separated (default --model)")
    parser.add_argument("--every", type=float, default=360, help="minutes between runs")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    features = [f.strip() for f in args.quantile_features.split(",") if f.strip()]
    if args.models and not args.backfill_only:
        parser.error("--models goes with --backfill-only")
    models = parse_models(args.models) if args.models else [(args.model, args.version)]
    if not all(all(m) for m in models):
        parser.error("--model and --version are required (or --models with --backfill-only)")
    if not args.backfill_only and not args.reference_start:
        parser.error("--reference-start is required (except with --backfill-only)")
    conn = connection()
    tags = {"model_id": args.model, "model_version": args.version}
    window = timedelta(hours=args.window_hours)
    look_back = timedelta(hours=round(args.backfill_days * 24))

    with InfluxDBClient(url=conn["url"], token=conn["token"], org=conn["org"], timeout=120_000) as client, \
            ResultsSink(conn["url"], conn["token"], conn["org"], conn["bucket"]) as sink:
        if args.backfill_only:
            stop = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            b = backfill_quantiles(client, conn, sink, models, features, stop - look_back, stop)
            print(f"backfill of {len(models)} model(s) since {(stop - look_back).isoformat()}: {b['gap_hours']} hours "
                  f"to aggregate found in {b['detect_s']:.2f} s, {b['hours']} aggregated ({b['points']} points; read "
                  f"{b['read_s']:.1f} s, compute {b['compute_s']:.1f} s, write {b['write_s']:.1f} s)", flush=True)
            return
        reference_start = datetime.fromisoformat(args.reference_start.replace("Z", "+00:00"))
        reference = read(client, conn, args.model, args.version, reference_start,
                         reference_start + timedelta(hours=args.reference_hours))
        if reference.empty:
            raise SystemExit("no capture in the reference period")
        size = max(30, int(len(reference) * args.window_hours / args.reference_hours))
        watch = DriftWatch(reference, TABULAR_NUMERIC, TABULAR_CATEGORICAL, window=size, model_id=args.model,
                           model_version=args.version, contiguous=False)
        labelled = reference.dropna(subset=["y_true"])
        reference_accuracy = float((labelled["y_true"] == labelled["y_pred"]).mean()) if len(labelled) else 1.0
        monitor = FairnessMonitor("language", "fr", ["en", "other"], tolerance=0.05)
        print(f"reference: {len(reference)} rows, window of about {size} rows, reference accuracy {reference_accuracy:.3f}")

        while True:
            now = datetime.now(timezone.utc)
            stop = now.replace(minute=0, second=0, microsecond=0)
            current = read(client, conn, args.model, args.version, stop - window, stop)
            if len(current) >= 30:
                results = watch.run(current)
                results += performance_results(current, reference_accuracy)
                recent = read(client, conn, args.model, args.version, stop - timedelta(hours=args.fairness_hours), stop)
                results += [r.model_dump() for r in fairness_results(recent)]
                results += [r.model_dump() for r in monitor.update(current)]
                n = sink.write_results(results, timestamp=stop, tags=tags)
                print(f"{stop.isoformat()}: {len(current)} rows analysed, {n} results written", flush=True)
            else:
                print(f"{stop.isoformat()}: {len(current)} rows, nothing to analyse", flush=True)
            if args.quantile_hours > 0:
                q = aggregate_quantiles(client, conn, sink, args.model, args.version,
                                        stop - timedelta(hours=args.quantile_hours), stop, features)
                print(f"{stop.isoformat()}: hourly quantiles of {q['hours']} hours written ({q['points']} points)", flush=True)
            if look_back > timedelta(0):  # older hours never aggregated (or with rows arrived since), oldest first
                b = backfill_quantiles(client, conn, sink, models, features, stop - look_back,
                                       stop - timedelta(hours=max(args.quantile_hours, 0)), args.backfill_max_hours)
                if b["gap_hours"]:
                    print(f"{stop.isoformat()}: quantile backfill: {b['hours']} of {b['gap_hours']} hours aggregated "
                          f"({b['points']} points), {b['remaining']} left for the next runs", flush=True)
            if args.once:
                break
            time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
