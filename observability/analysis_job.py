"""Periodic analysis: every `--every` minutes, analyse the last closed window of a model's captures and write the results.

The reference is read once at start (a period known to be sound). Each run reads the window that just closed from
InfluxDB, then writes calibrated drift with confirmation, performance (when labels have arrived), fairness on a rolling
window and the anytime-valid monitor to `pulsar_results`, where the dashboards read them.

    python analysis_job.py --model admissibilite --version 1.0 --reference-start 2026-10-26T00:00:00Z \\
        --reference-hours 48 --window-hours 6 --every 360
    python analysis_job.py ... --once      # analyse the last closed window and exit

Environment: INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG (default pulsarml), INFLUX_BUCKET (default monitoring).
"""

import argparse
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from influxdb_client import InfluxDBClient
from pulsar_data_collection.results import ResultsSink
from pulsar_metrics.fairness import FairnessMonitor

from analysis import TABULAR_CATEGORICAL, TABULAR_NUMERIC, DriftWatch, fairness_results, performance_results
from seed_demo import connection


def read(client, conn, model: str, version: str, start: datetime, stop: datetime) -> pd.DataFrame:
    query = f"""
    from(bucket: "{conn['bucket']}")
      |> range(start: {start.isoformat()}, stop: {stop.isoformat()})
      |> filter(fn: (r) => r["_measurement"] == "{model}_{version}_input_data")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--reference-start", required=True, help="ISO time, e.g. 2026-10-26T00:00:00Z")
    parser.add_argument("--reference-hours", type=float, default=48)
    parser.add_argument("--window-hours", type=float, default=6)
    parser.add_argument("--fairness-hours", type=float, default=24)
    parser.add_argument("--every", type=float, default=360, help="minutes between runs")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    conn = connection()
    tags = {"model_id": args.model, "model_version": args.version}
    window = timedelta(hours=args.window_hours)

    with InfluxDBClient(url=conn["url"], token=conn["token"], org=conn["org"], timeout=120_000) as client, \
            ResultsSink(conn["url"], conn["token"], conn["org"], conn["bucket"]) as sink:
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
            if args.once:
                break
            time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
