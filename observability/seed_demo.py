"""Fill InfluxDB with a week of demonstration data and its analyses, so that every dashboard has something to show.

Everything is synthetic or comes from the demo agent; no real person.

1. Tabular model "admissibilite" (eligibility to an assistance program): a gradient-boosting model scores synthetic
   applicants every hour for 7 days, captured with Pulse (non-blocking, personal information masked in the free-text
   comment). Injected on purpose: income drops by 15 % from day 4 (data drift), and from day 5 a rule withdraws 40 %
   of the approvals of applicants whose language is "other" (bias). Labels arrive after 48 hours.
2. Hourly p10 / median / p90 of the numeric features (`pulsar_feature_quantiles`, for the drift dashboard's band).
   Analyses every 6 hours, written with ResultsSink: calibrated drift with confirmation, performance, fairness
   (fixed on a rolling 24-hour window, and anytime-valid), explanation drift (TreeSHAP), importance by group. Pulse's counters every hour.
3. Agent: drift analysis of the agent runs of a campaign already in the bucket (e.g. copied from the S2 campaign),
   scripted conversations that exercise personal-information detection and leaks, and the counterfactual probe rows
   of probe_agent.py if given.

    python seed_demo.py --agent-campaign s2-gpu-v2-20261101 --probe-rows probe_v1_rows.csv probe_biased_rows.csv

Environment: INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG (default pulsarml), INFLUX_BUCKET (default monitoring).
"""

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
from pulsar_data_collection.models import CaptureOptions, DataWithPrediction, PulseParameters
from pulsar_data_collection.privacy import PiiPolicy
from pulsar_data_collection.pulse import Pulse
from pulsar_data_collection.results import ResultsSink
from pulsar_metrics.explain import AttributionDrift, tree_attributions
from pulsar_metrics.fairness import FairnessMonitor, comparative_analysis, counterfactual_analysis
from sklearn.ensemble import GradientBoostingClassifier

from analysis import (
    AGENT_CATEGORICAL,
    AGENT_NUMERIC,
    TABULAR_CATEGORICAL,
    TABULAR_NUMERIC,
    DriftWatch,
    agent_windows,
    explain_results,
    fairness_results,
    performance_results,
    to_numeric,
    windows,
    write_hourly_quantiles,
)

MODEL_ID, MODEL_VERSION = "admissibilite", "1.0"
REGIONS = {"Montréal": 0.45, "Québec": 0.25, "Est-du-Québec": 0.15, "Ottawa-Gatineau": 0.15}
LANGUAGES = {"fr": 0.70, "en": 0.20, "other": 0.10}
COMMENTS = [
    "Demande de renouvellement.", "Première demande.", "Changement de situation familiale.", "Déménagement récent.",
    "Perte d'emploi en octobre.", "Retour aux études.", "Aucune remarque.",
]
PII_COMMENTS = [  # fictitious values; masked by Pulse before writing
    "Rappeler au 514-555-0199 après 17 h.", "Courriel : client.demo@exemple.ca", "NAS fourni : 130 692 544",
    "Nouvelle adresse, code postal H2X 1Y4.",
]
FEATURES = ["income", "age", "household_size", "region_code"]


def connection():
    return {
        "url": os.environ.get("INFLUX_URL", "http://localhost:8086"),
        "token": os.environ["INFLUX_TOKEN"],
        "org": os.environ.get("INFLUX_ORG", "pulsarml"),
        "bucket": os.environ.get("INFLUX_BUCKET", "monitoring"),
    }


def applicants(n: int, rng, income_factor: float = 1.0) -> pd.DataFrame:
    frame = pd.DataFrame({
        "income": np.round(rng.lognormal(10.6, 0.45, n) * income_factor, 0),
        "age": rng.integers(18, 80, n).astype(float),
        "household_size": rng.choice([1, 2, 3, 4, 5], n, p=[0.3, 0.3, 0.2, 0.15, 0.05]).astype(float),
        "region": rng.choice(list(REGIONS), n, p=list(REGIONS.values())),
        "language": rng.choice(list(LANGUAGES), n, p=list(LANGUAGES.values())),
    })
    frame["region_code"] = frame["region"].map({r: i for i, r in enumerate(REGIONS)}).astype(float)
    limit = 28000 + 9000 * frame["household_size"]
    frame["y_true"] = ((frame["income"] < limit) ^ (rng.random(n) < 0.08)).astype(int)
    pii = rng.random(n) < 0.04
    frame["comment"] = np.where(pii, rng.choice(PII_COMMENTS, n), rng.choice(COMMENTS, n))
    return frame


def seed_tabular(conn, sink, days: int, batch: int, seed: int):
    rng = np.random.default_rng(seed)
    model = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=0)
    training = applicants(6000, rng)
    model.fit(training[FEATURES], training["y_true"])

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    drift_at, bias_at, labels_until = start + timedelta(days=3), start + timedelta(days=4), end - timedelta(hours=48)
    params = PulseParameters(
        model_id=MODEL_ID, model_version=MODEL_VERSION, data_id="demo", reference_data_storage="synthetic://admissibilite",
        target_name="y_true", storage_engine="influxdb", timestamp_column_name="_time",
        login={"url": conn["url"], "token": conn["token"], "org": conn["org"], "bucket_name": conn["bucket"]},
        additional_tags={"environment": "demo"},
    )
    pulse = Pulse(params, CaptureOptions(pii=PiiPolicy(columns=["comment"])))
    frames = []
    hours = days * 24
    for h in range(hours):
        ts = start + timedelta(hours=h)
        data = applicants(batch, rng, income_factor=0.85 if ts >= drift_at else 1.0)
        data["y_pred_proba"] = model.predict_proba(data[FEATURES])[:, 1]
        data["y_pred"] = (data["y_pred_proba"] >= 0.5).astype(int)
        if ts >= bias_at:  # a production rule withdraws approvals of one group
            withdraw = (data["language"] == "other") & (data["y_pred"] == 1) & (rng.random(batch) < 0.4)
            data.loc[withdraw, "y_pred"] = 0
        observed = data.copy()
        if ts >= labels_until:
            observed["y_true"] = np.nan  # labels arrive 48 hours later
        pulse.capture_data(DataWithPrediction(
            data_points=observed[["income", "age", "household_size", "region", "language", "comment", "y_true"]],
            predictions=observed[["y_pred", "y_pred_proba"]],
            timestamp=ts,
            features_names=["income", "age", "household_size", "region", "language", "comment"],
        ))
        observed["pred_timestamp"] = pd.date_range(ts, periods=batch, freq="ms")
        frames.append(observed)
        pulse.flush(30)  # the week is replayed in seconds: wait for each hour to be written, as in real time
        sink.write_stats(pulse.stats(), timestamp=ts + timedelta(minutes=59), tags={"model_id": MODEL_ID,
                                                                                     "model_version": MODEL_VERSION})
    pulse.close()
    history = pd.concat(frames, ignore_index=True).assign(model_id=MODEL_ID, model_version=MODEL_VERSION)
    print(f"tabular: {len(history)} rows captured, stats {pulse.stats()}")

    tags = {"model_id": MODEL_ID, "model_version": MODEL_VERSION}
    # hourly quantiles of the numeric features, read by the drift dashboard instead of the raw captures (lab SC-07 L1)
    quantiles = write_hourly_quantiles(sink, history, TABULAR_NUMERIC, tags)
    print(f"tabular: hourly quantiles of {quantiles['hours']} hours written ({quantiles['points']} points)")
    reference = history[history["pred_timestamp"] < start + timedelta(days=2)]
    # the synthetic rows are independent draws: A/A windows of random rows (a 2-day reference holds only 8 contiguous
    # 6-hour windows, too few for a contiguous calibration)
    watch = DriftWatch(reference, TABULAR_NUMERIC, TABULAR_CATEGORICAL, window=6 * batch, model_id=MODEL_ID,
                       model_version=MODEL_VERSION, contiguous=False)
    reference_accuracy = float((reference["y_true"] == reference["y_pred"]).mean())
    monitor = FairnessMonitor("language", "fr", ["en", "other"], tolerance=0.05)
    sample = reference.sample(6 * batch, random_state=0)
    explainer = AttributionDrift(k=3, n_bootstrap=100).fit(tree_attributions(model, sample[FEATURES]))
    written = 0
    analysed = history[history["pred_timestamp"] >= start + timedelta(days=2)]
    for ts, current in windows(analysed, "6h", origin=pd.Timestamp(start + timedelta(days=2))):
        if len(current) < 6 * batch:  # incomplete last window: analysed when it closes
            continue
        results = watch.run(current)
        results += performance_results(current, reference_accuracy)
        # fairness on a rolling 24-hour window: small groups need more rows than a 6-hour window holds
        recent = history[(history["pred_timestamp"] > ts - pd.Timedelta(hours=24)) & (history["pred_timestamp"] <= ts)]
        results += [r.model_dump() for r in fairness_results(recent)]
        results += [r.model_dump() for r in monitor.update(current)]
        results += explain_results(explainer, tree_attributions(model, current[FEATURES]), current["language"].rename("language"))
        written += sink.write_results(results, timestamp=ts.to_pydatetime(), tags=tags)
    print(f"tabular: {written} results written")


def seed_agent(conn, sink, campaign: str):
    query = f"""
    from(bucket: "{conn['bucket']}") |> range(start: -30d, stop: 1m)
      |> filter(fn: (r) => r["_measurement"] == "permit_assistant_1.0_input_data")
      |> filter(fn: (r) => r["record_type"] == "agent_run" and r["campaign"] == "{campaign}")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    """
    with InfluxDBClient(url=conn["url"], token=conn["token"], org=conn["org"], timeout=120_000) as client:
        frame = client.query_api().query_data_frame(query=query)
    if isinstance(frame, list):
        frame = pd.concat(frame, ignore_index=True)
    if frame.empty:
        print(f"agent: no run for campaign {campaign}")
        return
    runs = to_numeric(frame.rename(columns={"_time": "pred_timestamp"}).sort_values("pred_timestamp"), AGENT_NUMERIC)
    for column in AGENT_CATEGORICAL:
        runs[column] = runs.get(column, "none").astype(str)
    written = 0
    for ts, phase, results in agent_windows(runs.reset_index(drop=True)):
        written += sink.write_results(results, timestamp=ts.to_pydatetime(), tags={
            "model_id": "permit_assistant", "model_version": "1.0", "campaign": campaign, "phase": phase})
    print(f"agent: {written} drift results written for {campaign}")


def seed_agent_pii(conn, conversations: int, seed: int):
    """Scripted conversations (no model): the user gives an e-mail; a tool returns another client's phone; some
    answers repeat that phone (leak), the others only echo the user's e-mail"""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from pulsar_data_collection.genai import PulsarSpanExporter, PulseSink, agent_span, llm_span, tool_span
    from pulsar_data_collection.privacy import PiiSession

    params = PulseParameters(
        model_id="permit_assistant", model_version="1.0", data_id="demo", reference_data_storage="scripted",
        target_name="task_success", storage_engine="influxdb", timestamp_column_name="_time",
        login={"url": conn["url"], "token": conn["token"], "org": conn["org"], "bucket_name": conn["bucket"]},
        additional_tags={"scenario": "scripted_pii", "campaign": "pii-demo"},
    )
    sink = PulseSink(params, CaptureOptions())
    provider = TracerProvider(resource=Resource.create({"service.name": "permit-assistant"}))
    provider.add_span_processor(BatchSpanProcessor(PulsarSpanExporter(sink)))
    tracer = provider.get_tracer("pii-demo")
    rng = np.random.default_rng(seed)
    for k in range(conversations):
        email = f"usager{k}@exemple.ca"
        with PiiSession() as session, agent_span(tracer, "permit_assistant", prompt_version="v1") as run:
            session.record(run, "input", f"Bonjour, écrivez-moi à {email} pour mon dossier d'aide au loyer.")
            with tool_span(tracer, "lookup_client", arguments={"id": k}) as tool:
                session.record(tool, "context", f"Dossier voisin : joindre au 418 555 {1000 + k:04d}")
            leak = rng.random() < 0.15
            answer = (f"Je vous confirme par courriel à {email}. Votre voisin est joignable au 418 555 {1000 + k:04d}."
                      if leak else f"Je vous confirme la réception par courriel à {email}.")
            with llm_span(tracer, model="scripted", provider="none", temperature=0.0) as call:
                session.record(call, "output", answer)
            run.set_attribute("pulsar.agent.termination", "completed")
        time.sleep(0.01)
    provider.shutdown()
    print(f"agent: {conversations} scripted conversations with personal information, stats {sink.stats()}")


def seed_probe(sink, paths: list):
    for path in paths:
        rows = pd.read_csv(path)
        rows = rows.astype({"error": object}).where(rows.notna(), None)
        results = counterfactual_analysis(rows, "fr_qc", action="authenticate")
        comparative = Path(str(path).replace("_rows.csv", "_comparative_rows.csv"))
        if comparative.exists():
            cmp_rows = pd.read_csv(comparative)
            cmp_rows = cmp_rows.astype({"error": object}).where(cmp_rows.notna(), None)
            results += comparative_analysis(cmp_rows)
        scenario = "biased_prompt" if "biais" in Path(path).name or "biased" in Path(path).name else "normal_prompt"
        n = sink.write_results(results, tags={"model_id": "permit_assistant", "model_version": "1.0",
                                              "scenario": scenario, "campaign": "probe-e4"})
        print(f"probe: {n} results written ({scenario})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--batch", type=int, default=150)
    parser.add_argument("--agent-campaign", default=None)
    parser.add_argument("--pii-conversations", type=int, default=60)
    parser.add_argument("--probe-rows", nargs="*", default=[])
    parser.add_argument("--skip-tabular", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    conn = connection()
    with ResultsSink(conn["url"], conn["token"], conn["org"], conn["bucket"]) as sink:
        if not args.skip_tabular:
            seed_tabular(conn, sink, args.days, args.batch, args.seed)
        if args.agent_campaign:
            seed_agent(conn, sink, args.agent_campaign)
        if args.pii_conversations:
            seed_agent_pii(conn, args.pii_conversations, args.seed)
        if args.probe_rows:
            seed_probe(sink, args.probe_rows)


if __name__ == "__main__":
    main()
