"""Part 1: train a NO2 model on 2016-2018, then replay 2019-2021 week by week through Pulse into InfluxDB.

- Model: gradient-boosted trees (scikit-learn) predicting the hourly NO2 of the Montréal network from the calendar and
  the weather, trained on 2016-2018.
- Reference for the monitoring: out-of-fold predictions of 2016-2018 (each year predicted by a model trained on the
  other two), so that the reference errors are honest out-of-sample errors.
- Replay: every week of 2019-2021 is captured as one batch, "as in production": features, prediction and, as the
  ground truth arrives with the measurement, the observed NO2. The observation time is kept in the `obs_ts` field
  (epoch seconds); the points carry the replay time, so the bucket can keep a finite retention.

    python tabular_replay.py --data data --env-file ../../validation_e2e/.env --campaign no2-20261104
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from pulsar_data_collection.models import CaptureOptions, DataWithPrediction, PulseParameters
from pulsar_data_collection.pulse import Pulse
from sklearn.ensemble import HistGradientBoostingRegressor

from influx_env import ensure_bucket, load_env
from no2_data import FEATURES, load

HERE = Path(__file__).parent
MODEL_ID, MODEL_VERSION = "no2_montreal", "1.0"


def model():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)


def out_of_fold(train: pd.DataFrame) -> pd.Series:
    """Prediction of every year of `train` by a model fitted on the other years"""
    years = train.index.year
    predictions = pd.Series(index=train.index, dtype=float)
    for year in sorted(set(years)):
        fit = train[years != year]
        predictions[years == year] = model().fit(fit[FEATURES], fit["no2"]).predict(train.loc[years == year, FEATURES])
    return predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(HERE / "data"))
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--retention-days", type=int, default=90)
    args = parser.parse_args(argv)
    data = Path(args.data)
    login = load_env(args.env_file)
    ensure_bucket(login, args.retention_days)

    frame = load(data)
    train = frame[frame.index.year <= 2018]
    reference = train.assign(prediction=out_of_fold(train))
    reference.to_csv(data / "no2_reference.csv.gz", index_label="obs_time")
    deployed = model().fit(train[FEATURES], train["no2"])
    production = frame[frame.index.year >= 2019]

    params = PulseParameters(
        model_id=MODEL_ID, model_version=MODEL_VERSION, data_id="replay",
        reference_data_storage=str(data / "no2_reference.csv.gz"), target_name="no2", storage_engine="influxdb",
        timestamp_column_name="_time", login=login, additional_tags={"campaign": args.campaign},
    )
    started = time.perf_counter()
    weeks = 0
    with Pulse(data=params, options=CaptureOptions(spool_dir=data / ".spool")) as pulse:
        for week_start, week in production.groupby(production.index.to_period("W-SUN").start_time):
            features = week[FEATURES].reset_index(drop=True)
            predictions = pd.DataFrame({
                "prediction": deployed.predict(week[FEATURES]).round(3),
                "no2": week["no2"].round(3).to_numpy(),  # ground truth, known with the measurement in this replay
                "obs_ts": (week.index.astype("int64") // 10**9).astype(float).to_numpy(),
            })
            pulse.capture_data(DataWithPrediction(data_points=features, predictions=predictions,
                                                  timestamp=datetime.now(timezone.utc), features_names=FEATURES))
            weeks += 1
        pulse.flush(120)
        stats = pulse.stats()
    summary = {
        "campaign": args.campaign,
        "model": f"HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05), scikit-learn {sklearn.__version__}",
        "train_rows": len(train), "train_period": [str(train.index.min()), str(train.index.max())],
        "replay_rows": len(production), "replay_weeks": weeks,
        "replay_period": [str(production.index.min()), str(production.index.max())],
        "reference_mae_out_of_fold": float(np.abs(reference["no2"] - reference["prediction"]).mean()),
        "pulse_stats": {k: stats[k] for k in ("captured_rows", "written_rows", "dropped_rows", "rejected_rows")},
        "seconds": round(time.perf_counter() - started, 1),
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"tabular_replay_{args.campaign}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0 if stats["written_rows"] == stats["captured_rows"] else 1


if __name__ == "__main__":
    sys.exit(main())
