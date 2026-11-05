"""Hourly quantiles written for the drift dashboard (analysis.hourly_quantiles, analysis.write_hourly_quantiles).

    cd observability && python -m unittest discover -s tests -v
"""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import QUANTILE_MEASUREMENT, hourly_quantiles, write_hourly_quantiles  # noqa: E402


def frame(hours=3, per_hour=50, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    times = [start + timedelta(hours=h, seconds=int(s)) for h in range(hours) for s in rng.integers(0, 3600, per_hour)]
    n = len(times)
    return pd.DataFrame({
        "pred_timestamp": times,
        "income": rng.lognormal(10, 0.4, n),
        "age": rng.integers(18, 80, n).astype(float),
        "region": rng.choice(["Montréal", "Québec"], n),
        "flag": rng.random(n) < 0.5,
    })


class FakeSink:
    def __init__(self):
        self.calls = []

    def write_stats_many(self, items, timestamp=None, measurement=None):
        items = list(items)
        self.calls.append((timestamp, measurement, items))
        return len(items)


class HourlyQuantiles(unittest.TestCase):
    def test_exact_quantiles_per_hour_stamped_at_the_end(self):
        data = frame()
        rows = hourly_quantiles(data, ["income", "age"])
        self.assertEqual(len(rows), 3 * 2)
        first_hour = data[data["pred_timestamp"] < pd.Timestamp("2026-11-01T01:00:00Z")]
        end, feature, fields = rows[0]
        self.assertEqual(end, datetime(2026, 11, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(feature, "income")
        expected = np.quantile(first_hour["income"], [0.1, 0.5, 0.9])
        self.assertAlmostEqual(fields["p10"], expected[0])
        self.assertAlmostEqual(fields["median"], expected[1])
        self.assertAlmostEqual(fields["p90"], expected[2])
        self.assertEqual(fields["n"], 50.0)
        self.assertEqual([r[0] for r in rows], sorted(r[0] for r in rows))

    def test_categorical_boolean_and_missing_columns_are_skipped(self):
        rows = hourly_quantiles(frame(), ["income", "region", "flag", "not_there"])
        self.assertEqual({r[1] for r in rows}, {"income"})
        self.assertEqual(hourly_quantiles(frame(), ["region"]), [])

    def test_missing_values_are_left_out_and_empty_hours_give_no_point(self):
        data = frame(hours=2)
        data.loc[data["pred_timestamp"] >= pd.Timestamp("2026-11-01T01:00:00Z"), "income"] = np.nan
        data.loc[0, "age"] = np.nan
        rows = hourly_quantiles(data, ["income", "age"])
        self.assertEqual([(r[1], r[2]["n"]) for r in rows], [("income", 50.0), ("age", 49.0), ("age", 50.0)])

    def test_same_values_as_numpy_per_feature_with_infinities_and_gaps(self):
        data = frame(hours=4).sample(frac=1, random_state=3)
        data.iloc[::5, data.columns.get_loc("income")] = np.inf
        data.iloc[::7, data.columns.get_loc("age")] = np.nan
        hours = np.array([t.replace(minute=0, second=0, microsecond=0) for t in data["pred_timestamp"]])
        for end, feature, fields in hourly_quantiles(data, ["income", "age"]):
            values = data.loc[hours == end - timedelta(hours=1), feature].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            self.assertEqual([fields["p10"], fields["median"], fields["p90"], fields["n"]],
                             [*np.quantile(values, [0.1, 0.5, 0.9]).tolist(), float(len(values))])

    def test_empty_frame(self):
        self.assertEqual(hourly_quantiles(pd.DataFrame(), ["income"]), [])

    def test_idempotent_and_late_rows_change_only_their_hour(self):
        data = frame()
        before = hourly_quantiles(data, ["income"])
        self.assertEqual(before, hourly_quantiles(data.sample(frac=1, random_state=1), ["income"]))
        late = pd.DataFrame({"pred_timestamp": [pd.Timestamp("2026-11-01T01:30:00Z")], "income": [1e9], "age": [30.0]})
        after = hourly_quantiles(pd.concat([data, late], ignore_index=True), ["income"])
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[2], after[2])
        self.assertEqual(after[1][2]["n"], 51.0)
        self.assertNotEqual(before[1][2]["p90"], after[1][2]["p90"])


class WriteHourlyQuantiles(unittest.TestCase):
    def test_one_request_per_hour_with_the_tags_of_the_model(self):
        sink = FakeSink()
        out = write_hourly_quantiles(sink, frame(), ["income", "age", "region"], {"model_id": "m", "model_version": "1.0"})
        self.assertEqual((out["points"], out["requests"], out["hours"]), (6, 3, 3))
        timestamp, measurement, items = sink.calls[0]
        self.assertEqual(measurement, QUANTILE_MEASUREMENT)
        self.assertEqual(timestamp, datetime(2026, 11, 1, 1, tzinfo=timezone.utc))
        self.assertEqual([tags for _, tags in items], [
            {"model_id": "m", "model_version": "1.0", "feature_name": "income"},
            {"model_id": "m", "model_version": "1.0", "feature_name": "age"}])
        self.assertEqual(set(items[0][0]), {"p10", "median", "p90", "n"})
        self.assertTrue(all(isinstance(v, float) for fields, _ in items for v in fields.values()))


if __name__ == "__main__":
    unittest.main()
