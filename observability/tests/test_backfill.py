"""Backfill of the hourly quantiles (analysis.quantile_gaps, analysis.backfill_ranges, analysis_job.find_quantile_gaps,
analysis_job.backfill_quantiles).

    cd observability && python -m unittest discover -s tests -v

The integration test runs against a real InfluxDB 2 when PULSAR_TEST_INFLUX_URL and PULSAR_TEST_INFLUX_TOKEN are set
(PULSAR_TEST_INFLUX_ORG default pulsarml, PULSAR_TEST_INFLUX_BUCKET default monitoring); it writes a model of its own,
with a unique name, and deletes it at the end.
"""

import os
import sys
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis_job as job  # noqa: E402
from analysis import QUANTILE_MEASUREMENT, backfill_ranges, hourly_quantiles, quantile_gaps  # noqa: E402

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
H = timedelta(hours=1)


def at(hours: int) -> datetime:
    return T0 + hours * H


class QuantileGaps(unittest.TestCase):
    def test_missing_incomplete_and_complete_hours(self):
        raw = {at(1): 40, at(2): 40, at(3): 40, at(4): 40}
        agg = {at(2): 40.0, at(3): 20.0, at(4): 41.0}  # 3: a feature missing or late rows; 4: data deleted since
        self.assertEqual(quantile_gaps(raw, agg), [at(1), at(3)])

    def test_aggregates_without_captures_are_not_gaps(self):
        self.assertEqual(quantile_gaps({}, {at(1): 10.0}), [])

    def test_sorted_oldest_first(self):
        raw = {at(5): 1, at(1): 1, at(3): 1}
        self.assertEqual(quantile_gaps(raw, {}), [at(1), at(3), at(5)])


class BackfillRanges(unittest.TestCase):
    def test_consecutive_hours_merged(self):
        self.assertEqual(backfill_ranges([at(1), at(2), at(3), at(6)]), [(at(0), at(3)), (at(5), at(6))])

    def test_max_hours_takes_the_oldest(self):
        self.assertEqual(backfill_ranges([at(9), at(2), at(1), at(5)], max_hours=2), [(at(0), at(2))])
        self.assertEqual(backfill_ranges([at(1)], max_hours=0), [])

    def test_ranges_split_at_chunk_hours(self):
        ranges = backfill_ranges([at(h) for h in range(1, 61)], chunk_hours=24)
        self.assertEqual(ranges, [(at(0), at(24)), (at(24), at(48)), (at(48), at(60))])

    def test_empty(self):
        self.assertEqual(backfill_ranges([]), [])


class Record:
    def __init__(self, result, time, value, **tags):
        self.values = {"result": result, "_time": time, "_value": value, **tags}

    def get_time(self):
        return self.values["_time"]


class Column:
    def __init__(self, label, data_type):
        self.label, self.data_type = label, data_type


class Table:
    def __init__(self, records, value_type="long"):
        self.records = records
        self.columns = [Column("_measurement", "string"), Column("_field", "string"), Column("_value", value_type)]


class FakeClient:
    """Answers the gap query with the given tables and keeps the queries"""

    def __init__(self, tables=()):
        self.tables, self.queries = list(tables), []

    def query_api(self):
        return self

    def query(self, query):
        self.queries.append(query)
        return self.tables


CONN = {"bucket": "monitoring"}


class FindQuantileGaps(unittest.TestCase):
    def test_one_query_for_every_model_and_feature(self):
        client = FakeClient()
        models = [(f"m{i}", "1.0") for i in range(300)]
        gaps = job.find_quantile_gaps(client, CONN, models, ["income", "age"], at(0), at(24))
        self.assertEqual(len(client.queries), 1)
        self.assertEqual(gaps, {m: [] for m in models})
        query = client.queries[0]
        self.assertIn('r._measurement == "m299_1.0_input_data"', query)
        self.assertIn('(r.model_id == "m0" and r.model_version == "1.0")', query)
        self.assertIn('r._field == "age"', query)
        self.assertIn("range(start: 2026-11-01T00:00:00Z, stop: 2026-11-02T00:00:00Z)", query)
        # aggregates are stamped at the end of their hour: the hours of [start, stop) end in (start, stop]
        self.assertIn("range(start: 2026-11-01T01:00:00Z, stop: 2026-11-02T01:00:00Z)", query)

    def test_raw_counts_keyed_by_hour_end_and_compared_to_the_aggregates(self):
        tables = [
            Table([Record("raw", at(0), 20, _measurement="a_1.0_input_data"),
                   Record("raw", at(1), 20, _measurement="a_1.0_input_data"),
                   Record("raw", at(2), 20, _measurement="a_1.0_input_data"),
                   Record("raw", at(0), 5, _measurement="other_9_input_data")]),  # not asked: ignored
            Table([Record("raw", at(0), 8, _measurement="b_2_input_data")]),
            # aggregate of the hour [0, 1) is stamped at 1, the start of the window it falls in
            Table([Record("agg", at(1), 20.0, model_id="a", model_version="1.0"),
                   Record("agg", at(3), 20.0, model_id="a", model_version="1.0")]),
        ]
        gaps = job.find_quantile_gaps(FakeClient(tables), CONN, [("a", "1.0"), ("b", "2")], ["income"], at(0), at(3))
        self.assertEqual(gaps, {("a", "1.0"): [at(2)], ("b", "2"): [at(1)]})

    def test_disjunctions_stay_shallow_at_300_models(self):
        query = job.gaps_flux("monitoring", [(f"m{i}", "1.0") for i in range(300)], [f"f{j}" for j in range(20)],
                              at(0), at(1))
        depth = deepest = 0
        for char in query:
            depth += {"(": 1, ")": -1}.get(char, 0)
            deepest = max(deepest, depth)
        self.assertLess(deepest, 20)  # a flat chain of 300 `or` is refused by Flux: "Program is nested too deep"
        self.assertNotIn("contains(", query)  # not pushed down to storage in InfluxDB 2.6

    def test_quotes_are_escaped(self):
        query = job.gaps_flux("monitoring", [('we"ird', "1")], ["x"], at(0), at(1))
        self.assertIn('r.model_id == "we\\"ird"', query)


class BackfillQuantiles(unittest.TestCase):
    def test_bounded_per_run_oldest_first(self):
        calls = []
        gaps = {("a", "1.0"): [at(h) for h in (1, 2, 3, 10, 11)], ("b", "1.0"): []}
        original = job.find_quantile_gaps, job.aggregate_quantiles, job.numeric_features
        job.numeric_features = lambda client, conn, models, features, *a: ({m: features for m in models},
                                                                           {m: [] for m in models})
        job.find_quantile_gaps = lambda *args, **kw: gaps
        job.aggregate_quantiles = lambda client, conn, sink, m, v, a, b, features, chunk: (
            calls.append((m, a, b)) or {"rows": 1, "points": 2, "requests": 1, "read_s": 0, "compute_s": 0, "write_s": 0})
        try:
            out = job.backfill_quantiles(None, CONN, None, list(gaps), ["income"], at(0), at(24), max_hours=4)
        finally:
            job.find_quantile_gaps, job.aggregate_quantiles, job.numeric_features = original
        self.assertEqual(calls, [("a", at(0), at(3)), ("a", at(9), at(10))])
        self.assertEqual((out["gap_hours"], out["hours"], out["remaining"], out["points"]), (5, 4, 1, 4))

    def test_empty_window_queries_nothing(self):
        client = FakeClient()
        out = job.backfill_quantiles(client, CONN, None, [("a", "1")], ["x"], at(5), at(5))
        self.assertEqual((client.queries, out["hours"]), ([], 0))


def last_value(model, field, value, value_type):
    return Table([Record("_result", at(0), value, _measurement=f"{model}_1.0_input_data", _field=field)], value_type)


class NonNumericFeatures(unittest.TestCase):
    """Features stored as text or boolean are left out once per run, with one warning, instead of being seen as a gap
    (their raw values are counted, never aggregated) and re-aggregated at every run"""

    TYPES = [last_value("a", "income", 1.5, "double"), last_value("a", "age", 30, "long"),
             last_value("a", "region", "Québec", "string"), last_value("a", "flag", True, "boolean"),
             last_value("b", "income", 2.5, "double"), last_value("b", "age", "trente", "string"),
             last_value("b", "region", "Montréal", "string"), last_value("b", "flag", 1, "unsignedLong")]

    def test_split_by_stored_type_in_one_query(self):
        client = FakeClient(self.TYPES)
        numeric, excluded = job.numeric_features(client, CONN, [("a", "1.0"), ("b", "1.0"), ("c", "1.0")],
                                                 ["income", "age", "region", "flag"], at(0), at(24))
        self.assertEqual(len(client.queries), 1)
        self.assertIn("|> last()", client.queries[0])
        self.assertEqual(numeric, {("a", "1.0"): ["income", "age"], ("b", "1.0"): ["income", "flag"],
                                   ("c", "1.0"): ["income", "age", "region", "flag"]})  # c: no value, no gap
        self.assertEqual(excluded, {("a", "1.0"): ["region", "flag"], ("b", "1.0"): ["age", "region"],
                                    ("c", "1.0"): []})

    def test_a_field_numeric_in_one_shard_and_text_in_another_is_left_out(self):
        tables = [last_value("a", "age", 30, "long"), last_value("a", "age", "30", "string")]
        numeric, excluded = job.numeric_features(FakeClient(tables), CONN, [("a", "1.0")], ["age"], at(0), at(1))
        self.assertEqual((numeric, excluded), ({("a", "1.0"): []}, {("a", "1.0"): ["age"]}))

    def test_one_warning_naming_the_features_and_models(self):
        line = job.excluded_warning({("a", "1.0"): ["region", "flag"], ("b", "1.0"): ["region"]})
        self.assertEqual(len(line.splitlines()), 1)
        self.assertIn("region (a:1.0, b:1.0)", line)
        self.assertIn("flag (a:1.0)", line)
        many = job.excluded_warning({(f"m{i}", "1"): ["region"] for i in range(300)})
        self.assertIn("region (m0:1, m1:1, m2:1 and 297 more)", many)
        self.assertEqual(job.excluded_warning({("a", "1.0"): []}), "")

    def test_backfill_leaves_them_out_of_detection_and_aggregation(self):
        import contextlib
        import io

        detected, aggregated = [], []
        original = job.find_quantile_gaps, job.aggregate_quantiles
        job.find_quantile_gaps = lambda client, conn, models, features, start, stop: (
            detected.append((models, features)) or {m: [at(1)] for m in models})
        job.aggregate_quantiles = lambda client, conn, sink, m, v, a, b, features, chunk: (
            aggregated.append((m, features)) or {"rows": 1, "points": 1, "requests": 1, "read_s": 0, "compute_s": 0,
                                                 "write_s": 0})
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                out = job.backfill_quantiles(FakeClient(self.TYPES), CONN, None, [("a", "1.0"), ("b", "1.0")],
                                             ["income", "age", "region", "flag"], at(0), at(24))
        finally:
            job.find_quantile_gaps, job.aggregate_quantiles = original
        self.assertEqual(detected, [([("a", "1.0")], ["income", "age"]), ([("b", "1.0")], ["income", "flag"])])
        self.assertEqual(aggregated, [("a", ["income", "age"]), ("b", ["income", "flag"])])
        self.assertEqual(out["excluded"], {("a", "1.0"): ["region", "flag"], ("b", "1.0"): ["age", "region"]})
        lines = printed.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("warning: not stored as numbers"))

    def test_models_with_the_same_numeric_features_share_one_gap_query(self):
        tables = [last_value(f"m{i}", "region", "x", "string") for i in range(300)]
        detected = []
        original = job.find_quantile_gaps
        job.find_quantile_gaps = lambda client, conn, models, features, start, stop: (
            detected.append((len(models), features)) or {m: [] for m in models})
        try:
            with __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
                job.backfill_quantiles(FakeClient(tables), CONN, None, [(f"m{i}", "1.0") for i in range(300)],
                                       ["income", "region"], at(0), at(24))
        finally:
            job.find_quantile_gaps = original
        self.assertEqual(detected, [(300, ["income"])])


class ParseModels(unittest.TestCase):
    def test_pairs(self):
        self.assertEqual(job.parse_models("a:1.0, b_c:2:1"), [("a", "1.0"), ("b_c:2", "1")])
        with self.assertRaises(SystemExit):
            job.parse_models("a")


@unittest.skipUnless(os.environ.get("PULSAR_TEST_INFLUX_URL") and os.environ.get("PULSAR_TEST_INFLUX_TOKEN"),
                     "set PULSAR_TEST_INFLUX_URL and PULSAR_TEST_INFLUX_TOKEN to run against a real InfluxDB 2")
class BackfillAgainstInfluxDB(unittest.TestCase):
    """Captures of 6 closed hours and of the current hour, 2 hours aggregated beforehand: the backfill aggregates the
    4 others (2 per run with max_hours=2), never the current hour, writes nothing on a rerun, and re-aggregates an
    hour that receives a late row. The captures also hold a text field and a boolean field: asked as features, they
    are left out (they would otherwise be a gap at every run)."""

    FEATURES = ["income", "age"]

    @classmethod
    def setUpClass(cls):
        from influxdb_client import InfluxDBClient, Point
        from influxdb_client.client.write_api import SYNCHRONOUS
        from pulsar_data_collection.results import ResultsSink

        cls.conn = {"url": os.environ["PULSAR_TEST_INFLUX_URL"], "token": os.environ["PULSAR_TEST_INFLUX_TOKEN"],
                    "org": os.environ.get("PULSAR_TEST_INFLUX_ORG", "pulsarml"),
                    "bucket": os.environ.get("PULSAR_TEST_INFLUX_BUCKET", "monitoring")}
        cls.client = InfluxDBClient(url=cls.conn["url"], token=cls.conn["token"], org=cls.conn["org"], timeout=60_000)
        cls.writer = cls.client.write_api(write_options=SYNCHRONOUS)
        cls.sink = ResultsSink(cls.conn["url"], cls.conn["token"], cls.conn["org"], cls.conn["bucket"])
        cls.model, cls.version = f"qbf_test_{uuid.uuid4().hex[:8]}", "1.0"
        cls.Point = Point
        cls.stop = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        cls.start = cls.stop - 6 * H
        rng = np.random.default_rng(7)
        rows = []
        for h in range(7):  # 6 closed hours, then the current one (rows a few seconds after the hour starts)
            for i in range(12):
                rows.append({"pred_timestamp": cls.start + h * H + timedelta(seconds=1 + i * (299 if h < 6 else 1)),
                             "income": float(rng.lognormal(10, 0.4)), "age": float(rng.integers(18, 80)),
                             "region": str(rng.choice(["Montréal", "Québec"])), "flag": bool(rng.random() < 0.5)})
        cls.rows = pd.DataFrame(rows)
        cls.write_rows(cls.rows)

    @classmethod
    def write_rows(cls, frame):
        points = [cls.Point(job.captures(cls.model, cls.version)).tag("model_id", cls.model).time(r["pred_timestamp"])
                  .field("income", r["income"]).field("age", r["age"]).field("region", r["region"])
                  .field("flag", r["flag"]) for r in frame.to_dict("records")]
        cls.writer.write(bucket=cls.conn["bucket"], record=points)

    @classmethod
    def tearDownClass(cls):
        delete = cls.client.delete_api()
        for measurement, predicate in ((job.captures(cls.model, cls.version), ""),
                                       (QUANTILE_MEASUREMENT, f' AND model_id="{cls.model}"')):
            delete.delete(datetime(2000, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc) + 2 * H,
                          f'_measurement="{measurement}"{predicate}', bucket=cls.conn["bucket"], org=cls.conn["org"])
        cls.sink.close()
        cls.client.close()

    def stored(self) -> pd.DataFrame:
        frame = self.client.query_api().query_data_frame(f"""
        from(bucket: "{self.conn['bucket']}") |> range(start: {self.start.isoformat()}, stop: {(self.stop + 2 * H).isoformat()})
          |> filter(fn: (r) => r._measurement == "{QUANTILE_MEASUREMENT}" and r.model_id == "{self.model}")
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")""")
        frame = pd.concat(frame, ignore_index=True) if isinstance(frame, list) else frame
        return frame.sort_values(["_time", "feature_name"]).reset_index(drop=True) if len(frame) else frame

    def backfill(self, max_hours=None):
        return job.backfill_quantiles(self.client, self.conn, self.sink, [(self.model, self.version)], self.FEATURES,
                                      self.stop - 24 * H, self.stop, max_hours=max_hours)

    def test_backfill_is_bounded_idempotent_and_leaves_the_current_hour(self):
        models = [(self.model, self.version)]
        job.aggregate_quantiles(self.client, self.conn, self.sink, self.model, self.version, self.start + H,
                                self.start + 3 * H, self.FEATURES)
        gaps = job.find_quantile_gaps(self.client, self.conn, models, self.FEATURES, self.stop - 24 * H, self.stop)
        ends = [self.start + h * H for h in range(1, 7)]
        self.assertEqual(gaps[models[0]], [ends[0], ends[3], ends[4], ends[5]])

        first = self.backfill(max_hours=2)
        self.assertEqual((first["gap_hours"], first["hours"], first["remaining"], first["points"]), (4, 2, 2, 4))
        second = self.backfill(max_hours=2)
        self.assertEqual((second["gap_hours"], second["hours"], second["remaining"]), (2, 2, 0))
        again = self.backfill()
        self.assertEqual((again["gap_hours"], again["points"], again["rows"]), (0, 0, 0))

        asked = self.FEATURES + ["region", "flag"]
        unguarded = job.find_quantile_gaps(self.client, self.conn, models, asked, self.stop - 24 * H, self.stop)
        self.assertEqual(unguarded[models[0]], ends)  # their values are counted, never aggregated
        numeric, excluded = job.numeric_features(self.client, self.conn, models, asked, self.stop - 24 * H, self.stop)
        self.assertEqual((numeric[models[0]], excluded[models[0]]), (self.FEATURES, ["region", "flag"]))
        guarded = job.backfill_quantiles(self.client, self.conn, self.sink, models, asked, self.stop - 24 * H, self.stop)
        self.assertEqual((guarded["gap_hours"], guarded["points"], guarded["excluded"]),
                         (0, 0, {models[0]: ["region", "flag"]}))

        stored = self.stored()
        self.assertEqual(sorted(set(stored["_time"])), [pd.Timestamp(e) for e in ends])  # not the current hour
        closed = self.rows[self.rows["pred_timestamp"] < self.stop]
        expected = hourly_quantiles(closed, self.FEATURES)
        self.assertEqual(len(stored), len(expected))
        by_key = {(pd.Timestamp(r["_time"]), r["feature_name"]): r for r in stored.to_dict("records")}
        for end, feature, fields in expected:
            got = by_key[(pd.Timestamp(end), feature)]
            for name, value in fields.items():
                self.assertAlmostEqual(got[name], value, places=9)

        late = pd.DataFrame([{"pred_timestamp": self.start + 1.5 * H, "income": 1e9, "age": 30.0, "region": "Québec",
                              "flag": True}])
        self.write_rows(late)
        relate = self.backfill()
        self.assertEqual((relate["gap_hours"], relate["hours"]), (1, 1))
        n = self.stored().query("feature_name == 'income'").set_index("_time")["n"]
        self.assertEqual(n[pd.Timestamp(ends[1])], 13.0)


if __name__ == "__main__":
    unittest.main()
