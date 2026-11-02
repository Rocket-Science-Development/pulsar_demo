"""SC-14 - Schema evolution and type conflicts against the lab InfluxDB.

One model (measurement) per case, background mode with a spool directory (to verify rejections are never spooled nor
retried). Every row carries an int field `seq`; rows stored = points of `seq`. For each case: expected valid rows,
rows stored, rejected_rows counter, write_errors, spool files, and the reason recorded in last_error.
"""

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from lab.tests import _platform as C

PARAMS = {"batch_rows": 50, "mode": "background + spool_dir", "storage": "lab-db InfluxDB 2.6.1"}
CRITERIA = {"cases_counted_exactly": ("==", True), "spool_files_after": ("==", 0), "retry_writes_after_idle": ("==", 0)}
N = 50


def _batch(seq0, cols=20, f0=None, extra=None):
    rng = np.random.default_rng(seq0)
    x = pd.DataFrame(rng.normal(size=(N, cols)), columns=[f"f{i}" for i in range(cols)])
    if f0 is not None:
        x["f0"] = f0
    for k, v in (extra or {}).items():
        x[k] = v
    x["seq"] = np.arange(seq0, seq0 + N, dtype="int64")
    return x


def run(ctx):
    from pulsar_data_collection.models import CaptureOptions, DataWithPrediction

    env = C.influx_env()
    bucket = C.ensure_bucket(f"lab_sc14_{datetime.now(timezone.utc):%m%d%H%M%S}", env)
    ctx.params["bucket"] = bucket
    cases = {}

    def play(name, batches, together=False):
        """batches: list of (frame, valid: bool); together=True submits all before one flush (same write)"""
        p = C.make_pulse(f"sc14{name}", bucket, CaptureOptions(spool_dir=ctx.workdir / "spool" / name,
                                                              flush_interval_s=0.2 if not together else 2.0), env)
        for frame, _ in batches:
            p.capture_data(DataWithPrediction(timestamp=datetime.now(timezone.utc), predictions=None, data_points=frame,
                                              features_names=list(frame.columns)))
            if not together:
                p.flush(30)
        p.flush(60)
        errs_before = p.stats()["write_errors"]
        time.sleep(6)  # any retry loop would show up as new write errors
        st = p.stats()
        p.close()
        stored = C.count_rows(bucket, "seq", f"sc14{name}_", env)
        expected = sum(len(f) for f, ok in batches if ok)
        cases[name] = {"batches": len(batches), "rows": sum(len(f) for f, _ in batches), "expected_valid": expected,
                       "stored": stored, "written_rows": st["written_rows"], "rejected_rows": st["rejected_rows"],
                       "dropped_rows": st["dropped_rows"], "spooled_rows": st["spooled_rows"],
                       "write_errors": st["write_errors"], "retry_errors_after_idle": st["write_errors"] - errs_before,
                       "spool_files": st["spool_files"], "last_error": (st["last_error"] or "")[:300],
                       "counted_exactly": st["written_rows"] + st["rejected_rows"] == st["captured_rows"]
                       and stored == st["written_rows"]}
        return cases[name]

    s = iter(range(0, 10**9, N))
    # A. optional field present on every other batch
    play("optional", [(_batch(next(s), extra={"opt": 1.5} if k % 2 else None), True) for k in range(20)])
    # B. a float feature becomes a string for 5 batches, then float again
    play("float_to_str", [(_batch(next(s)), True) for _ in range(10)]
         + [(_batch(next(s), f0="high"), False) for _ in range(5)] + [(_batch(next(s)), True) for _ in range(10)])
    # C. an integer field receives one missing value: the column becomes float
    ints = lambda: np.arange(N, dtype="int64")  # noqa: E731
    with_nan = np.arange(N, dtype="float64")
    with_nan[7] = np.nan
    play("int_gets_nan", [(_batch(next(s), extra={"count": ints()}), True) for _ in range(10)]
         + [(_batch(next(s), extra={"count": with_nan}), False)] + [(_batch(next(s), extra={"count": ints()}), True) for _ in range(5)])
    # D. features added then removed
    play("add_remove", [(_batch(next(s), cols=20), True) for _ in range(5)] + [(_batch(next(s), cols=25), True) for _ in range(5)]
         + [(_batch(next(s), cols=10), True) for _ in range(5)])
    # E. a boolean becomes an integer
    play("bool_to_int", [(_batch(next(s), extra={"flag": np.ones(N, bool)}), True) for _ in range(5)]
         + [(_batch(next(s), extra={"flag": np.ones(N, "int64")}), False) for _ in range(3)])
    # F. a valid and a conflicting batch with the same columns land in the same write
    play("same_write", [(_batch(next(s)), True) for _ in range(3)])  # establishes f0 as float
    first = cases.pop("same_write")
    p_rows = [(_batch(next(s)), True), (_batch(next(s), f0="high"), False), (_batch(next(s)), True)]
    # reuse the same measurement: model id sc14same_write
    play("same_write", p_rows, together=True)
    cases["same_write"]["expected_valid"] += first["stored"]
    cases["same_write"]["note"] = "3 earlier batches establish f0 as float; then valid, conflicting, valid in one write"
    # G. mixed values inside one batch: 45 float rows, 5 rows with a string in f0
    play("mixed_rows", [(_batch(next(s)), True)])
    g_first = cases.pop("mixed_rows")
    mixed = _batch(next(s))
    mixed["f0"] = mixed["f0"].astype(object)
    mixed.loc[:4, "f0"] = "n/a"
    play("mixed_rows", [(mixed, False)])
    cases["mixed_rows"]["expected_valid"] = g_first["stored"] + 45
    cases["mixed_rows"]["note"] = "45 valid rows and 5 conflicting rows in one batch: InfluxDB partial write"

    # the stored counts of F and G include their first phase; written/rejected counters are of the second Pulse only
    for name, first_stored in (("same_write", first["stored"]), ("mixed_rows", g_first["stored"])):
        c = cases[name]
        c["stored_second_phase"] = c["stored"] - first_stored
        c["counted_exactly"] = c["stored_second_phase"] == c["written_rows"]
    (ctx.workdir / "cases.json").write_text(json.dumps(cases, indent=1))

    for name, c in cases.items():
        stored = c.get("stored_second_phase", c["stored"])
        if c["stored"] < c["expected_valid"]:
            ctx.limit(f"[{name}] valid rows lost: {c['expected_valid'] - c['stored']} of {c['expected_valid']} valid rows "
                      "not stored",
                      f"stored {c['stored']}, written_rows {c['written_rows']}, rejected_rows {c['rejected_rows']}; "
                      f"last_error: {c['last_error'][:160]}",
                      "Split a rejected write by batch (or by row on 422 partial writes) and retry the valid part; "
                      "cast columns to the stored field types before writing", "major")
        if stored != c["written_rows"]:
            ctx.limit(f"[{name}] counters disagree with the store: written_rows {c['written_rows']}, rejected_rows "
                      f"{c['rejected_rows']}, stored {stored}",
                      f"InfluxDB 422 partial write: the rows it accepted are counted as rejected; last_error: "
                      f"{c['last_error'][:160]}",
                      "Parse the 422 body (InfluxDB reports the rejected points) or rewrite row by row to count exactly",
                      "major")
        if c["retry_errors_after_idle"] or c["spool_files"]:
            ctx.limit(f"[{name}] rejected rows retried or spooled", json.dumps(c)[:300],
                      "Never spool StorageRejectedError", "major")
    exact = all(c["counted_exactly"] for c in cases.values())
    return {"cases_counted_exactly": exact, "spool_files_after": sum(c["spool_files"] for c in cases.values()),
            "retry_writes_after_idle": sum(c["retry_errors_after_idle"] for c in cases.values()),
            "rows_rejected_total": sum(c["rejected_rows"] for c in cases.values()),
            "valid_rows_lost_total": sum(max(0, c["expected_valid"] - c["stored"]) for c in cases.values()),
            "cases": {k: {kk: v[kk] for kk in ("expected_valid", "stored", "written_rows", "rejected_rows", "write_errors")}
                      for k, v in cases.items()}}
