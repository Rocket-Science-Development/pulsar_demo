"""Helpers of the "ml-streams" P2 tests (TS-04, TC-02, TC-04): per-window verdicts through the public Pulsar.ml API.

Built on _streams.py (Analyzer + AACalibrator + Benjamini-Hochberg + DriftPersistence); nothing here re-implements a
Pulsar.ml statistic, it only reads the result frames and counts outcomes.
"""

import os
import time

import numpy as np
import pandas as pd

from lab.tests import _streams as S

# keep the PC usable: numeric libraries limited to 8 threads (one test at a time, one process)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "8")


def truthy(series: pd.Series) -> pd.Series:
    """True where a verdict column (object dtype, True / False / None / NaN) holds True"""
    return series.map(lambda v: v is True or (isinstance(v, (bool, np.bool_)) and bool(v))).astype(bool)


class WindowStream:
    """Feeds windows to one analyzer and keeps per-window verdicts (BH, persistence-confirmed, performance)"""

    def __init__(self, analyzer, reference, cal=None, options=None, persistence=2, perf_persistence=2):
        self.analyzer, self.reference, self.cal = analyzer, reference.reset_index(drop=True), cal
        self.options = options if options is not None else (cal.options() if cal is not None else {})
        self.fp = S.DriftPersistence(persistence)
        self.pp = S.DriftPersistence(perf_persistence)
        self.rows, self.seconds = [], []

    def step(self, current, label=None, **extra):
        t0 = time.perf_counter()
        res = S.analyze(self.analyzer, current.reset_index(drop=True), self.reference, cal=self.cal, options=self.options)
        self.seconds.append(time.perf_counter() - t0)
        drift = res[res.metric_type == "drift"]
        perf = res[res.metric_type == "performance"]
        col = "drift_fdr" if self.cal is not None else "drift_status"
        fv = self.fp.update_to_pandas(S.persistence_frame(drift, col))
        flagged = drift[truthy(drift[col])] if col in drift else drift.iloc[:0]
        row = {"label": label, "n": len(current),
               "bh_features": sorted(set(flagged.feature_name)),
               "bh_pairs": sorted(f"{m}|{f}" for m, f in zip(flagged.metric_name, flagged.feature_name)),
               "confirmed": sorted(set(fv[fv.confirmed_drift].feature_name)) if not fv.empty else [],
               "errors": int(drift.error.notna().sum()) if "error" in drift else 0,
               "no_verdict": int(drift.drift_status.isna().sum()),
               **extra}
        if not perf.empty:
            pv = self.pp.update_to_pandas(perf)
            for _, p in perf.iterrows():
                row[f"perf_{p.metric_name}"] = None if pd.isna(p.metric_value) else float(p.metric_value)
                row[f"perf_{p.metric_name}_raw"] = None if p.drift_status is None or pd.isna(p.drift_status) else bool(p.drift_status)
                row[f"perf_{p.metric_name}_error"] = p.get("error") if isinstance(p.get("error"), str) else None
                if "label_coverage" in p:
                    row["label_coverage"] = None if pd.isna(p.label_coverage) else float(p.label_coverage)
            row["perf_confirmed"] = bool(pv.confirmed_drift.any()) if not pv.empty else False
        self.rows.append(row)
        return row, res


def share(rows, key_fn):
    return float(np.mean([bool(key_fn(r)) for r in rows])) if rows else None


def first(rows, key_fn):
    for i, r in enumerate(rows):
        if key_fn(r):
            return i
    return None
