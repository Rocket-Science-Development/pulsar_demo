"""Hourly NO2 of the Montréal monitoring network (NAPS) with the weather of Montréal-Trudeau airport (ECCC).

The target is the mean of eight NAPS stations of Montréal that report over the whole of 2016 to 2021 (at least five
reporting in the hour); the features are the hour, the day of the week, the day of the year and the hourly weather.
Only aggregated measurements: no person is involved.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd

STATIONS = [50103, 50109, 50128, 50129, 50134, 50135, 50136, 50138]
YEARS = range(2016, 2022)
WEATHER = {
    "Temp (°C)": "temp",
    "Dew Point Temp (°C)": "dew",
    "Rel Hum (%)": "rh",
    "Wind Dir (10s deg)": "wdir",
    "Wind Spd (km/h)": "wspd",
    "Stn Press (kPa)": "press",
}
FEATURES = ["hour", "dow", "doy_s", "doy_c", "temp", "dew", "rh", "wspd", "wd_s", "wd_c", "press"]
# Features whose distribution is tested for drift (the calendar features are the same every week by construction)
MONITORED = ["temp", "dew", "rh", "wspd", "wd_s", "wd_c", "press"]


def read_naps(path: Path) -> pd.DataFrame:
    """One NAPS yearly file (wide: one row per station and day, 24 hourly columns) to long hourly rows"""
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    start = next(k for k, line in enumerate(lines) if line.startswith("Pollutant//"))
    wide = pd.read_csv(io.StringIO("\n".join(lines[start:])), header=0)
    wide.columns = ["pollutant", "naps_id", "city", "province", "lat", "lon", "date"] + [f"H{h:02d}" for h in range(1, 25)]
    wide = wide[wide["naps_id"].isin(STATIONS)]
    long = wide.melt(id_vars=["naps_id", "date"], value_vars=[f"H{h:02d}" for h in range(1, 25)], var_name="h",
                     value_name="no2")
    long["no2"] = pd.to_numeric(long["no2"], errors="coerce").where(lambda s: s >= 0)  # -999: no data
    # "hour ending" local standard time: H01 covers 00:00-01:00, stamped at its start like the weather rows
    long["time"] = pd.to_datetime(long["date"]) + pd.to_timedelta(long["h"].str[1:].astype(int) - 1, unit="h")
    return long[["naps_id", "time", "no2"]]


def read_weather(data: Path) -> pd.DataFrame:
    frames = [pd.read_csv(data / f"yul_{y}_{m:02d}.csv", encoding="utf-8-sig") for y in YEARS for m in range(1, 13)]
    weather = pd.concat(frames, ignore_index=True)
    weather["time"] = pd.to_datetime(weather["Date/Time (LST)"])
    return weather.set_index("time")[list(WEATHER)].rename(columns=WEATHER)


def load(data: Path) -> pd.DataFrame:
    """Hourly frame indexed by local standard time: features, `no2` (target) and `stations` (reporting stations)"""
    data = Path(data)
    no2 = pd.concat([read_naps(data / f"NO2_{y}.csv") for y in YEARS], ignore_index=True)
    wide = no2.pivot_table(index="time", columns="naps_id", values="no2")
    target = pd.DataFrame({"no2": wide.mean(axis=1), "stations": wide.notna().sum(axis=1)})
    target = target[target["stations"] >= 5]
    frame = read_weather(data).join(target, how="inner")
    frame["hour"] = frame.index.hour
    frame["dow"] = frame.index.dayofweek
    day = frame.index.dayofyear
    frame["doy_s"], frame["doy_c"] = np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25)
    frame["wd_s"], frame["wd_c"] = np.sin(np.deg2rad(frame["wdir"] * 10)), np.cos(np.deg2rad(frame["wdir"] * 10))
    return frame.drop(columns=["wdir"]).sort_index()
