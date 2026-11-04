"""InfluxDB settings of the trial: environment variables, optionally read from an env file, and the trial bucket."""

import os
from pathlib import Path

from influxdb_client import BucketRetentionRules, InfluxDBClient

# Keys of a docker-compose env file of InfluxDB, mapped to the variables read by the scripts
ALIASES = {"DOCKER_INFLUXDB_INIT_ADMIN_TOKEN": "INFLUX_TOKEN", "DOCKER_INFLUXDB_INIT_ORG": "INFLUX_ORG"}


def load_env(path=None) -> dict:
    """INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG and INFLUX_BUCKET; values of `path` (KEY=VALUE lines) fill the gaps"""
    if path:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(ALIASES.get(key.strip(), key.strip()), value.strip())
    return {
        "url": os.environ.get("INFLUX_URL", "http://localhost:8087"),
        "token": os.environ["INFLUX_TOKEN"],
        "org": os.environ.get("INFLUX_ORG", "pulsarml"),
        "bucket_name": os.environ.get("INFLUX_BUCKET", "essai_gc"),
    }


def ensure_bucket(login: dict, retention_days: int = 90) -> None:
    """Create the bucket with a finite retention if it does not exist yet"""
    with InfluxDBClient(url=login["url"], token=login["token"], org=login["org"]) as client:
        api = client.buckets_api()
        if api.find_bucket_by_name(login["bucket_name"]) is None:
            rule = BucketRetentionRules(type="expire", every_seconds=retention_days * 86400)
            api.create_bucket(bucket_name=login["bucket_name"], retention_rules=rule, org=login["org"])
            print(f"bucket {login['bucket_name']} created (retention {retention_days} days)")


def query(login: dict, flux: str):
    import pandas as pd

    with InfluxDBClient(url=login["url"], token=login["token"], org=login["org"], timeout=300_000) as client:
        frame = client.query_api().query_data_frame(query=flux)
    return pd.concat(frame, ignore_index=True) if isinstance(frame, list) else frame
