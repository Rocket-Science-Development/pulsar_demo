"""Download the Government of Canada open data used by the trial, and record where each file comes from.

| Source | Licence | Files |
|---|---|---|
| NAPS, National Air Pollution Surveillance program (ECCC), hourly NO2 | Open Government Licence - Canada | `NO2_<year>.csv`, 2016 to 2021 |
| Historical climate data (ECCC), hourly, station MONTREAL INTL A (climate ID 7025251, station ID 51157) | ECCC Data Servers End-use Licence (commercial use allowed) | `yul_<year>_<month>.csv` |
| Government of Canada AI Register, minimum viable product (TBS) | Open Government Licence - Canada | `registre_ia.csv` |

Every file is written to `--data` (kept out of git) and listed in `sources.json` with its URL, download date, size and
SHA-256. A file already present is not downloaded again; its checksum is compared with the manifest instead.

    python download.py --data data
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).parent
NAPS = "https://data-donnees.az.ec.gc.ca/api/file?path=" + urllib.parse.quote(
    "/air/monitor/national-air-pollution-surveillance-naps-program/Data-Donnees/{year}/ContinuousData-DonneesContinu/"
    "HourlyData-DonneesHoraires/NO2_{year}.csv",
    safe="{}",
)
CLIMATE = (
    "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=51157&Year={year}&Month={month}"
    "&Day=1&timeframe=1&submit=Download+Data"
)
REGISTER = (
    "https://open.canada.ca/data/dataset/fcbc0200-79ba-4fa4-94a6-00e32facea6b/resource/"
    "369f6f34-148a-42ed-b581-8c164e941a89/download/gc-ai-register-mvp-registre-de-lia-du-gc-pmv-04-26.csv"
)
LICENCES = {
    "naps": ("Open Government Licence - Canada", "https://open.canada.ca/data/dataset/1b36a356-defd-4813-acea-47bc3abd859b"),
    "climate": (
        "Environment and Climate Change Canada Data Servers End-use Licence",
        "https://eccc-msc.github.io/open-data/licence/readme_en/",
    ),
    "register": ("Open Government Licence - Canada", "https://open.canada.ca/data/dataset/fcbc0200-79ba-4fa4-94a6-00e32facea6b"),
}
YEARS = range(2016, 2022)


def files():
    for year in YEARS:
        yield "naps", f"NO2_{year}.csv", NAPS.format(year=year)
    for year in YEARS:
        for month in range(1, 13):
            yield "climate", f"yul_{year}_{month:02d}.csv", CLIMATE.format(year=year, month=month)
    yield "register", "registre_ia.csv", REGISTER


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(HERE / "data"))
    parser.add_argument("--manifest", default=str(HERE / "sources.json"))
    parser.add_argument("--only", choices=sorted(LICENCES), nargs="*", help="download only these sources")
    args = parser.parse_args(argv)
    data, manifest_path = Path(args.data), Path(args.manifest)
    data.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"files": {}}
    session = requests.Session()
    mismatches = 0
    for source, name, url in files():
        if args.only and source not in args.only:
            continue
        path = data / name
        known = manifest["files"].get(name)
        if path.exists() and known:
            if sha256(path) != known["sha256"]:
                print(f"checksum mismatch: {name}", flush=True)
                mismatches += 1
            continue
        response = session.get(url, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
        licence, licence_url = LICENCES[source]
        manifest["files"][name] = {
            "source": source,
            "url": url,
            "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "licence": licence,
            "licence_url": licence_url,
        }
        print(f"{name}: {path.stat().st_size:,} bytes", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
        time.sleep(0.2)  # be gentle with the public servers
    print(f"{len(manifest['files'])} files in {manifest_path}; {mismatches} checksum mismatch(es)")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
