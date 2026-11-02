"""Check the lab registry: no field lost to YAML parsing, and the JSON copies match the YAML sources.

An unquoted comma inside a YAML flow mapping (`dataset: {name: a (b, c), ...}`) silently splits the value into extra
keys with null values. This check fails on any null value in the YAML registries, and on any `dataset` field of
`registry.json` or `approved_tonight.json` that differs from its YAML source.

    python -m lab.check_registry          # report problems, exit 1 if any
    python -m lab.check_registry --fix    # rewrite the dataset fields of the JSON copies from the YAML sources
"""

import json
import sys
from pathlib import Path

import yaml

LAB = Path(__file__).parent
SOURCES = ["registry_ml.yaml", "registry_llm.yaml", "registry_platform.yaml"]
# file -> json.dumps options that keep its current layout
COPIES = {"registry.json": {"indent": 1}, "approved_tonight.json": {}}


def nulls(value, path=""):
    if isinstance(value, dict):
        for k, v in value.items():
            if v is None:
                yield f"{path}/{k}"
            yield from nulls(v, f"{path}/{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from nulls(v, f"{path}/{v.get('id', i) if isinstance(v, dict) else i}")


def load_sources() -> tuple[dict, list]:
    tests, problems = {}, []
    for name in SOURCES:
        data = yaml.safe_load((LAB / name).read_text(encoding="utf-8"))
        problems += [f"{name}: null value at {p} (unquoted comma in a flow mapping?)" for p in nulls(data.get("tests", []))]
        tests.update({t["id"]: t for t in data.get("tests", [])})
    return tests, problems


def main(fix: bool = False) -> int:
    sources, problems = load_sources()
    for name, dump in COPIES.items():
        path = LAB / name
        if not path.exists():
            continue
        copy = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for test in copy:
            src = sources.get(test["id"])
            if src is None or "dataset" not in src or test.get("dataset") == src["dataset"]:
                continue
            if fix:
                test["dataset"], changed = src["dataset"], True
            else:
                problems.append(f"{name}: dataset of {test['id']} differs from its YAML source")
        if changed:
            path.write_text(json.dumps(copy, ensure_ascii=False, **dump), encoding="utf-8")
            print(f"{name}: dataset fields rewritten from the YAML sources")
    for p in problems:
        print(p)
    print("registry OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(fix="--fix" in sys.argv))
