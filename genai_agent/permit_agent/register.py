"""Retrieval tool over the Government of Canada AI Register (open data, Open Government Licence - Canada).

The register lists the AI systems of federal organizations, in English and French. `load_register()` keeps the
columns needed to answer questions about a system (organization, status, users, developer, personal information,
capabilities, description) and drops the vendor and personal-information-bank columns. `search_register` ranks the
systems by BM25 over their names, organization and description, and returns the best three records.

`REMOVED` lets a scenario take systems out of the corpus at run time (corpus loss).
"""

import math
import unicodedata
from collections import Counter

import pandas as pd

KEPT = {
    "ai_register_id": "id",
    "name_ai_system_en": "name_en",
    "name_ai_system_fr": "name_fr",
    "government_organization": "organization",
    "description_ai_system_en": "description_en",
    "description_ai_system_fr": "description_fr",
    "ai_system_primary_users_en": "primary_users",
    "developed_by_en": "developed_by",
    "ai_system_status_en": "status",
    "status_date": "status_year",
    "ai_system_capabilities_en": "capabilities_en",
    "involves_personal_information": "involves_personal_information",
}
TOP_K = 3
SYSTEMS: list = []
REMOVED: set = set()
_INDEX: dict = {}


def fold(text) -> str:
    """Lower case without accents (questions and records may be typed with or without them)"""
    text = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _terms(text) -> list:
    return [w for w in "".join(c if c.isalnum() else " " for c in fold(text)).split() if len(w) > 2]


def load_register(path) -> list:
    """Records of the register (kept columns only, stripped strings); also installs them as the search corpus"""
    frame = pd.read_csv(path, encoding="utf-8-sig")[list(KEPT)].rename(columns=KEPT)
    frame = frame.astype(object).where(frame.notna(), None)
    records = []
    for row in frame.to_dict("records"):
        row = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
        if row["status_year"] is not None:
            row["status_year"] = int(row["status_year"])
        records.append(row)
    install(records)
    return records


def install(records: list) -> None:
    SYSTEMS[:] = records
    docs = [_terms(" ".join(str(r[k] or "") for k in ("name_en", "name_fr", "organization", "description_en",
                                                      "description_fr", "capabilities_en"))) for r in records]
    df = Counter(t for d in docs for t in set(d))
    _INDEX.update(docs=[Counter(d) for d in docs], lengths=[len(d) for d in docs], df=df, n=len(docs),
                  avg=sum(map(len, docs)) / max(len(docs), 1))


def _bm25(query: str, k1: float = 1.2, b: float = 0.75) -> list:
    terms, n, avg = _terms(query), _INDEX["n"], _INDEX["avg"]
    scores = []
    for i, (counts, length) in enumerate(zip(_INDEX["docs"], _INDEX["lengths"])):
        if SYSTEMS[i]["id"] in REMOVED:
            continue
        s = 0.0
        for t in terms:
            if t in counts:
                idf = math.log(1 + (n - _INDEX["df"][t] + 0.5) / (_INDEX["df"][t] + 0.5))
                s += idf * counts[t] * (k1 + 1) / (counts[t] + k1 * (1 - b + b * length / avg))
        if s > 0:
            scores.append((s, i))
    return sorted(scores, reverse=True)[:TOP_K]


def passage(record: dict) -> str:
    """One-paragraph text of a record, used as retrieved context (faithfulness of the answer)"""
    pi = {"Y": "yes", "N": "no"}.get(record["involves_personal_information"], "not stated")
    return (
        f"{record['name_en']} / {record['name_fr']}. Organization: {record['organization']}. "
        f"Status: {record['status'] or 'not stated'} ({record['status_year'] or 'year not stated'}). "
        f"Primary users: {record['primary_users'] or 'not stated'}. Developed by: {record['developed_by'] or 'not stated'}. "
        f"Involves personal information: {pi}. Description: {(record['description_en'] or '')[:200]}"
    )


def search_register(query: str) -> list:
    """The three AI systems of the register that best match the query (BM25), with their main fields"""
    if not SYSTEMS:
        raise RuntimeError("register not loaded: call load_register() first")
    found = []
    for score, i in _bm25(query):
        record = SYSTEMS[i]
        found.append({"id": record["id"], "score": round(score, 2), "text": passage(record)})
    return found


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_register",
        "description": "Search the Government of Canada AI Register (AI systems of federal organizations: "
        "organization, status, users, developer, personal information, description). Pass the name of the system.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
}
SYSTEM_PROMPT = (
    "You answer questions about the Government of Canada AI Register. Always call search_register with the name of "
    "the system first, then answer only from the records returned. If no record matches the system, say that it is "
    "not in the register. Answer in the language of the question, in one short sentence."
)
