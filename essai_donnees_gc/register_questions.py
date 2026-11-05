"""Questions with known answers built from the structured fields of the AI Register, and their scoring.

Three kinds of question, in French and English, each about one system named in the question:

| Kind | Known answer (column) | Scored correct when the answer... |
|---|---|---|
| `organization` | `government_organization` | contains the English or the French name of the organization |
| `status` | `ai_system_status_en` | names the right status (production, development, retired) first |
| `personal_information` | `involves_personal_information` (Y / N) | says yes or no, as in the register |

The pool holds systems with a unique name in both languages, a known status and a known answer on personal
information. Scoring is by keywords, accent- and case-insensitive: strict (an organization given only by its acronym
counts as wrong).
"""

import random
import re

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "genai_agent"))

from permit_agent.register import _terms, fold  # noqa: E402

KINDS = ["organization", "status", "personal_information"]
TEMPLATES = {
    ("organization", "fr"): "Quelle organisation fédérale est responsable du système d'IA « {name_fr} » ?",
    ("organization", "en"): 'Which federal organization is responsible for the AI system "{name_en}"?',
    ("status", "fr"): "Quel est le statut actuel du système d'IA « {name_fr} » : en production, en développement ou "
    "hors service ?",
    ("status", "en"): 'What is the current status of the AI system "{name_en}": in production, in development or retired?',
    ("personal_information", "fr"): "Le système d'IA « {name_fr} » implique-t-il des renseignements personnels ? "
    "Répondez par oui ou par non.",
    ("personal_information", "en"): 'Does the AI system "{name_en}" involve personal information? Answer yes or no.',
}
STATUS = {
    "production": ["production"],
    "development": ["development", "developpement", "developing", "elaboration"],
    "retired": ["retired", "hors service", "retire", "decommission"],
}
NEGATIVE = [r"\bno\b", r"\bnon\b", r"\bnot\b", r"n't\b", r"\bpas\b", r"\baucun", r"\bnone\b"]
POSITIVE = [r"\byes\b", r"\boui\b", r"\binvolves\b", r"\bimplique\b", r"\butilise\b", r"\bcomporte\b"]


def status_of(record: dict):
    text = fold(record["status"] or "")
    return next((k for k, words in STATUS.items() if any(w in text for w in words)), None)


def pool(records: list, size: int, seed: int = 0) -> list:
    """Systems usable for questions: unique names, known status and personal information, names of 2 words or more"""
    names_en = [fold(r["name_en"]) for r in records]
    names_fr = [fold(r["name_fr"]) for r in records]
    usable = [
        r for r, en, fr in zip(records, names_en, names_fr)
        if r["name_en"] and r["name_fr"] and names_en.count(en) == 1 and names_fr.count(fr) == 1
        and status_of(r) and r["involves_personal_information"] in ("Y", "N") and len(_terms(r["name_en"])) >= 2
    ]
    return random.Random(seed).sample(usable, min(size, len(usable)))


def make_question(rng: random.Random, systems: list, language: str) -> dict:
    record, kind = rng.choice(systems), rng.choice(KINDS)
    return {"system": record["id"], "kind": kind, "language": language,
            "request": TEMPLATES[(kind, language)].format(**record)}


def score(answer: str, question: dict, record: dict) -> bool:
    """Whether the answer agrees with the register"""
    text = fold(answer or "")
    if question["kind"] == "organization":
        names = [re.sub(r"^(the|l'|la|le|les) ", "", fold(n).strip()) for n in record["organization"].split(" / ")]
        return any(n and n in text for n in names)
    if question["kind"] == "status":
        # the first status named is the answer ("in development, to be put in production in 2026" is development)
        found = [(text.find(w), k) for k, words in STATUS.items() for w in words if w in text]
        return bool(found) and min(found)[1] == status_of(record)
    negative = any(re.search(p, text) for p in NEGATIVE)
    positive = any(re.search(p, text) for p in POSITIVE)
    said = "N" if negative else "Y" if positive else None
    return said == record["involves_personal_information"]
