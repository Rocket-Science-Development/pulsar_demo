"""Tools of the demo agent: a synthetic public-services assistant (no real data, no real service).

`FAULTS` lets the drift scenarios degrade a tool (timeouts, changed behaviour) at run time.
"""

import random
import uuid

PROGRAMS = {
    "PRG-RENO": {
        "name": "Home renovation grant",
        "nom": "Subvention à la rénovation",
        "max_income": 80000,
        "rate": 0.15,
        "cap": 10000,
    },
    "PRG-GARDE": {
        "name": "Child care benefit",
        "nom": "Allocation pour la garde d'enfants",
        "max_income": 60000,
        "rate": 0.10,
        "cap": 6000,
    },
    "PRG-EMPLOI": {
        "name": "Job training support",
        "nom": "Aide à la formation professionnelle",
        "max_income": 50000,
        "rate": 0.08,
        "cap": 4000,
    },
    "PRG-LOGE": {"name": "Rent assistance", "nom": "Aide au loyer", "max_income": 40000, "rate": 0.12, "cap": 5000},
}
TOPICS = {
    "PRG-RENO": ["renovation", "rénovation", "home", "maison", "travaux"],
    "PRG-GARDE": ["child", "enfant", "garde", "daycare", "garderie"],
    "PRG-EMPLOI": ["job", "emploi", "training", "formation"],
    "PRG-LOGE": ["rent", "loyer", "housing", "logement"],
}

# Fault injection used by the drift scenarios: {"check_eligibility": {"timeout_rate": 0.3}}
FAULTS: dict = {}
_TOKENS: set = set()


def search_programs(topic: str) -> list:
    """Find public programs matching a topic"""
    text = topic.lower()
    found = [pid for pid, words in TOPICS.items() if any(w in text for w in words)]
    return [{"program_id": pid, "name": PROGRAMS[pid]["name"], "nom": PROGRAMS[pid]["nom"]} for pid in found] or [
        {"program_id": pid, "name": p["name"]} for pid, p in PROGRAMS.items()
    ]


def _program(program_id: str) -> dict:
    """Program by identifier, or by a name or topic word (models often pass "rent assistance" instead of PRG-LOGE)"""
    if program_id in PROGRAMS:
        return PROGRAMS[program_id]
    text = str(program_id).lower()
    found = [pid for pid, words in TOPICS.items() if any(w in text for w in words)]
    if len(found) != 1:
        raise ValueError(f"unknown program {program_id}")
    return PROGRAMS[found[0]]


def check_eligibility(program_id: str, annual_income: float, age: int) -> dict:
    """Check whether an applicant is eligible to a program"""
    fault = FAULTS.get("check_eligibility", {})
    if random.random() < fault.get("timeout_rate", 0.0):
        raise TimeoutError("eligibility service timed out")
    program = _program(program_id)
    if int(age) < 18:
        return {"eligible": False, "reason": "applicant must be 18 or older"}
    eligible = float(annual_income) <= program["max_income"]
    return {"eligible": eligible, "reason": "income within limit" if eligible else "income above limit"}


def compute_amount(program_id: str, annual_income: float) -> dict:
    """Estimate the amount an eligible applicant could receive"""
    program = _program(program_id)
    amount = min(program["cap"], round(program["rate"] * max(0.0, program["max_income"] - float(annual_income)), 2))
    return {"program_id": program_id, "estimated_amount": amount, "currency": "CAD"}


def authenticate(user_id: str) -> dict:
    """Authenticate the applicant before any file creation"""
    token = uuid.uuid4().hex[:12]
    _TOKENS.add(token)
    return {"token": token}


def create_file(program_id: str, token: str) -> dict:
    """Open an application file (requires a token from authenticate)"""
    if token not in _TOKENS:
        raise PermissionError("authentication required before creating a file")
    return {"file_id": f"DOS-{uuid.uuid4().hex[:8].upper()}", "program_id": program_id, "status": "open"}


# Small synthetic corpus of program rules (French and English) for the retrieval tool (RAG)
DOCUMENTS = [
    {"id": "DOC-LOGE-1", "text": "Aide au loyer : le revenu annuel du ménage doit être inférieur à 40 000 $. "
     "L'aide représente 12 % de l'écart avec ce plafond, jusqu'à 5 000 $ par année."},
    {"id": "DOC-LOGE-2", "text": "Rent assistance: household income must be below $40,000 a year. The benefit is 12% "
     "of the gap to that limit, up to $5,000 a year. Proof of lease is required."},
    {"id": "DOC-RENO-1", "text": "Subvention à la rénovation : offerte aux propriétaires dont le revenu est inférieur à "
     "80 000 $. Elle couvre 15 % des travaux admissibles, jusqu'à 10 000 $."},
    {"id": "DOC-RENO-2", "text": "Home renovation grant: for owners with an income below $80,000. It covers 15% of "
     "eligible work, up to $10,000. Quotes from licensed contractors are required."},
    {"id": "DOC-GARDE-1", "text": "Allocation pour la garde d'enfants : revenu familial inférieur à 60 000 $; jusqu'à "
     "6 000 $ par année, selon le nombre d'enfants de moins de 12 ans."},
    {"id": "DOC-GARDE-2", "text": "Child care benefit: family income below $60,000; up to $6,000 a year depending on "
     "the number of children under 12. A daycare receipt is required."},
    {"id": "DOC-EMPLOI-1", "text": "Aide à la formation professionnelle : revenu inférieur à 50 000 $; couvre 8 % de "
     "l'écart avec le plafond, jusqu'à 4 000 $, pour une formation reconnue."},
    {"id": "DOC-EMPLOI-2", "text": "Job training support: income below $50,000; covers 8% of the gap to the limit, up "
     "to $4,000, for an accredited training program."},
    {"id": "DOC-GEN-1", "text": "Toute demande exige une authentification avant l'ouverture d'un dossier. Les "
     "demandes incomplètes sont refusées après 30 jours."},
    {"id": "DOC-GEN-2", "text": "Every application requires authentication before a file is opened. Incomplete "
     "applications are rejected after 30 days."},
]


def _words(text: str) -> set:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(w) > 3}


def search_documents(query: str) -> list:
    """Retrieve the two passages of the program rules that share the most words with the query

    Scenario S3 removes documents with FAULTS["search_documents"] = {"removed": {ids}}.
    """
    removed = FAULTS.get("search_documents", {}).get("removed", set())
    corpus = [d for d in DOCUMENTS if d["id"] not in removed]
    terms = _words(query)
    ranked = sorted(corpus, key=lambda d: len(terms & _words(d["text"])), reverse=True)
    return [{"id": d["id"], "text": d["text"]} for d in ranked[:2] if terms & _words(d["text"])]


FUNCTIONS = {
    "search_documents": search_documents,
    "search_programs": search_programs,
    "check_eligibility": check_eligibility,
    "compute_amount": compute_amount,
    "authenticate": authenticate,
    "create_file": create_file,
}


def _schema(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS = [
    _schema(
        "search_documents",
        "Search the official rules of the programs (conditions, limits, required documents). Use it to answer questions "
        "about rules, and base the answer on the passages returned.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _schema("search_programs", "Find public programs matching a topic.", {"topic": {"type": "string"}}, ["topic"]),
    _schema(
        "check_eligibility",
        "Check eligibility of an applicant to a program.",
        {"program_id": {"type": "string"}, "annual_income": {"type": "number"}, "age": {"type": "integer"}},
        ["program_id", "annual_income", "age"],
    ),
    _schema(
        "compute_amount",
        "Estimate the amount an eligible applicant could receive.",
        {"program_id": {"type": "string"}, "annual_income": {"type": "number"}},
        ["program_id", "annual_income"],
    ),
    _schema(
        "authenticate", "Authenticate the applicant. Required before create_file.", {"user_id": {"type": "string"}}, ["user_id"]
    ),
    _schema(
        "create_file",
        "Open an application file for a program. Requires a token from authenticate.",
        {"program_id": {"type": "string"}, "token": {"type": "string"}},
        ["program_id", "token"],
    ),
]
ALLOWED_TOOLS = set(FUNCTIONS)
