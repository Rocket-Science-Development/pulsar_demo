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


FUNCTIONS = {
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
