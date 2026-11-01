"""Synthetic user requests, in French and English, with the tool a good run must use."""

import random

TEMPLATES = {
    "eligibility": [
        (
            "fr",
            "Je gagne {income} $ par année et j'ai {age} ans. Suis-je admissible à l'aide pour {topic_fr} ?",
            "check_eligibility",
        ),
        ("en", "I earn ${income} a year and I am {age}. Am I eligible for the {topic_en} program?", "check_eligibility"),
    ],
    "amount": [
        ("fr", "Avec un revenu de {income} $, combien pourrais-je recevoir pour {topic_fr} ?", "compute_amount"),
        ("en", "With an income of ${income}, how much could I get for {topic_en}?", "compute_amount"),
    ],
    "file": [
        ("fr", "Je m'appelle {user}. Ouvrez-moi un dossier pour {topic_fr}.", "create_file"),
        ("en", "My name is {user}. Please open an application file for {topic_en}.", "create_file"),
    ],
    "general": [
        ("fr", "Quels programmes existent pour {topic_fr} ?", "search_programs"),
        ("en", "Which programs exist for {topic_en}?", "search_programs"),
    ],
}
TOPICS = [
    ("la rénovation de ma maison", "home renovation"),
    ("la garde d'enfants", "child care"),
    ("la formation professionnelle", "job training"),
    ("le loyer", "rent assistance"),
]
USERS = ["Alex Tremblay", "Sam Roy", "Jordan Gagnon", "Charlie Côté"]

# Mix of request kinds; drift scenarios change it (e.g. many more file requests)
DEFAULT_MIX = {"eligibility": 0.35, "amount": 0.30, "file": 0.15, "general": 0.20}


def make_task(rng: random.Random, mix: dict = None, french_share: float = 0.5):
    mix = mix or DEFAULT_MIX
    kind = rng.choices(list(mix), weights=list(mix.values()))[0]
    candidates = TEMPLATES[kind]
    language = "fr" if rng.random() < french_share else "en"
    _, text, expected = next(t for t in candidates if t[0] == language)
    topic_fr, topic_en = rng.choice(TOPICS)
    request = text.format(
        income=rng.randrange(20000, 95000, 1000),
        age=rng.randint(19, 70),
        topic_fr=topic_fr,
        topic_en=topic_en,
        user=rng.choice(USERS),
    )
    return {"kind": kind, "language": language, "request": request, "expected_tool": expected}
