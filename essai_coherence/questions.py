"""Thirty questions with known short answers (public-services arithmetic, Canadian civics and geography), French and
English, for the consistency campaign. Synthetic and public knowledge: no personal data.

Each question is a dict for `pulsar_data_collection.genai.ConsistencyQuestion`: number, yes / no or label questions,
so that the final answer of every reply can be read and compared exactly. City questions are label questions over a
list of Canadian cities: a reply naming a city outside the list cannot be read (unparsed).
"""

CITIES = (
    "Winnipeg",
    "Halifax",
    "Iqaluit",
    "Whitehorse",
    "Charlottetown",
    "Yellowknife",
    "Regina",
    "Saskatoon",
    "Edmonton",
    "Calgary",
    "Victoria",
    "Vancouver",
    "Toronto",
    "Ottawa",
    "Montréal",
    "Québec",
    "Fredericton",
    "Moncton",
    "Saint John",
    "St. John's",
    "Dartmouth",
    "Sydney",
    "Brandon",
    "Summerside",
    "Gatineau",
)

_NUMBER = [
    (
        "n01",
        "Une prestation de 245 $ est versée toutes les deux semaines. Quel est le total sur un an (26 versements) ? "
        "Répondez par le nombre seulement.",
        "6370",
    ),
    (
        "n02",
        "A benefit of 180 dollars is paid every month. What is the total over 18 months? Answer with the number only.",
        "3240",
    ),
    ("n03", "Combien de provinces compte le Canada ? Répondez par un nombre seulement.", "10"),
    ("n04", "How many territories does Canada have? Answer with a number only.", "3"),
    (
        "n05",
        "Un loyer de 1250 $ par mois augmente de 4 %. Quel est le nouveau loyer mensuel en dollars ? Répondez par le "
        "nombre seulement.",
        "1300",
    ),
    (
        "n06",
        "A household earns 52000 dollars a year. A program pays 15 % of the income above 40000 dollars. How many "
        "dollars does it pay? Answer with the number only.",
        "1800",
    ),
    ("n07", "In what year did Canadian Confederation take place? Answer with the year only.", "1867"),
    ("n08", "Combien de jours compte une année bissextile ? Répondez par le nombre seulement.", "366"),
    (
        "n09",
        "A form has 12 sections and 7 are completed. What percentage is completed, rounded to the nearest whole "
        "number? Answer with the number only.",
        "58",
    ),
    ("n10", "Quel est le produit de 37 par 23 ? Répondez par le nombre seulement.", "851"),
]
_YES_NO = [
    ("y01", "Ottawa est-elle la capitale du Canada ? Répondez par oui ou par non.", "yes"),
    ("y02", "Is French an official language of Canada at the federal level? Answer yes or no.", "yes"),
    ("y03", "Le Nunavut est-il une province ? Répondez par oui ou par non.", "no"),
    ("y04", "Is the Prime Minister of Canada elected directly by all voters nationwide? Answer yes or no.", "no"),
    ("y05", "Is 2027 a leap year? Answer yes or no.", "no"),
    ("y06", "Le Québec a-t-il une frontière terrestre avec les États-Unis ? Répondez par oui ou par non.", "yes"),
    ("y07", "Is 391 a prime number? Answer yes or no.", "no"),
    ("y08", "Le lac Supérieur est-il entièrement situé au Canada ? Répondez par oui ou par non.", "no"),
    ("y09", "Does Canada's Privacy Act apply to federal government institutions? Answer yes or no.", "yes"),
    ("y10", "Is Vancouver the capital of British Columbia? Answer yes or no.", "no"),
]
_LABEL = [
    ("l01", "Quelle est la capitale du Manitoba ? Répondez par le nom de la ville seulement.", CITIES, "Winnipeg"),
    ("l02", "What is the capital of Nova Scotia? Answer with the city name only.", CITIES, "Halifax"),
    ("l03", "Quelle est la capitale du Nunavut ? Répondez par le nom de la ville seulement.", CITIES, "Iqaluit"),
    ("l04", "What is the capital of Yukon? Answer with the city name only.", CITIES, "Whitehorse"),
    (
        "l05",
        "Quelle est la capitale de l'Île-du-Prince-Édouard ? Répondez par le nom de la ville seulement.",
        CITIES,
        "Charlottetown",
    ),
    (
        "l06",
        "Which province has the largest population in Canada: Ontario, Quebec or British Columbia? Answer with the "
        "province name only.",
        ("Ontario", "Quebec", "British Columbia"),
        "Ontario",
    ),
    (
        "l07",
        "Quelle province est la plus grande en superficie : Québec, Ontario ou Colombie-Britannique ? Répondez par "
        "le nom seulement.",
        ("Québec", "Ontario", "Colombie-Britannique"),
        "Québec",
    ),
    (
        "l08",
        "Which ocean borders British Columbia: Atlantic, Pacific or Arctic? Answer with one word.",
        ("Atlantic", "Pacific", "Arctic"),
        "Pacific",
    ),
    (
        "l09",
        "Which territory has Yellowknife as its capital: Yukon, Northwest Territories or Nunavut? Answer with the " "name only.",
        ("Yukon", "Northwest Territories", "Nunavut"),
        "Northwest Territories",
    ),
    (
        "l10",
        "L'exercice financier du gouvernement fédéral commence le 1er avril. En quelle saison est-ce : hiver, "
        "printemps, été ou automne ? Répondez par un mot.",
        ("hiver", "printemps", "été", "automne"),
        "printemps",
    ),
]


def questions() -> list:
    out = [{"question_id": i, "text": t, "answer_type": "number", "expected": e} for i, t, e in _NUMBER]
    out += [{"question_id": i, "text": t, "answer_type": "yes_no", "expected": e} for i, t, e in _YES_NO]
    out += [
        {"question_id": i, "text": t, "answer_type": "label", "labels": tuple(labels), "expected": e}
        for i, t, labels, e in _LABEL
    ]
    return out
