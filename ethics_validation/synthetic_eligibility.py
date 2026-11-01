"""Semi-synthetic eligibility decisions with Canadian synthetic attributes and a bias injected on purpose.

No real person: every row is drawn from fixed distributions. Attributes: official language spoken at home (fr, en,
other), region (Montréal, Québec, Est-du-Québec, Ottawa-Gatineau), age group (18-34, 35-54, 55+). The decision rule
approves applicants from a latent need score; `bias` lowers the approval probability of one group by a known amount,
so that the power of a fairness test can be measured against the truth.
"""

from typing import Optional

import numpy as np
import pandas as pd

LANGUAGES = {"fr": 0.70, "en": 0.20, "other": 0.10}
REGIONS = {"Montréal": 0.45, "Québec": 0.25, "Est-du-Québec": 0.15, "Ottawa-Gatineau": 0.15}
AGE_GROUPS = {"18-34": 0.35, "35-54": 0.40, "55+": 0.25}


def _draw(rng, categories: dict, n: int):
    return rng.choice(list(categories), n, p=list(categories.values()))


def eligibility_decisions(
    n: int,
    approval_rate: float = 0.5,
    bias: Optional[dict] = None,
    group_sizes: Optional[dict] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """`n` synthetic applicants with a decision `y_pred` (1 = approved), a label `y_true` and a score

    Parameters
    ----------
    approval_rate : float
        Approval probability of an applicant without bias
    bias : dict, optional
        {"language": {"other": 0.15}}: approval probability lowered by 0.15 for that group (the label is not changed:
        the bias is in the decision, not in the need)
    group_sizes : dict, optional
        {"language": {"other": 200, "fr": 2000}}: exact group sizes instead of drawing them (power curves by size)
    """
    rng = np.random.default_rng(seed)
    if group_sizes:
        (attribute, sizes), = group_sizes.items()
        values = np.concatenate([[g] * k for g, k in sizes.items()])
        n = len(values)
        frame = pd.DataFrame({attribute: rng.permutation(values)})
    else:
        frame = pd.DataFrame({"language": _draw(rng, LANGUAGES, n)})
    for column, categories in (("language", LANGUAGES), ("region", REGIONS), ("age_group", AGE_GROUPS)):
        if column not in frame:
            frame[column] = _draw(rng, categories, n)
    need = rng.random(n)  # latent need, identical in distribution for every group
    probability = np.full(n, approval_rate)
    for attribute, shifts in (bias or {}).items():
        for group, delta in shifts.items():
            probability = np.where(frame[attribute] == group, probability - delta, probability)
    frame["y_true"] = (need < approval_rate).astype(int)
    frame["y_pred"] = (rng.random(n) < np.clip(probability, 0, 1)).astype(int)
    frame["y_pred_proba"] = np.clip(probability + rng.normal(0, 0.05, n), 0, 1)
    frame.insert(0, "applicant_id", [f"s{i:06d}" for i in range(n)])
    return frame
