"""IAC / Propaganda column names and the propaganda pooling choices.

Dataset-specific; to be replaced by a dataset spec file.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple


# How the 18 technique heads are pooled into the propaganda output (see
# pool_probabilities). The output keeps the name propaganda_max_pool whatever
# the strategy, so reports and saved predictions stay comparable.
PROPAGANDA_POOLS = ("max", "noisy_or", "lse")

# ---------------------------------------------------------------------------
# Column groups (testing.md §4.3)
# ---------------------------------------------------------------------------

PROPAGANDA_TECHNIQUE_COLUMNS: Tuple[str, ...] = (
    "Loaded_Language", "Slogans", "Black-and-White_Fallacy", "Flag-Waving",
    "Name_Calling", "Whataboutism", "Causal_Oversimplification",
    "Exaggeration,Minimisation", "Doubt", "Appeal_to_Authority", "Repetition",
    "Appeal_to_fear-prejudice", "Red_Herring", "Thought-terminating_Cliches",
    "Reductio_ad_hitlerum", "Bandwagon",
    "Obfuscation,Intentional_Vagueness,Confusion", "Straw_Men",
)

DISCOURSE_REGRESSION_COLUMNS: Tuple[str, ...] = (
    "disagree_agree", "attacking_respectful", "emotion_fact", "nasty_nice",
    "personal_audience", "defeater_undercutter", "negotiate_attack",
    "questioning_asserting",
)

DISCOURSE_BINARY_COLUMNS: Tuple[str, ...] = tuple(
    f"{c}_bi" for c in DISCOURSE_REGRESSION_COLUMNS
)

# For experiments that model propaganda techniques as separate binary heads,
# sanitize task names so Keras layer scopes remain valid (e.g., commas are invalid).
def _sanitize_task_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.\\/>-]", "_", name)
    if not safe or not re.match(r"^[A-Za-z0-9.]", safe):
        safe = f"task_{safe}"
    return safe


PROPAGANDA_BINARY_TASK_NAMES: Tuple[str, ...] = tuple(
    _sanitize_task_name(col) for col in PROPAGANDA_TECHNIQUE_COLUMNS
)

if len(set(PROPAGANDA_BINARY_TASK_NAMES)) != len(PROPAGANDA_BINARY_TASK_NAMES):
    raise ValueError(
        "Sanitized propaganda binary task names are not unique; adjust naming."
    )

PROPAGANDA_BINARY_LABEL_MAP: Dict[str, str] = dict(
    zip(PROPAGANDA_BINARY_TASK_NAMES, PROPAGANDA_TECHNIQUE_COLUMNS)
)


# Multi-task column of the paper's Table 3 (weighted F1, small BERT), keyed by
# the model output that reports it.
PAPER_TABLE3_MULTITASK_F1: Dict[str, Tuple[str, float]] = {
    "propaganda_max_pool": ("Propaganda", 61.74),
    "disagree_agree_bi": ("Disagree/Agree", 71.38),
    "emotion_fact_bi": ("Emotion/Fact", 63.93),
    "attacking_respectful_bi": ("Attacking/Respectful", 68.07),
    "nasty_nice_bi": ("Nasty/Nice", 73.69),
    "personal_audience_bi": ("Personal/Audience", 65.69),
    "defeater_undercutter_bi": ("Defeater/Undercutter", 55.65),
    "negotiate_attack_bi": ("Negotiate/Attack", 64.81),
    "questioning_asserting_bi": ("Questioning/Asserting", 63.23),
    "WA_bi": ("Argument Quality", 79.17),
}
