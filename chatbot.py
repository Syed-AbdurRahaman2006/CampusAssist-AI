"""FAQ retrieval engine using simple text similarity."""

import json
import re
from difflib import SequenceMatcher

from config import DATA_PATH, MATCH_THRESHOLD

# ── Load FAQ data once at startup ──────────────────────────────────
with open(DATA_PATH, encoding="utf-8") as f:
    _FAQ_DATA: list[dict] = json.load(f)

FALLBACK_ANSWER = (
    "Sorry, I couldn't find a matching answer. "
    "Please visit the college help desk or try rephrasing your question."
)


def _normalize(text: str) -> str:
    """Lowercase, strip, and collapse whitespace for fair comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    return re.sub(r"\s+", " ", text)       # collapse spaces


def get_faqs() -> list[dict]:
    """Return all FAQ entries."""
    return _FAQ_DATA


def find_answer(user_question: str) -> tuple[str, str | None, float]:
    """Return (answer, matched_faq_question, confidence_score).

    Uses `difflib.SequenceMatcher` to find the most similar FAQ.
    Falls back to a default message when the best score is below
    `MATCH_THRESHOLD`.
    """
    normalized_input = _normalize(user_question)

    best_score = 0.0
    best_faq: dict | None = None

    for faq in _FAQ_DATA:
        score = SequenceMatcher(
            None, normalized_input, _normalize(faq["question"])
        ).ratio()
        if score > best_score:
            best_score = score
            best_faq = faq

    if best_faq and best_score >= MATCH_THRESHOLD:
        return best_faq["answer"], best_faq["question"], round(best_score, 4)

    return FALLBACK_ANSWER, None, round(best_score, 4)
