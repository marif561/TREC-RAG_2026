"""Small, dependency-free text utilities shared by retrieval and generation."""

from __future__ import annotations

import re

TOKEN = re.compile(r"[a-z0-9]+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its may me might my no not of on or our should so such than that the their them then
    there these they this those to was we were what when where which while who why will with would
    you your about after also any before between both each more most other over some under very""".split()
)


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def content_terms(text: str) -> set[str]:
    """Lower-cased alphanumeric tokens minus stopwords and 1-character tokens."""
    return {t for t in tokens(text) if t not in STOPWORDS and len(t) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def overlap_fraction(a: set[str], b: set[str]) -> float:
    """|a ∩ b| / |a| — share of a's terms that also appear in b."""
    return len(a & b) / len(a) if a else 0.0


def split_sentences(text: str, min_chars: int = 0, max_chars: int | None = None) -> list[str]:
    sents = [s.strip() for s in SENT_SPLIT.split(text.replace("\n", " ")) if s.strip()]
    return [s for s in sents if len(s) >= min_chars and (max_chars is None or len(s) <= max_chars)]


def word_count(text: str) -> int:
    """Word count exactly as the track measures it: ``len(text.split())``."""
    return len(text.split())
