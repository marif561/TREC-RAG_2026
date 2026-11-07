"""Greedy coverage-driven answer assembly (Eq. 5).

With T the draft's content terms and A the term set of each sub-narrative,

    score(s) = |t(s) \\ T| + 3 · Σ_A max(0, |A ∩ (T ∪ t(s))| / |A| − |A ∩ T| / |A|)

The highest-scoring sentence that still fits is added until the word budget (1,000 words,
counted with ``str.split``) is used or no candidate scores above zero. Sentences stay in
selection order.
"""

from __future__ import annotations

from ..text import content_terms, word_count
from .sentences import Sentence


def sentence_score(s_terms: set[str], draft_terms: set[str], aspects: list[set[str]],
                   aspect_weight: float = 3.0) -> float:
    novelty = len(s_terms - draft_terms)
    gain = 0.0
    union = draft_terms | s_terms
    for a in aspects:
        if a:
            gain += max(0.0, len(a & union) / len(a) - len(a & draft_terms) / len(a))
    return novelty + aspect_weight * gain


def assemble(sentences: list[Sentence], aspect_texts: list[str], budget: int = 1000,
             aspect_weight: float = 3.0) -> list[Sentence]:
    aspects = [content_terms(a) for a in aspect_texts]
    remaining = [(s, content_terms(s.text), word_count(s.text)) for s in sentences]
    draft: list[Sentence] = []
    draft_terms: set[str] = set()
    used = 0
    while remaining:
        best, best_score = None, 0.0
        for item in remaining:
            s, terms, n = item
            if used + n > budget:
                continue
            sc = sentence_score(terms, draft_terms, aspects, aspect_weight)
            if sc > best_score:
                best, best_score = item, sc
        if best is None:
            break
        draft.append(best[0])
        draft_terms |= best[1]
        used += best[2]
        remaining.remove(best)
    return draft
