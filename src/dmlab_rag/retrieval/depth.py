"""Variable retrieval depth (Section 3.6).

Keep a document when max(Qwen, Ministral) UMBRELA grade ≥ 2, preserving the final order,
with a floor of one document and a ceiling of 100. (An earlier development policy used a
floor of 10; the uploaded test run used 1.)
"""

from __future__ import annotations

from . import Doc


def select_depth(ranked: list[Doc], min_grade: int = 2, floor: int = 1, ceiling: int = 100,
                 judges: tuple[str, ...] = ("uq", "um")) -> list[Doc]:
    def grade(d: Doc) -> float:
        vals = [d.signals.get(j) for j in judges]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else -1

    kept = [d for d in ranked if grade(d) >= min_grade][:ceiling]
    if len(kept) < floor:
        extra = [d for d in ranked if d not in kept]
        kept = kept + extra[: floor - len(kept)]
        kept.sort(key=ranked.index)
    return kept
