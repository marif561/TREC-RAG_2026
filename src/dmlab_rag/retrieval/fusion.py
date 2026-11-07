"""Unweighted reciprocal rank fusion (Eq. 3): RRF(d) = Σ_q 1 / (k + r_q(d)), k = 60."""

from __future__ import annotations

from . import Doc


def rrf(ranked_lists: dict[str, list[Doc]], k: int = 60, pool_size: int = 100) -> list[Doc]:
    scores: dict[str, float] = {}
    docs: dict[str, Doc] = {}
    for ranking in ranked_lists.values():
        for rank, d in enumerate(ranking, start=1):
            scores[d.docid] = scores.get(d.docid, 0.0) + 1.0 / (k + rank)
            docs.setdefault(d.docid, d)
    order = sorted(scores, key=lambda i: (-scores[i], i))[:pool_size]
    out = []
    for i in order:
        d = docs[i]
        d.signals["rrf"] = scores[i]
        out.append(d)
    return out
