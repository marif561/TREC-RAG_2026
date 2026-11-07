"""nDCG, local citation precision/recall (Eqs. 1-2) and paired bootstrap."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """TREC qrels: ``topic 0 docid grade`` per line."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            t, _, d, g = line.split()[:4]
            qrels[t][d] = int(g)
    return dict(qrels)


def ndcg_at_k(ranking: list[str], grades: dict[str, int], k: int) -> float:
    """Graded nDCG with gain = grade and log2 discount; unjudged documents count as 0."""
    dcg = sum(grades.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def citation_precision_recall(cited: list[list[str]], support: dict[tuple[int, str], float]) -> tuple[float, float]:
    """Weighted citation precision and recall (Eqs. 1-2).

    ``cited[i]`` is the list of documents cited by answer object i; ``support`` maps
    (i, docid) to w ∈ {0, 0.5, 1}. An object with no citation contributes 0 to recall.
    """
    pairs = [(i, d) for i, docs in enumerate(cited) for d in docs]
    prec = sum(support.get(p, 0.0) for p in pairs) / len(pairs) if pairs else 0.0
    rec = 0.0
    for i, docs in enumerate(cited):
        if docs:
            rec += sum(support.get((i, d), 0.0) for d in docs) / max(1, len(docs))
    rec /= len(cited) if cited else 1
    return prec, rec


def paired_bootstrap(a: list[float], b: list[float], n: int = 10_000, seed: int = 42,
                     alpha: float = 0.05) -> dict:
    """Mean of (b − a) over topics with a percentile bootstrap CI."""
    assert len(a) == len(b) and a
    diffs = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    m = len(diffs)
    boot = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    lo, hi = boot[int(alpha / 2 * n)], boot[int((1 - alpha / 2) * n) - 1]
    return {"mean": sum(diffs) / m, "lo": lo, "hi": hi,
            "improved": sum(d > 0 for d in diffs), "topics": m}
