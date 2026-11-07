"""WIN ensemble score (Section 3.4, Table 2).

Signals are normalised to [0, 1] and combined with weights frozen on the development
topics. When a signal is missing at test time (e.g. the coverage cache), its weight is
redistributed proportionally over the signals that are present.
"""

from __future__ import annotations

from . import Doc

# Frozen development weights (Table 2).
WIN_WEIGHTS = {
    "r": 0.3,    # rank percentile after fusion
    "g": 0.1,    # gpt-4.1-mini pointwise grade, 0-100
    "uq": 0.4,   # UMBRELA grade, Qwen3.5-9B, 0-4
    "um": 0.1,   # UMBRELA grade, Ministral-3-14B, 0-4
    "cov": 0.1,  # coverage pseudo-UMBRELA
}
SCALE = {"g": 100.0, "uq": 4.0, "um": 4.0, "cov": 4.0}


def rank_percentile(n: int, rank: int) -> float:
    """1.0 for the top document, 0.0 for the last (rank is 0-based)."""
    return 1.0 if n <= 1 else 1.0 - rank / (n - 1)


def normalise(signal: str, value: float | None) -> float | None:
    if value is None:
        return None
    return min(1.0, max(0.0, value / SCALE.get(signal, 1.0)))


def redistributed_weights(available: set[str], weights: dict[str, float] = WIN_WEIGHTS) -> dict[str, float]:
    present = {k: w for k, w in weights.items() if k in available}
    total = sum(present.values())
    if total == 0:
        raise ValueError("No ensemble signal available")
    return {k: w / total for k, w in present.items()}


def ensemble_scores(pool: list[Doc], weights: dict[str, float] = WIN_WEIGHTS) -> list[Doc]:
    """Score fused candidates and return them sorted by WIN score.

    Expects raw signals in ``doc.signals`` under the keys of ``weights`` (``r`` is
    computed here from fusion order). A signal missing for *every* document is treated as
    unavailable and its weight is redistributed; a value missing for a single document
    counts as 0.
    """
    n = len(pool)
    for rank, d in enumerate(pool):
        d.signals["r"] = rank_percentile(n, rank)
    available = {s for s in weights if any(d.signals.get(s) is not None for d in pool)}
    w = redistributed_weights(available, weights)
    for d in pool:
        d.signals["win"] = sum(wt * (normalise(s, d.signals.get(s)) or 0.0) for s, wt in w.items())
    return sorted(pool, key=lambda d: -d.signals["win"])
