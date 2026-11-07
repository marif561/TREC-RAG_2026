"""Listwise rerank of the head (Section 3.5, Eq. 4).

The listwise model reorders the top 30 documents, seeing up to 1,800 characters of each
plus the sub-narratives. Inside the head the submitted score is

    s(d) = α · pct_list(d) + (1 − α) · pct_ens(d),   α = 0.55

where both percentiles are computed within the head. Documents below the head keep their
ensemble order.
"""

from __future__ import annotations

from ..llm import ChatModel, parse_json
from ..prompts import render
from .ensemble import rank_percentile
from .judges import format_passages
from . import Doc


def listwise_order(narrative: str, sub_narratives: str, head: list[Doc], llm: ChatModel,
                   max_chars: int = 1800) -> list[int]:
    reply = parse_json(llm(render("listwise_rerank", narrative=narrative, sub_narratives=sub_narratives,
                                  passages=format_passages(head, max_chars))))
    order, seen = [], set()
    for i in reply.get("ordered_ids", []):
        i = int(i)
        if 0 <= i < len(head) and i not in seen:
            order.append(i)
            seen.add(i)
    # Any id the model dropped keeps its ensemble position after the ranked ones.
    return order + [i for i in range(len(head)) if i not in seen]


def interpolate(head: list[Doc], order: list[int], alpha: float = 0.55) -> list[Doc]:
    """Combine listwise and ensemble positions within the head (Eq. 4)."""
    n = len(head)
    list_rank = {idx: r for r, idx in enumerate(order)}
    for ens_rank, d in enumerate(head):
        d.signals["pct_list"] = rank_percentile(n, list_rank[ens_rank])
        d.signals["pct_ens"] = rank_percentile(n, ens_rank)
        d.signals["final"] = alpha * d.signals["pct_list"] + (1 - alpha) * d.signals["pct_ens"]
    return sorted(head, key=lambda d: -d.signals["final"])


def rerank(narrative: str, sub_narratives: str, ranked: list[Doc], llm: ChatModel,
           head_size: int = 30, alpha: float = 0.55, max_chars: int = 1800) -> list[Doc]:
    head, tail = ranked[:head_size], ranked[head_size:]
    if len(head) > 1:
        head = interpolate(head, listwise_order(narrative, sub_narratives, head, llm, max_chars), alpha)
    return head + tail
