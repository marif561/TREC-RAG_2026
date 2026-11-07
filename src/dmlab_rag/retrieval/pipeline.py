"""End-to-end DMLabWin055 for one narrative."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm import ChatModel
from ..topics import Topic
from . import Doc
from .dedup import DuplicateFilter
from .depth import select_depth
from .ensemble import WIN_WEIGHTS, ensemble_scores
from .fusion import rrf
from .judges import pointwise_grades, umbrela_grades
from .planner import plan_queries
from .rerank import rerank
from .search import Searcher
from .subnarratives import SubNarrative, as_bullets, generate_subnarratives


@dataclass
class RetrievalResult:
    topic: Topic
    sub_narratives: list[SubNarrative]
    reranked: list[Doc]            # full reranked pool (generation uses the top 40 of this)
    submitted: list[Doc]           # after the variable-depth cut
    views: dict[str, str] = field(default_factory=dict)


def retrieve(topic: Topic, searcher: Searcher, models: dict[str, ChatModel], cfg: dict) -> RetrievalResult:
    r = cfg["retrieval"]
    plan = plan_queries(topic.narrative, models["planner"], r["max_bag_terms"],
                        r["max_big_query_words"], r["max_sketches"])
    views = plan.views(r["max_views"])

    dedup = DuplicateFilter(r["dedup_prefix_chars"], r["dedup_jaccard_chars"], r["dedup_jaccard"])
    ranked_lists = {name: dedup.filter(searcher.search(q, r["hits_per_query"])) for name, q in views.items()}
    pool = rrf(ranked_lists, k=r["rrf_k"], pool_size=r["pool_size"])

    subs = generate_subnarratives(topic.narrative, models["subnarratives"], r["n_decompositions"],
                                  r.get("subnarrative_examples"))
    facets = as_bullets(subs)

    for d, g in zip(pool, pointwise_grades(topic.narrative, facets, pool, models["grader"])):
        d.signals["g"] = g
    for role, key in (("umbrela_qwen", "uq"), ("umbrela_ministral", "um")):
        if role in models:
            for d, u in zip(pool, umbrela_grades(topic.narrative, facets, pool, models[role])):
                d.signals[key] = u
    # The coverage pseudo-UMBRELA signal ("cov") was a development cache; when it is absent
    # its weight is redistributed automatically.
    weights = {**WIN_WEIGHTS, **r.get("weights", {})}
    ranked = ensemble_scores(pool, weights)

    reranked = rerank(topic.narrative, facets, ranked, models["listwise"], r["rerank_head"],
                      r["rerank_alpha"], r["rerank_max_chars"])
    submitted = select_depth(reranked, r["depth_min_grade"], r["depth_floor"], r["depth_ceiling"])
    return RetrievalResult(topic, subs, reranked, submitted, views)
