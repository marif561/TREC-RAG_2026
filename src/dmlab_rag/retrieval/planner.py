"""Query planning (Section 3.1): turn one narrative into up to seven lexical query views.

Views: (1) the original narrative, (2) a bag of up to 24 entities and keywords,
(3) one concatenated query of at most 120 words built from subquestions and keywords,
(4-7) two to four HyDE-style hypothetical evidence passages of 40-80 words.
All views are searched with BM25.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm import ChatModel, parse_json
from ..prompts import render


@dataclass
class QueryPlan:
    narrative: str
    keyword_bag: str
    big_query: str
    sketches: list[str]

    def views(self, max_views: int = 7) -> dict[str, str]:
        views = {"narrative": self.narrative}
        if self.keyword_bag:
            views["keywords"] = self.keyword_bag
        if self.big_query:
            views["big_query"] = self.big_query
        for i, s in enumerate(self.sketches):
            views[f"sketch_{i}"] = s
        return dict(list(views.items())[:max_views])


def _dedupe_terms(terms: list[str], limit: int) -> list[str]:
    seen, out = set(), []
    for t in terms:
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(t.strip())
    return out[:limit]


def _truncate_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def plan_queries(narrative: str, llm: ChatModel, max_bag_terms: int = 24,
                 max_big_query_words: int = 120, max_sketches: int = 4) -> QueryPlan:
    plan = parse_json(llm(render("sparse_query_plan", topic=narrative)))
    sk = parse_json(llm(render("evidence_sketches", topic=narrative)))

    bag = _dedupe_terms(list(plan.get("entity_terms", [])) + list(plan.get("keywords", [])), max_bag_terms)
    big = plan.get("sparse_big_query") or " ".join(plan.get("subquestions", []) + bag)
    sketches = [s for s in sk.get("evidence_sketches", []) if s.strip()][:max_sketches]
    return QueryPlan(narrative, " ".join(bag), _truncate_words(big, max_big_query_words), sketches)
