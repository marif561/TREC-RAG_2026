"""LLM relevance signals: a pointwise 0-100 grade and UMBRELA 0-4 grades."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from ..llm import ChatModel, parse_json
from ..prompts import render
from . import Doc

UMBRELA_SCORE = re.compile(r"##\s*final score:\s*([0-4])", re.I)


def parse_umbrela(text: str) -> int | None:
    m = UMBRELA_SCORE.search(text)
    return int(m.group(1)) if m else None


def format_passages(docs: list[Doc], max_chars: int) -> str:
    return "\n\n".join(f"[{i}] {d.text[:max_chars]}" for i, d in enumerate(docs))


def pointwise_grades(narrative: str, facets: str, docs: list[Doc], llm: ChatModel,
                     batch_size: int = 10, max_chars: int = 1500) -> list[float | None]:
    """gpt-4.1-mini grade from 0 to 100 for every passage, in batches (signal g)."""
    out: list[float | None] = [None] * len(docs)
    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        reply = parse_json(llm(render("pointwise_grade", topic=narrative, facets=facets,
                                      passages=format_passages(batch, max_chars))))
        for g in reply.get("grades", []):
            i = int(g["id"])
            if 0 <= i < len(batch):
                out[start + i] = float(g["score"])
    return out


def umbrela_grades(narrative: str, sub_narratives: str, docs: list[Doc], llm: ChatModel,
                   max_chars: int = 4000, workers: int = 8) -> list[int | None]:
    """One UMBRELA call per passage (signals u_q and u_m); unparseable replies give None."""
    def one(d: Doc) -> int | None:
        return parse_umbrela(llm(render("umbrela_grade", narrative=narrative,
                                        sub_narratives=sub_narratives, passage=d.text[:max_chars])))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, docs))
