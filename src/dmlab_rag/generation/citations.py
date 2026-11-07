"""Citation support check and repair (Section 4, Appendix A.3.9).

gpt-4.1-mini labels each (sentence, cited document) pair as support, partial_support or
not_support, seeing up to 4,000 characters of the document. not_support citations are
removed; partial_support citations are kept. Repair can leave a sentence with no
citation, which scores zero weighted citation recall.
"""

from __future__ import annotations

from collections import defaultdict

from ..llm import ChatModel, parse_json
from ..prompts import render
from ..retrieval import Doc
from .sentences import Sentence

LABEL_WEIGHT = {"support": 1.0, "partial_support": 0.5, "not_support": 0.0}


def label_support(narrative: str, sentences: list[Sentence], docs: list[Doc], llm: ChatModel,
                  max_chars: int = 4000) -> dict[tuple[int, int], str]:
    """Return {(sentence index, doc index): label} for every citation pair."""
    by_doc: dict[int, list[int]] = defaultdict(list)
    for si, s in enumerate(sentences):
        for di in s.citations:
            by_doc[di].append(si)
    labels: dict[tuple[int, int], str] = {}
    for di, sis in by_doc.items():
        numbered = "\n".join(f"{k + 1}. {sentences[si].text}" for k, si in enumerate(sis))
        reply = parse_json(llm(render("citation_support", narrative=narrative,
                                      passage=docs[di].text[:max_chars], numbered_sentences=numbered)))
        got = reply.get("labels", [])
        for k, si in enumerate(sis):
            lab = got[k] if k < len(got) and got[k] in LABEL_WEIGHT else "partial_support"
            labels[(si, di)] = lab
    return labels


def repair(sentences: list[Sentence], labels: dict[tuple[int, int], str]) -> list[Sentence]:
    out = []
    for si, s in enumerate(sentences):
        keep = [di for di in s.citations if labels.get((si, di), "partial_support") != "not_support"]
        out.append(Sentence(s.cid, s.text, keep))
    return out
