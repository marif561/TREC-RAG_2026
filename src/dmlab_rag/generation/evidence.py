"""Evidence harvesting (Section 4).

Evidence comes from the top 40 documents of the listwise run, before the variable-depth
cut. Documents are split into sentences of 50-420 characters, each sentence is scored by
its content-term overlap with the narrative and sub-narratives, and a round-robin pass
across documents keeps up to 80 sentences.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..retrieval import Doc
from ..text import content_terms, split_sentences


@dataclass(frozen=True)
class Evidence:
    eid: int
    doc_index: int      # index into the evidence document list
    docid: str
    text: str
    score: float


def harvest(docs: list[Doc], query_texts: list[str], top_docs: int = 40, min_chars: int = 50,
            max_chars: int = 420, max_sentences: int = 80) -> tuple[list[Doc], list[Evidence]]:
    docs = docs[:top_docs]
    q_terms = set().union(*(content_terms(t) for t in query_texts)) if query_texts else set()

    per_doc: list[list[tuple[float, str]]] = []
    for d in docs:
        scored = []
        for s in split_sentences(d.text, min_chars, max_chars):
            t = content_terms(s)
            scored.append((len(t & q_terms) / (len(t) ** 0.5) if t else 0.0, s))
        scored.sort(key=lambda x: -x[0])
        per_doc.append([x for x in scored if x[0] > 0])

    evidence: list[Evidence] = []
    depth = 0
    while len(evidence) < max_sentences and any(depth < len(p) for p in per_doc):
        for i, sents in enumerate(per_doc):
            if depth < len(sents) and len(evidence) < max_sentences:
                score, text = sents[depth]
                evidence.append(Evidence(len(evidence), i, docs[i].docid, text, score))
        depth += 1
    return docs, evidence


def numbered(evidence: list[Evidence]) -> str:
    return "\n".join(f"[{e.eid}] {e.text}" for e in evidence)
