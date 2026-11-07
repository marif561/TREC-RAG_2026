"""End-to-end DMLabRAGMerge for one narrative."""

from __future__ import annotations

from dataclasses import dataclass

from ..llm import ChatModel
from ..retrieval.pipeline import RetrievalResult
from ..text import word_count
from .assemble import assemble
from .citations import label_support, repair
from .claims import extract_claims
from .evidence import harvest
from .sentences import claims_to_sentences


@dataclass
class Answer:
    topic_id: str
    narrative: str
    references: list[str]                 # docids
    answer: list[dict]                    # [{"text": ..., "citations": [indices into references]}]

    @property
    def length(self) -> int:
        return sum(word_count(a["text"]) for a in self.answer)


def generate(ret: RetrievalResult, models: dict[str, ChatModel], cfg: dict) -> Answer:
    g = cfg["generation"]
    narrative = ret.topic.narrative
    sub_texts = [s.text for s in ret.sub_narratives]

    docs, evidence = harvest(ret.reranked, [narrative] + sub_texts, g["evidence_top_docs"],
                             g["evidence_min_chars"], g["evidence_max_chars"], g["evidence_max_sentences"])
    claims = extract_claims(narrative, ret.sub_narratives, evidence, models["claims_strong"],
                            models["claims_dense"], g)
    sentences = claims_to_sentences(narrative, claims, evidence, models["sentences"],
                                    max_citations=g["max_citations"])
    draft = assemble(sentences, sub_texts, g["word_budget"], g["aspect_weight"])
    if g.get("verify_citations", True):
        draft = repair(draft, label_support(narrative, draft, docs, models["citation_judge"],
                                            g["citation_max_chars"]))

    # References: documents cited before repair, in first-citation order. Uncited references
    # carry no penalty, so the list is kept unless ``prune_references`` is set.
    ref_order: list[int] = []
    for s in draft:
        for di in s.citations:
            if di not in ref_order:
                ref_order.append(di)
    if g.get("prune_references", False):
        ref_order = [di for di in ref_order if any(di in s.citations for s in draft)]
    pos = {di: i for i, di in enumerate(ref_order)}
    answer = [{"text": s.text, "citations": [pos[di] for di in s.citations if di in pos]} for s in draft]
    return Answer(ret.topic.topic_id, narrative, [docs[di].docid for di in ref_order], answer)
