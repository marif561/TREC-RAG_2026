"""Claim-to-sentence conversion (Section 4, Appendix A.3.8).

Each surviving claim becomes one 12-28-word sentence. A sentence may carry up to three
citations, drawn only from the documents that support its source claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..llm import ChatModel, parse_json
from ..prompts import render
from .claims import Claim
from .evidence import Evidence


@dataclass
class Sentence:
    cid: int
    text: str
    citations: list[int]   # indices into the evidence document list, strongest first


def allowed_citations(claim: Claim, evidence: list[Evidence], max_citations: int = 3) -> list[int]:
    by_id = {e.eid: e for e in evidence}
    out: list[int] = []
    for eid in claim.support:
        di = by_id[eid].doc_index
        if di not in out:
            out.append(di)
    return out[:max_citations]


def claims_to_sentences(topic: str, claims: list[Claim], evidence: list[Evidence], llm: ChatModel,
                        batch_size: int = 30, max_citations: int = 3) -> list[Sentence]:
    allowed = {c.cid: allowed_citations(c, evidence, max_citations) for c in claims}
    out: list[Sentence] = []
    for start in range(0, len(claims), batch_size):
        batch = claims[start:start + batch_size]
        payload = json.dumps([{"id": c.cid, "claim": c.text, "allowed_citations": allowed[c.cid]}
                              for c in batch], ensure_ascii=False)
        reply = parse_json(llm(render("claim_to_sentence", topic=topic, claims_with_allowed_citations=payload)))
        got = {int(s["id"]): s for s in reply.get("sentences", []) if "id" in s}
        for c in batch:
            s = got.get(c.cid, {})
            text = (s.get("text") or c.text).strip()
            cites = [int(x) for x in s.get("citations", []) if int(x) in allowed[c.cid]]
            out.append(Sentence(c.cid, text, (cites or allowed[c.cid])[:max_citations]))
    return out
