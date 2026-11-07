"""Atomic claim extraction in two passes (Section 4).

Pass 1 (strong model, gpt-5.6-sol): claims over all harvested evidence; every claim must
cite an evidence id.
Pass 2 (gpt-4.1-mini, denser): overlapping windows of 18 evidence sentences, plus targeted
extraction for sub-narratives whose terms overlap the current claims by less than 0.35.
Up to 90 claims are kept; near-duplicates at ≥ 0.85 content-term overlap are removed.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm import ChatModel, parse_json
from ..prompts import render
from ..retrieval.subnarratives import SubNarrative, as_bullets
from ..text import content_terms, overlap_fraction
from .evidence import Evidence, numbered


@dataclass
class Claim:
    cid: int
    text: str
    support: list[int]     # evidence ids
    source: str            # "pass1" | "window" | "targeted"


def _extract(llm: ChatModel, topic: str, evidence: list[Evidence], n_ask: int, aspects: str,
             source: str) -> list[Claim]:
    valid = {e.eid for e in evidence}
    reply = parse_json(llm(render("claim_extraction", n_ask=n_ask, topic=topic,
                                  aspects_or_priority_aspect=aspects,
                                  numbered_evidence_sentences=numbered(evidence))))
    out = []
    for c in reply.get("claims", []):
        support = [int(s) for s in c.get("support", []) if str(s).lstrip("-").isdigit() and int(s) in valid]
        text = (c.get("text") or "").strip()
        if text and support:                      # uncited claims are discarded
            out.append(Claim(-1, text, support, source))
    return out


class ClaimPool:
    def __init__(self, max_claims: int = 90, dup_overlap: float = 0.85):
        self.claims: list[Claim] = []
        self._terms: list[set[str]] = []
        self.max_claims = max_claims
        self.dup_overlap = dup_overlap

    def add(self, claims: list[Claim]) -> None:
        for c in claims:
            if len(self.claims) >= self.max_claims:
                return
            t = content_terms(c.text)
            if any(overlap_fraction(t, o) >= self.dup_overlap or overlap_fraction(o, t) >= self.dup_overlap
                   for o in self._terms):
                continue
            c.cid = len(self.claims)
            self.claims.append(c)
            self._terms.append(t)

    def term_set(self) -> set[str]:
        return set().union(*self._terms) if self._terms else set()


def windows(evidence: list[Evidence], size: int = 18, stride: int = 9) -> list[list[Evidence]]:
    if len(evidence) <= size:
        return [evidence] if evidence else []
    starts = list(range(0, len(evidence) - size + 1, stride))
    if starts[-1] + size < len(evidence):
        starts.append(len(evidence) - size)
    return [evidence[s:s + size] for s in starts]


def extract_claims(topic: str, subs: list[SubNarrative], evidence: list[Evidence],
                   strong: ChatModel, dense: ChatModel, cfg: dict) -> list[Claim]:
    pool = ClaimPool(cfg["max_claims"], cfg["claim_dup_overlap"])
    aspects = "Aspects to cover:\n" + as_bullets(subs) if subs else ""
    pool.add(_extract(strong, topic, evidence, cfg["pass1_n_ask"], aspects, "pass1"))

    for win in windows(evidence, cfg["window_size"], cfg["window_stride"]):
        pool.add(_extract(dense, topic, win, cfg["window_n_ask"], aspects, "window"))

    covered = pool.term_set()
    for sn in subs:
        if overlap_fraction(content_terms(sn.text), covered) < cfg["targeted_overlap"]:
            pool.add(_extract(dense, topic, evidence, cfg["targeted_n_ask"],
                              f"Priority aspect: {sn.text}", "targeted"))
            covered = pool.term_set()
    return pool.claims
