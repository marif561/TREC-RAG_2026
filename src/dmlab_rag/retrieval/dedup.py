"""Near-duplicate removal before fusion (Section 3.2).

Two documents are duplicates when their first 200 characters match, or when the token
Jaccard similarity over their first 1,200 characters reaches 0.9. About 39.6% of
ClimbMix documents have a duplicate, so this filter matters.
"""

from __future__ import annotations

from ..text import jaccard, tokens
from . import Doc


def _norm_prefix(text: str, n: int) -> str:
    return " ".join(text[:n].lower().split())


class DuplicateFilter:
    """Stateful filter: remembers kept documents so later views are checked against them."""

    def __init__(self, prefix_chars: int = 200, jaccard_chars: int = 1200, jaccard_threshold: float = 0.9):
        self.prefix_chars = prefix_chars
        self.jaccard_chars = jaccard_chars
        self.threshold = jaccard_threshold
        self.canonical: dict[str, str] = {}      # docid -> canonical docid
        self._prefixes: dict[str, str] = {}      # normalised prefix -> canonical docid
        self._token_sets: list[tuple[str, set[str]]] = []

    def canonical_id(self, doc: Doc) -> str:
        if doc.docid in self.canonical:
            return self.canonical[doc.docid]
        prefix = _norm_prefix(doc.text, self.prefix_chars)
        if prefix and prefix in self._prefixes:
            cid = self._prefixes[prefix]
        else:
            toks = set(tokens(doc.text[: self.jaccard_chars]))
            cid = next((d for d, s in self._token_sets if jaccard(toks, s) >= self.threshold), None)
            if cid is None:
                cid = doc.docid
                self._token_sets.append((cid, toks))
            if prefix:
                self._prefixes.setdefault(prefix, cid)
        self.canonical[doc.docid] = cid
        return cid

    def filter(self, ranked: list[Doc]) -> list[Doc]:
        """Collapse duplicates in a ranked list onto their first (canonical) document."""
        seen, out = set(), []
        for d in ranked:
            cid = self.canonical_id(d)
            if cid not in seen:
                seen.add(cid)
                out.append(d if cid == d.docid else Doc(cid, d.text, d.signals))
        return out
