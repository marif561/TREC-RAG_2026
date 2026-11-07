"""BM25 search backends.

DMLabWin055 builds no index of its own: it queries the shared ``climbmix-400b`` BM25
index through a Pyserini REST service. The endpoint shape is configurable because it is
set by whoever hosts the service. A local ``LuceneSearcher`` backend is provided for
anyone with a copy of the index.
"""

from __future__ import annotations

from typing import Protocol

from . import Doc


class Searcher(Protocol):
    def search(self, query: str, k: int) -> list[Doc]: ...


class RestSearcher:
    """Generic Pyserini-REST client.

    ``url_template`` is formatted with ``index``, ``query`` (URL-encoded) and ``k``.
    The response is expected to be a JSON list of hits (or an object holding one under
    ``hits_field``) with document id and text fields named by ``id_field``/``text_field``.
    """

    def __init__(self, base_url: str, index: str = "climbmix-400b",
                 url_template: str = "{base_url}/v1/indexes/{index}/search?query={query}&hits={k}",
                 hits_field: str | None = "candidates", id_field: str = "docid",
                 text_field: str = "doc", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.index = index
        self.url_template = url_template
        self.hits_field, self.id_field, self.text_field = hits_field, id_field, text_field
        self.timeout = timeout

    def search(self, query: str, k: int) -> list[Doc]:
        from urllib.parse import quote

        import requests

        url = self.url_template.format(base_url=self.base_url, index=self.index,
                                       query=quote(query), k=k)
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        hits = data[self.hits_field] if self.hits_field and isinstance(data, dict) else data
        out = []
        for h in hits[:k]:
            text = h.get(self.text_field, "")
            if isinstance(text, dict):  # some services return {"contents": ...} or {"text": ...}
                text = text.get("contents") or text.get("text") or text.get("segment") or ""
            out.append(Doc(str(h[self.id_field]), text))
        return out


class LuceneSearcherBackend:
    def __init__(self, index_path: str, k1: float = 0.9, b: float = 0.4):
        import json

        from pyserini.search.lucene import LuceneSearcher

        self._json = json
        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1, b)

    def search(self, query: str, k: int) -> list[Doc]:
        out = []
        for h in self.searcher.search(query, k):
            raw = self.searcher.doc(h.docid).raw()
            try:
                body = self._json.loads(raw)
                raw = body.get("contents") or body.get("text") or raw
            except ValueError:
                pass
            out.append(Doc(h.docid, raw))
        return out


def make_searcher(cfg: dict) -> Searcher:
    kind = cfg.get("backend", "rest")
    if kind == "rest":
        return RestSearcher(**{k: v for k, v in cfg.items() if k != "backend"})
    if kind == "lucene":
        return LuceneSearcherBackend(cfg["index_path"], cfg.get("k1", 0.9), cfg.get("b", 0.4))
    raise ValueError(f"Unknown search backend: {kind}")
