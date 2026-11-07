"""Sub-narrative generation (Section 3.3): two independent decompositions, merged.

Each decomposition has six to ten atomic facets marked "vital" or "okay". Few-shot style
examples come from official TREC 2025 RAG narratives (``data/subnarrative_examples.json``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..llm import ChatModel, parse_json
from ..prompts import render
from ..text import content_terms, jaccard


@dataclass(frozen=True)
class SubNarrative:
    text: str
    importance: str = "vital"


def format_examples(path: str | Path | None) -> str:
    if not path or not Path(path).exists():
        return ""
    examples = json.loads(Path(path).read_text(encoding="utf-8"))
    blocks = []
    for ex in examples:
        blocks.append("Example narrative:\n" + ex["narrative"] + "\nExample output:\n" +
                      json.dumps({"sub_narratives": ex["sub_narratives"]}, ensure_ascii=False))
    return "\n\n".join(blocks)


def merge_decompositions(decomps: list[list[SubNarrative]], dup_threshold: float = 0.6) -> list[SubNarrative]:
    """Union of decompositions; a facet whose content terms overlap an earlier one at
    Jaccard ≥ ``dup_threshold`` is merged into it ("vital" wins over "okay")."""
    merged: list[SubNarrative] = []
    terms: list[set[str]] = []
    for decomp in decomps:
        for sn in decomp:
            t = content_terms(sn.text)
            j = next((i for i, other in enumerate(terms) if jaccard(t, other) >= dup_threshold), None)
            if j is None:
                merged.append(sn)
                terms.append(t)
            elif sn.importance == "vital" and merged[j].importance != "vital":
                merged[j] = SubNarrative(merged[j].text, "vital")
    return merged


def generate_subnarratives(narrative: str, llm: ChatModel, n_decompositions: int = 2,
                           examples_path: str | Path | None = None) -> list[SubNarrative]:
    examples = format_examples(examples_path)
    decomps = []
    for i in range(n_decompositions):
        # A short suffix makes the cached prompts distinct so the decompositions are independent.
        prompt = render("sub_narratives", examples=examples, narrative=narrative)
        if i:
            prompt += f"\n(Independent decomposition #{i + 1}.)"
        rows = parse_json(llm(prompt)).get("sub_narratives", [])
        decomps.append([SubNarrative(r["text"].strip(), r.get("importance", "vital")) for r in rows
                        if r.get("text", "").strip()])
    return merge_decompositions(decomps)


def as_bullets(subs: list[SubNarrative]) -> str:
    return "\n".join(f"- [{s.importance}] {s.text}" for s in subs)
