"""Run-file writers and validators for the two submissions.

Always check the final files with the official TREC RAG 2026 validator; the checks here
cover the rules listed in Table 1 of the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

from .generation.pipeline import Answer
from .retrieval import Doc
from .text import word_count
from .topics import Topic


def write_trec_run(results: dict[str, list[Doc]], path: str | Path, run_id: str) -> None:
    """Standard six-column TREC run: ``topic Q0 docid rank score run_id``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for tid, docs in results.items():
            n = len(docs)
            for rank, d in enumerate(docs, start=1):
                f.write(f"{tid} Q0 {d.docid} {rank} {n - rank + 1} {run_id}\n")


def answer_record(ans: Answer, run_id: str, team_id: str) -> dict:
    return {
        "metadata": {"team_id": team_id, "run_id": run_id, "narrative_id": ans.topic_id,
                     "narrative": ans.narrative, "type": "automatic"},
        "references": ans.references,
        "response_length": ans.length,
        "answer": ans.answer,
    }


def write_answers(answers: list[Answer], path: str | Path, run_id: str, team_id: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for a in answers:
            f.write(json.dumps(answer_record(a, run_id, team_id), ensure_ascii=False) + "\n")


def validate_answers(path: str | Path, topics: list[Topic], max_words: int = 1024,
                     max_citations: int = 3) -> list[str]:
    """Return a list of problems (empty when the file passes)."""
    expected = {t.topic_id: t.narrative for t in topics}
    seen, problems = set(), []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        rec = json.loads(line)
        meta = rec["metadata"]
        tid = meta["narrative_id"]
        seen.add(tid)
        if tid not in expected:
            problems.append(f"line {n}: unknown narrative_id {tid}")
            continue
        if meta["narrative"] != expected[tid]:
            problems.append(f"line {n}: narrative text for {tid} differs from the topic file")
        words = sum(word_count(a["text"]) for a in rec["answer"])
        if words > max_words:
            problems.append(f"line {n}: {words} words > {max_words}")
        for a in rec["answer"]:
            if len(a["citations"]) > max_citations:
                problems.append(f"line {n}: answer object with {len(a['citations'])} citations")
            if any(not 0 <= c < len(rec["references"]) for c in a["citations"]):
                problems.append(f"line {n}: citation index out of range")
    missing = set(expected) - seen
    if missing:
        problems.append(f"{len(missing)} narratives missing, e.g. {sorted(missing)[:3]}")
    return problems
