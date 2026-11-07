"""Topic (narrative) loading. IDs and text are preserved exactly, as the track requires."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Topic:
    topic_id: str
    narrative: str


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_topics(path: str | Path, expected_sha256_prefix: str | None = None) -> list[Topic]:
    """Read a JSONL topic file with ``id`` (or ``topic_id``) and ``narrative`` (or ``title``)."""
    if expected_sha256_prefix:
        digest = sha256_file(path)
        if not digest.startswith(expected_sha256_prefix):
            raise ValueError(f"Topic file hash {digest[:8]}... does not match {expected_sha256_prefix}")
    topics = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            tid = str(row.get("id", row.get("topic_id", row.get("narrative_id"))))
            text = row.get("narrative", row.get("title", row.get("text")))
            topics.append(Topic(tid, text))
    return topics
