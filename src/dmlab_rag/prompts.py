"""Load the prompt templates from ``prompts/`` (verbatim from Appendix A.3 of the paper)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


@lru_cache(maxsize=None)
def template(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")


def render(name: str, **fields) -> str:
    return template(name).format(**fields)
