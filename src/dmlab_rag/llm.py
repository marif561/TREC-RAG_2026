"""Thin chat client for OpenAI-compatible endpoints, with a disk cache and JSON parsing.

Each role in ``configs/*.yaml`` (planner, grader, umbrela_qwen, ...) names a model and,
optionally, a ``base_url``. Hosted models use the OpenAI API; open-weight judges such as
Qwen3.5-9B and Ministral-3-14B can be served locally with vLLM, which exposes the same API.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Protocol

JSON_BLOCK = re.compile(r"\{.*\}", re.S)


class ChatModel(Protocol):
    def __call__(self, prompt: str, *, system: str | None = None) -> str: ...


def parse_json(text: str) -> dict:
    """Parse the first JSON object in a model reply (tolerates code fences and preambles)."""
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = JSON_BLOCK.search(text)
        if not m:
            raise ValueError(f"No JSON object in model output: {text[:200]!r}")
        return json.loads(m.group(0))


class OpenAIChat:
    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 temperature: float = 0.0, max_tokens: int = 4096, cache_dir: str | None = ".cache/llm"):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "EMPTY"))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = Path(cache_dir) / model.replace("/", "_") if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def __call__(self, prompt: str, *, system: str | None = None) -> str:
        key = hashlib.sha256(f"{system}\x00{prompt}".encode()).hexdigest()
        path = self.cache / f"{key}.txt" if self.cache else None
        if path and path.exists():
            return path.read_text(encoding="utf-8")
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        out = resp.choices[0].message.content or ""
        if path:
            path.write_text(out, encoding="utf-8")
        return out


def build_models(cfg: dict) -> dict[str, ChatModel]:
    """Instantiate one client per role in ``cfg['models']``."""
    cache = cfg.get("cache_dir", ".cache/llm")
    return {
        role: OpenAIChat(spec["model"], spec.get("base_url"), spec.get("api_key_env", "OPENAI_API_KEY"),
                         spec.get("temperature", 0.0), spec.get("max_tokens", 4096), cache)
        for role, spec in cfg["models"].items()
    }
