"""DMLabWin055 retrieval system."""

from dataclasses import dataclass, field


@dataclass
class Doc:
    docid: str
    text: str
    signals: dict[str, float] = field(default_factory=dict)
