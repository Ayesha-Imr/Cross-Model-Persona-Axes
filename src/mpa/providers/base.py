from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..config import ModelEntry


@dataclass
class GenResult:
    text: str
    in_tok: int = 0
    out_tok: int = 0


class Generator(Protocol):
    model: ModelEntry

    def generate(self, prompt: str, system: str | None = None) -> GenResult: ...

    def close(self) -> None: ...
