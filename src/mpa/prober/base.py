from __future__ import annotations

from typing import Protocol

import torch


class Prober(Protocol):
    num_layers: int
    hidden_dim: int

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Return tensor of shape (n, num_layers, hidden_dim) — mean-pooled per layer."""
        ...

    def close(self) -> None: ...
