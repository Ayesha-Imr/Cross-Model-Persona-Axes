from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .checkpoint import append_jsonl
from .config import Config


@dataclass
class CostTracker:
    cap_usd: float
    spent_usd: float = 0.0
    by_provider: dict[str, float] = field(default_factory=dict)
    log_path: Path | None = None

    def add(self, provider: str, in_tok: int, out_tok: int, cfg: Config) -> float:
        price = cfg.pricing.get(provider)
        if price is None:
            return 0.0
        cost = (in_tok / 1e6) * price.input_per_mtok + (out_tok / 1e6) * price.output_per_mtok
        self.spent_usd += cost
        self.by_provider[provider] = self.by_provider.get(provider, 0.0) + cost
        if self.log_path is not None:
            append_jsonl(self.log_path, {
                "provider": provider, "in_tok": in_tok, "out_tok": out_tok, "cost_usd": cost,
            })
        return cost

    def check_cap(self) -> None:
        if self.spent_usd > self.cap_usd:
            raise RuntimeError(
                f"Cost cap exceeded: ${self.spent_usd:.2f} > ${self.cap_usd:.2f}. "
                f"Raise `cost_cap_usd` and rerun (artifacts persist)."
            )

    def summary(self) -> str:
        parts = [f"{p}=${v:.3f}" for p, v in sorted(self.by_provider.items())]
        return f"total=${self.spent_usd:.3f} ({', '.join(parts)})"
