from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class GenParams(BaseModel):
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 512
    thinking_budget: int = 0  # google: cap reasoning tokens; 0 disables thinking


class ModelEntry(BaseModel):
    name: str
    provider: Literal["openai", "anthropic", "google", "cohere", "hf_local"]
    model_id: str
    family: str = "other"
    enabled: bool = True
    gen: GenParams = Field(default_factory=GenParams)


class AxisEntry(BaseModel):
    name: str
    trait_noun: str
    pos_system: str
    neg_system: str
    pos_threshold: float = 70.0
    neg_threshold: float = 30.0
    n_candidates: int = 50  # per polarity, before filter


class ProberCfg(BaseModel):
    model_id: str = "google/gemma-4-e4b-it"
    backend: Literal["hf", "vllm_lens"] = "hf"
    dtype: str = "bfloat16"
    batch_size: int = 8
    max_response_tokens: int = 512


class JudgeCfg(BaseModel):
    model: str = "gpt-4.1-mini-2025-04-14"
    max_workers: int = 20
    min_prob: float = 0.25


class PromptsCfg(BaseModel):
    source: str = "allenai/WildChat-1M"
    n: int = 50
    min_tokens: int = 30
    max_tokens: int = 300
    n_topic_clusters: int = 25


class ContrastiveCfg(BaseModel):
    n_seed_prompts: int = 25
    seed_prompts: list[str] = Field(default_factory=list)


class PricingEntry(BaseModel):
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0


class Config(BaseModel):
    seed: int = 0
    output_root: Path = Path("runs")
    cost_cap_usd: float = 25.0
    prober: ProberCfg = Field(default_factory=ProberCfg)
    judge: JudgeCfg = Field(default_factory=JudgeCfg)
    prompts: PromptsCfg = Field(default_factory=PromptsCfg)
    contrastive: ContrastiveCfg = Field(default_factory=ContrastiveCfg)
    axes: list[AxisEntry]
    models: list[ModelEntry]
    pricing: dict[str, PricingEntry] = Field(default_factory=dict)

    def enabled_models(self) -> list[ModelEntry]:
        return [m for m in self.models if m.enabled]

    def short_hash(self) -> str:
        """Stable hash — only fields where a change invalidates *all* artifacts.

        Excludes axes (incremental) and models (additive), so adding an axis
        or toggling a model reuses the same run dir.
        """
        import json
        payload = {
            "seed": self.seed,
            "prober": self.prober.model_dump(),
            "judge": self.judge.model_dump(),
            "prompts": self.prompts.model_dump(),
            "contrastive": self.contrastive.model_dump(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
