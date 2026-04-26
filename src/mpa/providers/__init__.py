from __future__ import annotations

from ..config import ModelEntry
from .base import Generator


def make_generator(model: ModelEntry) -> Generator:
    p = model.provider
    if p == "openai":
        from .openai_ import OpenAIGen
        return OpenAIGen(model)
    if p == "anthropic":
        from .anthropic_ import AnthropicGen
        return AnthropicGen(model)
    if p == "google":
        from .google_ import GoogleGen
        return GoogleGen(model)
    if p == "cohere":
        from .cohere_ import CohereGen
        return CohereGen(model)
    if p == "hf_local":
        from .hf_local import HFLocalGen
        return HFLocalGen(model)
    raise ValueError(f"unknown provider: {p}")
