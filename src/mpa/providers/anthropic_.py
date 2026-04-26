from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelEntry
from .base import GenResult


class AnthropicGen:
    def __init__(self, model: ModelEntry):
        from anthropic import Anthropic
        self.model = model
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def generate(self, prompt: str, system: str | None = None) -> GenResult:
        kwargs = dict(
            model=self.model.model_id,
            max_tokens=self.model.gen.max_tokens,
            temperature=self.model.gen.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        r = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return GenResult(text=text, in_tok=r.usage.input_tokens, out_tok=r.usage.output_tokens)

    def close(self) -> None:
        pass
