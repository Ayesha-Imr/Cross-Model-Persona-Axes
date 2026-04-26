from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelEntry
from .base import GenResult


class CohereGen:
    def __init__(self, model: ModelEntry):
        import cohere
        self.model = model
        self.client = cohere.ClientV2(api_key=os.environ.get("COHERE_API_KEY"))

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def generate(self, prompt: str, system: str | None = None) -> GenResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = self.client.chat(
            model=self.model.model_id,
            messages=messages,
            temperature=self.model.gen.temperature,
            max_tokens=self.model.gen.max_tokens,
        )
        text = "".join(c.text for c in r.message.content if getattr(c, "type", "") == "text")
        u = getattr(r, "usage", None)
        in_tok = getattr(u.tokens, "input_tokens", 0) if u and u.tokens else 0
        out_tok = getattr(u.tokens, "output_tokens", 0) if u and u.tokens else 0
        return GenResult(text=text, in_tok=int(in_tok or 0), out_tok=int(out_tok or 0))

    def close(self) -> None:
        pass
