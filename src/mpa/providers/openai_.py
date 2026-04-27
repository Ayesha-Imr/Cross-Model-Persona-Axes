from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelEntry
from .base import GenResult


class OpenAIGen:
    def __init__(self, model: ModelEntry):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def generate(self, prompt: str, system: str | None = None) -> GenResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = self.client.chat.completions.create(
            model=self.model.model_id,
            messages=messages,
            top_p=self.model.gen.top_p,
            max_completion_tokens=self.model.gen.max_tokens,
        )
        text = r.choices[0].message.content or ""
        u = r.usage
        return GenResult(text=text, in_tok=u.prompt_tokens, out_tok=u.completion_tokens)

    def close(self) -> None:
        pass
