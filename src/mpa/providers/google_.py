from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelEntry
from .base import GenResult


class GoogleGen:
    def __init__(self, model: ModelEntry):
        from google import genai
        self.model = model
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def generate(self, prompt: str, system: str | None = None) -> GenResult:
        from google.genai import types
        cfg = types.GenerateContentConfig(
            temperature=self.model.gen.temperature,
            top_p=self.model.gen.top_p,
            max_output_tokens=self.model.gen.max_tokens,
            system_instruction=system,
        )
        r = self.client.models.generate_content(
            model=self.model.model_id, contents=prompt, config=cfg,
        )
        text = r.text or ""
        usage = getattr(r, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
        out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
        return GenResult(text=text, in_tok=in_tok, out_tok=out_tok)

    def close(self) -> None:
        pass
