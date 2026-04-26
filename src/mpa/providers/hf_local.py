from __future__ import annotations

import gc
from typing import Any

from ..config import ModelEntry
from .base import GenResult


class HFLocalGen:
    """Local HF model generator. Uses vLLM if available (fast), else transformers.

    Holds one model in GPU at a time; call `close()` to free before loading the next.
    """

    def __init__(self, model: ModelEntry):
        self.model = model
        self._engine: Any | None = None
        self._backend: str | None = None
        self._tok = None

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        try:
            from vllm import LLM
            self._engine = LLM(model=self.model.model_id, dtype="bfloat16", trust_remote_code=True)
            self._backend = "vllm"
        except Exception:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model.model_id)
            self._engine = AutoModelForCausalLM.from_pretrained(
                self.model.model_id, torch_dtype=torch.bfloat16, device_map="auto",
            )
            self._backend = "hf"

    def generate(self, prompt: str, system: str | None = None) -> GenResult:
        self._ensure_loaded()
        if self._backend == "vllm":
            return self._gen_vllm(prompt, system)
        return self._gen_hf(prompt, system)

    def _gen_vllm(self, prompt: str, system: str | None) -> GenResult:
        from vllm import SamplingParams
        tok = self._engine.get_tokenizer()
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        sp = SamplingParams(
            temperature=self.model.gen.temperature,
            top_p=self.model.gen.top_p,
            max_tokens=self.model.gen.max_tokens,
        )
        out = self._engine.generate([text], sp, use_tqdm=False)[0]
        in_tok = len(out.prompt_token_ids)
        gen = out.outputs[0]
        return GenResult(text=gen.text, in_tok=in_tok, out_tok=len(gen.token_ids))

    def _gen_hf(self, prompt: str, system: str | None) -> GenResult:
        import torch
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        inputs = self._tok.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True,
        ).to(self._engine.device)
        with torch.no_grad():
            out = self._engine.generate(
                inputs,
                max_new_tokens=self.model.gen.max_tokens,
                temperature=self.model.gen.temperature,
                top_p=self.model.gen.top_p,
                do_sample=self.model.gen.temperature > 0,
                pad_token_id=self._tok.eos_token_id,
            )
        gen_ids = out[0, inputs.shape[1]:]
        text = self._tok.decode(gen_ids, skip_special_tokens=True)
        return GenResult(text=text, in_tok=int(inputs.shape[1]), out_tok=int(gen_ids.shape[0]))

    def close(self) -> None:
        if self._engine is None:
            return
        try:
            import torch
            if self._backend == "hf":
                del self._engine
                del self._tok
            else:
                del self._engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            self._engine = None
            self._tok = None
            self._backend = None
