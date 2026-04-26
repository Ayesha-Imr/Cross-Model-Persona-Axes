from __future__ import annotations

import gc
import logging
import os
from typing import Any

from ..config import ModelEntry
from .base import GenResult

log = logging.getLogger(__name__)


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
        force_hf = os.environ.get("MPA_FORCE_HF", "0") == "1"
        if not force_hf:
            try:
                from vllm import LLM
                log.info("[%s] loading via vLLM (enforce_eager=True)", self.model.name)
                self._engine = LLM(
                    model=self.model.model_id,
                    dtype="bfloat16",
                    trust_remote_code=True,
                    enforce_eager=True,
                    max_model_len=int(os.environ.get("MPA_VLLM_MAX_LEN", "4096")),
                    gpu_memory_utilization=float(
                        os.environ.get("MPA_VLLM_GPU_UTIL", "0.85")),
                )
                self._backend = "vllm"
                return
            except ImportError:
                log.warning("[%s] vLLM not installed; using transformers", self.model.name)
            except Exception as e:
                log.warning("[%s] vLLM load failed (%s: %s); falling back to transformers. "
                            "Set MPA_FORCE_HF=1 to skip the vLLM attempt.",
                            self.model.name, type(e).__name__, e)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log.info("[%s] loading via transformers", self.model.name)
        self._tok = AutoTokenizer.from_pretrained(self.model.model_id)
        self._engine = AutoModelForCausalLM.from_pretrained(
            self.model.model_id, dtype=torch.bfloat16, device_map="auto",
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
            msgs,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        ).to(self._engine.device)
        in_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            out = self._engine.generate(
                **inputs,
                max_new_tokens=self.model.gen.max_tokens,
                temperature=self.model.gen.temperature,
                top_p=self.model.gen.top_p,
                do_sample=self.model.gen.temperature > 0,
                pad_token_id=self._tok.eos_token_id,
            )
        gen_ids = out[0, in_len:]
        text = self._tok.decode(gen_ids, skip_special_tokens=True)
        return GenResult(text=text, in_tok=in_len, out_tok=int(gen_ids.shape[0]))

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
