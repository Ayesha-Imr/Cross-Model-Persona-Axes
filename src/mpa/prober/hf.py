from __future__ import annotations

import gc

import torch
from transformers import AutoModel, AutoTokenizer

from ..config import ProberCfg


_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class HFProber:
    """Mean-pool per-layer hidden states for each input text via output_hidden_states."""

    def __init__(self, cfg: ProberCfg):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            cfg.model_id,
            torch_dtype=_DTYPES[cfg.dtype],
            device_map="auto",
            output_hidden_states=True,
        ).eval()
        cfg_obj = self.model.config
        self.hidden_dim = getattr(cfg_obj, "hidden_size", None) or getattr(cfg_obj, "d_model")
        # +1 because hidden_states includes embedding output
        self.num_layers = (getattr(cfg_obj, "num_hidden_layers", None)
                           or getattr(cfg_obj, "num_layers")) + 1

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        out: list[torch.Tensor] = []
        bs = self.cfg.batch_size
        device = next(self.model.parameters()).device
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_response_tokens,
            ).to(device)
            res = self.model(**enc, output_hidden_states=True, return_dict=True)
            hs = torch.stack(res.hidden_states, dim=1)  # (B, L, T, H)
            mask = enc["attention_mask"].unsqueeze(1).unsqueeze(-1).to(hs.dtype)  # (B, 1, T, 1)
            pooled = (hs * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)  # (B, L, H)
            out.append(pooled.float().cpu())
        return torch.cat(out, dim=0)

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
