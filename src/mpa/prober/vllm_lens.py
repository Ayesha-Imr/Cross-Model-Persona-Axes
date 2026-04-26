from __future__ import annotations

import gc

import torch

from ..config import ProberCfg


class VLLMLensProber:
    """vLLM-Lens prober: extracts residual-stream activations via vLLM plugin.

    Forwards each text with a 1-token completion and `output_residual_stream`
    enabled for all layers, then mean-pools tokens within the prompt span.
    Falls back path is handled in `prober.__init__`.
    """

    def __init__(self, cfg: ProberCfg):
        from vllm import LLM
        import vllm_lens  # noqa: F401  -- triggers plugin registration
        self.cfg = cfg
        self.llm = LLM(model=cfg.model_id, dtype=cfg.dtype, trust_remote_code=True)
        hf_cfg = self.llm.llm_engine.model_config.hf_config
        self.hidden_dim = getattr(hf_cfg, "hidden_size", None) or hf_cfg.d_model
        self.num_layers = (getattr(hf_cfg, "num_hidden_layers", None) or hf_cfg.num_layers) + 1
        self.layer_indices = list(range(self.num_layers))

    def encode(self, texts: list[str]) -> torch.Tensor:
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=0.0, max_tokens=1,
            extra_args={"output_residual_stream": self.layer_indices},
        )
        outs = self.llm.generate(texts, sp, use_tqdm=False)
        pooled = []
        for o in outs:
            acts = o.outputs[0].metadata["activations"]["residual_stream"]  # {layer: tensor[T,H]}
            layer_means = []
            for li in self.layer_indices:
                t = acts[li]
                if not isinstance(t, torch.Tensor):
                    t = torch.as_tensor(t)
                layer_means.append(t.float().mean(dim=0))
            pooled.append(torch.stack(layer_means, dim=0))
        return torch.stack(pooled, dim=0)  # (n, L, H)

    def close(self) -> None:
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
