from __future__ import annotations

from ..config import ProberCfg
from .base import Prober


def make_prober(cfg: ProberCfg) -> Prober:
    if cfg.backend == "vllm_lens":
        try:
            from .vllm_lens import VLLMLensProber
            return VLLMLensProber(cfg)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "vllm_lens backend failed (%s); falling back to HF", e,
            )
    from .hf import HFProber
    return HFProber(cfg)
