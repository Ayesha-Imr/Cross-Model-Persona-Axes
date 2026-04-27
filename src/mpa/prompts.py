from __future__ import annotations

import hashlib
import logging
import random

from .config import PromptsCfg

log = logging.getLogger(__name__)


def _hash_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def sample_prompts(cfg: PromptsCfg, seed: int) -> list[dict]:
    """Sample N first-turn English prompts, topic de-skewed.

    Streams from UltraChat (`openbmb/UltraChat`); takes the first user message
    of each conversation, filters by token-length window, then de-skews topics
    via TF-IDF + KMeans.
    """
    from datasets import load_dataset
    rng = random.Random(seed)

    log.info("Loading %s (streaming)...", cfg.source)
    ds = load_dataset(cfg.source, split="train", streaming=True)

    pool: list[str] = []
    target_pool = max(cfg.n * 50, 1000)
    for ex in ds:
        text = _first_user_message(ex)
        if not text:
            continue
        n_tok = len(text.split())
        if not (cfg.min_tokens <= n_tok <= cfg.max_tokens):
            continue
        pool.append(text)
        if len(pool) >= target_pool:
            break
    log.info("Pool size: %d", len(pool))

    chosen = _topic_deskew(pool, cfg.n, cfg.n_topic_clusters, rng)
    return [{"id": _hash_id(t), "text": t, "topic_cluster": c} for c, t in chosen]


def _first_user_message(ex: dict) -> str | None:
    # UltraChat: {"id": ..., "data": ["user msg", "asst msg", ...]}
    data = ex.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            return (first.get("content") or "").strip() or None
    # Fallback for chat-style schemas (e.g. WildChat-shaped)
    conv = ex.get("conversation") or ex.get("messages")
    if isinstance(conv, list) and conv and isinstance(conv[0], dict):
        if conv[0].get("role") == "user":
            return (conv[0].get("content") or "").strip() or None
    return None


def _topic_deskew(pool: list[str], n: int, n_clusters: int, rng: random.Random) -> list[tuple[int, str]]:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    if len(pool) <= n:
        return [(0, t) for t in pool]
    vec = TfidfVectorizer(max_features=2048, stop_words="english")
    X = vec.fit_transform(pool)
    k = min(n_clusters, len(pool))
    km = KMeans(n_clusters=k, random_state=rng.randint(0, 2**31 - 1), n_init=4).fit(X)
    by_cluster: dict[int, list[str]] = {}
    for label, txt in zip(km.labels_, pool):
        by_cluster.setdefault(int(label), []).append(txt)
    chosen: list[tuple[int, str]] = []
    cluster_ids = list(by_cluster.keys())
    rng.shuffle(cluster_ids)
    while len(chosen) < n:
        progress = False
        for c in cluster_ids:
            if not by_cluster[c]:
                continue
            t = by_cluster[c].pop(rng.randrange(len(by_cluster[c])))
            chosen.append((c, t))
            progress = True
            if len(chosen) >= n:
                break
        if not progress:
            break
    return chosen
