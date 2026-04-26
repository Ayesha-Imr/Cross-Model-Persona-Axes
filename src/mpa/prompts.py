from __future__ import annotations

import hashlib
import logging
import random

from .config import PromptsCfg

log = logging.getLogger(__name__)


def _hash_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def sample_wildchat(cfg: PromptsCfg, seed: int) -> list[dict]:
    """Sample N single-turn English prompts, topic de-skewed.

    Uses a streaming pull from WildChat-1M. Filters to single-turn English with
    a token-length window, then clusters via TF-IDF + KMeans to spread topics.
    """
    from datasets import load_dataset
    rng = random.Random(seed)

    log.info("Loading %s (streaming)...", cfg.source)
    ds = load_dataset(cfg.source, split="train", streaming=True)

    pool: list[str] = []
    target_pool = max(cfg.n * 50, 1000)
    for ex in ds:
        if ex.get("language") and ex["language"] != "English":
            continue
        if ex.get("turn") != 1:
            continue
        conv = ex.get("conversation") or []
        if not conv or conv[0].get("role") != "user":
            continue
        text = (conv[0].get("content") or "").strip()
        n_tok = len(text.split())
        if not (cfg.min_tokens <= n_tok <= cfg.max_tokens):
            continue
        pool.append(text)
        if len(pool) >= target_pool:
            break
    log.info("Pool size: %d", len(pool))

    chosen = _topic_deskew(pool, cfg.n, cfg.n_topic_clusters, rng)
    return [{"id": _hash_id(t), "text": t, "topic_cluster": c} for c, t in chosen]


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
