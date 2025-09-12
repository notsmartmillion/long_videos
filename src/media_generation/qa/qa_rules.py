"""QA rules (Phase 6 stub)

Provides similarity scoring and simple continuity/diversity checks. Real
implementation should use embeddings for similarity and face/appearance models
for continuity. This stub is deterministic and lightweight.
"""

from __future__ import annotations

from typing import List, Dict, Any


def cosine_like(a: str, b: str) -> float:
    # Very rough token overlap score [0..1]
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union


def passes_similarity(caption: str, beat_text: str, threshold: float) -> bool:
    return cosine_like(caption, beat_text) >= threshold


def check_diversity(shot_types: List[str], window: int = 8, max_ratio: float = 0.4) -> bool:
    # Flag if any shot type dominates more than max_ratio in a sliding window
    n = len(shot_types)
    if n == 0:
        return True
    for i in range(n):
        w = shot_types[i:i+window]
        if not w:
            continue
        counts = {}
        for s in w:
            counts[s] = counts.get(s, 0) + 1
        if max(counts.values()) / len(w) > max_ratio:
            return False
    return True


