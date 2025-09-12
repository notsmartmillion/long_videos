import math
from typing import Optional

_embeddings_model = None


def _load_embeddings():  # optional
    global _embeddings_model
    if _embeddings_model is not None:
        return _embeddings_model
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _embeddings_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        _embeddings_model = None
    return _embeddings_model


def similarity_mode_from_config(cfg) -> str:
    try:
        sim = getattr(cfg, 'similarity', None)
        if sim and getattr(sim, 'mode', 'stub') == 'embeddings':
            try:
                import sentence_transformers  # type: ignore
                return 'embeddings'
            except Exception:
                return 'stub'
    except Exception:
        pass
    return 'stub'


def cosine_sim(a: str, b: str, mode: str = "stub") -> float:
    if mode == "embeddings":
        model = _load_embeddings()
        if model is not None:
            try:
                import numpy as np  # type: ignore
                va = model.encode([a])[0]
                vb = model.encode([b])[0]
                denom = (np.linalg.norm(va) * np.linalg.norm(vb))
                if denom == 0:
                    return 0.0
                return float(np.dot(va, vb) / denom)
            except Exception:
                pass
    # stub fallback: token overlap
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    denom = math.sqrt(len(ta) * len(tb))
    if denom == 0:
        return 0.0
    return inter / denom

# Topic-aware thresholds
TOPIC_THRESHOLDS = {
    "mythology": 0.70,
    "history": 0.68,
    "science": 0.60,
    "space": 0.62,
    "default": 0.62,
}


def get_threshold_for_topic(cfg, topic: str) -> float:
    try:
        base = float(getattr(getattr(cfg, 'qa', None), 'caption_similarity_threshold', 0.62))
    except Exception:
        base = 0.62
    return float(TOPIC_THRESHOLDS.get(topic, base))


