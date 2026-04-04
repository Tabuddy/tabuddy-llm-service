"""Semantic similarity matching for skill/capability comparison.

Uses SentenceTransformer embeddings (all-MiniLM-L6-v2) to compute
cosine similarity between skill strings, catching contextually similar
matches that Jaccard token overlap misses.

Model is lazy-loaded and globally cached to avoid reloading per call.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazily load and cache the SentenceTransformer model."""
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Semantic matcher model loaded (384-dim embeddings)")
    except Exception as e:
        logger.warning("Failed to load semantic matching model: %s", e)
        _model = False  # sentinel to avoid re-loading
        return None
    return _model


def semantic_similarity(a: str, b: str) -> float:
    """Compute cosine similarity between two skill strings.

    Returns:
        float in [0.0, 1.0]. 0 if model unavailable.
    """
    model = _get_model()
    if model is None:
        return _cosine_fallback(a, b)

    emb_a = model.encode(a, normalize_embeddings=True)
    emb_b = model.encode(b, normalize_embeddings=True)
    return float(_cosine_sim(emb_a, emb_b))


def semantic_best_match(query: str, candidates: list[str]) -> float:
    """Find the cosine similarity between query and the best matching candidate.

    Batch-encodes all candidates for efficiency.
    Returns best score in [0.0, 1.0]. Returns 0.0 if model unavailable.
    """
    if not candidates:
        return 0.0

    model = _get_model()
    if model is None:
        return max(_cosine_fallback(query, c) for c in candidates)

    emb_q = model.encode(query, normalize_embeddings=True)
    emb_c = model.encode(candidates, normalize_embeddings=True)
    sims = emb_c @ emb_q  # pre-normalized → dot product = cosine sim

    return float(max(sims))  # type: ignore[arg-type]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (handles 1-d and 2-d)."""
    sim = a @ b
    return max(0.0, float(sim))  # clamp to [0, 1]


def _cosine_fallback(a: str, b: str) -> float:
    """Word-overlap fallback when model is unavailable."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
