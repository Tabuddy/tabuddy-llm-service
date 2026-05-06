"""Deterministic similarity flagging for Stage 3 sub-stage 3a.

Given a list of dim dicts each carrying a pre-computed embedding, return
the unordered pairs whose cosine similarity exceeds a threshold (default
0.65 per the blueprint). The result feeds Stage 3 sub-stage 3b's LLM
reconciler, which decides MERGE / SPLIT / KEEP_SEPARATE per pair.

Pure math + data manipulation. No LLM, no DB, no embeddings API call.
The embedding generation lives in a separate module (TBD next).
"""

from __future__ import annotations

import math
from typing import Iterable


def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
    """Cosine similarity of two equal-length vectors. Returns 0.0 when
    either side has zero norm (degenerate / not-yet-computed embedding)
    rather than raising ZeroDivisionError."""
    a_list = list(a)
    b_list = list(b)
    dot = sum(x * y for x, y in zip(a_list, b_list))
    norm_a = math.sqrt(sum(x * x for x in a_list))
    norm_b = math.sqrt(sum(x * x for x in b_list))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def flag_overlapping_pairs(
    dims: list[dict],
    *,
    threshold: float = 0.65,
) -> list[dict]:
    """Return the unordered pairs of ``dims`` whose embedding cosine
    similarity is STRICTLY greater than ``threshold``.

    Each ``dim`` is a dict with at least ``tentative_id`` and ``embedding``
    keys. If a ``role`` key is present, the resulting pair carries
    ``a_role`` and ``b_role`` so downstream callers (Stage 3b) can tell
    intra-role overlap from cross-role.

    Each unordered pair appears once; self-pairs are excluded.
    """
    out: list[dict] = []
    for i, a in enumerate(dims):
        a_emb = a.get("embedding") or []
        for b in dims[i + 1:]:
            b_emb = b.get("embedding") or []
            sim = _cosine(a_emb, b_emb)
            if sim <= threshold:
                continue
            pair: dict = {
                "a_id": a["tentative_id"],
                "b_id": b["tentative_id"],
                "similarity": sim,
            }
            if "role" in a or "role" in b:
                pair["a_role"] = a.get("role")
                pair["b_role"] = b.get("role")
            out.append(pair)
    return out
