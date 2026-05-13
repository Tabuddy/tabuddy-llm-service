"""Stage 3 sub-stage 3a orchestrator.

Given a focal role's Stage 2 candidate dimensions and the approved Stage 2
dimensions of its adjacent roles, embed everything and return the list of
overlapping pairs that involve the focal role. Pairs entirely between two
non-focal adjacent roles are skipped — those overlaps are for the
adjacent roles' own Stage 3 runs to handle.

Each output pair carries:
  - a_role / b_role         (display names)
  - a_tentative_id / b_tentative_id
  - a_name / b_name         (dim display names, for the review UI)
  - similarity              (cosine similarity, > threshold)
  - kind                    ("intra_role" or "cross_role")
"""

from __future__ import annotations

import re

from skill_library_v3.dim_embedder import DimensionEmbedder
from skill_library_v3.dim_similarity import _cosine


# Generic dim-name noise: tokens that show up in many product-family dims
# and would inflate Jaccard scores without indicating real overlap. Kept
# small and conservative — only words that are universally non-distinctive.
_TOKEN_STOPWORDS: frozenset[str] = frozenset({
    "and", "or", "the", "of", "for", "in", "to", "as", "with",
    "operations", "services", "service", "systems", "system",
    "tools", "tool", "platform", "platforms", "suite", "suites",
    "products", "product", "engineering", "frameworks", "framework",
})


def _tokenize(text: str | None) -> set[str]:
    """Split text into a set of distinguishing alphanumeric tokens.
    Lowercase, drops stopwords + single-letter tokens. Used for the
    intra-role Jaccard overlap rule."""
    if not text:
        return set()
    raw = re.split(r"[^A-Za-z0-9]+", text)
    out: set[str] = set()
    for tok in raw:
        t = tok.lower()
        if len(t) < 2:
            continue
        if t in _TOKEN_STOPWORDS:
            continue
        out.add(t)
    return out


def _compute_token_jaccard(dim_a: dict, dim_b: dict) -> float:
    """Jaccard similarity over the union of in_scope tokens and
    exemplar_skills tokens. Returns 0.0 if either side has no tokens."""
    a_tokens = _tokenize(dim_a.get("in_scope", ""))
    for sk in dim_a.get("exemplar_skills") or []:
        a_tokens |= _tokenize(sk)
    b_tokens = _tokenize(dim_b.get("in_scope", ""))
    for sk in dim_b.get("exemplar_skills") or []:
        b_tokens |= _tokenize(sk)
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0


async def compute_overlap_pairs(
    *,
    role_slug: str,
    role_display: str,
    role_dimensions: list[dict],
    adjacent_dimensions_by_role: dict[str, list[dict]],
    embedder: DimensionEmbedder,
    threshold: float = 0.65,
    intra_role_threshold: float | None = None,
    intra_role_jaccard_threshold: float = 0.30,
) -> list[dict]:
    """Embed every dim, return overlapping pairs that involve the focal
    role. ``role_slug`` and ``role_display`` describe the focal role for
    output labels. ``adjacent_dimensions_by_role`` is keyed by adjacent
    role display name.

    Thresholds:
      * ``threshold`` (default 0.65) — embedding-similarity gate for
        CROSS-role pairs. Strict inequality.
      * ``intra_role_threshold`` (default = ``threshold``) — embedding
        gate for INTRA-role pairs. Same as cross-role by default for
        backward compatibility; lower it to surface intra-role pairs
        that read as separate concepts but share textual content.
      * ``intra_role_jaccard_threshold`` (default 0.30) — token-Jaccard
        gate for INTRA-role pairs over normalized
        ``in_scope`` + ``exemplar_skills`` tokens. An intra-role pair
        flags if EITHER embedding OR Jaccard exceeds its threshold.

    Cross-role pairs ignore the Jaccard rule entirely — vendor product
    names legitimately recur across roles (Vertex AI in MLOps and Data
    Engineer is normal, not a reconciliation candidate).
    """
    # Intra-role pairs deserve a more sensitive embedding gate — dims
    # within the same role are *expected* to be conceptually distinct,
    # so any embedding proximity is a stronger reconciliation signal
    # than the cross-role baseline.
    intra_emb_threshold = (
        0.55 if intra_role_threshold is None else intra_role_threshold
    )
    # Tag each dim with its role + focal flag so we can skip
    # adjacent-vs-adjacent pairs cheaply.
    tagged: list[dict] = []
    for d in role_dimensions:
        tagged.append({"role": role_display, "dim": d, "is_focal": True})
    for adj_role, adj_dims in (adjacent_dimensions_by_role or {}).items():
        for d in (adj_dims or []):
            tagged.append({"role": adj_role, "dim": d, "is_focal": False})

    if not tagged:
        return []

    # Embed each unique dim text. The embedder caches identical texts so
    # this is at-most O(unique_dims) network calls.
    embeddings: list[list[float]] = []
    for entry in tagged:
        d = entry["dim"]
        emb = await embedder.embed(d.get("name", ""), d.get("description", ""))
        embeddings.append(emb)

    out: list[dict] = []
    for i, ti in enumerate(tagged):
        for j in range(i + 1, len(tagged)):
            tj = tagged[j]
            # Skip pairs entirely between non-focal adjacents.
            if not (ti["is_focal"] or tj["is_focal"]):
                continue
            sim = _cosine(embeddings[i], embeddings[j])
            kind = "intra_role" if ti["role"] == tj["role"] else "cross_role"

            if kind == "intra_role":
                jaccard = _compute_token_jaccard(ti["dim"], tj["dim"])
                emb_hit = sim > intra_emb_threshold
                jaccard_hit = jaccard > intra_role_jaccard_threshold
                if not (emb_hit or jaccard_hit):
                    continue
                if emb_hit and jaccard_hit:
                    match_reason = "both"
                elif emb_hit:
                    match_reason = "embedding"
                else:
                    match_reason = "token_jaccard"
            else:
                if sim <= threshold:
                    continue
                jaccard = _compute_token_jaccard(ti["dim"], tj["dim"])
                match_reason = "embedding"

            out.append(
                {
                    "a_role": ti["role"],
                    "a_tentative_id": ti["dim"]["tentative_id"],
                    "a_name": ti["dim"].get("name", ""),
                    "b_role": tj["role"],
                    "b_tentative_id": tj["dim"]["tentative_id"],
                    "b_name": tj["dim"].get("name", ""),
                    "similarity": sim,
                    "token_jaccard": jaccard,
                    "kind": kind,
                    "match_reason": match_reason,
                }
            )
    return out
