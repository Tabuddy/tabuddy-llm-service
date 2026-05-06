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

from skill_library_v3.dim_embedder import DimensionEmbedder
from skill_library_v3.dim_similarity import _cosine


async def compute_overlap_pairs(
    *,
    role_slug: str,
    role_display: str,
    role_dimensions: list[dict],
    adjacent_dimensions_by_role: dict[str, list[dict]],
    embedder: DimensionEmbedder,
    threshold: float = 0.65,
) -> list[dict]:
    """Embed every dim, return overlapping pairs that involve the focal
    role. ``role_slug`` and ``role_display`` describe the focal role for
    output labels. ``adjacent_dimensions_by_role`` is keyed by adjacent
    role display name.
    """
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
            if sim <= threshold:
                continue
            kind = "intra_role" if ti["role"] == tj["role"] else "cross_role"
            out.append(
                {
                    "a_role": ti["role"],
                    "a_tentative_id": ti["dim"]["tentative_id"],
                    "a_name": ti["dim"].get("name", ""),
                    "b_role": tj["role"],
                    "b_tentative_id": tj["dim"]["tentative_id"],
                    "b_name": tj["dim"].get("name", ""),
                    "similarity": sim,
                    "kind": kind,
                }
            )
    return out
