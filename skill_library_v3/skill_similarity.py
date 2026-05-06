"""Stage 6 input prep: per-skill top-K similar skills via embedding.

Pure async function (with embedder injected). Each role's Stage 6 run
calls this once per skill to build the LLM's "candidate similar skills"
context — typically K=20 per blueprint.

Empty candidate list returns ``[]``. K=0 returns ``[]``. K larger than
the candidate count returns everything available, no padding. The
target itself is filtered out if it appears in candidates.
"""

from __future__ import annotations

from skill_library_v3.dim_embedder import DimensionEmbedder
from skill_library_v3.dim_similarity import _cosine


async def top_k_similar_skills(
    *,
    target: dict,
    candidates: list[dict],
    embedder: DimensionEmbedder,
    k: int = 20,
) -> list[dict]:
    """Return up to ``k`` candidates ordered by descending cosine
    similarity to ``target``. Target ``skill_id`` is filtered out of
    candidates. Each result is ``{skill_id, name, similarity}``."""
    if k <= 0 or not candidates:
        return []
    target_emb = await embedder.embed(target.get("name", ""), None)
    scored: list[dict] = []
    for c in candidates:
        if c.get("skill_id") == target.get("skill_id"):
            continue
        cand_emb = await embedder.embed(c.get("name", ""), None)
        sim = _cosine(target_emb, cand_emb)
        scored.append(
            {
                "skill_id": c.get("skill_id"),
                "name": c.get("name"),
                "similarity": sim,
            }
        )
    scored.sort(key=lambda r: r["similarity"], reverse=True)
    return scored[:k]
