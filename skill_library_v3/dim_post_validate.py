"""Stage 3c post-validators.

Two checks per the blueprint, run after :func:`apply_decisions` produces
a candidate ``ApplyResult``:

  * :func:`validate_exemplar_coverage` — every exemplar from the
    originals must have a home: either in some locked dim's
    ``exemplar_skills``, or in ``unassigned_exemplars`` (the
    Stage-4/5-handoff bucket for SPLITs that drop them by design).

  * :func:`validate_no_near_duplicates` — no two locked dims should
    have name+description embedding cosine similarity > 0.85. If 3b's
    decisions left two near-duplicates in the locked list, the human
    reviewer needs to see that flag.

Both return ``[]`` on success and a list of error dicts on violation,
matching the existing validator-log shape used by Stages 0/1/2.
"""

from __future__ import annotations

from skill_library_v3.dim_apply import ApplyResult
from skill_library_v3.dim_embedder import DimensionEmbedder
from skill_library_v3.dim_similarity import _cosine


def _all_exemplars_lower(dims: list[dict]) -> set[str]:
    out: set[str] = set()
    for d in dims:
        for s in d.get("exemplar_skills") or []:
            if not isinstance(s, str):
                continue
            key = s.strip().lower()
            if key:
                out.add(key)
    return out


def validate_exemplar_coverage(
    *,
    originals: list[dict],
    result: ApplyResult,
) -> list[dict]:
    """Every exemplar from ``originals`` must appear (case-insensitively)
    in some locked dim or in ``unassigned_exemplars``. Anything missing is
    surfaced as an ``exemplar_missing`` error.
    """
    locked_lc = _all_exemplars_lower(result.locked_dimensions)
    unassigned_lc = {s.strip().lower() for s in result.unassigned_exemplars if s and s.strip()}
    available = locked_lc | unassigned_lc

    errors: list[dict] = []
    seen_originals: set[str] = set()
    for d in originals:
        for skill in d.get("exemplar_skills") or []:
            if not isinstance(skill, str) or not skill.strip():
                continue
            key = skill.strip().lower()
            if key in seen_originals:
                continue
            seen_originals.add(key)
            if key not in available:
                errors.append(
                    {
                        "level": "error",
                        "code": "exemplar_missing",
                        "skill": skill,
                        "message": (
                            f"exemplar skill {skill!r} from original dim "
                            f"{d.get('tentative_id', '?')} is not in any locked dim "
                            "and not in unassigned_exemplars"
                        ),
                    }
                )
    return errors


async def validate_no_near_duplicates(
    *,
    locked_dimensions: list[dict],
    embedder: DimensionEmbedder,
    threshold: float = 0.85,
) -> list[dict]:
    """Embed every locked dim's name+description and flag pairs whose
    cosine similarity STRICTLY exceeds ``threshold`` (default 0.85,
    per blueprint).
    """
    if len(locked_dimensions) < 2:
        return []

    embeddings: list[list[float]] = []
    for d in locked_dimensions:
        emb = await embedder.embed(d.get("name", ""), d.get("description", ""))
        embeddings.append(emb)

    errors: list[dict] = []
    for i in range(len(locked_dimensions)):
        for j in range(i + 1, len(locked_dimensions)):
            sim = _cosine(embeddings[i], embeddings[j])
            if sim <= threshold:
                continue
            errors.append(
                {
                    "level": "error",
                    "code": "near_duplicate_locked_dims",
                    "a_id": locked_dimensions[i].get("tentative_id"),
                    "a_name": locked_dimensions[i].get("name"),
                    "b_id": locked_dimensions[j].get("tentative_id"),
                    "b_name": locked_dimensions[j].get("name"),
                    "similarity": sim,
                    "message": (
                        f"locked dims {locked_dimensions[i].get('name')!r} and "
                        f"{locked_dimensions[j].get('name')!r} have cosine "
                        f"similarity {sim:.3f} > {threshold}; reconciliation "
                        "left near-duplicates"
                    ),
                }
            )
    return errors
