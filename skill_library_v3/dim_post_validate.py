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

from skill_library_v3.dim_apply import ApplyResult, _split_csv
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


def validate_merge_provenance(
    *,
    originals: list[dict],
    locked_dimensions: list[dict],
) -> list[dict]:
    """Verify each merged dim's in_scope + exemplar_skills are a strict
    subset of its source dims' content.

    Each merged dim carries `_merge_provenance.source_dim_ids` (set by
    ``apply_decisions``). We recompute the set of legitimate tokens from
    those sources and flag any merged-dim token that didn't originate
    there as content drift.

    Failure modes caught:
      * the merged dim absorbed content from a third dim (cascading
        merge that the LLM smuggled in via the merge-into description)
      * a future regression in ``apply_decisions`` that loses the strict-
        union invariant
      * a provenance row pointing at a source id that no longer exists
        in the candidate set (``unknown_source_dim``)
    """
    by_id: dict[str, dict] = {d.get("tentative_id"): d for d in originals if d.get("tentative_id")}
    errors: list[dict] = []

    for merged in locked_dimensions:
        prov = merged.get("_merge_provenance")
        if not prov:
            continue
        source_ids = list(prov.get("source_dim_ids") or [])
        if not source_ids:
            continue

        # Build the legitimate-token sets from valid sources only.
        allowed_in_scope: set[str] = set()
        allowed_exemplars: set[str] = set()
        for sid in source_ids:
            src = by_id.get(sid)
            if src is None:
                errors.append(
                    {
                        "level": "warning",
                        "code": "unknown_source_dim",
                        "dim_id": merged.get("tentative_id"),
                        "missing_source_dim_id": sid,
                        "message": (
                            f"merged dim {merged.get('tentative_id')!r} provenance "
                            f"references source id {sid!r} which is not in the "
                            "candidate_dimensions list"
                        ),
                    }
                )
                continue
            for tok in _split_csv(src.get("in_scope")):
                allowed_in_scope.add(tok.lower())
            for sk in src.get("exemplar_skills") or []:
                if isinstance(sk, str) and sk.strip():
                    allowed_exemplars.add(sk.strip().lower())

        # Walk the merged dim's content and find tokens that aren't allowed.
        foreign_in_scope: list[str] = []
        for tok in _split_csv(merged.get("in_scope")):
            if tok.lower() not in allowed_in_scope:
                foreign_in_scope.append(tok)
        foreign_exemplars: list[str] = []
        for sk in merged.get("exemplar_skills") or []:
            if not isinstance(sk, str) or not sk.strip():
                continue
            if sk.strip().lower() not in allowed_exemplars:
                foreign_exemplars.append(sk.strip())

        if foreign_in_scope or foreign_exemplars:
            errors.append(
                {
                    "level": "error",
                    "code": "merge_content_drift",
                    "dim_id": merged.get("tentative_id"),
                    "dim_name": merged.get("name"),
                    "source_dim_ids": source_ids,
                    "foreign_in_scope": foreign_in_scope,
                    "foreign_exemplars": foreign_exemplars,
                    "message": (
                        f"merged dim {merged.get('name')!r} "
                        f"({merged.get('tentative_id')!r}) contains content not "
                        f"present in any source dim: "
                        f"in_scope={foreign_in_scope!r}, exemplars={foreign_exemplars!r}"
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
