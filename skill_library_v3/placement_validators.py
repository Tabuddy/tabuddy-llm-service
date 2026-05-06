"""Stage 5 deterministic placement validators.

Four checks per blueprint, all returning the warnings/errors dict shape
the rest of v3 uses:

  * embedding_cross_check    — does the LLM's primary_dimension agree
                                  with the embedding-nearest dim?
  * type_dim_consistency     — does the skill's TYPE belong in the
                                  primary dim's subject area? (heuristic)
  * cross_skill_consistency  — within a (type, subtype) group, do
                                  skills mostly share primary_dim?
                                  Outliers flagged.
  * secondary_dim_audit      — any secondary that's already implied by
                                  the skill's parent skills is
                                  redundant — strip and flag.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from skill_library_v3.dim_embedder import DimensionEmbedder
from skill_library_v3.dim_similarity import _cosine
from skill_library_v3.schemas.placement import PlacedSkill
from skill_library_v3.schemas.typology import TypedSkill


# ── 1. embedding cross-check ──────────────────────────────────────────────


async def embedding_cross_check(
    *,
    skill: PlacedSkill,
    dims: list[dict],
    embedder: DimensionEmbedder,
) -> list[dict]:
    """Compare the LLM's ``primary_dimension`` to the dim whose
    embedding is nearest to the skill's name. Mismatch -> warning.
    Per blueprint: every embedding mismatch goes to 100% human review.

    Two distinct findings:
      * ``primary_unknown`` — LLM picked a dim_id that's not in the
        role's dim set (LLM hallucination). Surfaced regardless of
        dim count.
      * ``embedding_mismatch`` — LLM's primary IS in the set but is
        not the embedding-nearest. Only meaningful with 2+ dims.
    """
    dim_ids = {d["tentative_id"] for d in dims}
    if skill.primary_dimension not in dim_ids:
        return [
            {
                "level": "warning",
                "code": "primary_unknown",
                "skill_id": skill.skill_id,
                "llm_primary": skill.primary_dimension,
                "message": (
                    f"LLM picked primary_dimension={skill.primary_dimension!r} "
                    "which is not in the role's locked dim set"
                ),
            }
        ]
    if len(dims) < 2:
        return []

    skill_emb = await embedder.embed(skill.name, None)
    sims_by_id: dict[str, float] = {}
    for d in dims:
        dim_emb = await embedder.embed(d.get("name", ""), d.get("description", ""))
        sims_by_id[d["tentative_id"]] = _cosine(skill_emb, dim_emb)

    nearest_id = max(sims_by_id, key=sims_by_id.get)

    if skill.primary_dimension != nearest_id:
        return [
            {
                "level": "warning",
                "code": "embedding_mismatch",
                "skill_id": skill.skill_id,
                "llm_primary": skill.primary_dimension,
                "embedding_nearest": nearest_id,
                "llm_primary_sim": sims_by_id[skill.primary_dimension],
                "embedding_nearest_sim": sims_by_id[nearest_id],
                "message": (
                    f"LLM picked {skill.primary_dimension!r} but embedding-"
                    f"nearest is {nearest_id!r} "
                    f"({sims_by_id[nearest_id]:.3f} vs "
                    f"{sims_by_id[skill.primary_dimension]:.3f})"
                ),
            }
        ]

    return []


# ── 2. type-dim consistency (heuristic) ───────────────────────────────────


# Per skill type, the keywords that should appear (case-insensitively) in
# the primary dim's name OR description for the placement to be coherent.
# Types not in this map have no automatic check (e.g. Concept, SoftSkill —
# they can land in many dims, no good heuristic).
_TYPE_DIM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Format":      ("format", "serialization", "protocol", "schema", "wire"),
    "Datastore":   ("database", "datastore", "storage", "store", "db", "warehouse"),
    "Language":    ("language", "programming", "coding"),
    "Library":     ("library", "framework", "data access", "access"),
    "Framework":   ("framework", "web", "application", "api"),
    "Service":     ("service", "platform", "cloud", "managed"),
    "Platform":    ("platform", "cloud", "infrastructure"),
    "Runtime":     ("runtime", "container", "execution", "vm"),
    "Protocol":    ("protocol", "networking", "communication", "wire", "api"),
    "Methodology": ("methodology", "process", "practice", "agile", "ci/cd", "delivery"),
    "Architecture": ("architecture", "pattern", "design", "system"),
}


def type_dim_consistency(
    *,
    typed: TypedSkill,
    placed: PlacedSkill,
    dims_by_id: dict[str, dict],
) -> list[dict]:
    """Soft heuristic: the primary dim's name/description should mention
    a keyword associated with the skill's type. Types without an entry
    in ``_TYPE_DIM_KEYWORDS`` skip the check rather than warn spuriously.
    """
    dim = dims_by_id.get(placed.primary_dimension)
    if dim is None:
        return [
            {
                "level": "warning",
                "code": "primary_unknown",
                "skill_id": typed.skill_id,
                "primary_dimension": placed.primary_dimension,
                "message": (
                    f"primary_dimension={placed.primary_dimension!r} not in "
                    "the role's locked dim set"
                ),
            }
        ]
    keywords = _TYPE_DIM_KEYWORDS.get(typed.type)
    if not keywords:
        return []
    haystack = (dim.get("name", "") + " " + dim.get("description", "")).lower()
    if any(kw in haystack for kw in keywords):
        return []
    return [
        {
            "level": "warning",
            "code": "type_dim_mismatch",
            "skill_id": typed.skill_id,
            "skill_type": typed.type,
            "primary_dimension": placed.primary_dimension,
            "primary_dim_name": dim.get("name"),
            "message": (
                f"{typed.type}-typed skill {typed.name!r} placed in dim "
                f"{dim.get('name')!r} which doesn't match expected "
                f"keywords for its type"
            ),
        }
    ]


# ── 3. cross-skill consistency ─────────────────────────────────────────────


# A group must have at least this many skills before we look for outliers.
# Smaller groups don't have a meaningful "majority".
_MIN_GROUP_SIZE_FOR_OUTLIER_CHECK = 3


def cross_skill_consistency(
    *,
    typed: list[TypedSkill],
    placed: list[PlacedSkill],
) -> list[dict]:
    """Group skills by ``(type, subtype)``; within each group, find
    primary-dim outliers. If 3 of 4 (Language, scripting_language) skills
    share dim X and the 4th is in dim Y, the 4th is the outlier.

    Group threshold: minimum 3 skills. Below that, "majority" is
    meaningless and we skip the check.
    """
    placed_by_id = {p.skill_id: p for p in placed}
    grouped: dict[tuple[str, str], list[TypedSkill]] = defaultdict(list)
    for t in typed:
        grouped[(t.type, t.subtype)].append(t)

    findings: list[dict] = []
    for key, members in grouped.items():
        if len(members) < _MIN_GROUP_SIZE_FOR_OUTLIER_CHECK:
            continue
        primaries = []
        for t in members:
            p = placed_by_id.get(t.skill_id)
            if p is not None:
                primaries.append((t.skill_id, p.primary_dimension))
        if not primaries:
            continue
        counts = Counter(dim for _, dim in primaries)
        majority_dim, majority_count = counts.most_common(1)[0]
        # An outlier is meaningful only when there's a real majority.
        if majority_count <= len(primaries) // 2:
            continue
        for skill_id, dim in primaries:
            if dim != majority_dim:
                findings.append(
                    {
                        "level": "warning",
                        "code": "cross_skill_outlier",
                        "skill_id": skill_id,
                        "type": key[0],
                        "subtype": key[1],
                        "majority_dim": majority_dim,
                        "outlier_dim": dim,
                        "majority_count": majority_count,
                        "group_size": len(primaries),
                        "message": (
                            f"{key[0]}/{key[1]} group: {majority_count}/{len(primaries)} "
                            f"skills are in {majority_dim!r}, but {skill_id!r} "
                            f"is in {dim!r}"
                        ),
                    }
                )
    return findings


# ── 4. secondary-dim audit ────────────────────────────────────────────────


def secondary_dim_audit(
    *,
    placed: PlacedSkill,
    parent_skill_dims_by_skill_id: dict[str, list[str]],
) -> list[dict]:
    """If a placed skill has secondary_dimensions but one of those dims
    is already covered by the skill's parent_skills (via
    ``parent_skill_dims_by_skill_id``), the secondary is redundant.

    Per blueprint, the right structural answer is parent_skills, not
    secondary_dimensions. Strip the redundant secondary and flag.
    """
    if not placed.secondary_dimensions:
        return []
    parent_dims = set(parent_skill_dims_by_skill_id.get(placed.skill_id, []))
    if not parent_dims:
        return []
    findings: list[dict] = []
    for sec in placed.secondary_dimensions:
        if sec in parent_dims:
            findings.append(
                {
                    "level": "warning",
                    "code": "secondary_redundant_with_parent",
                    "skill_id": placed.skill_id,
                    "redundant_dim": sec,
                    "message": (
                        f"secondary_dimension={sec!r} is already covered by "
                        f"this skill's parent_skills; strip it and let the "
                        "parent relationship carry the connection"
                    ),
                }
            )
    return findings
