"""Deterministic post-LLM checks for Stage 2 dimension lists.

Errors block the auto-approve gate; warnings don't. Lifecycle:
  errors empty   -> v2_run_log.status = 'approved'   (auto, no review queue)
  errors present -> v2_run_log.status = 'failed'     (no recovery via UI)

Same dict shape as Stages 0/1 validators so audit tooling is uniform.
"""

from __future__ import annotations

import re

from skill_library_v3.schemas.dimension import DimensionList


_FORBIDDEN_TOKENS: frozenset[str] = frozenset(
    {"miscellaneous", "misc", "other", "general"}
)
_TENT_ID_RE = re.compile(r"^d_init_\d{2,3}$")


def count_in_range(dlist: DimensionList) -> list[dict]:
    n = len(dlist.candidate_dimensions)
    if n < 12 or n > 20:
        return [
            {
                "level": "error",
                "code": "dimension_count_out_of_range",
                "location": "candidate_dimensions",
                "message": f"got {n} dimensions; must be 12-20.",
            }
        ]
    return []


def unique_dim_names(dlist: DimensionList) -> list[dict]:
    seen: dict[str, int] = {}
    out: list[dict] = []
    for i, d in enumerate(dlist.candidate_dimensions):
        key = d.name.strip().lower()
        if key in seen:
            out.append(
                {
                    "level": "error",
                    "code": "duplicate_dim_name",
                    "location": f"candidate_dimensions[{i}].name",
                    "message": (
                        f"name {d.name!r} duplicates dim at index {seen[key]}"
                    ),
                }
            )
        else:
            seen[key] = i
    return out


def no_misc_dimension(dlist: DimensionList) -> list[dict]:
    out: list[dict] = []
    for i, d in enumerate(dlist.candidate_dimensions):
        lower = d.name.lower()
        for forbidden in _FORBIDDEN_TOKENS:
            if forbidden in lower:
                out.append(
                    {
                        "level": "error",
                        "code": "forbidden_dim_name",
                        "location": f"candidate_dimensions[{i}].name",
                        "message": (
                            f"dim name {d.name!r} contains forbidden token "
                            f"{forbidden!r}; rename or fold into a sibling dim."
                        ),
                    }
                )
                break
    return out


def tentative_ids_well_formed(dlist: DimensionList) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for i, d in enumerate(dlist.candidate_dimensions):
        if not _TENT_ID_RE.match(d.tentative_id):
            out.append(
                {
                    "level": "error",
                    "code": "bad_tentative_id_format",
                    "location": f"candidate_dimensions[{i}].tentative_id",
                    "message": (
                        f"tentative_id {d.tentative_id!r} must match d_init_NN"
                    ),
                }
            )
        elif d.tentative_id in seen:
            out.append(
                {
                    "level": "error",
                    "code": "duplicate_tentative_id",
                    "location": f"candidate_dimensions[{i}].tentative_id",
                    "message": f"tentative_id {d.tentative_id!r} already used",
                }
            )
        seen.add(d.tentative_id)
    return out


def always_load_not_a_dimension(
    dlist: DimensionList, always_load_skills: list[str]
) -> list[dict]:
    """Cross-cutting always-load skills must not be turned into dimensions."""
    if not always_load_skills:
        return []
    universal_lc = {s.strip().lower() for s in always_load_skills if s.strip()}
    out: list[dict] = []
    for i, d in enumerate(dlist.candidate_dimensions):
        name_lc = d.name.strip().lower()
        if name_lc in universal_lc:
            out.append(
                {
                    "level": "error",
                    "code": "always_load_as_dimension",
                    "location": f"candidate_dimensions[{i}].name",
                    "message": (
                        f"dim {d.name!r} matches an always_load_skill; "
                        "those are cross-cutting and cannot be dimensions."
                    ),
                }
            )
    return out


def exemplars_present(dlist: DimensionList) -> list[dict]:
    """Pydantic enforces 3-15 exemplars; deterministic recheck catches
    edge cases like all-empty-string exemplars sneaking past Pydantic."""
    out: list[dict] = []
    for i, d in enumerate(dlist.candidate_dimensions):
        non_empty = [s for s in d.exemplar_skills if s and s.strip()]
        if len(non_empty) < 3:
            out.append(
                {
                    "level": "error",
                    "code": "exemplars_too_few",
                    "location": f"candidate_dimensions[{i}].exemplar_skills",
                    "message": (
                        f"only {len(non_empty)} non-empty exemplar(s); need >=3."
                    ),
                }
            )
    return out


def overlap_flags_engaged_when_adjacents_present(
    dlist: DimensionList,
    adjacent_dimensions_by_role: dict[str, list[dict]],
) -> list[dict]:
    """Soft signal — when adjacent roles' dims are in the input but the
    LLM produced no overlap_flags anywhere, that's likely the LLM ignored
    the adjacency context. Warning, not error."""
    has_adjacents = any(
        bool(dims) for dims in (adjacent_dimensions_by_role or {}).values()
    )
    if not has_adjacents:
        return []
    any_flags = any(d.overlap_flags for d in dlist.candidate_dimensions)
    if any_flags:
        return []
    return [
        {
            "level": "warning",
            "code": "overlap_flags_unengaged",
            "location": "candidate_dimensions",
            "message": (
                "adjacent roles have dimensions in the catalog, but no "
                "overlap_flags were emitted on any candidate dim. The LLM "
                "may have skipped the adjacency context."
            ),
        }
    ]


def run_dimension_validators(
    dlist: DimensionList,
    *,
    always_load_skills: list[str],
    adjacent_dimensions_by_role: dict[str, list[dict]],
) -> dict:
    """Run every check; partition into warnings/errors. Errors block
    auto-approve."""
    findings: list[dict] = []
    findings += count_in_range(dlist)
    findings += unique_dim_names(dlist)
    findings += no_misc_dimension(dlist)
    findings += tentative_ids_well_formed(dlist)
    findings += always_load_not_a_dimension(dlist, always_load_skills or [])
    findings += exemplars_present(dlist)
    findings += overlap_flags_engaged_when_adjacents_present(
        dlist, adjacent_dimensions_by_role or {}
    )

    warnings = [f for f in findings if f["level"] == "warning"]
    errors = [f for f in findings if f["level"] == "error"]
    return {"warnings": warnings, "errors": errors}
