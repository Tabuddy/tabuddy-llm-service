"""Stage 6 deterministic relationship validators.

Five checks per the blueprint, all returning the warnings/errors dict
shape used by the rest of v3:

  * validate_dag_no_cycles                 — parent_skills must form a
                                              DAG; cycles -> error
  * service_has_platform_parent            — Service-typed -> at least
                                              one Platform-typed parent
  * library_has_lang_or_framework_parent   — Library SHOULD have
                                              Language/Framework parent
                                              (warning, not error)
  * architecture_has_no_parents            — Architecture-typed must
                                              have empty parent_skills
  * suppress_symmetry                      — A.suppress_on_match must
                                              be a subset of A.parent_skills
"""

from __future__ import annotations

from skill_library_v3.schemas.relationships import SkillRelationships
from skill_library_v3.schemas.typology import TypedSkill


# ── 1. DAG cycle validator ────────────────────────────────────────────────


def validate_dag_no_cycles(
    relationships: list[SkillRelationships],
) -> list[dict]:
    """Detect cycles in the parent_skills graph via iterative DFS.

    Returns one finding per cycle reachable from a different starting
    node (so a single cycle A→B→A doesn't get N reports for an N-node
    graph). Each finding carries the full cycle path under ``cycle``.

    Algorithm: standard 3-color DFS — WHITE (unvisited), GRAY (on
    current DFS stack), BLACK (fully explored). Encountering a GRAY
    node from a child edge = back edge = cycle.
    """
    parents_by_id: dict[str, list[str]] = {
        r.skill_id: list(r.parent_skills) for r in relationships
    }
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {sid: WHITE for sid in parents_by_id}

    seen_cycle_keys: set[frozenset] = set()
    findings: list[dict] = []

    def dfs(start: str) -> None:
        # Iterative DFS with explicit stack of (node, parent-iter).
        stack: list[tuple[str, list[str]]] = [(start, list(parents_by_id.get(start, [])))]
        path: list[str] = [start]
        color[start] = GRAY
        while stack:
            node, remaining = stack[-1]
            if not remaining:
                color[node] = BLACK
                stack.pop()
                if path and path[-1] == node:
                    path.pop()
                continue
            next_parent = remaining.pop()
            if next_parent not in parents_by_id:
                # Parent id not in our skill set — defensive, skip.
                continue
            c = color.get(next_parent, WHITE)
            if c == BLACK:
                continue
            if c == GRAY:
                # Back edge: cycle. Extract the cycle path from the
                # first occurrence of next_parent in our current path.
                if next_parent in path:
                    idx = path.index(next_parent)
                    cycle = path[idx:] + [next_parent]
                else:
                    cycle = [next_parent, node]
                key = frozenset(cycle)
                if key not in seen_cycle_keys:
                    seen_cycle_keys.add(key)
                    findings.append(
                        {
                            "level": "error",
                            "code": "parent_cycle",
                            "cycle": cycle,
                            "message": (
                                f"parent_skills cycle detected: "
                                f"{' -> '.join(cycle)}"
                            ),
                        }
                    )
                continue
            # WHITE: descend.
            color[next_parent] = GRAY
            path.append(next_parent)
            stack.append((next_parent, list(parents_by_id.get(next_parent, []))))

    for sid in parents_by_id:
        if color[sid] == WHITE:
            dfs(sid)
    return findings


# ── 2a. Pre-validator auto-link (Service → Platform) ──────────────────────


def auto_link_service_to_platform(
    *,
    typed_skills: list[TypedSkill],
    relationships: list[SkillRelationships],
) -> tuple[list[SkillRelationships], list[dict]]:
    """Deterministic post-processor: when a Service-typed skill has no
    Platform-typed parent AND the role's typed_skills set contains a
    Platform-typed skill, add it as a parent.

    Fixes the common LLM mistake where Stage 6's top-K candidate list for a
    Service-typed skill doesn't surface the role's enterprise Platform
    (e.g. ``pega-platform``, ``salesforce``, ``servicenow``) — the LLM then
    falls back to a Protocol parent (``rest``, ``soap``) which violates the
    Service-Platform rule, blocking Stage 6 at awaiting_review.

    The auto-link only fires when:
      * The skill is typed Service.
      * Its current parent_skills has no Platform-typed entry.
      * The role's typed_skills DOES include at least one Platform-typed
        skill (we have something to link TO).

    If the Platform skill already appears in the Service's ``related_to``
    by mistake (schema forbids same id in both parent_skills and
    related_to), we MOVE it from related_to → parent_skills.

    Idempotent. Safe to run before ``service_has_platform_parent``.

    Returns ``(patched_relationships, fixes_applied)`` where
    ``fixes_applied`` is a list of audit dicts for downstream logging.
    """
    type_by_id = {t.skill_id: t.type for t in typed_skills}
    platform_ids = [t.skill_id for t in typed_skills if t.type == "Platform"]
    fixes_applied: list[dict] = []

    if not platform_ids:
        return list(relationships), fixes_applied

    out: list[SkillRelationships] = []
    for rel in relationships:
        # Only touch Service-typed skills.
        if type_by_id.get(rel.skill_id) != "Service":
            out.append(rel)
            continue

        has_platform_parent = any(
            type_by_id.get(p) == "Platform" for p in rel.parent_skills
        )
        if has_platform_parent:
            out.append(rel)
            continue

        # Pick the first Platform from typed_skills order — deterministic.
        platform_to_add = platform_ids[0]

        # Schema rule: parent_skills and related_to must be disjoint.
        # If platform_to_add is already in related_to (LLM mistake),
        # MOVE it (don't duplicate, that fails schema validation).
        new_related = [
            r for r in rel.related_to if r != platform_to_add
        ]
        new_parents = [*rel.parent_skills, platform_to_add]

        # Rebuild via the schema's constructor so the structural checks run.
        patched = SkillRelationships(
            skill_id=rel.skill_id,
            parent_skills=new_parents,
            child_skills=list(rel.child_skills),
            suppress_on_match=list(rel.suppress_on_match),
            requires=list(rel.requires),
            related_to=new_related,
        )
        out.append(patched)
        fixes_applied.append(
            {
                "code": "auto_linked_service_to_platform",
                "level": "info",
                "skill_id": rel.skill_id,
                "added_parent": platform_to_add,
                "original_parents": list(rel.parent_skills),
            }
        )

    return out, fixes_applied


# ── 2. Service has Platform parent ─────────────────────────────────────────


def service_has_platform_parent(
    *,
    typed_skills: list[TypedSkill],
    relationships: list[SkillRelationships],
) -> list[dict]:
    """Service-typed skills MUST have at least one Platform-typed parent.
    Per blueprint: hard rule — finding is an error."""
    type_by_id = {t.skill_id: t.type for t in typed_skills}
    rels_by_id = {r.skill_id: r for r in relationships}
    findings: list[dict] = []
    for t in typed_skills:
        if t.type != "Service":
            continue
        rel = rels_by_id.get(t.skill_id)
        parents = (rel.parent_skills if rel else [])
        if not any(type_by_id.get(p) == "Platform" for p in parents):
            findings.append(
                {
                    "level": "error",
                    "code": "service_missing_platform_parent",
                    "skill_id": t.skill_id,
                    "parent_skills": parents,
                    "message": (
                        f"Service-typed skill {t.skill_id!r} has no "
                        "Platform-typed parent (rule: every Service "
                        "must trace to a hosting Platform)"
                    ),
                }
            )
    return findings


# ── 3. Library has Language/Framework parent ─────────────────────────────


def library_has_lang_or_framework_parent(
    *,
    typed_skills: list[TypedSkill],
    relationships: list[SkillRelationships],
) -> list[dict]:
    """Library-typed skills SHOULD have a Language- or Framework-typed
    parent. Per blueprint: 'should' = warning, not error."""
    type_by_id = {t.skill_id: t.type for t in typed_skills}
    rels_by_id = {r.skill_id: r for r in relationships}
    findings: list[dict] = []
    for t in typed_skills:
        if t.type != "Library":
            continue
        rel = rels_by_id.get(t.skill_id)
        parents = (rel.parent_skills if rel else [])
        if not any(
            type_by_id.get(p) in ("Language", "Framework") for p in parents
        ):
            findings.append(
                {
                    "level": "warning",
                    "code": "library_missing_lang_or_framework_parent",
                    "skill_id": t.skill_id,
                    "parent_skills": parents,
                    "message": (
                        f"Library-typed skill {t.skill_id!r} has no "
                        "Language- or Framework-typed parent (soft rule)"
                    ),
                }
            )
    return findings


# ── 4. Architecture has no parents ────────────────────────────────────────


def architecture_has_no_parents(
    *,
    typed_skills: list[TypedSkill],
    relationships: list[SkillRelationships],
) -> list[dict]:
    """Architecture-typed skills MUST have empty parent_skills. Per
    blueprint: hard rule (Architectures are top-level concepts that
    influence systems; they don't 'belong to' a parent skill)."""
    rels_by_id = {r.skill_id: r for r in relationships}
    findings: list[dict] = []
    for t in typed_skills:
        if t.type != "Architecture":
            continue
        rel = rels_by_id.get(t.skill_id)
        parents = (rel.parent_skills if rel else [])
        if parents:
            findings.append(
                {
                    "level": "error",
                    "code": "architecture_has_parents",
                    "skill_id": t.skill_id,
                    "parent_skills": parents,
                    "message": (
                        f"Architecture-typed skill {t.skill_id!r} must "
                        f"have empty parent_skills; got {parents}"
                    ),
                }
            )
    return findings


# ── 5. Suppress symmetry ──────────────────────────────────────────────────


def suppress_symmetry(
    *,
    relationships: list[SkillRelationships],
) -> list[dict]:
    """Every id in ``A.suppress_on_match`` must also appear in
    ``A.parent_skills``. Suppression requires containment per the
    blueprint — if A 'absorbs' B at extraction time, A must already
    declare B as a parent."""
    findings: list[dict] = []
    for r in relationships:
        if not r.suppress_on_match:
            continue
        parent_set = set(r.parent_skills)
        for sup in r.suppress_on_match:
            if sup not in parent_set:
                findings.append(
                    {
                        "level": "error",
                        "code": "suppress_not_in_parents",
                        "skill_id": r.skill_id,
                        "suppress_id": sup,
                        "parent_skills": list(r.parent_skills),
                        "message": (
                            f"{r.skill_id!r}.suppress_on_match contains "
                            f"{sup!r} but it's not in parent_skills "
                            "(suppression requires containment)"
                        ),
                    }
                )
    return findings
