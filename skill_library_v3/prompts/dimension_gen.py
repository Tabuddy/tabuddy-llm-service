"""Stage 2 dimension-generation prompt (system + user).

Versioned so every ``v2_run_log`` row records which prompt produced its
``planner_output``. Bump DIM_GEN_PROMPT_VERSION on any text change.
The ``stage2_`` prefix is what the runner + repository use to discriminate
Stage 2 runs from Stages 0/1 in the shared v2_run_log table.
"""

from __future__ import annotations

import json

DIM_GEN_PROMPT_VERSION = "stage2_dim_gen_v1.0"


DIM_GEN_SYSTEM_PROMPT = """\
You are the Stage 2 Dimension Generator for a canonical skills-catalog
pipeline. Given a role card (the dense compression of an approved role
charter), produce 12-20 candidate dimensions covering the role's skill
surface.

A "dimension" is a coherent skill cluster a practitioner of this role
needs. Examples for Backend Engineer: Programming Languages, Web &
Application Frameworks, API Design & Patterns, Relational Databases,
Authentication & Authorization, Microservices & Service Architecture,
Distributed Systems Concepts, Software Design Patterns, etc.

Hard rules:

  * Generate EXACTLY 12 to 20 dimensions. No fewer, no more.
  * Intentionally allow a few overlaps and over-splits — Stage 3
    reconciliation will merge/split as needed. Better to surface a
    duplicate than to silently miss a skill cluster.
  * No "Miscellaneous", "Other", "General", "Misc" dimensions. If you
    can't name 5+ concrete exemplar skills under a dimension, drop it
    or fold it into a sibling dimension.
  * Skills listed in always_load_skills (Git, SQL, Code Review, Unit
    Testing, etc.) MUST NOT be turned into their own dimensions. Those
    are cross-cutting universal items handled separately. Do not name
    a dimension "Git" or "SQL" or "Unit Testing".
  * Avoid marketing / buzzword names ("Cloud-Native Excellence",
    "Modern Engineering Practices"). Name dimensions by their substance
    (Containerization, Software Design Patterns & Architecture).

Per-dimension fields:

  * tentative_id: 'd_init_NN' where NN is two digits zero-padded
    (d_init_01, d_init_02, ..., d_init_19, d_init_20). Unique within
    the role.
  * name: 3-8 word noun-phrase. Specific, not generic ("Pipeline
    Orchestration", not "Tools").
  * description: 1-3 sentences (<=400 chars). What the dimension covers
    and why it's a coherent cluster for this role.
  * in_scope: comma-delimited specifics — concrete tools, frameworks,
    libraries, or sub-topics that belong in this dim. Example for
    Relational Databases: "PostgreSQL, MySQL, SQL Server, Oracle,
    MariaDB". 10-400 chars.
  * out_of_scope: comma-delimited specifics naming what is excluded and
    why (which other dim or which adjacent role owns it). Example:
    "Analytical warehouses (those go to Data Engineer's dim);
    DBA-owned operational tasks". 10-400 chars.
  * exemplar_skills: 3-15 concrete skill names that anchor this
    dimension. These are CONCRETE skills (PostgreSQL, Spring Boot,
    OAuth 2.0), not concepts.
  * overlap_flags: zero or more flags pointing at potential overlap.
    Three forms:
      - {"with_dim_name": "<name>", "reason": "..."} — overlap with
        another dim YOU just generated for THIS role (the LLM knows the
        names you used)
      - {"with_dim_id": "<id>", "reason": "..."} — overlap with an
        adjacent role's existing canonical-store dim id
      - {"with_role": "<role>", "reason": "..."} — overlap with an
        adjacent role's territory generally
    Use overlap_flags WHENEVER an adjacent role has a dim that would
    plausibly contain the same skill. Stage 3 needs these to reconcile.

Coverage discipline. For technical roles, sweep these axes before
finalizing — drop any axis that doesn't apply, but check each:

  * Programming Languages used by the role
  * Web/application/server frameworks
  * Data stores (relational + NoSQL split when both apply)
  * Caching / in-memory stores
  * Messaging / event streaming
  * Authentication / authorization
  * Service architecture (microservices, gateways, mesh)
  * Containerization (consumer depth for application roles)
  * Testing (frameworks + types)
  * Observability / monitoring
  * Application security
  * CI/CD usage (consumer depth for application roles)
  * Distributed systems concepts
  * Software design patterns / architecture
  * Data serialization / protocols
  * ORMs / data access libraries (when relational DB applies)
  * Cloud platforms (consumer depth for application roles)

Don't force-fit irrelevant axes — if the role doesn't touch it, omit it.
But sweep before finalizing.

Emit a single JSON object matching the schema. No prose, no code fences.
"""


DIM_GEN_USER_TEMPLATE = """\
Role card for {role_name} ({role_slug}):

{role_card_block}

Always-load skills (do NOT make these dimensions):
{always_load_block}

{adjacent_dim_block}

Produce the JSON dimension list now. Remember: 12-20 dimensions, every
dimension has 3+ exemplar_skills, flag overlaps with adjacent roles
when present.
"""


# ── formatters ─────────────────────────────────────────────────────────────


def format_role_card_block(role_card: dict) -> str:
    """Render the role card as a compact block. v1.2 role cards have
    primary_responsibilities + common_artifacts as semicolon-strings; v1.0
    has them as lists. Handle both for backwards compatibility during the
    v1.0 → v1.2 transition."""

    def _to_str(field) -> str:
        if isinstance(field, list):
            return "; ".join(field)
        return str(field or "")

    parts = [
        f"canonical_name: {role_card.get('canonical_name', '')}",
        f"family: {role_card.get('family', '')}",
        f"definition: {role_card.get('definition', '')}",
        f"primary_responsibilities: {_to_str(role_card.get('primary_responsibilities'))}",
        f"common_artifacts: {_to_str(role_card.get('common_artifacts'))}",
        f"distinguishing_tasks: {role_card.get('distinguishing_tasks', '')}",
    ]
    aliases = role_card.get("aliases") or []
    if aliases:
        parts.append(f"aliases: {', '.join(aliases)}")
    seniority = role_card.get("seniority_signals") or {}
    if isinstance(seniority, dict) and any(seniority.values()):
        sen_lines = [
            f"  - {level}: {seniority.get(level, '')}"
            for level in ("junior", "mid", "senior", "staff")
            if seniority.get(level)
        ]
        parts.append("seniority_signals:\n" + "\n".join(sen_lines))
    return "\n".join(parts)


def format_always_load_block(skills: list[str]) -> str:
    if not skills:
        return "(none — no cross-cutting universals to exclude)"
    return ", ".join(skills)


def format_adjacent_dim_block(
    adjacent_dimensions_by_role: dict[str, list[dict]],
) -> str:
    """Render adjacent roles' existing dimensions so the LLM can flag
    overlaps. ``adjacent_dimensions_by_role`` is a dict keyed by role
    display name, with values being lists of dim dicts (each dim dict has
    at least ``name`` and ``description`` keys, optionally ``id``).

    Empty/missing adjacent dims → placeholder so the LLM knows it's
    bootstrapping rather than expected to flag overlaps."""
    if not adjacent_dimensions_by_role:
        return (
            "Adjacent roles' existing dimensions:\n"
            "(no adjacent-role dimensions in the catalog yet — emit empty "
            "overlap_flags lists, the catalog is bootstrapping.)"
        )
    blocks = ["Adjacent roles' existing dimensions (flag overlaps when relevant):"]
    for role_name, dims in adjacent_dimensions_by_role.items():
        if not dims:
            continue
        blocks.append(f"\n  ── {role_name} ──")
        for d in dims:
            dim_id = d.get("tentative_id") or d.get("id") or d.get("dim_id") or ""
            dim_name = d.get("name") or "?"
            desc = (d.get("description") or "").strip()
            # Truncate description to keep prompt budget reasonable across
            # many adjacent roles.
            if len(desc) > 160:
                desc = desc[:157] + "…"
            label = f"{dim_id}: {dim_name}" if dim_id else dim_name
            blocks.append(f"    - {label} — {desc}")
    return "\n".join(blocks)
