"""Stage 1 role-card prompt (system + user).

Versioned so every ``v2_run_log`` row records which prompt produced its
``planner_output`` — bump ROLE_CARD_PROMPT_VERSION on any text change.
The ``stage1_`` prefix is what the runner + repository use to discriminate
Stage 1 runs from Stage 0 charter runs in the shared v2_run_log table.
"""

from __future__ import annotations

import json

from skill_library_v3.schemas.role_card import FAMILY_VALUES

ROLE_CARD_PROMPT_VERSION = "stage1_anchor_v1.2"


ROLE_CARD_SYSTEM_PROMPT = """\
You compress an approved role charter into a structured "role card" that
downstream stages of a canonical skills-catalog pipeline will reference.
Be terse but precise. Drop adjective stuffing; keep the load-bearing nouns
and verbs. Stage 2 (dimension generation) reads this card as its primary
input, so DENSITY MATTERS more than verbosity.

Output rules:
  * canonical_name: the human-readable role name (echo the one provided
    unless the charter clearly uses a different canonical wording).

  * aliases: 3 to 6 alternate spellings or commonly-used short forms that
    refer to THE EXACT SAME ROLE. Aliases are surface-form variants of
    the same job, not adjacent roles in the same family. Aim to include
    at least ONE short-form abbreviation when one exists in the field
    (e.g. "BE" for Backend Engineer, "DE" for Data Engineer, "QA" for
    Quality Assurance Engineer, "FE" for Frontend Engineer).

    Acceptable examples:
      - Abbreviations: "BE" for Backend Engineer, "DE" for Data Engineer,
        "FE" for Frontend Engineer, "ML" for ML Engineer
      - Spelling variants: "Front-End Developer" for Frontend Developer
      - Deprecated/legacy wordings: "Webmaster" for Web Engineer
      - Marketplace shorthand for the same role:
          "Backend Developer", "Server-side Engineer", "Backend Software
          Engineer", "Server Developer" all for Backend Engineer

    DO NOT include distinct roles, even when they share a tech stack or
    family with the canonical role. The following are NOT aliases — they
    are separate roles that will get their own charter:
      - For DevOps Engineer:
          "Site Reliability Engineer", "Platform Engineer",
          "Build and Release Engineer", "Production Engineer",
          "DevSecOps Engineer" — all distinct disciplines.
      - For Data Engineer:
          "Data Scientist", "Analytics Engineer", "ML Engineer".
      - For MLOps Engineer:
          "ML Engineer", "Data Scientist", "AI Engineer".
      - For Backend Engineer:
          "Full Stack Engineer", "API Engineer" (when listed as its
          own role), "Platform Engineer".

    The litmus test: if the candidate alias has its own meaningful job-
    posting volume distinct from the canonical role, it is NOT an alias.
    Do not include the canonical_name itself.

  * family: pick exactly ONE from the closed list provided. If the role
    legitimately straddles two families, pick the dominant one — Stage 2
    will still see the full charter.

  * definition: ONE sentence, <=200 characters. Lead with the load-bearing
    verb. Mention the request lifecycle / data flow / ownership boundary
    that distinguishes this role.

  * primary_responsibilities: a SINGLE STRING containing 5 to 9
    semicolon-delimited responsibility areas. Each item is a short
    domain-specific phrase (3-8 words). Use specific technical language
    where appropriate (e.g. "API design and implementation; OLTP database
    modeling; authentication and authorization; caching and performance;
    message-queue integration"). NOT a JSON array — ONE STRING with
    "; " between items, no trailing semicolon.

  * common_artifacts: a SINGLE STRING containing 3 to 10 semicolon-
    delimited concrete deliverables. Be specific — "REST/GraphQL/gRPC
    services; OLTP database schemas; OpenAPI specifications; technical
    design documents; service repositories; contract tests" rather than
    "service interfaces; data models". NOT a JSON array.

  * distinguishing_tasks: 1-3 sentences. What separates this role from
    the adjacent roles named in the charter. Include the request-
    lifecycle / artifact-ownership boundary that other roles do NOT
    cross.

  * seniority_signals: one short descriptor per band (junior / mid /
    senior / staff). Each describes scope and autonomy at that level
    (e.g. "implements features within an existing service with code-
    review supervision" vs "owns cross-service architecture for a
    domain"). Each band must be DISTINCT.

  * adjacent_roles: list of 5-15 role-name strings (flat strings, not
    objects) of the adjacent roles for this role's catalog position.
    Carry forward EVERY adjacent role the charter named — never drop
    one. You may ADD canonical adjacent roles the charter omitted (for
    example: if the charter only listed Backend Engineer as adjacent,
    you may add "Frontend Engineer", "Full Stack Engineer", "DevOps
    Engineer" etc. to the role-card list as long as they're well-known
    sibling roles in the same family). Use canonical role names ("Site
    Reliability Engineer", not "SRE"; "Database Administrator", not
    "DBA").

  * always_load_skills: list of 3-10 cross-cutting skill names that
    every practitioner of this role uses regardless of specialization
    (e.g. "Git", "SQL", "Code Review", "Unit Testing", "Agile / Scrum",
    "English communication"). Carry forward EVERY skill the charter
    named — never drop one. You may add genuinely cross-cutting
    universal skills the charter omitted. Skills here will be excluded
    from dimension generation in Stage 2 — they're "load these for
    every candidate" universal items, not dimensional differentiators.

Skill names of role-specific tools and technologies (Spring Boot, Kafka,
PostgreSQL, etc.) MUST stay OUT of the role card — those go to Stage 4
skill placement, not here. always_load_skills is the only field where
named skills appear, and only for genuinely universal ones.

Emit only a single JSON object matching the schema. No prose, no code
fences.
"""


ROLE_CARD_USER_TEMPLATE = """\
Role to anchor: {role_name}
Slug to echo as role_id: {role_slug}

Approved charter (Stage 0 output):
{charter_block}

Closed list of role families — pick exactly one:
{family_enum}

Produce the JSON role card now.
"""


# ── formatters ─────────────────────────────────────────────────────────────


# One-line description per family so the LLM has anchors when classifying
# instead of guessing from the family name alone. Kept inline rather than
# moved to a config file because it's prompt content — versioned together.
_FAMILY_DESCRIPTIONS: dict[str, str] = {
    "Software Engineering":
        "application code authorship across backend, frontend, mobile, embedded.",
    "Infrastructure & Platform":
        "SRE, DevOps, network, storage, virtualization, compute platform reliability.",
    "Data & ML":
        "data pipelines, warehousing, analytics, ML model training and MLOps.",
    "Security":
        "AppSec, SecOps, GRC, identity, threat detection, secure design review.",
    "ERP & Enterprise":
        "SAP, Oracle, Workday, Dynamics — functional and technical.",
    "Digital & Martech":
        "analytics platforms, ad tech, CMS, CRM, CDP integration and operation.",
    "QA & SDET":
        "test strategy, automation, performance, quality engineering.",
    "Hardware & Firmware":
        "silicon, board bring-up, firmware, embedded device drivers.",
    "Support/Ops Engineering":
        "production support, customer-facing operations, escalation, runbooks.",
}


def format_charter_block(charter: dict) -> str:
    """Render the charter JSON with explicit section headers so the LLM
    parses it as structure rather than blob."""
    if not isinstance(charter, dict):
        return json.dumps(charter)

    parts: list[str] = []
    parts.append(f"DEFINITION:\n  {charter.get('definition', '').strip()}")

    in_scope = charter.get("in_scope") or []
    if in_scope:
        parts.append(
            "IN_SCOPE:\n" + "\n".join(f"  - {item}" for item in in_scope)
        )

    out_of_scope = charter.get("out_of_scope") or []
    if out_of_scope:
        rendered = []
        for item in out_of_scope:
            task = item.get("task", "?") if isinstance(item, dict) else str(item)
            owned_by = item.get("owned_by", "?") if isinstance(item, dict) else "?"
            rendered.append(f"  - {task}  ->  owned by {owned_by}")
        parts.append("OUT_OF_SCOPE:\n" + "\n".join(rendered))

    adjacent = charter.get("adjacent_roles") or []
    if adjacent:
        rendered = []
        for item in adjacent:
            role = item.get("role", "?") if isinstance(item, dict) else str(item)
            note = item.get("boundary_note", "") if isinstance(item, dict) else ""
            rendered.append(f"  - {role}: {note}")
        parts.append("ADJACENT_ROLES:\n" + "\n".join(rendered))

    always_load = charter.get("always_load_skills") or []
    if always_load:
        parts.append(
            "ALWAYS_LOAD_SKILLS:\n  " + ", ".join(always_load)
        )

    return "\n\n".join(parts)


def format_family_enum() -> str:
    """Render the family enum as a dashed list with one-line descriptions
    so the LLM has disambiguation anchors when picking."""
    lines = []
    for name in FAMILY_VALUES:
        lines.append(f"  - {name} — {_FAMILY_DESCRIPTIONS.get(name, '')}")
    return "\n".join(lines)
