"""Stage 5 placer prompt (system + user).

The blueprint flags secondary-dim overuse as the #1 failure mode of this
stage. The system prompt is opinionated and lists explicit reject
patterns so the LLM has to actively justify any non-empty
secondary_dimensions.
"""

from __future__ import annotations

import json


PLACER_PROMPT_VERSION = "stage5_placement_v1.0"


PLACER_SYSTEM_PROMPT = """\
You place skills in a role's locked dimension structure.

Each skill gets EXACTLY ONE primary_dimension (its home). Secondary
dimensions are RARE and require strict justification.

PRIMARY DIM RULE
================
Pick the dimension where this skill is most prominently used and
conceptually anchored. Type drives this:

  * Language → the role's languages dim
  * Library  → the dim of the library's primary function (data access,
               testing, observability, etc.)
  * Tool     → the dim covering its primary use case
  * Datastore → the dim about persistence/storage for this role
  * Format   → the dim about serialization/protocols
  * Concept / Methodology / Architecture → the dim that conceptually
               owns this idea

SECONDARY DIM RULE — HIGH BAR (most skills get NONE)
====================================================
Add a secondary dim ONLY IF ALL THREE conditions hold:

  1. The skill genuinely operates with roughly equal weight in two
     distinct conceptual areas of THIS role's dim structure.
  2. Browsing the secondary dim would feel incomplete without this skill.
  3. The relationship cannot already be expressed by parent_skills,
     related_to, or requires (Stage 6 fields, but consider them as
     concepts here).

REJECT THESE PATTERNS for secondary_dimensions:

  * "It's an AWS service so D02 is secondary"
        → no, that's a parent_skill relationship.
  * "It's written in Python so D01 is secondary"
        → no, that's a parent_skill relationship.
  * "It can be used in pipelines"
        → no, too speculative.
  * "It's also a kind of storage"
        → no, that's the type/subtype, not a secondary dim.

DEFAULT: secondary_dimensions = []. If you list any, your reasoning
must explicitly justify it against the THREE conditions above.

Output rules:

  * primary_dimension: a dim_id from the provided list. Do NOT invent
    new dim_ids. Echo one of the given ids exactly.
  * secondary_dimensions: empty list (default), or up to 2 dim_ids.
    Same constraint — must come from the provided list. The primary
    must NOT also appear in secondary_dimensions.
  * placement_confidence: 0.0-1.0. Calibrate honestly — a Language
    going into the languages dim is ~0.95; an ambiguous tool that
    could fit two dims is ~0.6.
  * reasoning: 2-3 sentences explaining the primary choice and (if any)
    secondary justification against the three conditions above.

Emit one JSON object: ``{{"skills": [...]}}``. No prose, no code fences.
"""


PLACER_USER_TEMPLATE = """\
Role: {role_card_block}

Available dimensions for this role (pick primary from this list):
{dims_block}

Skills to place:
{skills_block}

Place each skill. Default secondary_dimensions=[]. If non-empty, justify
explicitly against the three conditions in the system prompt.
"""


def format_role_card_block(role_card: dict) -> str:
    parts = [
        f"canonical_name: {role_card.get('canonical_name', '')}",
        f"family: {role_card.get('family', '')}",
        f"definition: {role_card.get('definition', '')}",
    ]
    resp = role_card.get("primary_responsibilities")
    if resp:
        if isinstance(resp, list):
            resp = "; ".join(resp)
        parts.append(f"primary_responsibilities: {resp}")
    art = role_card.get("common_artifacts")
    if art:
        if isinstance(art, list):
            art = "; ".join(art)
        parts.append(f"common_artifacts: {art}")
    return "\n".join(parts)


def format_dims_block(dims: list[dict]) -> str:
    """Render the role's locked dimensions as a compact list with
    tentative_id, name, description, in_scope. The LLM picks
    tentative_id values from this list as primary / secondary."""
    lines = []
    for d in dims:
        in_scope = d.get("in_scope") or ""
        out_of_scope = d.get("out_of_scope") or ""
        lines.append(
            f"  - {d.get('tentative_id', '?')}: {d.get('name', '?')}\n"
            f"      description: {d.get('description', '')}\n"
            f"      in_scope: {in_scope}\n"
            f"      out_of_scope: {out_of_scope}"
        )
    return "\n".join(lines)


def format_skills_block(typed_skills: list[dict]) -> str:
    """Render the input batch — each skill carries skill_id, name,
    type, subtype so the LLM can apply the type-driven primary rule."""
    rows = []
    for s in typed_skills:
        rows.append(
            {
                "skill_id": s.get("skill_id"),
                "name": s.get("name"),
                "type": s.get("type"),
                "subtype": s.get("subtype"),
            }
        )
    return json.dumps(rows, indent=2, ensure_ascii=False)
