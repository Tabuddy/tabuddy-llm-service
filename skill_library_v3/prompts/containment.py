"""Stage 6 containment-resolution prompt (system + user).

Per blueprint, the five relationship distinctions are subtle. The
system prompt teaches them with examples and explicitly warns about
the composition-vs-containment trap (PySpark = Python + Spark, where
both parents are real but neither is suppressed at extraction time).
"""

from __future__ import annotations

import json


CONTAINMENT_PROMPT_VERSION = "stage6_containment_v1.0"


CONTAINMENT_SYSTEM_PROMPT = """\
You identify the relationship structure for one skill. Containment
relationships drive a longest-match algorithm at extraction time, so
precision matters — wrong parent_skills or sloppy suppress_on_match
leads to either over-counting (every Python mention also credits
"Backend Engineer skill") or silent skill drops.

The five relationship buckets:

  * parent_skills — this skill is a sub-component of these.
        Examples:
          - AWS Lambda's parent is AWS.
          - PySpark's parents are Python AND Spark (both legitimate).
          - React Router's parent is React.
        NOT a parent: a related skill, a prerequisite, or "another skill
        used in the same role".

  * child_skills — rare to set directly; usually inferred from parents
        on the OTHER side of the relationship. Set only when a skill
        clearly contains specific named children (e.g., AWS contains
        AWS Lambda + AWS S3 + …).

  * suppress_on_match — when this skill is matched in text, the listed
        skills should NOT also be credited. Use ONLY when:
          (a) the relationship is "X contains Y", AND
          (b) Y is a STRICT sub-mention of X (matching "AWS Lambda"
              should suppress a separate credit for "AWS").
        DO NOT use suppress for COMPOSITION:
          - PySpark = Python + Spark composition. A "PySpark" mention
            DOES credit Python and Spark — both are legit. Suppressing
            them would lose information.
        Suppression requires containment: every id in suppress_on_match
        MUST also be in parent_skills.

  * requires — prerequisite knowledge needed to use this skill, but
        NOT a containment relation. Example: AWS Lambda requires
        knowledge of "AWS IAM", but IAM isn't a parent of Lambda.

  * related_to — lateral skills (alternatives, complements). NOT
        parents and NOT children.
        Examples:
          - AWS Lambda's related_to: Azure Functions, Google Cloud Functions.
          - Python's related_to: Java, Go, Ruby (alternative languages).

Critical type-rules (the deterministic validator will reject violations):

  * Service-typed skills MUST have at least one Platform-typed parent.
  * Library-typed skills SHOULD (not must) have a Language- or
    Framework-typed parent.
  * Architecture-typed skills MUST have empty parent_skills (Architecture
    is top-level — Microservices, Hexagonal, CQRS — they don't have
    parent skills, only related_to peers).

Default behaviour: empty arrays. Don't fill buckets just because the
candidate list is non-empty. A skill with no clear parent in the
candidate list should have parent_skills=[].

Emit ONE JSON object matching this shape:

  {{
    "skill_id": "<echo the target's skill_id>",
    "parent_skills": [...],
    "child_skills": [...],
    "suppress_on_match": [...],
    "requires": [...],
    "related_to": [...]
  }}

The ``skill_id`` MUST equal the target's ``skill_id``. No prose, no
fences.
"""


CONTAINMENT_USER_TEMPLATE = """\
Target skill:
{target_block}

Top similar skills (candidates for parent / child / requires /
related_to / suppress_on_match — pick from these or omit):
{candidates_block}

Produce ONE JSON SkillRelationships object with the five relationship
arrays. Default each to []. Apply the type-rules and the
suppression-requires-containment rule.
"""


def format_target_block(typed: dict, placed: dict) -> str:
    """Render the target skill's identity, type, and placement so the
    LLM has the same view the validators will check against."""
    return (
        f"  skill_id:           {typed.get('skill_id')}\n"
        f"  name:               {typed.get('name')}\n"
        f"  type:               {typed.get('type')}\n"
        f"  subtype:            {typed.get('subtype')}\n"
        f"  primary_dimension:  {placed.get('primary_dimension')}\n"
        f"  secondary_dimensions: {placed.get('secondary_dimensions') or []}"
    )


def format_candidates_block(candidates: list[dict]) -> str:
    """Render the top-K similar candidates as a compact JSON list with
    similarity scores so the LLM has both name and signal strength."""
    if not candidates:
        return "  (no similar candidates available)"
    rows = [
        {
            "skill_id": c.get("skill_id"),
            "name": c.get("name"),
            "similarity": round(float(c.get("similarity", 0.0)), 4),
        }
        for c in candidates
    ]
    return json.dumps(rows, indent=2, ensure_ascii=False)
