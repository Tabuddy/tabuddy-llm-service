"""Stage 6 containment-resolution prompt (system + user).

Per blueprint, the five relationship distinctions are subtle. The
system prompt teaches them with examples and explicitly warns about
the composition-vs-containment trap (PySpark = Python + Spark, where
both parents are real but neither is suppressed at extraction time).
"""

from __future__ import annotations

import json


CONTAINMENT_PROMPT_VERSION = "stage6_containment_v1.1"


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

Type-rules — these are HARD rules the deterministic validator enforces.
You MUST satisfy them whenever the candidate list permits:

  * Service-typed targets MUST have at least one Platform-typed parent.
    Scan the candidate list for any skill that names AWS, Azure, GCP,
    a chain platform (Ethereum/Solana/etc.), or any other hosting
    Platform — if one is there and is the genuine host of this
    Service, parent_skills MUST include it. Do not leave parent_skills
    empty for a Service when a Platform candidate exists.
  * Library-typed targets SHOULD (not must) have a Language- or
    Framework-typed parent — include it when one is in the candidate
    list and it's the language/framework the library is for.
  * Architecture-typed targets MUST have empty parent_skills
    (Architectures are top-level — Microservices, Hexagonal, CQRS —
    they don't have parent skills, only related_to peers).

related_to density discipline. The candidate list is top-K (typically
20) — its size is NOT a target. Aim for 5-10 related_to per skill:
  - If you would emit 15+ related_to, you are dumping the candidate
    list and adding low-signal links. Pick the 5-10 most genuinely
    lateral (alternatives or close complements) and stop. Do not dump.
  - If you would emit 0-2 related_to for a well-known industry skill
    (a language, a major protocol, a major testing methodology, an IAM
    standard), you are being lazy. Re-scan the candidate list — there
    are almost always 5+ legitimate laterals for any well-known skill.

Apart from these rules, default to empty arrays — don't pad
child_skills, suppress_on_match, or requires just because the
candidate list is non-empty.

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
