"""Skill-driven dimension generation — fork of Stage 2 for the JD pipeline.

The original Stage 2 ([skill_library_v3/agents/dimension_gen.py]) is
role-card-driven: it produces 12-20 dimensions for a whole role from
its role card. The 3-API JD pipeline needs the inverse — given ONE
unknown skill plus optional JD / role hint context, produce 1-3
candidate dimensions in the same v3 schema (CandidateDimension) so
downstream similarity / merge / placement / containment can run
unchanged.

This module mirrors the v3 prompt + agent pattern (BaseLLMAgent
subclass, versioned prompt, JSON-mode output) but with a focused
single-skill input shape.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from skill_library_v2.agents.base import BaseLLMAgent
from skill_library_v3.schemas.dimension import CandidateDimension


SKILL_DIM_GEN_PROMPT_VERSION = "skill_dim_gen_v1.0"


SKILL_DIM_GEN_SYSTEM_PROMPT = """\
You are the Skill-Driven Dimension Generator for a canonical skills
catalog. Given ONE engineering skill (and optional context from the
JD it was extracted from + the role it likely belongs to), produce
1-3 candidate dimensions that this skill belongs to.

A "dimension" is a coherent skill cluster (e.g. "Container
Orchestration", "Distributed Systems Concepts", "API Design &
Patterns"). Most skills belong to exactly ONE dimension; a few
genuinely span two areas; rarely three. Default to 1 — only emit
2-3 when the skill is clearly multi-anchored (e.g. PySpark spans
"Programming Languages" AND "Stream / Batch Processing").

Hard rules:
  * Generate EXACTLY 1, 2, or 3 dimensions. Never 0, never more than 3.
  * REUSE existing catalog dimensions when ANY listed dimension is a
    reasonable fit. Copy the existing dim's tentative_id verbatim into
    your output's tentative_id field. Do NOT invent a near-synonym
    when the catalog already has the concept.
  * Only INVENT a new dimension when no catalog entry fits. New
    dimensions get a tentative_id of the form `d_init_NN` (two-digit
    zero-padded, unique within this response, starting at d_init_01).
  * No "Miscellaneous", "Other", "General", "Misc" dimensions.
  * Avoid marketing / buzzword names ("Cloud-Native Excellence"). Name
    dimensions by their substance (e.g. "Containerization").

Per-dimension fields (mirror the v3 Stage 2 schema):
  * tentative_id: existing catalog id (verbatim) when reusing, else
    `d_init_NN`.
  * name: 3-8 word noun-phrase. Specific, not generic.
  * description: 1-3 sentences (>=20 chars, <=400 chars). What the
    dimension covers and why this skill belongs in it.
  * in_scope: comma-delimited specifics (10-400 chars) — concrete
    tools, frameworks, sub-topics that belong in this dimension. The
    target skill MUST appear in this list.
  * out_of_scope: comma-delimited specifics (10-400 chars) naming what
    is excluded and why (which other dimension owns it).
  * exemplar_skills: 3-15 concrete skill names that anchor this
    dimension. The target skill MUST appear in this list.
  * overlap_flags: zero or more flags pointing at potential overlaps
    with other catalog dimensions (use `with_dim_id` referencing the
    catalog id, with a one-sentence reason).

Emit a single JSON object matching the schema below. No prose, no
fences.

Schema:
  {
    "candidate_dimensions": [
      {
        "tentative_id": "<id>",
        "name": "...",
        "description": "...",
        "in_scope": "...",
        "out_of_scope": "...",
        "exemplar_skills": ["...", "..."],
        "overlap_flags": [
          {"with_dim_id": "<existing dim id>", "reason": "..."}
        ]
      }
    ]
  }
"""


SKILL_DIM_GEN_USER_TEMPLATE = """\
Target skill: {skill_name}

Context (optional — use if helpful, ignore if empty):
  jd_excerpt: {jd_excerpt}
  role_hint:  {role_hint}

Existing dimension catalog (REUSE when a fit exists):
{catalog_block}

Produce 1-3 candidate dimensions for {skill_name!r} now. Reuse
existing tentative_ids verbatim when a catalog dim fits. Only invent
a new dim (with a `d_init_NN` id) when nothing in the catalog fits.
"""


class SkillDimensionList(BaseModel):
    """Skill-driven dim-gen output. Same shape as v3 DimensionList but
    with the candidate_dimensions count clamped to 1-3 (role-driven
    Stage 2 produces 12-20)."""

    candidate_dimensions: list[CandidateDimension] = Field(
        min_length=1, max_length=3,
    )


def format_existing_dim_catalog(catalog: list[dict]) -> str:
    """Render the existing dim catalog as a compact list the LLM can
    cite by tentative_id. Each catalog entry should carry at least
    ``tentative_id``, ``name``, ``description`` (description is
    truncated to keep prompts small).
    """
    if not catalog:
        return "  (catalog empty — invent new dimensions starting at d_init_01)"
    lines: list[str] = []
    for d in catalog:
        dim_id = (
            d.get("tentative_id")
            or d.get("slug")
            or d.get("id")
            or "?"
        )
        name = d.get("name") or d.get("display_name") or "?"
        desc = (d.get("description") or d.get("rationale") or "").strip()
        if len(desc) > 160:
            desc = desc[:157] + "..."
        if desc:
            lines.append(f"  - {dim_id}: {name} -- {desc}")
        else:
            lines.append(f"  - {dim_id}: {name}")
    return "\n".join(lines)


class SkillDrivenDimGenAgent(BaseLLMAgent):
    """Generates 1-3 candidate dimensions for ONE unknown skill.

    Mirrors :class:`skill_library_v3.agents.dimension_gen.Stage2DimensionGeneratorAgent`
    but with a single-skill input shape. Uses tier=generation so the
    LLM has up-to-date breadth on emerging tools / frameworks.
    """

    tier = "generation"

    def __init__(self) -> None:
        super().__init__(
            agent_name="skill_driven_dim_gen",
            prompt_version=SKILL_DIM_GEN_PROMPT_VERSION,
        )

    async def generate(
        self,
        *,
        skill_name: str,
        jd_excerpt: str = "",
        role_hint: str = "",
        existing_dims: list[dict] | None = None,
    ) -> list[CandidateDimension]:
        catalog = existing_dims or []
        user_prompt = SKILL_DIM_GEN_USER_TEMPLATE.format(
            skill_name=skill_name,
            jd_excerpt=(jd_excerpt or "").strip() or "(none)",
            role_hint=(role_hint or "").strip() or "(none)",
            catalog_block=format_existing_dim_catalog(catalog),
        )
        result = await self.call_json(
            system_prompt=SKILL_DIM_GEN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=SkillDimensionList,
            temperature=0.3,
        )
        return list(result.candidate_dimensions)


__all__ = [
    "SKILL_DIM_GEN_PROMPT_VERSION",
    "SKILL_DIM_GEN_SYSTEM_PROMPT",
    "SKILL_DIM_GEN_USER_TEMPLATE",
    "SkillDimensionList",
    "SkillDrivenDimGenAgent",
    "format_existing_dim_catalog",
]
