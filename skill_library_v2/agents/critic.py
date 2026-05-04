"""Skill Critic agent — reviews one Generator batch for a single dimension.

LangGraph node: ``critic_skills``. Runs on the ``reasoning`` tier
(o4-mini), a different family from the Generator's gpt-5.4, which per
v2 §5 produces a genuinely independent second opinion.

No tool use — the Critic only needs the generator's output + role/dim
context.
"""

from __future__ import annotations

import logging
from typing import Any

from skill_library_v2.agents.base import BaseLLMAgent
from skill_library_v2.prompts.critic import (
    CRITIC_PROMPT_VERSION,
    CRITIC_SYSTEM_PROMPT,
    CRITIC_USER_TEMPLATE,
    format_categorizer_proposals_for_critic,
    format_skills_for_critic,
)
from skill_library_v2.schemas.category import CategorizerOutput
from skill_library_v2.schemas.role import DimensionSlice
from skill_library_v2.schemas.skill import CriticOutput, SkillDraft
from skill_library_v2.state import PlanGraphState

logger = logging.getLogger(__name__)


class CriticAgent(BaseLLMAgent):
    tier = "reasoning"

    def __init__(self) -> None:
        super().__init__(
            agent_name="critic",
            prompt_version=CRITIC_PROMPT_VERSION,
        )

    async def run(
        self,
        *,
        role_id: str,
        role_display: str,
        dim: DimensionSlice,
        skills: list[SkillDraft],
        categorizer_output: CategorizerOutput,
        generator_notes: str,
    ) -> CriticOutput:
        skills_json = format_skills_for_critic(
            [s.model_dump() for s in skills]
        )
        proposals_block = format_categorizer_proposals_for_critic(
            [p.model_dump() for p in categorizer_output.proposals]
        )
        user_prompt = CRITIC_USER_TEMPLATE.format(
            role_display=role_display,
            role_id=role_id,
            dim_id=dim.dimension_id,
            dim_name=dim.dimension_name,
            dim_rationale=dim.rationale,
            expected_kinds=", ".join(dim.expected_skill_kinds) or "(none)",
            categorizer_proposals_block=proposals_block,
            skills_json=skills_json,
            generator_notes=generator_notes or "(none)",
        )
        return await self.call_json(
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=CriticOutput,
        )


_critic_singleton: CriticAgent | None = None


def _get_critic() -> CriticAgent:
    global _critic_singleton
    if _critic_singleton is None:
        _critic_singleton = CriticAgent()
    return _critic_singleton


async def critic_skills(state: PlanGraphState) -> dict[str, Any]:
    """LangGraph node: run Critic for one dimension's generator output."""
    dim_id: str = state["dim_id"]
    dim = DimensionSlice.model_validate(state["dim"])

    role_id: str = state["role_id"]
    role_display: str = state["role_display"]

    generated = state.get("generated") or {}
    skills: list[SkillDraft] = generated.get(dim_id) or []

    # If the Generator failed this dim, the Critic has nothing to critique.
    if not skills:
        logger.info(
            "[critic] dim=%s role=%s: no skills to critique (generator failed?)",
            dim_id, role_id,
        )
        return {
            "dim_critic_verdict": {dim_id: "reject"},
            "dim_critic_corrections": {dim_id: []},
        }

    gen_notes = (state.get("dim_generator_notes") or {}).get(dim_id, "")

    agent = _get_critic()
    logger.info(
        "[critic] dim=%s role=%s: reviewing %d skills",
        dim_id, role_id, len(skills),
    )

    try:
        output: CriticOutput = await agent.run(
            role_id=role_id,
            role_display=role_display,
            dim=dim,
            skills=skills,
            generator_notes=gen_notes,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[critic] dim=%s role=%s FAILED: %s",
            dim_id, role_id, exc,
        )
        # Critic failure is non-fatal — treat as approve so validator can
        # still run its deterministic checks.
        return {
            "dim_critic_verdict": {dim_id: "approve"},
            "dim_critic_corrections": {dim_id: []},
            "errors": (state.get("errors") or []) + [
                {"node": "critic_skills",
                 "message": f"{type(exc).__name__}: {exc}",
                 "retryable": False}
            ],
        }

    logger.info(
        "[critic] dim=%s role=%s: verdict=%s corrections=%d",
        dim_id, role_id, output.overall_verdict, len(output.corrections),
    )

    return {
        "dim_critic_verdict": {dim_id: output.overall_verdict},
        "dim_critic_corrections": {dim_id: output.corrections},
    }
