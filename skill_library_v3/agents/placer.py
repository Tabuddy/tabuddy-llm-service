"""Stage 5 skill-placer agent — assigns each typed skill to one primary
dimension (and rarely up to 2 secondary dimensions) within a role.

Subclasses :class:`skill_library_v2.agents.base.BaseLLMAgent`. Tests
subclass this and override ``call_json`` to capture prompts.
"""

from __future__ import annotations

from skill_library_v2.agents.base import BaseLLMAgent
from skill_library_v3.prompts.placer import (
    PLACER_PROMPT_VERSION,
    PLACER_SYSTEM_PROMPT,
    PLACER_USER_TEMPLATE,
    format_dims_block,
    format_role_card_block,
    format_skills_block,
)
from skill_library_v3.schemas.placement import PlacedSkill, PlacedSkillBatch
from skill_library_v3.schemas.typology import TypedSkill


class Stage5PlacerAgent(BaseLLMAgent):
    """Places one batch of typed skills into a role's locked dimensions.

    tier=generation -> gpt-5.4-mini, consistent with the rest of v3.
    Stage 5's prompt is the most fragile in the pipeline (secondary-dim
    overuse is the #1 failure mode); the system prompt encodes the
    blueprint's reject-patterns explicitly.
    """

    tier = "generation"

    def __init__(self) -> None:
        super().__init__(
            agent_name="stage5_placer",
            prompt_version=PLACER_PROMPT_VERSION,
        )

    async def place_skills(
        self,
        *,
        typed_skills: list[TypedSkill],
        role_card: dict,
        locked_dimensions: list[dict],
    ) -> list[PlacedSkill]:
        """Place a batch of typed skills. Empty input returns ``[]``
        without calling the LLM."""
        if not typed_skills:
            return []
        # Convert TypedSkill objects to dicts for the prompt formatter.
        skill_dicts = [s.model_dump() for s in typed_skills]
        user_prompt = PLACER_USER_TEMPLATE.format(
            role_card_block=format_role_card_block(role_card),
            dims_block=format_dims_block(locked_dimensions),
            skills_block=format_skills_block(skill_dicts),
        )
        result = await self.call_json(
            system_prompt=PLACER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=PlacedSkillBatch,
            temperature=0.1,
        )
        return list(result.skills)
