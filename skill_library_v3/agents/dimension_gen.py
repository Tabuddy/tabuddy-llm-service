"""Stage 2 dimension-generation agent — produces 12-20 candidate dimensions
from an approved role card.

Subclasses :class:`skill_library_v2.agents.base.BaseLLMAgent` to inherit
JSON parse + corrective retry + tier client selection.
"""

from __future__ import annotations

from skill_library_v2.agents.base import BaseLLMAgent
from skill_library_v3.prompts.dimension_gen import (
    DIM_GEN_PROMPT_VERSION,
    DIM_GEN_SYSTEM_PROMPT,
    DIM_GEN_USER_TEMPLATE,
    format_adjacent_dim_block,
    format_always_load_block,
    format_role_card_block,
)
from skill_library_v3.schemas.dimension import DimensionList


class Stage2DimensionGeneratorAgent(BaseLLMAgent):
    """Generates a candidate dimension list from a role card.

    tier=generation -> gpt-5.4-mini. Same tier as Stages 0/1; the model's
    breadth across emerging roles + 2025 tooling is what makes the
    coverage axes work.
    """

    tier = "generation"

    def __init__(self) -> None:
        super().__init__(
            agent_name="stage2_dimension_generator",
            prompt_version=DIM_GEN_PROMPT_VERSION,
        )

    async def generate(
        self,
        *,
        role_slug: str,
        role_name: str,
        role_card: dict,
        always_load_skills: list[str],
        adjacent_dimensions_by_role: dict[str, list[dict]],
    ) -> DimensionList:
        user_prompt = DIM_GEN_USER_TEMPLATE.format(
            role_name=role_name,
            role_slug=role_slug,
            role_card_block=format_role_card_block(role_card),
            always_load_block=format_always_load_block(always_load_skills),
            adjacent_dim_block=format_adjacent_dim_block(
                adjacent_dimensions_by_role
            ),
        )
        result = await self.call_json(
            system_prompt=DIM_GEN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=DimensionList,
            temperature=0.3,
        )
        # Force role_id back to the slug we passed.
        result.role_id = role_slug
        return result
