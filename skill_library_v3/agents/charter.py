"""Stage 0 charter agent.

Subclasses :class:`skill_library_v2.agents.base.BaseLLMAgent` to inherit JSON
parse + corrective retry + tier client selection + the parameter-restriction
handling that o-series models need.
"""

from __future__ import annotations

from skill_library_v2.agents.base import BaseLLMAgent
from skill_library_v3.prompts.charter import (
    CHARTER_PROMPT_VERSION,
    CHARTER_SYSTEM_PROMPT,
    CHARTER_USER_TEMPLATE,
    format_adjacent_role_list,
    format_jd_samples_block,
)
from skill_library_v3.schemas.charter import CharterOutput


class Stage0CharterAgent(BaseLLMAgent):
    """Generates a role charter from a role name + adjacent-role context.

    tier=generation → gpt-5.4-mini (overridable via GENERATION_DEPLOYMENT).
    The 2025 generation model has the breadth to articulate fresh boundaries
    for emerging roles (MLOps vs DevOps, AI Engineer vs ML Engineer) where
    o4-mini's older training cutoff produces stale framings.
    """

    tier = "generation"

    def __init__(self) -> None:
        super().__init__(
            agent_name="stage0_charter",
            prompt_version=CHARTER_PROMPT_VERSION,
        )

    async def generate(
        self,
        *,
        role_slug: str,
        role_name: str,
        adjacent_roles: list[dict],
        jd_samples: list[dict] | None = None,
    ) -> CharterOutput:
        user_prompt = CHARTER_USER_TEMPLATE.format(
            role_name=role_name,
            role_slug=role_slug,
            adjacent_role_list=format_adjacent_role_list(adjacent_roles),
            jd_samples=format_jd_samples_block(jd_samples or []),
        )
        result = await self.call_json(
            system_prompt=CHARTER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=CharterOutput,
            temperature=0.2,  # ignored on o-series via _accepts_temperature
        )
        # Force role_id/role_name to the values we passed in. The LLM
        # occasionally rewrites the slug to its own preferred shape.
        result.role_id = role_slug
        result.role_name = role_name
        return result
