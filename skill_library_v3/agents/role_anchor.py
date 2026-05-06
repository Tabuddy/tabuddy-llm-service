"""Stage 1 role-anchoring agent — compresses an approved Stage 0 charter
into a structured role card that downstream stages reference.

Subclasses :class:`skill_library_v2.agents.base.BaseLLMAgent` to inherit
JSON parse + corrective retry + tier client selection.
"""

from __future__ import annotations

from skill_library_v2.agents.base import BaseLLMAgent
from skill_library_v3.prompts.role_card import (
    ROLE_CARD_PROMPT_VERSION,
    ROLE_CARD_SYSTEM_PROMPT,
    ROLE_CARD_USER_TEMPLATE,
    format_charter_block,
    format_family_enum,
)
from skill_library_v3.schemas.role_card import RoleCard


class Stage1RoleAnchorAgent(BaseLLMAgent):
    """Generates a role card from an approved charter.

    tier=generation -> gpt-5.4-mini (matches Stage 0 charter agent). The
    2025 generation model has up-to-date framings for emerging roles like
    AI Engineer / GenAI Engineer that the role card needs to bucket
    correctly into the closed family enum.
    """

    tier = "generation"

    def __init__(self) -> None:
        super().__init__(
            agent_name="stage1_role_anchor",
            prompt_version=ROLE_CARD_PROMPT_VERSION,
        )

    async def generate(
        self,
        *,
        role_slug: str,
        role_name: str,
        charter: dict,
    ) -> RoleCard:
        user_prompt = ROLE_CARD_USER_TEMPLATE.format(
            role_name=role_name,
            role_slug=role_slug,
            charter_block=format_charter_block(charter),
            family_enum=format_family_enum(),
        )
        result = await self.call_json(
            system_prompt=ROLE_CARD_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=RoleCard,
            temperature=0.2,
        )
        # Force role_id back to the slug we passed; LLMs occasionally
        # rewrite it. Keep the LLM's canonical_name unless it's empty.
        result.role_id = role_slug
        if not result.canonical_name:
            result.canonical_name = role_name
        return result
