"""Stage 4 type-assigner agent — assigns one TypedSkill per input skill,
batched to keep LLM calls efficient.

Subclasses :class:`skill_library_v2.agents.base.BaseLLMAgent` to inherit
JSON parse + corrective retry + tier client selection. Tests subclass
this agent and override ``call_json`` to verify prompt construction.
"""

from __future__ import annotations

from skill_library_v2.agents.base import BaseLLMAgent
from skill_library_v3.prompts.type_assigner import (
    TYPE_ASSIGNER_PROMPT_VERSION,
    TYPE_ASSIGNER_SYSTEM_PROMPT,
    TYPE_ASSIGNER_USER_TEMPLATE,
    format_skills_batch_block,
    format_typology_block,
)
from skill_library_v3.schemas.typology import TypedSkill, TypedSkillBatch


class Stage4TypeAssignerAgent(BaseLLMAgent):
    """Types one batch of skills (50-100 per call per blueprint).

    tier=generation -> gpt-5.4-mini, consistent with the rest of v3.
    Stage 4 typing benefits from current-world knowledge of tools and
    platforms (e.g. recognizing emerging AI runtimes), so generation
    is the right tier.
    """

    tier = "generation"

    def __init__(self) -> None:
        super().__init__(
            agent_name="stage4_type_assigner",
            prompt_version=TYPE_ASSIGNER_PROMPT_VERSION,
        )

    async def assign_types(self, *, skills: list[dict]) -> list[TypedSkill]:
        """Type a batch of skills. Empty batch returns ``[]`` without
        calling the LLM."""
        if not skills:
            return []
        system_prompt = TYPE_ASSIGNER_SYSTEM_PROMPT.format(
            typology_block=format_typology_block(),
        )
        user_prompt = TYPE_ASSIGNER_USER_TEMPLATE.format(
            skills_block=format_skills_batch_block(skills),
        )
        result = await self.call_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=TypedSkillBatch,
            temperature=0.1,
        )
        return list(result.skills)
