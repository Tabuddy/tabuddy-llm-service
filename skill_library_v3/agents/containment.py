"""Stage 6 containment-resolution agent — produces one
SkillRelationships record per skill, given top-K similar candidates.

Per-skill (not batched): each skill has its own candidate list, so
batching adds prompt complexity without meaningful efficiency.
"""

from __future__ import annotations

from skill_library_v2.agents.base import BaseLLMAgent
from skill_library_v3.prompts.containment import (
    CONTAINMENT_PROMPT_VERSION,
    CONTAINMENT_SYSTEM_PROMPT,
    CONTAINMENT_USER_TEMPLATE,
    format_candidates_block,
    format_target_block,
)
from skill_library_v3.schemas.placement import PlacedSkill
from skill_library_v3.schemas.relationships import SkillRelationships
from skill_library_v3.schemas.typology import TypedSkill


class Stage6ContainmentAgent(BaseLLMAgent):
    """One LLM call per skill. Returns SkillRelationships."""

    tier = "nano"

    def __init__(self) -> None:
        super().__init__(
            agent_name="stage6_containment",
            prompt_version=CONTAINMENT_PROMPT_VERSION,
        )

    async def resolve(
        self,
        *,
        typed: TypedSkill,
        placed: PlacedSkill,
        candidates: list[dict],
    ) -> SkillRelationships:
        """Resolve one skill's relationships. ``candidates`` are the
        top-K similar skills (from skill_similarity.top_k_similar_skills);
        the LLM picks parent / child / requires / related_to /
        suppress_on_match from this list (or omits)."""
        user_prompt = CONTAINMENT_USER_TEMPLATE.format(
            target_block=format_target_block(typed.model_dump(), placed.model_dump()),
            candidates_block=format_candidates_block(candidates),
        )
        result = await self.call_json(
            system_prompt=CONTAINMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=SkillRelationships,
            temperature=0.1,
        )
        # Force the skill_id back to the target's so any LLM echo-drift
        # gets corrected.
        if result.skill_id != typed.skill_id:
            result = SkillRelationships(
                skill_id=typed.skill_id,
                parent_skills=result.parent_skills,
                child_skills=result.child_skills,
                suppress_on_match=result.suppress_on_match,
                requires=result.requires,
                related_to=result.related_to,
            )
        return result
