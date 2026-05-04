"""Skill Generator agent — fills each Categorizer-proposed bucket with concrete skills.

LangGraph node: ``generate_skills`` (invoked by ``dim_worker.process_dim``).
Runs on the ``generation`` tier with a function-calling tool loop:
``web_search`` / ``scrape_page`` / ``submit_skills`` (terminator).
"""

from __future__ import annotations

import logging

from skill_library_v2.agents.base import BaseLLMAgent
from skill_library_v2.prompts.skill_generator import (
    SKILL_GENERATOR_PROMPT_VERSION,
    SKILL_GENERATOR_SYSTEM_PROMPT,
    SKILL_GENERATOR_USER_TEMPLATE,
    format_categorizer_proposals,
)
from skill_library_v2.schemas.category import CategorizerOutput
from skill_library_v2.schemas.role import DimensionSlice
from skill_library_v2.schemas.skill import SkillGeneratorOutput
from skill_library_v2.tools.skill_generator_tools import (
    TERMINATOR_TOOL,
    TOOL_SCHEMAS,
    dispatch_tool,
)

logger = logging.getLogger(__name__)


def _format_already(already: dict[str, list[str]]) -> str:
    if not already:
        return "(none)"
    return "\n".join(
        f"  - {dim}: {len(slugs)} slug(s) e.g. {slugs[:5]}"
        for dim, slugs in already.items()
    )


def _format_corrections(corrs: list[dict]) -> str:
    if not corrs:
        return "(none)"
    return "\n".join(
        f"  - {c.get('slug')}: {c.get('issue_type')} — {c.get('suggested_fix')}"
        for c in corrs
    )


def _format_hints(hints: list[dict]) -> str:
    if not hints:
        return "(none)"
    return "\n".join(
        f"  - {h.get('title','?')} :: {h.get('url','?')}" for h in hints[:5]
    )


class SkillGeneratorAgent(BaseLLMAgent):
    tier = "generation"

    def __init__(self) -> None:
        super().__init__(
            agent_name="skill_generator",
            prompt_version=SKILL_GENERATOR_PROMPT_VERSION,
        )

    async def run(
        self,
        *,
        role_id: str,
        role_display: str,
        role_archetype: str,
        dim: DimensionSlice,
        categorizer_output: CategorizerOutput,
        already_emitted: dict[str, list[str]],
        corrections: list[dict],
        role_web_hints: list[dict],
    ) -> SkillGeneratorOutput:
        proposals_dicts = [p.model_dump() for p in categorizer_output.proposals]
        user_prompt = SKILL_GENERATOR_USER_TEMPLATE.format(
            role_id=role_id,
            role_display=role_display,
            role_archetype=role_archetype or "(not set)",
            dim_id=dim.dimension_id,
            dim_name=dim.dimension_name,
            dim_rationale=dim.rationale,
            categorizer_proposals_block=format_categorizer_proposals(proposals_dicts),
            already_emitted_block=_format_already(already_emitted),
            corrections_block=_format_corrections(corrections),
            role_web_hints_block=_format_hints(role_web_hints),
        )
        return await self.call_with_tools(
            system_prompt=SKILL_GENERATOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=TOOL_SCHEMAS,
            tool_dispatch=dispatch_tool,
            terminator_tool=TERMINATOR_TOOL,
            schema=SkillGeneratorOutput,
            max_iters=6,
            temperature=0.0,
        )


_singleton: SkillGeneratorAgent | None = None


def _get_skill_generator() -> SkillGeneratorAgent:
    global _singleton
    if _singleton is None:
        _singleton = SkillGeneratorAgent()
    return _singleton
