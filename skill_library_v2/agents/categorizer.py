"""Categorizer agent — proposes (category, sub_category) buckets for one dim.

Runs on the ``generation`` tier with a function-calling tool loop:
``web_search`` / ``scrape_page`` / ``submit_categories`` (terminator).
"""

from __future__ import annotations

import logging

from skill_library_v2.agents.base import BaseLLMAgent
from skill_library_v2.prompts.categorizer import (
    CATEGORIZER_PROMPT_VERSION,
    CATEGORIZER_SYSTEM_PROMPT,
    CATEGORIZER_USER_TEMPLATE,
)
from skill_library_v2.schemas.category import CategorizerOutput
from skill_library_v2.schemas.role import DimensionSlice
from skill_library_v2.tools.categorizer_tools import (
    TERMINATOR_TOOL,
    TOOL_SCHEMAS,
    dispatch_tool,
)

logger = logging.getLogger(__name__)


def _format_other_dims(others: list[DimensionSlice]) -> str:
    if not others:
        return "(none)"
    return "\n".join(
        f"  - {d.dimension_id}: {d.dimension_name}" for d in others
    )


def _format_web_hints(hints: list[dict]) -> str:
    if not hints:
        return "(none)"
    return "\n".join(
        f"  - {h.get('title','?')} :: {h.get('url','?')}" for h in hints[:5]
    )


class CategorizerAgent(BaseLLMAgent):
    tier = "generation"

    def __init__(self) -> None:
        super().__init__(
            agent_name="categorizer",
            prompt_version=CATEGORIZER_PROMPT_VERSION,
        )

    async def run(
        self,
        *,
        role_id: str,
        role_display: str,
        role_archetype: str,
        dim: DimensionSlice,
        other_dims: list[DimensionSlice],
        role_web_hints: list[dict],
    ) -> CategorizerOutput:
        user_prompt = CATEGORIZER_USER_TEMPLATE.format(
            role_id=role_id,
            role_display=role_display,
            role_archetype=role_archetype or "(not set)",
            dim_id=dim.dimension_id,
            dim_name=dim.dimension_name,
            dim_rationale=dim.rationale,
            difficulty_hint=dim.difficulty_hint,
            expected_kinds=", ".join(dim.expected_skill_kinds) or "(none)",
            other_dims_block=_format_other_dims(other_dims),
            role_web_hints_block=_format_web_hints(role_web_hints),
        )
        return await self.call_with_tools(
            system_prompt=CATEGORIZER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=TOOL_SCHEMAS,
            tool_dispatch=dispatch_tool,
            terminator_tool=TERMINATOR_TOOL,
            schema=CategorizerOutput,
            max_iters=4,
            temperature=0.0,
        )


_singleton: CategorizerAgent | None = None


def _get_categorizer() -> CategorizerAgent:
    global _singleton
    if _singleton is None:
        _singleton = CategorizerAgent()
    return _singleton
