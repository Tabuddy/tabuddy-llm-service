"""Planner agent — LangGraph entry node.

Receives a role (``role_id`` + ``role_display`` seeded in the state), optionally
gathers 2-4 Brave Search snippets for grounding, then calls the Azure OpenAI
reasoning tier with a chain-of-thought prompt to produce a
:class:`PlannerOutput`. The output is merged into the graph state for
downstream agents (Dimension Router, Retrieval Service, Generator) to fan out
over.

Phase 1 note: Phase 2 will add the Dimension Router between the Planner and
the Generator; the ``difficulty_hint`` field on each dimension is already
emitted here so the Router's routing rule has the input it needs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from skill_library_v2.agents.base import BaseLLMAgent
from skill_library_v2.prompts.planner import (
    PLANNER_PROMPT_VERSION,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_TEMPLATE,
    format_enum_block,
    format_web_hints_block,
)
from skill_library_v2.schemas.role import PlannerOutput, WebHint
from skill_library_v2.state import PlanGraphState, ReviewItem
from skill_library_v2.tools.web_search import BraveResult, search as web_search

logger = logging.getLogger(__name__)


class PlannerAgent(BaseLLMAgent):
    tier = "reasoning"

    def __init__(self) -> None:
        super().__init__(
            agent_name="planner",
            prompt_version=PLANNER_PROMPT_VERSION,
        )

    async def gather_web_hints(self, role_display: str) -> list[WebHint]:
        """Best-effort web grounding. Returns [] if no search provider is configured."""
        queries = [
            f'"{role_display}" responsibilities required skills',
            f'"{role_display}" job description hiring 2026',
        ]
        results_per_query: list[list[BraveResult]] = await asyncio.gather(
            *(web_search(q, count=5) for q in queries),
            return_exceptions=False,
        )

        seen: set[str] = set()
        hints: list[WebHint] = []
        for results in results_per_query:
            for r in results:
                if r.url in seen:
                    continue
                seen.add(r.url)
                hints.append(
                    WebHint(
                        title=r.title,
                        url=r.url,
                        description=r.description,
                    )
                )
                if len(hints) >= 6:
                    break
            if len(hints) >= 6:
                break
        return hints

    async def run(self, role_id: str, role_display: str) -> tuple[PlannerOutput, list[WebHint]]:
        hints = await self.gather_web_hints(role_display)
        user_prompt = PLANNER_USER_TEMPLATE.format(
            role_id=role_id,
            role_display=role_display,
            enum_block=format_enum_block(),
            web_hints_block=format_web_hints_block(
                [h.model_dump() for h in hints]
            ),
        )
        output = await self.call_json(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=PlannerOutput,
        )
        return output, hints


_planner_singleton: PlannerAgent | None = None


def _get_planner() -> PlannerAgent:
    global _planner_singleton
    if _planner_singleton is None:
        _planner_singleton = PlannerAgent()
    return _planner_singleton


async def plan_role(state: PlanGraphState) -> dict[str, Any]:
    """LangGraph node. Returns a state patch — keys are merged into the
    state by the graph runtime, preserving append-only semantics.
    """
    role_id = state.get("role_id")
    role_display = state.get("role_display")
    if not role_id or not role_display:
        raise ValueError(
            "plan_role requires 'role_id' and 'role_display' in the seed state."
        )

    agent = _get_planner()
    logger.info("[planner] planning role %r (%s)", role_display, role_id)

    output, hints = await agent.run(role_id=role_id, role_display=role_display)

    review_additions: list[ReviewItem] = [
        {
            "item_type": "dimension_flagged",
            "payload": {"dimension_id": dim_id, "role_id": role_id},
            "reason": "Planner could not cleanly map this dimension to the "
                      "skill_category enum.",
        }
        for dim_id in output.flagged_for_review
    ]

    logger.info(
        "[planner] emitted %d dimensions (%d flagged) for role %s",
        len(output.dimensions),
        len(output.flagged_for_review),
        role_id,
    )

    return {
        "prompt_version": PLANNER_PROMPT_VERSION,
        "model_snapshot": agent.model_name,
        "role_archetype": output.role_archetype,
        "dimensions": output.dimensions,
        "planner_reasoning": output.reasoning,
        "planner_web_hints": hints,
        "review_queue": (state.get("review_queue") or []) + review_additions,
    }
