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
from urllib.parse import urlparse

from skill_library_v2.agents.base import BaseLLMAgent
from skill_library_v2.prompts.planner import (
    PLANNER_PROMPT_VERSION,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_TEMPLATE,
    format_enum_block,
    format_page_extracts_block,
    format_web_hints_block,
)
from skill_library_v2.schemas.role import PageExtract, PlannerOutput, WebHint
from skill_library_v2.state import PlanGraphState, ReviewItem
from skill_library_v2.tools.web_search import (
    BraveResult,
    fetch_extracts,
    search as web_search,
)

logger = logging.getLogger(__name__)


# Aggregators, opinion forums, and content farms that dominate SEO for
# career-advice queries but carry almost no signal for a skill Planner.
_BLOCKED_DOMAINS: frozenset[str] = frozenset({
    "indeed.com",
    "ziprecruiter.com",
    "glassdoor.com",
    "simplyhired.com",
    "monster.com",
    "reddit.com",
    "quora.com",
    "medium.com",
    "substack.com",
    "coursera.org",
    "udemy.com",
    "linkedin.com",
})

# Authoritative technical sources — hits here get sorted to the top of the
# hint list so the LLM sees them first within its attention window.
_PREFERRED_DOMAIN_SUBSTRINGS: tuple[str, ...] = (
    "roadmap.sh",
    "stackoverflow.blog",
    "github.com",
    "github.io",
    "developers.google.com",
    "learn.microsoft.com",
    "aws.amazon.com",
    "engineering.",
    ".dev/blog",
    "onetonline.org",
    "levels.fyi",
    "lever.co",
    "greenhouse.io",
    "workable.com",
)


def _host(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return netloc[4:] if netloc.startswith("www.") else netloc


def _is_blocked(url: str) -> bool:
    host = _host(url)
    if not host:
        return True
    return any(host == d or host.endswith("." + d) for d in _BLOCKED_DOMAINS)


def _preference_rank(url: str) -> int:
    host = _host(url)
    for i, sub in enumerate(_PREFERRED_DOMAIN_SUBSTRINGS):
        if sub in host or sub in url.lower():
            return i
    return len(_PREFERRED_DOMAIN_SUBSTRINGS)


class PlannerAgent(BaseLLMAgent):
    tier = "reasoning"

    def __init__(self) -> None:
        super().__init__(
            agent_name="planner",
            prompt_version=PLANNER_PROMPT_VERSION,
        )

    async def gather_web_hints(self, role_display: str) -> list[WebHint]:
        """Best-effort web grounding. Returns [] if no search provider is configured.

        Queries target *technical* sources (roadmaps, engineering blogs, real job
        postings) rather than career-advice aggregators; results from aggregator
        and opinion-forum domains are filtered out before ranking.
        """
        queries = [
            f'"{role_display}" roadmap technical skills tech stack',
            f'"{role_display}" engineering blog required technologies',
            f'"{role_display}" job posting requirements technologies framework',
        ]
        results_per_query: list[list[BraveResult]] = await asyncio.gather(
            *(web_search(q, count=10) for q in queries),
            return_exceptions=False,
        )

        seen: set[str] = set()
        candidates: list[BraveResult] = []
        raw_total = 0
        for results in results_per_query:
            raw_total += len(results)
            for r in results:
                if r.url in seen or _is_blocked(r.url):
                    continue
                seen.add(r.url)
                candidates.append(r)

        candidates.sort(key=lambda r: _preference_rank(r.url))
        top = candidates[:6]

        logger.info(
            "[planner] web hints: %d raw → %d after dedupe+blocklist → %d after cap",
            raw_total, len(candidates), len(top),
        )
        for r in top:
            logger.debug("[planner] hint: %s (%s)", r.url, _host(r.url))

        return [
            WebHint(title=r.title, url=r.url, description=r.description)
            for r in top
        ]

    async def run(
        self, role_id: str, role_display: str,
    ) -> tuple[PlannerOutput, list[WebHint], list[PageExtract]]:
        hints = await self.gather_web_hints(role_display)
        extracts = await fetch_extracts(hints) if hints else []
        hint_dicts = [h.model_dump() for h in hints]
        user_prompt = PLANNER_USER_TEMPLATE.format(
            role_id=role_id,
            role_display=role_display,
            enum_block=format_enum_block(),
            web_hints_block=format_web_hints_block(hint_dicts),
            page_extracts_block=format_page_extracts_block(
                [e.model_dump() for e in extracts],
                hints=hint_dicts,
            ),
        )
        output = await self.call_json(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=PlannerOutput,
        )
        return output, hints, extracts


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

    output, hints, extracts = await agent.run(
        role_id=role_id, role_display=role_display,
    )

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
        "planner_page_extracts": extracts,
        # The `review_queue` reducer appends — return only the new additions.
        "review_queue": review_additions,
    }
