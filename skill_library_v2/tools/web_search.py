"""Unified web-search dispatcher for skill_library_v2.

Preference order per call:

1. Brave Search, if ``BRAVE_API_KEY`` is configured.
2. Firecrawl Search, if ``FIRECRAWL_API_KEY`` is configured.
3. Empty list (pipeline is expected to tolerate missing grounding).

Both backends return :class:`BraveResult`, so callers (Planner today;
Retrieval Service and Existence Critic tomorrow) treat the result shape
uniformly.
"""

from __future__ import annotations

import logging

from skill_library_v2.config import get_settings
from skill_library_v2.tools.brave_search import BraveResult
from skill_library_v2.tools.brave_search import close_client as _close_brave
from skill_library_v2.tools.brave_search import search as _brave_search
from skill_library_v2.tools.firecrawl_search import close_client as _close_firecrawl
from skill_library_v2.tools.firecrawl_search import search as _firecrawl_search

logger = logging.getLogger(__name__)


async def search(
    query: str,
    count: int = 10,
    freshness: str | None = None,
) -> list[BraveResult]:
    """Run a single web search via the first configured provider.

    ``freshness`` is only honored by Brave; Firecrawl ignores it.
    """
    settings = get_settings()

    if settings.brave_api_key:
        results = await _brave_search(query, count=count, freshness=freshness)
        if results:
            return results
        if settings.firecrawl_api_key:
            logger.info("Brave returned 0 hits for %r; falling back to Firecrawl.", query)
            return await _firecrawl_search(query, count=count)
        return results

    if settings.firecrawl_api_key:
        return await _firecrawl_search(query, count=count)

    logger.warning(
        "No web-search provider configured (BRAVE_API_KEY / FIRECRAWL_API_KEY); "
        "skipping search for %r.", query,
    )
    return []


async def close_clients() -> None:
    """Release HTTP clients for both backends (idempotent)."""
    await _close_brave()
    await _close_firecrawl()


__all__ = ["BraveResult", "search", "close_clients"]
