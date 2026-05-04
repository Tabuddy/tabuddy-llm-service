"""Unified web-search dispatcher for skill_library_v2.

Preference order per call:

1. Brave Search, if ``BRAVE_API_KEY`` is configured.
2. Firecrawl Search, if ``FIRECRAWL_API_KEY`` is configured.
3. Empty list (pipeline is expected to tolerate missing grounding).

Both backends return :class:`BraveResult`, so callers (Planner today;
Retrieval Service and Existence Critic tomorrow) treat the result shape
uniformly.

Also exposes :func:`fetch_extracts` — a fan-out step that takes the top-N
ranked hints, scrapes each via Firecrawl, and runs the extractor to
produce signal-dense :class:`PageExtract` objects for the Planner prompt.
"""

from __future__ import annotations

import asyncio
import logging

from skill_library_v2.config import get_settings
from skill_library_v2.schemas.role import PageExtract, WebHint
from skill_library_v2.tools.brave_search import BraveResult
from skill_library_v2.tools.brave_search import close_client as _close_brave
from skill_library_v2.tools.brave_search import search as _brave_search
from skill_library_v2.tools.content_extractor import extract_signal
from skill_library_v2.tools.firecrawl_scrape import close_client as _close_firecrawl_scrape
from skill_library_v2.tools.firecrawl_scrape import scrape as _firecrawl_scrape
from skill_library_v2.tools.firecrawl_search import close_client as _close_firecrawl
from skill_library_v2.tools.firecrawl_search import search as _firecrawl_search

logger = logging.getLogger(__name__)

# Per-page budget and default fan-out. Kept as module-level defaults (not
# V2Settings knobs) until we have a second caller that needs different values.
_DEFAULT_SCRAPE_LIMIT = 3
_DEFAULT_PAGE_CHARS = 2500


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


async def fetch_extracts(
    hints: list[WebHint],
    limit: int = _DEFAULT_SCRAPE_LIMIT,
    per_page_chars: int = _DEFAULT_PAGE_CHARS,
) -> list[PageExtract]:
    """Scrape the top-``limit`` hints and return filtered page extracts.

    * Scrapes are issued in parallel via Firecrawl ``/v1/scrape``.
    * A scrape that fails (HTTP error, empty markdown, cached failure) or
      extracts to an empty string is dropped — no fallback to the snippet.
    * Output is ordered to match the input ``hints`` order so prompt
      citations ``[1]``, ``[2]``, ... line up with the snippet block.
    """
    targets = hints[:limit]
    if not targets:
        logger.info("[web_search] fetch_extracts: no hints to scrape.")
        return []

    logger.info("[web_search] scraping top %d/%d hints", len(targets), len(hints))

    markdowns = await asyncio.gather(
        *(_firecrawl_scrape(h.url) for h in targets),
        return_exceptions=False,
    )

    extracts: list[PageExtract] = []
    total_chars = 0
    for hint, md in zip(targets, markdowns):
        if not md:
            logger.warning("[web_search] scrape produced no markdown for %s", hint.url)
            continue
        filtered = extract_signal(md, max_chars=per_page_chars)
        if not filtered:
            logger.debug(
                "[web_search] extractor found no signal lines for %s (%d chars raw)",
                hint.url, len(md),
            )
            continue
        logger.info(
            "[web_search] scrape %s: %d chars raw → %d chars filtered",
            hint.url, len(md), len(filtered),
        )
        extracts.append(
            PageExtract(title=hint.title, url=hint.url, content=filtered)
        )
        total_chars += len(filtered)

    logger.info(
        "[web_search] page extracts: %d/%d URLs produced content (%d chars total)",
        len(extracts), len(targets), total_chars,
    )
    return extracts


async def close_clients() -> None:
    """Release HTTP clients for all backends (idempotent)."""
    await _close_brave()
    await _close_firecrawl()
    await _close_firecrawl_scrape()


__all__ = ["BraveResult", "search", "fetch_extracts", "close_clients"]
