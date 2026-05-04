"""Async Firecrawl scrape client — fetch cleaned markdown for a single URL.

Used by the Planner's page-extract step. Firecrawl renders JS, strips
chrome, and returns clean markdown, so SPA-heavy authoritative sources
(roadmap.sh, Greenhouse ATS pages) actually produce readable content.
Degrades gracefully to ``None`` when ``FIRECRAWL_API_KEY`` is unset or the
URL cannot be scraped.

Docs: https://docs.firecrawl.dev/features/scrape
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from skill_library_v2.config import get_settings
from skill_library_v2.tools.exceptions import (
    FirecrawlCreditsExhausted,
    firecrawl_credits_tripped,
    trip_firecrawl_credits,
)

logger = logging.getLogger(__name__)

_FIRECRAWL_SCRAPE_ENDPOINT = "https://api.firecrawl.dev/v1/scrape"
_TIMEOUT = httpx.Timeout(45.0, connect=5.0)
_CACHE_TTL_SECONDS = 24 * 3600   # 24 hours
_CACHE_MAX_ENTRIES = 256

_client_lock = asyncio.Lock()
_async_client: httpx.AsyncClient | None = None

# In-process URL → (timestamp, markdown|None). ``None`` is cached too so a
# known-failing URL doesn't burn credits on re-runs within the TTL.
_cache: dict[str, tuple[float, str | None]] = {}


def _cache_get(url: str) -> tuple[bool, str | None]:
    """Return (hit, value). ``value`` may be ``None`` (cached failure)."""
    entry = _cache.get(url)
    if entry is None:
        return (False, None)
    ts, val = entry
    if time.monotonic() - ts > _CACHE_TTL_SECONDS:
        _cache.pop(url, None)
        return (False, None)
    return (True, val)


def _cache_put(url: str, value: str | None) -> None:
    if len(_cache) >= _CACHE_MAX_ENTRIES:
        oldest = sorted(_cache.items(), key=lambda kv: kv[1][0])[: _CACHE_MAX_ENTRIES // 4]
        for k, _ in oldest:
            _cache.pop(k, None)
    _cache[url] = (time.monotonic(), value)


async def _get_client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is not None:
        return _async_client
    async with _client_lock:
        if _async_client is None:
            _async_client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _async_client


async def close_client() -> None:
    global _async_client
    if _async_client is not None:
        await _async_client.aclose()
        _async_client = None


def _extract_markdown(payload: dict[str, Any]) -> str | None:
    data = payload.get("data")
    if isinstance(data, dict):
        md = data.get("markdown")
        if isinstance(md, str) and md.strip():
            return md
    return None


async def scrape(url: str) -> str | None:
    """Fetch ``url`` via Firecrawl and return cleaned markdown.

    Returns ``None`` (with a warning log) when:
      * ``FIRECRAWL_API_KEY`` is not configured
      * Firecrawl returns a 4xx / 5xx after retries
      * The response contains no ``markdown`` field or an empty one

    The return value is cached for 24h keyed by URL — repeat runs for the
    same role don't burn scrape credits on the same page twice.
    """
    settings = get_settings()
    if not settings.firecrawl_api_key:
        logger.warning("FIRECRAWL_API_KEY not set; skipping scrape for %s.", url)
        return None

    if firecrawl_credits_tripped():
        raise FirecrawlCreditsExhausted(
            "Firecrawl quota exhausted earlier in this run; aborting scrape to "
            "preserve a clean checkpoint stop."
        )

    hit, cached = _cache_get(url)
    if hit:
        return cached

    body = {"url": url, "formats": ["markdown"]}
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.firecrawl_api_key}",
    }

    client = await _get_client()
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.TransportError, httpx.RemoteProtocolError)),
            reraise=True,
        ):
            with attempt:
                resp = await client.post(
                    _FIRECRAWL_SCRAPE_ENDPOINT, json=body, headers=headers,
                )
                if resp.status_code == 429:
                    logger.warning("Firecrawl scrape rate-limited; backing off.")
                    raise httpx.RemoteProtocolError("rate limited")
                if resp.status_code == 402:
                    body = resp.text[:200]
                    trip_firecrawl_credits(body)
                    raise FirecrawlCreditsExhausted(
                        f"Firecrawl quota exhausted (HTTP 402) on scrape: {body}"
                    )
                if resp.status_code >= 400:
                    logger.warning(
                        "Firecrawl scrape returned HTTP %s for %s: %s",
                        resp.status_code, url, resp.text[:200],
                    )
                    _cache_put(url, None)
                    return None
                payload = resp.json()
                if payload.get("success") is False:
                    logger.warning("Firecrawl scrape reported failure for %s: %s", url, payload)
                    _cache_put(url, None)
                    return None
                markdown = _extract_markdown(payload)
                if markdown is None:
                    logger.warning("Firecrawl scrape returned no markdown for %s.", url)
                _cache_put(url, markdown)
                return markdown
    except FirecrawlCreditsExhausted:
        raise  # propagate to the batch runner for a clean checkpoint stop
    except Exception as exc:
        logger.exception("Firecrawl scrape failed for %s: %s", url, exc)
        _cache_put(url, None)
        return None

    return None
