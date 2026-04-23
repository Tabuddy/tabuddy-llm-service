"""Async Firecrawl search client — fallback grounding provider.

Used when ``BRAVE_API_KEY`` is not configured but ``FIRECRAWL_API_KEY`` is.
Returns results shaped as :class:`BraveResult` so the Planner doesn't care
which backend served the hits. Degrades gracefully to ``[]`` on any failure.

Docs: https://docs.firecrawl.dev/features/search
"""

from __future__ import annotations

import asyncio
import hashlib
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
from skill_library_v2.tools.brave_search import BraveResult

logger = logging.getLogger(__name__)

_FIRECRAWL_ENDPOINT = "https://api.firecrawl.dev/v1/search"
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
_CACHE_TTL_SECONDS = 3600
_CACHE_MAX_ENTRIES = 512

_client_lock = asyncio.Lock()
_async_client: httpx.AsyncClient | None = None

_cache: dict[str, tuple[float, list[BraveResult]]] = {}


def _cache_key(query: str, count: int) -> str:
    payload = f"firecrawl|{query.strip().lower()}|{count}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> list[BraveResult] | None:
    hit = _cache.get(key)
    if hit is None:
        return None
    ts, results = hit
    if time.monotonic() - ts > _CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return results


def _cache_put(key: str, results: list[BraveResult]) -> None:
    if len(_cache) >= _CACHE_MAX_ENTRIES:
        oldest = sorted(_cache.items(), key=lambda kv: kv[1][0])[: _CACHE_MAX_ENTRIES // 4]
        for k, _ in oldest:
            _cache.pop(k, None)
    _cache[key] = (time.monotonic(), results)


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


def _parse_results(payload: dict[str, Any], limit: int) -> list[BraveResult]:
    data = payload.get("data")
    if isinstance(data, dict):
        data = data.get("web") or data.get("results") or []
    if not isinstance(data, list):
        return []
    out: list[BraveResult] = []
    for item in data[:limit]:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("link")
        title = item.get("title")
        if not url or not title:
            continue
        description = (
            item.get("description")
            or item.get("snippet")
            or item.get("summary")
            or ""
        )
        out.append(
            BraveResult(
                title=str(title),
                url=str(url),
                description=str(description),
                age=None,
            )
        )
    return out


async def search(query: str, count: int = 10) -> list[BraveResult]:
    """Run a single Firecrawl search.

    Returns ``[]`` (with a warning log) if ``FIRECRAWL_API_KEY`` is unset or
    Firecrawl is unreachable after retries.
    """
    settings = get_settings()
    if not settings.firecrawl_api_key:
        logger.warning("FIRECRAWL_API_KEY not set; skipping Firecrawl search for %r.", query)
        return []

    count = max(1, min(int(count), 20))
    key = _cache_key(query, count)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    body = {"query": query, "limit": count}
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
                resp = await client.post(_FIRECRAWL_ENDPOINT, json=body, headers=headers)
                if resp.status_code == 429:
                    logger.warning("Firecrawl rate-limited; backing off.")
                    raise httpx.RemoteProtocolError("rate limited")
                if resp.status_code >= 400:
                    logger.warning(
                        "Firecrawl returned HTTP %s for %r: %s",
                        resp.status_code, query, resp.text[:200],
                    )
                    return []
                payload = resp.json()
                if payload.get("success") is False:
                    logger.warning("Firecrawl reported failure for %r: %s", query, payload)
                    return []
                results = _parse_results(payload, count)
                _cache_put(key, results)
                return results
    except Exception as exc:
        logger.exception("Firecrawl search failed for %r: %s", query, exc)
        return []

    return []
