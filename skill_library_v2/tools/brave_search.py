"""Async Brave Search client for skill_library_v2.

Used by the Planner to gather role context and by downstream validators
(Existence Critic, Retrieval Service) to verify evidence URLs. Degrades
gracefully to ``[]`` when ``BRAVE_API_KEY`` is not configured — Brave is
augmentation, not a hard dependency.

Docs: https://api.search.brave.com/app/documentation/web-search/get-started
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from skill_library_v2.config import get_settings

logger = logging.getLogger(__name__)

_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_CACHE_TTL_SECONDS = 3600   # 1 hour
_CACHE_MAX_ENTRIES = 512

_client_lock = asyncio.Lock()
_async_client: httpx.AsyncClient | None = None


@dataclass(frozen=True, slots=True)
class BraveResult:
    title: str
    url: str
    description: str
    age: str | None = None


# ── In-process LRU cache keyed by query fingerprint ──────────────────────────
_cache: dict[str, tuple[float, list[BraveResult]]] = {}


def _cache_key(query: str, count: int, freshness: str | None) -> str:
    payload = f"{query.strip().lower()}|{count}|{freshness or ''}"
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
    """Optional teardown, e.g. on FastAPI shutdown."""
    global _async_client
    if _async_client is not None:
        await _async_client.aclose()
        _async_client = None


def _parse_results(payload: dict[str, Any], limit: int) -> list[BraveResult]:
    web = (payload.get("web") or {}).get("results") or []
    out: list[BraveResult] = []
    for item in web[:limit]:
        url = item.get("url")
        title = item.get("title")
        if not url or not title:
            continue
        out.append(
            BraveResult(
                title=str(title),
                url=str(url),
                description=str(item.get("description") or ""),
                age=item.get("age"),
            )
        )
    return out


async def search(
    query: str,
    count: int = 10,
    freshness: str | None = None,
) -> list[BraveResult]:
    """Run a single Brave web search.

    Args:
        query: Free-text query string.
        count: Number of results to return (1-20).
        freshness: Optional recency filter (e.g. ``"pw"`` = past week,
            ``"pm"`` = past month, ``"py"`` = past year).

    Returns:
        Up to ``count`` :class:`BraveResult` entries. Returns ``[]`` (with a
        warning log) when ``BRAVE_API_KEY`` is unset or Brave is unreachable
        after retries — callers must treat an empty result as "no hint",
        never as an error.
    """
    settings = get_settings()
    if not settings.brave_api_key:
        logger.warning("BRAVE_API_KEY not set; skipping Brave search for %r.", query)
        return []

    count = max(1, min(int(count), 20))
    key = _cache_key(query, count, freshness)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    params: dict[str, Any] = {"q": query, "count": count}
    if freshness:
        params["freshness"] = freshness
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": settings.brave_api_key,
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
                resp = await client.get(_BRAVE_ENDPOINT, params=params, headers=headers)
                if resp.status_code == 429:
                    logger.warning("Brave rate-limited; backing off.")
                    raise httpx.RemoteProtocolError("rate limited")
                if resp.status_code >= 400:
                    logger.warning(
                        "Brave returned HTTP %s for %r: %s",
                        resp.status_code, query, resp.text[:200],
                    )
                    return []
                payload = resp.json()
                results = _parse_results(payload, count)
                _cache_put(key, results)
                return results
    except Exception as exc:
        logger.exception("Brave search failed for %r: %s", query, exc)
        return []

    return []
