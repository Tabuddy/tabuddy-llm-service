"""FastAPI router for the /canonical-skill UI + JSON APIs.

Browses the relational skill catalog (categories, sub_categories,
canonical_skills, skill_aliases, roles, dimensions + their join tables)
produced by the skill_library_v2 pipeline.

**Always reads from prod.** Pin DSN + schema explicitly so this UI cannot
accidentally hit a local DB even if ``DATABASE_URL`` / ``SKILL_LIBRARY_SCHEMA``
are set for the rest of the app.

Resolution order (first non-empty wins):
  - ``CANONICAL_SKILL_DB_URL``  (explicit override)
  - ``PG_PROD``                 (default — prod connection)

Schema:
  - ``CANONICAL_SKILL_DB_SCHEMA`` (override)
  - default ``dev`` (prod catalog lives in the ``dev`` schema)

Routes:
    GET  /canonical-skill                                → HTML
    GET  /api/canonical-skill/stats                      → JSON counts
    GET  /api/canonical-skill/roles                      → JSON list
    GET  /api/canonical-skill/roles/{id_or_slug}         → JSON detail
    GET  /api/canonical-skill/dimensions                 → JSON list (filterable)
    GET  /api/canonical-skill/dimensions/{id_or_slug}    → JSON detail
    GET  /api/canonical-skill/skills                     → JSON list (filterable)
    GET  /api/canonical-skill/skills/{id_or_slug}        → JSON detail
    GET  /api/canonical-skill/categories                 → JSON list
"""
from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from skill_library_repository import SkillLibraryRepository

logger = logging.getLogger(__name__)

canonical_skill_router = APIRouter(tags=["canonical-skill"])

_templates = Jinja2Templates(directory="templates")


@lru_cache(maxsize=1)
def _repo() -> SkillLibraryRepository:
    db_url = (
        os.getenv("CANONICAL_SKILL_DB_URL")
        or os.getenv("PG_PROD")
        or ""
    )
    schema = os.getenv("CANONICAL_SKILL_DB_SCHEMA", "dev")
    if not db_url:
        raise RuntimeError(
            "PG_PROD (or CANONICAL_SKILL_DB_URL) must be set — the "
            "/canonical-skill UI is read-only on the prod catalog."
        )
    db_url = _normalize_postgres_url(db_url)
    logger.info(
        "[canonical-skill] connecting to prod (schema=%s, host=%s)",
        schema, _redact_host(db_url),
    )
    return SkillLibraryRepository(database_url=db_url, schema=schema)


def _normalize_postgres_url(url: str) -> str:
    """Tolerate unencoded special chars in the password.

    libpq (psycopg2) follows RFC 3986: it splits userinfo/host on the FIRST
    '@', so a raw '@' in the password breaks DSN parsing (you get errors
    like "could not translate host name \"p3R@1.2.3.4\""). Many .env values
    are pasted with the raw password. We split on the LAST '@' (the actual
    userinfo/host boundary) and percent-encode the password if needed.
    Already-encoded chars (%XX) are preserved.
    """
    from urllib.parse import quote

    if "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "@" not in rest:
        return url

    userinfo, host_etc = rest.rsplit("@", 1)
    if ":" not in userinfo:
        return url

    user, password = userinfo.split(":", 1)
    # safe='%' keeps existing %XX escapes intact (idempotent on a clean URL).
    encoded = quote(password, safe="%")
    if encoded == password:
        return url  # already clean
    logger.info(
        "[canonical-skill] auto-encoded special chars in PG_PROD password"
    )
    return f"{scheme}://{user}:{encoded}@{host_etc}"


def _redact_host(url: str) -> str:
    """Pull the host:port out of a Postgres URL for log lines without leaking
    credentials. Best-effort — falls back to '?' if parsing fails."""
    try:
        after_at = url.rsplit("@", 1)[-1]
        return after_at.split("/", 1)[0]
    except Exception:
        return "?"


# ── HTML shell ──────────────────────────────────────────────────────────────


@canonical_skill_router.get("/canonical-skill", response_class=HTMLResponse)
async def canonical_skill_page(request: Request):
    return _templates.TemplateResponse(request, "canonical_skill.html", {})


# ── JSON APIs ───────────────────────────────────────────────────────────────


@canonical_skill_router.get("/api/canonical-skill/stats")
async def api_stats() -> dict[str, Any]:
    return await asyncio.to_thread(_repo().get_catalog_stats)


@canonical_skill_router.get("/api/canonical-skill/roles")
async def api_list_roles() -> dict[str, Any]:
    items = await asyncio.to_thread(_repo().list_roles)
    return {"items": items, "total": len(items)}


@canonical_skill_router.get("/api/canonical-skill/roles/{id_or_slug}")
async def api_get_role(id_or_slug: str) -> dict[str, Any]:
    detail = await asyncio.to_thread(_repo().get_role_detail, id_or_slug)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"role not found: {id_or_slug}")
    return detail


@canonical_skill_router.get("/api/canonical-skill/dimensions")
async def api_list_dimensions(
    role_id: int | None = Query(default=None),
    q: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _repo().list_dimensions,
        role_id=role_id, q=q, limit=limit, offset=offset,
    )


@canonical_skill_router.get("/api/canonical-skill/dimensions/{id_or_slug}")
async def api_get_dimension(id_or_slug: str) -> dict[str, Any]:
    detail = await asyncio.to_thread(_repo().get_dimension_detail, id_or_slug)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"dimension not found: {id_or_slug}")
    return detail


@canonical_skill_router.get("/api/canonical-skill/skills")
async def api_list_skills(
    dimension_id: int | None = Query(default=None),
    category_id: int | None = Query(default=None),
    q: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _repo().list_skills,
        dimension_id=dimension_id, category_id=category_id,
        q=q, limit=limit, offset=offset,
    )


@canonical_skill_router.get("/api/canonical-skill/skills/{id_or_slug}")
async def api_get_skill(id_or_slug: str) -> dict[str, Any]:
    detail = await asyncio.to_thread(_repo().get_skill_detail, id_or_slug)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"skill not found: {id_or_slug}")
    return detail


@canonical_skill_router.get("/api/canonical-skill/categories")
async def api_list_categories() -> dict[str, Any]:
    items = await asyncio.to_thread(_repo().list_categories)
    return {"items": items, "total": len(items)}
