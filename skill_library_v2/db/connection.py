"""Async Postgres plumbing for skill_library_v2.

One module-level ``asyncpg.Pool`` singleton, lazily created on first use.
Helper functions wrap the two writes Phase 1 cares about:

- :func:`record_run` — insert/update a row in ``v2_run_log``.
- :func:`enqueue_review` — bulk-insert rows into ``v2_review_queue``.

Both are intentionally small; Phase 2+ will add reads against
``canonical_skills`` / ``skill_aliases`` when the Retrieval Service lands.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Sequence
from uuid import UUID

import asyncpg

from skill_library_v2.config import get_settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()


async def get_pool() -> asyncpg.Pool:
    """Lazily create (or return) the shared asyncpg connection pool."""
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is None:
            dsn = get_settings().pg_dsn
            logger.info("Creating asyncpg pool for skill_library_v2 (dsn host omitted).")
            _pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=1,
                max_size=10,
                command_timeout=30,
            )
    assert _pool is not None
    return _pool


async def close_pool() -> None:
    """Optional teardown — call from FastAPI shutdown or CLI exit."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# ─── v2_run_log ──────────────────────────────────────────────────────────────

_INSERT_RUN_SQL = """
INSERT INTO v2_run_log (
    run_id, role_id, role_display, prompt_version, model_snapshot,
    planner_output, planner_reasoning, status, started_at
) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, NOW())
ON CONFLICT (run_id) DO UPDATE
   SET planner_output    = EXCLUDED.planner_output,
       planner_reasoning = EXCLUDED.planner_reasoning,
       status            = EXCLUDED.status,
       prompt_version    = EXCLUDED.prompt_version,
       model_snapshot    = EXCLUDED.model_snapshot;
"""

_COMPLETE_RUN_SQL = """
UPDATE v2_run_log
   SET status = $2,
       completed_at = NOW(),
       error_message = $3
 WHERE run_id = $1;
"""


async def record_run(
    *,
    run_id: UUID | str,
    role_id: str,
    role_display: str,
    prompt_version: str,
    model_snapshot: str,
    planner_output: dict[str, Any],
    planner_reasoning: str,
    status: str = "planned",
) -> None:
    """Upsert a run row with the Planner's output and 'planned' status."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            _INSERT_RUN_SQL,
            _as_uuid(run_id),
            role_id,
            role_display,
            prompt_version,
            model_snapshot,
            json.dumps(planner_output, default=_json_default),
            planner_reasoning,
            status,
        )


async def mark_run_complete(
    run_id: UUID | str,
    *,
    status: str = "complete",
    error_message: str | None = None,
) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(_COMPLETE_RUN_SQL, _as_uuid(run_id), status, error_message)


# ─── v2_review_queue ─────────────────────────────────────────────────────────

_INSERT_REVIEW_SQL = """
INSERT INTO v2_review_queue (run_id, item_type, payload, reason)
VALUES ($1, $2, $3::jsonb, $4);
"""


async def enqueue_review(
    run_id: UUID | str | None,
    items: Sequence[dict[str, Any]],
) -> int:
    """Bulk insert review-queue entries; returns the count written."""
    if not items:
        return 0
    pool = await get_pool()
    run_uuid = _as_uuid(run_id) if run_id is not None else None
    rows = [
        (
            run_uuid,
            str(item.get("item_type", "unknown")),
            json.dumps(item.get("payload", {}), default=_json_default),
            str(item.get("reason", "")),
        )
        for item in items
    ]
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(_INSERT_REVIEW_SQL, rows)
    return len(rows)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _as_uuid(value: UUID | str) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def _json_default(obj: Any) -> Any:
    # asyncpg wants JSON-safe types; Pydantic models arrive as objects.
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")
