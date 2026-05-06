"""One-shot migration: purge a role's old role-prefixed dimension rows
and re-fire Stage 8 so the new global-slug + dedup loader writes the
canonical rows fresh.

Usage:
    uv run python scripts/migrate_dim_slugs.py <role_slug>

Example:
    uv run python scripts/migrate_dim_slugs.py frontend-engineer

The DELETEs run in a single transaction so the catalog can never sit
in a half-purged state. After the purge commits, Stage 8 is re-fired
on its own connection — that part is idempotent (ON CONFLICT
everywhere), so re-running the script is safe.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from skill_library_v3.db.connection import connect
from skill_library_v3.runner import run_stage_8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate_dim_slugs")


def purge_role_prefixed_dims(role_slug: str) -> int:
    """Delete every dim row (and its M:N edges) whose slug starts with
    ``{role_slug}--``. Returns the number of dim rows purged."""
    pattern = f"{role_slug}--%"
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM dimensions WHERE slug LIKE %s",
                (pattern,),
            )
            ids = [r[0] for r in cur.fetchall()]
            if not ids:
                logger.info("no role-prefixed dims for %s; nothing to purge", role_slug)
                return 0
            logger.info(
                "purging %d dim rows for %s + their join-table edges",
                len(ids), role_slug,
            )
            # Order matters: delete children before parents.
            cur.execute(
                "DELETE FROM dimension_skills WHERE dimension_id = ANY(%s)",
                (ids,),
            )
            cur.execute(
                "DELETE FROM dimension_categories WHERE dimension_id = ANY(%s)",
                (ids,),
            )
            cur.execute(
                "DELETE FROM role_dimensions WHERE dimension_id = ANY(%s)",
                (ids,),
            )
            cur.execute(
                "DELETE FROM dimensions WHERE id = ANY(%s)",
                (ids,),
            )
        conn.commit()
    return len(ids)


def purge_stage8_run_log(role_slug: str) -> int:
    """Drop the role's Stage 8 v2_run_log row(s) so the rerun creates a
    fresh row instead of the polling endpoint showing a stale 'approved'
    status that doesn't reflect the catalog state."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM v2_review_queue
                 WHERE run_id IN (
                     SELECT run_id FROM v2_run_log
                      WHERE role_id = %s
                        AND prompt_version LIKE %s
                 )
                """,
                (role_slug, "stage8_load_%"),
            )
            cur.execute(
                """
                DELETE FROM v2_run_log
                 WHERE role_id = %s
                   AND prompt_version LIKE %s
                """,
                (role_slug, "stage8_load_%"),
            )
            n = cur.rowcount or 0
        conn.commit()
    return n


def main(role_slug: str) -> None:
    purged_dims = purge_role_prefixed_dims(role_slug)
    purged_runs = purge_stage8_run_log(role_slug)
    logger.info(
        "purge complete: %d dim rows + %d stage8 run-log rows",
        purged_dims, purged_runs,
    )
    logger.info("re-firing Stage 8 for %s ...", role_slug)
    asyncio.run(run_stage_8(role_slug))
    logger.info("done")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
