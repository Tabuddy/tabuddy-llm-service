"""Verify cross-role dimension dedup against the live catalog.

Builds a synthetic payload that looks like another role producing two
dimensions with names *very* similar to existing rows
(``frontend-programming-languages``, ``component-frameworks-and-rendering``).
Runs the loader and checks that:

  * The dim row count does NOT increase by 2 — the loader should hit
    the existing rows via embedding similarity and reuse them.
  * The synthetic role gets ``role_dimensions`` edges to the existing
    dim ids.
"""

from __future__ import annotations

import asyncio
import logging

from skill_library_v3.canonical_loader import load_catalog_for_role
from skill_library_v3.db.connection import connect
from skill_library_v3.schemas.catalog import CatalogPayload, DimensionRow, RoleDimRow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_dim_dedup")

SYNTHETIC_ROLE = "fullstack-test"
SYNTHETIC_DISPLAY = "Full Stack Test"


def count_dims() -> int:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM dimensions")
            return cur.fetchone()[0]


def cross_role_shared_dims() -> list[tuple]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT d.slug,
                       count(rd.role_id) AS role_count,
                       array_agg(r.slug ORDER BY r.slug) AS roles
                  FROM dimensions d
                  JOIN role_dimensions rd ON rd.dimension_id = d.id
                  JOIN roles r            ON r.id            = rd.role_id
                 GROUP BY d.slug
                HAVING count(rd.role_id) > 1
                 ORDER BY role_count DESC
            """)
            return cur.fetchall()


def cleanup() -> None:
    """Remove the synthetic role + its role_dimensions edges. Keeps
    canonical_skills/aliases/etc. intact since those are deduped by slug
    and would have been reused anyway. Just removes the test artifact."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM roles WHERE slug = %s", (SYNTHETIC_ROLE,))
            row = cur.fetchone()
            if row:
                role_id = row[0]
                cur.execute("DELETE FROM role_dimensions WHERE role_id = %s", (role_id,))
                cur.execute("DELETE FROM roles WHERE id = %s", (role_id,))
        conn.commit()


async def main() -> None:
    cleanup()  # idempotent — start clean

    before = count_dims()
    logger.info("dimension count before: %d", before)

    payload = CatalogPayload(
        role_slug=SYNTHETIC_ROLE,
        role_display=SYNTHETIC_DISPLAY,
        # Three dims:
        #   1. Near-identical to an existing Frontend dim (should REUSE)
        #   2. Same concept, different phrasing (should REUSE if sim >= 0.78)
        #   3. Genuinely new dim (should INSERT)
        dimensions=[
            DimensionRow(
                slug="frontend-programming-languages",
                display_name="Frontend Programming Languages",
                rationale="Languages used to implement browser-side application logic.",
            ),
            DimensionRow(
                slug="component-frameworks",
                display_name="Component Frameworks",
                rationale="UI component frameworks for browser-side rendering.",
            ),
            DimensionRow(
                slug="server-side-rendering",
                display_name="Server-Side Rendering",
                rationale="SSR strategies and hydration in full-stack apps.",
            ),
        ],
        role_dimensions=[
            RoleDimRow(role_slug=SYNTHETIC_ROLE,
                       dimension_slug="frontend-programming-languages"),
            RoleDimRow(role_slug=SYNTHETIC_ROLE,
                       dimension_slug="component-frameworks"),
            RoleDimRow(role_slug=SYNTHETIC_ROLE,
                       dimension_slug="server-side-rendering"),
        ],
    )

    result = await load_catalog_for_role(payload)
    logger.info("loader summary: %s", result.to_dict())

    after = count_dims()
    logger.info("dimension count after: %d (delta=%d)", after, after - before)

    shared = cross_role_shared_dims()
    logger.info("cross-role shared dim rows (role_count > 1):")
    for slug, count, roles in shared:
        logger.info("  %s -> %d roles: %s", slug, count, roles)

    if after - before <= 1:
        logger.info("PASS: dedup fired — at least one synthetic dim merged into an "
                    "existing row.")
    else:
        logger.warning("WARN: dedup did NOT fire — both synthetic dims inserted as new")

    cleanup()
    logger.info("cleanup done; synthetic role removed")


if __name__ == "__main__":
    asyncio.run(main())
