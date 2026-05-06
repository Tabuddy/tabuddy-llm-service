"""psycopg2 connection helper for the skill_library_v3 DB.

Mirrors the simple pattern from :mod:`skill_library_repository` — a sync
psycopg2 connect using SKILL_LIBRARY_PG_DSN. Async callers wrap calls with
``asyncio.to_thread`` (the same idiom canonical_skill_api.py uses).

On prod, the catalog lives in a non-default schema (``skill-library``) inside
the shared ``postgres`` database. v3's repository uses unqualified table
names, so we set ``search_path`` per connection when ``SKILL_LIBRARY_SCHEMA``
is set. Locally the env var is unset, the v3 DB is itself named
``skill-library`` with tables in ``public``, and no SET is issued.
"""

from __future__ import annotations

import logging

import psycopg2

from skill_library_v3.config import get_settings

logger = logging.getLogger(__name__)

_LOGGED_ONCE = False


def connect():
    """Open a fresh psycopg2 connection to the v3 (skill-library) database."""
    global _LOGGED_ONCE
    settings = get_settings()
    conn = psycopg2.connect(settings.skill_library_pg_dsn)
    schema = (settings.skill_library_schema or "").strip()
    if schema:
        # Include ``dev`` after the user schema and before ``public``: prod
        # has the pgvector extension installed in ``dev``, so without ``dev``
        # on the path ``'...'::vector`` casts fail with "type 'vector' does
        # not exist". Postgres silently ignores nonexistent schemas in
        # search_path, so this is a no-op on local.
        with conn.cursor() as cur:
            cur.execute(
                'SET search_path TO "{schema}", dev, public'.format(
                    schema=schema.replace('"', '""')
                )
            )
        conn.commit()
    if not _LOGGED_ONCE:
        logger.info(
            "[v3-db] connected; search_path schema=%r "
            "(empty = default public; otherwise '<schema>, dev, public')",
            schema,
        )
        _LOGGED_ONCE = True
    return conn
