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

import os

import psycopg2

from skill_library_v3.config import get_settings


def connect():
    """Open a fresh psycopg2 connection to the v3 (skill-library) database."""
    conn = psycopg2.connect(get_settings().skill_library_pg_dsn)
    schema = os.getenv("SKILL_LIBRARY_SCHEMA", "").strip()
    if schema:
        with conn.cursor() as cur:
            cur.execute(
                'SET search_path TO "{schema}", public'.format(
                    schema=schema.replace('"', '""')
                )
            )
        conn.commit()
    return conn
