"""psycopg2 connection helper for the skill_library_v3 DB.

Mirrors the simple pattern from :mod:`skill_library_repository` — a sync
psycopg2 connect using SKILL_LIBRARY_PG_DSN. Async callers wrap calls with
``asyncio.to_thread`` (the same idiom canonical_skill_api.py uses).
"""

from __future__ import annotations

import psycopg2

from skill_library_v3.config import get_settings


def connect():
    """Open a fresh psycopg2 connection to the v3 (skill-library) database."""
    return psycopg2.connect(get_settings().skill_library_pg_dsn)
