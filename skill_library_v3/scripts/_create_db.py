"""One-shot helper: CREATE DATABASE 'skill-library' if it doesn't exist.

Connects to the maintenance ``postgres`` database on the same host as
SKILL_LIBRARY_PG_DSN (since you can't ``CREATE DATABASE`` while connected to
the target database itself). Uses autocommit because CREATE DATABASE rejects
running inside a transaction.

Usage:
    uv run python -m skill_library_v3.scripts._create_db
"""

from __future__ import annotations

import logging
import sys
from urllib.parse import urlparse, urlunparse

import psycopg2
import psycopg2.extensions

from skill_library_v3.config import get_settings

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    target_dsn = get_settings().skill_library_pg_dsn
    parsed = urlparse(target_dsn)
    target_db = (parsed.path or "/").lstrip("/")
    if not target_db:
        logger.error("SKILL_LIBRARY_PG_DSN is missing a database name")
        return 1

    # Swap to the maintenance DB on the same host to issue CREATE DATABASE.
    admin_dsn = urlunparse(parsed._replace(path="/postgres"))

    conn = psycopg2.connect(admin_dsn)
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
            if cur.fetchone():
                logger.info("[v3] database %r already exists", target_db)
                return 0
            # Identifier interpolation: the database name has a hyphen, so
            # double-quote it. We can't parameterise identifiers in psycopg2,
            # but target_db came from our own config, not user input.
            cur.execute(f'CREATE DATABASE "{target_db}"')
            logger.info("[v3] created database %r", target_db)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
