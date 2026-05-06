"""Bootstrap a fresh ``skill-library`` schema on the prod Postgres DB.

What this does:
  1. Connect to ``PG_PROD`` (read from .env).
  2. ``CREATE SCHEMA IF NOT EXISTS "skill-library"``.
  3. Set ``search_path`` to ``"skill-library", public`` so:
       - new types/tables/indexes land in skill-library
       - ``vector(1536)`` resolves via the existing ``vector`` extension in public
       - ``gin_trgm_ops`` resolves via ``pg_trgm`` in public
  4. Apply ``db/schema.sql`` (with the two CREATE EXTENSION lines stripped —
     extensions are already installed on prod by an admin).
  5. Apply ``db/schema_v2_additions.sql``.
  6. Apply ``skill_library_v3/db/v3_additions.sql`` (jd_samples + two run-log
     partial indexes used by Stage 0 / Stage 1 lookups).
  7. Verify the expected table set.

Safe to re-run: every statement is idempotent (CREATE TABLE IF NOT EXISTS,
CREATE TYPE wrapped in DO blocks for the new ones, ON CONFLICT helpers).
The original ``schema.sql`` uses bare CREATE TABLE / CREATE TYPE — second
runs of those will error, so the script bails early when the schema is
already populated.

Does NOT migrate data. That is a separate task.
"""

from __future__ import annotations

import logging
import os
import re
import sys

import psycopg2
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bootstrap_prod_skill_library")

SCHEMA_NAME = "skill-library"

EXPECTED_TABLES = {
    "categories", "sub_categories", "canonical_skills", "skill_aliases",
    "roles", "dimensions", "role_dimensions", "dimension_skills",
    "dimension_categories", "skill_tags", "skill_relationships",
    "v2_run_log", "v2_review_queue", "role_aliases",
    "jd_samples",
}
EXPECTED_TYPES = {
    "skill_nature", "skill_volatility", "skill_lifespan", "version_strategy",
    "entity_source", "alias_type", "match_strategy", "relationship_type",
    "relationship_direction", "license_type",
}


# Strip the two CREATE EXTENSION statements from schema.sql before
# applying — extensions already exist on prod and the connecting user
# isn't a superuser, so re-creating them would fail.
_EXTENSION_LINE_RE = re.compile(
    r"^CREATE\s+EXTENSION\s+IF\s+NOT\s+EXISTS.*?;",
    re.IGNORECASE | re.MULTILINE,
)


def _read_sql(path: str, *, strip_extensions: bool) -> str:
    with open(path, encoding="utf-8") as f:
        sql = f.read()
    if strip_extensions:
        sql = _EXTENSION_LINE_RE.sub("-- (extension line stripped — installed in public)", sql)
    return sql


def _table_set(cur, schema: str) -> set[str]:
    cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = %s",
        (schema,),
    )
    return {r[0] for r in cur.fetchall()}


def _type_set(cur, schema: str) -> set[str]:
    cur.execute(
        "SELECT t.typname FROM pg_type t "
        "JOIN pg_namespace n ON n.oid = t.typnamespace "
        "WHERE n.nspname = %s AND t.typtype = 'e'",
        (schema,),
    )
    return {r[0] for r in cur.fetchall()}


def main() -> None:
    load_dotenv()
    dsn = os.environ.get("PG_PROD")
    if not dsn:
        logger.error("PG_PROD env var not set")
        sys.exit(2)

    sql_main = _read_sql("db/schema.sql", strip_extensions=True)
    sql_addn = _read_sql("db/schema_v2_additions.sql", strip_extensions=True)
    sql_v3 = _read_sql(
        "skill_library_v3/db/v3_additions.sql",
        strip_extensions=True,
    )

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
                (SCHEMA_NAME,),
            )
            schema_exists = cur.fetchone() is not None
            if schema_exists:
                existing = _table_set(cur, SCHEMA_NAME)
                if existing:
                    logger.error(
                        "schema %r already has %d tables (%s) — refusing to "
                        "re-apply DDL on a populated schema. Drop / pick a "
                        "different schema name first.",
                        SCHEMA_NAME, len(existing),
                        sorted(existing)[:5],
                    )
                    sys.exit(3)
                logger.info("schema %r exists but is empty — proceeding", SCHEMA_NAME)
            else:
                cur.execute(f'CREATE SCHEMA "{SCHEMA_NAME}"')
                logger.info("created schema %r", SCHEMA_NAME)

            # Set search_path for THIS session only. Order matters:
            #   1. skill-library — where new types/tables/indexes land.
            #   2. dev          — where the prod admin installed the
            #                     ``vector`` extension (per pg_extension).
            #   3. public       — where ``pg_trgm`` is installed; safe
            #                     fallback for anything else.
            cur.execute(f'SET search_path TO "{SCHEMA_NAME}", dev, public')

            logger.info("applying db/schema.sql ...")
            cur.execute(sql_main)
            logger.info("applying db/schema_v2_additions.sql ...")
            cur.execute(sql_addn)
            logger.info("applying skill_library_v3/db/v3_additions.sql ...")
            cur.execute(sql_v3)

            tables = _table_set(cur, SCHEMA_NAME)
            types = _type_set(cur, SCHEMA_NAME)
            missing_tables = EXPECTED_TABLES - tables
            missing_types = EXPECTED_TYPES - types
            if missing_tables:
                logger.error("missing tables: %s", sorted(missing_tables))
            if missing_types:
                logger.error("missing types: %s", sorted(missing_types))
            if missing_tables or missing_types:
                conn.rollback()
                sys.exit(4)
            logger.info(
                "verified: %d tables, %d enum types in schema %r",
                len(tables), len(types), SCHEMA_NAME,
            )
        conn.commit()
    logger.info("done — prod %r schema is ready", SCHEMA_NAME)


if __name__ == "__main__":
    main()
