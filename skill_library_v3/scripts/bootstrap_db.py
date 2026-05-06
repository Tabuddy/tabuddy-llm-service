"""One-shot DB bootstrap for the new ``skill-library`` Postgres database.

Applies, in order:
  1. db/schema.sql                          (existing canonical schema)
  2. db/schema_v2_additions.sql             (v2 run log + review queue)
  3. skill_library_v3/db/v3_additions.sql   (index + jd_samples)

Idempotent for v3_additions (CREATE TABLE IF NOT EXISTS / CREATE INDEX
IF NOT EXISTS). Steps 1+2 are NOT idempotent — they raise on a
pre-existing schema. Pass ``--skip-base`` to apply only the v3 additions
on top of an already-bootstrapped DB.

Usage:
    uv run python -m skill_library_v3.scripts.bootstrap_db
    uv run python -m skill_library_v3.scripts.bootstrap_db --skip-base
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import psycopg2
import psycopg2.errors

from skill_library_v3.config import get_settings

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_FILES_BASE = (
    REPO_ROOT / "db" / "schema.sql",
    REPO_ROOT / "db" / "schema_v2_additions.sql",
)
SCHEMA_FILES_V3 = (
    REPO_ROOT / "skill_library_v3" / "db" / "v3_additions.sql",
)


def _apply_sql_file(dsn: str, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    logger.info("[v3 bootstrap] applying %s (%d bytes)", path.name, len(sql))
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip db/schema.sql + db/schema_v2_additions.sql; apply only v3_additions.sql.",
    )
    args = parser.parse_args(argv)

    dsn = get_settings().skill_library_pg_dsn
    logger.info("[v3 bootstrap] target DSN: %s", _redact(dsn))

    files: tuple[Path, ...] = SCHEMA_FILES_V3 if args.skip_base else (
        *SCHEMA_FILES_BASE,
        *SCHEMA_FILES_V3,
    )
    for path in files:
        if not path.exists():
            logger.error("[v3 bootstrap] missing schema file: %s", path)
            return 1
        try:
            _apply_sql_file(dsn, path)
        except psycopg2.errors.DuplicateTable as exc:
            logger.error(
                "[v3 bootstrap] %s already applied (%s). "
                "Re-run with --skip-base if the canonical schema is already in place.",
                path.name,
                exc.diag.message_primary,
            )
            return 2
        except psycopg2.Error as exc:
            logger.exception("[v3 bootstrap] failed on %s: %s", path.name, exc)
            return 3

    logger.info("[v3 bootstrap] done.")
    return 0


def _redact(dsn: str) -> str:
    """Strip credentials from a DSN for log output."""
    if "@" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1) if "://" in dsn else ("postgresql", dsn)
    after_at = rest.rsplit("@", 1)[-1]
    return f"{scheme}://***@{after_at}"


if __name__ == "__main__":
    sys.exit(main())
