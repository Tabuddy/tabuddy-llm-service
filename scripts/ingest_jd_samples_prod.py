"""Run the v3 JD-sample ingest against the **production** ``skill-library``
schema.

Wrapper around ``skill_library_v3.scripts.ingest_jd_samples`` that:
  1. Reads ``PG_PROD`` from .env (the prod Postgres DSN — same value the
     bootstrap_prod_skill_library.py script uses).
  2. Overrides ``SKILL_LIBRARY_PG_DSN`` and ``SKILL_LIBRARY_SCHEMA`` *before*
     any ``skill_library_v3`` import, so the cached ``V3Settings`` picks up
     the prod values.
  3. Dispatches to the existing ingest ``main()`` — argv passes through, so
     all upstream flags (``--inspect-only``, ``--title-col``, ``--body-col``,
     ``--csv``, ``--max-rows``, ``--backfill-only``) work unchanged.

Usage (from repo root):
    # Dry-run: download CSV and print columns + first row, no DB writes
    uv run python scripts/ingest_jd_samples_prod.py --inspect-only

    # Real ingest (autodetects title/body columns)
    uv run python scripts/ingest_jd_samples_prod.py

    # Backfill only (re-fuzzy-match unmatched rows after adding more roles)
    uv run python scripts/ingest_jd_samples_prod.py --backfill-only

Auth:
  - kagglehub picks up KAGGLE_USERNAME/KAGGLE_KEY env vars or
    ``~/.kaggle/kaggle.json`` automatically. If neither is configured the
    download fails with a link to https://www.kaggle.com/docs/api.
  - PG_PROD must be set in ``.env`` (this is what the bootstrap script
    already requires).

Schema target: ``skill-library`` schema in the prod ``postgres`` database.
The connection helper applies ``SET search_path TO "skill-library", public``
on connect, so ingest writes land in the right place.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Allow running as `python scripts/ingest_jd_samples_prod.py` without
# requiring PYTHONPATH=. — put the repo root on sys.path so v3 imports
# resolve. (Other scripts in this directory rely on PYTHONPATH being set;
# this one fixes itself for convenience.)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv


def _bootstrap_prod_env() -> None:
    """Point SKILL_LIBRARY_* env vars at prod, BEFORE any v3 import."""
    load_dotenv()
    pg_prod = os.environ.get("PG_PROD", "").strip()
    if not pg_prod:
        sys.stderr.write(
            "PG_PROD is not set in .env — required to target the prod "
            "skill-library DB. Aborting.\n"
        )
        sys.exit(2)
    os.environ["SKILL_LIBRARY_PG_DSN"] = pg_prod
    os.environ["SKILL_LIBRARY_SCHEMA"] = "skill-library"


def main() -> int:
    _bootstrap_prod_env()

    # Late import: the ingest module pulls in skill_library_v3.config, which
    # caches V3Settings via @lru_cache. We must not import it before the env
    # overrides above are in place.
    from skill_library_v3.config import get_settings
    from skill_library_v3.scripts.ingest_jd_samples import main as ingest_main

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    redacted = settings.skill_library_pg_dsn
    if "@" in redacted:
        scheme, rest = redacted.split("://", 1) if "://" in redacted else ("postgresql", redacted)
        redacted = f"{scheme}://***@{rest.rsplit('@', 1)[-1]}"
    logging.info(
        "[ingest-prod] target DSN: %s | schema: %r",
        redacted,
        settings.skill_library_schema,
    )

    return ingest_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
