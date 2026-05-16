"""Environment-backed settings for skill_library_v3.

Mirrors :mod:`skill_library_v2.config` but isolates the v3 DB DSN so the
new pipeline cannot accidentally write into the legacy `tabuddy` database.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


# Allow the env file to be overridden at process-launch time so a single
# checkout can be pointed at prod, staging, or local without editing `.env`.
# Set ENV_FILE=.env.prod (etc.) in the shell before invoking uvicorn / scripts.
_ENV_FILE = os.getenv("ENV_FILE", ".env")

# Force-load the chosen env file into os.environ with override=True. Pydantic
# Settings reads os.environ BEFORE env_file, so without this any stale value
# in the shell (e.g. SKILL_LIBRARY_PG_DSN left over from a prior session)
# would beat the env_file values silently.
load_dotenv(_ENV_FILE, override=True)


class V3Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    skill_library_pg_dsn: str = (
        "postgresql://postgres:postgres@localhost:5432/skill-library"
    )
    # Optional non-default Postgres schema to put on search_path for v3 queries.
    # Local dev: leave empty — local DB is itself named ``skill-library`` with
    # tables in ``public``. Prod: set to ``skill-library`` so the unqualified
    # table refs in repository.py resolve to the prod ``skill-library`` schema
    # of the shared ``postgres`` database.
    skill_library_schema: str = ""
    kaggle_username: str | None = None
    kaggle_key: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> V3Settings:
    return V3Settings()
