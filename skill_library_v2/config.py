"""Environment-backed settings for skill_library_v2.

Reads from .env via pydantic-settings (already a repo dep). Kept minimal so
the pipeline can boot with just AZURE_OPEN_AI_KEY + PG_DSN; BRAVE_API_KEY is
optional and downstream code must degrade gracefully when it is absent.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


# Same convention as V3Settings — ENV_FILE overrides the default `.env`.
_ENV_FILE = os.getenv("ENV_FILE", ".env")
# Inject env file values into os.environ with override so they beat any
# stale values inherited from the shell. See V3Settings for the long version.
load_dotenv(_ENV_FILE, override=True)


class V2Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    azure_open_ai_key: str | None = None
    brave_api_key: str | None = None
    firecrawl_api_key: str | None = None
    pg_dsn: str = "postgresql://postgres:postgres@localhost:5432/tabuddy"
    reasoning_deployment: str = "o4-mini"
    fast_deployment: str = "gpt-5.4-mini"
    generation_deployment: str = "gpt-5.4"


@lru_cache(maxsize=1)
def get_settings() -> V2Settings:
    return V2Settings()
