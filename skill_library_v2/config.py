"""Environment-backed settings for skill_library_v2.

Reads from .env via pydantic-settings (already a repo dep). Kept minimal so
the pipeline can boot with just AZURE_OPEN_AI_KEY + PG_DSN; BRAVE_API_KEY is
optional and downstream code must degrade gracefully when it is absent.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class V2Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    azure_open_ai_key: str | None = None
    brave_api_key: str | None = None
    firecrawl_api_key: str | None = None
    pg_dsn: str = "postgresql://postgres:postgres@localhost:5432/tabuddy"
    reasoning_deployment: str = "o4-mini"
    fast_deployment: str = "gpt-4o-mini"


@lru_cache(maxsize=1)
def get_settings() -> V2Settings:
    return V2Settings()
