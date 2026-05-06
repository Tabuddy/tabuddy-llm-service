"""Environment-backed settings for skill_library_v3.

Mirrors :mod:`skill_library_v2.config` but isolates the v3 DB DSN so the
new pipeline cannot accidentally write into the legacy `tabuddy` database.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class V3Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    skill_library_pg_dsn: str = (
        "postgresql://postgres:postgres@localhost:5432/skill-library"
    )
    kaggle_username: str | None = None
    kaggle_key: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> V3Settings:
    return V3Settings()
