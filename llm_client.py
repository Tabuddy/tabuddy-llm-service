"""Centralized Azure OpenAI client factory.

Provides two client tiers:
  - **fast** (gpt-4o-mini): block tagging, skill normalization, categorization
  - **reasoning** (o4-mini): JD atom extraction, role-fit scoring, section scoring
"""

from __future__ import annotations

import os
import logging
from openai import AsyncAzureOpenAI, AzureOpenAI

logger = logging.getLogger(__name__)

_AZURE_ENDPOINT = "https://tabuddy-azure-sponsor.openai.azure.com/"
_AZURE_API_VERSION = "2024-12-01-preview"

FAST_MODEL = "gpt-4o-mini"
REASONING_MODEL = "o4-mini"

# ── Singletons ────────────────────────────────────────────────────────────────
_fast_async: AsyncAzureOpenAI | None = None
_reasoning_async: AsyncAzureOpenAI | None = None
_fast_sync: AzureOpenAI | None = None


def _api_key() -> str | None:
    return os.getenv("AZURE_OPEN_AI_KEY")


# ── Async clients ─────────────────────────────────────────────────────────────

def get_fast_client() -> AsyncAzureOpenAI | None:
    """Return async client for gpt-4o-mini (fast extraction tasks)."""
    global _fast_async
    if _fast_async is not None:
        return _fast_async
    key = _api_key()
    if not key:
        return None
    _fast_async = AsyncAzureOpenAI(
        api_key=key,
        azure_endpoint=_AZURE_ENDPOINT,
        api_version=_AZURE_API_VERSION,
    )
    return _fast_async


def get_reasoning_client() -> AsyncAzureOpenAI | None:
    """Return async client for o4-mini (reasoning-heavy tasks)."""
    global _reasoning_async
    if _reasoning_async is not None:
        return _reasoning_async
    key = _api_key()
    if not key:
        return None
    _reasoning_async = AsyncAzureOpenAI(
        api_key=key,
        azure_endpoint=_AZURE_ENDPOINT,
        api_version=_AZURE_API_VERSION,
    )
    return _reasoning_async


# ── Sync client (for blocking contexts like _match_skill_llm) ────────────────

def get_fast_sync_client() -> AzureOpenAI | None:
    """Return sync client for gpt-4o-mini."""
    global _fast_sync
    if _fast_sync is not None:
        return _fast_sync
    key = _api_key()
    if not key:
        return None
    _fast_sync = AzureOpenAI(
        api_key=key,
        azure_endpoint=_AZURE_ENDPOINT,
        api_version=_AZURE_API_VERSION,
    )
    return _fast_sync
