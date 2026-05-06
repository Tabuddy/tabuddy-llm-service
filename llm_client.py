"""Centralized Azure OpenAI client factory.

Provides three client tiers:
  - **fast** (gpt-5.4-mini): block tagging, skill normalization, categorization
  - **reasoning** (o4-mini): JD atom extraction, role-fit scoring, section scoring
  - **generation** (gpt-5.4): skill Generator — needs current-world knowledge
"""

from __future__ import annotations

import os
import logging
from openai import AsyncAzureOpenAI, AzureOpenAI

logger = logging.getLogger(__name__)

_AZURE_ENDPOINT = "https://tabuddy-azure-sponsor.openai.azure.com/"
_AZURE_API_VERSION = "2024-12-01-preview"

FAST_MODEL = os.getenv("FAST_DEPLOYMENT", "gpt-5.4-mini")
REASONING_MODEL = os.getenv("REASONING_DEPLOYMENT", "o4-mini")
GENERATION_MODEL = os.getenv("GENERATION_DEPLOYMENT", "gpt-5-mini")
EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# Models that use reasoning-style API: reject `temperature` (only default 1.0)
# and `max_tokens` (need `max_completion_tokens` instead). Add to this set
# whenever a new o-series or gpt-5 reasoning variant is introduced.
RESTRICTED_PARAM_MODELS: frozenset[str] = frozenset({
    "o1", "o1-mini", "o1-preview", "o3-mini", "o4-mini",
    "gpt-5-mini",
})

# ── Singletons ────────────────────────────────────────────────────────────────
_fast_async: AsyncAzureOpenAI | None = None
_reasoning_async: AsyncAzureOpenAI | None = None
_generation_async: AsyncAzureOpenAI | None = None
_embedding_async: AsyncAzureOpenAI | None = None
_fast_sync: AzureOpenAI | None = None


def _api_key() -> str | None:
    return os.getenv("AZURE_OPEN_AI_KEY")


# ── Async clients ─────────────────────────────────────────────────────────────

def get_fast_client() -> AsyncAzureOpenAI | None:
    """Return async client for gpt-5.4-mini (fast extraction tasks)."""
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


def get_generation_client() -> AsyncAzureOpenAI | None:
    """Return async client for gpt-5.4 (skill Generator with current knowledge)."""
    global _generation_async
    if _generation_async is not None:
        return _generation_async
    key = _api_key()
    if not key:
        return None
    _generation_async = AsyncAzureOpenAI(
        api_key=key,
        azure_endpoint=_AZURE_ENDPOINT,
        api_version=_AZURE_API_VERSION,
    )
    return _generation_async


def get_embedding_client() -> AsyncAzureOpenAI | None:
    """Return async client for text-embedding-3-small (1536-dim vectors)."""
    global _embedding_async
    if _embedding_async is not None:
        return _embedding_async
    key = _api_key()
    if not key:
        return None
    _embedding_async = AsyncAzureOpenAI(
        api_key=key,
        azure_endpoint=_AZURE_ENDPOINT,
        api_version=_AZURE_API_VERSION,
    )
    return _embedding_async


# ── Sync client (for blocking contexts like _match_skill_llm) ────────────────

def get_fast_sync_client() -> AzureOpenAI | None:
    """Return sync client for gpt-5.4-mini."""
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
