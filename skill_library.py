"""Unified skill library: merges static SKILL_ALIASES with MongoDB learned skills.

Usage:
  await skill_library.init()           # call once at startup
  aliases = skill_library.get_aliases()  # returns merged dict every call
  await skill_library.learn(alias, canonical)  # persist new skill

Design:
  - Static dict (skills_dictionary.SKILL_ALIASES) is the base layer.
  - MongoDB learned_skills overlays on top (MongoDB wins on conflict).
  - In-memory cache is refreshed on startup and on every learn() call.
  - If MongoDB is unavailable, only static dict is used — no errors.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# ── In-memory merged cache ────────────────────────────────────────────────────
_merged: dict[str, str] = {}
_initialized: bool = False


async def init() -> None:
    """Load static aliases and overlay MongoDB learned skills.

    Safe to call multiple times — subsequent calls refresh the cache.
    """
    global _merged, _initialized

    from skills_dictionary import SKILL_ALIASES
    import db

    base = dict(SKILL_ALIASES)  # copy to avoid mutating the original

    learned = await db.get_all_learned_skills()
    if learned:
        base.update(learned)
        logger.info(
            "Skill library: %d static + %d learned = %d total aliases",
            len(SKILL_ALIASES), len(learned), len(base),
        )
    else:
        logger.info(
            "Skill library: %d static aliases (no MongoDB entries)", len(base))

    _merged = base
    _initialized = True


def get_aliases() -> dict[str, str]:
    """Return the merged skill alias dict.  Falls back to static if not initialized."""
    if _merged:
        return _merged
    from skills_dictionary import SKILL_ALIASES
    return SKILL_ALIASES


async def learn(alias: str, canonical: str, confidence: float = 0.85) -> None:
    """Persist a new alias → canonical mapping and update the in-memory cache.

    Called automatically by normalizer.py after every successful LLM
    normalization of an unknown skill.

    Silently no-ops if MongoDB is unavailable or inputs are invalid.
    """
    if not alias or not canonical:
        return

    key = alias.strip().lower()
    if not key:
        return

    # Update in-memory cache immediately (no wait for DB round-trip)
    _merged[key] = canonical

    import db
    await db.upsert_learned_skill(key, canonical, source="llm", confidence=confidence)
