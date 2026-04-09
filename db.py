"""MongoDB async client + helpers for the auto-growing skill library.

Collection: learned_skills
Schema:
  { alias: str (pk), canonical: str, source: "llm"|"manual",
    confidence: float, first_seen: datetime, hit_count: int }

Graceful degradation: if MONGODB_URI is not set or the connection fails,
all functions are no-ops and the service operates with the static skill dict only.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Module-level client singleton ─────────────────────────────────────────────
_client = None
_db = None
_collection = None
_available: bool = False


def _get_mongo_uri() -> str | None:
    return os.getenv("MONGODB_URI")


async def init_db() -> bool:
    """Initialize MongoDB connection.  Call once at startup.

    Returns True if connection succeeded, False if unavailable.
    Failure is non-fatal — the service degrades to static-only mode.
    """
    global _client, _db, _collection, _available

    uri = _get_mongo_uri()
    if not uri:
        logger.info(
            "MONGODB_URI not set — skill library runs in static-only mode")
        return False

    try:
        from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore

        _client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=3000)
        # Ping to verify connection before we commit to using it
        await _client.admin.command("ping")

        db_name = os.getenv("MONGODB_DB", "tabuddy")
        _db = _client[db_name]
        _collection = _db["learned_skills"]

        # Ensure unique index on alias
        await _collection.create_index("alias", unique=True)

        _available = True
        logger.info("✅ MongoDB connected — skill library persistence enabled")
        return True

    except Exception as e:
        logger.warning(
            "MongoDB unavailable (%s) — skill library runs in static-only mode", e
        )
        _client = None
        _db = None
        _collection = None
        _available = False
        return False


def is_available() -> bool:
    """Return True if MongoDB connection is live."""
    return _available


async def get_all_learned_skills() -> dict[str, str]:
    """Fetch all learned skills as {alias: canonical} dict.

    Returns empty dict if MongoDB unavailable.
    """
    if not _available or _collection is None:
        return {}
    try:
        docs = await _collection.find(
            {}, {"alias": 1, "canonical": 1, "_id": 0}
        ).to_list(length=50_000)
        return {d["alias"]: d["canonical"] for d in docs if "alias" in d and "canonical" in d}
    except Exception as e:
        logger.warning("Failed to fetch learned skills from MongoDB: %s", e)
        return {}


async def upsert_learned_skill(
    alias: str,
    canonical: str,
    source: str = "llm",
    confidence: float = 0.85,
) -> bool:
    """Insert or update a learned skill mapping.

    On duplicate alias, increments hit_count and updates canonical if
    confidence is higher.  Returns True on success, False on error.
    """
    if not _available or _collection is None:
        return False

    alias = alias.strip().lower()
    if not alias or not canonical.strip():
        return False

    try:
        now = datetime.now(timezone.utc)
        await _collection.update_one(
            {"alias": alias},
            {
                "$setOnInsert": {
                    "alias": alias,
                    "first_seen": now,
                },
                "$set": {
                    "canonical": canonical,
                    "source": source,
                    "confidence": confidence,
                    "last_seen": now,
                },
                "$inc": {"hit_count": 1},
            },
            upsert=True,
        )
        return True
    except Exception as e:
        logger.debug("upsert_learned_skill failed for %r: %s", alias, e)
        return False


async def get_skill_hit_counts(top_n: int = 100) -> list[dict]:
    """Return the top-N most frequently encountered learned skills.

    Useful for monitoring which unknown skills are being auto-learned.
    """
    if not _available or _collection is None:
        return []
    try:
        docs = await _collection.find(
            {},
            {"alias": 1, "canonical": 1, "hit_count": 1, "_id": 0},
        ).sort("hit_count", -1).limit(top_n).to_list(length=top_n)
        return docs
    except Exception as e:
        logger.warning("get_skill_hit_counts failed: %s", e)
        return []
