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
    category: str = "Other",
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
                    "category": category,
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


async def search_learned_skills(
    query: str = "",
    category: str = "",
    source: str = "",
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[dict], int]:
    """Search / filter learned skills for the admin panel.

    Returns (docs, total_count).
    """
    if not _available or _collection is None:
        return [], 0

    try:
        filt: dict = {}
        if query:
            filt["$or"] = [
                {"alias": {"$regex": query, "$options": "i"}},
                {"canonical": {"$regex": query, "$options": "i"}},
            ]
        if category:
            filt["category"] = category
        if source:
            filt["source"] = source

        total = await _collection.count_documents(filt)
        docs = await (
            _collection.find(filt, {"_id": 0})
            .sort("last_seen", -1)
            .skip(skip)
            .limit(limit)
            .to_list(length=limit)
        )
        return docs, total
    except Exception as e:
        logger.warning("search_learned_skills failed: %s", e)
        return [], 0


async def delete_learned_skill(alias: str) -> bool:
    """Delete a learned skill by alias key."""
    if not _available or _collection is None:
        return False
    try:
        result = await _collection.delete_one({"alias": alias.strip().lower()})
        if result.deleted_count:
            return True
        return False
    except Exception as e:
        logger.warning("delete_learned_skill failed for %r: %s", alias, e)
        return False


async def get_skill_categories() -> list[str]:
    """Return distinct category values from learned_skills."""
    if not _available or _collection is None:
        return []
    try:
        cats = await _collection.distinct("category")
        return sorted([c for c in cats if c])
    except Exception as e:
        logger.warning("get_skill_categories failed: %s", e)
        return []


async def seed_static_skills() -> int:
    """Bulk-insert static SKILL_ALIASES into MongoDB (skip existing).

    Uses a fast count check to skip entirely on subsequent startups.
    Only runs the full diff + insert on first boot or when new static
    skills are added to the code.
    """
    if not _available or _collection is None:
        return 0

    from skills_dictionary import SKILL_ALIASES, SKILL_CATEGORIES

    # Fast path: if static count matches, nothing to do
    try:
        static_count = await _collection.count_documents({"source": "static"})
        if static_count >= len(SKILL_ALIASES):
            return 0
    except Exception as e:
        logger.warning("seed_static_skills: count check failed: %s", e)
        return 0

    now = datetime.now(timezone.utc)

    # Only fetch existing aliases when we actually need to diff
    existing = set()
    try:
        cursor = _collection.find({}, {"alias": 1, "_id": 0})
        async for doc in cursor:
            existing.add(doc["alias"])
    except Exception as e:
        logger.warning("seed_static_skills: failed to read existing: %s", e)
        return 0

    # Build insert batch for aliases not yet in DB
    to_insert = []
    for alias_key, canonical in SKILL_ALIASES.items():
        key = alias_key.strip().lower()
        if key in existing:
            continue
        category = SKILL_CATEGORIES.get(canonical, "Other")
        to_insert.append({
            "alias": key,
            "canonical": canonical,
            "source": "static",
            "category": category,
            "confidence": 1.0,
            "first_seen": now,
            "last_seen": now,
            "hit_count": 0,
        })

    if not to_insert:
        logger.info("Seed: all %d static skills already present",
                    len(SKILL_ALIASES))
        return 0

    try:
        result = await _collection.insert_many(to_insert, ordered=False)
        inserted = len(result.inserted_ids)
        logger.info(
            "Seed: inserted %d new static skills into MongoDB", inserted)
        return inserted
    except Exception as e:
        logger.warning("seed_static_skills bulk insert error: %s", e)
        return 0
