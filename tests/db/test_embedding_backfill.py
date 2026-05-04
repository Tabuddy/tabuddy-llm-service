"""Test the embedding backfill against a live DB with mocked Azure embedder."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select, text

from db.loader import load_payload
from db.models import CanonicalSkill, Category, SkillAlias, SubCategory
from db.session import get_engine, get_session


_TRUNCATE_SQL = text("""
    TRUNCATE TABLE
        skill_relationships, skill_tags, dimension_categories,
        dimension_skills, role_dimensions, skill_aliases,
        canonical_skills, sub_categories, categories, dimensions, roles
    RESTART IDENTITY CASCADE
""")


_SAMPLE = (
    Path(__file__).resolve().parents[2]
    / "evals" / "samples" / "skill_library_two_roles_sample.json"
)


pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set; skipping DB tests.",
)


def _fake_vector(seed_text: str) -> list[float]:
    """Deterministic 1536-dim pseudo-vector keyed off the input text length.

    Good enough for "are the columns populated?" tests; not for similarity.
    """
    base = (sum(ord(c) for c in seed_text) % 100) / 100.0
    return [base] * 1536


class _FakeEmbedder:
    async def embed(self, texts):
        return [_fake_vector(t) for t in texts]


@pytest.fixture(scope="module")
async def loaded_db_no_embeddings():
    """Load the sample with embeddings disabled at load time so we can verify
    backfill populates them."""
    # AZURE_OPEN_AI_KEY may or may not be set; force loader to skip embedding.
    saved = os.environ.pop("AZURE_OPEN_AI_KEY", None)
    try:
        eng = get_engine()
        async with eng.begin() as conn:
            await conn.execute(_TRUNCATE_SQL)
        payload = json.loads(_SAMPLE.read_text(encoding="utf-8"))
        await load_payload(payload)
        yield
        async with eng.begin() as conn:
            await conn.execute(_TRUNCATE_SQL)
    finally:
        if saved is not None:
            os.environ["AZURE_OPEN_AI_KEY"] = saved


async def test_backfill_populates_aliases_and_skills(loaded_db_no_embeddings):
    from db import embedding_backfill

    fake = _FakeEmbedder()
    with patch.object(
        embedding_backfill, "get_embedder", return_value=fake,
    ):
        counts = await embedding_backfill.main_async(target="all", limit=None)

    assert counts["names"] > 0
    assert counts["aliases"] > 0

    async with get_session() as s:
        nulls = (await s.execute(
            select(CanonicalSkill).where(CanonicalSkill.name_embedding.is_(None))
        )).scalars().all()
        assert len(nulls) == 0
        nulls_a = (await s.execute(
            select(SkillAlias).where(SkillAlias.alias_embedding.is_(None))
        )).scalars().all()
        assert len(nulls_a) == 0


async def test_backfill_idempotent(loaded_db_no_embeddings):
    from db import embedding_backfill

    fake = _FakeEmbedder()
    with patch.object(
        embedding_backfill, "get_embedder", return_value=fake,
    ):
        await embedding_backfill.main_async(target="all", limit=None)
        # Second call should embed nothing — all rows already non-null.
        counts2 = await embedding_backfill.main_async(target="all", limit=None)

    for k in ("names", "aliases", "categories", "sub_categories"):
        assert counts2[k] == 0, f"{k!r} re-embedded {counts2[k]} rows on second call"
