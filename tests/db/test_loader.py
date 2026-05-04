"""Smoke test: load the two-role golden fixture into a live DB, assert counts.

Fixtures use TRUNCATE rather than drop+create so the schema (views, search
functions, custom indexes) laid down by ``alembic upgrade head`` stays
intact between tests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from sqlalchemy import select, text

from db.loader import load_payload
from db.models import CanonicalSkill, Category, Role, SubCategory
from db.session import get_engine, get_session

_SAMPLE = (
    Path(__file__).resolve().parents[2]
    / "evals" / "samples" / "skill_library_two_roles_sample.json"
)

# Tables to TRUNCATE between tests. CASCADE handles FK chains.
_TRUNCATE_SQL = text("""
    TRUNCATE TABLE
        skill_relationships,
        skill_tags,
        dimension_categories,
        dimension_skills,
        role_dimensions,
        skill_aliases,
        canonical_skills,
        sub_categories,
        categories,
        dimensions,
        roles
    RESTART IDENTITY CASCADE
""")


def _has_db() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


pytestmark = pytest.mark.skipif(
    not _has_db(),
    reason="DATABASE_URL not set; skipping DB tests.",
)


@pytest.fixture(scope="module")
async def setup_schema():
    """Wipe rows; assume schema already migrated."""
    eng = get_engine()
    async with eng.begin() as conn:
        await conn.execute(_TRUNCATE_SQL)
    yield
    async with eng.begin() as conn:
        await conn.execute(_TRUNCATE_SQL)


async def test_load_two_role_sample(setup_schema):
    payload = json.loads(_SAMPLE.read_text(encoding="utf-8"))
    counts = await load_payload(payload)
    assert counts["categories"] >= 3
    assert counts["sub_categories"] >= 6
    assert counts["canonical_skills"] >= 15
    assert counts["roles"] == 2
    assert counts["dimensions"] >= 4

    async with get_session() as s:
        roles = (await s.execute(select(Role))).scalars().all()
        assert {r.slug for r in roles} == {"frontend_developer", "data_engineer"}
        cats = (await s.execute(select(Category))).scalars().all()
        cat_slugs = {c.slug for c in cats}
        assert "framework" in cat_slugs
        assert "data_tool" in cat_slugs


async def test_load_idempotent(setup_schema):
    payload = json.loads(_SAMPLE.read_text(encoding="utf-8"))
    counts1 = await load_payload(payload)
    counts2 = await load_payload(payload)
    assert counts2["canonical_skills"] == counts1["canonical_skills"]

    async with get_session() as s:
        n_skills = (await s.execute(select(CanonicalSkill))).scalars().all()
        # No duplicate slugs after second load.
        slugs = [sk.slug for sk in n_skills]
        assert len(slugs) == len(set(slugs))
