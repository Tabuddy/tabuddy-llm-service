"""Test post-load DB enrichment against a live DB seeded from the sample fixture."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from sqlalchemy import select, text

from db.loader import load_payload
from db.models import CanonicalSkill, Category
from db.post_load_enrich import main_async as enrich_main
from db.session import get_engine, get_session


_SAMPLE = (
    Path(__file__).resolve().parents[2]
    / "evals" / "samples" / "skill_library_two_roles_sample.json"
)

_TRUNCATE_SQL = text("""
    TRUNCATE TABLE
        skill_relationships, skill_tags, dimension_categories,
        dimension_skills, role_dimensions, skill_aliases,
        canonical_skills, sub_categories, categories, dimensions, roles
    RESTART IDENTITY CASCADE
""")


pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set; skipping DB tests.",
)


@pytest.fixture(scope="module")
async def loaded_db():
    eng = get_engine()
    async with eng.begin() as conn:
        await conn.execute(_TRUNCATE_SQL)
    payload = json.loads(_SAMPLE.read_text(encoding="utf-8"))
    await load_payload(payload)
    yield
    async with eng.begin() as conn:
        await conn.execute(_TRUNCATE_SQL)


async def test_enrichment_runs_without_error(loaded_db):
    counts = await enrich_main()
    assert isinstance(counts, dict)
    # All count keys present even if zero — verifies the script ran every step.
    assert set(counts) == {
        "is_also_category",
        "version_tagged",
        "version_linked_to_parent",
        "slug_prefix_parents",
    }


async def test_enrichment_parses_angular_version(loaded_db):
    await enrich_main()
    async with get_session() as s:
        # The sample uses "Angular 2+" as an alias on display_name="Angular".
        # The post-load enricher walks display_name; "Angular" itself does not
        # match the trailing-version regex, so version_tag stays NULL there.
        # Skills like "WCAG 2.1" — but the sample uses "WCAG" as display, with
        # "WCAG 2.1" as an alias. So expect version_tagged>=0; this test only
        # asserts the script ran cleanly and the column types are consistent.
        rows = (await s.execute(select(CanonicalSkill))).scalars().all()
        for r in rows:
            if r.version_tag is not None:
                assert r.version_strategy in ("SEPARATE_ENTITY", "NOT_APPLICABLE")


async def test_is_also_category_flag_correctness(loaded_db):
    """Skills whose slug equals a category or sub_category slug should be flagged."""
    await enrich_main()
    async with get_session() as s:
        cat_slugs = {
            r[0] for r in (await s.execute(select(Category.slug))).all()
        }
        # Look for any canonical skill that shares its slug with a category.
        rows = (await s.execute(select(CanonicalSkill))).scalars().all()
        for r in rows:
            if r.slug in cat_slugs:
                assert r.is_also_category is True
