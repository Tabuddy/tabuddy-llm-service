"""Test the taxonomy-normalization helper against a live DB.

Verifies the user-flagged scenario: "Caching" exists in the DB; the next
run's LLM emits "caching_pattern"/"Caching Pattern". The normalizer should
return the existing id via trigram match (or embedding when available).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import insert, text

from db.models import Category
from db.normalization import find_match
from db.session import get_engine, get_session


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
async def fresh_schema():
    eng = get_engine()
    async with eng.begin() as conn:
        await conn.execute(_TRUNCATE_SQL)
    yield
    async with eng.begin() as conn:
        await conn.execute(_TRUNCATE_SQL)


async def _seed_caching(session) -> int:
    res = await session.execute(
        insert(Category)
        .values(slug="caching", display_name="Caching", description=None)
        .returning(Category.id)
    )
    cid = res.scalar_one()
    await session.flush()
    return cid


async def test_exact_slug_hits(fresh_schema):
    async with get_session() as s:
        async with s.begin():
            cid = await _seed_caching(s)
        async with s.begin():
            m = await find_match(
                s, "categories", "caching", "Caching", incoming_embedding=None,
            )
            assert m.existing_id == cid
            assert m.kind == "exact_slug"


async def _seed_data_tool(session) -> int:
    res = await session.execute(
        insert(Category)
        .values(slug="data_tool", display_name="Data Tool", description=None)
        .returning(Category.id)
    )
    cid = res.scalar_one()
    await session.flush()
    return cid


async def test_pluralization_drift_merges_via_trigram(fresh_schema):
    """Stem drift ('data_tool' ≈ 'data_tools', similarity 0.75) is the
    trigram path's job — well above the 0.7 threshold."""
    async with get_session() as s:
        async with s.begin():
            await s.execute(text("DELETE FROM categories"))
            cid = await _seed_data_tool(s)
        async with s.begin():
            m = await find_match(
                s, "categories",
                incoming_slug="data_tools",
                incoming_display="Data Tools",
                incoming_embedding=None,
            )
            assert m.existing_id == cid, (
                f"expected stem drift 'data_tools' to merge into 'data_tool' (id={cid}), got {m}"
            )
            assert m.kind in {"trigram_slug", "trigram_display"}


async def test_caching_pattern_merges_via_embedding(fresh_schema):
    """Semantic drift is the embedding cascade's job — 'Caching' ≡ 'Caching Pattern'.

    Seeds the existing category with a hand-rolled embedding, then queries
    with a near-identical vector. Trigram for these two strings is only ~0.5
    (below the 0.7 typo-only threshold), so the merge MUST come from the
    embedding cascade.
    """
    # 1536-dim vector; identical = cosine 1.0, slightly perturbed = ~0.99.
    seed_vec = [0.5] * 1536
    near_vec = [0.5 + (0.001 if i % 5 == 0 else 0.0) for i in range(1536)]
    seed_lit = "[" + ",".join(repr(float(x)) for x in seed_vec) + "]"

    async with get_session() as s:
        async with s.begin():
            await s.execute(text("DELETE FROM categories"))
            res = await s.execute(text(
                "INSERT INTO categories (slug, display_name, name_embedding) "
                "VALUES ('caching', 'Caching', CAST(:emb AS vector)) RETURNING id"
            ), {"emb": seed_lit})
            cid = res.scalar_one()
        async with s.begin():
            m = await find_match(
                s, "categories",
                incoming_slug="caching_pattern",
                incoming_display="Caching Pattern",
                incoming_embedding=near_vec,
            )
            assert m.existing_id == cid, (
                f"expected 'Caching Pattern' to merge into 'caching' (id={cid}) "
                f"via embedding cosine, got {m}"
            )
            assert m.kind == "embedding"
            assert m.score is not None and m.score >= 0.85


async def test_no_match_below_threshold(fresh_schema):
    async with get_session() as s:
        async with s.begin():
            await s.execute(text("DELETE FROM categories"))
            await _seed_caching(s)
        async with s.begin():
            m = await find_match(
                s, "categories",
                incoming_slug="unrelated_thing_xyz",
                incoming_display="Totally Unrelated Concept",
                incoming_embedding=None,
            )
            assert m.existing_id is None
            assert m.kind == "no_match"


async def test_invalid_table_rejected(fresh_schema):
    async with get_session() as s:
        async with s.begin():
            with pytest.raises(ValueError):
                await find_match(s, "evil_table", "x", "X", None)


# ── Substring-direction guard (framework vs css_framework) ─────────────────


async def _seed_css_framework(session) -> int:
    res = await session.execute(
        insert(Category)
        .values(slug="css_framework", display_name="CSS Framework", description=None)
        .returning(Category.id)
    )
    cid = res.scalar_one()
    await session.flush()
    return cid


async def _seed_framework(session) -> int:
    res = await session.execute(
        insert(Category)
        .values(slug="framework", display_name="Framework", description=None)
        .returning(Category.id)
    )
    cid = res.scalar_one()
    await session.flush()
    return cid


async def test_framework_does_not_merge_into_css_framework(fresh_schema):
    """Shorter, general slug must not be absorbed by longer specialized one."""
    async with get_session() as s:
        async with s.begin():
            await s.execute(text("DELETE FROM categories"))
            await _seed_css_framework(s)
        async with s.begin():
            m = await find_match(
                s, "categories",
                incoming_slug="framework",
                incoming_display="Framework",
                incoming_embedding=None,
            )
            assert m.kind == "no_match", (
                f"expected substring guard to block merge, got {m}"
            )
            assert m.existing_id is None


async def test_css_framework_does_not_merge_into_framework(fresh_schema):
    """Longer specialized slug must not be absorbed by shorter general one."""
    async with get_session() as s:
        async with s.begin():
            await s.execute(text("DELETE FROM categories"))
            await _seed_framework(s)
        async with s.begin():
            m = await find_match(
                s, "categories",
                incoming_slug="css_framework",
                incoming_display="CSS Framework",
                incoming_embedding=None,
            )
            assert m.kind == "no_match", (
                f"expected substring guard to block merge, got {m}"
            )
            assert m.existing_id is None
