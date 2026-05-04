"""Bidirectional query helpers test."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from sqlalchemy import text

from db.loader import load_payload
from db.queries import roles_for_skills, skills_for_role
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
async def seeded_db():
    eng = get_engine()
    async with eng.begin() as conn:
        await conn.execute(_TRUNCATE_SQL)
    payload = json.loads(_SAMPLE.read_text(encoding="utf-8"))
    await load_payload(payload)
    yield
    async with eng.begin() as conn:
        await conn.execute(_TRUNCATE_SQL)


async def test_skills_for_frontend_developer(seeded_db):
    async with get_session() as s:
        skills = await skills_for_role(s, "frontend_developer")
    slugs = {sk.slug for sk in skills}
    assert "react" in slugs
    assert "javascript" in slugs
    assert "typescript" in slugs
    # And NOT a data-engineer-only skill
    assert "apache_airflow" not in slugs


async def test_skills_for_data_engineer(seeded_db):
    async with get_session() as s:
        skills = await skills_for_role(s, "data_engineer")
    slugs = {sk.slug for sk in skills}
    assert "apache_airflow" in slugs
    assert "snowflake" in slugs
    assert "react" not in slugs


async def test_roles_for_skills_frontend(seeded_db):
    async with get_session() as s:
        matches = await roles_for_skills(s, ["react", "typescript", "next_js"])
    assert matches, "expected at least one match"
    assert matches[0].role_slug == "frontend_developer"
    assert matches[0].matched_skill_count >= 2


async def test_roles_for_skills_data(seeded_db):
    async with get_session() as s:
        matches = await roles_for_skills(
            s, ["snowflake", "apache_airflow", "apache_kafka"],
        )
    assert matches
    assert matches[0].role_slug == "data_engineer"
    assert matches[0].matched_skill_count >= 2


async def test_roles_for_skills_unknown_slug_no_crash(seeded_db):
    async with get_session() as s:
        matches = await roles_for_skills(s, ["totally_made_up_skill_xyz"])
    assert matches == []
