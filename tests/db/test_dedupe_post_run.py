"""Tests for the post-run dedup pass.

The pipeline writes the same canonical concept under different slugs from
different roles (e.g. ``aws_parameter_store`` vs ``aws_systems_manager_parameter_store``
vs ``aws_ssm_parameter_store``). Same problem for dimensions, sub-categories,
and (more rarely) categories. ``db.dedupe_post_run`` finds those groups by
identical display_name (lowercased) and merges them, repointing every join
table at the canonical row before deleting the duplicates.

These tests use TRUNCATE-CASCADE (matching tests/db/test_loader.py) so the
schema (FK constraints, generated columns, vector indexes) stays intact.
"""
from __future__ import annotations

import os

import pytest
from sqlalchemy import insert, select, text

from db.models import (
    CanonicalSkill,
    Category,
    Dimension,
    DimensionSkill,
    Role,
    RoleDimension,
    SkillAlias,
    SubCategory,
)
from db.session import get_engine, get_session


_TRUNCATE_SQL = text("""
    TRUNCATE TABLE
        skill_relationships, skill_tags, dimension_categories,
        dimension_skills, role_dimensions, skill_aliases,
        canonical_skills, sub_categories, categories, dimensions, roles
    RESTART IDENTITY CASCADE
""")


# Safety: these tests TRUNCATE the entire pipeline schema. Refuse to run
# unless the user has explicitly opted in via TEST_DATABASE_URL — otherwise
# we would silently wipe a real run's data when CI or a developer runs
# pytest on a machine with DATABASE_URL pointed at a populated DB.
_TEST_DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not _TEST_DB_URL,
    reason=(
        "TEST_DATABASE_URL not set; skipping DB tests. To run, point "
        "TEST_DATABASE_URL at a SEPARATE Postgres database from the one "
        "in DATABASE_URL — these tests TRUNCATE all pipeline tables."
    ),
)


@pytest.fixture(scope="function")
async def fresh_db():
    # Override DATABASE_URL with TEST_DATABASE_URL for the duration of the
    # test so that get_engine() / get_session() create a separate engine
    # bound to the test DB, leaving any production DB untouched.
    original = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = _TEST_DB_URL  # type: ignore[assignment]
    # Clear any cached engine from a previous run that might still point at
    # the production DB.
    import db.session
    db.session._engine = None  # type: ignore[attr-defined]
    db.session._session_factory = None  # type: ignore[attr-defined]
    eng = get_engine()
    async with eng.begin() as conn:
        await conn.execute(_TRUNCATE_SQL)
    yield
    # Restore for the rest of the test session.
    if original is not None:
        os.environ["DATABASE_URL"] = original
    db.session._engine = None  # type: ignore[attr-defined]
    db.session._session_factory = None  # type: ignore[attr-defined]


# ── helpers to seed records ──────────────────────────────────────────────


async def _ensure_default_category(session) -> int:
    """Ensure a 'default' category exists; return its id (cached after first call)."""
    res = await session.execute(
        text("SELECT id FROM categories WHERE slug = 'default_test_cat' LIMIT 1")
    )
    row = res.first()
    if row:
        return row.id
    res = await session.execute(
        insert(Category).values(slug="default_test_cat", display_name="Default Test Cat")
        .returning(Category.id)
    )
    return res.scalar_one()


async def _seed_skill(session, slug: str, display_name: str, **kwargs) -> int:
    """Insert a minimal canonical_skill, return its id."""
    if "category_id" not in kwargs:
        kwargs["category_id"] = await _ensure_default_category(session)
    res = await session.execute(
        insert(CanonicalSkill)
        .values(
            slug=slug,
            display_name=display_name,
            skill_nature=kwargs.pop("skill_nature", "TOOL"),
            **kwargs,
        )
        .returning(CanonicalSkill.id)
    )
    return res.scalar_one()


async def _seed_dimension(session, slug: str, display_name: str) -> int:
    res = await session.execute(
        insert(Dimension)
        .values(slug=slug, display_name=display_name, source="AUTOMATED_DISCOVERY")
        .returning(Dimension.id)
    )
    return res.scalar_one()


async def _seed_role(session, slug: str, display_name: str) -> int:
    res = await session.execute(
        insert(Role)
        .values(slug=slug, display_name=display_name, source="AUTOMATED_DISCOVERY")
        .returning(Role.id)
    )
    return res.scalar_one()


async def _seed_category(session, slug: str, display_name: str) -> int:
    res = await session.execute(
        insert(Category)
        .values(slug=slug, display_name=display_name)
        .returning(Category.id)
    )
    return res.scalar_one()


# ── Test 1: skill duplicate-group discovery ──────────────────────────────


async def test_find_skill_duplicate_groups_by_display_name(fresh_db):
    """Three skills with identical display_name should form one group of size 3."""
    from db.dedupe_post_run import find_skill_duplicate_groups

    async with get_session() as s:
        async with s.begin():
            id_a = await _seed_skill(s, "aws_parameter_store", "AWS Systems Manager Parameter Store")
            id_b = await _seed_skill(s, "aws_systems_manager_parameter_store", "AWS Systems Manager Parameter Store")
            id_c = await _seed_skill(s, "aws_ssm_parameter_store", "AWS Systems Manager Parameter Store")
            id_other = await _seed_skill(s, "aws_secrets_manager", "AWS Secrets Manager")

        async with s.begin():
            groups = await find_skill_duplicate_groups(s)

    assert len(groups) == 1, f"expected 1 dup group, got {len(groups)}: {groups}"
    g = groups[0]
    assert g["display_name_lower"] == "aws systems manager parameter store"
    assert sorted(g["ids"]) == sorted([id_a, id_b, id_c])
    assert g["canonical_id"] == min(id_a, id_b, id_c), (
        "canonical should be the lowest id (oldest)"
    )
    assert id_other not in g["ids"]


async def test_find_skill_duplicate_groups_returns_empty_when_no_dups(fresh_db):
    from db.dedupe_post_run import find_skill_duplicate_groups

    async with get_session() as s:
        async with s.begin():
            await _seed_skill(s, "react", "React")
            await _seed_skill(s, "vue_js", "Vue.js")

        async with s.begin():
            groups = await find_skill_duplicate_groups(s)

    assert groups == []


# ── Test 2: skill merge with full join repointing ────────────────────────


async def test_merge_skills_repoints_dimension_skills_and_aliases(fresh_db):
    """After merging, all joins from dup ids point to canonical, and dups are deleted.

    Tricky part: when a (canonical, alias_text) row already exists AND a dup also
    has that same alias_text, we must drop the dup's alias row before repointing
    or the UNIQUE constraint fires.
    """
    from db.dedupe_post_run import merge_skills

    async with get_session() as s:
        async with s.begin():
            dim_id = await _seed_dimension(s, "test_dim", "Test Dim")
            canonical_id = await _seed_skill(s, "aws_param_store", "AWS Parameter Store")
            dup_a = await _seed_skill(s, "aws_systems_manager_parameter_store", "AWS Parameter Store")
            dup_b = await _seed_skill(s, "aws_ssm_parameter_store", "AWS Parameter Store")

            # Both canonical and dup_a have alias 'SSM'; merging should drop the dup's
            await s.execute(insert(SkillAlias).values(
                skill_id=canonical_id, alias_text="SSM", alias_type="ACRONYM",
            ))
            await s.execute(insert(SkillAlias).values(
                skill_id=dup_a, alias_text="SSM", alias_type="ACRONYM",
            ))
            # dup_b has unique alias 'Parameter Store' → should successfully repoint
            await s.execute(insert(SkillAlias).values(
                skill_id=dup_b, alias_text="Parameter Store", alias_type="ABBREVIATION",
            ))

            # All 3 skills are wired to dim_id via dimension_skills; after merge,
            # the unique (dim_id, skill_id) means dup rows must be dropped before
            # repointing or unique constraint fires.
            await s.execute(insert(DimensionSkill).values(dimension_id=dim_id, skill_id=canonical_id))
            await s.execute(insert(DimensionSkill).values(dimension_id=dim_id, skill_id=dup_a))
            await s.execute(insert(DimensionSkill).values(dimension_id=dim_id, skill_id=dup_b))

        async with s.begin():
            merged_count = await merge_skills(s, canonical_id, [dup_a, dup_b])

    # Verify in a fresh transaction
    async with get_session() as s:
        # Dups deleted
        ids = (await s.execute(
            select(CanonicalSkill.id).where(CanonicalSkill.id.in_([dup_a, dup_b]))
        )).all()
        assert ids == [], f"dup skills should be deleted, found: {ids}"

        # Canonical still present
        cs = (await s.execute(
            select(CanonicalSkill.id).where(CanonicalSkill.id == canonical_id)
        )).first()
        assert cs is not None

        # Aliases: canonical kept SSM (not duped), gained Parameter Store from dup_b
        aliases = (await s.execute(
            select(SkillAlias.alias_text).where(SkillAlias.skill_id == canonical_id)
        )).scalars().all()
        assert sorted(aliases) == sorted(["SSM", "Parameter Store"]), aliases

        # dimension_skills: 1 row, pointing at canonical
        ds = (await s.execute(
            select(DimensionSkill.skill_id).where(DimensionSkill.dimension_id == dim_id)
        )).scalars().all()
        assert ds == [canonical_id], f"dimension_skills should have only canonical: {ds}"

    assert merged_count == 2


# ── Test 3: dimension dedup with role-scoped prefix protection ───────────


async def test_find_dimension_duplicate_groups_skips_role_scoped_prefixes(fresh_db):
    """`ebs_*` and `sap_*` and `oracle_*` are intentionally role-scoped — never merge."""
    from db.dedupe_post_run import find_dimension_duplicate_groups

    async with get_session() as s:
        async with s.begin():
            id_generic_a = await _seed_dimension(s, "auth_iam", "Auth & Identity Standards")
            id_generic_b = await _seed_dimension(s, "auth_identity_standards", "Auth & Identity Standards")
            id_generic_c = await _seed_dimension(s, "auth_iam_standards", "Auth & IAM Standards")
            # Role-scoped — same display_name pattern but should NOT merge with above
            id_ebs = await _seed_dimension(s, "ebs_authorization_security", "Authorizations & Security Model")
            id_sap = await _seed_dimension(s, "sap_authorization_security", "SAP Authorization & Security")

        async with s.begin():
            groups = await find_dimension_duplicate_groups(s)

    # Only the two generic auth_iam dims share an exact display_name; the third
    # has slightly different wording ("IAM Standards") and should not be in the
    # exact-match group.
    assert len(groups) == 1
    g = groups[0]
    assert sorted(g["ids"]) == sorted([id_generic_a, id_generic_b])
    assert id_ebs not in g["ids"]
    assert id_sap not in g["ids"]


# ── Test 4: dry-run mode ─────────────────────────────────────────────────


async def test_dedupe_dry_run_does_not_commit(fresh_db):
    """dry_run=True returns the merge plan without actually deleting/repointing."""
    from db.dedupe_post_run import run_dedup_pass

    async with get_session() as s:
        async with s.begin():
            id_a = await _seed_skill(s, "kafka", "Apache Kafka")
            id_b = await _seed_skill(s, "apache_kafka", "Apache Kafka")

    plan = await run_dedup_pass(dry_run=True)

    # Plan should mention the merge
    skill_merges = plan.get("skills", [])
    assert len(skill_merges) == 1
    assert sorted(skill_merges[0]["ids"]) == sorted([id_a, id_b])

    # But the dup should still be in the DB (no commit)
    async with get_session() as s:
        rows = (await s.execute(
            select(CanonicalSkill.id).where(CanonicalSkill.id.in_([id_a, id_b]))
        )).scalars().all()
        assert len(rows) == 2, "dry_run should not delete anything"
