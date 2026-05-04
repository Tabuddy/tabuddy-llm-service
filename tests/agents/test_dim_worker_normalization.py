"""Tests for the clamp-to-80 + alias-dedup normalization in dim_worker.

The LLM routinely emits 81-110 skills due to the coverage push, and
occasionally repeats an alias within one skill. dim_worker normalizes both
deterministically before validation runs, so the worker doesn't waste a
retry round-trip on issues that are trivial to fix.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from skill_library_v2.agents.dim_worker import process_dim
from skill_library_v2.schemas.category import (
    CategorizerOutput,
    DimensionCategoryProposal,
)
from skill_library_v2.schemas.skill import (
    CriticOutput,
    SkillDraft,
    SkillGeneratorOutput,
)


def _make_skill(i: int, aliases: list[str] | None = None) -> SkillDraft:
    return SkillDraft(
        display_name=f"Skill{i}",
        slug=f"skill_{i}",
        category_slug="framework",
        sub_category_slug="frontend_framework",
        skill_nature="TOOL",
        aliases=aliases or [],
    )


def _cat_out(dim_id: str = "d") -> CategorizerOutput:
    return CategorizerOutput(
        dim_id=dim_id,
        proposals=[DimensionCategoryProposal(
            category_slug="framework", category_display="Framework",
            sub_category_slug="frontend_framework",
            sub_category_display="Frontend Framework",
            rationale="UI libs", expected_skill_count_hint=5,
        )],
    )


def _crit_out(dim_id: str = "d") -> CriticOutput:
    return CriticOutput(
        dim_id=dim_id,
        corrections=[],
        approved_slugs=[],
        overall_verdict="approve",
    )


def _state(dim_id: str = "d") -> dict:
    return {
        "dim_id": dim_id,
        "dim": {
            "dimension_id": dim_id, "dimension_name": "D",
            "rationale": "r", "expected_skill_kinds": ["FRAMEWORK"],
            "difficulty_hint": "well_known",
        },
        "role_id": "r", "role_display": "R",
        "dimensions": [], "planner_web_hints": [],
    }


def test_schema_no_longer_caps_at_80():
    """Pre-fix the schema raised ValidationError on >80; now the cap lives in dim_worker."""
    skills = [_make_skill(i) for i in range(100)]
    out = SkillGeneratorOutput(dim_id="d", skills=skills)
    assert len(out.skills) == 100


async def test_dim_worker_clamps_overshoot_to_80():
    """LLM emits 105 skills; dim_worker truncates to 80 before validator runs."""
    skill_out = SkillGeneratorOutput(
        dim_id="d", skills=[_make_skill(i) for i in range(105)],
    )
    with patch("skill_library_v2.agents.dim_worker._get_categorizer") as gc, \
         patch("skill_library_v2.agents.dim_worker._get_skill_generator") as gs, \
         patch("skill_library_v2.agents.dim_worker._get_critic") as gcrit:
        gc.return_value = type("M", (), {"run": AsyncMock(return_value=_cat_out())})()
        gs.return_value = type("M", (), {"run": AsyncMock(return_value=skill_out)})()
        gcrit.return_value = type("M", (), {"run": AsyncMock(return_value=_crit_out())})()

        result = await process_dim(_state())

    assert result["dim_status"]["d"] == "done"
    assert len(result["generated"]["d"]) == 80
    assert result["generated"]["d"][0].slug == "skill_0"
    assert result["generated"]["d"][-1].slug == "skill_79"


async def test_dim_worker_dedupes_aliases_case_insensitive_first_wins():
    """Duplicate aliases (Protobuf/protobuf/PROTOBUF) are deduped before validate."""
    skill_with_dups = _make_skill(0, aliases=["Protobuf", "protobuf", "PROTOBUF", "gRPC IDL"])
    other_skills = [_make_skill(i) for i in range(1, 8)]
    skill_out = SkillGeneratorOutput(
        dim_id="d", skills=[skill_with_dups, *other_skills],
    )

    with patch("skill_library_v2.agents.dim_worker._get_categorizer") as gc, \
         patch("skill_library_v2.agents.dim_worker._get_skill_generator") as gs, \
         patch("skill_library_v2.agents.dim_worker._get_critic") as gcrit:
        gc.return_value = type("M", (), {"run": AsyncMock(return_value=_cat_out())})()
        gs.return_value = type("M", (), {"run": AsyncMock(return_value=skill_out)})()
        gcrit.return_value = type("M", (), {"run": AsyncMock(return_value=_crit_out())})()

        result = await process_dim(_state())

    assert result["dim_status"]["d"] == "done", (
        f"validator should pass after dedup; got {result.get('errors')}"
    )
    survivor = result["generated"]["d"][0]
    assert survivor.aliases == ["Protobuf", "gRPC IDL"]


async def test_dim_worker_strips_blank_aliases():
    """Blank/whitespace aliases get filtered out by the dedup pass."""
    skill = _make_skill(0, aliases=["", "  ", "real", "real"])
    skill_out = SkillGeneratorOutput(
        dim_id="d",
        skills=[skill, *[_make_skill(i) for i in range(1, 8)]],
    )
    with patch("skill_library_v2.agents.dim_worker._get_categorizer") as gc, \
         patch("skill_library_v2.agents.dim_worker._get_skill_generator") as gs, \
         patch("skill_library_v2.agents.dim_worker._get_critic") as gcrit:
        gc.return_value = type("M", (), {"run": AsyncMock(return_value=_cat_out())})()
        gs.return_value = type("M", (), {"run": AsyncMock(return_value=skill_out)})()
        gcrit.return_value = type("M", (), {"run": AsyncMock(return_value=_crit_out())})()

        result = await process_dim(_state())

    assert result["dim_status"]["d"] == "done"
    survivor = result["generated"]["d"][0]
    assert survivor.aliases == ["real"]
