"""Mock the categorizer + skill_generator + critic; assert dim_worker patches the new state shape."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

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


async def test_process_dim_patches_dim_categories_and_skills():
    cat_out = CategorizerOutput(
        dim_id="frontend_frameworks",
        proposals=[DimensionCategoryProposal(
            category_slug="framework",
            category_display="Framework",
            sub_category_slug="frontend_framework",
            sub_category_display="Frontend Framework",
            rationale="UI libraries",
            expected_skill_count_hint=5,
        )],
    )
    skill_out = SkillGeneratorOutput(
        dim_id="frontend_frameworks",
        skills=[
            SkillDraft(
                display_name=f"Skill{i}",
                slug=f"skill_{i}",
                category_slug="framework",
                sub_category_slug="frontend_framework",
                skill_nature="TOOL",
            )
            for i in range(8)
        ],
    )
    crit_out = CriticOutput(
        dim_id="frontend_frameworks",
        corrections=[],
        approved_slugs=["skill_0"],
        overall_verdict="approve",
    )

    with patch("skill_library_v2.agents.dim_worker._get_categorizer") as gc, \
         patch("skill_library_v2.agents.dim_worker._get_skill_generator") as gs, \
         patch("skill_library_v2.agents.dim_worker._get_critic") as gcrit:
        gc.return_value = type("M", (), {"run": AsyncMock(return_value=cat_out)})()
        gs.return_value = type("M", (), {"run": AsyncMock(return_value=skill_out)})()
        gcrit.return_value = type("M", (), {"run": AsyncMock(return_value=crit_out)})()

        state = {
            "dim_id": "frontend_frameworks",
            "dim": {
                "dimension_id": "frontend_frameworks",
                "dimension_name": "Frontend Frameworks",
                "rationale": "UI frameworks",
                "expected_skill_kinds": ["FRAMEWORK"],
                "difficulty_hint": "well_known",
            },
            "role_id": "frontend_developer",
            "role_display": "Frontend Developer",
            "dimensions": [],
            "planner_web_hints": [],
        }
        out = await process_dim(state)

    assert out["dim_status"]["frontend_frameworks"] == "done"
    assert out["dim_categories"]["frontend_frameworks"].dim_id == "frontend_frameworks"
    assert len(out["generated"]["frontend_frameworks"]) == 8
    assert out["generated"]["frontend_frameworks"][0].slug == "skill_0"
    assert out["dim_critic_verdict"]["frontend_frameworks"] == "approve"
    assert out["dim_retry_count"]["frontend_frameworks"] == 0


async def test_process_dim_fails_when_skill_in_unknown_bucket():
    """Validator should reject and the worker should fail after MAX_ATTEMPTS."""
    cat_out = CategorizerOutput(
        dim_id="d", proposals=[DimensionCategoryProposal(
            category_slug="framework", category_display="F",
            sub_category_slug="frontend_framework", sub_category_display="FF",
            rationale="r", expected_skill_count_hint=5,
        )],
    )
    skill_out = SkillGeneratorOutput(
        dim_id="d",
        skills=[SkillDraft(
            display_name=f"S{i}", slug=f"s_{i}",
            category_slug="unknown",  # not in cat_out
            sub_category_slug="nope",
            skill_nature="TOOL",
        ) for i in range(8)],
    )
    crit_out = CriticOutput(
        dim_id="d", corrections=[], approved_slugs=[], overall_verdict="approve",
    )

    with patch("skill_library_v2.agents.dim_worker._get_categorizer") as gc, \
         patch("skill_library_v2.agents.dim_worker._get_skill_generator") as gs, \
         patch("skill_library_v2.agents.dim_worker._get_critic") as gcrit:
        gc.return_value = type("M", (), {"run": AsyncMock(return_value=cat_out)})()
        gs.return_value = type("M", (), {"run": AsyncMock(return_value=skill_out)})()
        gcrit.return_value = type("M", (), {"run": AsyncMock(return_value=crit_out)})()

        state = {
            "dim_id": "d",
            "dim": {
                "dimension_id": "d", "dimension_name": "D",
                "rationale": "r", "expected_skill_kinds": ["FRAMEWORK"],
                "difficulty_hint": "well_known",
            },
            "role_id": "r", "role_display": "R",
            "dimensions": [], "planner_web_hints": [],
        }
        out = await process_dim(state)

    assert out["dim_status"]["d"] == "failed"
    assert out["dim_retry_count"]["d"] == 3  # MAX_ATTEMPTS
    assert out["dim_categories"]["d"].dim_id == "d"
    # An error should be logged
    assert any("exhausted" in e["message"] for e in out.get("errors", []))
