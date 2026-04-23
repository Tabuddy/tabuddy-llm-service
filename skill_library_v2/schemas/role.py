"""Role-level schemas — the Planner's input and output contracts.

The Planner decomposes a single role into a minimal set of orthogonal skill
dimensions. These schemas are the *contract* downstream agents rely on;
changes here will cascade. See ``tabuddy_v2_architecture.md`` §13 and §2.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from skill_library_v2.schemas.skill import SKILL_CATEGORY_ENUM, SkillCategory

DifficultyHint = Literal["well_known", "ambiguous"]


class RoleInput(BaseModel):
    """Seed for a pipeline run. The caller provides both fields."""

    role_id: str = Field(
        ...,
        description="Machine-safe slug for the role (e.g., 'backend_engineer').",
    )
    role_display: str = Field(
        ...,
        description="Human-readable role name (e.g., 'Backend Engineer').",
    )

    @field_validator("role_id")
    @classmethod
    def _slug_shape(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "_-" for c in v):
            raise ValueError("role_id must be a slug of [a-zA-Z0-9_-]")
        return v.lower()


class WebHint(BaseModel):
    """Brave Search snippet that informed the Planner's reasoning."""

    title: str
    url: str
    description: str


class DimensionSlice(BaseModel):
    """One orthogonal competency axis for the role.

    Downstream agents (Retrieval Service, Generator) iterate over these and
    produce skills *within* each dimension. Quality here bounds quality
    downstream — a vague dimension produces noisy skills.
    """

    dimension_id: str = Field(
        ...,
        description="Stable slug, e.g. 'languages', 'web_frameworks'.",
    )
    dimension_name: str = Field(..., description="Human-readable display name.")
    rationale: str = Field(
        ...,
        description="One-paragraph explanation of why this dimension is "
                    "role-critical and non-overlapping with the others.",
    )
    expected_skill_kinds: list[SkillCategory] = Field(
        ...,
        min_length=1,
        description="skill_category enum values this dimension will populate.",
    )
    difficulty_hint: DifficultyHint = Field(
        default="well_known",
        description="'ambiguous' routes to the expensive path in the "
                    "Dimension Router (Phase 2).",
    )

    @field_validator("dimension_id")
    @classmethod
    def _slug_shape(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "_-" for c in v):
            raise ValueError("dimension_id must be a slug of [a-zA-Z0-9_-]")
        return v.lower()

    @field_validator("expected_skill_kinds")
    @classmethod
    def _known_categories(cls, v: list[str]) -> list[str]:
        unknown = [c for c in v if c not in SKILL_CATEGORY_ENUM]
        if unknown:
            raise ValueError(
                f"unknown skill_category values: {unknown}. "
                f"Allowed: {SKILL_CATEGORY_ENUM}"
            )
        return v


class PlannerOutput(BaseModel):
    """What the Planner LLM returns (and what we persist to v2_run_log)."""

    role_archetype: str = Field(
        ...,
        description="One-sentence classification of the role's broadest family.",
    )
    dimensions: list[DimensionSlice] = Field(
        ...,
        min_length=3,
        max_length=15,
        description="3-15 orthogonal dimensions. 5-10 typical.",
    )
    reasoning: str = Field(
        ...,
        description="Compression of the CoT steps; persisted for audit and "
                    "used by reviewers to understand the Planner's judgment.",
    )
    flagged_for_review: list[str] = Field(
        default_factory=list,
        description="dimension_id values the Planner could not confidently "
                    "map to a skill_category; routed to v2_review_queue.",
    )
