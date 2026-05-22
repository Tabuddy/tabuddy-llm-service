"""Pydantic schema for the Stage 1 role card.

A role card is a tight compression of an approved Stage 0 charter that
downstream stages (Stage 2 dimension generation, Stage 5 placement, etc.)
read as a single artifact instead of consuming the full charter prose.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# Closed list of role families. The Stage 0 planner agent prompt enumerates
# these same nine families; keeping them in sync keeps Stage 1's classification
# anchored to the same vocabulary the planner already reasons about.
FAMILY_VALUES: tuple[str, ...] = (
    "Software Engineering",
    "Infrastructure & Platform",
    "Data & ML",
    "Security",
    "ERP & Enterprise",
    "Digital & Martech",
    "QA & SDET",
    "Hardware & Firmware",
    "Support/Ops Engineering",
)

RoleFamily = Literal[
    "Software Engineering",
    "Infrastructure & Platform",
    "Data & ML",
    "Security",
    "ERP & Enterprise",
    "Digital & Martech",
    "QA & SDET",
    "Hardware & Firmware",
    "Support/Ops Engineering",
]


class SeniorityBand(BaseModel):
    junior: str = Field(min_length=10, max_length=600)
    mid: str = Field(min_length=10, max_length=600)
    senior: str = Field(min_length=10, max_length=600)
    staff: str = Field(min_length=10, max_length=600)


class RoleAliasOnlyOutput(BaseModel):
    """Subset of Stage 1 role card for LinkedIn title-only backfill."""

    canonical_name: str = Field(min_length=2, max_length=80)
    aliases: list[str] = Field(default_factory=list, max_length=10)


class RoleCard(BaseModel):
    """Stage 1 output — written to v2_run_log.planner_output of a stage1 row.

    v1.2: ``primary_responsibilities`` and ``common_artifacts`` are
    semicolon-delimited strings (denser for Stage 2's prompt budget) rather
    than the v1.0/v1.1 ``list[str]``. ``adjacent_roles`` and
    ``always_load_skills`` are now first-class fields on the role card so
    Stage 2 can read everything from one artifact instead of joining back
    to the charter.
    """

    role_id: str
    canonical_name: str = Field(min_length=2, max_length=80)
    aliases: list[str] = Field(default_factory=list, max_length=10)
    family: RoleFamily
    definition: str = Field(min_length=20, max_length=400)
    primary_responsibilities: str = Field(min_length=30, max_length=1200)
    common_artifacts: str = Field(min_length=20, max_length=1200)
    distinguishing_tasks: str = Field(min_length=20, max_length=600)
    seniority_signals: SeniorityBand
    adjacent_roles: list[str] = Field(default_factory=list, min_length=0, max_length=15)
    always_load_skills: list[str] = Field(default_factory=list, min_length=0, max_length=10)
