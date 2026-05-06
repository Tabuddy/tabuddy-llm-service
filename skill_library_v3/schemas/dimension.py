"""Pydantic schemas for the Stage 2 dimension generation output.

Stage 2 produces 12-20 candidate dimensions per role. The output is
intentionally allowed to contain a few duplicates / over-splits — Stage 3
reconciliation (a future plan) will resolve them via embedding similarity
and LLM merge/split decisions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class OverlapFlag(BaseModel):
    """One overlap signal between this candidate dimension and an existing
    dimension. Stage 3 reconciliation will resolve these.

    Three reference modes (one of these should be populated):
      * with_dim_id: a canonical-store dimension id
      * with_dim_name: a dimension display name (for cross-role hits where
        we don't have a stable id yet)
      * with_role: an adjacent role name when the overlap is cross-role
        ("Kafka also appears under Data Engineer's Stream Processing dim")
    """

    with_dim_id: str | None = None
    with_dim_name: str | None = None
    with_role: str | None = None
    reason: str = Field(min_length=10, max_length=500)


class CandidateDimension(BaseModel):
    """One candidate dimension. Stage 3 may merge, split, or keep-separate."""

    tentative_id: str = Field(pattern=r"^d_init_\d{2,3}$")
    name: str = Field(min_length=3, max_length=80)
    description: str = Field(min_length=20, max_length=600)
    in_scope: str = Field(min_length=10, max_length=600)
    out_of_scope: str = Field(min_length=10, max_length=600)
    exemplar_skills: list[str] = Field(min_length=3, max_length=15)
    overlap_flags: list[OverlapFlag] = Field(default_factory=list)


class DimensionList(BaseModel):
    """Stage 2 output — written to v2_run_log.planner_output of a stage2 row.

    ``role_id`` is optional at parse time because Stage2DimensionGeneratorAgent
    overrides it post-LLM with the slug we passed in. The LLM is told via
    prompt to focus on candidate_dimensions; if it omits the top-level
    role_id, that's fine — the agent fills it in.
    """

    role_id: str = ""
    candidate_dimensions: list[CandidateDimension] = Field(min_length=10, max_length=22)
