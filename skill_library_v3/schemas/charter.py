"""Pydantic schemas for the Stage 0 charter output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class OutOfScopeItem(BaseModel):
    task: str = Field(min_length=4)
    owned_by: str = Field(min_length=2)


class AdjacentRole(BaseModel):
    role: str = Field(min_length=2)
    boundary_note: str = Field(min_length=4)


class CharterOutput(BaseModel):
    """Stage 0 charter — produced by Stage0CharterAgent, written to
    v2_run_log.planner_output, gated by human review before Stage 1 starts."""

    role_id: str
    role_name: str
    definition: str = Field(min_length=20, max_length=600)
    in_scope: list[str] = Field(min_length=8, max_length=12)
    out_of_scope: list[OutOfScopeItem] = Field(default_factory=list)
    adjacent_roles: list[AdjacentRole] = Field(default_factory=list)
    always_load_skills: list[str] = Field(default_factory=list)
