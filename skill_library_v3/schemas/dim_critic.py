"""Pydantic schema for the Stage 2.5 critic LLM response.

Kept lenient — defaults on every field, no enum constraints — because
the LLM's first-pass output is corrected post-hoc by
``post_process_critic_response`` rather than rejected at parse time.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CriticProposedDim(BaseModel):
    axis_slug: str = ""
    name: str = ""
    description: str = ""
    in_scope: str = ""
    exemplar_skills: list[str] = Field(default_factory=list)
    out_of_scope: str = ""


class CriticOutput(BaseModel):
    covered: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    proposed_dims: list[CriticProposedDim] = Field(default_factory=list)
