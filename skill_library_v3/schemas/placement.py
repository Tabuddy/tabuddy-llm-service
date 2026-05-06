"""Pydantic schemas for the Stage 5 skill-placement output.

A PlacedSkill anchors one skill in one role's locked-dim structure:
exactly ONE primary dimension (its home) and (rarely) up to 2 secondary
dimensions. The schema's cross-field validators encode the blueprint's
"secondary is rare" property — the primary cannot also be secondary,
duplicates are rejected, and the cap of 2 is structural.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class PlacedSkill(BaseModel):
    """Stage 5 output for one skill within one role."""

    skill_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    primary_dimension: str = Field(min_length=1)
    secondary_dimensions: list[str] = Field(default_factory=list, max_length=2)
    placement_confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=20, max_length=600)

    @model_validator(mode="after")
    def _no_primary_in_secondaries(self) -> "PlacedSkill":
        if self.primary_dimension in self.secondary_dimensions:
            raise ValueError(
                "primary_dimension must not appear in secondary_dimensions; "
                "the primary is the home, secondaries are additional anchors"
            )
        if len(self.secondary_dimensions) != len(set(self.secondary_dimensions)):
            raise ValueError(
                "secondary_dimensions must not contain duplicates"
            )
        return self


class PlacedSkillBatch(BaseModel):
    """Wrapper so ``call_json`` can validate the LLM's batch response.

    OpenAI JSON-object response_format requires a top-level object, not
    a bare list. The LLM emits ``{"skills": [...]}``.
    """

    skills: list[PlacedSkill] = Field(default_factory=list)
