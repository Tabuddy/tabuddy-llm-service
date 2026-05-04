"""Categorizer agent output schemas.

The Categorizer reads a DimensionSlice and proposes the (category, sub_category)
buckets the dimension expects. The Skill Generator then fills each bucket.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


def _check_slug(v: str) -> str:
    if not _SLUG_RE.match(v):
        raise ValueError(
            f"slug must be lowercase snake_case ([a-z0-9_]+): {v!r}"
        )
    return v


class DimensionCategoryProposal(BaseModel):
    """One (category, sub_category) bucket the categorizer proposes for a dim.

    Slugs only — the loader resolves them to live DB ids via
    ``ON CONFLICT (slug) DO NOTHING`` upserts. The aggregator deduplicates by
    slug across dimensions, so the same (cat, sub_cat) pair appears once in
    the final ``categories`` and ``sub_categories`` tables even if multiple
    dims emit it.
    """

    category_slug: str = Field(..., max_length=80)
    category_display: str = Field(..., max_length=120)
    sub_category_slug: str = Field(..., max_length=120)
    sub_category_display: str = Field(..., max_length=160)
    rationale: str = Field(
        ...,
        description="One-sentence why this bucket is in scope for the dim.",
    )
    expected_skill_count_hint: int = Field(
        default=10,
        ge=1,
        le=80,
        description="Soft target for the Skill Generator. Not enforced.",
    )

    @field_validator("category_slug", "sub_category_slug")
    @classmethod
    def _slug(cls, v: str) -> str:
        return _check_slug(v)


class CategorizerOutput(BaseModel):
    """All buckets the categorizer proposes for one dimension."""

    dim_id: str
    proposals: list[DimensionCategoryProposal] = Field(
        ...,
        min_length=1,
        description="Variable length. Typical 2-8 per dim; can exceed 8 if "
                    "the dim genuinely spans more buckets.",
    )
    notes: str = Field(default="")
