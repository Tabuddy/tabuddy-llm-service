"""Stage 8 catalog payload schemas.

CatalogPayload is the deterministic intermediate the loader produces
from a role's approved Stage 1-7 outputs. The persistence layer reads
this payload and writes it into the canonical relational tables
(``categories``, ``sub_categories``, ``dimensions``, ``canonical_skills``,
``skill_aliases``, ``role_dimensions``, ``dimension_skills``,
``dimension_categories``, ``skill_relationships``, ``skill_tags``).

We carry slugs (not DB ids) at this layer so the transform stays pure
and the persistence layer alone has to reason about ordering / FK
resolution.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# Closed enums mirroring db/schema.sql. Mismatch with the DB enum =
# load failure on INSERT, so test coverage on these is critical.
SkillNature = Literal[
    "TOOL", "CONCEPT", "PRACTICE", "PLATFORM", "LANGUAGE", "CREDENTIAL",
    "LIBRARY", "FRAMEWORK", "RUNTIME", "PROTOCOL", "STANDARD",
    "METHODOLOGY", "CLOUD_SERVICE", "PATTERN", "API", "ALGORITHM",
]
SkillVolatility = Literal["STABLE", "EVOLVING", "EMERGING", "DEPRECATED"]
SkillLifespan = Literal["EVERGREEN", "MULTI_YEAR", "SHORT_LIVED"]
VersionStrategy = Literal["MERGE_TO_PARENT", "SEPARATE_ENTITY", "NOT_APPLICABLE"]
EntitySource = Literal["MANUAL_CURATION", "AUTOMATED_DISCOVERY", "USER_SUBMISSION"]
AliasType = Literal[
    "CANONICAL", "ABBREVIATION", "VERSION", "MISSPELLING",
    "REGIONAL_VARIANT", "DEPRECATED_NAME", "COLLOQUIAL", "FULL_NAME", "ACRONYM",
]
License = Literal[
    "proprietary", "mit", "apache_2", "bsd", "gpl_v2", "gpl_v3", "lgpl",
    "mpl", "isc", "unlicense", "cc_by", "cc_by_sa", "other_open", "unknown",
]
RelationshipType = Literal[
    "TRANSFERS_TO", "REQUIRES", "COMPETES_WITH", "CO_EVOLVES_WITH",
]


class CategoryRow(BaseModel):
    slug: str = Field(min_length=1, max_length=80)
    display_name: str = Field(min_length=1, max_length=120)
    description: str | None = None


class SubCategoryRow(BaseModel):
    slug: str = Field(min_length=1, max_length=120)
    display_name: str = Field(min_length=1, max_length=160)
    category_slug: str = Field(min_length=1, max_length=80)
    description: str | None = None


class DimensionRow(BaseModel):
    slug: str = Field(min_length=1, max_length=160)
    display_name: str = Field(min_length=1, max_length=200)
    rationale: str | None = None
    difficulty_hint: str = Field(default="well_known", max_length=20)


class SkillRow(BaseModel):
    slug: str = Field(min_length=1, max_length=120)
    display_name: str = Field(min_length=1, max_length=100)
    category_slug: str = Field(min_length=1, max_length=80)
    sub_category_slug: str | None = None
    parent_skill_slug: str | None = None
    skill_nature: SkillNature
    volatility: SkillVolatility = "STABLE"
    typical_lifespan: SkillLifespan = "EVERGREEN"
    version_strategy: VersionStrategy = "NOT_APPLICABLE"
    version_tag: str | None = Field(default=None, max_length=50)
    is_extractable: bool = True
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    # Stage 7 enrichment promoted to first-class columns.
    vendor: str | None = Field(default=None, max_length=120)
    license: License | None = None
    year_introduced: int | None = None
    maturity_reasoning: str | None = None


class AliasRow(BaseModel):
    skill_slug: str = Field(min_length=1, max_length=120)
    alias_text: str = Field(min_length=1, max_length=200)
    alias_type: AliasType
    is_primary: bool = False


class RoleAliasRow(BaseModel):
    role_slug: str = Field(min_length=1)
    alias_text: str = Field(min_length=1, max_length=200)
    alias_type: AliasType
    is_primary: bool = False


class RoleDimRow(BaseModel):
    role_slug: str = Field(min_length=1)
    dimension_slug: str = Field(min_length=1)


class DimSkillRow(BaseModel):
    dimension_slug: str = Field(min_length=1)
    skill_slug: str = Field(min_length=1)


class DimCatRow(BaseModel):
    dimension_slug: str = Field(min_length=1)
    category_slug: str = Field(min_length=1)
    sub_category_slug: str | None = None
    rationale: str | None = None


class RelRow(BaseModel):
    source_skill_slug: str = Field(min_length=1)
    target_skill_slug: str = Field(min_length=1)
    relationship_type: RelationshipType
    base_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str | None = None


class TagRow(BaseModel):
    skill_slug: str = Field(min_length=1)
    tag: str = Field(min_length=1, max_length=80)


class CatalogPayload(BaseModel):
    """One role's complete contribution to the canonical catalog."""

    role_slug: str = Field(min_length=1)
    role_display: str = Field(min_length=1)

    categories: list[CategoryRow] = Field(default_factory=list)
    sub_categories: list[SubCategoryRow] = Field(default_factory=list)
    dimensions: list[DimensionRow] = Field(default_factory=list)
    skills: list[SkillRow] = Field(default_factory=list)
    aliases: list[AliasRow] = Field(default_factory=list)
    role_aliases: list[RoleAliasRow] = Field(default_factory=list)

    role_dimensions: list[RoleDimRow] = Field(default_factory=list)
    dimension_skills: list[DimSkillRow] = Field(default_factory=list)
    dimension_categories: list[DimCatRow] = Field(default_factory=list)
    relationships: list[RelRow] = Field(default_factory=list)
    tags: list[TagRow] = Field(default_factory=list)
