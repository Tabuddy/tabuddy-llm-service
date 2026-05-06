"""Stage 7 enrichment schemas.

Five sub-models corresponding to the five parallel agents (7a-7e), plus
a merged ``SkillEnrichment`` record that's one row of the Stage 7 output
list. Cross-field validation catches LLM mistakes early — e.g.
ambiguity_flag=true requires non-empty confused_with, versioned=true
requires current_version.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ── closed enums ──────────────────────────────────────────────────────────


LICENSE_VALUES: tuple[str, ...] = (
    "proprietary",
    "mit",
    "apache_2",
    "bsd",
    "gpl_v2",
    "gpl_v3",
    "lgpl",
    "mpl",
    "isc",
    "unlicense",
    "cc_by",
    "cc_by_sa",
    "other_open",
    "unknown",
)

License = Literal[
    "proprietary", "mit", "apache_2", "bsd", "gpl_v2", "gpl_v3", "lgpl",
    "mpl", "isc", "unlicense", "cc_by", "cc_by_sa", "other_open", "unknown",
]


MATURITY_VALUES: tuple[str, ...] = ("well_known", "emerging", "niche", "deprecated")

MaturityLevel = Literal["well_known", "emerging", "niche", "deprecated"]


# ── 7a: Vendor + License ──────────────────────────────────────────────────


class VendorLicense(BaseModel):
    """Output of agent 7a. Concept / SoftSkill / Methodology entries
    legitimately have null vendor and null license."""

    vendor: str | None = Field(default=None, max_length=120)
    license: License | None = None
    year_introduced: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _year_sanity(self) -> "VendorLicense":
        if self.year_introduced is None:
            return self
        # Programming history starts ~1960 (FORTRAN, COBOL); newer than
        # next year is the LLM hallucinating future dates.
        max_year = datetime.now().year + 1
        if self.year_introduced < 1960 or self.year_introduced > max_year:
            raise ValueError(
                f"year_introduced={self.year_introduced} outside plausible "
                f"range [1960, {max_year}]"
            )
        return self


# ── 7b: Maturity ──────────────────────────────────────────────────────────


class Maturity(BaseModel):
    """Output of agent 7b. The four levels mirror the blueprint exactly."""

    maturity: MaturityLevel
    reasoning: str = Field(min_length=20, max_length=600)
    confidence: float = Field(ge=0.0, le=1.0)


# ── 7c: Context Keywords ──────────────────────────────────────────────────


class ContextKeywords(BaseModel):
    """Output of agent 7c. 8-15 distinctive co-occurring terms — the
    skill's lexical neighborhood, useful for downstream extraction
    boosting and for surfacing related skills the LLM might miss."""

    context_keywords: list[str] = Field(min_length=5, max_length=18)

    @model_validator(mode="after")
    def _no_empty_strings(self) -> "ContextKeywords":
        for kw in self.context_keywords:
            if not isinstance(kw, str) or not kw.strip():
                raise ValueError(
                    "context_keywords must not contain empty / whitespace-only entries"
                )
        return self


# ── 7d: Ambiguity ─────────────────────────────────────────────────────────


class Ambiguity(BaseModel):
    """Output of agent 7d. When ``ambiguity_flag`` is true, ``confused_with``
    MUST list at least one skill the LLM thinks could be confused — an
    empty list with the flag set is meaningless and gets rejected."""

    ambiguity_flag: bool
    confused_with: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=20, max_length=600)

    @model_validator(mode="after")
    def _flag_implies_confused_with(self) -> "Ambiguity":
        if self.ambiguity_flag and not self.confused_with:
            raise ValueError(
                "ambiguity_flag=True requires at least one entry in confused_with"
            )
        return self


# ── 7e: Versioning ────────────────────────────────────────────────────────


class Versioning(BaseModel):
    """Output of agent 7e. When ``versioned`` is true, ``current_version``
    must be set so we know what the canonical version is."""

    versioned: bool
    current_version: str | None = Field(default=None, max_length=40)
    version_aliases: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _versioned_implies_current(self) -> "Versioning":
        if self.versioned and not self.current_version:
            raise ValueError(
                "versioned=True requires current_version to be set"
            )
        return self


# ── merged record + batch ─────────────────────────────────────────────────


class SkillEnrichment(BaseModel):
    """One skill's enrichment — assembled from the five agents' outputs.
    Stored as one element of ``planner_output.enrichments`` on the
    Stage 7 v2_run_log row."""

    skill_id: str = Field(min_length=1)
    vendor_license: VendorLicense
    maturity: Maturity
    context_keywords: ContextKeywords
    ambiguity: Ambiguity
    versioning: Versioning


class SkillEnrichmentBatch(BaseModel):
    """Wrapper for batch LLM responses. Stage 7 doesn't actually batch
    (per-skill agents are already parallelized), but keeping the wrapper
    pattern consistent with prior stages avoids surprises in the runner."""

    skills: list[SkillEnrichment] = Field(default_factory=list)
