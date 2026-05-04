"""Skill-level schemas and the canonical skill_category enum mirror.

`SKILL_CATEGORY_ENUM` mirrors the Postgres enum declared in
``db/schema.sql`` (lines 13-23). Keep the two in lockstep — any change to the
DB enum requires a paired update here, otherwise the Planner will emit
category values the DB will reject.

`SkillDraft` and `ProposedNewSkill` are forward-declared for Phase 2's
Generator; Phase 1 (Planner) does not produce skills. They live here now so
the ``PlanGraphState`` shape is stable across phases.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

# ─── Mirror of db/schema.sql : skill_category enum ──────────────────────────
SKILL_CATEGORY_ENUM: tuple[str, ...] = (
    "LANGUAGE",
    "FRAMEWORK",
    "LIBRARY",
    "DATABASE",
    "CLOUD_PLATFORM",
    "CLOUD_SERVICE",
    "DEVOPS_TOOL",
    "ARCHITECTURE_CONCEPT",
    "METHODOLOGY",
    "PROTOCOL",
    "OPERATING_SYSTEM",
    "SOFT_SKILL",
    "DOMAIN_CONCEPT",
    "CERTIFICATION",
    "BUILD_TOOL",
    "TESTING_TOOL",
    "MONITORING_TOOL",
    "SECURITY_TOOL",
    "DATA_TOOL",
    "DESIGN_TOOL",
    "COLLABORATION_TOOL",
    "MESSAGE_BROKER",
    "CONTAINER_RUNTIME",
    "ORCHESTRATION_TOOL",
    "WEB_SERVER",
    "ORM",
    "HARDWARE_TOOL",
    "IDE_TOOL",
    "VERSION_CONTROL",
)

SkillCategory = Literal[
    "LANGUAGE", "FRAMEWORK", "LIBRARY", "DATABASE", "CLOUD_PLATFORM",
    "CLOUD_SERVICE", "DEVOPS_TOOL", "ARCHITECTURE_CONCEPT", "METHODOLOGY",
    "PROTOCOL", "OPERATING_SYSTEM", "SOFT_SKILL", "DOMAIN_CONCEPT",
    "CERTIFICATION", "BUILD_TOOL", "TESTING_TOOL", "MONITORING_TOOL",
    "SECURITY_TOOL", "DATA_TOOL", "DESIGN_TOOL", "COLLABORATION_TOOL",
    "MESSAGE_BROKER", "CONTAINER_RUNTIME", "ORCHESTRATION_TOOL", "WEB_SERVER",
    "ORM", "HARDWARE_TOOL", "IDE_TOOL", "VERSION_CONTROL",
]

# Expanded nature set. The first 6 were the v1 values; the additional 7 cover
# values the LLM emits naturally when it sees the corresponding skill type
# (React=LIBRARY, Angular=FRAMEWORK, Node.js=RUNTIME, S3=CLOUD_SERVICE, MQTT=
# PROTOCOL, ISO 27001=STANDARD, Scrum=METHODOLOGY). Adding them avoids dim
# retries that can never converge.
SkillNature = Literal[
    "TOOL", "CONCEPT", "PRACTICE", "PLATFORM", "LANGUAGE", "CREDENTIAL",
    "LIBRARY", "FRAMEWORK", "RUNTIME", "PROTOCOL", "STANDARD",
    "METHODOLOGY", "CLOUD_SERVICE",
    # Round 3 additions surfaced from the 78-role run:
    "PATTERN", "API", "ALGORITHM",
]


class SkillDraft(BaseModel):
    """Concrete skill emitted by the Skill Generator.

    ``category_slug`` and ``sub_category_slug`` MUST match a proposal in the
    Categorizer's output for this dim — the validator enforces this. Slugs are
    resolved to live DB ids by the loader (``db/loader.py``).
    """

    display_name: str
    slug: str
    category_slug: str = Field(
        ..., max_length=80,
        description="Slug of one of the Categorizer's proposed categories.",
    )
    sub_category_slug: str = Field(
        ..., max_length=120,
        description="Slug of one of the Categorizer's proposed sub-categories.",
    )
    skill_nature: SkillNature
    aliases: list[str] = Field(default_factory=list)


class ProposedNewSkill(SkillDraft):
    """Skill not found in the catalog; must carry evidence for review.

    Unused in the current pipeline; retained for forward-compatibility with a
    future two-bucket Generator output where novel skills carry provenance.
    """

    evidence_urls: list[HttpUrl]
    evidence_type: Literal[
        "github", "package_registry", "official_docs",
        "stackoverflow_tag", "job_posting_corpus",
    ]
    justification: str


# ─── Skill Generator + Critic outputs ──────────────────────────────────────

class SkillGeneratorOutput(BaseModel):
    """What the Skill Generator emits for one dimension.

    Each ``SkillDraft`` must reference a (category_slug, sub_category_slug)
    pair that appears in the Categorizer's output for this dim.
    """

    dim_id: str = Field(..., description="dimension_id this batch belongs to.")
    skills: list[SkillDraft] = Field(
        ...,
        min_length=3,
        description=(
            "Concrete named skills under this dimension. 12-30 typical for "
            "narrow dims; 30-60 for broad dims. Hard cap of 80 enforced by "
            "dim_worker (overshoot is truncated, not rejected — the coverage-"
            "push prompt routinely emits 81-110)."
        ),
    )
    notes: str = Field(
        default="",
        description="Optional rationale or caveats from the Skill Generator.",
    )


# Backwards-compatible alias for one transitional commit. Will be removed in
# the dim_worker rewrite.
GeneratorOutput = SkillGeneratorOutput


CriticIssueType = Literal[
    "hallucinated",
    "wrong_category",
    "wrong_subcategory",
    "out_of_scope",
    "duplicate_alias",
    "non_skill_concept",
    "too_broad",
    "too_narrow",
    "under_covered",
]


class Correction(BaseModel):
    """One issue the Critic identified in the Generator's output."""

    slug: str = Field(..., description="The offending skill's slug.")
    issue_type: CriticIssueType
    suggested_fix: str = Field(
        ...,
        description="How the Generator should address this on the next pass.",
    )


class CriticOutput(BaseModel):
    """Per-dimension Critic verdict + corrections list."""

    dim_id: str
    corrections: list[Correction] = Field(default_factory=list)
    approved_slugs: list[str] = Field(
        default_factory=list,
        description="Slugs the Critic explicitly approved (no issues found).",
    )
    overall_verdict: Literal["approve", "revise", "reject"] = Field(
        ...,
        description=(
            "'approve' = no corrections; 'revise' = corrections listed, "
            "Generator should re-run; 'reject' = batch is unsalvageable, "
            "force a retry from scratch."
        ),
    )
