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

SkillNature = Literal["TOOL", "CONCEPT", "PRACTICE", "PLATFORM", "LANGUAGE", "CREDENTIAL"]


class SkillDraft(BaseModel):
    """Forward declaration — filled by Generator in Phase 2."""

    display_name: str
    slug: str
    category: SkillCategory
    sub_category: str | None = None
    skill_nature: SkillNature
    aliases: list[str] = Field(default_factory=list)


class ProposedNewSkill(SkillDraft):
    """Skill not found in the catalog; must carry evidence for review."""

    evidence_urls: list[HttpUrl]
    evidence_type: Literal[
        "github", "package_registry", "official_docs",
        "stackoverflow_tag", "job_posting_corpus",
    ]
    justification: str
