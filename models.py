"""Pydantic models for the resume meta-tagging pipeline."""

from __future__ import annotations
from pydantic import BaseModel


# ── Candidate Details ──
class CandidateDetails(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    linkedin_url: str | None = None
    github_url: str | None = None


# ── Resume Block (internal) ──
class ResumeBlock(BaseModel):
    block_name: str  # Summary, Experience_0, Project_0, Skills_Dump, Education
    raw_text: str
    block_type: str  # summary | experience | project | skills_dump | education | other


# ── Zoned block preview (for debugging: what text went into each block before LLM) ──
class ZonedBlockPreview(BaseModel):
    block_name: str
    block_type: str
    raw_text: str


# ── Skill Trace (per-occurrence in a block) ──
class SkillTrace(BaseModel):
    provenance: str  # block_name it came from
    action_verb: str  # "Architected", "Led", "Built", "Listed"
    context: str  # 5-15 word exact quote from block
    co_dependent_skills: list[str] = []
    metric: str | None = None  # non-financial scale metric, null if none
    # Filled when aggregate(..., enrich_skill_temporal=True); from experience_detail
    last_used_date: str | None = None  # ISO date YYYY-MM-DD, end of role or "as of" if current
    computed_years_with_skill: float | None = None  # tenure in that block / role (approx.)


# ── Per-block skill entry (used inside BlockTagResult) ──
class BlockSkillEntry(BaseModel):
    skill: str
    action_verb: str
    context: str
    co_dependent_skills: list[str] = []
    metric: str | None = None


# ── Experience detail per role ──
class ExperienceDetail(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    is_current: bool = False
    key_achievements: list[str] = []
    quantifiers: list[str] = []
    tech_stack: list[str] = []


# ── Project detail ──
class ProjectDetail(BaseModel):
    project_name: str | None = None
    description: str | None = None
    tech_stack: list[str] = []
    quantifiers: list[str] = []
    key_highlights: list[str] = []
    # Verbatim project block (Client, Role, Technologies, description, responsibilities)
    full_block_text: str | None = None


class BlockTagResult(BaseModel):
    block_name: str
    block_type: str = "other"
    # Verbatim zoned block body (for UI / audit; same as hybrid block raw_text)
    raw_text: str | None = None
    skills: list[BlockSkillEntry] = []
    quantifiers: list[str] = []
    experience_detail: ExperienceDetail | None = None
    project_detail: ProjectDetail | None = None


# ── Context Meta-Tags ──
class SummaryMetaTags(BaseModel):
    key_highlights: list[str] = []
    quantifiers: list[str] = []
    years_of_experience: str | None = None
    domain_focus: list[str] = []


class ExperienceMetaTags(BaseModel):
    total_companies: int = 0
    current_company: str | None = None
    current_role: str | None = None
    experience_timeline: list[ExperienceDetail] = []
    top_quantifiers: list[str] = []
    combined_tech_stack: list[str] = []


class ProjectMetaTags(BaseModel):
    total_projects: int = 0
    projects: list[ProjectDetail] = []
    combined_tech_stack: list[str] = []


class SkillCategoryTags(BaseModel):
    languages: list[str] = []
    frameworks: list[str] = []
    databases: list[str] = []
    cloud_devops: list[str] = []
    tools: list[str] = []
    methodologies: list[str] = []
    domain_skills: list[str] = []
    soft_skills: list[str] = []


class ContextMetaTags(BaseModel):
    summary_tags: SummaryMetaTags = SummaryMetaTags()
    experience_tags: ExperienceMetaTags = ExperienceMetaTags()
    project_tags: ProjectMetaTags = ProjectMetaTags()
    skill_categories: SkillCategoryTags = SkillCategoryTags()
    resume_strength_signals: list[str] = []


# ── Global Parameters (7 extraction targets) ──
class ScaleAnchor(BaseModel):
    extracted_metric: str
    extracted_unit: str
    provenance: str
    raw_evidence: str


class EnumExtraction(BaseModel):
    normalized_value: str
    provenance: str
    raw_evidence: str


class GlobalParameters(BaseModel):
    scale_anchors: list[ScaleAnchor] = []
    codebase_lifecycle: list[EnumExtraction] = []
    product_domain: list[EnumExtraction] = []
    architectural_paradigm: list[EnumExtraction] = []
    compliance_exposure: list[EnumExtraction] = []
    leadership_footprint: list[EnumExtraction] = []
    cross_functional_area: list[EnumExtraction] = []


# ── Trajectory-based global skill index ──
class GlobalSkillEntry(BaseModel):
    skill: str
    # Sum of per-trace computed_years_with_skill when temporal enrich is on (naive sum across roles).
    computed_years_with_skill: float | None = None
    traces: list[SkillTrace]


class ExtractedLink(BaseModel):
    uri: str
    page: int
    anchor_text: str | None = None
    title: str | None = None
    x0: float | None = None
    y0: float | None = None
    x1: float | None = None
    y1: float | None = None
    top: float | None = None
    bottom: float | None = None
    width: float | None = None
    height: float | None = None


class HybridParseStage1Response(BaseModel):
    candidate: CandidateDetails
    extracted_links: list[ExtractedLink] = []
    parsed_text: str = ""
    zoned_blocks_preview: list[ZonedBlockPreview] = []


# ── Final Aggregated Output ──
class ResumeTaggingResponse(BaseModel):
    candidate: CandidateDetails
    extracted_links: list[ExtractedLink] = []
    context_meta_tags: ContextMetaTags
    global_skill_index: list[GlobalSkillEntry]
    blocks: list[BlockTagResult]
    global_parameters: GlobalParameters
    reasoning_log: list[str] = []
    # Debug: parsed text from file and zoned blocks (to verify parsing/zoning vs LLM)
    parsed_text: str = ""
    zoned_blocks_preview: list[ZonedBlockPreview] = []
