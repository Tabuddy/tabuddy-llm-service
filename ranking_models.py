"""Pydantic models for the resume ranking pipeline."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel


# ── Experience Level ──────────────────────────────────────────────────────────
class ExperienceLevel(str, Enum):
    SENIOR = "SENIOR"    # 7+ years
    MID = "MID"          # 3-7 years
    FRESHER = "FRESHER"  # 0-3 years


# ── SetFit Tier Classification ────────────────────────────────────────────────
class TierPrediction(BaseModel):
    label: str
    score: float


class TierClassification(BaseModel):
    tier1: TierPrediction
    tier2: TierPrediction | None = None
    tier3: TierPrediction | None = None
    hierarchy_path: str
    final_label: str
    low_confidence: bool = False  # True if any tier score < threshold


# ── JD Atom Models ────────────────────────────────────────────────────────────
class JDAtom(BaseModel):
    atom_type: str
    raw_text: str
    normalized_entity: str
    bundle_id: str | None = None


class JDCapabilityAtom(JDAtom):
    capability_type: str | None = None
    capability_subtype: str | None = None
    ownership_level: str | None = None      # Primary Owner | Shared Owner | Support
    intensity_level: str | None = None      # High | Medium | Low
    classification: str | None = None       # PRIMARY | SECONDARY
    motion: str | None = None


class JDScaleAtom(JDAtom):
    value: str | None = None


class JDProfile(BaseModel):
    raw_jd: str
    tier_classification: TierClassification
    # Global atoms
    role_family: str | None = None
    role_objective: str | None = None
    role_scope: str | None = None
    delivery_mode: str | None = None
    industry_vertical: str | None = None
    industry_sub_vertical: str | None = None
    business_model: str | None = None
    compliance_context: list[str] = []
    # Experience requirements (extracted from JD)
    required_years_min: float | None = None     # e.g. 7 for "7+ years", 0.5 for "6 months"
    required_years_max: float | None = None     # e.g. 10 for "7-10 years"
    expected_seniority: str | None = None       # "senior", "mid", "junior", "fresher"
    required_experience_text: str | None = None # raw text, e.g. "6+ months", "3-5 years"
    # Capability atoms
    core_capabilities: list[JDCapabilityAtom] = []
    secondary_capabilities: list[JDCapabilityAtom] = []
    adjacent_capabilities: list[JDCapabilityAtom] = []
    # Scale & impact
    scale_atoms: list[JDScaleAtom] = []
    impact_atoms: list[JDAtom] = []
    # Context
    stakeholder_atoms: list[JDAtom] = []
    process_methodology_atoms: list[JDAtom] = []
    environment_atoms: list[JDAtom] = []
    certification_atoms: list[JDAtom] = []
    # Derived normalized sets (for fast matching)
    required_skills_normalized: list[str] = []   # all capability normalized_entities
    required_tech_normalized: list[str] = []      # tech-specific atoms


# ── Scoring Breakdown ─────────────────────────────────────────────────────────
class DimensionScore(BaseModel):
    dimension: str
    raw_score: float          # 0-100
    weight: float             # 0-1
    weighted_score: float     # raw_score * weight
    evidence: list[str] = []  # brief explanation strings


class ResumeRankResult(BaseModel):
    rank: int = 0
    candidate_name: str | None = None
    candidate_email: str | None = None
    final_score: float                        # 0-100
    experience_level: ExperienceLevel
    tier_classification: TierClassification
    dimension_scores: list[DimensionScore]
    resume_strength_signals: list[str] = []
    tier_mismatch: bool = False
    low_confidence_classification: bool = False


class RankingSession(BaseModel):
    jd_profile: JDProfile
    results: list[ResumeRankResult]           # sorted desc by final_score
    total_resumes: int
    processing_errors: list[str] = []


# ── API Request/Response Models ───────────────────────────────────────────────
class ParseJDRequest(BaseModel):
    jd_text: str


class ScoreResumeRequest(BaseModel):
    jd_profile: JDProfile


class ClassifyTextRequest(BaseModel):
    text: str
