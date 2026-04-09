"""Pydantic models for the resume ranking pipeline."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, model_validator


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


# ── 4-Layer Classification (new taxonomy) ─────────────────────────────────────
class LayerPrediction(BaseModel):
    label: str
    score: float
    # "setfit" | "llm_rescue" | "inferred" | "mapped"
    confidence_source: str = "setfit"


class LayeredClassification(BaseModel):
    """4-layer hierarchical classification: Pillar → Layer → Activity → Platform/Tool.

    compatible_layers enables multi-expertise matching — a full-stack resume
    can be compatible with both App_Frontend and App_Backend JDs.
    """
    pillar: LayerPrediction
    layer: LayerPrediction | None = None
    activity: LayerPrediction | None = None
    platform_tool: LayerPrediction | None = None
    hierarchy_path: str
    final_label: str
    low_confidence: bool = False
    # e.g. ["App_Frontend", "App_Backend", "App_Fullstack"]
    compatible_layers: list[str] = []

    def to_legacy_tier(self) -> TierClassification:
        """Map LayeredClassification back to TierClassification for backward compat."""
        _PILLAR_TO_TIER2 = {
            "Application Engineering": "App_Engineering",
            "Data & Intelligence": "Data_Intelligence",
            "Infrastructure & Cloud": "Infra_Cloud",
            "Cyber Security": "Cyber_Security",
            "Product & Design": "Product_Design",
            "Platform & Tooling": "Infra_Cloud",
        }
        _PLATFORM_TO_STACK = {
            "Java": "Stack_Java",
            "Python": "Stack_Python",
            "Node.js": "Stack_Node",
            "React": "Stack_React",
            "React Native": "Stack_ReactNative",
            "Angular": "Stack_Angular",
            "iOS/Swift": "Stack_iOS",
            "Android/Kotlin": "Stack_Android",
        }
        tier1 = TierPrediction(label="Digital", score=self.pillar.score)
        tier2: TierPrediction | None = None
        tier3: TierPrediction | None = None

        t2_label = _PILLAR_TO_TIER2.get(self.pillar.label)
        if t2_label:
            tier2 = TierPrediction(label=t2_label, score=self.pillar.score)

        if self.platform_tool:
            stack = _PLATFORM_TO_STACK.get(self.platform_tool.label)
            if stack:
                tier3 = TierPrediction(
                    label=stack, score=self.platform_tool.score)

        path_parts = ["Digital"]
        if tier2:
            path_parts.append(tier2.label)
        if tier3:
            path_parts.append(tier3.label)

        return TierClassification(
            tier1=tier1,
            tier2=tier2,
            tier3=tier3,
            hierarchy_path=" > ".join(path_parts),
            final_label=tier3.label if tier3 else (
                tier2.label if tier2 else "Digital"),
            low_confidence=self.low_confidence,
        )


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
    layered_classification: LayeredClassification | None = None  # new 4-layer taxonomy
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
    # e.g. 7 for "7+ years", 0.5 for "6 months"
    required_years_min: float | None = None
    required_years_max: float | None = None     # e.g. 10 for "7-10 years"
    # "senior", "mid", "junior", "fresher"
    expected_seniority: str | None = None
    # raw text, e.g. "6+ months", "3-5 years"
    required_experience_text: str | None = None
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
    # all capability normalized_entities
    required_skills_normalized: list[str] = []
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
    layered_classification: LayeredClassification | None = None  # new 4-layer taxonomy
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
