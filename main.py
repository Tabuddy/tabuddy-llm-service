from __future__ import annotations
from hybrid_block_builder import build_hybrid_blocks
from docling_api import docling_router
from models import (
    ResumeTaggingResponse,
    HybridParseStage1Response,
    ZonedBlockPreview,
    ExtractedLink,
    ResumeBlock,
)
from aggregator import aggregate
from meta_tag_extractor import extract_context_meta_tags
from global_parameter_extractor import extract_global_parameters
from block_tagger import tag_all_blocks
from nlp_filter import filter_blocks
from candidate_extractor import extract_candidate_details, extract_candidate_details_for_hybrid
from resume_zoner import zone_resume
from resume_parser import extract_docx_plain_text, extract_pdf_links, extract_text
import skill_library
import db
from normalizer import normalize_skills
from ranking_models import (
    JDProfile,
    RankingSession,
    ResumeRankResult,
    ParseJDRequest,
    ScoreResumeRequest,
    ClassifyTextRequest,
    TierClassification,
    TierPrediction,
)
from resume_scorer import score_resume, detect_experience_level, estimate_candidate_years
from jd_parser import parse_jd, _llm_classify_text
from section_scorer import score_all_sections
from skill_matcher import process_jd
from non_skill_repository import NonSkillRepository
from unknown_word_classifier import AzureUnknownWordClassifier
from reverse_planner_llm import AzureReversePlannerLLM
from skill_library_repository import SkillLibraryRepository
from jd_pipeline_run_repository import JdPipelineRunRepository
from skill_library_v2.agents.planner import PlannerAgent
from new_skill_orchestrator import (
    NewSkillMetaV3,
    _derive_legacy_fields,
    _run_enrichment,
    enrich_one_existing_canonical_skill,
    enrich_unmatched_skills_batched,
    find_dimensions_by_identity_skill_library,
)
from skill_library_v3.schemas.typology import SkillType, TypedSkill
from skill_library_v3.catalog_transform import (
    map_maturity_to_volatility,
    map_versioned_to_strategy,
)
from cost_tracker import CostAccumulator
from pathlib import Path
import json
import tempfile
import time
from datetime import datetime
import logging
import asyncio
from typing import Any, cast
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from model_azure import download_models_from_azure, upload_models_to_azure
from model_pipeline import router as model_pipeline_router
from canonical_skill_api import canonical_skill_router
from generate_skills_router import generate_skills_router
from skill_library_v3.db import repository as v3_repo

# Suppress noisy third-party progress bars BEFORE any imports that pull in tqdm
import os
os.environ["TQDM_DISABLE"] = "1"


# ── Ranking system imports ────────────────────────────────────────────────────


logger = logging.getLogger(__name__)

_BATCH_RESUME_CONCURRENCY = max(
    1, int(os.getenv("RANKING_BATCH_CONCURRENCY", "4"))
)


load_dotenv(os.getenv("ENV_FILE", ".env"), override=True)

logging.basicConfig(level=logging.INFO)

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logging.getLogger("RapidOCR").setLevel(logging.WARNING)
logging.getLogger("rapidocr").setLevel(logging.WARNING)


# Download models synchronously before Uvicorn starts the ASGI lifespan.
# This strictly evades the Uvicorn 60-second lifespan timeout logic.
if os.getenv("ENV") == "PROD":
    logger.info(
        "☁️ Production environment detected. Syncing models from Azure before starting server...")
    download_models_from_azure()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize MongoDB skill library on startup; shutdown cleanly."""
    logger.info("🚀 Starting TABuddy service (LLM-based classification)...")

    # Initialize MongoDB connection and merged skill library
    await db.init_db()
    await db.seed_static_skills()
    await skill_library.init()

    # v3 stage 0: reap any 'running' charter runs left behind by a prior
    # uvicorn crash so the UI doesn't show them as in-flight forever.
    try:
        await asyncio.to_thread(v3_repo.recover_orphan_runs, stale_after_seconds=600)
    except Exception:  # noqa: BLE001 — never block startup on this
        logger.exception("[v3] orphan-run recovery failed; continuing startup")

    # Ensure JD pipeline history tables exist (never blocks startup).
    try:
        await asyncio.to_thread(JdPipelineRunRepository().ensure_schema)
    except Exception as exc:
        logger.warning(
            "JD pipeline history schema bootstrap failed (continuing): %s", exc
        )

    yield
    logger.info("Shutting down TABuddy service")


app = FastAPI(
    title="TABuddy – Resume Intelligence & Ranking Service",
    description=(
        "Contextual resume parsing, skill extraction, global parameter tagging, "
        "and deterministic resume ranking for HR tech."
    ),
    version="0.4.0",
    lifespan=lifespan,
)

app.include_router(docling_router)
app.include_router(model_pipeline_router)
app.include_router(canonical_skill_router)
app.include_router(generate_skills_router)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


# ── Skill Normalization models (existing) ──
class SkillsRequest(BaseModel):
    skills: list[str] = Field(..., min_length=1, max_length=500)


class NormalizedSkillItem(BaseModel):
    original: str
    normalized: str
    method: str  # dictionary | fuzzy | llm | unmatched
    confidence: float


class SkillsResponse(BaseModel):
    normalized_skills: list[str]
    details: list[NormalizedSkillItem]


class JDSkillPipelineRequest(BaseModel):
    jd_text: str = Field(..., min_length=1)


class JdRoleHint(BaseModel):
    """Umbrella role from API 1 (same LLM call as word classifier); soft prior for API 2."""

    display_name: str
    slug: str = ""
    role_archetype: str | None = None
    rationale: str | None = None
    role_aliases: list[str] = Field(default_factory=list)


class FinalSkillItem(BaseModel):
    skill_name: str
    is_primary: bool


class RoleSignalItem(BaseModel):
    # role_id is None for "Case NEW" synthesized roles whose catalog entry
    # is still being upserted by v3 in the background.
    role_id: int | None = None
    slug: str
    display_name: str
    score: float                   # 0.0–1.0 (× 100 = percentage)
    matched_count: int | None = None   # skill_match only
    total_count: int | None = None     # skill_match only


class Stage3SignalsResponse(BaseModel):
    skill_match_roles: list[RoleSignalItem] = Field(default_factory=list)
    alias_match_roles: list[RoleSignalItem] = Field(default_factory=list)
    kra_match_roles: list[RoleSignalItem] = Field(default_factory=list)
    alias_found: bool = False      # True when at least one alias_match_roles result returned


class Stage4DecisionResponse(BaseModel):
    case: str
    chosen_role: RoleSignalItem | None = None
    confidence: float = 0.0
    llm2_fired: bool = False
    llm2_reasoning: str | None = None
    alias_collision_detected: bool = False
    queued: bool = False
    reasoning: str = ""
    is_new_role: bool = False
    new_role_display_name: str | None = None
    new_role_slug: str | None = None


class CandidateSkillAttached(BaseModel):
    skill_name: str
    is_primary: bool = False
    queue_id: int
    role_display_name: str
    role_slug: str
    status: str = "pending"


class CandidateKraAttached(BaseModel):
    queue_id: int
    role_display_name: str
    role_slug: str
    best_kra_similarity: float | None = None
    r_and_r_preview: str = ""
    status: str = "pending"


class Stage5UpdatesResponse(BaseModel):
    centroid_updated: bool = False
    centroid_n_after: int | None = None
    queue_entry_id: int | None = None
    collision_log_id: int | None = None
    new_skills_attached: list[CandidateSkillAttached] = Field(default_factory=list)
    new_kra_attached: CandidateKraAttached | None = None
    v3_pipeline_triggered: bool = False
    v3_run_id: str | None = None
    v3_role_slug: str | None = None


class JDSkillPipelineResponse(BaseModel):
    final_skills: list[FinalSkillItem] = Field(default_factory=list)
    run_id: str | None = None
    jd_role: JdRoleHint | None = None
    nano_parsed: dict | None = None
    stage3_signals: Stage3SignalsResponse | None = None
    stage4_decision: Stage4DecisionResponse | None = None
    stage5_updates: Stage5UpdatesResponse | None = None
    rejected: bool = False
    rejection_reason: str | None = None


# Archetypes that count as "tech" for our software-engineering-focused
# catalog. Override via TECH_ARCHETYPES env var (comma-separated).
_DEFAULT_TECH_ARCHETYPES = "Engineering,Data,DevOps,Security,QA,Research"


def _is_tech_jd(nano_parsed: dict | None) -> tuple[bool, str]:
    """Return ``(is_tech, reason_when_not_tech)``.

    Reject when:
      * Stage 1 marked JD_type=fail (parser couldn't make sense of it).
      * role_archetype is not in the allow-list (e.g., "Other", "Management",
        "Product", "Design" — none of these align with our SE-focused catalog).
    """
    if not nano_parsed:
        return False, "Stage 1 returned no nano_parsed payload"
    if nano_parsed.get("JD_type") == "fail":
        return False, "Stage 1 marked JD_type=fail (unparseable)"
    arch = (nano_parsed.get("role_archetype") or "").strip()
    allowed_raw = os.getenv("TECH_ARCHETYPES", _DEFAULT_TECH_ARCHETYPES)
    allowed = {a.strip() for a in allowed_raw.split(",") if a.strip()}
    if not arch:
        return False, "Stage 1 produced no role_archetype"
    if arch not in allowed:
        return False, (
            f"role_archetype {arch!r} is not in TECH_ARCHETYPES "
            f"(allowed: {sorted(allowed)})"
        )
    return True, ""


# ── Skill Detail / Reverse Planner models ────────────────────────────────────
class ExtractDetailsRequest(BaseModel):
    final_skills: list[str] = Field(default_factory=list)
    llm_skills: list[str] = Field(default_factory=list)
    primary_skills: list[str] = Field(default_factory=list)
    jd_role_hint: JdRoleHint | None = None
    run_id: str | None = None


class CanonicalSkillSummary(BaseModel):
    id: int
    slug: str | None = None
    display_name: str | None = None
    category_id: int | None = None
    sub_category_id: int | None = None
    skill_nature: str | None = None
    volatility: str | None = None
    is_extractable: bool | None = None
    is_also_category: bool | None = None
    typical_lifespan: str | None = None


class AliasMatch(BaseModel):
    input_term: str
    matched_via: str  # "alias" | "display_name"
    matched_canonical: CanonicalSkillSummary
    existing_alias_id: int | None = None
    existing_alias_text: str | None = None
    alias_persisted: bool
    alias_persist_skipped_reason: str | None = None


class RoleSummary(BaseModel):
    source: str  # "db" | "llm"
    id: int | None = None
    slug: str
    display_name: str
    role_archetype: str | None = None
    rationale: str | None = None


class DimensionSummary(BaseModel):
    source: str  # "db" | "llm"
    id: int | None = None
    slug: str
    display_name: str
    rationale: str | None = None
    difficulty_hint: str | None = None


class DimensionDetail(BaseModel):
    input_skill: str
    dimension: DimensionSummary
    roles_from_db: list[RoleSummary] = Field(default_factory=list)
    llm_role: RoleSummary | None = None


class AliasInfo(BaseModel):
    id: int
    alias_text: str
    alias_type: str | None = None
    match_strategy: str | None = None
    is_primary: bool | None = None


class SkillDetail(BaseModel):
    """One entry per skill in final_skills.

    `source_tag` tells you where the dimension/role info came from:
      - "db"           — canonical match in the library; aliases + dimensions
                         + roles all or partially come from the DB.
      - "llm"          — an llm_skills term not in the library; dimensions and
                         roles were inferred.
    """

    input_skill: str
    source_tag: str
    was_in_llm_skills: bool
    canonical: CanonicalSkillSummary | None = None
    matched_via: str | None = None  # "alias" | "display_name" | "embedding_alias" | "embedding_display_name"
    aliases_in_db: list[AliasInfo] = Field(default_factory=list)
    new_alias_persisted: bool = False
    new_alias_text: str | None = None
    dimensions: list[DimensionDetail] = Field(default_factory=list)
    new_skill_meta: NewSkillMetaV3 | None = None


class ChosenRole(BaseModel):
    source: str  # "db" | "llm" | "single_candidate"
    id: int | None = None
    slug: str
    display_name: str
    role_archetype: str | None = None
    rationale: str | None = None


class ExtractDetailsResponse(BaseModel):
    input_final_skills: list[str]
    input_llm_skills: list[str]
    alias_matches: list[AliasMatch]
    new_aliases_persisted: int
    unmatched_skills: list[str]
    dimensions: list[DimensionDetail]
    skills_detail: list[SkillDetail]
    candidate_roles: list[RoleSummary]
    chosen_role: ChosenRole | None = None
    run_id: str | None = None


class FinalInputSkillTag(BaseModel):
    skill: str
    tag: str  # "in_db" | "new"


class PersistenceItem(BaseModel):
    input_skill: str
    skill_tag: str  # "in_db" | "new"
    skill_id: int | None = None
    dimension: DimensionSummary
    roles_from_db: list[RoleSummary] = Field(default_factory=list)
    llm_role: RoleSummary | None = None
    dimension_id: int | None = None
    chosen_role_id: int | None = None
    matched_chosen_role: bool
    skill_dimension_saved: bool = False
    role_dimension_saved: bool = False
    skipped_reason: str | None = None
    outcome_line: str | None = Field(
        default=None,
        description="Human-readable outcome for the Stage 03 UI (skill / dim / role-link).",
    )


class PersistenceReport(BaseModel):
    skill_dimension_saved: int = 0
    role_dimension_saved: int = 0
    new_skills_created: int = 0
    skipped: int = 0
    items: list[PersistenceItem] = Field(default_factory=list)


class PlannerGeneratedOutput(BaseModel):
    generated: bool = False
    role_id: str | None = None
    role_display: str | None = None
    payload: dict[str, Any] | None = None
    saved_role_dimensions: int = 0
    saved_dimensions_created: int = 0


class FinalRoleOutputRequest(ExtractDetailsResponse):
    """Accept full output payload from /skills/extract-details."""


class FinalRoleOutputResponse(BaseModel):
    chosen_role: ChosenRole | None = None
    final_input_skills: list[FinalInputSkillTag] = Field(default_factory=list)
    persistence: PersistenceReport
    planner_output: PlannerGeneratedOutput | None = None
    run_id: str | None = None
    # ``in_db`` — chosen role resolved to an existing DB row (roles or role_aliases).
    # ``auto_created`` — role was not in DB; inserted after Stage 4 confirmed it.
    # ``human_review_required`` — role absent; skipped insert (writes disabled).
    chosen_role_resolution: str | None = None
    llm_cost_api1_usd: float | None = None
    llm_cost_api2_usd: float | None = None
    llm_cost_api3_usd: float | None = None
    llm_cost_total_usd: float | None = None


# ── JD Pipeline History models ───────────────────────────────────────────────
class JdRunSummary(BaseModel):
    """Compact row for the history list page."""

    id: str
    created_at: datetime
    updated_at: datetime
    status: str
    chosen_role_display: str | None = None
    chosen_role_id: int | None = None
    final_skills_count: int | None = None
    final_skills: list[str] | None = None
    jd_role_hint_display: str | None = None
    duration_ms: int | None = None
    error_message: str | None = None
    jd_text_preview: str | None = None
    jd_text_length: int | None = None
    stage4_case: str | None = None
    v3_pipeline_triggered: bool = False
    v3_run_id: str | None = None
    v3_role_slug: str | None = None
    llm_cost_api1_usd: float | None = None
    llm_cost_api2_usd: float | None = None
    llm_cost_api3_usd: float | None = None
    llm_cost_total_usd: float | None = None


class JdRunListResponse(BaseModel):
    runs: list[JdRunSummary]
    limit: int
    offset: int


class V3StageRow(BaseModel):
    stage_index: int           # 0..8
    stage_name: str            # 'charter', 'anchor', 'dim_gen', ...
    prompt_version: str | None = None
    status: str                # 'pending' | 'running' | 'approved' | 'failed'
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None


class V3RunStatus(BaseModel):
    role_slug: str
    role_display: str | None = None
    stages: list[V3StageRow]   # always 9 rows (one per stage)
    terminal_status: str       # 'pending' | 'running' | 'approved' | 'failed'
    last_updated: datetime | None = None


class JdRunArtifact(BaseModel):
    id: int
    artifact_kind: str
    artifact_id: int | None = None
    artifact_text: str | None = None
    created_at: datetime


class JdRunDetail(BaseModel):
    id: str
    jd_text: str
    status: str
    api_parser_response: dict[str, Any] | None = None
    api1_response: dict[str, Any] | None = None
    api2_response: dict[str, Any] | None = None
    api3_response: dict[str, Any] | None = None
    chosen_role_display: str | None = None
    chosen_role_id: int | None = None
    final_skills_count: int | None = None
    final_skills: list[str] | None = None
    jd_role_hint_display: str | None = None
    error_message: str | None = None
    duration_ms: int | None = None
    created_at: datetime
    updated_at: datetime
    llm_cost_api1_usd: float | None = None
    llm_cost_api2_usd: float | None = None
    llm_cost_api3_usd: float | None = None
    llm_cost_total_usd: float | None = None
    artifacts: list[JdRunArtifact] = Field(default_factory=list)
    #: Denormalized view for history UIs (JD → skills → dims → API3 links).
    history_view: dict[str, Any] | None = None


class PdfLinkItem(BaseModel):
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


class PdfLinksResponse(BaseModel):
    links: list[PdfLinkItem]


def _clean_strings(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for v in values or []:
        s = (str(v) if v is not None else "").strip()
        if s:
            out.append(s)
    return out


def _dedupe_case_insensitive(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _extract_jd_sentence(jd_text: str, word: str, max_chars: int = 180) -> str:
    """Return the first JD sentence containing word (case-insensitive, up to max_chars)."""
    import re as _re
    for sent in _re.split(r"[.!\?\n;]+", jd_text):
        if _re.search(r"\b" + _re.escape(word) + r"\b", sent, _re.IGNORECASE):
            return sent.strip()[:max_chars]
    # fallback: find any occurrence ignoring word boundaries (handles C++, .NET etc.)
    for sent in _re.split(r"[.!\?\n;]+", jd_text):
        if word.lower() in sent.lower():
            return sent.strip()[:max_chars]
    return ""


def _embed_for_skill_library_db(text: str) -> list[float] | None:
    """1536-dim vectors for ``skill-library`` pgvector columns (Azure OpenAI).

    Must match ``skill_matcher`` / existing ``canonical_skills.name_embedding``.
    Returns None when the client is unavailable so inserts can omit embedding.
    """
    from skill_matcher import _azure_embed_sync

    trimmed = (text or "").strip()
    if not trimmed:
        return None
    vecs = _azure_embed_sync([trimmed])
    if not vecs or not vecs[0]:
        return None
    return list(vecs[0])


# ── API 3: reconciliation-aware dimension + new-skill helpers ───────────────

_PIPELINE_DIM_PREFIXES = ("d_init_", "d_merge_", "d_split_")


def _is_pipeline_generated_dim_slug(slug: str | None) -> bool:
    if not slug or not isinstance(slug, str):
        return False
    s = slug.strip().lower()
    return any(s.startswith(p) for p in _PIPELINE_DIM_PREFIXES)


def _catalog_slug_from_merge_log(
    merge_log: list[dict[str, Any]] | None,
) -> dict[str, str]:
    """Map tentative / merge ``into`` ids (lower) -> catalog dimension slug.

    When Stage 3 MERGE collapses a pipeline dim with an existing library
    dimension, callers should persist links against the catalog row, not a
    new ``d_merge_*`` row.
    """
    out: dict[str, str] = {}
    for entry in merge_log or []:
        merged_from = list(entry.get("merged_from") or [])
        into = entry.get("into")
        catalog_slugs: list[str] = []
        for key in ("a_dim_id", "b_dim_id"):
            val = entry.get(key)
            if val is None:
                continue
            s = str(val).strip()
            if s and not _is_pipeline_generated_dim_slug(s):
                catalog_slugs.append(s)
        if not catalog_slugs:
            continue
        catalog = catalog_slugs[0]
        for tid in merged_from:
            if tid is None:
                continue
            ts = str(tid).strip()
            if ts:
                out[ts.lower()] = catalog
        if into:
            out[str(into).strip().lower()] = catalog
    return out


def _resolve_dimension_id_from_reconciliation_sync(
    repo: SkillLibraryRepository,
    *,
    dimension_slug: str | None,
    merge_slug_map: dict[str, str],
) -> int | None:
    """Resolve ``dimensions.id`` using merge reconciliation when applicable."""
    slug = (dimension_slug or "").strip()
    if not slug:
        return None
    slug_lc = slug.lower()
    if not _is_pipeline_generated_dim_slug(slug):
        found = repo.find_dimensions_by_names(slugs=[slug])
        rec = found.get(slug_lc)
        if rec is None and found:
            rec = next(iter(found.values()))
        if rec and rec.get("id") is not None:
            return int(rec["id"])
        return None
    catalog_slug = merge_slug_map.get(slug_lc)
    if not catalog_slug:
        return None
    found = repo.find_dimensions_by_names(slugs=[catalog_slug])
    cl = catalog_slug.strip().lower()
    rec = found.get(cl)
    if rec is None and found:
        rec = next(iter(found.values()))
    if rec and rec.get("id") is not None:
        return int(rec["id"])
    return None


def _has_v3_meta_for_new_skill_persist(meta: NewSkillMetaV3 | None) -> bool:
    if meta is None:
        return False
    if meta.locked_dimensions:
        return True
    for m in meta.merge_log or []:
        for key in ("a_dim_id", "b_dim_id"):
            val = m.get(key)
            if val and not _is_pipeline_generated_dim_slug(str(val)):
                return True
    return False


def _append_canonical_skill_proposed_artifact(
    artifacts: list[dict[str, Any]],
    skill_name: str,
    meta: NewSkillMetaV3,
) -> None:
    derived = meta.derived
    typed = meta.typed
    if derived is not None:
        artifact_text = (
            f"{skill_name} | type={derived.category} subtype={derived.sub_category} "
            f"nature={derived.skill_nature} lifespan={derived.typical_lifespan}"
        )
    elif typed is not None:
        artifact_text = f"{skill_name} | type={typed.type} subtype={typed.subtype}"
    else:
        artifact_text = skill_name
    if len(artifact_text) > 500:
        artifact_text = artifact_text[:497] + "..."
    artifacts.append({
        "kind": "canonical_skill_proposed",
        "artifact_text": artifact_text,
    })


def _apply_new_skill_v3_enrichment_to_db(
    repo: SkillLibraryRepository,
    *,
    skill_id: int,
    display_name: str,
    meta: NewSkillMetaV3,
    typical_lifespan: str,
) -> bool:
    """Persist Stage 7 enrichment + Stage 4 typing like ``catalog_transform`` + Stage 8 loader.

    Returns True if ``canonical_skills`` enrichment columns were updated successfully.
    """
    enr = meta.enrichment
    derived = meta.derived
    typed = meta.typed

    if enr and enr.maturity:
        volatility = map_maturity_to_volatility(enr.maturity.maturity)
    elif derived is not None:
        volatility = str(derived.volatility or "STABLE")
    else:
        volatility = "STABLE"

    vendor: str | None = None
    license_value: str | None = None
    year_introduced: int | None = None
    maturity_reasoning: str | None = None
    version_strategy = "NOT_APPLICABLE"
    version_tag: str | None = None

    if enr is not None:
        vl = enr.vendor_license
        vendor = (str(vl.vendor).strip() if vl.vendor else None) or None
        if vl.license is not None:
            license_value = str(getattr(vl.license, "value", vl.license)).strip().lower()
        year_introduced = vl.year_introduced
        if enr.maturity:
            maturity_reasoning = enr.maturity.reasoning
        ver = enr.versioning
        version_strategy = map_versioned_to_strategy(bool(ver.versioned))
        version_tag = (
            str(ver.current_version).strip() if ver.current_version else None
        )

    confidence = float(typed.confidence) if typed is not None else 0.7
    confidence = max(0.0, min(1.0, confidence))

    tl = typical_lifespan
    if derived is not None and getattr(derived, "typical_lifespan", None):
        tl = str(derived.typical_lifespan)

    enrichment_row_ok = False
    try:
        repo.update_canonical_skill_enrichment_v3(
            skill_id,
            volatility=volatility,
            vendor=vendor,
            license_value=license_value,
            year_introduced=year_introduced,
            maturity_reasoning=maturity_reasoning,
            version_strategy=version_strategy,
            version_tag=version_tag,
            confidence=confidence,
            typical_lifespan=tl,
        )
        enrichment_row_ok = True
    except Exception as exc:
        logger.warning(
            "update_canonical_skill_enrichment_v3 failed for skill_id=%s: %s",
            skill_id,
            exc,
        )

    if enr is not None and enr.versioning and enr.versioning.version_aliases:
        disp_lc = display_name.strip().lower()
        va_keys: list[str] = []
        for k in enr.versioning.version_aliases.keys():
            kk = str(k).strip()
            if not kk or kk.lower() == disp_lc:
                continue
            va_keys.append(kk)
        if va_keys:
            try:
                repo.add_aliases(
                    [(skill_id, k) for k in va_keys],
                    alias_type="VERSION",
                    match_strategy="CASE_INSENSITIVE",
                )
            except Exception as exc:
                logger.warning(
                    "VERSION aliases insert failed skill_id=%s: %s",
                    skill_id,
                    exc,
                )

    if enr is not None and enr.context_keywords and enr.context_keywords.context_keywords:
        try:
            repo.add_skill_tags(skill_id, list(enr.context_keywords.context_keywords))
        except Exception as exc:
            logger.warning("skill_tags insert failed skill_id=%s: %s", skill_id, exc)

    return enrichment_row_ok


def _typology_type_from_skill_nature(skill_nature: str | None) -> SkillType:
    from skill_library_v3.catalog_transform import map_type_to_skill_nature
    from skill_library_v3.schemas.typology import TYPOLOGY_VALUES

    target = (skill_nature or "").strip().upper()
    for label in TYPOLOGY_VALUES:
        if map_type_to_skill_nature(label) == target:
            return cast(SkillType, label)
    return "Concept"


def _primary_dimension_slug_for_library_skill(sd: SkillDetail) -> str:
    for dd in sd.dimensions or []:
        dim = dd.dimension
        if dim is None:
            continue
        slug = (dim.slug or "").strip()
        if slug:
            return slug
    return "general"


async def _backfill_existing_skill_catalog_enrichment(
    repo: SkillLibraryRepository,
    sd: SkillDetail,
    cost_acc: CostAccumulator,
) -> bool:
    """Stage 7 + v3 persist when the skill has no catalog context tags
    (``skill_tags``). Thin wrapper around
    ``new_skill_orchestrator.enrich_one_existing_canonical_skill`` — the
    actual enrichment + DB-write logic lives there so the nightly backfill
    cron can call it directly without importing main.py."""
    if sd.canonical is None or sd.canonical.id is None:
        return False
    return await enrich_one_existing_canonical_skill(
        repo,
        skill_id=int(sd.canonical.id),
        cost_acc=cost_acc,
    )


def _try_create_new_canonical_skill_sync(
    repo: SkillLibraryRepository,
    sd: SkillDetail,
) -> tuple[int | None, str | None, bool]:
    """Insert canonical skill + primary alias for one v3-shaped LLM skill.

    Third tuple element is True when enrichment columns were updated on ``canonical_skills``.
    """
    meta = sd.new_skill_meta
    if meta is None or not _has_v3_meta_for_new_skill_persist(meta):
        return None, "no_persistable_v3_meta", False
    display_name = sd.input_skill.strip()
    if not display_name:
        return None, "empty_display_name", False
    try:
        derived = meta.derived
        typed = meta.typed
        if derived is not None:
            cat_name = (derived.category or "General").strip() or "General"
            sub_token = (derived.sub_category or "general").strip() or "general"
            sub_name = sub_token.replace("_", " ").strip().title() or "General"
            skill_nature = str(derived.skill_nature or "TOOL")
            typical_lifespan = str(derived.typical_lifespan or "EVERGREEN")
        elif typed is not None:
            cat_name = (typed.type or "General").strip() or "General"
            sub_token = (typed.subtype or "general").strip() or "general"
            sub_name = sub_token.replace("_", " ").strip().title() or "General"
            skill_nature = "TOOL"
            typical_lifespan = "EVERGREEN"
        else:
            cat_name = "General"
            sub_name = "General"
            skill_nature = "TOOL"
            typical_lifespan = "EVERGREEN"
        if len(sub_name) > 120:
            sub_name = sub_name[:120]

        cat = repo.find_or_create_category(display_name=cat_name)
        sub = repo.find_or_create_sub_category(
            category_id=int(cat["id"]),
            display_name=sub_name,
        )
        emb = _embed_for_skill_library_db(display_name)
        row = repo.create_canonical_skill(
            display_name=display_name,
            category_id=int(cat["id"]),
            sub_category_id=int(sub["id"]),
            skill_nature=skill_nature,
            typical_lifespan=typical_lifespan,
            source="llm",
            name_embedding=emb,
        )
        sid = int(row["id"])
        repo.add_aliases(
            [(sid, display_name)],
            alias_type="CANONICAL",
            match_strategy="CASE_INSENSITIVE",
        )
        enrichment_ok = _apply_new_skill_v3_enrichment_to_db(
            repo,
            skill_id=sid,
            display_name=display_name,
            meta=meta,
            typical_lifespan=typical_lifespan,
        )
        return sid, None, enrichment_ok
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", False


def _extra_dimension_rows_from_skills_detail(
    skills_detail: list[SkillDetail],
) -> list[DimensionDetail]:
    """Add locked / merge-only dimension rows missing from ``skills_detail.dimensions``."""
    out: list[DimensionDetail] = []
    keys: set[tuple[str, str]] = set()
    for sd in skills_detail:
        if sd.source_tag != "llm" or not sd.new_skill_meta:
            continue
        meta = sd.new_skill_meta
        existing_slugs = {(d.dimension.slug or "").strip().lower() for d in sd.dimensions}
        for ld in meta.locked_dimensions:
            tid = (ld.tentative_id or "").strip()
            if not tid:
                continue
            tlc = tid.lower()
            if tlc in existing_slugs:
                continue
            key = (sd.input_skill.strip().lower(), tlc)
            if key in keys:
                continue
            keys.add(key)
            mat = meta.enrichment.maturity if meta.enrichment else None
            diff = mat.maturity if mat else None
            out.append(
                DimensionDetail(
                    input_skill=sd.input_skill,
                    dimension=DimensionSummary(
                        id=None,
                        slug=tid,
                        display_name=ld.name,
                        rationale=ld.description,
                        difficulty_hint=diff,
                        source="llm",
                    ),
                    roles_from_db=[],
                    llm_role=None,
                )
            )
        if not meta.locked_dimensions and meta.merge_log:
            seen_cat: set[str] = set()
            for entry in meta.merge_log:
                for key in ("a_dim_id", "b_dim_id"):
                    val = entry.get(key)
                    if not val or _is_pipeline_generated_dim_slug(str(val)):
                        continue
                    sl = str(val).strip()
                    slc = sl.lower()
                    if slc in seen_cat or slc in existing_slugs:
                        continue
                    seen_cat.add(slc)
                    k2 = (sd.input_skill.strip().lower(), slc)
                    if k2 in keys:
                        continue
                    keys.add(k2)
                    disp = (
                        entry.get("b_name")
                        or entry.get("into_name")
                        or sl.replace("-", " ").title()
                    )
                    out.append(
                        DimensionDetail(
                            input_skill=sd.input_skill,
                            dimension=DimensionSummary(
                                id=None,
                                slug=sl,
                                display_name=disp,
                                rationale=entry.get("reasoning"),
                                difficulty_hint=None,
                                source="db",
                            ),
                            roles_from_db=[],
                            llm_role=None,
                        )
                    )
    return out


def _merge_dimension_worklist(req: FinalRoleOutputRequest) -> list[DimensionDetail]:
    """``req.dimensions`` plus locked / merge-only rows, de-duplicated by (skill, slug)."""
    seen: set[tuple[str, str]] = set()
    out: list[DimensionDetail] = []
    for dd in req.dimensions:
        sk = dd.input_skill.strip().lower()
        sl = (dd.dimension.slug or "").strip().lower()
        if sl:
            key = (sk, sl)
            if key in seen:
                continue
            seen.add(key)
        out.append(dd)
    for dd in _extra_dimension_rows_from_skills_detail(req.skills_detail):
        sk = dd.input_skill.strip().lower()
        sl = (dd.dimension.slug or "").strip().lower()
        if not sl:
            continue
        key = (sk, sl)
        if key in seen:
            continue
        seen.add(key)
        out.append(dd)
    return out


def _norm_token(v: str | None) -> str:
    return (v or "").strip().lower().replace("_", "-")


def _skill_db_tag(sd: SkillDetail) -> str:
    return "in_db" if sd.canonical is not None else "new"


def _role_matches_chosen(dd: DimensionDetail, chosen: ChosenRole | None) -> bool:
    if chosen is None:
        return False

    chosen_slug = _norm_token(chosen.slug)
    chosen_name = _norm_token(chosen.display_name)

    if dd.llm_role is not None:
        llm_slug = _norm_token(dd.llm_role.slug)
        llm_name = _norm_token(dd.llm_role.display_name)
        if (chosen_slug and llm_slug == chosen_slug) or (
            chosen_name and llm_name == chosen_name
        ):
            return True

    for r in dd.roles_from_db:
        db_slug = _norm_token(r.slug)
        db_name = _norm_token(r.display_name)
        if (chosen_slug and db_slug == chosen_slug) or (
            chosen_name and db_name == chosen_name
        ):
            return True
    return False


def _should_create_missing_role(
    chosen: ChosenRole,
    *,
    matched_rows: int,
    total_rows: int,
) -> bool:
    """Conditional create gate for chosen role missing in DB."""
    if not (chosen.display_name or "").strip() or not (chosen.slug or "").strip():
        return False
    if not (chosen.role_archetype or "").strip():
        return False
    if total_rows <= 0:
        return False
    # Require at least 2 matching rows or 30% coverage.
    return matched_rows >= 2 or (matched_rows / total_rows) >= 0.30


# ── Existing: Skill normalization endpoint ──
@app.post("/normalize-skills", response_model=SkillsResponse)
async def normalize_skills_endpoint(req: SkillsRequest):
    results = await normalize_skills(req.skills)
    return SkillsResponse(
        normalized_skills=[r.normalized for r in results],
        details=[
            NormalizedSkillItem(
                original=r.original,
                normalized=r.normalized,
                method=r.method,
                confidence=r.confidence,
            )
            for r in results
        ],
    )


def _extract_text(field) -> str:
    """Safely extract .text from a nano field that may be a dict, list, or None."""
    if isinstance(field, dict):
        return (field.get("text") or "").strip()
    if isinstance(field, list):
        return "\n\n".join(
            (item.get("text") or "").strip()
            for item in field
            if isinstance(item, dict) and item.get("text")
        ).strip()
    return ""


async def _llm_parse_jd_nano(
    jd_text: str,
    cost_acc: "CostAccumulator | None" = None,
) -> dict:
    """Call fast tier (FAST_MODEL, default gpt-4o-mini) with job_parser.txt to extract structured JD fields."""
    from llm_client import FAST_MODEL, get_fast_client

    client = get_fast_client()
    if client is None:
        raise RuntimeError("LLM client unavailable — check AZURE_OPEN_AI_KEY")

    system_prompt = (Path(__file__).parent / "job_parser.txt").read_text(encoding="utf-8")
    resp = await client.chat.completions.create(
        model=FAST_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse this job description:\n\n{jd_text}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        timeout=90,
    )
    if cost_acc is not None and resp.usage is not None:
        cost_acc.add(FAST_MODEL, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


async def _llm_extract_skills_with_tags(
    text: str,
    cost_acc: "CostAccumulator | None" = None,
) -> list[FinalSkillItem]:
    """Extract skills with primary/secondary tags from the given text (R&R + tech requirements).

    Role extraction is handled upstream by the nano parser — this function returns skills only.
    """
    from llm_client import get_fast_client, SKILL_MODEL

    client = get_fast_client()
    if client is None:
        raise RuntimeError("LLM client unavailable — check AZURE_OPEN_AI_KEY")

    system_prompt = (
        "You are a precise technical skill extractor for job descriptions.\n"
        "Extract only SPECIFIC, NAMED skills that belong to one of these categories:\n\n"
        "  LANGUAGE      — e.g. Python, Java, TypeScript, JavaScript, SQL, Apex\n"
        "  FRAMEWORK     — e.g. React, Spring Boot, Django, Next.js, Lightning (Salesforce)\n"
        "  LIBRARY       — e.g. Redux, Pandas, NumPy, Hibernate, Lodash\n"
        "  TOOL          — e.g. Git, Docker, Jenkins, Postman, Webpack, Salesforce CLI\n"
        "  PLATFORM      — e.g. Salesforce, Kubernetes, AWS, Azure, GCP, Hadoop, Spark\n"
        "  CLOUD_SERVICE — e.g. AWS Lambda, S3, RDS, Azure Blob, GCP BigQuery\n"
        "  DATABASE      — e.g. MySQL, PostgreSQL, MongoDB, Redis, Cassandra\n"
        "  METHODOLOGY   — e.g. Agile, Scrum, TDD, CI/CD, DevOps\n"
        "  PROTOCOL      — e.g. REST, GraphQL, gRPC, WebSocket, OAuth2, SOQL, SOSL\n"
        "  STANDARD      — e.g. OpenAPI, OWASP, WCAG, ISO 27001\n"
        "  PATTERN       — e.g. Microservices, Event-Driven Architecture, CQRS, MVC\n"
        "  PRACTICE      — e.g. Code Review, Pair Programming, A/B Testing\n"
        "  CONCEPT       — e.g. Machine Learning, Distributed Systems (only when no specific tool fits)\n\n"
        "STRICT REJECT RULES — never include these:\n"
        "  1. Certifications: anything like 'Salesforce Certified X', 'AWS Certified X', 'PMP', etc.\n"
        "  2. Vague soft skills or activities: Troubleshooting, Debugging, Architecture, Software Development,\n"
        "     Change Management, Compliance, Governance, Vendor Management, Communication, Leadership\n"
        "  3. Broad category words: 'databases', 'cloud platforms', 'server-side development',\n"
        "     'back-end web development', 'Metadata repository', 'Deployments'\n"
        "  4. Expanded names when the acronym IS the skill: use 'SOQL' not 'Salesforce Object Query Language',\n"
        "     use 'SOSL' not 'Salesforce Object Search Language'\n\n"
        "BAD examples (reject these):\n"
        "  'Salesforce Object Query Language', 'Salesforce Certified Platform Developer I',\n"
        "  'Troubleshooting', 'Debugging', 'Architecture', 'Software Development', 'Metadata repository'\n\n"
        "GOOD examples (extract these):\n"
        "  'Salesforce', 'Apex', 'Lightning', 'SOQL', 'SOSL', 'JavaScript', 'Git', 'Agile'\n\n"
        "Additional rules:\n"
        "  - Use the canonical short name (e.g. 'Node.js', 'PostgreSQL', 'AWS')\n"
        "  - Each skill is a separate entry — do not group\n"
        "  - is_primary=true for core/required, false for nice-to-have\n\n"
        "Return valid JSON only:\n"
        "{\n"
        "  \"skills\": [{\"skill_name\": \"...\", \"is_primary\": true/false}, ...]\n"
        "}"
    )
    user_message = f"Extract all specific technical skills from this job description:\n\n{text}"
    logger.info(
        "[skills/classifier] model=%s user_message_chars=%d (raw skill slice chars=%d)",
        SKILL_MODEL,
        len(user_message),
        len(text),
    )
    logger.info(
        "[skills/classifier] USER_MESSAGE_BEGIN\n%s\n[skills/classifier] USER_MESSAGE_END",
        user_message,
    )
    resp = await client.chat.completions.create(
        model=SKILL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        timeout=120,
    )
    if cost_acc is not None and resp.usage is not None:
        cost_acc.add(SKILL_MODEL, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    raw_str = resp.choices[0].message.content or "{}"
    logger.info(
        "[skills/classifier] RAW_MESSAGE_CONTENT_BEGIN\n%s\n[skills/classifier] RAW_MESSAGE_CONTENT_END",
        raw_str,
    )
    raw = json.loads(raw_str)
    items = []
    for s in raw.get("skills", []):
        if isinstance(s, dict) and (s.get("skill_name") or "").strip():
            items.append(FinalSkillItem(
                skill_name=str(s["skill_name"]).strip(),
                is_primary=bool(s.get("is_primary", False)),
            ))
    logger.info(
        "[skills/classifier] parsed_skills count=%d names=%s",
        len(items),
        [i.skill_name for i in items],
    )
    return items


@app.post("/skills/extract-from-jd", response_model=JDSkillPipelineResponse)
async def extract_skills_from_jd_endpoint(req: JDSkillPipelineRequest):
    jd_text = (req.jd_text or "").strip()
    if not jd_text:
        raise HTTPException(status_code=400, detail="jd_text cannot be empty")

    cost_acc = CostAccumulator()
    try:
        # Stage 1 — FAST_MODEL parses JD structure (job_parser.txt): company, role, R&R, etc.
        nano_parsed = await _llm_parse_jd_nano(jd_text, cost_acc=cost_acc)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Nano JD parsing failed: {exc}",
        ) from exc

    # ── Tech-archetype gate — early-exit non-tech JDs ─────────────────────────
    # The skill catalog is software-engineering-focused. Reject JDs whose
    # archetype doesn't fit (Lab Technician, Product Manager, Project Admin,
    # Synthetic Biology Scientist, etc.) so we don't waste LLM tokens on
    # Stage 2/3/4/5 or fire v3 to create catalog entries for off-domain roles.
    is_tech, reject_reason = _is_tech_jd(nano_parsed)
    if not is_tech:
        nano_role_for_log = (nano_parsed or {}).get("role") or "—"
        arch_for_log = (nano_parsed or {}).get("role_archetype") or "—"
        logger.info(
            "[skills/extract-from-jd] REJECT non-tech JD: role=%r archetype=%r reason=%s",
            nano_role_for_log, arch_for_log, reject_reason,
        )
        rejected_response = JDSkillPipelineResponse(
            final_skills=[],
            jd_role=JdRoleHint(
                display_name=nano_role_for_log,
                slug="",
                role_archetype=arch_for_log,
                rationale=reject_reason,
            ) if nano_parsed else None,
            nano_parsed=nano_parsed,
            rejected=True,
            rejection_reason=reject_reason,
        )
        # Persist a thin record so the history list shows the rejection.
        try:
            history_repo = JdPipelineRunRepository()
            rejected_run_id = await asyncio.to_thread(
                history_repo.start_run,
                jd_text=jd_text,
                api1_response=rejected_response,
                api_parser_response=nano_parsed,
                jd_role_hint_display=nano_role_for_log,
                llm_cost_usd=cost_acc.total_cost_usd,
            )
            rejected_response.run_id = rejected_run_id
        except Exception as exc:
            logger.warning("[skills/extract-from-jd] reject-run start_run failed: %s", exc)
        return rejected_response

    # Extract role from nano output
    jd_role: JdRoleHint | None = None
    nano_role = (nano_parsed.get("role") or "").strip()
    nano_role_aliases: list[str] = [
        a.strip() for a in (nano_parsed.get("role_aliases") or [])
        if isinstance(a, str) and a.strip()
    ]
    if nano_role:
        jd_role = JdRoleHint(
            display_name=nano_role,
            slug="",
            role_archetype=(nano_parsed.get("role_archetype") or "").strip() or None,
            role_aliases=nano_role_aliases,
        )
    logger.info(
        "[skills/extract-from-jd] role=%r role_aliases=%s",
        nano_role or "(none)",
        nano_role_aliases or [],
    )

    # Build skill extractor input: role + technical_requirements + roles_and_responsibilities
    _parts: list[str] = []
    if nano_role:
        _parts.append(f"Role: {nano_role}")
    tech_req_text = _extract_text(nano_parsed.get("technical_requirements"))
    if tech_req_text:
        _parts.append(tech_req_text)
    r_and_r_text = _extract_text(nano_parsed.get("roles_and_responsibilities"))
    if r_and_r_text:
        _parts.append(r_and_r_text)
    skill_input = "\n\n".join(_parts) if _parts else jd_text
    skill_input_source = "structured_composite" if _parts else "full_jd_fallback"

    logger.info(
        "[skills/extract-from-jd] skill_input_source=%s JD_type=%s has_role=%s tech_req_chars=%d r_and_r_chars=%d skill_input_chars=%d jd_chars=%d",
        skill_input_source,
        nano_parsed.get("JD_type"),
        bool(nano_role),
        len(tech_req_text),
        len(r_and_r_text),
        len(skill_input),
        len(jd_text),
    )
    logger.info(
        "[skills/extract-from-jd] skill_input → chars=%d  est_tokens=%d",
        len(skill_input),
        len(skill_input) // 4,
    )

    try:
        # Stage 2 — skill extractor runs on structured text only (not full JD)
        final_skills = await _llm_extract_skills_with_tags(skill_input, cost_acc=cost_acc)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM skill extraction failed: {exc}",
        ) from exc

    # ── Stage 2 (R&R embedding) + Stage 3 (DB lookups) + Stage 3.5 (alias validator) ──
    stage3_signals: Stage3SignalsResponse | None = None
    s3 = None
    try:
        from jd_similarity_matcher import Stage3Result, run_stage2_and_stage3

        skills_for_stage3 = [s.skill_name for s in final_skills]
        s3: Stage3Result = await run_stage2_and_stage3(
            r_and_r_text,
            skills_for_stage3,
            nano_role,
            role_aliases=nano_role_aliases,
            top_k=5,
            cost_acc=cost_acc,
        )
        stage3_signals = Stage3SignalsResponse(
            skill_match_roles=[
                RoleSignalItem(
                    role_id=r.role_id,
                    slug=r.slug,
                    display_name=r.display_name,
                    score=r.score,
                    matched_count=r.matched_count,
                    total_count=r.total_count,
                )
                for r in s3.skill_match_roles
            ],
            alias_match_roles=[
                RoleSignalItem(role_id=r.role_id, slug=r.slug, display_name=r.display_name, score=r.score)
                for r in s3.alias_match_roles
            ],
            kra_match_roles=[
                RoleSignalItem(role_id=r.role_id, slug=r.slug, display_name=r.display_name, score=r.score)
                for r in s3.kra_match_roles
            ],
            alias_found=s3.alias_found,
        )
        logger.info(
            "[skills/extract-from-jd] stage3 skill_roles=%d alias_roles=%d kra_roles=%d alias_found=%s",
            len(s3.skill_match_roles),
            len(s3.alias_match_roles),
            len(s3.kra_match_roles),
            s3.alias_found,
        )
        for r in s3.skill_match_roles:
            logger.info(
                "[stage3a/skill_match] role=%r slug=%s score=%.2f%% (%d/%d skills)",
                r.display_name, r.slug, r.score * 100,
                r.matched_count or 0, r.total_count or 0,
            )
        for r in s3.alias_match_roles:
            logger.info(
                "[stage3b/alias_match] role=%r slug=%s score=%.2f%%",
                r.display_name, r.slug, r.score * 100,
            )
        for r in s3.kra_match_roles:
            logger.info(
                "[stage3c/kra_match]   role=%r slug=%s score=%.2f%%",
                r.display_name, r.slug, r.score * 100,
            )
    except Exception as exc:
        logger.warning("[skills/extract-from-jd] Stage 2/3 failed (non-fatal): %s", exc)
        s3 = None

    # ── Stage 1 role resolution (NEW) ─────────────────────────────────────────
    # Resolve the verbatim Stage 1 role title against the canonical catalog
    # BEFORE Stage 4 runs. The result is the authoritative "is this role in
    # the catalog?" signal — sidesteps Stage 3b's noisy trigram fuzzy fallback.
    stage1_resolved = None
    if s3 is not None and nano_role:
        try:
            from jd_classifier import resolve_stage1_role
            from jd_similarity_matcher import _SKILL_SCHEMA, _pg_connect

            def _embed_role_for_stage1(role_text: str) -> list[float] | None:
                from skill_matcher import _azure_embed_sync

                vecs = _azure_embed_sync([role_text], cost_acc=cost_acc)
                if not vecs or not vecs[0]:
                    return None
                return list(vecs[0])

            def _resolve_stage1():
                conn = _pg_connect()
                try:
                    with conn:
                        return resolve_stage1_role(
                            conn,
                            nano_role,
                            schema=_SKILL_SCHEMA,
                            embed_fn=_embed_role_for_stage1,
                        )
                finally:
                    conn.close()

            stage1_resolved = await asyncio.to_thread(_resolve_stage1)
            if stage1_resolved is not None:
                logger.info(
                    "[stage1/resolve] %r -> %s (id=%d, match_kind=%s)",
                    nano_role, stage1_resolved.slug,
                    stage1_resolved.role_id, stage1_resolved.match_kind,
                )
            else:
                logger.info(
                    "[stage1/resolve] %r is NOT in canonical catalog "
                    "(new-role candidate)",
                    nano_role,
                )
        except Exception as exc:
            logger.warning("[stage1/resolve] failed: %s", exc)

    # ── Stage 4 (Decision Routing) — compulsory ──────────────────────────────
    stage4_response: Stage4DecisionResponse | None = None
    decision = None
    if s3 is not None:
        try:
            from jd_classifier import llm2_resolve_role, stage4_route_decision

            async def _llm2_with_cost(
                *, r_and_r_text: str, candidates: list,
            ):
                return await llm2_resolve_role(
                    r_and_r_text, candidates, cost_acc=cost_acc,
                )

            decision = await stage4_route_decision(
                s3, r_and_r_text=r_and_r_text, role_name_input=nano_role,
                stage1_resolved=stage1_resolved,
                llm2_fn=_llm2_with_cost,
            )
            stage4_response = Stage4DecisionResponse(
                case=decision.case,
                chosen_role=(
                    RoleSignalItem(
                        role_id=decision.chosen_role.role_id,
                        slug=decision.chosen_role.slug,
                        display_name=decision.chosen_role.display_name,
                        score=decision.chosen_role.score,
                    ) if decision.chosen_role else None
                ),
                confidence=decision.confidence,
                llm2_fired=decision.llm2_fired,
                llm2_reasoning=decision.llm2_reasoning,
                alias_collision_detected=decision.alias_collision_detected,
                queued=decision.queued,
                reasoning=decision.reasoning,
                is_new_role=decision.is_new_role,
                new_role_display_name=decision.new_role_display_name,
                new_role_slug=decision.new_role_slug,
            )
            logger.info(
                "[stage4] case=%s chosen=%s confidence=%.2f queued=%s llm2=%s",
                decision.case,
                decision.chosen_role.slug if decision.chosen_role else None,
                decision.confidence, decision.queued, decision.llm2_fired,
            )
        except Exception as exc:
            logger.warning("[skills/extract-from-jd] Stage 4 failed: %s", exc)

    # ── History start_run (before Stage 5 so we have a run_id to reference) ──
    response = JDSkillPipelineResponse(
        final_skills=final_skills,
        jd_role=jd_role,
        nano_parsed=nano_parsed,
        stage3_signals=stage3_signals,
        stage4_decision=stage4_response,
    )
    history_repo = JdPipelineRunRepository()
    run_id = await asyncio.to_thread(
        history_repo.start_run,
        jd_text=jd_text,
        api1_response=response,
        api_parser_response=nano_parsed,
        jd_role_hint_display=jd_role.display_name if jd_role else None,
        llm_cost_usd=cost_acc.total_cost_usd,
    )
    response.run_id = run_id

    # ── v3 auto-fire (Case NEW, must run BEFORE Stage 5) ─────────────────────
    # Stage 4 routed to Case NEW: chosen_role is synthesized (role_id=None).
    # When JD_V3_AUTO_TRIGGER=true, upsert the role now so Stage 5 has a real
    # role_id to attach unknown skills/KRA to. v3 Stage 0 cascade fires in
    # the background. The pipeline ALWAYS returns a chosen_role to the
    # caller — v3 is for catalog enrichment, not for the current JD's
    # classification answer.
    if (
        decision is not None
        and decision.is_new_role
        and decision.chosen_role is not None
        and decision.new_role_display_name
    ):
        auto_trigger = os.getenv("JD_V3_AUTO_TRIGGER", "false").lower() == "true"
        if auto_trigger:
            try:
                from skill_library_v3 import runner as v3_runner

                created = await asyncio.to_thread(
                    v3_runner.create_initial_run,
                    decision.new_role_display_name,
                )
                run_uuid = created["run_id"]
                role_slug = created["role_slug"]
                # Resolve real role_id (upsert_role just persisted the row).
                from jd_similarity_matcher import _pg_connect
                def _resolve_role_id():
                    conn = _pg_connect()
                    try:
                        with conn.cursor() as cur:
                            cur.execute(
                                "SELECT id FROM roles WHERE slug = %s LIMIT 1",
                                (role_slug,),
                            )
                            r = cur.fetchone()
                            return int(r[0]) if r else None
                    finally:
                        conn.close()
                real_role_id = await asyncio.to_thread(_resolve_role_id)
                # Back-fill chosen_role so Stage 5 + history see the real ID.
                decision.chosen_role.role_id = real_role_id
                decision.chosen_role.slug = role_slug
                decision.new_role_slug = role_slug
                if stage4_response is not None and stage4_response.chosen_role:
                    stage4_response.chosen_role.role_id = real_role_id
                    stage4_response.chosen_role.slug = role_slug
                    stage4_response.new_role_slug = role_slug
                asyncio.create_task(v3_runner.run_stage_0(run_uuid))
                if response.stage5_updates is None:
                    response.stage5_updates = Stage5UpdatesResponse()
                response.stage5_updates.v3_pipeline_triggered = True
                response.stage5_updates.v3_run_id = str(run_uuid)
                response.stage5_updates.v3_role_slug = role_slug
                logger.info(
                    "[v3/auto-fire] role=%r run_id=%s slug=%s role_id=%s "
                    "(Stage 0 dispatched)",
                    decision.new_role_display_name, run_uuid, role_slug,
                    real_role_id,
                )
            except Exception as exc:
                logger.warning(
                    "[v3/auto-fire] failed for role=%r: %s",
                    decision.new_role_display_name, exc,
                )
        else:
            logger.info(
                "[v3/auto-fire] would fire for role=%r slug=%s "
                "(JD_V3_AUTO_TRIGGER=false, skipping)",
                decision.new_role_display_name, decision.new_role_slug,
            )

    # ── Stage 5 (Post-Classification Updates) — compulsory ───────────────────
    new_skills_attached: list[CandidateSkillAttached] = []
    new_kra_attached: CandidateKraAttached | None = None
    if decision is not None and run_id is not None and s3 is not None:
        try:
            from jd_classifier import (
                NEW_KRA_THRESHOLD, detect_unknown_skills,
                enqueue_kra_candidate, enqueue_skill_candidates,
                query_matched_skill_names, stage5_apply_updates,
            )
            from jd_similarity_matcher import _SKILL_SCHEMA, _pg_connect

            jd_skill_dicts = [
                {"skill_name": s.skill_name, "is_primary": s.is_primary}
                for s in final_skills
            ]

            def _run_stage5():
                conn = _pg_connect()
                try:
                    with conn:
                        updates = stage5_apply_updates(
                            conn,
                            jd_run_id=run_id,
                            decision=decision,
                            r_and_r_embedding=s3.r_and_r_embedding,
                            r_and_r_text=r_and_r_text,
                            role_name_input=nano_role,
                            s3=s3,
                        )
                        skill_attachments: list[CandidateSkillAttached] = []
                        kra_attachment: CandidateKraAttached | None = None
                        # For Case NEW with v3 auto-trigger off, chosen_role
                        # is synthesized but role_id is None — skip skill/KRA
                        # FK-dependent writes since there's no roles row yet.
                        has_real_role_id = (
                            decision.chosen_role is not None
                            and decision.chosen_role.role_id is not None
                        )
                        if has_real_role_id and not decision.queued:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "UPDATE jd_pipeline_runs "
                                    "SET chosen_role_id = %s, "
                                    "    chosen_role_display = %s "
                                    "WHERE id = %s",
                                    (
                                        decision.chosen_role.role_id,
                                        decision.chosen_role.display_name,
                                        run_id,
                                    ),
                                )

                            matched = query_matched_skill_names(
                                conn,
                                [d["skill_name"] for d in jd_skill_dicts],
                                schema=_SKILL_SCHEMA,
                            )
                            unknown = detect_unknown_skills(jd_skill_dicts, matched)
                            if unknown:
                                ids = enqueue_skill_candidates(
                                    conn,
                                    jd_run_id=run_id,
                                    role_id=decision.chosen_role.role_id,
                                    unknown_skills=unknown,
                                )
                                # Pair each new id back with its source skill;
                                # `enqueue_skill_candidates` preserves input order
                                # and skips ON CONFLICT rows, so we walk in lockstep.
                                for s_in, qid in zip(unknown, ids):
                                    skill_attachments.append(
                                        CandidateSkillAttached(
                                            skill_name=s_in["skill_name"],
                                            is_primary=bool(s_in.get("is_primary", False)),
                                            queue_id=qid,
                                            role_display_name=decision.chosen_role.display_name,
                                            role_slug=decision.chosen_role.slug,
                                        )
                                    )

                            chosen_kra_score = next(
                                (
                                    r.score for r in s3.kra_match_roles
                                    if r.role_id == decision.chosen_role.role_id
                                ),
                                0.0,
                            )
                            if chosen_kra_score < NEW_KRA_THRESHOLD and r_and_r_text:
                                kid = enqueue_kra_candidate(
                                    conn,
                                    jd_run_id=run_id,
                                    role_id=decision.chosen_role.role_id,
                                    r_and_r_text=r_and_r_text[:4000],
                                    best_kra_similarity=chosen_kra_score,
                                )
                                if kid is not None:
                                    kra_attachment = CandidateKraAttached(
                                        queue_id=kid,
                                        role_display_name=decision.chosen_role.display_name,
                                        role_slug=decision.chosen_role.slug,
                                        best_kra_similarity=float(chosen_kra_score),
                                        r_and_r_preview=r_and_r_text[:200],
                                    )
                        return updates, skill_attachments, kra_attachment
                finally:
                    conn.close()

            updates, new_skills_attached, new_kra_attached = await asyncio.to_thread(_run_stage5)
            # Preserve v3 auto-fire fields populated earlier (BEFORE Stage 5)
            # — building a fresh Stage5UpdatesResponse here would clobber
            # v3_pipeline_triggered / v3_run_id / v3_role_slug to defaults.
            existing_v3 = response.stage5_updates
            response.stage5_updates = Stage5UpdatesResponse(
                centroid_updated=updates.centroid_updated,
                centroid_n_after=updates.centroid_n_after,
                queue_entry_id=updates.queue_entry_id,
                collision_log_id=updates.collision_log_id,
                new_skills_attached=new_skills_attached,
                new_kra_attached=new_kra_attached,
                v3_pipeline_triggered=(
                    existing_v3.v3_pipeline_triggered if existing_v3 else False
                ),
                v3_run_id=existing_v3.v3_run_id if existing_v3 else None,
                v3_role_slug=existing_v3.v3_role_slug if existing_v3 else None,
            )
            logger.info(
                "[stage5] centroid=%s n=%s queue=%s collision=%s new_skills=%d new_kra=%s",
                updates.centroid_updated, updates.centroid_n_after,
                updates.queue_entry_id, updates.collision_log_id,
                len(new_skills_attached),
                new_kra_attached.queue_id if new_kra_attached else None,
            )
        except Exception as exc:
            logger.warning("[skills/extract-from-jd] Stage 5 failed: %s", exc)

    # Persist final api1_response (now with stage4/5 fields)
    # back to jd_pipeline_runs so the history list + detail page can read
    # them. start_run captured an earlier snapshot; this overwrites with
    # the post-Stage-5/v3 state.
    if run_id is not None:
        cost_acc.log_summary("skills/extract-from-jd", logger)
        await asyncio.to_thread(
            history_repo.update_api1_response,
            run_id,
            response,
            llm_cost_usd=cost_acc.total_cost_usd,
        )
        # If Stage 4 queued (no confident classification), clear the
        # chosen_role columns. The legacy API3 finalize path may have
        # written a value via /skills/final-role-output; we run AFTER it
        # so the clear wins. Prevents the contradictory UI where the
        # narrative says "queued" but the history list says "Data Engineer".
        if decision is not None and decision.queued:
            def _clear_chosen():
                from jd_similarity_matcher import _pg_connect
                conn = _pg_connect()
                try:
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE jd_pipeline_runs "
                                "SET chosen_role_id = NULL, "
                                "    chosen_role_display = NULL "
                                "WHERE id = %s",
                                (run_id,),
                            )
                finally:
                    conn.close()
            try:
                await asyncio.to_thread(_clear_chosen)
            except Exception as exc:
                logger.warning(
                    "[stage4/clear-chosen] failed for run_id=%s: %s",
                    run_id, exc,
                )

    return response


def _canonical_summary_from_row(row: dict) -> CanonicalSkillSummary:
    def _enum_str(v):
        if v is None:
            return None
        return str(v)

    return CanonicalSkillSummary(
        id=int(row["skill_id"]),
        slug=row.get("skill_slug"),
        display_name=row.get("skill_display_name"),
        category_id=row.get("category_id"),
        sub_category_id=row.get("sub_category_id"),
        skill_nature=_enum_str(row.get("skill_nature")),
        volatility=_enum_str(row.get("volatility")),
        is_extractable=row.get("is_extractable"),
        is_also_category=row.get("is_also_category"),
        typical_lifespan=_enum_str(row.get("typical_lifespan")),
    )


def _role_summary_from_db_row(row: dict) -> RoleSummary:
    return RoleSummary(
        source="db",
        id=int(row["role_id"]),
        slug=str(row.get("slug") or ""),
        display_name=str(row.get("display_name") or ""),
        role_archetype=row.get("role_archetype"),
    )


def _role_summary_from_llm(role: dict) -> RoleSummary:
    return RoleSummary(
        source="llm",
        slug=str(role.get("slug") or ""),
        display_name=str(role.get("display_name") or ""),
        role_archetype=role.get("role_archetype"),
        rationale=role.get("rationale"),
    )


@app.post("/skills/extract-details", response_model=ExtractDetailsResponse)
async def extract_skill_details_endpoint(req: ExtractDetailsRequest):
    """Reverse planner: alias resolution -> dimension lookup -> role inference.

    For EVERY skill in final_skills:
      - If matched in DB (whether it came from llm_skills or initial_skills),
        attach the canonical row, its known aliases, its DB dimensions, and
        the roles linked to those dimensions. source_tag = "db".
      - If it came from llm_skills and did NOT match the DB, run the reverse
        planner (grounded against the existing dimensions catalogue) to
        infer dimensions; for any dimension still not in the DB, ask the
        LLM for a role too. source_tag = "llm".
      - If it came from initial_skills and somehow did NOT match the DB,
        surface it tagged "unmatched" with no dimensions (rare).

    Persists new aliases for llm_skills that match a canonical skill
    (idempotent). Does NOT persist new canonical_skills, dimensions, or roles.
    """
    final_skills = _dedupe_case_insensitive(_clean_strings(req.final_skills))
    llm_skills = _dedupe_case_insensitive(_clean_strings(req.llm_skills))
    llm_skills_lower = {s.lower() for s in llm_skills}
    primary_skills_lower = {s.strip().lower() for s in (req.primary_skills or []) if s.strip()}

    cost_acc = CostAccumulator()
    repo = SkillLibraryRepository()
    planner = AzureReversePlannerLLM(accumulator=cost_acc)

    # ── Stage 1: alias resolution for ALL final_skills ──────────────────────
    # We resolve every final_skill (not just llm_skills) so initial_skills
    # also get their DB info attached. Alias persistence only fires for
    # llm_skills, since initial_skills already came from canonical lookup.
    alias_lookup: dict[str, dict] = {}
    if final_skills:
        alias_lookup = await asyncio.to_thread(
            repo.find_canonical_skills_by_aliases, final_skills
        )
        unmatched_after_exact = [
            t for t in final_skills if t.strip().lower() not in alias_lookup
        ]
        if unmatched_after_exact:
            emb_lookup = await asyncio.to_thread(
                repo.find_canonical_skills_by_alias_embedding,
                unmatched_after_exact,
                cost_acc=cost_acc,
            )
            for key, rec in emb_lookup.items():
                alias_lookup[key] = rec
            if emb_lookup:
                logger.info(
                    "[skills/extract-details] embedding_skill_match count=%d terms=%s",
                    len(emb_lookup),
                    [
                        {
                            "term": k,
                            "canonical": v.get("skill_display_name"),
                            "sim": v.get("embedding_similarity"),
                            "via": v.get("matched_via"),
                        }
                        for k, v in emb_lookup.items()
                    ],
                )

    alias_matches: list[AliasMatch] = []
    unmatched_llm_skills: list[str] = []
    aliases_to_insert: list[tuple[int, str]] = []

    # Per-final_skill bookkeeping; we'll build SkillDetail entries from these.
    matched_per_final: dict[str, dict] = {}
    new_alias_per_final: dict[str, str] = {}

    for term in final_skills:
        hit = alias_lookup.get(term.lower())
        was_llm = term.lower() in llm_skills_lower
        if not hit:
            if was_llm:
                unmatched_llm_skills.append(term)
            continue

        canonical = _canonical_summary_from_row(hit)
        existing_alias_id = hit.get("alias_id")
        existing_alias_text = hit.get("alias_text")
        matched_via = str(hit.get("matched_via") or "").strip()
        if not matched_via:
            matched_via = "alias" if existing_alias_id is not None else "display_name"

        matched_per_final[term] = {
            "canonical": canonical,
            "matched_via": matched_via,
            "existing_alias_id": existing_alias_id,
            "existing_alias_text": existing_alias_text,
        }

        # Only auto-persist NEW aliases for llm_skills. For initial_skills
        # the term already came from a canonical lookup, so there's nothing
        # to learn.
        if not was_llm:
            continue

        already_same = (
            existing_alias_text is not None
            and str(existing_alias_text).strip().lower() == term.strip().lower()
        )
        # Also skip if the term is exactly the canonical display_name (no new
        # info to add to the alias table).
        is_display_name = (
            canonical.display_name is not None
            and canonical.display_name.strip().lower() == term.strip().lower()
        )

        if already_same or is_display_name:
            alias_matches.append(AliasMatch(
                input_term=term,
                matched_via=matched_via,
                matched_canonical=canonical,
                existing_alias_id=existing_alias_id,
                existing_alias_text=existing_alias_text,
                alias_persisted=False,
                alias_persist_skipped_reason=(
                    "alias_text already exists for this canonical skill"
                    if already_same
                    else "term is the canonical display_name"
                ),
            ))
        else:
            aliases_to_insert.append((canonical.id, term))
            new_alias_per_final[term] = term
            alias_matches.append(AliasMatch(
                input_term=term,
                matched_via=matched_via,
                matched_canonical=canonical,
                existing_alias_id=existing_alias_id,
                existing_alias_text=existing_alias_text,
                alias_persisted=True,
            ))

    # TODO: REMOVE COMMENT AFTER TESTING — persist new synonyms for matched LLM terms into skill_aliases.
    # new_aliases_persisted = 0
    # persisted_alias_pairs: list[tuple[int, str]] = []
    # if aliases_to_insert:
    #     try:
    #         new_aliases_persisted = await asyncio.to_thread(
    #             repo.add_aliases, aliases_to_insert
    #         )
    #         persisted_alias_pairs = list(aliases_to_insert)
    #     except Exception as exc:
    #         logger.exception("Persisting aliases failed: %s", exc)
    #         for am in alias_matches:
    #             if am.alias_persisted:
    #                 am.alias_persisted = False
    #                 am.alias_persist_skipped_reason = f"db error: {exc}"
    #         new_alias_per_final.clear()
    #         new_aliases_persisted = 0
    #         persisted_alias_pairs = []
    new_aliases_persisted = 0
    persisted_alias_pairs: list[tuple[int, str]] = []
    for am in alias_matches:
        if am.alias_persisted:
            am.alias_persisted = False
            am.alias_persist_skipped_reason = (
                "TODO: REMOVE AFTER TESTING — alias DB write disabled"
            )
    # Keep skills_detail/UI consistent with skipped DB persist.
    new_alias_per_final.clear()

    # ── Stage 2: pull DB enrichment for every matched skill ─────────────────
    matched_skill_ids = sorted({
        int(info["canonical"].id) for info in matched_per_final.values()
    })

    aliases_by_skill: dict[int, list[dict]] = {}
    dims_by_skill: dict[int, list[dict]] = {}
    roles_by_dim_id: dict[int, list[dict]] = {}

    if matched_skill_ids:
        aliases_by_skill, dims_by_skill = await asyncio.gather(
            asyncio.to_thread(repo.fetch_aliases_for_skill_ids, matched_skill_ids),
            asyncio.to_thread(repo.fetch_dimensions_for_skill_ids, matched_skill_ids),
        )
        all_dim_ids = sorted({
            int(d["id"]) for dims in dims_by_skill.values() for d in dims
        })
        if all_dim_ids:
            roles_by_dim_id = await asyncio.to_thread(
                repo.fetch_roles_for_dimensions, all_dim_ids
            )

    # ── Stage 3: v3 per-skill orchestrator for unmatched llm_skills ─────────
    # Each unknown skill goes through the full v3 stack (dim-gen -> sim/recon
    # -> type -> place -> contain -> enrich). Result is a v3-shaped
    # NewSkillMetaV3 payload kept on SkillDetail.new_skill_meta.
    new_skill_meta_by_skill: dict[str, NewSkillMetaV3] = {}

    # Run the orchestrator for every unmatched term that came from llm_skills.
    # We intentionally do not pre-filter single lowercase tokens: extractors
    # often emit brand names like "firebase" / "flutter" / "jetpack", which
    # must still be enriched and dimension-placed.

    role_hint_text = ""
    if req.jd_role_hint is not None and (req.jd_role_hint.display_name or "").strip():
        role_hint_text = req.jd_role_hint.display_name.strip()

    if unmatched_llm_skills:
        # Cost-regression: one batched LLM call replaces the per-skill
        # enrich_new_skill loop. Per-skill enrichment (Stage 2 dim-gen,
        # Stage 3 reconciler, Stage 7 sub-agents) used to fire 6+ LLM calls
        # per unmatched skill and pushed API 2 cost to $0.40+ for JDs with
        # 20+ unmatched skills. Now we only get category/nature/lifespan
        # in-request; richer enrichment (vendor/license, maturity, ambiguity,
        # versioning, dim placement) is deferred to the v3 background
        # pipeline that fires for Case NEW.
        try:
            new_skill_meta_by_skill.update(
                await enrich_unmatched_skills_batched(planner, unmatched_llm_skills)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[skills/extract-details] batched unmatched-skill enrichment "
                "failed for %d skills: %s",
                len(unmatched_llm_skills), exc,
            )

    # Backfill role labels on Stage 3 logs from DB dimension-role links.
    # Some logs can miss b_role/a_role depending on upstream payload shape.
    if new_skill_meta_by_skill:
        dim_slug_refs: set[str] = set()
        for meta in new_skill_meta_by_skill.values():
            for dim in (meta.locked_dimensions or []):
                for ov in (dim.overlap_flags or []):
                    for ref in (ov.with_dim_id, ov.with_dim_name):
                        s = str(ref or "").strip()
                        if s:
                            dim_slug_refs.add(s)
            for row in (meta.keep_log or []):
                for key in ("a_dim_id", "b_dim_id"):
                    slug = str(row.get(key) or "").strip()
                    if slug:
                        dim_slug_refs.add(slug)
            for row in (meta.merge_log or []):
                for key in ("a_dim_id", "b_dim_id"):
                    slug = str(row.get(key) or "").strip()
                    if slug:
                        dim_slug_refs.add(slug)
                for slug in (row.get("merged_from") or []):
                    s = str(slug or "").strip()
                    if s:
                        dim_slug_refs.add(s)
            for row in (meta.split_log or []):
                for key in ("a_dim_id", "b_dim_id"):
                    slug = str(row.get(key) or "").strip()
                    if slug:
                        dim_slug_refs.add(slug)
                for slug in (row.get("split_from") or []):
                    s = str(slug or "").strip()
                    if s:
                        dim_slug_refs.add(s)

        dim_rows_by_key: dict[str, dict] = {}
        roles_by_dim_id_for_logs: dict[int, list[dict]] = {}
        if dim_slug_refs:
            dim_rows_by_key = await asyncio.to_thread(
                repo.find_dimensions_by_names,
                [],
                list(dim_slug_refs),
            )
            dim_ids_for_logs = sorted({
                int(v.get("id"))
                for v in dim_rows_by_key.values()
                if v.get("id") is not None
            })
            if dim_ids_for_logs:
                roles_by_dim_id_for_logs = await asyncio.to_thread(
                    repo.fetch_roles_for_dimensions,
                    dim_ids_for_logs,
                )

        def _roles_for_dim_slug(dim_slug: str) -> str:
            key = str(dim_slug or "").strip().lower()
            if not key:
                return ""
            dim_row = dim_rows_by_key.get(key)
            if not dim_row:
                return ""
            dim_id = dim_row.get("id")
            if dim_id is None:
                return ""
            role_rows = roles_by_dim_id_for_logs.get(int(dim_id), [])
            if not role_rows:
                return ""
            names = [str(r.get("display_name") or "").strip() for r in role_rows]
            names = [n for n in names if n]
            return ", ".join(names)

        for meta in new_skill_meta_by_skill.values():
            for row in (meta.keep_log or []):
                if not str(row.get("a_role") or "").strip():
                    row["a_role"] = _roles_for_dim_slug(str(row.get("a_dim_id") or ""))
                if not str(row.get("b_role") or "").strip():
                    row["b_role"] = _roles_for_dim_slug(str(row.get("b_dim_id") or ""))
            for row in (meta.merge_log or []):
                if not str(row.get("a_role") or "").strip():
                    row["a_role"] = _roles_for_dim_slug(str(row.get("a_dim_id") or ""))
                if not str(row.get("b_role") or "").strip():
                    row["b_role"] = _roles_for_dim_slug(str(row.get("b_dim_id") or ""))
            for row in (meta.split_log or []):
                if not str(row.get("a_role") or "").strip():
                    row["a_role"] = _roles_for_dim_slug(str(row.get("a_dim_id") or ""))
                if not str(row.get("b_role") or "").strip():
                    row["b_role"] = _roles_for_dim_slug(str(row.get("b_dim_id") or ""))
            # Fallback path used by UI when keep_log is empty:
            # enrich overlap flags with DB-attached role for referenced dims.
            for dim in (meta.locked_dimensions or []):
                for ov in (dim.overlap_flags or []):
                    if str(ov.with_role or "").strip():
                        continue
                    ref_slug = str(ov.with_dim_id or "").strip()
                    if not ref_slug:
                        ref_slug = str(ov.with_dim_name or "").strip()
                    resolved = _roles_for_dim_slug(ref_slug)
                    if resolved:
                        ov.with_role = resolved

    # Build (skill, llm_dim) pairs from each unknown skill's v3 locked dims.
    # The DimensionSummary uses tentative_id as the slug so downstream API 3
    # treats it as an LLM-proposed dimension (no DB id).
    llm_pairs: list[tuple[str, dict]] = []
    for skill, meta in new_skill_meta_by_skill.items():
        for cd in meta.locked_dimensions:
            llm_pairs.append((skill, {
                "slug": cd.tentative_id,
                "display_name": cd.name,
                "rationale": cd.description,
                "difficulty_hint": None,
            }))

    # Even though dims came from v3 (locked), the legacy catalog may already
    # have one of these as a real DB row. Probing find_dimensions_by_names
    # lets us promote those entries from source="llm" to source="db" so the
    # API 3 link path can attach to the existing dim instead of proposing a
    # duplicate.
    all_display_names: list[str] = [
        d.get("display_name") for (_, d) in llm_pairs if d.get("display_name")
    ]
    all_slugs: list[str] = [
        d.get("slug") for (_, d) in llm_pairs if d.get("slug")
    ]
    db_dim_map_for_llm: dict[str, dict] = {}
    if llm_pairs:
        db_dim_map_for_llm = await asyncio.to_thread(
            find_dimensions_by_identity_skill_library,
            all_display_names,
            all_slugs,
        )
        new_dim_ids = sorted({
            int(rec["id"]) for rec in db_dim_map_for_llm.values()
            if rec.get("id") is not None and int(rec["id"]) not in roles_by_dim_id
        })
        if new_dim_ids:
            extra_roles = await asyncio.to_thread(
                repo.fetch_roles_for_dimensions, new_dim_ids
            )
            roles_by_dim_id.update(extra_roles)

    # Build (skill, llm_dim, db_dim) tuples for each unknown skill.
    pending_llm_pair_meta: list[tuple[str, dict, dict | None]] = []
    for skill, d in llm_pairs:
        name_key = (d.get("display_name") or "").lower()
        slug_key = (d.get("slug") or "").lower()
        db_row = (
            db_dim_map_for_llm.get(name_key)
            or db_dim_map_for_llm.get(slug_key)
        )
        pending_llm_pair_meta.append((skill, d, db_row))

    # The v3 orchestrator does its own placement / containment per skill,
    # so per-pair LLM role inference is no longer needed. Roles for the
    # JD will come from DB-matched dims (Stage 2 above) and from the
    # role-picker prompt at Stage 6 below.
    llm_roles_by_idx: dict[int, dict | None] = {}

    # ── Stage 4: aggregate dimensions per skill + candidate roles ───────────
    dimension_details: list[DimensionDetail] = []
    candidate_roles: list[RoleSummary] = []
    seen_role_keys: set[str] = set()

    def _add_candidate_role(role: RoleSummary) -> None:
        key = f"{role.source}:{role.id or ''}:{role.slug.lower()}"
        if key in seen_role_keys:
            return
        seen_role_keys.add(key)
        candidate_roles.append(role)

    # 4a. DB-matched skills: build their DimensionDetails from DB.
    db_dim_details_by_skill: dict[str, list[DimensionDetail]] = {}
    for term, info in matched_per_final.items():
        canonical: CanonicalSkillSummary = info["canonical"]
        sid = int(canonical.id)
        details_for_term: list[DimensionDetail] = []
        for d_row in dims_by_skill.get(sid, []) or []:
            dim_summary = DimensionSummary(
                source="db",
                id=int(d_row["id"]),
                slug=str(d_row.get("slug") or ""),
                display_name=str(d_row.get("display_name") or ""),
                rationale=d_row.get("rationale"),
                difficulty_hint=str(d_row.get("difficulty_hint") or "") or None,
            )
            db_role_rows = roles_by_dim_id.get(int(d_row["id"]), [])
            db_role_summaries = [
                _role_summary_from_db_row(r) for r in db_role_rows
            ]
            for r in db_role_summaries:
                _add_candidate_role(r)
            detail = DimensionDetail(
                input_skill=term,
                dimension=dim_summary,
                roles_from_db=db_role_summaries,
                llm_role=None,
            )
            details_for_term.append(detail)
            dimension_details.append(detail)
        db_dim_details_by_skill[term] = details_for_term

    # 4b. Unmatched llm_skills: build DimensionDetails from llm_pairs.
    llm_dim_details_by_skill: dict[str, list[DimensionDetail]] = {}
    for idx, (skill, d, db_row) in enumerate(pending_llm_pair_meta):
        if db_row is not None:
            dim_summary = DimensionSummary(
                source="db",
                id=int(db_row["id"]),
                slug=str(db_row.get("slug") or d.get("slug") or ""),
                display_name=str(
                    db_row.get("display_name") or d.get("display_name") or ""
                ),
                rationale=db_row.get("rationale") or d.get("rationale"),
                difficulty_hint=(
                    str(db_row.get("difficulty_hint") or d.get("difficulty_hint") or "")
                    or None
                ),
            )
            db_role_rows = roles_by_dim_id.get(int(db_row["id"]), [])
            db_role_summaries = [
                _role_summary_from_db_row(r) for r in db_role_rows
            ]
            for r in db_role_summaries:
                _add_candidate_role(r)
            llm_role = llm_roles_by_idx.get(idx)
            llm_role_summary = (
                _role_summary_from_llm(llm_role) if llm_role else None
            )
            if llm_role_summary:
                _add_candidate_role(llm_role_summary)
            detail = DimensionDetail(
                input_skill=skill,
                dimension=dim_summary,
                roles_from_db=db_role_summaries,
                llm_role=llm_role_summary,
            )
        else:
            dim_summary = DimensionSummary(
                source="llm",
                id=None,
                slug=str(d.get("slug") or ""),
                display_name=str(d.get("display_name") or ""),
                rationale=d.get("rationale"),
                difficulty_hint=d.get("difficulty_hint"),
            )
            llm_role = llm_roles_by_idx.get(idx)
            llm_role_summary = (
                _role_summary_from_llm(llm_role) if llm_role else None
            )
            if llm_role_summary:
                _add_candidate_role(llm_role_summary)
            detail = DimensionDetail(
                input_skill=skill,
                dimension=dim_summary,
                roles_from_db=[],
                llm_role=llm_role_summary,
            )
        llm_dim_details_by_skill.setdefault(skill, []).append(detail)
        dimension_details.append(detail)

    # ── Stage 5: build per-skill detail (one entry per final_skill) ─────────
    skills_detail: list[SkillDetail] = []
    for term in final_skills:
        was_llm = term.lower() in llm_skills_lower
        info = matched_per_final.get(term)
        if info is not None:
            canonical: CanonicalSkillSummary = info["canonical"]
            sid = int(canonical.id)
            alias_rows = aliases_by_skill.get(sid, []) or []
            alias_objs = [
                AliasInfo(
                    id=int(r["id"]),
                    alias_text=str(r["alias_text"]),
                    alias_type=str(r.get("alias_type")) if r.get("alias_type") is not None else None,
                    match_strategy=str(r.get("match_strategy")) if r.get("match_strategy") is not None else None,
                    is_primary=bool(r.get("is_primary")) if r.get("is_primary") is not None else None,
                )
                for r in alias_rows
            ]
            new_alias_text = new_alias_per_final.get(term)
            skills_detail.append(SkillDetail(
                input_skill=term,
                source_tag="db",
                was_in_llm_skills=was_llm,
                canonical=canonical,
                matched_via=info["matched_via"],
                aliases_in_db=alias_objs,
                new_alias_persisted=new_alias_text is not None,
                new_alias_text=new_alias_text,
                dimensions=db_dim_details_by_skill.get(term, []),
            ))
            continue

        if was_llm:
            v3_meta = new_skill_meta_by_skill.get(term)
            skills_detail.append(SkillDetail(
                input_skill=term,
                source_tag="llm",
                was_in_llm_skills=True,
                canonical=None,
                matched_via=None,
                aliases_in_db=[],
                new_alias_persisted=False,
                new_alias_text=None,
                dimensions=llm_dim_details_by_skill.get(term, []),
                new_skill_meta=v3_meta,
            ))
        else:
            # initial_skills term that didn't canonicalize — surface it but
            # without dimensions so the consumer can flag it.
            skills_detail.append(SkillDetail(
                input_skill=term,
                source_tag="unmatched",
                was_in_llm_skills=False,
                canonical=None,
                matched_via=None,
                aliases_in_db=[],
                new_alias_persisted=False,
                new_alias_text=None,
                dimensions=[],
            ))

    # ── Stage 6: pick a single chosen role ──────────────────────────────────
    chosen_role: ChosenRole | None = None
    if len(candidate_roles) == 0 and req.jd_role_hint is not None and (req.jd_role_hint.display_name or "").strip():
        h = req.jd_role_hint
        chosen_role = ChosenRole(
            source="llm",
            id=None,
            slug=h.slug or "",
            display_name=h.display_name.strip(),
            role_archetype=h.role_archetype,
            rationale=h.rationale,
        )
    elif len(candidate_roles) == 1:
        c = candidate_roles[0]
        chosen_role = ChosenRole(
            source="single_candidate",
            id=c.id,
            slug=c.slug,
            display_name=c.display_name,
            role_archetype=c.role_archetype,
            rationale=c.rationale,
        )
    elif len(candidate_roles) > 1:
        # Build raw payload, then deduplicate by normalised slug (backend_engineer
        # and backend-engineer are the same role), preferring DB candidates.
        _raw_payload: list[dict] = []
        for c in candidate_roles:
            payload = {
                "source": c.source,
                "slug": c.slug,
                "display_name": c.display_name,
            }
            if c.id is not None:
                payload["id"] = c.id
            _raw_payload.append(payload)

        _seen_norm: dict[str, int] = {}
        _deduped: list[dict] = []
        for p in _raw_payload:
            norm = p["slug"].lower().replace("_", "-")
            if norm in _seen_norm:
                # Upgrade to DB candidate if the existing slot is LLM
                if p.get("id") is not None and _deduped[_seen_norm[norm]].get("id") is None:
                    _deduped[_seen_norm[norm]] = p
            else:
                _seen_norm[norm] = len(_deduped)
                _deduped.append(p)

        # Keep DB-sourced first, then LLM; cap total at 25
        _db_cands = [p for p in _deduped if p.get("id") is not None]
        _llm_cands = [p for p in _deduped if p.get("id") is None]
        cand_payload = (_db_cands + _llm_cands)[:25]

        # Pivot to dimension-centric map: roles_from_db sent once per unique
        # dimension instead of once per (skill × dimension) pair.
        _dim_groups: dict[str, dict] = {}
        for dd in dimension_details:
            slug = dd.dimension.slug
            if slug not in _dim_groups:
                _dim_groups[slug] = {
                    "dimension": {"slug": slug, "display_name": dd.dimension.display_name},
                    "skills": [],
                    "roles_from_db": [r.display_name for r in dd.roles_from_db[:3]],
                    "llm_roles": [],
                }
            _dim_groups[slug]["skills"].append(dd.input_skill)
            if dd.llm_role is not None:
                lr = dd.llm_role
                _dim_groups[slug]["llm_roles"].append({
                    "skill": dd.input_skill,
                    "slug": lr.slug,
                    "display_name": lr.display_name,
                })
        dimension_role_map = list(_dim_groups.values())

        jd_role_hint_ctx = None
        if req.jd_role_hint is not None and (
            req.jd_role_hint.display_name or ""
        ).strip():
            h = req.jd_role_hint
            jd_role_hint_ctx = {
                "display_name": h.display_name.strip(),
                "slug": (h.slug or "").strip(),
                "role_archetype": (h.role_archetype or "").strip() or None,
                "rationale": (h.rationale or "").strip() or None,
            }

        context = {
            "final_skills": [
                {"skill_name": s, "is_primary": s.lower() in primary_skills_lower}
                for s in final_skills
            ],
            "jd_role_hint": jd_role_hint_ctx,
            "dimension_role_map": dimension_role_map,
        }
        try:
            picked = await planner.pick_role(cand_payload, context)
        except Exception as exc:
            logger.warning("pick_role failed: %s", exc)
            picked = None

        # Lookup so we can snap chosen_role onto the canonical DB row when
        # the picker chose an existing DB role.
        db_role_lookup = {
            c.id: c for c in candidate_roles
            if c.source == "db" and c.id is not None
        }

        if picked:
            src_id = picked.get("source_role_id")
            db_role = (
                db_role_lookup.get(src_id) if isinstance(src_id, int) else None
            )
            if db_role is not None:
                chosen_role = ChosenRole(
                    source="db",
                    id=db_role.id,
                    slug=db_role.slug,
                    display_name=db_role.display_name,
                    role_archetype=db_role.role_archetype,
                    rationale=picked.get("rationale"),
                )
            else:
                chosen_role = ChosenRole(
                    source="llm",
                    id=None,
                    slug=str(picked.get("slug") or ""),
                    display_name=str(picked.get("display_name") or ""),
                    role_archetype=picked.get("role_archetype"),
                    rationale=picked.get("rationale"),
                )
        else:
            c = candidate_roles[0]
            chosen_role = ChosenRole(
                source=c.source,
                id=c.id,
                slug=c.slug,
                display_name=c.display_name,
                role_archetype=c.role_archetype,
                rationale=c.rationale,
            )

    cost_acc.log_summary("skills/extract-details", logger)
    response = ExtractDetailsResponse(
        input_final_skills=final_skills,
        input_llm_skills=llm_skills,
        alias_matches=alias_matches,
        new_aliases_persisted=new_aliases_persisted,
        unmatched_skills=unmatched_llm_skills,
        dimensions=dimension_details,
        skills_detail=skills_detail,
        candidate_roles=candidate_roles,
        chosen_role=chosen_role,
        run_id=req.run_id,
    )

    # ── History persistence (only if API 1 minted a run_id) ────
    if req.run_id:
        history_repo = JdPipelineRunRepository()
        await asyncio.to_thread(
            history_repo.attach_api2,
            req.run_id,
            response,
            llm_cost_usd=cost_acc.total_cost_usd,
        )
        if persisted_alias_pairs:
            artifact_items = [
                {
                    "kind": "alias_added",
                    "artifact_id": int(skill_id),
                    "artifact_text": str(alias_text),
                }
                for (skill_id, alias_text) in persisted_alias_pairs
            ]
            await asyncio.to_thread(
                history_repo.record_artifacts_bulk, req.run_id, artifact_items
            )

    # Post-success reconciliation diagnostics. Helps verify whether B-role data is actually present
    # in API 2 payloads sent to the frontend.
    keep_total = keep_missing_b = 0
    merge_total = merge_missing_b = 0
    split_total = split_missing_b = 0
    missing_b_dim_samples: list[str] = []
    for sd in response.skills_detail:
        meta = sd.new_skill_meta
        if not meta:
            continue
        for row in (meta.keep_log or []):
            keep_total += 1
            b_role = str(row.get("b_role") or "").strip()
            if not b_role:
                keep_missing_b += 1
                b_dim = str(row.get("b_dim_id") or "").strip()
                if b_dim and len(missing_b_dim_samples) < 12:
                    missing_b_dim_samples.append(b_dim)
        for row in (meta.merge_log or []):
            merge_total += 1
            b_role = str(row.get("b_role") or "").strip()
            if not b_role:
                merge_missing_b += 1
                b_dim = str(row.get("b_dim_id") or "").strip()
                if b_dim and len(missing_b_dim_samples) < 12:
                    missing_b_dim_samples.append(b_dim)
        for row in (meta.split_log or []):
            split_total += 1
            b_role = str(row.get("b_role") or "").strip()
            if not b_role:
                split_missing_b += 1
                b_dim = str(row.get("b_dim_id") or "").strip()
                if b_dim and len(missing_b_dim_samples) < 12:
                    missing_b_dim_samples.append(b_dim)

    logger.info(
        "API2_SUCCESS_ROLE_COVERAGE run_id=%s keep=%d missing_b=%d merge=%d missing_b=%d split=%d missing_b=%d missing_b_dims=%s",
        req.run_id,
        keep_total,
        keep_missing_b,
        merge_total,
        merge_missing_b,
        split_total,
        split_missing_b,
        sorted(set(missing_b_dim_samples)),
    )

    return response


def _pipeline_llm_cost_fields(
    c1: float | None = None,
    c2: float | None = None,
    c3: float | None = None,
) -> dict[str, float | None]:
    """Per-API USD costs plus sum of any non-null API totals."""
    vals = [float(x) for x in (c1, c2, c3) if x is not None]
    total = round(sum(vals), 8) if vals else None
    return {
        "llm_cost_api1_usd": float(c1) if c1 is not None else None,
        "llm_cost_api2_usd": float(c2) if c2 is not None else None,
        "llm_cost_api3_usd": float(c3) if c3 is not None else None,
        "llm_cost_total_usd": total,
    }


def _pipeline_llm_cost_fields_from_tuple(
    costs: tuple[float | None, float | None, float | None] | None,
) -> dict[str, float | None]:
    if costs is None:
        return _pipeline_llm_cost_fields()
    return _pipeline_llm_cost_fields(costs[0], costs[1], costs[2])


def _log_jd_pipeline_llm_cost_sum(
    logger: logging.Logger,
    costs: tuple[float | None, float | None, float | None],
    run_id: str,
) -> None:
    c1, c2, c3 = costs
    a = float(c1) if c1 is not None else 0.0
    b = float(c2) if c2 is not None else 0.0
    c = float(c3) if c3 is not None else 0.0
    logger.info(
        "[JD pipeline] LLM cost total (API1+API2+API3) api1=$%.6f api2=$%.6f api3=$%.6f sum=$%.6f run_id=%s",
        a,
        b,
        c,
        a + b + c,
        run_id,
    )


@app.post("/skills/final-role-output", response_model=FinalRoleOutputResponse)
async def final_role_output_endpoint(req: FinalRoleOutputRequest):
    api3_started_at = time.monotonic()
    cost_acc = CostAccumulator()
    repo = SkillLibraryRepository()
    rev_planner = AzureReversePlannerLLM(accumulator=cost_acc)
    history_repo = JdPipelineRunRepository() if req.run_id else None
    artifacts: list[dict[str, Any]] = []
    dimension_worklist = _merge_dimension_worklist(req)

    # API 1 / API 2: alias + non-skill writes stay disabled for testing.
    # API 3: catalogue persistence ON (canonical inserts for new v3 skills,
    # dimension create only when not merged into an existing dim, links).
    _API3_writes_enabled = True

    # 1) Build final input skills with in_db/new tags.
    final_input_skills = [
        FinalInputSkillTag(skill=sd.input_skill, tag=_skill_db_tag(sd))
        for sd in req.skills_detail
    ]

    chosen = req.chosen_role
    persistence = PersistenceReport()
    if chosen is None:
        empty_response = FinalRoleOutputResponse(
            chosen_role=None,
            final_input_skills=final_input_skills,
            persistence=persistence,
            planner_output=None,
            run_id=req.run_id,
            chosen_role_resolution=None,
        )
        cost_acc.log_summary("skills/final-role-output", logger)
        if history_repo is not None:
            duration_ms = int((time.monotonic() - api3_started_at) * 1000)
            await asyncio.to_thread(
                history_repo.attach_api3,
                req.run_id,
                empty_response,
                chosen_role_display=None,
                chosen_role_id=None,
                final_skills=req.input_final_skills,
                duration_ms=duration_ms,
                llm_cost_usd=cost_acc.total_cost_usd,
            )
            costs = await asyncio.to_thread(
                history_repo.get_run_llm_costs_usd, req.run_id
            )
            if costs is not None:
                _log_jd_pipeline_llm_cost_sum(logger, costs, req.run_id)
                cost_fields = _pipeline_llm_cost_fields_from_tuple(costs)
                empty_response = empty_response.model_copy(update=cost_fields)
        return empty_response

    # 2) Resolve role in DB.
    # Check roles table first (by id / slug / display_name), then fall back to
    # role_aliases table. Only create a new role when absent from both tables.
    resolved_role = await asyncio.to_thread(
        repo.find_role_by_identity,
        role_id=chosen.id,
        slug=chosen.slug,
        display_name=chosen.display_name,
    )
    if resolved_role is None:
        resolved_role = await asyncio.to_thread(
            repo.find_role_by_alias_name, chosen.display_name or ""
        )
        if resolved_role is not None:
            logger.info(
                "[final-role-output] chosen role %r resolved via role_aliases → id=%s slug=%s",
                chosen.display_name, resolved_role.get("id"), resolved_role.get("slug"),
            )

    role_missing_initially = resolved_role is None
    planner_output: PlannerGeneratedOutput | None = None
    role_auto_inserted = False

    if role_missing_initially and _API3_writes_enabled:
        matched_rows = sum(
            1 for dd in dimension_worklist if _role_matches_chosen(dd, chosen)
        )
        should_create = _should_create_missing_role(
            chosen,
            matched_rows=matched_rows,
            total_rows=len(dimension_worklist),
        )
        if should_create:
            try:
                role_embed_text = (chosen.display_name or "").strip() + " " + (chosen.role_archetype or "").strip()
                role_vec = await asyncio.to_thread(_embed_for_skill_library_db, role_embed_text.strip())
                similar_roles = (
                    await asyncio.to_thread(repo.find_similar_roles_by_embedding, role_vec)
                    if role_vec is not None
                    else []
                )
                role_reused = False
                if similar_roles:
                    dedup = await rev_planner.confirm_or_reuse_entity(
                        "role",
                        {"display_name": chosen.display_name, "role_archetype": chosen.role_archetype},
                        similar_roles,
                    )
                    if dedup.get("action") == "use_existing" and isinstance(dedup.get("existing_id"), int):
                        resolved_role = await asyncio.to_thread(
                            repo.find_role_by_identity, role_id=dedup["existing_id"]
                        )
                        role_reused = True
                if resolved_role is None:
                    resolved_role = await asyncio.to_thread(
                        repo.create_role,
                        slug=chosen.slug,
                        display_name=chosen.display_name,
                        role_archetype=chosen.role_archetype,
                        source=chosen.source or "llm",
                        name_embedding=role_vec,
                    )
                    if (
                        resolved_role is not None
                        and resolved_role.get("id") is not None
                        and not role_reused
                    ):
                        role_auto_inserted = True
                        artifacts.append({
                            "kind": "role_created",
                            "artifact_id": int(resolved_role["id"]),
                            "artifact_text": str(
                                resolved_role.get("display_name") or chosen.display_name
                            ),
                        })
            except Exception as exc:
                logger.warning("create_role failed for chosen role %r: %s", chosen.slug, exc)

    chosen_role_out = chosen
    if resolved_role is not None:
        chosen_role_out = ChosenRole(
            source="db",
            id=int(resolved_role["id"]),
            slug=str(resolved_role.get("slug") or chosen.slug),
            display_name=str(resolved_role.get("display_name") or chosen.display_name),
            role_archetype=resolved_role.get("role_archetype") or chosen.role_archetype,
            rationale=chosen.rationale,
        )
    elif chosen is not None:
        # Chosen role not in DB: never mint a fake id — human review before insert.
        chosen_role_out = ChosenRole(
            source=chosen.source or "llm",
            id=None,
            slug=chosen.slug,
            display_name=chosen.display_name,
            role_archetype=chosen.role_archetype,
            rationale=chosen.rationale,
        )

    # 3) Map skill -> canonical id for persistence.
    skill_lookup = await asyncio.to_thread(
        repo.find_canonical_skills_by_aliases,
        req.input_final_skills,
    )
    final_skill_by_lower: dict[str, str] = {
        s.lower(): s for s in req.input_final_skills
    }
    skill_id_by_input: dict[str, int] = {}
    for k, rec in skill_lookup.items():
        src = final_skill_by_lower.get(k)
        if src and rec.get("skill_id") is not None:
            skill_id_by_input[src] = int(rec["skill_id"])
    skill_id_by_input_lower: dict[str, int] = {
        s.lower(): sid for s, sid in skill_id_by_input.items()
    }

    new_skill_metas = {
        sd.input_skill: sd.new_skill_meta
        for sd in req.skills_detail
        if sd.source_tag == "llm"
        and sd.new_skill_meta is not None
        and sd.input_skill.lower() not in skill_id_by_input_lower
    }
    skills_created_this_run: set[str] = set()
    skill_enrichment_saved_count = 0
    existing_skill_catalog_backfill_count = 0

    # 3.5) New (canonical-miss) LLM skills: insert canonical + primary alias
    # when v3 meta has locked dimensions and/or merge targets; else artifact only.
    if not _API3_writes_enabled:
        for skill_name, meta in new_skill_metas.items():
            _append_canonical_skill_proposed_artifact(artifacts, skill_name, meta)
    else:
        for sd in req.skills_detail:
            if sd.source_tag != "llm":
                continue
            if sd.input_skill.lower() in skill_id_by_input_lower:
                continue
            meta = sd.new_skill_meta
            if not _has_v3_meta_for_new_skill_persist(meta):
                if meta is not None:
                    _append_canonical_skill_proposed_artifact(
                        artifacts, sd.input_skill, meta
                    )
                continue
            sid, err, enrichment_ok = await asyncio.to_thread(
                _try_create_new_canonical_skill_sync, repo, sd
            )
            if sid is not None:
                skill_id_by_input[sd.input_skill] = sid
                skill_id_by_input_lower[sd.input_skill.lower()] = sid
                skills_created_this_run.add(sd.input_skill.strip().lower())
                persistence.new_skills_created += 1
                if enrichment_ok:
                    skill_enrichment_saved_count += 1
                artifacts.append({
                    "kind": "canonical_skill_added",
                    "artifact_id": sid,
                    "artifact_text": sd.input_skill[:500],
                })
            else:
                if meta is not None:
                    _append_canonical_skill_proposed_artifact(
                        artifacts, sd.input_skill, meta
                    )
                logger.warning(
                    "API3 new canonical skill not created for %r: %s",
                    sd.input_skill,
                    err,
                )

    meta_merge_by_skill_lower: dict[str, dict[str, str]] = {}
    for sd in req.skills_detail:
        key = sd.input_skill.strip().lower()
        if sd.new_skill_meta and sd.new_skill_meta.merge_log:
            meta_merge_by_skill_lower[key] = _catalog_slug_from_merge_log(
                sd.new_skill_meta.merge_log
            )
        else:
            meta_merge_by_skill_lower.setdefault(key, {})

    # 3.6) Existing library skills with zero context tags (skill_tags): run
    # Stage 7 + save. Default OFF — this fires the 5-up enrichment per skill
    # which used to be a quiet cost leak whenever API 3 saw matched-but-
    # not-yet-enriched skills. Re-enable for catalog-backfill batches by
    # setting API3_BACKFILL_LIBRARY_ENRICHMENT=1.
    if _API3_writes_enabled and os.getenv(
        "API3_BACKFILL_LIBRARY_ENRICHMENT", "0"
    ).strip().lower() in ("1", "true", "yes"):
        for sd in req.skills_detail:
            if sd.source_tag != "db" or sd.canonical is None or sd.canonical.id is None:
                continue
            sid = int(sd.canonical.id)
            try:
                missing = await asyncio.to_thread(
                    repo.canonical_skill_missing_catalog_enrichment, sid
                )
            except Exception:
                missing = False
            if not missing:
                continue
            try:
                ok = await _backfill_existing_skill_catalog_enrichment(repo, sd, cost_acc)
            except Exception as exc:
                logger.warning(
                    "API3 existing-skill catalog enrichment backfill failed skill_id=%s: %s",
                    sid,
                    exc,
                )
                ok = False
            if ok:
                existing_skill_catalog_backfill_count += 1
                artifacts.append({
                    "kind": "library_enrichment_backfilled",
                    "artifact_id": sid,
                    "artifact_text": (sd.input_skill or "")[:500],
                })

    # 4) Persist per (skill, dimension):
    # - always attempt skill-dimension for DB-backed skills
    # - persist role-dimension only when row matches chosen role and role exists in DB
    for dd in dimension_worklist:
        skill_name = dd.input_skill
        skill_id = skill_id_by_input_lower.get(skill_name.lower())
        skill_tag = "in_db" if skill_id is not None else "new"

        matched = _role_matches_chosen(dd, chosen)
        dim_id = dd.dimension.id
        dim_slug = dd.dimension.slug
        dim_name = dd.dimension.display_name
        item = PersistenceItem(
            input_skill=skill_name,
            skill_tag=skill_tag,
            skill_id=skill_id,
            dimension=dd.dimension,
            roles_from_db=dd.roles_from_db,
            llm_role=dd.llm_role,
            dimension_id=(int(dim_id) if dim_id is not None else None),
            chosen_role_id=(
                int(chosen_role_out.id)
                if (chosen_role_out is not None and chosen_role_out.id is not None)
                else None
            ),
            matched_chosen_role=matched,
        )

        if skill_id is None:
            meta_sd = next(
                (
                    x
                    for x in req.skills_detail
                    if x.input_skill.strip().lower() == skill_name.strip().lower()
                ),
                None,
            )
            meta = meta_sd.new_skill_meta if meta_sd else None
            if _has_v3_meta_for_new_skill_persist(meta):
                item.skipped_reason = "skill_not_in_db_create_failed"
                item.outcome_line = "Skipped — new skill not created (see logs)"
            else:
                item.skipped_reason = "skill_not_in_db_v3_proposed"
                item.outcome_line = "Skipped — no persistable v3 meta for new skill"
            persistence.skipped += 1
            if dim_id is None:
                artifacts.append({
                    "kind": "dimension_proposed",
                    "artifact_text": (dim_name or "")[:500],
                })
            artifacts.append({
                "kind": "dimension_skill_link_proposed",
                "artifact_text": f"{skill_name} \u2194 {dim_name}"[:500],
            })
            if (
                matched
                and chosen_role_out is not None
                and chosen_role_out.id is not None
            ):
                artifacts.append({
                    "kind": "role_dimension_link_proposed",
                    "artifact_text": (
                        f"{chosen_role_out.display_name} \u2194 {dim_name}"
                    )[:500],
                })
            persistence.items.append(item)
            continue

        if not _API3_writes_enabled:
            item.skipped_reason = (
                "TODO: REMOVE AFTER TESTING — api3_writes_enabled=False (writes disabled)"
            )
            item.outcome_line = "Writes disabled"
            persistence.skipped += 1
            persistence.items.append(item)
            continue

        try:
            merge_map = meta_merge_by_skill_lower.get(skill_name.strip().lower(), {})
            slug_in = (dim_slug or "").strip()
            slug_lc = slug_in.lower()
            merge_val_slugs = {str(v).strip().lower() for v in merge_map.values() if v}
            merge_reconciliation = False
            catalog_slug_match = False
            embedding_dedup = False
            dim_row_created = False

            if dim_id is None:
                recon_id = await asyncio.to_thread(
                    _resolve_dimension_id_from_reconciliation_sync,
                    repo,
                    dimension_slug=dim_slug,
                    merge_slug_map=merge_map,
                )
                if recon_id is not None:
                    dim_id = recon_id
                    if slug_in and _is_pipeline_generated_dim_slug(slug_in) and merge_map.get(slug_lc):
                        merge_reconciliation = True
                    elif slug_in and slug_lc in merge_val_slugs:
                        merge_reconciliation = True
                    elif slug_in and not _is_pipeline_generated_dim_slug(slug_in):
                        catalog_slug_match = True
            if dim_id is None:
                dim_embed_text = (dim_name or "").strip() + " " + (dd.dimension.rationale or "").strip()
                dim_vec = await asyncio.to_thread(_embed_for_skill_library_db, dim_embed_text.strip())
                similar_dims = (
                    await asyncio.to_thread(repo.find_similar_dimensions_by_embedding, dim_vec)
                    if dim_vec is not None
                    else []
                )
                if similar_dims:
                    dedup = await rev_planner.confirm_or_reuse_entity(
                        "dimension",
                        {"display_name": dim_name, "rationale": dd.dimension.rationale},
                        similar_dims,
                    )
                    if dedup.get("action") == "use_existing" and isinstance(dedup.get("existing_id"), int):
                        dim_id = dedup["existing_id"]
                        embedding_dedup = True
                if dim_id is None:
                    dim_row = await asyncio.to_thread(
                        repo.find_or_create_dimension,
                        slug=dim_slug,
                        display_name=dim_name,
                        rationale=dd.dimension.rationale,
                        difficulty_hint=dd.dimension.difficulty_hint,
                        source=dd.dimension.source,
                        name_embedding=dim_vec,
                    )
                    created_id = int(dim_row["id"])
                    dim_row_created = bool(dim_row.get("created"))
                    if dim_row_created:
                        artifacts.append({
                            "kind": "dimension_created",
                            "artifact_id": created_id,
                            "artifact_text": dim_name,
                        })
                    dim_id = created_id
            dim_id = int(dim_id)
            item.dimension_id = dim_id

            sd_inserted = await asyncio.to_thread(
                repo.upsert_dimension_skill_link,
                skill_id=skill_id,
                dimension_id=dim_id,
            )
            if sd_inserted:
                persistence.skill_dimension_saved += 1
                artifacts.append({
                    "kind": "dimension_skill_link",
                    "artifact_id": dim_id,
                    "artifact_text": f"{skill_name} \u2194 {dim_name}",
                })
            item.skill_dimension_saved = True

            if (
                matched
                and chosen_role_out is not None
                and chosen_role_out.id is not None
            ):
                rd_inserted = await asyncio.to_thread(
                    repo.upsert_role_dimension_link,
                    role_id=int(chosen_role_out.id),
                    dimension_id=dim_id,
                )
                if rd_inserted:
                    persistence.role_dimension_saved += 1
                    artifacts.append({
                        "kind": "role_dimension_link",
                        "artifact_id": dim_id,
                        "artifact_text": (
                            f"{chosen_role_out.display_name} \u2194 {dim_name}"
                        ),
                    })
                item.role_dimension_saved = True
            elif matched and (chosen_role_out is None or chosen_role_out.id is None):
                item.skipped_reason = "chosen_role_not_resolved_in_db"

            sk_lower = skill_name.strip().lower()
            skill_bits: list[str] = []
            if sk_lower in skills_created_this_run:
                skill_bits.append("New skill saved")

            if merge_reconciliation:
                dim_bit = "Existing dimension (reconciliation merge)"
            elif embedding_dedup:
                dim_bit = "Existing dimension (embedding dedup)"
            elif dim_row_created:
                dim_bit = "New dimension saved (reconciliation separate)"
            elif catalog_slug_match:
                dim_bit = "Existing dimension (catalog slug)"
            else:
                dim_bit = "Existing dimension (library)"

            if item.role_dimension_saved:
                role_bit = "Role↔dimension saved"
            elif matched and (chosen_role_out is None or chosen_role_out.id is None):
                role_bit = "Role↔dimension not saved — chosen role not in DB (human review)"
            elif not matched:
                role_bit = "Role↔dimension skipped (dimension not under chosen role)"
            else:
                role_bit = "Role↔dimension unchanged (already linked)"

            item.outcome_line = " · ".join([*skill_bits, dim_bit, role_bit])
        except Exception as exc:
            item.skipped_reason = f"db_error: {exc}"
            item.outcome_line = f"Error: {exc}"
            persistence.skipped += 1

        persistence.items.append(item)

    # 5) Planner output when chosen role was missing in DB initially.
    # PlannerAgent is off by default. Set ENABLE_PLANNER_FOR_MISSING_ROLE=1 to re-enable.
    if role_missing_initially and os.getenv(
        "ENABLE_PLANNER_FOR_MISSING_ROLE", ""
    ).strip().lower() in ("1", "true", "yes"):
        planner_output = PlannerGeneratedOutput(
            generated=True,
            role_id=(chosen.slug or "").replace("-", "_"),
            role_display=chosen.display_name,
            payload=None,
        )
        try:
            planner = PlannerAgent()
            planner._accumulator = cost_acc
            plan, hints = await planner.run(
                role_id=planner_output.role_id or "unknown_role",
                role_display=planner_output.role_display or chosen.slug,
            )
            planner_output.payload = {
                "role_archetype": plan.role_archetype,
                "dimensions": [d.model_dump() for d in plan.dimensions],
                "reasoning": plan.reasoning,
                "flagged_for_review": plan.flagged_for_review,
                "web_hints": [h.model_dump() for h in hints],
            }

            # Persist planner-generated dimensions for the chosen role if we have DB role id.
            if (
                chosen_role_out is not None
                and chosen_role_out.id is not None
                and _API3_writes_enabled
            ):
                planner_saved_role_dims = 0
                planner_saved_dim_creates = 0
                for d in plan.dimensions:
                    dim_before = await asyncio.to_thread(
                        repo.find_dimensions_by_names,
                        [d.dimension_name],
                        [d.dimension_id],
                    )
                    existed = bool(dim_before)
                    planner_dim_embed_text = (d.dimension_name or "").strip() + " " + (d.rationale or "").strip()
                    planner_dim_vec = await asyncio.to_thread(_embed_for_skill_library_db, planner_dim_embed_text.strip())
                    final_dim_id: int | None = None
                    if not existed:
                        similar_dims = (
                            await asyncio.to_thread(
                                repo.find_similar_dimensions_by_embedding, planner_dim_vec
                            )
                            if planner_dim_vec is not None
                            else []
                        )
                        if similar_dims:
                            dedup = await rev_planner.confirm_or_reuse_entity(
                                "dimension",
                                {"display_name": d.dimension_name, "rationale": d.rationale},
                                similar_dims,
                            )
                            if dedup.get("action") == "use_existing" and isinstance(dedup.get("existing_id"), int):
                                final_dim_id = dedup["existing_id"]
                                existed = True
                    if final_dim_id is None:
                        dim_row = await asyncio.to_thread(
                            repo.find_or_create_dimension,
                            slug=d.dimension_id,
                            display_name=d.dimension_name,
                            rationale=d.rationale,
                            difficulty_hint=d.difficulty_hint,
                            source="llm",
                            name_embedding=planner_dim_vec,
                        )
                        final_dim_id = int(dim_row["id"])
                    if not existed:
                        planner_saved_dim_creates += 1
                        artifacts.append({
                            "kind": "dimension_created",
                            "artifact_id": int(dim_row["id"]),
                            "artifact_text": d.dimension_name,
                        })
                    rd_ins = await asyncio.to_thread(
                        repo.upsert_role_dimension_link,
                        role_id=int(chosen_role_out.id),
                        dimension_id=final_dim_id,
                    )
                    if rd_ins:
                        planner_saved_role_dims += 1
                        artifacts.append({
                            "kind": "role_dimension_link",
                            "artifact_id": int(dim_row["id"]),
                            "artifact_text": (
                                f"{chosen_role_out.display_name} \u2194 {d.dimension_name}"
                            ),
                        })
                planner_output.saved_dimensions_created = planner_saved_dim_creates
                planner_output.saved_role_dimensions = planner_saved_role_dims
        except Exception as exc:
            logger.warning("planner generation failed for missing role=%r: %s", chosen.slug, exc)
            planner_output.payload = {"error": str(exc)}

    cost_acc.log_summary("skills/final-role-output", logger)
    if chosen_role_out is not None and chosen_role_out.id is not None:
        if role_auto_inserted:
            chosen_role_resolution = "auto_created"
        else:
            chosen_role_resolution = "in_db"
    else:
        chosen_role_resolution = "human_review_required"
    response = FinalRoleOutputResponse(
        chosen_role=chosen_role_out,
        final_input_skills=final_input_skills,
        persistence=persistence,
        planner_output=planner_output,
        run_id=req.run_id,
        chosen_role_resolution=chosen_role_resolution,
    )

    # ── History persistence (only if API 1 minted a run_id) ────
    if history_repo is not None:
        duration_ms = int((time.monotonic() - api3_started_at) * 1000)
        chosen_role_id_for_history: int | None = None
        chosen_role_display_for_history: str | None = None
        if chosen_role_out is not None:
            chosen_role_display_for_history = chosen_role_out.display_name
            if chosen_role_out.id is not None:
                chosen_role_id_for_history = int(chosen_role_out.id)
        await asyncio.to_thread(
            history_repo.attach_api3,
            req.run_id,
            response,
            chosen_role_display=chosen_role_display_for_history,
            chosen_role_id=chosen_role_id_for_history,
            final_skills=req.input_final_skills,
            duration_ms=duration_ms,
            llm_cost_usd=cost_acc.total_cost_usd,
        )
        if artifacts:
            await asyncio.to_thread(
                history_repo.record_artifacts_bulk, req.run_id, artifacts
            )

    logger.info(
        "API3_SUCCESS new_skill_enrichment_saved=%s existing_skill_catalog_backfill=%s run_id=%s",
        skill_enrichment_saved_count,
        existing_skill_catalog_backfill_count,
        req.run_id or "",
    )
    if history_repo is not None and req.run_id:
        costs = await asyncio.to_thread(
            history_repo.get_run_llm_costs_usd, req.run_id
        )
        if costs is not None:
            _log_jd_pipeline_llm_cost_sum(logger, costs, req.run_id)
            response = response.model_copy(
                update=_pipeline_llm_cost_fields_from_tuple(costs)
            )

    return response


@app.get("/skills/runs", response_model=JdRunListResponse)
async def list_jd_pipeline_runs(
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
):
    """Return a paginated history of JD pipeline runs (most recent first)."""
    repo = JdPipelineRunRepository()
    try:
        rows = await asyncio.to_thread(
            repo.list_runs, limit=limit, offset=offset, status=status
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list JD pipeline runs: {exc}",
        ) from exc
    summaries: list[JdRunSummary] = []
    for row in rows:
        # `#>>` returns text — coerce the bool slot back to a Python bool.
        raw_trig = row.pop("v3_pipeline_triggered_raw", None)
        row["v3_pipeline_triggered"] = (
            str(raw_trig).lower() == "true" if raw_trig is not None else False
        )
        row.update(
            _pipeline_llm_cost_fields(
                row.get("llm_cost_api1_usd"),
                row.get("llm_cost_api2_usd"),
                row.get("llm_cost_api3_usd"),
            )
        )
        summaries.append(JdRunSummary(**row))
    return JdRunListResponse(
        runs=summaries,
        limit=max(1, min(int(limit), 200)),
        offset=max(0, int(offset)),
    )


_V3_STAGE_PREFIXES: list[tuple[int, str, str]] = [
    (0, "charter",     "stage0_charter_"),
    (1, "anchor",      "stage1_anchor_"),
    (2, "dim_gen",     "stage2_dim_gen_"),
    (3, "reconciler",  "stage3_recon_"),
    (4, "typing",      "stage4_type_"),
    (5, "placement",   "stage5_place_"),
    (6, "containment", "stage6_contain_"),
    (7, "enrichment",  "stage7_enrich_"),
    (8, "catalog_load","stage8_load_"),
]


def _v3_stage_index(prompt_version: str) -> int | None:
    """Map a v2_run_log.prompt_version like 'stage3_recon_v1.4' to 0..8."""
    pv = (prompt_version or "").lower()
    for idx, _name, prefix in _V3_STAGE_PREFIXES:
        if pv.startswith(prefix):
            return idx
    return None


@app.get("/v3/runs/{role_slug}/status", response_model=V3RunStatus)
async def v3_run_status(role_slug: str):
    """Return a 9-stage timeline for the v3 pipeline of a given role_slug.

    Reads ``v2_run_log`` (which uses ``role_id`` as the role slug for v3
    runs) and maps each row to a stage index by prompt_version prefix.
    Stages that have not started yet appear with ``status='pending'``.
    """
    from jd_similarity_matcher import _pg_connect

    def _load() -> tuple[list[dict[str, Any]], str | None]:
        conn = _pg_connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT prompt_version, status, started_at, completed_at,
                           role_display
                      FROM v2_run_log
                     WHERE role_id = %s
                     ORDER BY started_at ASC
                    """,
                    (role_slug,),
                )
                cols = [c[0] for c in cur.description]
                return (
                    [dict(zip(cols, r)) for r in cur.fetchall()],
                    None,
                )
        finally:
            conn.close()

    try:
        rows, _ = await asyncio.to_thread(_load)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"v3 status load failed: {exc}") from exc

    if not rows:
        # Role slug not found at all — surface that explicitly.
        raise HTTPException(
            status_code=404,
            detail=f"No v3 run rows for role_slug={role_slug!r}",
        )

    # Index rows by stage. Multiple rows per stage are possible (re-runs);
    # the most recent one wins.
    by_idx: dict[int, dict[str, Any]] = {}
    role_display: str | None = None
    for r in rows:
        idx = _v3_stage_index(r.get("prompt_version") or "")
        if idx is None:
            continue
        role_display = role_display or r.get("role_display")
        # ORDER BY started_at ASC means later entries overwrite earlier — that's
        # the desired latest-run-wins behavior.
        by_idx[idx] = r

    stages: list[V3StageRow] = []
    last_updated: datetime | None = None
    for idx, name, _prefix in _V3_STAGE_PREFIXES:
        r = by_idx.get(idx)
        if r is None:
            stages.append(V3StageRow(stage_index=idx, stage_name=name, status="pending"))
            continue
        started = r.get("started_at")
        completed = r.get("completed_at")
        duration_ms = (
            int((completed - started).total_seconds() * 1000)
            if started and completed else None
        )
        if started and (last_updated is None or started > last_updated):
            last_updated = started
        if completed and (last_updated is None or completed > last_updated):
            last_updated = completed
        stages.append(V3StageRow(
            stage_index=idx,
            stage_name=name,
            prompt_version=r.get("prompt_version"),
            status=str(r.get("status") or "pending"),
            started_at=started,
            completed_at=completed,
            duration_ms=duration_ms,
        ))

    # Terminal status: stage 8 winning state, else 'running' if any stage
    # is running, else 'pending'.
    stage8 = stages[8]
    if stage8.status in ("approved", "failed"):
        terminal = stage8.status
    elif any(s.status == "running" for s in stages):
        terminal = "running"
    elif any(s.status == "failed" for s in stages):
        terminal = "failed"
    else:
        terminal = "pending"

    return V3RunStatus(
        role_slug=role_slug,
        role_display=role_display,
        stages=stages,
        terminal_status=terminal,
        last_updated=last_updated,
    )


def _norm_skill_key_hist(s: str | None) -> str:
    return (s or "").strip().lower()


def _build_jd_history_view(
    api1: dict[str, Any] | None,
    api2: dict[str, Any] | None,
    api3: dict[str, Any] | None,
    *,
    api_parser_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shape stored API1–3 JSON into one object for history pages (no DB migration)."""
    api1 = api1 or {}
    api2 = api2 or {}
    api3 = api3 or {}

    persist_tag_by_skill: dict[str, str] = {}
    for x in api3.get("final_input_skills") or []:
        if isinstance(x, dict) and x.get("skill"):
            persist_tag_by_skill[_norm_skill_key_hist(str(x["skill"]))] = str(
                x.get("tag") or ""
            )

    detail_by_skill: dict[str, dict[str, Any]] = {}
    for sd in api2.get("skills_detail") or []:
        if isinstance(sd, dict) and sd.get("input_skill"):
            detail_by_skill[_norm_skill_key_hist(str(sd["input_skill"]))] = sd

    pers = api3.get("persistence")
    raw_items: list[dict[str, Any]] = []
    if isinstance(pers, dict):
        for it in pers.get("items") or []:
            if isinstance(it, dict):
                raw_items.append(it)

    persist_rows_by_skill: dict[str, list[dict[str, Any]]] = {}
    for it in raw_items:
        sk = str(it.get("input_skill") or "")
        nk = _norm_skill_key_hist(sk)
        if nk:
            persist_rows_by_skill.setdefault(nk, []).append(it)

    skills_out: list[dict[str, Any]] = []
    for fs in api1.get("final_skills") or []:
        if not isinstance(fs, dict):
            continue
        name = str(fs.get("skill_name") or "").strip()
        if not name:
            continue
        nk = _norm_skill_key_hist(name)
        sd = detail_by_skill.get(nk, {})
        source_tag = str(sd.get("source_tag") or "")
        canonical = sd.get("canonical") if isinstance(sd.get("canonical"), dict) else None
        meta = sd.get("new_skill_meta") if isinstance(sd.get("new_skill_meta"), dict) else None

        if not sd:
            skill_row_kind = "incomplete"
            skill_row_label = "No API 2 row (run stopped after API 1 or history missing)"
        elif source_tag == "llm":
            skill_row_kind = "new_skill"
            skill_row_label = "New / unmatched skill (orchestrated in API 2)"
        elif source_tag == "unmatched":
            skill_row_kind = "unmatched"
            skill_row_label = "Unmatched (no library row)"
        else:
            skill_row_kind = "library_skill"
            skill_row_label = "Existing skill (matched library)"

        dims_out: list[dict[str, Any]] = []
        for dd in sd.get("dimensions") or []:
            if not isinstance(dd, dict):
                continue
            dim = dd.get("dimension") if isinstance(dd.get("dimension"), dict) else {}
            did = dim.get("id")
            src = str(dim.get("source") or "").lower()
            if src == "db" and did is not None:
                dim_kind = "library"
                dim_kind_label = "Library dimension (catalog)"
            elif src == "llm" or did is None:
                dim_kind = "proposed"
                dim_kind_label = "Proposed / LLM dimension (no DB id yet)"
            else:
                dim_kind = "other"
                dim_kind_label = "Dimension"
            dims_out.append(
                {
                    "display_name": dim.get("display_name"),
                    "slug": dim.get("slug"),
                    "source": dim.get("source"),
                    "id": did,
                    "dim_kind": dim_kind,
                    "dim_kind_label": dim_kind_label,
                    "roles_from_db": [
                        str(r.get("display_name") or "")
                        for r in (dd.get("roles_from_db") or [])
                        if isinstance(r, dict)
                    ],
                }
            )

        locked_out: list[dict[str, Any]] = []
        if meta:
            for ld in meta.get("locked_dimensions") or []:
                if not isinstance(ld, dict):
                    continue
                tid = str(ld.get("tentative_id") or "")
                locked_out.append(
                    {
                        "tentative_id": tid,
                        "name": ld.get("name"),
                        "description": (str(ld.get("description") or ""))[:500],
                        "placement_note": (
                            "Reuses catalog slug"
                            if tid and not tid.startswith(("d_init_", "d_merge_", "d_split_"))
                            else "Pipeline tentative id"
                        ),
                    }
                )

        persist_tag = persist_tag_by_skill.get(nk, "")
        link_rows: list[dict[str, Any]] = []
        for it in persist_rows_by_skill.get(nk, []):
            dim = it.get("dimension") if isinstance(it.get("dimension"), dict) else {}
            link_rows.append(
                {
                    "dimension_name": dim.get("display_name"),
                    "dimension_slug": dim.get("slug"),
                    "dimension_id": dim.get("id"),
                    "skill_dimension_saved": bool(it.get("skill_dimension_saved")),
                    "role_dimension_saved": bool(it.get("role_dimension_saved")),
                    "outcome_line": it.get("outcome_line"),
                    "skipped_reason": it.get("skipped_reason"),
                }
            )

        # Skill enrichment: library canonical profile vs v3 orchestrator enrichment.
        enrichment_source: str | None = None
        library_canonical_profile: dict[str, Any] | None = None
        orchestrator_enrichment: dict[str, Any] | None = None
        orchestrator_typed: dict[str, Any] | None = None
        orchestrator_derived: dict[str, Any] | None = None
        if source_tag == "db" and canonical:
            enrichment_source = "library"
            library_canonical_profile = {
                "id": canonical.get("id"),
                "slug": canonical.get("slug"),
                "display_name": canonical.get("display_name"),
                "category_id": canonical.get("category_id"),
                "sub_category_id": canonical.get("sub_category_id"),
                "skill_nature": canonical.get("skill_nature"),
                "volatility": canonical.get("volatility"),
                "typical_lifespan": canonical.get("typical_lifespan"),
                "is_extractable": canonical.get("is_extractable"),
                "is_also_category": canonical.get("is_also_category"),
            }
        elif source_tag == "llm":
            enrichment_source = "orchestrator"
            enr = meta.get("enrichment") if isinstance(meta, dict) else None
            if isinstance(enr, dict):
                orchestrator_enrichment = enr
            ts = meta.get("typed") if isinstance(meta, dict) else None
            if isinstance(ts, dict):
                orchestrator_typed = ts
            dv = meta.get("derived") if isinstance(meta, dict) else None
            if isinstance(dv, dict):
                orchestrator_derived = dv

        library_aliases_snapshot: list[dict[str, Any]] = []
        for a in sd.get("aliases_in_db") or []:
            if not isinstance(a, dict):
                continue
            library_aliases_snapshot.append(
                {
                    "alias_text": a.get("alias_text"),
                    "alias_type": a.get("alias_type"),
                    "is_primary": a.get("is_primary"),
                    "match_strategy": a.get("match_strategy"),
                }
            )

        skills_out.append(
            {
                "skill_name": name,
                "is_primary": bool(fs.get("is_primary")),
                "source_tag": source_tag,
                "skill_row_kind": skill_row_kind,
                "skill_row_label": skill_row_label,
                "persist_tag": persist_tag,
                "persist_tag_label": (
                    "API 3: existing canonical (in_db)"
                    if persist_tag == "in_db"
                    else (
                        "API 3: new canonical path (new)"
                        if persist_tag == "new"
                        else "API 3 not run or tag missing"
                    )
                ),
                "canonical_id": (canonical or {}).get("id"),
                "canonical_display_name": (canonical or {}).get("display_name"),
                "canonical_slug": (canonical or {}).get("slug"),
                "dimensions": dims_out,
                "locked_dimensions": locked_out,
                "has_new_skill_meta": meta is not None,
                "api3_link_rows": link_rows,
                "enrichment_source": enrichment_source,
                "library_canonical_profile": library_canonical_profile,
                "orchestrator_enrichment": orchestrator_enrichment,
                "orchestrator_typed": orchestrator_typed,
                "orchestrator_derived": orchestrator_derived,
                "library_aliases_snapshot": library_aliases_snapshot,
            }
        )

    cr = api3.get("chosen_role") or api2.get("chosen_role")
    chosen = cr if isinstance(cr, dict) else None

    pers_summary: dict[str, Any] = {}
    if isinstance(pers, dict):
        pers_summary = {
            "new_skills_created": pers.get("new_skills_created", 0),
            "skill_dimension_saved": pers.get("skill_dimension_saved", 0),
            "role_dimension_saved": pers.get("role_dimension_saved", 0),
            "skipped": pers.get("skipped", 0),
        }

    return {
        "pipeline": {
            "api1": "POST /skills/extract-from-jd",
            "api2": "POST /skills/extract-details",
            "api3": "POST /skills/final-role-output",
        },
        "api_parser_response": api_parser_response,
        "chosen_role": chosen,
        "chosen_role_resolution": api3.get("chosen_role_resolution"),
        "persistence_summary": pers_summary,
        "persistence_item_count": len(raw_items),
        "persistence_items": raw_items,
        "skills": skills_out,
    }


def _merge_history_library_skill_details(
    history_view: dict[str, Any],
    repo: SkillLibraryRepository,
) -> None:
    """Attach live catalog aliases, tags, and enrichment columns for library-matched skills."""
    skills = history_view.get("skills") or []
    for s in skills:
        if str(s.get("source_tag") or "") != "db":
            continue
        cid = s.get("canonical_id")
        if cid is None:
            continue
        try:
            detail = repo.get_skill_detail(int(cid))
        except Exception as exc:
            logger.debug(
                "history merge skipped for canonical_id=%s: %s",
                cid,
                exc,
            )
            continue
        if not detail:
            continue
        s["library_db_aliases"] = detail.get("aliases") or []
        s["library_db_tags"] = detail.get("tags") or []
        s["library_db_enrichment"] = {
            "vendor": detail.get("vendor"),
            "license": detail.get("license"),
            "year_introduced": detail.get("year_introduced"),
            "maturity_reasoning": detail.get("maturity_reasoning"),
            "confidence": detail.get("confidence"),
            "version_tag": detail.get("version_tag"),
            "version_strategy": detail.get("version_strategy"),
            "category_display": detail.get("category_display"),
            "sub_category_display": detail.get("sub_category_display"),
        }


@app.get("/skills/runs/{run_id}", response_model=JdRunDetail)
async def get_jd_pipeline_run(run_id: str):
    """Return the full row for one run, including artifacts."""
    repo = JdPipelineRunRepository()
    try:
        row = await asyncio.to_thread(repo.get_run, run_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch JD pipeline run: {exc}",
        ) from exc
    if not row:
        raise HTTPException(status_code=404, detail="run_id not found")
    artifacts_raw = row.pop("artifacts", []) or []
    artifacts = [JdRunArtifact(**a) for a in artifacts_raw]
    history_view = _build_jd_history_view(
        row.get("api1_response"),
        row.get("api2_response"),
        row.get("api3_response"),
        api_parser_response=row.get("api_parser_response"),
    )
    lib_repo = SkillLibraryRepository()
    await asyncio.to_thread(_merge_history_library_skill_details, history_view, lib_repo)
    row.update(
        _pipeline_llm_cost_fields(
            row.get("llm_cost_api1_usd"),
            row.get("llm_cost_api2_usd"),
            row.get("llm_cost_api3_usd"),
        )
    )
    return JdRunDetail(**row, artifacts=artifacts, history_view=history_view)


@app.post("/extract-pdf-links", response_model=PdfLinksResponse)
async def extract_pdf_links_endpoint(file: UploadFile = File(...)):
    """Return hyperlink targets (URI annotations) from a PDF upload."""
    filename = (file.filename or "").lower()
    if not filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported for link extraction.",
        )
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    links = await asyncio.to_thread(extract_pdf_links, raw)
    return PdfLinksResponse(links=[PdfLinkItem(**item) for item in links])


def _resolve_hybrid_raw_text(
    raw_plain: str,
    docling_markdown: str,
    *,
    filename: str,
) -> str:
    """Prefer python-docx/pdf text; if empty, use Docling markdown (table-heavy DOCX)."""
    plain = (raw_plain or "").strip()
    md = (docling_markdown or "").strip()
    if plain:
        return plain
    if md:
        logger.info(
            "Hybrid: empty plain-text parse for %s; using Docling markdown as raw_text.",
            filename or "upload",
        )
        return md
    lower = (filename or "").lower()
    if lower.endswith((".docx", ".doc")):
        detail = "Could not extract text from DOCX."
    elif lower.endswith(".pdf"):
        detail = "Could not extract text from PDF."
    else:
        detail = "Could not extract text from document."
    raise HTTPException(status_code=422, detail=detail)


# ── NEW: Resume upload & contextual meta-tagging endpoint ──
@app.post("/parse-resume", response_model=ResumeTaggingResponse)
async def parse_resume_endpoint(file: UploadFile = File(...)):
    """Full resume parsing pipeline:

    1. Parse file → raw text (PDF/DOCX/TXT)
    2. Extract candidate details (name, email, phone, LinkedIn, GitHub)
    3. Zone resume into blocks (Header, Summary, Experience_0..N, Project_0..N,
       Skills_Dump, Education) — experience & project sections sub-split
       into per-role/per-project blocks using date patterns
    4. NLP Filter: POS-tag Summary; reclassify as Skills_Dump if verb_count < 3
    5. Block Tagging: Single LLM call extracts per-block:
       - Skills with action_verb, context, co_dependent_skills, metrics
       - Experience details: company, role, duration, achievements, quantifiers, tech
       - Project details: name, description, tech_stack, quantifiers, highlights
    6. Global Parameter Extraction: 7 taxonomy parameters via LLM
    7. Context Meta-Tag Extraction: Builds summary/experience/project meta-tags,
       categorizes skills, generates resume strength signals
    8. Aggregate into final JSON with trajectory-based global_skill_index
    """
    # Step 1: Parse
    t0 = time.perf_counter()
    raw_text = await extract_text(file)
    logger.info("Step 1 (parse) took %.2fs", time.perf_counter() - t0)

    # Step 2 + 3: Candidate details & Zone resume (both need only raw_text)
    t1 = time.perf_counter()
    candidate = extract_candidate_details(raw_text)
    blocks = zone_resume(raw_text)
    logger.info("Steps 2+3 (candidate+zone) took %.2fs",
                time.perf_counter() - t1)

    # Step 4: NLP Filter
    t2 = time.perf_counter()
    blocks = await asyncio.to_thread(filter_blocks, blocks)
    logger.info("Step 4 (NLP filter) took %.2fs", time.perf_counter() - t2)

    # Steps 5+6+7 PARALLEL:
    #   Chain A: tag_all_blocks → extract_context_meta_tags (step 5 → step 7)
    #   Chain B: extract_global_parameters (step 6)
    # Step 7 starts as soon as step 5 finishes, while step 6 may still run.
    t3 = time.perf_counter()

    async def _tag_and_build_meta(blks):
        t5 = time.perf_counter()
        br = await tag_all_blocks(blks)
        logger.info("  Step 5 (block tagging) took %.2fs",
                    time.perf_counter() - t5)
        t7 = time.perf_counter()
        cmt = await extract_context_meta_tags(br)
        logger.info("  Step 7 (meta-tags) took %.2fs",
                    time.perf_counter() - t7)
        return br, cmt

    (block_results, context_meta_tags), (global_params, reasoning_log) = (
        await asyncio.gather(
            _tag_and_build_meta(blocks),
            extract_global_parameters(blocks),
        )
    )
    logger.info("Steps 5+6+7 (parallel) took %.2fs", time.perf_counter() - t3)

    # Step 8: Aggregate (include parsed text and zoned blocks for frontend debug)
    response = aggregate(
        candidate, block_results, global_params,
        context_meta_tags, reasoning_log,
        parsed_text=raw_text,
        zoned_blocks=blocks,
    )
    logger.info("Total pipeline took %.2fs", time.perf_counter() - t0)

    return response


@app.post("/parse-resume-hybrid", response_model=ResumeTaggingResponse)
async def parse_resume_hybrid_endpoint(file: UploadFile = File(...)):
    """Hybrid pipeline:
    - normal parser text + docling markdown in parallel
    - one LLM call builds blocks compatible with the existing pipeline
    - then reuse the same downstream workflow as /parse-resume
    """
    t0 = time.perf_counter()

    # We need bytes for both parsers, so read once.
    raw_bytes = await file.read()
    filename = (file.filename or "upload").lower()

    # Run both parsers concurrently to save time:
    # - normal parser -> raw_text
    # - docling -> docling_markdown
    # - pdf links -> extracted_links (pdf only)
    #
    # These are CPU-heavy/sync operations, so we run them in threads.
    from resume_parser import _parse_pdf

    def _normal_parse() -> str:
        if filename.endswith(".pdf"):
            return _parse_pdf(raw_bytes)
        if filename.endswith(".docx") or filename.endswith(".doc"):
            return extract_docx_plain_text(raw_bytes)
        # txt and fallback
        return raw_bytes.decode("utf-8", errors="replace")

    def _docling_parse() -> str:
        try:
            from docling_client import convert_path_to_markdown

            # Write to temp so Docling can infer format.
            suffix = Path(filename).suffix or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name
            try:
                return convert_path_to_markdown(tmp_path, device="cpu")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Docling conversion failed in hybrid flow: %s", e)
            return ""

    parse_tasks = [
        asyncio.to_thread(_normal_parse),
        asyncio.to_thread(_docling_parse),
    ]
    wants_pdf_links = filename.endswith(".pdf")
    if wants_pdf_links:
        parse_tasks.append(asyncio.to_thread(extract_pdf_links, raw_bytes))

    parse_results = await asyncio.gather(*parse_tasks)
    docling_markdown = parse_results[1]
    extracted_links = parse_results[2] if wants_pdf_links else []
    raw_text = _resolve_hybrid_raw_text(
        parse_results[0],
        docling_markdown,
        filename=filename,
    )

    # Build blocks via LLM using both texts (fallback inside builder if needed).
    blocks = await build_hybrid_blocks(
        raw_text,
        docling_markdown,
    )

    # Same deterministic filter + tagging/meta/global flow as default.
    blocks = await asyncio.to_thread(filter_blocks, blocks)

    # Candidate regex on Header block matches zoned preview; raw_text alone can reorder lines.
    candidate = extract_candidate_details_for_hybrid(blocks, raw_text)

    async def _tag_and_build_meta(blks):
        br = await tag_all_blocks(blks, normalize_skills_per_block=True)
        cmt = await extract_context_meta_tags(br)
        return br, cmt

    (block_results, context_meta_tags), (global_params, reasoning_log) = (
        await asyncio.gather(
            _tag_and_build_meta(blocks),
            extract_global_parameters(blocks),
        )
    )

    response = aggregate(
        candidate,
        block_results,
        global_params,
        context_meta_tags,
        reasoning_log,
        parsed_text=raw_text,
        zoned_blocks=blocks,
        extracted_links=extracted_links,
        enrich_skill_temporal=True,
    )

    logger.info("Hybrid pipeline total took %.2fs", time.perf_counter() - t0)
    return response


@app.post("/parse-resume-hybrid-stage1", response_model=HybridParseStage1Response)
async def parse_resume_hybrid_stage1_endpoint(file: UploadFile = File(...)):
    """Stage 1 of hybrid flow: parse + extracted links + hybrid blocks.

    Returns the same content your UI shows in Parsed Text / Zoned Blocks,
    without running downstream tagging/meta/global extraction.
    """
    t0 = time.perf_counter()
    raw_bytes = await file.read()
    filename = (file.filename or "upload").lower()

    from resume_parser import _parse_pdf

    def _normal_parse() -> str:
        if filename.endswith(".pdf"):
            return _parse_pdf(raw_bytes)
        if filename.endswith(".docx") or filename.endswith(".doc"):
            return extract_docx_plain_text(raw_bytes)
        return raw_bytes.decode("utf-8", errors="replace")

    def _docling_parse() -> str:
        try:
            from docling_client import convert_path_to_markdown

            suffix = Path(filename).suffix or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name
            try:
                return convert_path_to_markdown(tmp_path, device="cpu")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Docling conversion failed in hybrid stage1: %s", e)
            return ""

    parse_tasks = [
        asyncio.to_thread(_normal_parse),
        asyncio.to_thread(_docling_parse),
    ]
    wants_pdf_links = filename.endswith(".pdf")
    if wants_pdf_links:
        parse_tasks.append(asyncio.to_thread(extract_pdf_links, raw_bytes))

    parse_results = await asyncio.gather(*parse_tasks)
    docling_markdown = parse_results[1]
    extracted_links_raw = parse_results[2] if wants_pdf_links else []
    raw_text = _resolve_hybrid_raw_text(
        parse_results[0],
        docling_markdown,
        filename=filename,
    )

    blocks = await build_hybrid_blocks(raw_text, docling_markdown)
    candidate = extract_candidate_details_for_hybrid(blocks, raw_text)

    zoned_preview = [
        ZonedBlockPreview(
            block_name=b.block_name,
            block_type=b.block_type,
            raw_text=b.raw_text,
        )
        for b in (blocks or [])
    ]

    resp = HybridParseStage1Response(
        candidate=candidate,
        extracted_links=[ExtractedLink(**x) for x in extracted_links_raw],
        parsed_text=raw_text,
        zoned_blocks_preview=zoned_preview,
    )
    logger.info("Hybrid stage1 total took %.2fs", time.perf_counter() - t0)
    return resp


@app.post("/parse-resume-hybrid-stage2", response_model=ResumeTaggingResponse)
async def parse_resume_hybrid_stage2_endpoint(
    payload: HybridParseStage1Response,
):
    """Stage 2 of hybrid flow: run the downstream pipeline after blocks are built.

    Input payload should be exactly the response from:
    - `/parse-resume-hybrid-stage1`
    """
    t0 = time.perf_counter()

    # Rebuild the internal ResumeBlock objects from the stage1 preview.
    stage1_blocks: list[ResumeBlock] = [
        ResumeBlock(
            block_name=z.block_name,
            block_type=z.block_type,
            raw_text=z.raw_text,
        )
        for z in (payload.zoned_blocks_preview or [])
    ]

    # Deterministic filter (may reclassify "summary" -> "skills_dump")
    blocks = await asyncio.to_thread(filter_blocks, stage1_blocks)

    async def _tag_and_build_meta(blks):
        br = await tag_all_blocks(blks, normalize_skills_per_block=True)
        cmt = await extract_context_meta_tags(br)
        return br, cmt

    (block_results, context_meta_tags), (global_params, reasoning_log) = (
        await asyncio.gather(
            _tag_and_build_meta(blocks),
            extract_global_parameters(blocks),
        )
    )

    response = aggregate(
        payload.candidate,
        block_results,
        global_params,
        context_meta_tags,
        reasoning_log,
        parsed_text=payload.parsed_text,
        zoned_blocks=stage1_blocks,  # keep stage1 preview in UI/debug
        extracted_links=payload.extracted_links,
        enrich_skill_temporal=True,
    )

    logger.info("Hybrid stage2 total took %.2fs", time.perf_counter() - t0)
    return response


# ═══════════════════════════════════════════════════════════════════════════════
# RESUME RANKING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


# ── Helper: full resume pipeline reused internally ────────────────────────────
async def _run_full_resume_pipeline(file: UploadFile) -> ResumeTaggingResponse:
    """Parse resume via local hybrid pipeline (normal text + Docling in parallel)."""
    raw_bytes = await file.read()
    filename = file.filename or "resume.pdf"

    from resume_parser import _parse_pdf

    fname_lower = filename.lower()

    def _normal_parse() -> str:
        if fname_lower.endswith(".pdf"):
            return _parse_pdf(raw_bytes)
        if fname_lower.endswith(".docx") or fname_lower.endswith(".doc"):
            return extract_docx_plain_text(raw_bytes)
        return raw_bytes.decode("utf-8", errors="replace")

    def _docling_parse() -> str:
        try:
            from docling_client import convert_path_to_markdown
            suffix = Path(filename).suffix or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name
            try:
                return convert_path_to_markdown(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(
                "Docling failed in ranking fallback pipeline: %s", e)
            return ""

    plain, docling_md = await asyncio.gather(
        asyncio.to_thread(_normal_parse),
        asyncio.to_thread(_docling_parse),
    )
    raw_text = _resolve_hybrid_raw_text(
        plain,
        docling_md,
        filename=filename,
    )

    blocks = await build_hybrid_blocks(raw_text, docling_md)
    blocks = await asyncio.to_thread(filter_blocks, blocks)
    candidate = extract_candidate_details_for_hybrid(blocks, raw_text)

    async def _tag_and_meta(blks):
        br = await tag_all_blocks(blks, normalize_skills_per_block=True)
        cmt = await extract_context_meta_tags(br)
        return br, cmt

    (block_results, context_meta_tags), (global_params, reasoning_log) = (
        await asyncio.gather(
            _tag_and_meta(blocks),
            extract_global_parameters(blocks),
        )
    )

    return aggregate(
        candidate, block_results, global_params,
        context_meta_tags, reasoning_log,
        parsed_text=raw_text, zoned_blocks=blocks,
    )


# ── 1. Parse & classify a Job Description ────────────────────────────────────
@app.post("/ranking/parse-jd", response_model=JDProfile)
async def ranking_parse_jd(req: ParseJDRequest):
    """Parse a raw JD text into structured capability atoms + SetFit tier classification.

    Returns a JDProfile that can be stored client-side and passed to
    /ranking/score-resume or /ranking/batch-rank.
    """
    if not req.jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text cannot be empty")

    # Input validation: detect if resume text was accidentally sent as JD
    import re as _re
    _resume_signals = [
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # email
        # 10-digit phone
        r"\b\d{10}\b",
        r"linkedin\.com/in/",                                   # linkedin
        r"github\.com/",                                        # github
    ]
    signal_count = sum(
        1 for pat in _resume_signals
        if _re.search(pat, req.jd_text[:500], _re.IGNORECASE)
    )
    if signal_count >= 2:
        logger.warning(
            "⚠️  /ranking/parse-jd received text that looks like a resume "
            "(contains %d personal-info signals in first 500 chars). "
            "Proceeding, but accuracy may be degraded.",
            signal_count,
        )

    t0 = time.perf_counter()
    profile = await parse_jd(req.jd_text)
    logger.info("JD parsing took %.2fs", time.perf_counter() - t0)
    return profile


# ── 2. Score a single resume against a pre-parsed JD ─────────────────────────
@app.post("/ranking/score-resume", response_model=ResumeRankResult)
async def ranking_score_resume(
    jd_profile: ScoreResumeRequest,
    file: UploadFile = File(...),
):
    """Score a single uploaded resume against a pre-parsed JDProfile.

    Returns a ResumeRankResult with full 6-dimension score breakdown.
    """
    t0 = time.perf_counter()
    resume = await _run_full_resume_pipeline(file)

    # Classify resume with LLM
    resume_text = resume.parsed_text[:2000] if resume.parsed_text else ""
    resume_lc = await _llm_classify_text(resume_text)
    resume_tier = resume_lc.to_legacy_tier()

    # Run LLM section scoring
    candidate_years = estimate_candidate_years(resume)
    section_scores = await score_all_sections(
        jd_profile.jd_profile, resume, candidate_years)

    result = score_resume(jd_profile.jd_profile, resume,
                          resume_tier, resume_lc=resume_lc,
                          section_scores=section_scores)
    logger.info("Single resume scoring took %.2fs", time.perf_counter() - t0)
    return result


# ── 3. Batch rank multiple resumes against a JD ───────────────────────────────
@app.post("/ranking/batch-rank", response_model=RankingSession)
async def ranking_batch_rank(
    jd_text: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """Parse JD once, then score all uploaded resumes in parallel and rank them.

    Returns a RankingSession with all results sorted by final_score descending.
    """
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text cannot be empty")
    if not files:
        raise HTTPException(
            status_code=400, detail="At least one resume file required")
    if len(files) > 30:
        raise HTTPException(
            status_code=400, detail="Maximum 30 resumes per batch")

    t0 = time.perf_counter()

    # Step 1: Parse JD (single LLM call)
    jd_profile = await parse_jd(jd_text)
    logger.info("JD parsed in %.2fs", time.perf_counter() - t0)

    # Step 2: Process all resumes in parallel
    sem = asyncio.Semaphore(_BATCH_RESUME_CONCURRENCY)

    async def _score_one(f: UploadFile) -> ResumeRankResult | str:
        try:
            async with sem:
                resume = await _run_full_resume_pipeline(f)
                resume_text = resume.parsed_text[:
                                                 2000] if resume.parsed_text else ""
                resume_lc = await _llm_classify_text(resume_text)
                resume_tier = resume_lc.to_legacy_tier()
                # Run LLM section scoring in parallel
                candidate_years = estimate_candidate_years(resume)
                section_scores = await score_all_sections(
                    jd_profile, resume, candidate_years)
                return score_resume(
                    jd_profile, resume, resume_tier,
                    resume_lc=resume_lc,
                    section_scores=section_scores,
                )
        except Exception as e:
            fname = getattr(f, "filename", "unknown")
            logger.exception("Error scoring resume %s: %s", fname, e)
            return f"Error processing {fname}: {e}"

    outcomes = await asyncio.gather(*[_score_one(f) for f in files])

    results: list[ResumeRankResult] = []
    errors: list[str] = []
    for o in outcomes:
        if isinstance(o, ResumeRankResult):
            results.append(o)
        else:
            errors.append(str(o))

    # Sort by final_score descending, assign ranks
    results.sort(key=lambda r: r.final_score, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    logger.info(
        "Batch rank: %d resumes, %.2fs total. Errors: %d",
        len(results), time.perf_counter() - t0, len(errors),
    )

    return RankingSession(
        jd_profile=jd_profile,
        results=results,
        total_resumes=len(files),
        processing_errors=errors,
    )


# ── 4. Expose SetFit classification directly ──────────────────────────────────
@app.post("/classify-text", response_model=TierClassification)
async def classify_text_endpoint(req: ClassifyTextRequest):
    """Run LLM-based classification on arbitrary text.

    Useful for debugging classification of JDs and resumes.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    lc = await _llm_classify_text(req.text[:3000])
    return lc.to_legacy_tier()


# ── 5. Resume Ranking Test UI ─────────────────────────────────────────────────
@app.get("/resume-ranking", response_class=HTMLResponse)
async def resume_ranking_ui(request: Request):
    """Jinja2 test page for the resume ranking system."""
    return templates.TemplateResponse(
        request,
        "resume_ranking.html",
        {
            "setfit_loaded": False,
            "loaded_tiers": [],
        },
    )


# ── 6. Skill Library Panel ───────────────────────────────────────────────────

class AdminSkillPayload(BaseModel):
    alias: str = Field(..., min_length=1, max_length=200)
    canonical: str = Field(..., min_length=1, max_length=200)
    category: str = Field(default="Other", max_length=100)


@app.get("/skill-library", response_class=HTMLResponse)
async def skill_library_ui(request: Request):
    """Jinja2 panel for viewing and managing the skill library."""
    return templates.TemplateResponse(request, "skill_library.html", {})


@app.get("/api/skill-library")
async def api_skill_library(
    q: str = "",
    category: str = "",
    source: str = "",
    skip: int = 0,
    limit: int = 50,
):
    """Search / list learned skills for the admin panel."""
    limit = min(limit, 200)
    skills, total = await db.search_learned_skills(
        query=q, category=category, source=source, skip=skip, limit=limit,
    )

    # Compute quick counts for the stats bar
    _, llm_total = await db.search_learned_skills(source="llm", limit=1)
    _, admin_total = await db.search_learned_skills(source="admin", limit=1)
    _, static_total = await db.search_learned_skills(source="static", limit=1)
    categories = await db.get_skill_categories()

    # Serialize datetimes to ISO strings
    for s in skills:
        for key in ("first_seen", "last_seen"):
            if key in s and s[key] is not None:
                s[key] = s[key].isoformat()

    return {
        "skills": skills,
        "total": total,
        "static_count": static_total,
        "llm_count": llm_total,
        "admin_count": admin_total,
        "category_count": len(categories),
    }


@app.get("/api/skill-library/categories")
async def api_skill_categories():
    """Return distinct categories from the skill library."""
    return await db.get_skill_categories()


@app.post("/api/skill-library", status_code=201)
async def api_add_skill(payload: AdminSkillPayload):
    """Admin-add a skill to the library."""
    await skill_library.learn(
        alias=payload.alias,
        canonical=payload.canonical,
        confidence=1.0,
        category=payload.category,
        source="admin",
    )
    return {"ok": True, "alias": payload.alias.strip().lower(), "canonical": payload.canonical}


@app.delete("/api/skill-library/{alias}")
async def api_delete_skill(alias: str):
    """Delete a learned skill from the library."""
    deleted = await db.delete_learned_skill(alias)
    if not deleted:
        raise HTTPException(
            status_code=404, detail="Skill not found or DB unavailable")
    # Also remove from in-memory cache
    merged = skill_library.get_aliases()
    merged.pop(alias.strip().lower(), None)
    return {"ok": True}
