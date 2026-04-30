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
from skill_library_v2.agents.planner import PlannerAgent
from pathlib import Path
import tempfile
import time
import logging
import asyncio
from typing import Any
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from model_azure import download_models_from_azure, upload_models_to_azure
from model_pipeline import router as model_pipeline_router

# Suppress noisy third-party progress bars BEFORE any imports that pull in tqdm
import os
os.environ["TQDM_DISABLE"] = "1"


# ── Ranking system imports ────────────────────────────────────────────────────


logger = logging.getLogger(__name__)

_BATCH_RESUME_CONCURRENCY = max(
    1, int(os.getenv("RANKING_BATCH_CONCURRENCY", "4"))
)


load_dotenv()

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

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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


class JDSkillPipelineResponse(BaseModel):
    initial_skills: list[str]
    unknown_words: list[str]
    filtered_unknown_words: list[str]
    llm_skills: list[str]
    llm_non_skills: list[str]
    final_skills: list[str]
    final_non_skills: list[str]


# ── Skill Detail / Reverse Planner models ────────────────────────────────────
class ExtractDetailsRequest(BaseModel):
    final_skills: list[str] = Field(default_factory=list)
    llm_skills: list[str] = Field(default_factory=list)


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


class NewSkillMeta(BaseModel):
    """LLM-inferred metadata for a skill not yet in the canonical library."""
    category: str | None = None
    sub_category: str | None = None
    skill_nature: str | None = None
    typical_lifespan: str | None = None


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
    matched_via: str | None = None  # "alias" | "display_name"
    aliases_in_db: list[AliasInfo] = Field(default_factory=list)
    new_alias_persisted: bool = False
    new_alias_text: str | None = None
    dimensions: list[DimensionDetail] = Field(default_factory=list)
    new_skill_meta: NewSkillMeta | None = None


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


@app.post("/skills/extract-from-jd", response_model=JDSkillPipelineResponse)
async def extract_skills_from_jd_endpoint(req: JDSkillPipelineRequest):
    jd_text = (req.jd_text or "").strip()
    if not jd_text:
        raise HTTPException(status_code=400, detail="jd_text cannot be empty")

    stage1 = await asyncio.to_thread(process_jd, jd_text)
    initial_skills = _clean_strings(stage1.get("skills"))
    unknown_words = _clean_strings(stage1.get("unknown_words"))

    repo = NonSkillRepository()
    filtered_unknown_words = await asyncio.to_thread(
        repo.filter_non_skills, unknown_words
    )

    llm_skills: list[str] = []
    llm_non_skills: list[str] = []
    if filtered_unknown_words:
        classifier = AzureUnknownWordClassifier()
        try:
            llm_result = await classifier.classify_words(filtered_unknown_words)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Stage-2 LLM classification failed: {exc}",
            ) from exc

        llm_skills = _clean_strings(llm_result.get("skills"))
        llm_non_skills = _clean_strings(llm_result.get("non_skills"))

        # Guardrail: deployments is an activity-domain skill for downstream
        # reverse-planning and should never be persisted as non-skill.
        if "deployments" in {w.lower() for w in filtered_unknown_words}:
            skills_lc = {w.lower() for w in llm_skills}
            llm_non_skills = [w for w in llm_non_skills if w.lower() != "deployments"]
            if "deployments" not in skills_lc:
                llm_skills.append("deployments")

        await asyncio.to_thread(repo.add_non_skills, llm_non_skills)

    final_skills = _dedupe_case_insensitive(initial_skills + llm_skills)
    final_non_skills = _dedupe_case_insensitive(llm_non_skills)

    return JDSkillPipelineResponse(
        initial_skills=initial_skills,
        unknown_words=unknown_words,
        filtered_unknown_words=filtered_unknown_words,
        llm_skills=llm_skills,
        llm_non_skills=llm_non_skills,
        final_skills=final_skills,
        final_non_skills=final_non_skills,
    )


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

    repo = SkillLibraryRepository()
    planner = AzureReversePlannerLLM()

    # ── Stage 1: alias resolution for ALL final_skills ──────────────────────
    # We resolve every final_skill (not just llm_skills) so initial_skills
    # also get their DB info attached. Alias persistence only fires for
    # llm_skills, since initial_skills already came from canonical lookup.
    alias_lookup: dict[str, dict] = {}
    if final_skills:
        alias_lookup = await asyncio.to_thread(
            repo.find_canonical_skills_by_aliases, final_skills
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

    new_aliases_persisted = 0
    if aliases_to_insert:
        try:
            new_aliases_persisted = await asyncio.to_thread(
                repo.add_aliases, aliases_to_insert
            )
        except Exception as exc:
            logger.exception("Persisting aliases failed: %s", exc)
            for am in alias_matches:
                if am.alias_persisted:
                    am.alias_persisted = False
                    am.alias_persist_skipped_reason = f"db error: {exc}"
            new_alias_per_final.clear()
            new_aliases_persisted = 0

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

    # ── Stage 3: reverse planner for unmatched llm_skills ───────────────────
    # Pass the existing dimension catalogue so the LLM reuses canonical slugs
    # instead of minting near-synonyms. Run skill enrichment concurrently.
    dimension_catalogue: list[dict] = []
    dimensions_by_skill: dict[str, list[dict]] = {}
    skill_enrichment: dict[str, dict] = {}

    if unmatched_llm_skills:
        try:
            dimension_catalogue = await asyncio.to_thread(
                repo.fetch_dimension_catalogue
            )
        except Exception as exc:
            logger.warning(
                "fetch_dimension_catalogue failed; LLM will run ungrounded: %s",
                exc,
            )
            dimension_catalogue = []

        dims_result, enrich_result = await asyncio.gather(
            planner.infer_dimensions(
                unmatched_llm_skills,
                dimension_catalogue=dimension_catalogue,
            ),
            planner.enrich_new_skills(unmatched_llm_skills),
            return_exceptions=True,
        )
        if isinstance(dims_result, Exception):
            raise HTTPException(
                status_code=502,
                detail=f"Reverse planner LLM (infer_dimensions) failed: {dims_result}",
            ) from dims_result
        dimensions_by_skill = dims_result
        if isinstance(enrich_result, Exception):
            logger.warning("enrich_new_skills failed: %s", enrich_result)
        else:
            skill_enrichment = enrich_result

    # Collect every (skill, llm_dim) pair for DB lookup and possible LLM role.
    llm_pairs: list[tuple[str, dict]] = []
    all_display_names: list[str] = []
    all_slugs: list[str] = []
    for skill, dims in dimensions_by_skill.items():
        for d in dims or []:
            llm_pairs.append((skill, d))
            if d.get("display_name"):
                all_display_names.append(d["display_name"])
            if d.get("slug"):
                all_slugs.append(d["slug"])

    db_dim_map_for_llm: dict[str, dict] = {}
    if llm_pairs:
        db_dim_map_for_llm = await asyncio.to_thread(
            repo.find_dimensions_by_names,
            all_display_names,
            all_slugs,
        )
        # Make sure we have roles for any newly-matched DB dims too.
        new_dim_ids = sorted({
            int(rec["id"]) for rec in db_dim_map_for_llm.values()
            if rec.get("id") is not None and int(rec["id"]) not in roles_by_dim_id
        })
        if new_dim_ids:
            extra_roles = await asyncio.to_thread(
                repo.fetch_roles_for_dimensions, new_dim_ids
            )
            roles_by_dim_id.update(extra_roles)

    # Per (skill, dimension) pair: infer an LLM role even when the dimension
    # is catalogue-grounded in the DB, so the picker can reconcile DB
    # role_dimensions vs semantic fit (e.g. IaC / monitoring → DevOps).
    pending_llm_roles: list[tuple[int, str, dict]] = []
    pending_llm_pair_meta: list[tuple[str, dict, dict | None]] = []

    for skill, d in llm_pairs:
        name_key = (d.get("display_name") or "").lower()
        slug_key = (d.get("slug") or "").lower()
        db_row = (
            db_dim_map_for_llm.get(name_key)
            or db_dim_map_for_llm.get(slug_key)
        )
        pending_llm_pair_meta.append((skill, d, db_row))
        idx = len(pending_llm_pair_meta) - 1
        dim_for_infer = {**d}
        if db_row is not None:
            if db_row.get("display_name"):
                dim_for_infer["display_name"] = str(db_row["display_name"])
            if db_row.get("slug"):
                dim_for_infer["slug"] = str(db_row["slug"])
            rat = db_row.get("rationale") or dim_for_infer.get("rationale")
            if rat:
                dim_for_infer["rationale"] = str(rat)
            dh = db_row.get("difficulty_hint") or dim_for_infer.get(
                "difficulty_hint"
            )
            if dh:
                dim_for_infer["difficulty_hint"] = str(dh)
        pending_llm_roles.append((idx, skill, dim_for_infer))

    # Also infer per-dimension roles for canonical DB-matched skills (stage 4a)
    # so llm_role is not empty for rows like Loki/GCP/CloudFormation.
    pending_db_roles: list[tuple[tuple[str, int], str, dict]] = []
    for term, info in matched_per_final.items():
        canonical: CanonicalSkillSummary = info["canonical"]
        sid = int(canonical.id)
        for d_row in dims_by_skill.get(sid, []) or []:
            dim_id = int(d_row["id"])
            pending_db_roles.append((
                (term, dim_id),
                term,
                {
                    "display_name": str(d_row.get("display_name") or ""),
                    "slug": str(d_row.get("slug") or ""),
                    "rationale": d_row.get("rationale"),
                    "difficulty_hint": (
                        str(d_row.get("difficulty_hint") or "") or None
                    ),
                },
            ))

    llm_roles_by_idx: dict[int, dict | None] = {}
    if pending_llm_roles:
        async def _one(idx: int, skill: str, dim: dict):
            try:
                return idx, await planner.infer_role_for_skill(skill, dim)
            except Exception as exc:
                logger.warning(
                    "infer_role_for_skill failed for skill=%r dim=%r: %s",
                    skill, dim.get("display_name"), exc,
                )
                return idx, None

        results = await asyncio.gather(*[
            _one(i, s, d) for (i, s, d) in pending_llm_roles
        ])
        for idx, role in results:
            llm_roles_by_idx[idx] = role

    llm_roles_for_db_dims: dict[tuple[str, int], dict | None] = {}
    if pending_db_roles:
        async def _one_db(
            key: tuple[str, int], skill: str, dim: dict
        ) -> tuple[tuple[str, int], dict | None]:
            try:
                return key, await planner.infer_role_for_skill(skill, dim)
            except Exception as exc:
                logger.warning(
                    "infer_role_for_skill failed for DB skill=%r dim=%r: %s",
                    skill, dim.get("display_name"), exc,
                )
                return key, None

        db_results = await asyncio.gather(*[
            _one_db(k, s, d) for (k, s, d) in pending_db_roles
        ])
        for key, role in db_results:
            llm_roles_for_db_dims[key] = role

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
            llm_role = llm_roles_for_db_dims.get((term, int(d_row["id"])))
            llm_role_summary = (
                _role_summary_from_llm(llm_role) if llm_role else None
            )
            if llm_role_summary:
                _add_candidate_role(llm_role_summary)
            detail = DimensionDetail(
                input_skill=term,
                dimension=dim_summary,
                roles_from_db=db_role_summaries,
                llm_role=llm_role_summary,
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
            enrich = skill_enrichment.get(term)
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
                new_skill_meta=NewSkillMeta(**enrich) if enrich else None,
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
    if len(candidate_roles) == 1:
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
        cand_payload = []
        for c in candidate_roles:
            payload = {
                "source": c.source,
                "slug": c.slug,
                "display_name": c.display_name,
                "role_archetype": c.role_archetype,
            }
            if c.id is not None:
                payload["id"] = c.id
            cand_payload.append(payload)

        # Per (skill, dimension): DB-linked roles + separate per-pair LLM role.
        skill_dimension_role_map: list[dict] = []
        for dd in dimension_details:
            roles_from_db_payload = [
                {
                    "source": "db",
                    "id": r.id,
                    "slug": r.slug,
                    "display_name": r.display_name,
                    "role_archetype": r.role_archetype,
                }
                for r in dd.roles_from_db
            ]
            llm_payload = None
            if dd.llm_role is not None:
                lr = dd.llm_role
                llm_payload = {
                    "source": "llm",
                    "id": lr.id,
                    "slug": lr.slug,
                    "display_name": lr.display_name,
                    "role_archetype": lr.role_archetype,
                }
                if lr.rationale:
                    llm_payload["rationale"] = lr.rationale
            skill_dimension_role_map.append({
                "skill": dd.input_skill,
                "dimension": {
                    "source": dd.dimension.source,
                    "id": dd.dimension.id,
                    "slug": dd.dimension.slug,
                    "display_name": dd.dimension.display_name,
                },
                "roles_from_db": roles_from_db_payload,
                "llm_role": llm_payload,
            })

        context = {
            "final_skills": final_skills,
            "matched_canonical_skills": [
                info["canonical"].display_name
                for info in matched_per_final.values()
                if info["canonical"].display_name
            ],
            "unmatched_llm_skills": unmatched_llm_skills,
            "dimensions": [
                {"display_name": dd.dimension.display_name,
                 "slug": dd.dimension.slug,
                 "source": dd.dimension.source}
                for dd in dimension_details
            ],
            "skill_dimension_role_map": skill_dimension_role_map,
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

    return ExtractDetailsResponse(
        input_final_skills=final_skills,
        input_llm_skills=llm_skills,
        alias_matches=alias_matches,
        new_aliases_persisted=new_aliases_persisted,
        unmatched_skills=unmatched_llm_skills,
        dimensions=dimension_details,
        skills_detail=skills_detail,
        candidate_roles=candidate_roles,
        chosen_role=chosen_role,
    )


@app.post("/skills/final-role-output", response_model=FinalRoleOutputResponse)
async def final_role_output_endpoint(req: FinalRoleOutputRequest):
    repo = SkillLibraryRepository()

    # 1) Build final input skills with in_db/new tags.
    final_input_skills = [
        FinalInputSkillTag(skill=sd.input_skill, tag=_skill_db_tag(sd))
        for sd in req.skills_detail
    ]

    chosen = req.chosen_role
    persistence = PersistenceReport()
    if chosen is None:
        return FinalRoleOutputResponse(
            chosen_role=None,
            final_input_skills=final_input_skills,
            persistence=persistence,
            planner_output=None,
        )

    # 2) Resolve role in DB (or conditionally create when missing).
    resolved_role = await asyncio.to_thread(
        repo.find_role_by_identity,
        role_id=chosen.id,
        slug=chosen.slug,
        display_name=chosen.display_name,
    )
    role_missing_initially = resolved_role is None
    planner_output: PlannerGeneratedOutput | None = None

    if role_missing_initially:
        matched_rows = sum(
            1 for dd in req.dimensions if _role_matches_chosen(dd, chosen)
        )
        should_create = _should_create_missing_role(
            chosen,
            matched_rows=matched_rows,
            total_rows=len(req.dimensions),
        )
        if should_create:
            try:
                resolved_role = await asyncio.to_thread(
                    repo.create_role,
                    slug=chosen.slug,
                    display_name=chosen.display_name,
                    role_archetype=chosen.role_archetype,
                    source=chosen.source or "llm",
                )
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

    # 3.5) Create canonical_skills rows for new LLM-discovered skills that carry
    # enrichment metadata (category, skill_nature, typical_lifespan from API 2).
    new_skill_metas = {
        sd.input_skill: sd.new_skill_meta
        for sd in req.skills_detail
        if sd.source_tag == "llm"
        and sd.new_skill_meta is not None
        and sd.input_skill not in skill_id_by_input
    }
    for skill_name, meta in new_skill_metas.items():
        if not meta.category or not meta.skill_nature or not meta.typical_lifespan:
            logger.warning(
                "Skipping new skill %r — incomplete enrichment metadata", skill_name
            )
            continue
        try:
            cat_row = await asyncio.to_thread(
                repo.find_or_create_category,
                display_name=meta.category,
            )
            cat_id = int(cat_row["id"])
            sub_cat_id: int | None = None
            if meta.sub_category:
                sub_cat_row = await asyncio.to_thread(
                    repo.find_or_create_category,
                    display_name=meta.sub_category,
                )
                sub_cat_id = int(sub_cat_row["id"])
            skill_row = await asyncio.to_thread(
                repo.create_canonical_skill,
                display_name=skill_name,
                category_id=cat_id,
                sub_category_id=sub_cat_id,
                skill_nature=meta.skill_nature,
                typical_lifespan=meta.typical_lifespan,
            )
            skill_id_by_input[skill_name] = int(skill_row["id"])
            persistence.new_skills_created += 1
        except Exception as exc:
            logger.warning("Failed to create canonical skill %r: %s", skill_name, exc)

    # 4) Persist per (skill, dimension):
    # - always attempt skill-dimension for DB-backed skills
    # - persist role-dimension only when row matches chosen role and role exists in DB
    for dd in req.dimensions:
        skill_name = dd.input_skill
        skill_tag = "in_db" if skill_name in skill_id_by_input else "new"

        matched = _role_matches_chosen(dd, chosen)
        dim_id = dd.dimension.id
        dim_slug = dd.dimension.slug
        dim_name = dd.dimension.display_name
        skill_id = skill_id_by_input.get(skill_name)
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

        if skill_name not in skill_id_by_input:
            item.skipped_reason = "skill_not_in_db"
            persistence.skipped += 1
            persistence.items.append(item)
            continue

        try:
            if dim_id is None:
                dim_row = await asyncio.to_thread(
                    repo.find_or_create_dimension,
                    slug=dim_slug,
                    display_name=dim_name,
                    rationale=dd.dimension.rationale,
                    difficulty_hint=dd.dimension.difficulty_hint,
                    source=dd.dimension.source,
                )
                dim_id = int(dim_row["id"])
            dim_id = int(dim_id)
            item.dimension_id = dim_id

            sd_inserted = await asyncio.to_thread(
                repo.upsert_dimension_skill_link,
                skill_id=skill_id_by_input[skill_name],
                dimension_id=dim_id,
            )
            if sd_inserted:
                persistence.skill_dimension_saved += 1
            # False here usually means mapping already existed; treat as persisted.
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
                # False here usually means mapping already existed; still persisted.
                item.role_dimension_saved = True
            elif matched and (chosen_role_out is None or chosen_role_out.id is None):
                item.skipped_reason = "chosen_role_not_resolved_in_db"
        except Exception as exc:
            item.skipped_reason = f"db_error: {exc}"
            persistence.skipped += 1

        persistence.items.append(item)

    # 5) Planner output when chosen role was missing in DB initially.
    if role_missing_initially:
        planner_output = PlannerGeneratedOutput(
            generated=True,
            role_id=(chosen.slug or "").replace("-", "_"),
            role_display=chosen.display_name,
            payload=None,
        )
        try:
            planner = PlannerAgent()
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
            if chosen_role_out is not None and chosen_role_out.id is not None:
                planner_saved_role_dims = 0
                planner_saved_dim_creates = 0
                for d in plan.dimensions:
                    dim_before = await asyncio.to_thread(
                        repo.find_dimensions_by_names,
                        [d.dimension_name],
                        [d.dimension_id],
                    )
                    existed = bool(dim_before)
                    dim_row = await asyncio.to_thread(
                        repo.find_or_create_dimension,
                        slug=d.dimension_id,
                        display_name=d.dimension_name,
                        rationale=d.rationale,
                        difficulty_hint=d.difficulty_hint,
                        source="llm",
                    )
                    if not existed:
                        planner_saved_dim_creates += 1
                    rd_ins = await asyncio.to_thread(
                        repo.upsert_role_dimension_link,
                        role_id=int(chosen_role_out.id),
                        dimension_id=int(dim_row["id"]),
                    )
                    if rd_ins:
                        planner_saved_role_dims += 1
                planner_output.saved_dimensions_created = planner_saved_dim_creates
                planner_output.saved_role_dimensions = planner_saved_role_dims
        except Exception as exc:
            logger.warning("planner generation failed for missing role=%r: %s", chosen.slug, exc)
            planner_output.payload = {"error": str(exc)}

    return FinalRoleOutputResponse(
        chosen_role=chosen_role_out,
        final_input_skills=final_input_skills,
        persistence=persistence,
        planner_output=planner_output,
    )


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
        "resume_ranking.html",
        {
            "request": request,
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
    return templates.TemplateResponse("skill_library.html", {"request": request})


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
