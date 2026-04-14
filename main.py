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
from pathlib import Path
import tempfile
import time
import logging
import asyncio
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
        br = await tag_all_blocks(blks)
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
        br = await tag_all_blocks(blks)
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
        br = await tag_all_blocks(blks)
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
