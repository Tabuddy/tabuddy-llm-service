from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import textwrap
import time
import uuid
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from docling_client import convert_path_to_markdown
from llm_client import FAST_MODEL, get_fast_client
from meta_tag_extractor import (
    _build_experience_tags,
    _build_project_tags,
    _build_summary_tags,
)
from models import (
    BlockTagResult,
    ContextMetaTags,
    ExperienceDetail,
    GlobalParameters,
    ProjectDetail,
    ResumeBlock,
)
from resume_parser import extract_docx_plain_text
from resume_zoner import zone_resume

logger = logging.getLogger(__name__)

_MAX_IMAGE_PAGES = max(1, int(os.getenv("HYBRID_IMG_MAX_PAGES", "8")))
_MAX_COMPLETION_TOKENS = int(os.getenv("HYBRID_IMG_MAX_COMPLETION_TOKENS", "8000"))
_DEBUG_IMAGE_ROOT = Path(os.getenv("HYBRID_IMG_DEBUG_DIR", "img"))

_BLOCK_BUILDER_SYSTEM_PROMPT = """\
You are a Resume Block Builder for a technical recruiting system.

You MUST output a JSON object with a `blocks` array matching this schema:

Block schema:
{
  "block_name": string,
  "block_type": "summary" | "experience" | "project" | "skills_dump" | "education" | "other",
  "raw_text": string
}

Hard rules:
1) Return ONLY valid JSON. No markdown fences. No extra keys besides `blocks`.
2) Build blocks so the downstream extractor can parse them reliably.
3) Use BOTH inputs:
   - normal_text: plain text extracted from the PDF/DOCX/TXT
   - docling_markdown: Docling markdown for layout-aware hints
4) Do not invent facts (no fake employers, dates, or projects). Prefer verbatim copy from the inputs for resumes that use labeled subsections.
5) Prefer accurate boundaries: identify section headings (Summary, Experience, Projects, Skills, Education).
6) Exclude top-level section headings (e.g. "EXPERIENCE", "WORK EXPERIENCE") from `raw_text` when possible.
7) CRITICAL for `block_type="experience"`: each experience `raw_text` MUST begin with the verbatim job header from the inputs before any narrative bullets.
8) CRITICAL for `block_type="project"`: when labels like Project Name, Client, Role, Technologies, Project Description, Responsibilities exist, keep them in the project block with headings preserved.
9) For `block_type="experience"`: keep technologies/modules/technical environment lines with that role.
10) Skills specific to a project or role belong inside that project/experience block, not only in `Skills_Dump`.

Naming conventions:
- Header/intro content before the first known section: block_name = "Header", block_type="other"
- Summary section: block_name = "Summary", block_type="summary"
- Skills section: block_name = "Skills_Dump", block_type="skills_dump"
- Experience section: block_name = "Experience_0", "Experience_1", ... block_type="experience"
- Projects section: block_name = "Project_0", "Project_1", ... block_type="project"
- Education section: block_name = "Education", block_type="education"
- Anything leftover: block_type="other" with block_name like "Other_0", "Other_1", ...
"""

_IMAGE_BLOCK_BUILDER_SYSTEM_PROMPT = """\
You are a Resume OCR + Resume Block Builder for a technical recruiting system.

You will receive page images for exactly one resume, in reading order.

Return ONLY valid JSON with this schema:
{
  "raw_text": string,
  "blocks": [
    {
      "block_name": string,
      "block_type": "summary" | "experience" | "project" | "skills_dump" | "education" | "other",
      "raw_text": string
    }
  ]
}

Rules:
1) `raw_text` should be the readable resume text reconstructed from the page images in natural reading order.
2) Do not invent text that is not visible in the images.
3) Preserve role headers, project labels, technologies lines, dates, employer names, and section headings whenever visible.
4) Each experience block must begin with the role header/date line before responsibility bullets.
5) Each project block must retain labeled fields like Project Name, Client, Role, Technologies, Project Description, Responsibilities when present.
6) Use the same block naming conventions as the existing pipeline:
   Header, Summary, Skills_Dump, Experience_0..N, Project_0..N, Education, Other_0..N.
7) Return JSON only. No markdown fences. No explanations.
"""

_IMAGE_ONE_SHOT_ANALYSIS_PROMPT = """\
You are a Resume OCR + Resume Intelligence system for a technical recruiting platform.

You will receive page images for exactly one resume, in reading order.

Return ONLY valid JSON with this schema:
{
  "raw_text": "string",
  "blocks": [
    {
      "block_name": "string",
      "block_type": "summary" | "experience" | "project" | "skills_dump" | "education" | "other",
      "raw_text": "string",
      "skills": [
        {
          "skill": "string",
          "action_verb": "string",
          "context": "string",
          "co_dependent_skills": ["string"],
          "metric": "string or null"
        }
      ],
      "quantifiers": ["string"],
      "experience_detail": {
        "company": "string or null",
        "role": "string or null",
        "duration": "string or null",
        "is_current": true,
        "key_achievements": ["string"],
        "quantifiers": ["string"],
        "tech_stack": ["string"]
      },
      "project_detail": {
        "project_name": "string or null",
        "description": "string or null",
        "tech_stack": ["string"],
        "quantifiers": ["string"],
        "key_highlights": ["string"]
      }
    }
  ],
  "context_meta_tags": {
    "summary_tags": {
      "key_highlights": ["string"],
      "quantifiers": ["string"],
      "years_of_experience": "string or null",
      "domain_focus": ["string"]
    },
    "experience_tags": {
      "total_companies": 0,
      "current_company": "string or null",
      "current_role": "string or null",
      "experience_timeline": [],
      "top_quantifiers": ["string"],
      "combined_tech_stack": ["string"]
    },
    "project_tags": {
      "total_projects": 0,
      "projects": [],
      "combined_tech_stack": ["string"]
    },
    "skill_categories": {
      "languages": ["string"],
      "frameworks": ["string"],
      "databases": ["string"],
      "cloud_devops": ["string"],
      "tools": ["string"],
      "methodologies": ["string"],
      "domain_skills": ["string"],
      "soft_skills": ["string"]
    },
    "resume_strength_signals": ["string"]
  },
  "global_parameters": {
    "scale_anchors": [
      {
        "extracted_metric": "string",
        "extracted_unit": "string",
        "provenance": "string",
        "raw_evidence": "string"
      }
    ],
    "codebase_lifecycle": [{"normalized_value": "string", "provenance": "string", "raw_evidence": "string"}],
    "product_domain": [{"normalized_value": "string", "provenance": "string", "raw_evidence": "string"}],
    "architectural_paradigm": [{"normalized_value": "string", "provenance": "string", "raw_evidence": "string"}],
    "compliance_exposure": [{"normalized_value": "string", "provenance": "string", "raw_evidence": "string"}],
    "leadership_footprint": [{"normalized_value": "string", "provenance": "string", "raw_evidence": "string"}],
    "cross_functional_area": [{"normalized_value": "string", "provenance": "string", "raw_evidence": "string"}]
  },
  "reasoning_log": ["string"]
}

Rules:
1) Reconstruct `raw_text` from the images in reading order.
2) Build high-quality blocks. Each experience block must begin with the role header/date line before bullets.
3) Preserve project labels like Project Name, Client, Role, Technologies, Project Description, Responsibilities when visible.
4) Use block names: Header, Summary, Skills_Dump, Experience_0..N, Project_0..N, Education, Other_0..N.
5) Do not invent facts not visible in the images.
6) `skills` should contain canonical names, short exact context quotes, and string quantifiers only.
7) `context_meta_tags` and `global_parameters` must be grounded in the same extracted content.
8) Return JSON only, no markdown fences, no explanation.
"""


def _truncate_for_prompt(s: str, head: int = 25000, tail: int = 8000) -> str:
    s = s or ""
    if len(s) <= head + tail:
        return s
    return s[:head] + "\n\n[TRUNCATED]\n\n" + s[-tail:]


def _extract_json_object(content: str) -> dict:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return a JSON object.")
    return json.loads(text[start:end + 1])


def _coerce_blocks(raw_blocks: object) -> list[ResumeBlock]:
    out: list[ResumeBlock] = []
    for item in raw_blocks or []:
        if not isinstance(item, dict):
            continue
        block_type = item.get("block_type")
        if block_type not in {
            "summary",
            "experience",
            "project",
            "skills_dump",
            "education",
            "other",
        }:
            continue
        block_name = str(item.get("block_name") or block_type).strip()
        raw_text = str(item.get("raw_text") or "").strip()
        if not block_name or not raw_text:
            continue
        out.append(
            ResumeBlock(
                block_name=block_name,
                block_type=block_type,
                raw_text=raw_text,
            )
        )
    return out


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _coerce_block_results(raw_blocks: object) -> list[BlockTagResult]:
    out: list[BlockTagResult] = []
    for item in raw_blocks or []:
        if not isinstance(item, dict):
            continue
        block_type = str(item.get("block_type") or "other").strip() or "other"
        block_name = str(item.get("block_name") or block_type).strip() or block_type

        raw_skills = item.get("skills") if isinstance(item.get("skills"), list) else []
        skills = []
        for skill in raw_skills:
            if not isinstance(skill, dict):
                continue
            skill_name = str(skill.get("skill") or "").strip()
            if not skill_name:
                continue
            skills.append(
                {
                    "skill": skill_name,
                    "action_verb": str(skill.get("action_verb") or "Applied").strip() or "Applied",
                    "context": str(skill.get("context") or "").strip(),
                    "co_dependent_skills": _string_list(skill.get("co_dependent_skills")),
                    "metric": None if skill.get("metric") in (None, "") else str(skill.get("metric")).strip(),
                }
            )

        exp_detail = item.get("experience_detail")
        if not isinstance(exp_detail, dict):
            exp_detail = None
        proj_detail = item.get("project_detail")
        if not isinstance(proj_detail, dict):
            proj_detail = None

        out.append(
            BlockTagResult(
                block_name=block_name,
                block_type=block_type,
                skills=skills,
                quantifiers=_string_list(item.get("quantifiers")),
                experience_detail=(
                    ExperienceDetail(
                        company=exp_detail.get("company"),
                        role=exp_detail.get("role"),
                        duration=exp_detail.get("duration"),
                        is_current=bool(exp_detail.get("is_current", False)),
                        key_achievements=_string_list(exp_detail.get("key_achievements")),
                        quantifiers=_string_list(exp_detail.get("quantifiers")),
                        tech_stack=_string_list(exp_detail.get("tech_stack")),
                    )
                    if exp_detail else None
                ),
                project_detail=(
                    ProjectDetail(
                        project_name=proj_detail.get("project_name"),
                        description=proj_detail.get("description"),
                        tech_stack=_string_list(proj_detail.get("tech_stack")),
                        quantifiers=_string_list(proj_detail.get("quantifiers")),
                        key_highlights=_string_list(proj_detail.get("key_highlights")),
                    )
                    if proj_detail else None
                ),
            )
        )
    return out


def _merge_missing_section_types(
    blocks: list[ResumeBlock],
    fallback_text: str,
) -> list[ResumeBlock]:
    if not blocks:
        return zone_resume(fallback_text)

    existing_types = {b.block_type for b in blocks}
    merged = list(blocks)
    for zoned in zone_resume(fallback_text):
        if (
            zoned.block_type in {"summary", "experience", "project", "skills_dump", "education"}
            and zoned.block_type not in existing_types
        ):
            merged.append(zoned)
    return merged


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _render_text_to_images(text: str) -> list[Image.Image]:
    text = (text or "").strip()
    if not text:
        return []

    page_size = (1240, 1754)
    margin_x = 72
    margin_y = 72
    font = _load_font(24)
    title_font = _load_font(28)
    line_height = 34
    max_lines = 44
    wrap_width = 88

    logical_lines: list[str] = []
    for paragraph in text.splitlines():
        para = paragraph.rstrip()
        if not para:
            logical_lines.append("")
            continue
        logical_lines.extend(textwrap.wrap(para, width=wrap_width) or [""])

    pages: list[Image.Image] = []
    for index in range(0, len(logical_lines), max_lines):
        img = Image.new("RGB", page_size, "white")
        draw = ImageDraw.Draw(img)
        draw.text((margin_x, margin_y), "Resume Page", fill="black", font=title_font)

        y = margin_y + 52
        for line in logical_lines[index:index + max_lines]:
            draw.text((margin_x, y), line, fill="black", font=font)
            y += line_height
        pages.append(img)
    return pages


def _docx_bytes_to_render_text(file_bytes: bytes) -> str:
    docling_markdown = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        try:
            docling_markdown = convert_path_to_markdown(tmp_path, device="cpu")
        except Exception as exc:
            logger.warning("DOCX Docling markdown conversion failed: %s", exc)
            docling_markdown = ""
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if docling_markdown.strip():
        return docling_markdown
    return extract_docx_plain_text(file_bytes)


def _pdf_bytes_to_images(file_bytes: bytes) -> list[Image.Image]:
    import pypdfium2 as pdfium

    scale = float(os.getenv("HYBRID_IMG_PDF_SCALE", "1.3"))
    pdf = pdfium.PdfDocument(file_bytes)
    images: list[Image.Image] = []
    try:
        page_count = len(pdf)
        logger.info("Hybrid-img: PDF renderer detected %d page(s)", page_count)
        for page_index in range(page_count):
            page = pdf[page_index]
            try:
                bitmap = page.render(
                    scale=scale,
                    rotation=0,
                    prefer_bgrx=False,
                )
                pil_image = bitmap.to_pil().convert("RGB")
                images.append(pil_image)
            finally:
                page.close()
        return images
    finally:
        pdf.close()


def _document_bytes_to_images(file_bytes: bytes, filename: str) -> list[Image.Image]:
    lower = (filename or "").lower()
    if lower.endswith(".pdf"):
        return _pdf_bytes_to_images(file_bytes)

    if lower.endswith((".docx", ".doc")):
        try:
            text = _docx_bytes_to_render_text(file_bytes)
        except Exception as exc:
            logger.warning("DOCX text extraction for image rendering failed: %s", exc)
            text = ""
        return _render_text_to_images(text)

    try:
        text = file_bytes.decode("utf-8", errors="replace")
    except Exception:
        text = ""
    return _render_text_to_images(text)


def _image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    prepared = image.convert("RGB")
    prepared.thumbnail((1400, 1800))
    prepared.save(buffer, format="JPEG", quality=85, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _safe_stem(filename: str) -> str:
    stem = Path(filename or "upload").stem or "upload"
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return cleaned[:80] or "upload"


def _persist_debug_images(images: list[Image.Image], filename: str) -> Path | None:
    if not images:
        return None

    run_dir = _DEBUG_IMAGE_ROOT / f"{_safe_stem(filename)}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for idx, image in enumerate(images, start=1):
        out_path = run_dir / f"page_{idx:02d}.png"
        image.convert("RGB").save(out_path, format="PNG")

    return run_dir


def _extract_experience_achievements_from_raw_text(raw_text: str, *, limit: int = 5) -> list[str]:
    lines = [line.strip() for line in (raw_text or "").splitlines() if line.strip()]
    if not lines:
        return []

    achievements: list[str] = []
    for line in lines[2:]:
        cleaned = line.lstrip("-*• ").strip()
        if len(cleaned) < 12:
            continue
        achievements.append(cleaned)
        if len(achievements) >= limit:
            break
    return achievements


def _backfill_experience_details_from_blocks(
    zoned_blocks: list[ResumeBlock],
    block_results: list[BlockTagResult],
) -> list[BlockTagResult]:
    raw_by_name = {block.block_name: block.raw_text for block in zoned_blocks}
    updated: list[BlockTagResult] = []

    for result in block_results:
        if result.block_type == "experience" and result.experience_detail:
            if not result.experience_detail.key_achievements:
                result.experience_detail.key_achievements = _extract_experience_achievements_from_raw_text(
                    raw_by_name.get(result.block_name, ""),
                )
        updated.append(result)
    return updated


def _normalize_context_meta_tags_payload(payload: dict | None) -> dict:
    data = dict(payload or {})

    project_tags = data.get("project_tags")
    if isinstance(project_tags, dict):
        normalized_project_tags = dict(project_tags)
        projects = normalized_project_tags.get("projects")
        if isinstance(projects, list):
            normalized_projects: list[dict] = []
            for item in projects:
                if isinstance(item, dict):
                    normalized_projects.append(item)
                elif item is not None:
                    name = str(item).strip()
                    if name:
                        normalized_projects.append(
                            {
                                "project_name": name,
                                "description": None,
                                "tech_stack": [],
                                "quantifiers": [],
                                "key_highlights": [],
                            }
                        )
            normalized_project_tags["projects"] = normalized_projects
        data["project_tags"] = normalized_project_tags

    experience_tags = data.get("experience_tags")
    if isinstance(experience_tags, dict):
        normalized_experience_tags = dict(experience_tags)
        timeline = normalized_experience_tags.get("experience_timeline")
        if isinstance(timeline, list):
            normalized_timeline: list[dict] = []
            for item in timeline:
                if isinstance(item, dict):
                    normalized_timeline.append(item)
            normalized_experience_tags["experience_timeline"] = normalized_timeline
        data["experience_tags"] = normalized_experience_tags

    return data


async def build_hybrid_blocks(
    normal_text: str,
    docling_markdown: str,
    *,
    max_completion_tokens: int = _MAX_COMPLETION_TOKENS,
) -> list[ResumeBlock]:
    try:
        client = get_fast_client()
        if client is None:
            logger.warning("No Azure OpenAI key; falling back to deterministic zoner.")
            return zone_resume(normal_text)

        payload = {
            "normal_text": _truncate_for_prompt(normal_text),
            "docling_markdown": _truncate_for_prompt(docling_markdown),
        }

        resp = await client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": _BLOCK_BUILDER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_completion_tokens=max_completion_tokens,
        )
        parsed = _extract_json_object(resp.choices[0].message.content or "")
        blocks = _coerce_blocks(parsed.get("blocks"))
        return _merge_missing_section_types(blocks, normal_text)
    except Exception as exc:
        logger.exception("Hybrid block builder failed: %s", exc)
        return zone_resume(normal_text)


async def build_hybrid_blocks_from_images(
    file_bytes: bytes,
    filename: str,
    *,
    max_completion_tokens: int = _MAX_COMPLETION_TOKENS,
) -> tuple[str, list[ResumeBlock]]:
    try:
        started_at = time.perf_counter()
        client = get_fast_client()
        if client is None:
            logger.warning("No Azure OpenAI key; skipping image block builder.")
            return "", []

        convert_started_at = time.perf_counter()
        images = await asyncio.to_thread(_document_bytes_to_images, file_bytes, filename)
        logger.info(
            "Hybrid-img: document-to-images for %s produced %d page(s) in %.2fs",
            filename,
            len(images),
            time.perf_counter() - convert_started_at,
        )
        if not images:
            logger.warning("No document images could be created for %s.", filename)
            return "", []

        save_started_at = time.perf_counter()
        image_dir = await asyncio.to_thread(_persist_debug_images, images, filename)
        logger.info(
            "Hybrid-img: saved %d debug image(s) for %s to %s in %.2fs",
            len(images),
            filename,
            image_dir,
            time.perf_counter() - save_started_at,
        )

        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": "These images contain one resume in page order. Extract the resume text and split it into blocks. Return JSON only.",
            }
        ]
        payload_started_at = time.perf_counter()
        for image in images[:_MAX_IMAGE_PAGES]:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_to_data_url(image),
                    },
                }
            )
        logger.info(
            "Hybrid-img: prepared %d image payload(s) for %s in %.2fs",
            min(len(images), _MAX_IMAGE_PAGES),
            filename,
            time.perf_counter() - payload_started_at,
        )

        llm_started_at = time.perf_counter()
        resp = await client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": _IMAGE_BLOCK_BUILDER_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_completion_tokens=max_completion_tokens,
        )
        logger.info(
            "Hybrid-img: vision LLM call for %s finished in %.2fs",
            filename,
            time.perf_counter() - llm_started_at,
        )

        parse_started_at = time.perf_counter()
        parsed = _extract_json_object(resp.choices[0].message.content or "")
        raw_text = str(parsed.get("raw_text") or "").strip()
        blocks = _coerce_blocks(parsed.get("blocks"))
        if not blocks and raw_text:
            blocks = zone_resume(raw_text)
        elif raw_text:
            blocks = _merge_missing_section_types(blocks, raw_text)
        logger.info(
            "Hybrid-img: parsed LLM response for %s into raw_text=%d chars, blocks=%d in %.2fs (total %.2fs)%s",
            filename,
            len(raw_text),
            len(blocks),
            time.perf_counter() - parse_started_at,
            time.perf_counter() - started_at,
            f", debug_images={image_dir}" if image_dir else "",
        )
        return raw_text, blocks
    except Exception as exc:
        logger.exception("Hybrid image block builder failed: %s", exc)
        return "", []


async def analyze_resume_from_images_one_shot(
    file_bytes: bytes,
    filename: str,
    *,
    max_completion_tokens: int = 12000,
) -> tuple[str, list[ResumeBlock], list[BlockTagResult], ContextMetaTags, GlobalParameters, list[str]]:
    try:
        started_at = time.perf_counter()
        client = get_fast_client()
        if client is None:
            logger.warning("No Azure OpenAI key; skipping one-shot image analysis.")
            return "", [], [], ContextMetaTags(), GlobalParameters(), ["LLM unavailable"]

        convert_started_at = time.perf_counter()
        images = await asyncio.to_thread(_document_bytes_to_images, file_bytes, filename)
        logger.info(
            "Hybrid-img one-shot: document-to-images for %s produced %d page(s) in %.2fs",
            filename,
            len(images),
            time.perf_counter() - convert_started_at,
        )
        if not images:
            return "", [], [], ContextMetaTags(), GlobalParameters(), ["No document images available"]

        # Debug image persistence is intentionally disabled here.
        # The vision request below already sends each page directly as an inline
        # data URL, so storing page images on disk is not required.
        image_dir = None

        # save_started_at = time.perf_counter()
        # image_dir = await asyncio.to_thread(_persist_debug_images, images, filename)
        # logger.info(
        #     "Hybrid-img one-shot: saved %d debug image(s) for %s to %s in %.2fs",
        #     len(images),
        #     filename,
        #     image_dir,
        #     time.perf_counter() - save_started_at,
        # )

        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": "Analyze this single resume from the attached page images and return the full JSON schema only.",
            }
        ]
        payload_started_at = time.perf_counter()
        for image in images[:_MAX_IMAGE_PAGES]:
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image)}})
        logger.info(
            "Hybrid-img one-shot: prepared %d image payload(s) for %s in %.2fs",
            min(len(images), _MAX_IMAGE_PAGES),
            filename,
            time.perf_counter() - payload_started_at,
        )

        llm_started_at = time.perf_counter()
        resp = await client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": _IMAGE_ONE_SHOT_ANALYSIS_PROMPT},
                {"role": "user", "content": content},
            ],
            max_completion_tokens=max_completion_tokens,
        )
        logger.info(
            "Hybrid-img one-shot: vision LLM call for %s finished in %.2fs",
            filename,
            time.perf_counter() - llm_started_at,
        )

        parse_started_at = time.perf_counter()
        parsed = _extract_json_object(resp.choices[0].message.content or "")
        raw_text = str(parsed.get("raw_text") or "").strip()
        raw_blocks = parsed.get("blocks")
        zoned_blocks = _coerce_blocks(raw_blocks)
        block_results = _coerce_block_results(raw_blocks)

        if raw_text and zoned_blocks:
            zoned_blocks = _merge_missing_section_types(zoned_blocks, raw_text)

        if raw_text and len(block_results) < len(zoned_blocks):
            by_name = {b.block_name: b for b in block_results}
            merged_results: list[BlockTagResult] = []
            for zb in zoned_blocks:
                merged_results.append(
                    by_name.get(
                        zb.block_name,
                        BlockTagResult(block_name=zb.block_name, block_type=zb.block_type),
                    )
                )
            block_results = merged_results

        block_results = _backfill_experience_details_from_blocks(zoned_blocks, block_results)

        context_meta_tags = ContextMetaTags.model_validate(
            _normalize_context_meta_tags_payload(parsed.get("context_meta_tags"))
        )
        context_meta_tags.summary_tags = _build_summary_tags(block_results)
        context_meta_tags.experience_tags = _build_experience_tags(block_results)
        context_meta_tags.project_tags = _build_project_tags(block_results)
        global_parameters = GlobalParameters.model_validate(parsed.get("global_parameters") or {})
        reasoning_log = _string_list(parsed.get("reasoning_log"))

        logger.info(
            "Hybrid-img one-shot: parsed full response for %s into raw_text=%d chars, blocks=%d, block_results=%d in %.2fs (total %.2fs)%s",
            filename,
            len(raw_text),
            len(zoned_blocks),
            len(block_results),
            time.perf_counter() - parse_started_at,
            time.perf_counter() - started_at,
            f", debug_images={image_dir}" if image_dir else "",
        )
        return raw_text, zoned_blocks, block_results, context_meta_tags, global_parameters, reasoning_log
    except Exception as exc:
        logger.exception("Hybrid image one-shot analysis failed: %s", exc)
        return "", [], [], ContextMetaTags(), GlobalParameters(), [f"One-shot analysis failed: {exc}"]
