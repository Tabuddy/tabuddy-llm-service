from __future__ import annotations

import json
import logging
import os

from openai import AsyncAzureOpenAI

from models import ResumeBlock
from resume_zoner import zone_resume

logger = logging.getLogger(__name__)

_AZURE_ENDPOINT = "https://tabuddy-azure-sponsor.openai.azure.com/"
_AZURE_DEPLOYMENT = "gpt-4o-mini"
_AZURE_API_VERSION = "2024-12-01-preview"

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
4) Do not invent facts (no fake employers, dates, or projects). Prefer **verbatim copy** from the inputs for resumes that use labeled subsections (see rules 8–10).
5) Prefer accurate boundaries: identify section headings (Summary, Experience, Projects, Skills, Education).
6) Exclude top-level section headings (e.g. "EXPERIENCE", "WORK EXPERIENCE") from `raw_text` when possible.
7) CRITICAL for `block_type="experience"`: each experience `raw_text` MUST begin with the **verbatim** job header from the inputs before any narrative bullets:
   - Line 1: job title and date range as written (often contains "|", "–", or month names).
   - Line 2 (if present in source): company name and location line.
   Only after those lines, include responsibilities / bullet paragraphs **that appear under that employment / organizational experience section in the source only**.
   Never start an experience block with mid-role prose (e.g. "Working as a lead...") if the title+company lines exist above it in `normal_text` or `docling_markdown`.
   **Stacked job lines (no duties in between)**: If **Organizational Experiences** / employment is only **two (or more) consecutive title lines**—each line is a full role (e.g. ends with a date range) and the **very next non-blank line starts another role**—then **each** `experience` block must contain **only that one role line** (plus any bullets that appear **directly under that role** before the next role line). **Stop** each block **before** the next role line and **before** the next major section heading such as **Technical Skills**, **Skills**, **Core Competencies**, or **Project Experiences**. Do **not** append **Technical Skills** (or any skills-inventory lines) to `Experience_0` just because they follow the first role in the file; that content belongs in `skills_dump`, not `experience`.
8) CRITICAL for `block_type="project"`: many resumes use fixed labels—**Project Name**, **Client**, **Role**, **Technologies** (or "Technology:"), **Project Description**, **Responsibilities**. When those appear in the source for a project, the project's `raw_text` MUST include **every such section** that exists, in sensible order, with **headings preserved** (same or minimally normalized wording). Include the **full Technologies line** (comma-separated tools are important). Include the **full Project Description** paragraph(s). Under **Responsibilities**, include **every bullet or line** from the source for that project—do not summarize down to 1–2 bullets unless the source truly has that few. If you must shorten for length, drop the **least** specific tail content last—never drop an entire Responsibilities or Technologies section that exists in the inputs.
9) For `block_type="experience"`: if the source has **MODULES**, **Technologies**, **Technical environment**, **Skills / Tools used**, or long comma-separated module/platform lines **immediately under that same role** (between that role's header and the **next** role or section heading), copy them **in full** into that experience block (same order as source). Do **not** treat the resume's global **Technical Skills** / **Skills** section (the block that lists Selenium, JUnit, "Programming Languages:", etc. for the whole CV) as belonging to `Experience_0` unless those lines are explicitly nested under that role in the source—not merely because they appear below the employment list.
10) **Skills_dump** vs project tech: listing tools under a project ("Technologies: Java, Selenium…") belongs in the **project** (or experience) block, not only in Skills_Dump. Keep project-scoped tech with the project.

Naming conventions (for consistency with the existing pipeline):
- Header/intro content before the first known section: block_name = "Header", block_type="other"
- Summary section: block_name = "Summary", block_type="summary"
- Skills section: block_name = "Skills_Dump", block_type="skills_dump"
- Experience section: block_name = "Experience_0", "Experience_1", ... block_type="experience"
  - Each Experience block should correspond to ONE company cluster (company + its roles + that company's bullet points).
- Projects section: block_name = "Project_0", "Project_1", ... block_type="project"
  - Each Project block should correspond to ONE project cluster.
- Education section: block_name = "Education", block_type="education"
- Anything leftover: block_type="other" with block_name like "Other_0", "Other_1", ...

Content size guidance:
- Keep each `raw_text` reasonably sized so the downstream LLM can extract from it.
  If you must truncate: (1) never remove experience job title + company + date header lines, nor **Company** / **Duration** / **Position** / **Module** label lines when the resume uses that layout; (2) for experience, never drop the **Responsibilities** heading or cut duty bullets from the **middle**—trim only from the **end** of the longest responsibility list if unavoidable; (3) for projects, never remove **Technologies**, **Project Description**, or the **Responsibilities** heading—trim from the **end** of the longest bullet list first, not from the middle of responsibilities.
"""


def _get_client() -> AsyncAzureOpenAI | None:
    api_key = os.getenv("AZURE_OPEN_AI_KEY")
    if not api_key:
        return None
    return AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=_AZURE_ENDPOINT,
        api_version=_AZURE_API_VERSION,
    )


def _truncate_for_prompt(s: str, head: int = 25000, tail: int = 8000) -> str:
    s = s or ""
    if len(s) <= head + tail:
        return s
    return s[:head] + "\n\n[TRUNCATED]\n\n" + s[-tail:]


async def build_hybrid_blocks(
    normal_text: str,
    docling_markdown: str,
    *,
    max_completion_tokens: int = 8000,
) -> list[ResumeBlock]:
    try:
        client = _get_client()
        if client is None:
            logger.warning(
                "No Azure OpenAI key – falling back to deterministic zoner"
            )
            return zone_resume(normal_text)

        payload = {
            "normal_text": _truncate_for_prompt(normal_text),
            "docling_markdown": _truncate_for_prompt(docling_markdown),
        }

        input_data = json.dumps(payload, ensure_ascii=False)

        resp = await client.chat.completions.create(
            model=_AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _BLOCK_BUILDER_SYSTEM_PROMPT},
                {"role": "user", "content": input_data},
            ],
            max_completion_tokens=max_completion_tokens,
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0]

        parsed = json.loads(content)
        raw_blocks = parsed.get("blocks") or []

        out: list[ResumeBlock] = []
        for b in raw_blocks:
            if not isinstance(b, dict):
                continue
            bt = b.get("block_type")
            if bt not in (
                "summary",
                "experience",
                "project",
                "skills_dump",
                "education",
                "other",
            ):
                continue
            bn = b.get("block_name") or bt
            rt = (b.get("raw_text") or "").strip()
            if not rt:
                continue
            out.append(ResumeBlock(block_name=bn, block_type=bt, raw_text=rt))

        if not out:
            logger.warning("Hybrid block builder returned no blocks – fallback")
            return zone_resume(normal_text)

        # Ensure minimal blocks exist.
        existing_types = {b.block_type for b in out}
        zoned = zone_resume(normal_text)
        for z in zoned:
            if (
                z.block_type
                in ("summary", "experience", "project", "skills_dump", "education")
                and z.block_type not in existing_types
            ):
                out.append(z)

        return out
    except Exception as e:
        logger.exception("Hybrid block builder failed: %s", e)
        return zone_resume(normal_text)

