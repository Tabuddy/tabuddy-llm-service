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
4) Do not invent content. Only copy/paraphrase small portions from the inputs.
5) Prefer accurate boundaries: identify section headings (Summary, Experience, Projects, Skills, Education).
6) Exclude top-level section headings (e.g. "EXPERIENCE", "WORK EXPERIENCE") from `raw_text` when possible.
7) CRITICAL for `block_type="experience"`: each experience `raw_text` MUST begin with the **verbatim** job header from the inputs before any narrative bullets:
   - Line 1: job title and date range as written (often contains "|", "–", or month names).
   - Line 2 (if present in source): company name and location line.
   Only after those lines, include responsibilities / bullet paragraphs.
   Never start an experience block with mid-role prose (e.g. "Working as a lead...") if the title+company lines exist above it in `normal_text` or `docling_markdown`.

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
  If needed, truncate body text but **never** drop the title + company/date header lines for experience blocks.
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
    max_completion_tokens: int = 6000,
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

