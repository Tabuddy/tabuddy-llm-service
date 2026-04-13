"""Block Tagger – Extracts skills, experience details, project details, and
quantifiers from resume blocks via parallel per-block LLM calls.

Each block gets rich structured metadata: skills with trace info, experience/
project details with achievements and quantifiers, and tech stack per role.
Blocks are processed concurrently for speed.
"""

from __future__ import annotations

import asyncio
import os
import json
import logging
from pydantic import ValidationError
from openai import AsyncAzureOpenAI
from models import (
    ResumeBlock,
    BlockTagResult,
    BlockSkillEntry,
    ExperienceDetail,
    ProjectDetail,
)

logger = logging.getLogger(__name__)

_BLOCK_TAGGER_CONCURRENCY = max(
    1, int(os.getenv("BLOCK_TAGGER_MAX_CONCURRENCY", "4"))
)

_AZURE_ENDPOINT = "https://tabuddy-azure-sponsor.openai.azure.com/"
_AZURE_DEPLOYMENT = "gpt-4o-mini"
_AZURE_API_VERSION = "2024-12-01-preview"

_client: AsyncAzureOpenAI | None = None


def _get_client() -> AsyncAzureOpenAI | None:
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("AZURE_OPEN_AI_KEY")
    if not api_key:
        return None
    _client = AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=_AZURE_ENDPOINT,
        api_version=_AZURE_API_VERSION,
    )
    return _client


_SKILL_EXTRACTION_PROMPT = """\
You are a precision Resume Analyzer. Extract structured data from ONE resume block.

The block has a `block_type` telling you what to extract:

FOR ALL BLOCKS (except "other"):
Extract "skills" array. Each skill has:
- "skill": Canonical name (e.g. "React" not "ReactJS")
- "action_verb": Exact verb used ("Led","Built","Designed"). "Applied" if unclear. "Listed" for skills_dump.
- "context": 5-15 word exact quote showing usage. "Listed in skills section" for skills_dump.
- "co_dependent_skills": Other skills in same action. [] if none.
- "metric": Non-financial scale metric ("40% improvement","1M users"). null if none.
Extract "quantifiers": ALL numeric claims (percentages, counts, scales, durations).

WHAT IS A SKILL: Languages, frameworks, tools, databases, methodologies, architectural patterns, domain competencies, business tools, soft/leadership skills.
NOT A SKILL: Company names, locations, job titles, dates, generic words (engineering, team, development), financial metrics.

FOR block_type="experience", also extract "experience_detail":
- "company", "role", "duration" (date range as written), "is_current" (true if "Present"/"Current" in dates)
- "key_achievements": Up to 5 impactful 1-sentence achievements with quantified results
- "quantifiers": ALL numeric claims from this block
- "tech_stack": Technologies used in this role

FOR block_type="project", also extract "project_detail":
- "project_name" (infer 3-5 word name if none), "description" (1-2 sentences)
- "tech_stack", "quantifiers", "key_highlights"

FOR block_type="summary": Extract ALL skills (even casual mentions), quantifiers, domain focus areas.

Return ONLY valid JSON:
{
  "skills": [{"skill":"","action_verb":"","context":"","co_dependent_skills":[],"metric":null}],
  "quantifiers": [],
  "experience_detail": {"company":null,"role":null,"duration":null,"is_current":false,"key_achievements":[],"quantifiers":[],"tech_stack":[]},
  "project_detail": {"project_name":null,"description":null,"tech_stack":[],"quantifiers":[],"key_highlights":[]}
}
Include experience_detail ONLY for experience blocks, project_detail ONLY for project blocks. Set unused detail to null.
"""

_SCHEMA_RETRY_HINT = (
    "Your previous JSON failed our schema. Return ONLY valid JSON. "
    "Every quantifiers entry (top-level, experience_detail.quantifiers, "
    "project_detail.quantifiers) must be a STRING (e.g. \"70\" or \"10%\"), never a raw number."
)


def _build_block_tag_result(block: ResumeBlock, bd: dict) -> BlockTagResult:
    """Build BlockTagResult from parsed LLM JSON; may raise ValidationError."""
    skills = [
        BlockSkillEntry(
            skill=s.get("skill", ""),
            action_verb=s.get("action_verb", "Applied"),
            context=s.get("context", ""),
            co_dependent_skills=s.get("co_dependent_skills", []),
            metric=s.get("metric"),
        )
        for s in bd.get("skills", [])
        if s.get("skill")
    ]

    exp_detail = None
    exp_data = bd.get("experience_detail")
    if exp_data and isinstance(exp_data, dict):
        exp_detail = ExperienceDetail(
            company=exp_data.get("company"),
            role=exp_data.get("role"),
            duration=exp_data.get("duration"),
            is_current=exp_data.get("is_current", False),
            key_achievements=exp_data.get("key_achievements", []),
            quantifiers=exp_data.get("quantifiers", []),
            tech_stack=exp_data.get("tech_stack", []),
        )

    proj_detail = None
    proj_data = bd.get("project_detail")
    if proj_data and isinstance(proj_data, dict):
        proj_detail = ProjectDetail(
            project_name=proj_data.get("project_name"),
            description=proj_data.get("description"),
            tech_stack=proj_data.get("tech_stack", []),
            quantifiers=proj_data.get("quantifiers", []),
            key_highlights=proj_data.get("key_highlights", []),
        )

    return BlockTagResult(
        block_name=block.block_name,
        block_type=block.block_type,
        skills=skills,
        quantifiers=bd.get("quantifiers", []),
        experience_detail=exp_detail,
        project_detail=proj_detail,
    )


async def _tag_single_block(
    client: AsyncAzureOpenAI, block: ResumeBlock,
) -> BlockTagResult:
    """Send a single block to the LLM for extraction; one retry on JSON/schema failure."""
    input_data = {
        "block_name": block.block_name,
        "block_type": block.block_type,
        "raw_text": block.raw_text,
    }
    for attempt in range(2):
        try:
            payload = dict(input_data)
            if attempt == 1:
                payload["_schema_retry"] = _SCHEMA_RETRY_HINT

            response = await client.chat.completions.create(
                model=_AZURE_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": _SKILL_EXTRACTION_PROMPT},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                max_completion_tokens=8000,
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            logger.info(
                "Block tagger [%s] LLM: finish=%s, len=%d, reasoning=%s, completion=%s",
                block.block_name,
                choice.finish_reason,
                len(content),
                getattr(response.usage.completion_tokens_details,
                        "reasoning_tokens", "N/A")
                if response.usage and response.usage.completion_tokens_details else "N/A",
                response.usage.completion_tokens if response.usage else "N/A",
            )

            if not content.strip():
                logger.warning(
                    "Block [%s] LLM returned empty – using fallback", block.block_name)
                return _fallback_extract(block)

            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
            bd = json.loads(content)
            return _build_block_tag_result(block, bd)

        except json.JSONDecodeError as e:
            if attempt == 0:
                logger.warning(
                    "Block [%s] invalid JSON, retrying LLM once: %s",
                    block.block_name,
                    e,
                )
                continue
            logger.error(
                "Block [%s] invalid JSON after retry: %s", block.block_name, e)
            return _fallback_extract(block)

        except ValidationError as e:
            if attempt == 0:
                logger.warning(
                    "Block [%s] schema validation failed, retrying LLM once: %s",
                    block.block_name,
                    e,
                )
                continue
            logger.error(
                "Block [%s] schema validation failed after retry: %s",
                block.block_name,
                e,
            )
            return _fallback_extract(block)

        except Exception as e:
            logger.exception("Block [%s] LLM failed: %s", block.block_name, e)
            return _fallback_extract(block)

    return _fallback_extract(block)


async def tag_all_blocks(blocks: list[ResumeBlock]) -> list[BlockTagResult]:
    """Process each block in parallel via concurrent LLM calls."""
    client = _get_client()

    if client is None:
        logger.warning("No Azure OpenAI key – using fallback extraction")
        return [_fallback_extract(b) for b in blocks]

    sem = asyncio.Semaphore(_BLOCK_TAGGER_CONCURRENCY)

    async def _guarded_tag(block: ResumeBlock) -> BlockTagResult:
        async with sem:
            return await _tag_single_block(client, block)

    tasks: list = []
    task_indices: list[int] = []
    results: list[BlockTagResult] = [
        BlockTagResult(block_name=b.block_name, block_type=b.block_type)
        for b in blocks
    ]

    for i, block in enumerate(blocks):
        if block.block_type == "other":
            # Skip header/contact blocks – nothing to extract
            continue
        if block.block_type == "skills_dump":
            # Deterministic – no LLM needed
            results[i] = _parse_skills_dump(block)
            continue
        # All other types → LLM call
        tasks.append(_guarded_tag(block))
        task_indices.append(i)

    if tasks:
        llm_results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, res in zip(task_indices, llm_results):
            if isinstance(res, Exception):
                logger.error("Block [%s] raised: %s",
                             blocks[idx].block_name, res)
                results[idx] = _fallback_extract(blocks[idx])
            else:
                results[idx] = res

    return results


def _fallback_extract(block: ResumeBlock) -> BlockTagResult:
    """Fallback when LLM is unavailable – deterministic extraction."""
    if block.block_type in ("other", "education"):
        return BlockTagResult(block_name=block.block_name, block_type=block.block_type)

    if block.block_type == "skills_dump":
        return _parse_skills_dump(block)

    if block.block_type == "experience":
        return _parse_experience_block(block)

    if block.block_type == "project":
        return _parse_project_block(block)

    return BlockTagResult(block_name=block.block_name, block_type=block.block_type)


def _parse_experience_block(block: ResumeBlock) -> BlockTagResult:
    """Deterministic extraction of experience details from a block."""
    import re

    lines = block.raw_text.strip().split("\n")
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    is_current = False
    quantifiers: list[str] = []

    # Date range pattern
    date_re = re.compile(
        r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
        r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"['\u2019]?\s*\d{2,4}"
        r"|(?:19|20)\d{2})"
        r"\s*[-–—to]+\s*"
        r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
        r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"['\u2019]?\s*\d{2,4}"
        r"|(?:19|20)\d{2}"
        r"|Present|Current|Ongoing|Now|Till\s+Date)",
        re.IGNORECASE,
    )

    # Quantifier patterns
    quant_re = re.compile(
        r"\d+[%xX]\s+\w+|\d+\+?\s*(?:years?|engineers?|members?|users?|customers?"
        r"|requests?|transactions?|team|people|stores?|regions?|countries?)"
        r"|\d+[KkMmBb]\+?\s*\w+|\d+(?:\.\d+)?%",
        re.IGNORECASE,
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Find date range
        date_match = date_re.search(stripped)
        if date_match and not duration:
            duration = date_match.group(0).strip()
            if re.search(r"Present|Current|Ongoing|Now", duration, re.IGNORECASE):
                is_current = True
            # Line with date often has company name
            before_date = stripped[: date_match.start()].strip()
            if before_date and not company:
                company = before_date.rstrip(
                    ",").rstrip("-").rstrip("–").strip()

        # Find quantifiers
        for qm in quant_re.finditer(stripped):
            quantifiers.append(qm.group(0))

    # Heuristic: first non-empty line without date might be company/role
    for line in lines[:4]:
        stripped = line.strip()
        if not stripped:
            continue
        if not date_re.search(stripped):
            # Could be company or role line
            if not company:
                company = stripped
            elif not role:
                role = stripped
        else:
            # Date line — check if there's a role title nearby
            # type: ignore[union-attr]
            before = stripped[: date_re.search(stripped).start()].strip()
            if before and not company:
                company = before

    # Extract key achievements from bullet points
    achievements: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and (stripped.startswith(("•", "-", "*", "·", "▪")) or
                         re.match(r"^\d+\.", stripped)):
            achievement = stripped.lstrip("•-*·▪ ").strip()
            if len(achievement) > 20:
                achievements.append(achievement[:200])

    exp_detail = ExperienceDetail(
        company=company,
        role=role,
        duration=duration,
        is_current=is_current,
        key_achievements=achievements[:5],
        quantifiers=quantifiers,
        tech_stack=[],
    )

    return BlockTagResult(
        block_name=block.block_name,
        block_type=block.block_type,
        quantifiers=quantifiers,
        experience_detail=exp_detail,
    )


def _parse_project_block(block: ResumeBlock) -> BlockTagResult:
    """Deterministic extraction of project details from a block."""
    import re

    lines = block.raw_text.strip().split("\n")
    project_name: str | None = None
    highlights: list[str] = []
    quantifiers: list[str] = []

    quant_re = re.compile(
        r"\d+[%xX]\s+\w+|\d+\+?\s*(?:years?|users?|requests?|transactions?"
        r"|team|people|stores?|regions?)"
        r"|\d+[KkMmBb]\+?\s*\w+|\d+(?:\.\d+)?%",
        re.IGNORECASE,
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not project_name and len(stripped) < 80:
            project_name = stripped
        if stripped.startswith(("•", "-", "*", "·", "▪")) or re.match(r"^\d+\.", stripped):
            highlight = stripped.lstrip("•-*·▪ ").strip()
            if len(highlight) > 15:
                highlights.append(highlight[:200])
        for qm in quant_re.finditer(stripped):
            quantifiers.append(qm.group(0))

    proj_detail = ProjectDetail(
        project_name=project_name,
        description=None,
        tech_stack=[],
        quantifiers=quantifiers,
        key_highlights=highlights[:5],
    )

    return BlockTagResult(
        block_name=block.block_name,
        block_type=block.block_type,
        quantifiers=quantifiers,
        project_detail=proj_detail,
    )


def _parse_skills_dump(block: ResumeBlock) -> BlockTagResult:
    """Parse a skills_dump block by splitting on delimiters."""
    import re
    from skills_dictionary import SKILL_ALIASES

    raw = block.raw_text
    tokens = re.split(r"[,|•·\n\r\t]+", raw)
    skills = []
    for token in tokens:
        cleaned = token.strip().strip("-").strip("*").strip()
        if not cleaned or len(cleaned) > 50:
            continue
        key = cleaned.strip().lower()
        if key in SKILL_ALIASES:
            canonical = SKILL_ALIASES[key]
        else:
            canonical = cleaned
        skills.append(
            BlockSkillEntry(
                skill=canonical,
                action_verb="Listed",
                context="Listed in skills section",
                co_dependent_skills=[],
                metric=None,
            )
        )
    return BlockTagResult(
        block_name=block.block_name,
        block_type=block.block_type,
        skills=skills,
    )
