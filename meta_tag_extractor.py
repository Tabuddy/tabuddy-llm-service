"""Context Meta-Tag Extractor – synthesizes high-level resume meta-tags from
block-level extraction results.

Builds SummaryMetaTags, ExperienceMetaTags, ProjectMetaTags, SkillCategoryTags,
and resume strength signals from the structured block results.
Uses an LLM call to categorize skills and generate strength signals.
"""

from __future__ import annotations

import os
import json
import logging
from openai import AsyncAzureOpenAI
from models import (
    BlockTagResult,
    ExperienceDetail,
    ProjectDetail,
    ContextMetaTags,
    SummaryMetaTags,
    ExperienceMetaTags,
    ProjectMetaTags,
    SkillCategoryTags,
)

logger = logging.getLogger(__name__)

_AZURE_ENDPOINT = "https://tabuddy-azure-sponsor.openai.azure.com/"
_AZURE_DEPLOYMENT = "o4-mini"
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


_META_TAG_PROMPT = """\
You are a Resume Intelligence Analyst. Given extracted resume data, you must produce two outputs:

1. SKILL CATEGORIZATION: Classify each unique skill into exactly ONE category.
2. RESUME STRENGTH SIGNALS: Generate 3-8 meta-signals that tell a recruiter what stands out.

INPUT: You will receive a JSON object with:
- "all_skills": array of unique skill names extracted from the resume
- "all_quantifiers": array of all quantifier phrases found across the resume
- "experience_count": number of experience/role blocks
- "project_count": number of project blocks
- "has_summary": whether resume has a summary section
- "summary_quantifiers": quantifiers specifically from the summary
- "top_achievements": notable achievements from experience blocks

TASK 1 - SKILL CATEGORIZATION:
Place each skill into exactly ONE of these categories:
• "languages": Programming/scripting languages (Python, Java, SQL, R, C++, JavaScript, Go, etc.)
• "frameworks": Frameworks & libraries (React, Django, Spring Boot, TensorFlow, Flask, etc.)
• "databases": Database technologies (PostgreSQL, MongoDB, Redis, DynamoDB, Elasticsearch, etc.)
• "cloud_devops": Cloud platforms, DevOps, infrastructure (AWS, Docker, Kubernetes, Terraform, CI/CD, etc.)
• "tools": Developer/business tools (Git, Jira, Figma, Tableau, VS Code, Postman, etc.)
• "methodologies": Development methodologies & practices (Agile, Scrum, TDD, A/B Testing, etc.)
• "domain_skills": Domain-specific competencies (Machine Learning, Product Management, Data Analysis, etc.)
• "soft_skills": Leadership, communication, management skills (Cross-functional Leadership, Mentoring, etc.)

Rules:
- Each skill appears in exactly ONE category
- Use the canonical skill name as provided
- If a skill could fit multiple categories, prefer the most specific one

TASK 2 - RESUME STRENGTH SIGNALS:
Generate 3-8 short, punchy meta-signals that highlight what's notable about this resume.
Focus on:
- Quantified achievements ("Has quantified scale metrics: 1M users, 99.9% uptime")
- Technical depth ("Deep backend expertise: microservices, distributed systems")
- Leadership evidence ("Team leadership: managed teams of 8-12 engineers")
- Project breadth ("Built 4 independent projects with measurable impact")
- Domain expertise ("Strong fintech/payments domain experience")
- Scale signals ("Experience with high-scale systems: 10K RPS, 500TB data")

Rules:
- Each signal should be 5-15 words
- Ground signals in actual data — don't invent claims
- Focus on what a recruiter would find most valuable
- If quantifiers are present, reference specific numbers

Return ONLY valid JSON, no markdown fences:

{
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
}
"""


def _build_summary_tags(block_results: list[BlockTagResult]) -> SummaryMetaTags:
    """Build summary meta-tags from summary block results."""
    for br in block_results:
        if br.block_type == "summary":
            # Extract domain focus from skills in summary
            domain_focus = [
                s.skill for s in br.skills
                if s.action_verb != "Listed"
            ]
            # key_highlights: prefer skills with metrics; fallback to skill contexts
            highlights = [s.context for s in br.skills if s.metric]
            if not highlights:
                highlights = [
                    s.context for s in br.skills
                    if s.context and s.context != "Listed in skills section"
                ]
            return SummaryMetaTags(
                key_highlights=highlights[:10],
                quantifiers=br.quantifiers,
                years_of_experience=_extract_years(br.quantifiers),
                domain_focus=domain_focus[:10],
            )
    return SummaryMetaTags()


def _extract_years(quantifiers: list[str]) -> str | None:
    """Find years of experience from quantifiers."""
    import re
    for q in quantifiers:
        m = re.search(r"(\d+\+?\s*years?)", q, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _build_experience_tags(
    block_results: list[BlockTagResult],
) -> ExperienceMetaTags:
    """Build experience meta-tags from experience block results."""
    timeline: list[ExperienceDetail] = []
    all_quantifiers: list[str] = []
    all_tech: list[str] = []
    current_company: str | None = None
    current_role: str | None = None

    for br in block_results:
        if br.block_type == "experience" and br.experience_detail:
            ed = br.experience_detail

            # If the LLM didn't return key_achievements for this role but we have
            # skills with rich contexts for this block, synthesize achievements
            # from those contexts so the UI can display bullet points.
            if not ed.key_achievements:
                contexts: list[str] = []
                seen_ctx: set[str] = set()
                for s in br.skills:
                    ctx = (s.context or "").strip()
                    if not ctx or ctx == "Listed in skills section":
                        continue
                    if ctx.lower() in seen_ctx:
                        continue
                    seen_ctx.add(ctx.lower())
                    contexts.append(ctx)
                    if len(contexts) >= 6:
                        break
                if contexts:
                    ed.key_achievements = contexts

            timeline.append(ed)
            all_quantifiers.extend(ed.quantifiers)
            all_tech.extend(ed.tech_stack)
            if ed.is_current:
                current_company = ed.company
                current_role = ed.role

    # Deduplicate tech stack preserving order
    seen: set[str] = set()
    unique_tech: list[str] = []
    for t in all_tech:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            unique_tech.append(t)

    return ExperienceMetaTags(
        total_companies=len(timeline),
        current_company=current_company,
        current_role=current_role,
        experience_timeline=timeline,
        top_quantifiers=all_quantifiers,
        combined_tech_stack=unique_tech,
    )


def _build_project_tags(block_results: list[BlockTagResult]) -> ProjectMetaTags:
    """Build project meta-tags from project block results."""
    projects: list[ProjectDetail] = []
    all_tech: list[str] = []

    for br in block_results:
        if br.block_type == "project" and br.project_detail:
            projects.append(br.project_detail)
            all_tech.extend(br.project_detail.tech_stack)

    seen: set[str] = set()
    unique_tech: list[str] = []
    for t in all_tech:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            unique_tech.append(t)

    return ProjectMetaTags(
        total_projects=len(projects),
        projects=projects,
        combined_tech_stack=unique_tech,
    )


async def extract_context_meta_tags(
    block_results: list[BlockTagResult],
) -> ContextMetaTags:
    """Build context meta-tags from all block extraction results.

    Deterministically builds summary/experience/project tags, then uses
    LLM to categorize skills and generate strength signals.
    """
    # Deterministic aggregation
    summary_tags = _build_summary_tags(block_results)
    experience_tags = _build_experience_tags(block_results)
    project_tags = _build_project_tags(block_results)

    # Collect all unique skills and quantifiers for LLM categorization
    all_skills: set[str] = set()
    all_quantifiers: list[str] = []
    top_achievements: list[str] = []

    for br in block_results:
        for s in br.skills:
            all_skills.add(s.skill)
        all_quantifiers.extend(br.quantifiers)
        if br.experience_detail:
            top_achievements.extend(br.experience_detail.key_achievements[:3])

    # LLM call for skill categorization + strength signals
    skill_categories, strength_signals = await _categorize_skills(
        skills=sorted(all_skills),
        quantifiers=all_quantifiers,
        experience_count=experience_tags.total_companies,
        project_count=project_tags.total_projects,
        has_summary=bool(
            summary_tags.key_highlights or summary_tags.domain_focus),
        summary_quantifiers=summary_tags.quantifiers,
        top_achievements=top_achievements,
    )

    return ContextMetaTags(
        summary_tags=summary_tags,
        experience_tags=experience_tags,
        project_tags=project_tags,
        skill_categories=skill_categories,
        resume_strength_signals=strength_signals,
    )


async def _categorize_skills(
    skills: list[str],
    quantifiers: list[str],
    experience_count: int,
    project_count: int,
    has_summary: bool,
    summary_quantifiers: list[str],
    top_achievements: list[str],
) -> tuple[SkillCategoryTags, list[str]]:
    """Use LLM to categorize skills and generate strength signals."""
    if not skills:
        return SkillCategoryTags(), []

    client = _get_client()
    if client is None:
        logger.warning("No Azure OpenAI key – returning uncategorized skills")
        return SkillCategoryTags(domain_skills=skills), [
            "LLM unavailable – skills uncategorized"
        ]

    input_data = {
        "all_skills": skills,
        "all_quantifiers": quantifiers[:30],
        "experience_count": experience_count,
        "project_count": project_count,
        "has_summary": has_summary,
        "summary_quantifiers": summary_quantifiers,
        "top_achievements": top_achievements[:10],
    }

    try:
        response = await client.chat.completions.create(
            model=_AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _META_TAG_PROMPT},
                {"role": "user", "content": json.dumps(input_data)},
            ],
            max_completion_tokens=5000,
        )
        content = response.choices[0].message.content or "{}"
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0]

        parsed = json.loads(content)
        sc = parsed.get("skill_categories", {})

        return (
            SkillCategoryTags(
                languages=sc.get("languages", []),
                frameworks=sc.get("frameworks", []),
                databases=sc.get("databases", []),
                cloud_devops=sc.get("cloud_devops", []),
                tools=sc.get("tools", []),
                methodologies=sc.get("methodologies", []),
                domain_skills=sc.get("domain_skills", []),
                soft_skills=sc.get("soft_skills", []),
            ),
            parsed.get("resume_strength_signals", []),
        )
    except Exception as e:
        logger.exception("Skill categorization failed: %s", e)
        return SkillCategoryTags(domain_skills=skills), [
            f"Categorization error: {e}"
        ]
