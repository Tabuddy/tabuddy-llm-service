"""Section Scorer — JD-aware per-section resume scoring via LLM.

Scores each resume section against the JD profile using specialized prompts.
Uses o4-mini for reasoning-heavy tasks (role fit, experience, summary)
and gpt-4o-mini for structured matching tasks (skills, projects).

Each scorer returns a SectionScore with per-item MatchDetail breakdowns.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from llm_client import (
    get_fast_client, get_reasoning_client,
    FAST_MODEL, REASONING_MODEL,
)
from ranking_models import (
    JDProfile, SectionScore, MatchDetail,
)
from models import ResumeTaggingResponse

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

_ROLE_FIT_PROMPT = """\
You are an expert technical recruiter performing a Role-Fit assessment.

Given a Job Description profile and a candidate's resume summary, assess how well this candidate's overall profile aligns with the JD role.

JD PROFILE:
- Role Family: {role_family}
- Role Objective: {role_objective}
- Role Scope: {role_scope}
- Industry: {industry}
- Required Experience: {required_experience}
- Core Capabilities: {core_capabilities}
- Required Skills: {required_skills}

CANDIDATE PROFILE:
- Summary: {candidate_summary}
- Domain Focus: {domain_focus}
- Years of Experience: {candidate_years}
- Current Role: {current_role}
- Key Strengths: {strength_signals}
- Tech Stack Highlights: {tech_highlights}

SCORING RULES:
1. Score 0-100 based on OVERALL role alignment, not just keyword matching.
2. A candidate with similar but different tools (e.g., Spark ETL vs Informatica ETL) should get 50-70 for transferable skills.
3. Consider: role seniority match, domain relevance, capability breadth, technology transferability.
4. Do NOT penalize heavily for missing specific vendor tools if the candidate has equivalent experience with competing tools.
5. DO penalize if the candidate's career direction is fundamentally different (e.g., frontend dev for data architect role).
6. Treat JD year ranges correctly: `min_years` is the hard threshold; `max_years` is an upper preference and NOT a disqualifier.

Return ONLY a JSON object:
{{
  "score": <0-100>,
  "reasoning": "<2-3 sentence explanation of the score>",
  "strengths": ["<what aligns well>"],
  "gaps": ["<what's missing or weak>"]
}}
"""

_EXPERIENCE_PROMPT = """\
You are an expert technical recruiter evaluating a candidate's work experience against a job description.

JD REQUIREMENTS:
- Role: {role_family} — {role_objective}
- Required Years: {required_experience}
- Core Technologies needed: {required_skills}
- Industry: {industry}
- Key Capabilities: {core_capabilities}

CANDIDATE EXPERIENCE TIMELINE:
{experience_entries}

SCORING RULES:
1. Evaluate EACH experience entry for relevance to the JD.
2. For each entry, assess: technology overlap, role relevance, domain match, achievement quality, scale of work.
3. A candidate using Spark/Hive/BigQuery for data pipelines has TRANSFERABLE experience to a role requiring Informatica/Snowflake — score as "transferable" (40-70), not "gap".
4. Weight recent experience more heavily than older experience.
5. Leadership and architecture experience are valuable for senior roles.
6. Consider career progression — growing responsibility signals are positive.
7. For JD year ranges, evaluate against `min_years` as threshold. Do not penalize candidates for exceeding `max_years`.

For each experience entry, return a match assessment.
Return ONLY a JSON object:
{{
  "score": <0-100 overall experience quality score>,
  "reasoning": "<2-3 sentence summary>",
  "entries": [
    {{
      "item": "<company - role>",
      "match_type": "exact|transferable|partial|gap",
      "score": <0-100>,
      "explanation": "<why this score>"
    }}
  ]
}}
"""

_SKILLS_PROMPT = """\
You are an expert technical recruiter performing a detailed skill-gap analysis.

JD REQUIRED SKILLS & TECHNOLOGIES:
{jd_skills}

CANDIDATE'S SKILLS (with context):
{candidate_skills}

SCORING RULES:
1. For EACH JD required skill, determine if the candidate has it.
2. Match types:
   - "exact": Candidate has this exact skill/technology (score: 90-100)
   - "transferable": Candidate has a directly competing/equivalent tool (score: 50-75)
     Examples: Spark ETL ↔ Informatica ETL, BigQuery ↔ Snowflake, Hive ↔ Redshift, GCP ↔ AWS
   - "partial": Candidate has related but not equivalent experience (score: 25-50)
     Examples: SQL knowledge helps with any data platform, data modeling is tool-agnostic
   - "gap": Candidate has no relevant experience with this skill (score: 0-15)
3. Consider skill context — a skill used in production projects scores higher than one merely "listed".
4. For vendor-specific tools (Informatica IICS, Snowflake), check if candidate has equivalent open-source/competing tools.
5. If resume evidence mentions XML transformation, treat XSLT as at least transferable unless explicitly contradicted.
6. If resume evidence shows CI/CD, GitHub, or branch workflows, treat Git as transferable (not hard gap).

Return ONLY a JSON object:
{{
  "score": <0-100 overall skill match score>,
  "reasoning": "<2-3 sentence summary of skill alignment>",
  "skills": [
    {{
      "item": "<JD required skill>",
      "match_type": "exact|transferable|partial|gap",
      "score": <0-100>,
      "explanation": "<what candidate has that matches or why it's a gap>"
    }}
  ]
}}
"""

_PROJECTS_PROMPT = """\
You are an expert technical recruiter evaluating project experience against a job description.

JD REQUIREMENTS:
- Role: {role_family} — {role_objective}
- Core Technologies: {required_skills}
- Industry: {industry}

CANDIDATE'S PROJECTS:
{project_entries}

SCORING RULES:
1. Evaluate each project for: tech stack alignment with JD, domain relevance, scale/complexity, innovation.
2. Projects using data engineering tools (even different ones from JD) show relevant capability.
3. Consider project scope and impact indicators.

Return ONLY a JSON object:
{{
  "score": <0-100 overall project relevance score>,
  "reasoning": "<2-3 sentence summary>",
  "projects": [
    {{
      "item": "<project name>",
      "match_type": "exact|transferable|partial|gap",
      "score": <0-100>,
      "explanation": "<relevance assessment>"
    }}
  ]
}}
"""

_DOMAIN_PROMPT = """\
You are an expert recruiter assessing domain/industry alignment between a candidate and a job.

JD DOMAIN REQUIREMENTS:
- Industry: {industry}
- Sub-vertical: {sub_vertical}
- Compliance: {compliance}
- Business Model: {business_model}

CANDIDATE'S DOMAIN EXPOSURE:
- Domains worked in: {candidate_domains}
- Industries from experience: {candidate_industries}
- Compliance exposure: {compliance_exposure}

SCORING RULES:
1. Assess how well the candidate's domain experience matches the JD industry needs.
2. "FinTech" experience is PARTIALLY relevant to "investment management/financial services" — score as transferable (40-65).
3. "E-commerce" at scale (Walmart) shows enterprise data skills even if not financial services.
4. Consider adjacent domain experience as partial credit, not zero.
5. If JD does not specify a domain, score generously (70+).

Return ONLY a JSON object:
{{
  "score": <0-100>,
  "reasoning": "<2-3 sentence assessment>",
  "matches": [
    {{
      "item": "<domain area>",
      "match_type": "exact|transferable|partial|gap",
      "score": <0-100>,
      "explanation": "<why>"
    }}
  ]
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_json_parse(text: str) -> dict | None:
    """Parse JSON from LLM response, handling markdown fences and reasoning."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost balanced JSON object
    start = text.find("{")
    if start < 0:
        logger.warning("No JSON object found in LLM response: %.200s", text)
        return None

    # Walk forward to find the matching closing brace
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break

    # Fallback: try first-{ to last-} (original approach)
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse LLM JSON response: %.200s", text)
    return None


def _build_experience_text(resume: ResumeTaggingResponse) -> str:
    """Build formatted experience entries for prompt."""
    entries = []
    for exp in resume.context_meta_tags.experience_tags.experience_timeline:
        parts = [f"- {exp.company} | {exp.role} | {exp.duration}"]
        if exp.is_current:
            parts[0] += " (CURRENT)"
        if exp.tech_stack:
            parts.append(f"  Tech: {', '.join(exp.tech_stack)}")
        if exp.key_achievements:
            for ach in exp.key_achievements[:3]:
                parts.append(f"  • {ach}")
        if exp.quantifiers:
            parts.append(f"  Metrics: {', '.join(exp.quantifiers[:3])}")
        entries.append("\n".join(parts))
    return "\n\n".join(entries) if entries else "No experience entries found."


def _build_skills_text(resume: ResumeTaggingResponse) -> str:
    """Build formatted candidate skills with context for prompt."""
    lines = []
    sc = resume.context_meta_tags.skill_categories

    # Contextual skills from global_skill_index (most valuable)
    contextual = []
    listed = []
    for entry in resume.global_skill_index:
        is_contextual = any(
            t.action_verb.lower() not in {"listed"} for t in entry.traces
        )
        if is_contextual:
            contexts = [t.context for t in entry.traces if t.context][:2]
            contextual.append(
                f"  - {entry.skill} (used in: {'; '.join(contexts)})" if contexts
                else f"  - {entry.skill}"
            )
        else:
            listed.append(entry.skill)

    if contextual:
        lines.append("Skills demonstrated in projects/experience:")
        lines.extend(contextual[:20])

    if listed:
        lines.append(f"\nSkills listed: {', '.join(listed[:30])}")

    # Categorized skills
    cats = []
    if sc.languages:
        cats.append(f"Languages: {', '.join(sc.languages)}")
    if sc.frameworks:
        cats.append(f"Frameworks: {', '.join(sc.frameworks)}")
    if sc.databases:
        cats.append(f"Databases: {', '.join(sc.databases)}")
    if sc.cloud_devops:
        cats.append(f"Cloud/DevOps: {', '.join(sc.cloud_devops)}")
    if sc.tools:
        cats.append(f"Tools: {', '.join(sc.tools)}")
    if sc.domain_skills:
        cats.append(f"Domain: {', '.join(sc.domain_skills)}")

    if cats:
        lines.append("\nCategorized:")
        lines.extend(f"  {c}" for c in cats)

    # Raw-text fallback: surface explicit tech mentions even when structured
    # extraction is incomplete.
    raw = (resume.parsed_text or "").lower()
    if raw:
        keyword_list = [
            "python", "java", "docker", "docker compose", "kubernetes",
            "knowledge graph", "rdf", "sparql", "xslt", "embedding",
            "embeddings", "vector", "azure storage", "git",
        ]
        raw_hits: list[str] = []
        for kw in keyword_list:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, raw):
                raw_hits.append(kw)
        if raw_hits:
            lines.append(
                "\nRaw text tech mentions: " + ", ".join(sorted(set(raw_hits)))
            )

    return "\n".join(lines) if lines else "No skills found."


def _build_projects_text(resume: ResumeTaggingResponse) -> str:
    """Build formatted project entries for prompt."""
    entries = []
    for proj in resume.context_meta_tags.project_tags.projects:
        parts = [f"- {proj.project_name}"]
        if proj.description:
            parts.append(f"  Description: {proj.description[:200]}")
        if proj.tech_stack:
            parts.append(f"  Tech: {', '.join(proj.tech_stack)}")
        if proj.key_highlights:
            for h in proj.key_highlights[:3]:
                parts.append(f"  • {h}")
        entries.append("\n".join(parts))
    return "\n\n".join(entries) if entries else "No project entries found."


def _jd_skills_text(jd: JDProfile) -> str:
    """Build JD skills list for prompt."""
    lines = []
    for cap in jd.core_capabilities:
        tag = "REQUIRED" if cap.classification == "PRIMARY" else "Preferred"
        lines.append(f"- [{tag}] {cap.normalized_entity}: {cap.raw_text}")
    for cap in jd.secondary_capabilities:
        lines.append(f"- [Preferred] {cap.normalized_entity}: {cap.raw_text}")
    if jd.required_tech_normalized:
        lines.append(
            f"\nTech keywords: {', '.join(jd.required_tech_normalized)}")
    if jd.required_skills_normalized:
        lines.append(
            f"Normalized skills: {', '.join(jd.required_skills_normalized)}")
    return "\n".join(lines)


def _format_jd_experience_requirement(jd: JDProfile) -> str:
    """Format JD experience bounds clearly for LLM prompts."""
    min_y = jd.required_years_min
    max_y = jd.required_years_max

    if min_y is None and max_y is None:
        return jd.required_experience_text or "Not specified"

    if min_y is not None and max_y is not None:
        return (
            f"min_years={min_y:.1f}, max_years={max_y:.1f} "
            "(use min_years as requirement; max_years is preference)"
        )
    if min_y is not None:
        return f"min_years={min_y:.1f}"
    return f"max_years={max_y:.1f} (preference)"


def _parse_section_score(
    raw: dict | None,
    section_name: str,
    detail_key: str = "skills",
) -> SectionScore:
    """Parse LLM JSON into SectionScore, with safe fallbacks."""
    if not raw:
        return SectionScore(
            section_name=section_name,
            score=50.0,
            reasoning="LLM scoring unavailable — using neutral default.",
        )

    score = float(raw.get("score", 50))
    score = max(0.0, min(100.0, score))
    reasoning = raw.get("reasoning", "")

    details = []
    for item in raw.get(detail_key, []):
        if isinstance(item, dict):
            details.append(MatchDetail(
                item=item.get("item", ""),
                match_type=item.get("match_type", "partial"),
                score=max(0.0, min(100.0, float(item.get("score", 50)))),
                explanation=item.get("explanation", ""),
            ))

    return SectionScore(
        section_name=section_name,
        score=score,
        reasoning=reasoning,
        match_details=details,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION SCORERS
# ══════════════════════════════════════════════════════════════════════════════

async def score_role_fit(
    jd: JDProfile,
    resume: ResumeTaggingResponse,
    candidate_years: float,
) -> SectionScore:
    """Assess overall role-fit using o4-mini reasoning model."""
    client = get_reasoning_client()
    if not client:
        return SectionScore(
            section_name="role_fit",
            score=50.0,
            reasoning="LLM unavailable — using neutral role-fit default.",
        )

    # Build candidate profile from resume
    summary = ""
    for tag in resume.context_meta_tags.summary_tags.domain_focus:
        summary += tag + ". "
    for h in resume.context_meta_tags.summary_tags.key_highlights[:3]:
        summary += h + ". "

    exp_tags = resume.context_meta_tags.experience_tags
    current_role = f"{exp_tags.current_role}" if exp_tags.current_role else "Not specified"

    tech_highlights = []
    # Include a wider candidate tech view to avoid false "missing skill"
    # conclusions from only the latest few experience entries.
    for exp in exp_tags.experience_timeline:
        tech_highlights.extend(exp.tech_stack)
    for proj in resume.context_meta_tags.project_tags.projects:
        tech_highlights.extend(proj.tech_stack)
    for entry in resume.global_skill_index:
        if entry.skill:
            tech_highlights.append(entry.skill)
    tech_highlights = list(dict.fromkeys(tech_highlights))[:40]

    caps_text = "; ".join(
        c.normalized_entity for c in jd.core_capabilities[:8])
    skills_text = ", ".join(jd.required_skills_normalized[:10])

    prompt = _ROLE_FIT_PROMPT.format(
        role_family=jd.role_family or "Not specified",
        role_objective=jd.role_objective or "Not specified",
        role_scope=jd.role_scope or "Not specified",
        industry=jd.industry_vertical or "Not specified",
        required_experience=_format_jd_experience_requirement(jd),
        core_capabilities=caps_text,
        required_skills=skills_text,
        candidate_summary=summary or "No summary available",
        domain_focus=", ".join(
            resume.context_meta_tags.summary_tags.domain_focus) or "Not specified",
        candidate_years=f"{candidate_years:.1f} years",
        current_role=current_role,
        strength_signals="; ".join(
            resume.context_meta_tags.resume_strength_signals[:5]),
        tech_highlights=", ".join(tech_highlights),
    )

    try:
        response = await client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4096,
        )
        content = response.choices[0].message.content or ""
        raw = _safe_json_parse(content)
        result = _parse_section_score(raw, "role_fit", "strengths")
        # Enrich with gaps as match details
        if raw and "gaps" in raw:
            for gap in raw["gaps"]:
                result.match_details.append(MatchDetail(
                    item=gap, match_type="gap", score=0, explanation="Missing from candidate profile",
                ))
        if raw and "strengths" in raw:
            strength_details = []
            for s in raw["strengths"]:
                if isinstance(s, str):
                    strength_details.append(MatchDetail(
                        item=s, match_type="exact", score=80, explanation="Strength alignment",
                    ))
            result.match_details = strength_details + result.match_details
        return result
    except Exception as e:
        logger.exception("Role-fit scoring failed: %s", e)
        return SectionScore(
            section_name="role_fit", score=50.0,
            reasoning=f"Role-fit scoring error: {e}",
        )


async def score_experience_section(
    jd: JDProfile,
    resume: ResumeTaggingResponse,
) -> SectionScore:
    """Score experience section against JD using o4-mini."""
    client = get_reasoning_client()
    if not client:
        return SectionScore(
            section_name="experience", score=50.0,
            reasoning="LLM unavailable.",
        )

    caps_text = "; ".join(
        c.normalized_entity for c in jd.core_capabilities[:8])
    skills_text = ", ".join(jd.required_skills_normalized[:10])

    prompt = _EXPERIENCE_PROMPT.format(
        role_family=jd.role_family or "Not specified",
        role_objective=jd.role_objective or "Not specified",
        required_experience=_format_jd_experience_requirement(jd),
        required_skills=skills_text,
        industry=jd.industry_vertical or "Not specified",
        core_capabilities=caps_text,
        experience_entries=_build_experience_text(resume),
    )

    try:
        response = await client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4096,
        )
        content = response.choices[0].message.content or ""
        raw = _safe_json_parse(content)
        return _parse_section_score(raw, "experience", "entries")
    except Exception as e:
        logger.exception("Experience section scoring failed: %s", e)
        return SectionScore(
            section_name="experience", score=50.0,
            reasoning=f"Experience scoring error: {e}",
        )


async def score_skills_section(
    jd: JDProfile,
    resume: ResumeTaggingResponse,
) -> SectionScore:
    """Score skills against JD using gpt-4o-mini."""
    client = get_fast_client()
    if not client:
        return SectionScore(
            section_name="skills", score=50.0,
            reasoning="LLM unavailable.",
        )

    prompt = _SKILLS_PROMPT.format(
        jd_skills=_jd_skills_text(jd),
        candidate_skills=_build_skills_text(resume),
    )

    try:
        response = await client.chat.completions.create(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=3000,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        raw = _safe_json_parse(content)
        return _parse_section_score(raw, "skills", "skills")
    except Exception as e:
        logger.exception("Skills section scoring failed: %s", e)
        return SectionScore(
            section_name="skills", score=50.0,
            reasoning=f"Skills scoring error: {e}",
        )


async def score_projects_section(
    jd: JDProfile,
    resume: ResumeTaggingResponse,
) -> SectionScore:
    """Score project experience against JD using gpt-4o-mini."""
    client = get_fast_client()
    if not client:
        return SectionScore(
            section_name="projects", score=50.0,
            reasoning="LLM unavailable.",
        )

    prompt = _PROJECTS_PROMPT.format(
        role_family=jd.role_family or "Not specified",
        role_objective=jd.role_objective or "Not specified",
        required_skills=", ".join(jd.required_skills_normalized[:10]),
        industry=jd.industry_vertical or "Not specified",
        project_entries=_build_projects_text(resume),
    )

    try:
        response = await client.chat.completions.create(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        raw = _safe_json_parse(content)
        return _parse_section_score(raw, "projects", "projects")
    except Exception as e:
        logger.exception("Projects section scoring failed: %s", e)
        return SectionScore(
            section_name="projects", score=50.0,
            reasoning=f"Projects scoring error: {e}",
        )


async def score_domain_fit(
    jd: JDProfile,
    resume: ResumeTaggingResponse,
) -> SectionScore:
    """Assess domain/industry alignment using o4-mini."""
    client = get_reasoning_client()
    if not client:
        return SectionScore(
            section_name="domain", score=50.0,
            reasoning="LLM unavailable.",
        )

    # Gather candidate domain signals
    candidate_domains = list(
        resume.context_meta_tags.summary_tags.domain_focus)
    candidate_industries = []
    gp = resume.global_parameters
    if gp and gp.product_domain:
        for pd in gp.product_domain:
            val = pd.normalized_value if hasattr(
                pd, "normalized_value") else str(pd)
            candidate_industries.append(val)

    compliance_exposure = []
    if gp and gp.compliance_exposure:
        for ce in gp.compliance_exposure:
            val = ce.normalized_value if hasattr(
                ce, "normalized_value") else str(ce)
            compliance_exposure.append(val)

    prompt = _DOMAIN_PROMPT.format(
        industry=jd.industry_vertical or "Not specified",
        sub_vertical=jd.industry_sub_vertical or "Not specified",
        compliance=", ".join(
            jd.compliance_context) if jd.compliance_context else "None specified",
        business_model=jd.business_model or "Not specified",
        candidate_domains=", ".join(
            candidate_domains) if candidate_domains else "Not specified",
        candidate_industries=", ".join(
            candidate_industries) if candidate_industries else "Not specified",
        compliance_exposure=", ".join(
            compliance_exposure) if compliance_exposure else "None found",
    )

    try:
        response = await client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4096,
        )
        content = response.choices[0].message.content or ""
        raw = _safe_json_parse(content)
        return _parse_section_score(raw, "domain", "matches")
    except Exception as e:
        logger.exception("Domain scoring failed: %s", e)
        return SectionScore(
            section_name="domain", score=50.0,
            reasoning=f"Domain scoring error: {e}",
        )


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

async def score_all_sections(
    jd: JDProfile,
    resume: ResumeTaggingResponse,
    candidate_years: float,
) -> list[SectionScore]:
    """Run all section scorers in parallel. Returns list of SectionScores.

    Projects section is only scored for candidates with no work experience
    (freshers). For candidates with experience entries, projects are skipped
    since their work history already demonstrates capability.
    """
    import asyncio

    has_experience = bool(
        resume.context_meta_tags.experience_tags.experience_timeline
    )

    tasks = [
        score_role_fit(jd, resume, candidate_years),
        score_experience_section(jd, resume),
        score_skills_section(jd, resume),
    ]
    names = ["role_fit", "experience", "skills"]

    # Only score projects for candidates with no work experience
    if not has_experience:
        tasks.append(score_projects_section(jd, resume))
        names.append("projects")

    tasks.append(score_domain_fit(jd, resume))
    names.append("domain")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    sections: list[SectionScore] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.exception("Section scorer %s failed: %s", names[i], r)
            sections.append(SectionScore(
                section_name=names[i], score=50.0,
                reasoning=f"Scoring failed: {r}",
            ))
        else:
            sections.append(r)

    return sections
