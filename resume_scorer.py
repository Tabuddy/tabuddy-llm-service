"""Resume Scorer — Deterministic 6-Dimension Scoring Engine.

Zero LLM calls. All scoring is pure arithmetic based on structured data
from both the JDProfile and the ResumeTaggingResponse.

Dimensions & Adaptive Weights:
┌─────────────────────────┬────────┬──────┬─────────┐
│ Dimension               │ Senior │ Mid  │ Fresher │
├─────────────────────────┼────────┼──────┼─────────┤
│ Tier Alignment          │  15%   │  15% │   15%   │
│ Core Capability Match   │  25%   │  30% │   15%   │
│ Skill Match             │  10%   │  20% │   35%   │
│ Experience Quality      │  30%   │  20% │    5%   │
│ Scale & Impact          │  10%   │   5% │    5%   │
│ Domain & Context        │  10%   │  10% │   25%   │
└─────────────────────────┴────────┴──────┴─────────┘
"""

from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING

from ranking_models import (
    ExperienceLevel,
    DimensionScore,
    ResumeRankResult,
    JDProfile,
    TierClassification,
)
from models import ResumeTaggingResponse  # existing model from the service
from skills_dictionary import SKILL_ALIASES

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Weight Profiles (must sum to 1.0) ─────────────────────────────────────────
_WEIGHTS: dict[ExperienceLevel, dict[str, float]] = {
    ExperienceLevel.SENIOR: {
        "tier_alignment": 0.15,
        "capability_match": 0.25,
        "skill_match": 0.10,
        "experience_quality": 0.30,
        "scale_impact": 0.10,
        "domain_context": 0.10,
    },
    ExperienceLevel.MID: {
        "tier_alignment": 0.15,
        "capability_match": 0.30,
        "skill_match": 0.20,
        "experience_quality": 0.20,
        "scale_impact": 0.05,
        "domain_context": 0.10,
    },
    ExperienceLevel.FRESHER: {
        "tier_alignment": 0.15,
        "capability_match": 0.15,
        "skill_match": 0.35,
        "experience_quality": 0.05,
        "scale_impact": 0.05,
        "domain_context": 0.25,
    },
}

# Tier hierarchy paths for matching
_TIER1_LABELS = {"Digital", "Physical", "Service"}
_TIER2_LABELS = {
    "App_Engineering", "Data_Intelligence", "Infra_Cloud",
    "Product_Design", "Cyber_Security",
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Experience Level Detection
# ══════════════════════════════════════════════════════════════════════════════

MONTH_NAMES = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6,
    "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_NOW = 2026  # April 2026; kept static to avoid test-time drift


def _month_index(name: str) -> int | None:
    """Map a month name/abbrev to 1-12, or None."""
    return MONTH_NAMES.get(name.lower().strip())


def _parse_months_from_duration(duration: str) -> float:
    """Parse a duration string into fractional years.

    Handles:
      - "Aug 2025 - Present" → months from start to now / 12
      - "May 2024 - July 2024" → exact month diff / 12
      - "6 months", "1 year 3 months" → explicit counts
      - "2024 - 2025" → year-only ranges
    """
    d = duration.lower().strip()

    # Explicit month mentions: "6 months", "8 months"
    explicit_months = re.search(r"(\d+(?:\.\d+)?)\s*months?", d)
    if explicit_months:
        return float(explicit_months.group(1)) / 12.0

    # "1 year 3 months", "2 years 6 months"
    year_month_match = re.search(r"(\d+(?:\.\d+)?)\s*years?\s*(\d+)\s*months?", d)
    if year_month_match:
        return float(year_month_match.group(1)) + float(year_month_match.group(2)) / 12.0

    # Just years: "2 years", "1.5 year"
    year_only = re.search(r"(\d+(?:\.\d+)?)\s*year", d)
    if year_only:
        return float(year_only.group(1))

    # Date range: "Aug 2025 - Present", "May 2024 - July 2024"
    range_pat = re.compile(
        r"([a-z]+)[\s]+(\d{4})\s*[-–—to]+\s*(present|current|([a-z]+)\s+(\d{4}))"
    )
    m = range_pat.search(d)
    if m:
        start_mon = _month_index(m.group(1))
        start_year = int(m.group(2))
        if start_mon is None:
            return 0.0

        if m.group(3) and m.group(3).lower() in ("present", "current"):
            end_mon = 4  # April
            end_year = _NOW
        else:
            end_mon = _month_index(m.group(4))
            end_year = int(m.group(5)) if m.group(5) else _NOW
            if end_mon is None:
                return 0.0

        total = (end_year - start_year) * 12 + (end_mon - start_mon)
        return max(total / 12.0, 0.083)  # minimum 1 month

    # Year-only range: "2024 - 2025"
    year_range = re.search(r"(\d{4})\s*[-–—to]+\s*(\d{4})", d)
    if year_range:
        yrs = int(year_range.group(2)) - int(year_range.group(1))
        return max(float(yrs), 0.5)

    return 0.0


_ROLES_INTERNSHIP = {"intern", "trainee", "co-op", "co_op", "apprentice"}


def _is_internship_role(role: str) -> bool:
    """Check if a role string indicates an internship/apprenticeship."""
    role_lower = role.lower()
    return any(r in role_lower for r in _ROLES_INTERNSHIP)


def _parse_years_from_text(text: str) -> float | None:
    """Extract numeric years from strings like '5 years', '7+ years', '3.5 years'."""
    m = re.search(r"(\d+(?:\.\d+)?)\+?\s*year", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def estimate_candidate_years(resume: ResumeTaggingResponse) -> float:
    """Estimate the candidate's total years of experience from resume meta-tags.

    Priority:
      1. Explicit year count from summary_tags.years_of_experience
      2. Parse each timeline entry's duration to months (supports months for freshers)
      3. Approximate from timeline: intern roles -> 0.5yr each, others -> 2yr each
      4. Fallback to 0
    """
    # 1. Explicit years statement
    yoe_str = resume.context_meta_tags.summary_tags.years_of_experience
    if yoe_str:
        yrs = _parse_years_from_text(yoe_str)
        if yrs is not None:
            return yrs

    # 2. Parse date ranges from experience timeline using month-aware parsing
    total_years: float = 0.0
    all_internships = True
    for exp in resume.context_meta_tags.experience_tags.experience_timeline:
        if exp.duration:
            months = _parse_months_from_duration(exp.duration)
            total_years += months
            if not _is_internship_role(exp.role or ""):
                all_internships = False
        else:
            all_internships = False

    if total_years > 0:
        return total_years

    # 3. Fallback from quantifiers
    all_quants = resume.context_meta_tags.experience_tags.top_quantifiers
    max_years_found: float = 0.0
    for q in all_quants:
        y = _parse_years_from_text(q)
        if y and y > max_years_found:
            max_years_found = y

    if max_years_found > 0:
        return max_years_found

    # 4. Rough estimate from number of companies — internship-aware
    total_companies = resume.context_meta_tags.experience_tags.total_companies
    if total_companies > 0:
        timeline = resume.context_meta_tags.experience_tags.experience_timeline
        intern_count = sum(1 for e in timeline if _is_internship_role(e.role or ""))
        if intern_count == total_companies:
            # All roles are internships — ~6 months each
            return max(total_companies * 0.5, 0.25)
        return max(total_companies * 2.0, 0.5)

    return 0.0


def detect_experience_level(resume: ResumeTaggingResponse) -> ExperienceLevel:
    """Determine experience level from resume meta-tags — purely deterministic.

    Priority:
      1. Explicit year count from summary_tags.years_of_experience
      2. Approximate from experience timeline count + leadership signals
      3. Presence of experience blocks vs skills_dump only
    """
    years = estimate_candidate_years(resume)
    if years >= 7:
        return ExperienceLevel.SENIOR
    elif years >= 3:
        return ExperienceLevel.MID
    else:
        return ExperienceLevel.FRESHER


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Fuzzy + Exact Skill Matching Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_term(text: str) -> str:
    """Lowercase, strip punctuation, collapse spaces."""
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def _fuzzy_match_score(a: str, b: str) -> float:
    """Jaccard similarity on word-tokens for soft matching (0.0-1.0)."""
    tokens_a = set(_normalize_term(a).split())
    tokens_b = set(_normalize_term(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _normalize_to_canonical(skill: str) -> str | None:
    """If skill (lowercased) is in SKILL_ALIASES, return canonical name.

    Also checks if the normalized form matches any alias key. Returns None
    if no alias is found so the caller falls back to the original string.
    """
    norm = _normalize_term(skill)
    if not norm:
        return None
    # Direct alias lookup
    if norm in SKILL_ALIASES:
        return SKILL_ALIASES[norm]
    # Check if any alias (normalized) matches our input
    for alias_key, canonical in SKILL_ALIASES.items():
        if _normalize_term(alias_key) == norm:
            return canonical
    return None


def _semantic_similarity_score(query: str, candidates: list[str]) -> float:
    """Find best semantic (cosine similarity) match using SentenceTransformer.

    Returns score in [0.0, 1.0]. Falls back to Jaccard if model unavailable.
    """
    from semantic_matcher import semantic_best_match
    return semantic_best_match(query, candidates)


def _best_match_score(query: str, candidates: list[str], threshold: float = 0.4) -> float:
    """Find highest match score between query and a list of candidates.

    Three-tier matching:
      1. Exact / substring match → returns 1.0 immediately
      2. Dictionary-normalized exact/substring match → returns 1.0
      3. Jaccard token overlap → returns score if >= threshold
      4. Semantic embedding similarity (fallback) → returns score
         if semantic >= 0.75 (full) or >= 0.6 (partial)
    """
    if not candidates:
        return 0.0

    norm_q = _normalize_term(query)
    if not norm_q:
        return 0.0

    # Normalize candidates too
    norm_candidates = [_normalize_term(c) for c in candidates if _normalize_term(c)]
    if not norm_candidates:
        return 0.0

    # Tier 1: Exact / substring match on normalized forms
    for norm_c in norm_candidates:
        if norm_q == norm_c or norm_q in norm_c or norm_c in norm_q:
            return 1.0

    # Tier 2: Dictionary-normalized match
    query_canonical = _normalize_to_canonical(query)
    if query_canonical:
        norm_q_canonical = _normalize_term(query_canonical)
        for idx, norm_c in enumerate(norm_candidates):
            # Check if candidate normalizes to same canonical
            cand_canonical = _normalize_to_canonical(candidates[idx])
            if cand_canonical:
                norm_c_canonical = _normalize_term(cand_canonical)
                if (norm_q_canonical == norm_c_canonical
                        or norm_q_canonical in norm_c_canonical
                        or norm_c_canonical in norm_q_canonical):
                    return 0.9  # Very high but not 1.0 to distinguish from exact

    # Tier 3: Jaccard token overlap
    best = 0.0
    for norm_c in norm_candidates:
        score = _fuzzy_match_score(norm_q, norm_c)
        if score > best:
            best = score

    if best >= threshold:
        return best

    # Tier 4: Semantic embedding similarity (for contextually similar skills)
    # e.g. "CI/CD Automation" vs "CI/CD Implementation"
    semantic_score = _semantic_similarity_score(norm_q, norm_candidates)
    if semantic_score >= 0.55:
        # Contextual semantic match — score maps to cosine similarity
        return max(best, min(semantic_score, 0.7))

    # Tier 5: LLM fallback — light model check for edge cases
    # e.g. "Microservices Architecture" vs "Service-Oriented Design"
    for c in norm_candidates:
        if _match_skill_llm(norm_q, c):
            return max(best, 0.35)  # Weak match, but non-zero

    return 0.0


# ── LLM-based skill context fallback ──────────────────────────────────────────
_LLM_SKILL_CACHE: dict[str, bool] = {}


def _match_skill_llm(skill_a: str, skill_b: str) -> bool:
    """Use gpt-4.1-nano to check if two skill strings are contextually related.

    Only fires when all 4 tiers return no match. Results are cached to avoid
    redundant API calls.

    Returns True if the LLM judges the skills as related, False otherwise.
    """
    # Normalize key for caching (order-independent)
    key = "|".join(sorted([skill_a, skill_b]))
    if key in _LLM_SKILL_CACHE:
        return _LLM_SKILL_CACHE[key]

    result = _skill_llm_call(skill_a, skill_b)
    _LLM_SKILL_CACHE[key] = result
    return result


def _skill_llm_call(skill_a: str, skill_b: str) -> bool:
    """Single LLM call to check contextual skill relation."""
    try:
        import os
        from openai import AzureOpenAI

        api_key = os.getenv("AZURE_OPEN_AI_KEY")
        if not api_key:
            return False

        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint="https://tabuddy-azure-sponsor.openai.azure.com/",
            api_version="2024-12-01-preview",
        )

        prompt = (
            f"Are these two skills contextually related in the software "
            f"engineering domain?\n"
            f"Skill 1: {skill_a}\n"
            f"Skill 2: {skill_b}\n\n"
            f"Answer YES or NO only."
        )

        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        content = (response.choices[0].message.content or "").strip().upper()
        return content.startswith("YES")
    except Exception as e:
        logger.debug("LLM skill match failed for %r vs %r: %s", skill_a, skill_b, e)
        return False


def _collect_resume_skills(resume: ResumeTaggingResponse) -> dict[str, str]:
    """Collect all resume skills with their provenance.

    Returns: { normalized_skill: provenance_type }
    Where provenance_type: 'contextual' (from experience/project) or 'listed' (skills_dump)
    """
    skills: dict[str, str] = {}

    # Skills from global_skill_index with provenance tracking
    for entry in resume.global_skill_index:
        norm = _normalize_term(entry.skill)
        if not norm:
            continue
        is_contextual = any(
            t.action_verb.lower() not in {"listed"}
            for t in entry.traces
        )
        if is_contextual:
            skills[norm] = "contextual"
        elif norm not in skills:
            skills[norm] = "listed"

    # Tech stack from experience blocks — definitively contextual
    for exp in resume.context_meta_tags.experience_tags.experience_timeline:
        for tech in exp.tech_stack:
            norm = _normalize_term(tech)
            if norm and len(norm) > 1:
                skills[norm] = "contextual"

    # Tech stack from project blocks — also contextual
    for proj in resume.context_meta_tags.project_tags.projects:
        for tech in proj.tech_stack:
            norm = _normalize_term(tech)
            if norm and len(norm) > 1:
                skills[norm] = "contextual"

    # Skill categories — listed (don't overwrite contextual)
    sc = resume.context_meta_tags.skill_categories
    for cat_list in [sc.languages, sc.frameworks, sc.databases,
                     sc.cloud_devops, sc.tools, sc.methodologies]:
        for s in cat_list:
            norm = _normalize_term(s)
            if norm and len(norm) > 1 and norm not in skills:
                skills[norm] = "listed"

    return skills


def _collect_all_resume_tech(resume: ResumeTaggingResponse) -> list[str]:
    """Collect all tech stack items from experience + project timelines."""
    tech: list[str] = []
    for exp in resume.context_meta_tags.experience_tags.experience_timeline:
        tech.extend(exp.tech_stack)
    for proj in resume.context_meta_tags.project_tags.projects:
        tech.extend(proj.tech_stack)
    return tech


# Generic words that don't represent specific technologies
_GENERIC_SKILL_WORDS = {
    "development", "implementation", "engineering", "management", "design",
    "architecture", "integration", "optimization", "methodology", "monitoring",
    "deployment", "testing", "analysis", "building", "creating", "maintaining",
    "configuration", "administration", "operations", "automation", "programming",
    "best", "practices", "and", "the", "for", "with", "using",
}


def _decompose_compound_skill(compound: str) -> list[str]:
    """Break compound capability terms into individual tech words for matching.

    E.g., 'docker implementation' → ['docker']
          'restful api development' → ['restful', 'api']
          'full-stack development' → ['fullstack', 'full', 'stack']
    """
    words = _normalize_term(compound).split()
    return [w for w in words if w not in _GENERIC_SKILL_WORDS and len(w) > 1]


# ══════════════════════════════════════════════════════════════════════════════
# DIMENSION SCORERS
# ══════════════════════════════════════════════════════════════════════════════

def score_tier_alignment(
    jd_tier: TierClassification,
    resume_tier: TierClassification,
) -> tuple[float, list[str]]:
    """Score how well the resume's tier hierarchy matches the JD.

    Returns (score_0_100, evidence_list)
    """
    evidence: list[str] = []
    jd_t1 = jd_tier.tier1.label
    res_t1 = resume_tier.tier1.label

    if jd_t1 == "Unknown" or res_t1 == "Unknown":
        evidence.append("Classification unavailable — tier alignment not assessed")
        return 50.0, evidence

    # Tier 1 match
    if jd_t1 != res_t1:
        evidence.append(f"Tier 1 mismatch: JD={jd_t1}, Resume={res_t1}")
        return 10.0, evidence

    evidence.append(f"✓ Tier 1 match: {jd_t1}")
    score = 40.0

    # Tier 2 match (if applicable)
    jd_t2 = jd_tier.tier2.label if jd_tier.tier2 else None
    res_t2 = resume_tier.tier2.label if resume_tier.tier2 else None

    if jd_t2 and res_t2:
        if jd_t2 == res_t2:
            score += 40.0
            evidence.append(f"✓ Tier 2 match: {jd_t2}")
        else:
            score += 10.0
            evidence.append(f"△ Tier 2 partial: JD={jd_t2}, Resume={res_t2}")
    elif jd_t2 and not res_t2:
        score += 5.0
        evidence.append(f"△ JD Tier 2={jd_t2}, resume unclassified at Tier 2")
    else:
        score += 20.0  # JD doesn't need tier 2, same tier 1 is enough

    # Tier 3 match (if applicable)
    jd_t3 = jd_tier.tier3.label if jd_tier.tier3 else None
    res_t3 = resume_tier.tier3.label if resume_tier.tier3 else None

    if jd_t3 and res_t3:
        if jd_t3 == res_t3:
            score += 20.0
            evidence.append(f"✓ Tier 3 match: {jd_t3}")
        else:
            score += 5.0
            evidence.append(f"△ Tier 3 partial: JD={jd_t3}, Resume={res_t3}")
    elif not jd_t3:
        score += 20.0  # JD doesn't specify tier 3

    return min(score, 100.0), evidence


def score_capability_match(
    jd: JDProfile,
    resume: ResumeTaggingResponse,
    exp_level: ExperienceLevel,
) -> tuple[float, list[str]]:
    """Match JD core/secondary capabilities against resume meta-tags.

    PRIMARY capabilities weighted 2x, ownership level affects weight.
    For freshers, also checks project descriptions.
    """
    evidence: list[str] = []
    all_caps = jd.core_capabilities + jd.secondary_capabilities + jd.adjacent_capabilities

    if not all_caps:
        evidence.append("No capability atoms extracted from JD")
        return 50.0, evidence

    # Collect resume capability signals
    resume_skill_set = _collect_resume_skills(resume)
    resume_tech = [_normalize_term(t) for t in _collect_all_resume_tech(resume)]
    resume_signals: list[str] = list(resume_skill_set.keys()) + resume_tech

    # Also include strength signals and domain focus
    for sig in resume.context_meta_tags.resume_strength_signals:
        resume_signals.append(_normalize_term(sig))
    for focus in resume.context_meta_tags.summary_tags.domain_focus:
        resume_signals.append(_normalize_term(focus))

    total_weight = 0.0
    matched_weight = 0.0

    for cap in all_caps:
        # Weight: PRIMARY = 2x, SECONDARY = 1x, ownership bonus
        base_w = 2.0 if cap.classification == "PRIMARY" else 1.0
        if cap.ownership_level == "Primary Owner":
            base_w *= 1.2
        elif cap.ownership_level == "Support":
            base_w *= 0.8

        total_weight += base_w
        match = _best_match_score(cap.normalized_entity, resume_signals)
        matched_weight += base_w * match

        if match >= 0.8:
            evidence.append(f"✓ {cap.normalized_entity} ({cap.classification})")
        elif match >= 0.4:
            evidence.append(f"△ ~{cap.normalized_entity} (partial)")

    score = (matched_weight / total_weight) * 100 if total_weight else 0.0
    no_match_count = sum(
        1 for cap in all_caps
        if _best_match_score(cap.normalized_entity, resume_signals) < 0.3
    )
    if no_match_count:
        evidence.append(f"✗ {no_match_count} capability/ies not found in resume")

    return min(score, 100.0), evidence


def score_skill_match(
    jd: JDProfile,
    resume: ResumeTaggingResponse,
    exp_level: ExperienceLevel,
) -> tuple[float, list[str]]:
    """Match JD required skills against resume skills.

    - Skills used in experience/project blocks score 1.5x vs listed skills
    - Compound JD terms are decomposed for better atomic matching
    - Co-dependency bonus for bundled skills
    - Also matches individual tech keywords extracted from JD text
    """
    evidence: list[str] = []
    required = jd.required_skills_normalized
    required_tech = jd.required_tech_normalized

    if not required and not required_tech:
        evidence.append("No specific skills extracted from JD")
        return 60.0, evidence

    resume_skills = _collect_resume_skills(resume)
    all_resume_skills_norm = list(resume_skills.keys())

    # ── Match compound capability terms ─────────────────────────────────────
    compound_matched = 0.0
    compound_total = len(required)
    compound_missing: list[str] = []

    for req_skill in required:
        best_score = _best_match_score(req_skill, all_resume_skills_norm)

        # If compound term doesn't match well, decompose and try parts
        if best_score < 0.6:
            parts = _decompose_compound_skill(req_skill)
            for part in parts:
                part_score = _best_match_score(part, all_resume_skills_norm)
                if part_score > best_score:
                    best_score = part_score

        if best_score >= 0.8:
            # Check provenance: contextual or listed?
            prov = resume_skills.get(_normalize_term(req_skill), "listed")
            if prov != "contextual":
                parts = _decompose_compound_skill(req_skill)
                for part in parts:
                    if resume_skills.get(part) == "contextual":
                        prov = "contextual"
                        break
            if prov == "contextual":
                compound_matched += 1.5
                evidence.append(f"✓ {req_skill} (experienced in context)")
            else:
                compound_matched += 1.0
                evidence.append(f"✓ {req_skill} (listed)")
        elif best_score >= 0.4:
            compound_matched += 0.5
            evidence.append(f"△ ~{req_skill} (partial match)")
        else:
            compound_missing.append(req_skill)

    # ── Match individual tech keywords ──────────────────────────────────────
    tech_matched = 0.0
    tech_total = 0
    for tech in required_tech:
        # Skip terms already covered by compound capability matches
        if any(_normalize_term(tech) in _normalize_term(r) for r in required):
            continue
        tech_total += 1
        match = _best_match_score(tech, all_resume_skills_norm)
        if match >= 0.8:
            prov = resume_skills.get(tech, "listed")
            if prov == "contextual":
                tech_matched += 1.5
                evidence.append(f"✓ {tech} (experienced in context)")
            else:
                tech_matched += 1.0
                evidence.append(f"✓ {tech} (listed)")
        elif match >= 0.4:
            tech_matched += 0.5

    # ── Combined score ──────────────────────────────────────────────────────
    total = compound_total + tech_total
    matched = compound_matched + tech_matched
    base_score = (matched / total) * 100 if total else 0.0

    # Co-dependency bonus
    bundle_bonus = _score_bundle_codependency(jd, resume_skills, all_resume_skills_norm)
    if bundle_bonus > 0:
        evidence.append(f"✓ Co-dependent skill bundle bonus: +{bundle_bonus:.0f}pts")

    final_score = min(base_score + bundle_bonus, 100.0)

    if compound_missing[:3]:
        evidence.append(f"✗ Missing: {', '.join(compound_missing[:3])}")

    return final_score, evidence


def _score_bundle_codependency(
    jd: JDProfile,
    resume_skills: dict[str, str],
    all_resume_skills: list[str],
) -> float:
    """Bonus for co-occurring skill groups (e.g. React + Redux, K8s + Terraform)."""
    bonus = 0.0
    # Group JD capabilities by bundle_id
    bundles: dict[str, list[str]] = {}
    for cap in jd.core_capabilities + jd.secondary_capabilities:
        bid = cap.bundle_id
        if bid:
            bundles.setdefault(bid, []).append(cap.normalized_entity)

    for bid, skills in bundles.items():
        if len(skills) < 2:
            continue
        # Check if resume has ALL skills in this bundle
        matches = sum(
            1 for s in skills
            if _best_match_score(s, all_resume_skills) >= 0.7
        )
        if matches == len(skills):
            bonus += 5.0  # Full bundle match bonus
        elif matches >= len(skills) * 0.6:
            bonus += 2.0  # Partial bundle match

    return min(bonus, 15.0)  # Cap bonus at 15 points


def score_experience_quality(
    jd: JDProfile,
    resume: ResumeTaggingResponse,
    exp_level: ExperienceLevel,
) -> tuple[float, list[str]]:
    """Score the quality and relevance of experience history.

    Sub-scores:
    - Role relevance: how well resume roles match JD role_family / capabilities
    - Achievement density: quantifiers and accomplishments per role
    - Recency: current role relevance
    - Trajectory: leadership progression signals
    """
    evidence: list[str] = []
    timeline = resume.context_meta_tags.experience_tags.experience_timeline

    if not timeline:
        if exp_level == ExperienceLevel.FRESHER:
            evidence.append("No experience blocks — expected for fresher")
            return 60.0, evidence  # Neutral for freshers
        evidence.append("No work experience blocks detected")
        return 10.0, evidence

    scores: list[float] = []

    # Build JD signal words for matching
    jd_signals: list[str] = []
    if jd.role_family:
        jd_signals.append(jd.role_family)
    if jd.role_objective:
        jd_signals.append(jd.role_objective)
    for cap in jd.core_capabilities:
        jd_signals.append(cap.normalized_entity)
    jd_signals_norm = [_normalize_term(s) for s in jd_signals]

    for i, exp in enumerate(timeline):
        role_str = f"{exp.role or ''} {exp.company or ''}"
        # Also match tech_stack against JD signals for relevance
        tech_signals = [_normalize_term(t) for t in exp.tech_stack]
        role_relevance = _best_match_score(role_str, jd_signals_norm) if jd_signals_norm else 0.5
        tech_relevance = max(
            (_best_match_score(t, jd_signals_norm) for t in tech_signals),
            default=0.0,
        ) if tech_signals and jd_signals_norm else 0.0
        relevance = max(role_relevance, tech_relevance)

        # Achievement quality: count quantified achievements
        quant_count = len(exp.quantifiers)
        achievement_count = len(exp.key_achievements)
        # Fallback: count achievements that contain numbers as pseudo-quantifiers
        if quant_count == 0 and achievement_count > 0:
            quant_count = sum(
                1 for ach in exp.key_achievements
                if re.search(r'\d+', ach)
            )
        quality_boost = min(quant_count * 5 + achievement_count * 3, 30)

        # Recency: current role or most recent (first in timeline = most recent)
        recency_factor = 1.3 if (exp.is_current or i == 0) else 1.0

        role_score = min((relevance * 60 + quality_boost) * recency_factor, 100)
        scores.append(role_score)

        total_metrics = quant_count + achievement_count
        if relevance >= 0.5:
            evidence.append(f"✓ {exp.role or 'Role'} at {exp.company or 'Co'}: relevant ({total_metrics} metrics)")
        elif relevance >= 0.2:
            evidence.append(f"△ {exp.role or 'Role'} at {exp.company or 'Co'}: adjacent role ({total_metrics} metrics)")

    # Leadership trajectory bonus
    leadership = resume.global_parameters.leadership_footprint
    if leadership:
        evidence.append(f"✓ Leadership signals: {len(leadership)} entries")

    base_score = sum(scores) / len(scores) if scores else 0.0

    # Leadership bonus for senior/mid roles
    leadership_bonus = min(len(leadership) * 5, 15) if exp_level in (
        ExperienceLevel.SENIOR, ExperienceLevel.MID
    ) else 0
    if leadership_bonus:
        evidence.append(f"  Leadership bonus: +{leadership_bonus}pts")

    return min(base_score + leadership_bonus, 100.0), evidence


def score_scale_and_impact(
    jd: JDProfile,
    resume: ResumeTaggingResponse,
) -> tuple[float, list[str]]:
    """Match scale and impact signals between JD and resume.

    Checks: scale_anchors, quantified outcomes, measurable_outcomes.
    Falls back to key_achievements when pipeline quantifiers are sparse.
    """
    evidence: list[str] = []

    # Collect resume scale signals
    resume_scale: list[str] = []
    for anchor in resume.global_parameters.scale_anchors:
        resume_scale.append(
            _normalize_term(f"{anchor.extracted_metric} {anchor.extracted_unit}")
        )
    for q in resume.context_meta_tags.experience_tags.top_quantifiers[:20]:
        resume_scale.append(_normalize_term(q))
    for sig in resume.context_meta_tags.resume_strength_signals:
        resume_scale.append(_normalize_term(sig))

    # Fallback: also pull metrics from key_achievements and project highlights
    for exp in resume.context_meta_tags.experience_tags.experience_timeline:
        for ach in exp.key_achievements:
            resume_scale.append(_normalize_term(ach))
        for q in exp.quantifiers:
            resume_scale.append(_normalize_term(q))
    for proj in resume.context_meta_tags.project_tags.projects:
        for h in proj.key_highlights:
            resume_scale.append(_normalize_term(h))
        for q in proj.quantifiers:
            resume_scale.append(_normalize_term(q))

    # Count all quantified items (achievements + quantifiers)
    quant_count = (
        len(resume.context_meta_tags.experience_tags.top_quantifiers)
        + sum(len(e.quantifiers) for e in resume.context_meta_tags.experience_tags.experience_timeline)
        + sum(len(p.quantifiers) for p in resume.context_meta_tags.project_tags.projects)
    )

    if not resume_scale:
        evidence.append("No scale or impact metrics found in resume")
        return 20.0, evidence

    # Score JD scale atoms against resume scale
    total_atoms = len(jd.scale_atoms) + len(jd.impact_atoms)

    if total_atoms == 0:
        # No specific scale requirements in JD — resume having metrics is a bonus
        bonus = min(len(resume.global_parameters.scale_anchors) * 10 + quant_count * 3, 40)
        evidence.append(f"No specific scale requirements in JD. Resume has {quant_count} quantified items")
        return 50.0 + bonus, evidence

    matched = 0.0
    for atom in jd.scale_atoms + jd.impact_atoms:
        match = _best_match_score(atom.normalized_entity, resume_scale)
        matched += match
        if match >= 0.5:
            evidence.append(f"✓ Scale/impact: {atom.normalized_entity}")

    score = (matched / total_atoms) * 70  # Base

    # Bonus for having quantified achievements when JD demands it
    if quant_count >= 5:
        score += 20
        evidence.append(f"✓ {quant_count} quantified achievements found")
    elif quant_count >= 2:
        score += 10
        evidence.append(f"△ {quant_count} quantified achievements")
    elif quant_count >= 1:
        score += 5

    return min(score, 100.0), evidence


def score_domain_context(
    jd: JDProfile,
    resume: ResumeTaggingResponse,
) -> tuple[float, list[str]]:
    """Score domain alignment: industry, compliance, architecture, methodology.

    This is especially important for freshers (higher weight) as it signals
    whether their projects/education align with the JD domain.
    """
    evidence: list[str] = []
    sub_scores: list[float] = []

    # 1. Industry/Product Domain match
    jd_domains: list[str] = []
    if jd.industry_vertical:
        jd_domains.append(jd.industry_vertical)
    if jd.industry_sub_vertical:
        jd_domains.append(jd.industry_sub_vertical)

    resume_domains: list[str] = []
    for pd in resume.global_parameters.product_domain:
        resume_domains.append(pd.normalized_value)
    for sig in resume.context_meta_tags.resume_strength_signals:
        resume_domains.append(sig)

    if jd_domains and resume_domains:
        domain_match = max(
            _best_match_score(jd_d, resume_domains)
            for jd_d in jd_domains
        )
        sub_scores.append(domain_match * 100)
        if domain_match >= 0.5:
            evidence.append(f"✓ Domain match: {jd_domains[0]}")
        else:
            evidence.append(f"△ Domain partial: JD={jd_domains[0]}")
    elif not jd_domains:
        sub_scores.append(60.0)  # Neutral if JD doesn't specify
        evidence.append("No specific domain requirement in JD")

    # 2. Architectural paradigm match
    jd_arch: list[str] = [a.normalized_entity for a in jd.process_methodology_atoms if a.normalized_entity]
    resume_arch: list[str] = [a.normalized_value for a in resume.global_parameters.architectural_paradigm]
    resume_arch += [a.normalized_value for a in resume.global_parameters.codebase_lifecycle]

    if jd_arch and resume_arch:
        arch_match = max(
            (_best_match_score(ja, resume_arch) for ja in jd_arch),
            default=0.0,
        )
        sub_scores.append(arch_match * 100)
        if arch_match >= 0.5:
            evidence.append(f"✓ Arch/methodology match")
    else:
        sub_scores.append(50.0)

    # 3. Compliance match
    jd_compliance = jd.compliance_context
    resume_compliance = [c.normalized_value for c in resume.global_parameters.compliance_exposure]

    if jd_compliance and resume_compliance:
        compliance_hits = sum(
            1 for jc in jd_compliance
            if _best_match_score(jc, resume_compliance) >= 0.6
        )
        comp_score = (compliance_hits / len(jd_compliance)) * 100
        sub_scores.append(comp_score)
        if compliance_hits:
            evidence.append(f"✓ Compliance overlap: {compliance_hits}/{len(jd_compliance)} requirements")
        else:
            evidence.append(f"✗ No compliance overlap (JD requires: {', '.join(jd_compliance[:2])})")
    elif jd_compliance and not resume_compliance:
        sub_scores.append(10.0)  # Penalty for missing required compliance
        evidence.append(f"✗ Missing compliance: {', '.join(jd_compliance[:2])}")
    else:
        sub_scores.append(70.0)  # No compliance requirement

    # 4. Cross-functional area match
    jd_cf: list[str] = [a.normalized_entity for a in jd.stakeholder_atoms]
    resume_cf: list[str] = [a.normalized_value for a in resume.global_parameters.cross_functional_area]

    if jd_cf and resume_cf:
        cf_match = max((_best_match_score(j, resume_cf) for j in jd_cf), default=0.0)
        sub_scores.append(cf_match * 100)
        if cf_match >= 0.5:
            evidence.append(f"✓ Cross-functional match")
    else:
        sub_scores.append(50.0)

    final_score = sum(sub_scores) / len(sub_scores) if sub_scores else 50.0
    return min(final_score, 100.0), evidence


def score_experience_adequacy(
    jd: JDProfile,
    candidate_years: float,
) -> tuple[float, list[str] | None]:
    """Score candidate's experience against JD requirements.

    Returns (score_0_100, evidence_or_None). If JD has no experience requirements,
    returns (None, None) so this dimension is excluded from scoring.

    Penalty profile:
      - candidate_years >= JD min: 100.0 (fully qualified)
      - Slight gap (<50% short): proportional penalty
      - Severe gap (>=50% short): hard penalty, score floored at 10
      - No experience at all for senior JD: 5.0
    """
    evidence: list[str] = []
    jd_min = jd.required_years_min
    jd_max = jd.required_years_max

    if jd_min is None and jd.expected_seniority is None:
        return None, None  # No experience requirement in JD

    # Determine effective minimum
    if jd_min is None:
        seniority_min = {
            "fresher": 0.0,
            "junior": 1.0,
            "mid": 3.0,
            "senior": 7.0,
            "principal": 10.0,
        }
        jd_min = seniority_min.get(jd.expected_seniority or "", 0.0)

    if jd_min <= 0:
        return None, None  # Any experience level acceptable

    # Calculate gap
    gap = jd_min - candidate_years

    if gap <= 0:
        # Candidate meets or exceeds requirement
        evidence.append(
            f"Experience adequate: {candidate_years:.1f} yrs >= {jd_min:.1f} required"
        )
        if jd_max and candidate_years > jd_max:
            bonus = 95.0  # Slightly over is fine but not ideal
            evidence.append(
                f"Note: {candidate_years:.1f} yrs exceeds {jd_max:.1f} max (no penalty)"
            )
            return bonus, evidence
        return 100.0, evidence

    # Candidate is short — compute penalty
    gap_ratio = gap / jd_min  # 0.0 = at threshold, 1.0 = 0 experience

    if gap_ratio < 0.3:
        # Within 30% of requirement — slight penalty
        score = 100.0 - (gap_ratio / 0.3) * 20  # 80-100
        evidence.append(
            f"Near threshold: {candidate_years:.1f} yrs (need {jd_min:.1f}, ~{gap:.1f} short)"
        )
    elif gap_ratio < 0.6:
        # 30-60% short — moderate penalty
        score = 80.0 - ((gap_ratio - 0.3) / 0.3) * 40  # 40-80
        evidence.append(
            f"Experience gap: {candidate_years:.1f} yrs vs {jd_min:.1f} required ({gap:.1f} years short)"
        )
    else:
        # Severe gap — hard floor at 5-10
        score = max(10.0 - (gap_ratio - 0.6) * 12.5, 5.0)
        evidence.append(
            f"Severe experience gap: {candidate_years:.1f} yrs vs {jd_min:.1f} required ({gap:.1f} years short)"
        )

    return score, evidence


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def score_resume(
    jd: JDProfile,
    resume: ResumeTaggingResponse,
    resume_tier: TierClassification,
    rank: int = 0,
) -> ResumeRankResult:
    """Score a parsed resume against a parsed JD.

    Args:
        jd: Parsed job description profile
        resume: Full resume tagging response from the hybrid pipeline
        resume_tier: SetFit classification of the resume
        rank: Position in final ranking (set by caller after sorting)

    Returns:
        ResumeRankResult with full dimension breakdown
    """
    # Detect experience level & estimate actual years
    exp_level = detect_experience_level(resume)
    candidate_years = estimate_candidate_years(resume)
    weights = _WEIGHTS[exp_level]

    # ── Dimension 1: Tier Alignment ──────────────────────────────────────────
    tier_raw, tier_ev = score_tier_alignment(
        jd.tier_classification, resume_tier
    )
    # ── Dimension 2: Core Capability Match ───────────────────────────────────
    cap_raw, cap_ev = score_capability_match(jd, resume, exp_level)

    # ── Dimension 3: Skill Match ─────────────────────────────────────────────
    skill_raw, skill_ev = score_skill_match(jd, resume, exp_level)

    # ── Dimension 4: Experience Quality ──────────────────────────────────────
    exp_raw, exp_ev = score_experience_quality(jd, resume, exp_level)

    # ── Dimension 5: Scale & Impact ──────────────────────────────────────────
    scale_raw, scale_ev = score_scale_and_impact(jd, resume)

    # ── Dimension 6: Domain & Context ────────────────────────────────────────
    domain_raw, domain_ev = score_domain_context(jd, resume)

    # ── Dimension 7 (conditional): Experience Adequacy ───────────────────────
    exp_adequacy_raw, exp_adequacy_ev = score_experience_adequacy(jd, candidate_years)
    has_exp_adequacy = exp_adequacy_raw is not None

    # ── Assemble Dimension Scores ─────────────────────────────────────────────
    dimension_scores = [
        DimensionScore(
            dimension="Tier Alignment",
            raw_score=round(tier_raw, 1),
            weight=weights["tier_alignment"],
            weighted_score=round(tier_raw * weights["tier_alignment"], 1),
            evidence=tier_ev,
        ),
        DimensionScore(
            dimension="Core Capability Match",
            raw_score=round(cap_raw, 1),
            weight=weights["capability_match"],
            weighted_score=round(cap_raw * weights["capability_match"], 1),
            evidence=cap_ev,
        ),
        DimensionScore(
            dimension="Skill Match",
            raw_score=round(skill_raw, 1),
            weight=weights["skill_match"],
            weighted_score=round(skill_raw * weights["skill_match"], 1),
            evidence=skill_ev,
        ),
        DimensionScore(
            dimension="Experience Quality",
            raw_score=round(exp_raw, 1),
            weight=weights["experience_quality"],
            weighted_score=round(exp_raw * weights["experience_quality"], 1),
            evidence=exp_ev,
        ),
        DimensionScore(
            dimension="Scale & Impact",
            raw_score=round(scale_raw, 1),
            weight=weights["scale_impact"],
            weighted_score=round(scale_raw * weights["scale_impact"], 1),
            evidence=scale_ev,
        ),
        DimensionScore(
            dimension="Domain & Context",
            raw_score=round(domain_raw, 1),
            weight=weights["domain_context"],
            weighted_score=round(domain_raw * weights["domain_context"], 1),
            evidence=domain_ev,
        ),
    ]

    if has_exp_adequacy:
        # When Experience Adequacy is active, reduce other weights proportionally
        # and give Experience Adequacy a 20% weight
        adeq_weight = 0.20
        scale_factor = 1.0 - adeq_weight  # 0.8
        dimension_scores.append(
            DimensionScore(
                dimension="Experience Adequacy",
                raw_score=round(exp_adequacy_raw, 1),
                weight=adeq_weight,
                weighted_score=round(exp_adequacy_raw * adeq_weight, 1),
                evidence=exp_adequacy_ev or [],
            )
        )
        # Scale down other weights proportionally
        for ds in dimension_scores[:-1]:
            ds.weight *= scale_factor
            ds.weighted_score = round(ds.raw_score * ds.weight, 1)

    final_score = sum(d.weighted_score for d in dimension_scores)

    # Tier mismatch hard penalty (only if tier1 completely wrong)
    tier_mismatch = (
        jd.tier_classification.tier1.label != "Unknown"
        and resume_tier.tier1.label != "Unknown"
        and jd.tier_classification.tier1.label != resume_tier.tier1.label
    )
    if tier_mismatch:
        final_score *= 0.7  # 30% penalty for wrong industry entirely

    cand = resume.candidate
    return ResumeRankResult(
        rank=rank,
        candidate_name=cand.name,
        candidate_email=cand.email,
        final_score=round(min(final_score, 100.0), 1),
        experience_level=exp_level,
        tier_classification=resume_tier,
        dimension_scores=dimension_scores,
        resume_strength_signals=resume.context_meta_tags.resume_strength_signals,
        tier_mismatch=tier_mismatch,
        low_confidence_classification=(
            resume_tier.low_confidence or jd.tier_classification.low_confidence
        ),
    )
