import os
import json
import asyncio
import logging
from dataclasses import dataclass

from rapidfuzz import fuzz, process
from openai import AsyncAzureOpenAI

import skill_library
from skills_dictionary import SKILL_ALIASES  # kept for static fuzzy key list

logger = logging.getLogger(__name__)

# ── Thresholds ──
FUZZY_THRESHOLD = 80  # minimum score to accept a fuzzy match


@dataclass
class NormalizedSkill:
    original: str
    normalized: str
    method: str  # "dictionary" | "fuzzy" | "llm" | "unmatched"
    confidence: float  # 0.0 – 1.0


# ────────────────────────────────────────────
# Tier 1 – Dictionary exact match
# ────────────────────────────────────────────
def _dict_lookup(skill: str) -> NormalizedSkill | None:
    key = skill.strip().lower()
    aliases = skill_library.get_aliases()
    if key in aliases:
        return NormalizedSkill(
            original=skill,
            normalized=aliases[key],
            method="dictionary",
            confidence=1.0,
        )
    return None


# ────────────────────────────────────────────
# Tier 2 – Fuzzy match against alias keys
# ────────────────────────────────────────────

def _get_alias_keys() -> list[str]:
    """Return current alias keys from merged skill library."""
    return list(skill_library.get_aliases().keys())


def _fuzzy_match(skill: str) -> NormalizedSkill | None:
    key = skill.strip().lower()
    aliases = skill_library.get_aliases()
    alias_keys = _get_alias_keys()
    candidates = process.extract(
        key,
        alias_keys,
        scorer=fuzz.WRatio,
        score_cutoff=FUZZY_THRESHOLD,
        limit=8,
    )
    for matched_alias, score, _ in candidates:
        # Guard against short-string false positives:
        # reject if the lengths differ too much.
        len_ratio = min(len(key), len(matched_alias)) / \
            max(len(key), len(matched_alias))
        if len_ratio < 0.5:
            continue
        return NormalizedSkill(
            original=skill,
            normalized=aliases[matched_alias],
            method="fuzzy",
            confidence=round(score / 100, 2),
        )
    return None


# ────────────────────────────────────────────
# Tier 3 – LLM fallback (Azure OpenAI)
# ────────────────────────────────────────────
_AZURE_ENDPOINT = "https://tabuddy-azure-sponsor.openai.azure.com/"
_AZURE_DEPLOYMENT = "gpt-4o-mini"
_AZURE_API_VERSION = "2024-12-01-preview"

_client: AsyncAzureOpenAI | None = None


def _get_azure_client() -> AsyncAzureOpenAI | None:
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


_LLM_SYSTEM_PROMPT = """\
You are a skill normalizer. Given a list of tech/professional skill strings, \
return their canonical (standard) forms along with a category.

Rules:
- Use the most widely recognized name (e.g. "React" not "ReactJS").
- Preserve proper casing (e.g. "Node.js", "PostgreSQL", "C#").
- If the input is already canonical, return it as-is.
- If you don't recognize a skill, return the original string unchanged.
- Assign each skill ONE category from this list:
  Web Development, Mobile Development, Backend Development, Frontend Development,
  AI/ML, Data Science, DevOps/Cloud, Databases, Programming Languages,
  Testing/QA, Design/UI/UX, Marketing, Project Management, Cybersecurity,
  Networking, Blockchain, Game Development, Embedded/IoT, Soft Skills, Other
- Return ONLY a JSON array of objects with "name" and "category" keys, no explanation.

Example:
Input: ["Reactjs", "PSQL", "k8s", "some-unknown-tool"]
Output: [{"name": "React", "category": "Frontend Development"}, {"name": "PostgreSQL", "category": "Databases"}, {"name": "Kubernetes", "category": "DevOps/Cloud"}, {"name": "some-unknown-tool", "category": "Other"}]
"""


async def _llm_normalize(skills: list[str]) -> list[NormalizedSkill]:
    """Batch-normalize unmatched skills via Azure OpenAI."""
    client = _get_azure_client()
    if client is None:
        # No API key configured – return as unmatched
        return [
            NormalizedSkill(
                original=s, normalized=s, method="unmatched", confidence=0.0
            )
            for s in skills
        ]

    try:
        response = await client.chat.completions.create(
            model=_AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(skills)},
            ],
            max_completion_tokens=16000,
        )
        content = response.choices[0].message.content or "[]"
        normalized: list[dict] = json.loads(content)

        # Safety: ensure lengths match
        if len(normalized) != len(skills):
            raise ValueError("LLM returned mismatched array length")

        results = []
        for orig, entry in zip(skills, normalized):
            if isinstance(entry, dict):
                norm_name = entry.get("name", orig)
                category = entry.get("category", "Other")
            else:
                # Backward compat: plain string
                norm_name = str(entry)
                category = "Other"

            norm_name = skill_library.normalize_canonical_case(norm_name)

            results.append(
                NormalizedSkill(
                    original=orig,
                    normalized=norm_name,
                    method="llm",
                    confidence=0.85,
                )
            )
            # Persist every LLM-tier skill to MongoDB (fire-and-forget),
            # including same-name canonical forms like "Snowflake".
            asyncio.create_task(
                skill_library.learn(
                    orig, norm_name,
                    confidence=0.85,
                    category=category,
                )
            )

        return results
    except Exception as e:
        logger.warning("LLM normalization failed: %s", e)
        return [
            NormalizedSkill(
                original=s, normalized=s, method="unmatched", confidence=0.0
            )
            for s in skills
        ]


# ────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────
async def normalize_skills(skills: list[str]) -> list[NormalizedSkill]:
    """
    Normalize a list of skill strings.

    Pipeline:  Dictionary → Fuzzy → LLM fallback
    """
    results: list[NormalizedSkill | None] = [None] * len(skills)
    llm_pending: list[tuple[int, str]] = []  # (index, original_skill)

    for i, skill in enumerate(skills):
        # Tier 1: Dictionary
        result = _dict_lookup(skill)
        if result:
            results[i] = result
            continue

        # Tier 2: Fuzzy
        result = _fuzzy_match(skill)
        if result:
            results[i] = result
            continue

        # Collect for Tier 3
        llm_pending.append((i, skill))

    # Tier 3: Batch LLM call for remaining
    if llm_pending:
        llm_skills = [s for _, s in llm_pending]
        llm_results = await _llm_normalize(llm_skills)
        for (idx, _), llm_result in zip(llm_pending, llm_results):
            results[idx] = llm_result

    return [r for r in results if r is not None]
