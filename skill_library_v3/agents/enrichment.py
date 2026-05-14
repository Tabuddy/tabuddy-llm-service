"""Stage 7 enrichment agents — five focused-output agents (7a-7e).

The runner fans out per skill via ``asyncio.gather`` over all five
agents in parallel. Each agent has its own prompt-version + Pydantic
output schema; the runner merges the five outputs into one
SkillEnrichment record.
"""

from __future__ import annotations

from skill_library_v2.agents.base import BaseLLMAgent
from skill_library_v3.prompts.enrichment import (
    AMBIGUITY_PROMPT_VERSION,
    AMBIGUITY_SYSTEM_PROMPT,
    AMBIGUITY_USER_TEMPLATE,
    CONTEXT_KEYWORDS_PROMPT_VERSION,
    CONTEXT_KEYWORDS_SYSTEM_PROMPT,
    CONTEXT_KEYWORDS_USER_TEMPLATE,
    MATURITY_PROMPT_VERSION,
    MATURITY_SYSTEM_PROMPT,
    MATURITY_USER_TEMPLATE,
    VENDOR_LICENSE_PROMPT_VERSION,
    VENDOR_LICENSE_SYSTEM_PROMPT,
    VENDOR_LICENSE_USER_TEMPLATE,
    VERSIONING_PROMPT_VERSION,
    VERSIONING_SYSTEM_PROMPT,
    VERSIONING_USER_TEMPLATE,
    format_skill_block,
)
from skill_library_v3.schemas.enrichment import (
    Ambiguity,
    ContextKeywords,
    Maturity,
    VendorLicense,
    Versioning,
)


# ── 7a: Vendor + License ──────────────────────────────────────────────────


class Stage7VendorLicenseAgent(BaseLLMAgent):
    """Agent 7a — vendor + license + year_introduced + confidence.

    Runs on `fast` (gpt-4o-mini) — vendor identity is stable factual knowledge
    and pattern-matching, doesn't need gpt-5.x reasoning depth.
    """

    tier = "fast"

    def __init__(self) -> None:
        super().__init__(
            agent_name="stage7_vendor_license",
            prompt_version=VENDOR_LICENSE_PROMPT_VERSION,
        )

    async def enrich(self, *, skill: dict) -> VendorLicense:
        user_prompt = VENDOR_LICENSE_USER_TEMPLATE.format(
            skill_block=format_skill_block(skill),
        )
        return await self.call_json(
            system_prompt=VENDOR_LICENSE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=VendorLicense,
            temperature=0.1,
        )


# ── 7b: Maturity ──────────────────────────────────────────────────────────


class Stage7MaturityAgent(BaseLLMAgent):
    """Agent 7b — maturity level + reasoning + confidence.

    Stays on `generation` tier (the rest of Stage 7 is on `nano`). Hybrid
    eval against the v1.4 baseline showed nano shifts ~46% of maturity bands
    by 2+ levels (e.g. Argo CD: well_known → emerging) — the maturity badge
    is the most-visible Stage 7 field in the admin UI, so we trade ~$0.15
    of nano savings to keep this one agent calibrated.
    """

    tier = "generation"

    def __init__(self) -> None:
        super().__init__(
            agent_name="stage7_maturity",
            prompt_version=MATURITY_PROMPT_VERSION,
        )

    async def enrich(self, *, skill: dict) -> Maturity:
        user_prompt = MATURITY_USER_TEMPLATE.format(
            skill_block=format_skill_block(skill),
        )
        return await self.call_json(
            system_prompt=MATURITY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=Maturity,
            temperature=0.1,
        )


# ── 7c: Context Keywords ──────────────────────────────────────────────────


class Stage7ContextKeywordsAgent(BaseLLMAgent):
    """Agent 7c — 8-15 distinctive co-occurring terms.

    Runs on `fast` (gpt-4o-mini) — pure recall/pattern task, no reasoning load.
    """

    tier = "fast"

    def __init__(self) -> None:
        super().__init__(
            agent_name="stage7_context_keywords",
            prompt_version=CONTEXT_KEYWORDS_PROMPT_VERSION,
        )

    async def enrich(self, *, skill: dict) -> ContextKeywords:
        user_prompt = CONTEXT_KEYWORDS_USER_TEMPLATE.format(
            skill_block=format_skill_block(skill),
        )
        return await self.call_json(
            system_prompt=CONTEXT_KEYWORDS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=ContextKeywords,
            temperature=0.2,
        )


# ── 7d: Ambiguity ─────────────────────────────────────────────────────────


class Stage7AmbiguityAgent(BaseLLMAgent):
    """Agent 7d — ambiguity flag + confused_with skill_ids.

    Runs on `nano` (gpt-5.4-nano). gpt-4o-mini blew up false positives
    17x on this task in the v1.4 eval (2 baseline flags -> 34 flags),
    hallucinating skill_ids and conflating umbrella/sub-service relations
    (e.g. flagging "aws" as confused with "aws_lambda") with ambiguity.
    Nano stays calibrated against the baseline.
    """

    tier = "nano"

    def __init__(self) -> None:
        super().__init__(
            agent_name="stage7_ambiguity",
            prompt_version=AMBIGUITY_PROMPT_VERSION,
        )

    async def enrich(self, *, skill: dict) -> Ambiguity:
        user_prompt = AMBIGUITY_USER_TEMPLATE.format(
            skill_block=format_skill_block(skill),
        )
        return await self.call_json(
            system_prompt=AMBIGUITY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=Ambiguity,
            temperature=0.1,
        )


# ── 7e: Versioning ────────────────────────────────────────────────────────


class Stage7VersioningAgent(BaseLLMAgent):
    """Agent 7e — versioned flag + current_version + alias map."""

    tier = "nano"

    def __init__(self) -> None:
        super().__init__(
            agent_name="stage7_versioning",
            prompt_version=VERSIONING_PROMPT_VERSION,
        )

    async def enrich(self, *, skill: dict) -> Versioning:
        user_prompt = VERSIONING_USER_TEMPLATE.format(
            skill_block=format_skill_block(skill),
        )
        return await self.call_json(
            system_prompt=VERSIONING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=Versioning,
            temperature=0.1,
        )
