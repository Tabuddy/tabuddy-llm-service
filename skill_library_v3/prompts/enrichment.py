"""Stage 7 enrichment prompts (5 agents, 7a-7e).

Each agent gets its own focused prompt + version constant. The five
versions are emitted into a single ``stage7_enrichment_v1.0`` row by
the runner; per-agent versions exist so we can iterate on one aspect
without bumping the other four.
"""

from __future__ import annotations

import json
from datetime import datetime

from skill_library_v3.schemas.enrichment import LICENSE_VALUES, MATURITY_VALUES


# ── prompt-version constants ──────────────────────────────────────────────


VENDOR_LICENSE_PROMPT_VERSION = "stage7_vendor_license_v1.0"
MATURITY_PROMPT_VERSION = "stage7_maturity_v1.0"
CONTEXT_KEYWORDS_PROMPT_VERSION = "stage7_context_keywords_v1.0"
AMBIGUITY_PROMPT_VERSION = "stage7_ambiguity_v1.0"
VERSIONING_PROMPT_VERSION = "stage7_versioning_v1.0"

# Row-level prompt version for the v2_run_log row (matches
# STAGE7_PROMPT_VERSION_PREFIX). Per-agent versions above are stored
# inside planner_output for traceability of each focused output.
ENRICHMENT_PROMPT_VERSION = "stage7_enrichment_v1.0"


# ── 7a: Vendor + License ──────────────────────────────────────────────────


def _license_enum_block() -> str:
    return ", ".join(LICENSE_VALUES)


VENDOR_LICENSE_SYSTEM_PROMPT = f"""\
You attach vendor + license metadata to one engineering skill.

Output exactly one JSON object with these keys:

  {{
    "vendor": "<canonical vendor / maintainer name, or null>",
    "license": "<one of: {_license_enum_block()}, or null>",
    "year_introduced": <int 1960..{datetime.now().year + 1}, or null>,
    "confidence": <float in [0.0, 1.0]>
  }}

Rules:
  * vendor=null is correct for Concept / Methodology / SoftSkill skills
    (e.g., "CAP theorem", "Idempotency", "Code review"). Don't invent
    a vendor.
  * license=null is correct when the skill itself is a concept (no
    licensable artifact) OR when the license is genuinely unknown to
    you. Use "unknown" only when you know there IS a license but can't
    pin which.
  * year_introduced should reflect when the skill / tool / framework
    first became public. For concepts, null.
  * confidence: 0.95+ for canonical knowledge ("Python is maintained by
    PSF, OSI-Python License"), 0.7-0.9 for second-hand knowledge, <0.5
    if you're guessing.

No prose, no fences. JSON only.
"""


VENDOR_LICENSE_USER_TEMPLATE = """\
Skill:
{skill_block}

Emit one JSON VendorLicense object.
"""


# ── 7b: Maturity ──────────────────────────────────────────────────────────


def _maturity_enum_block() -> str:
    return " | ".join(MATURITY_VALUES)


MATURITY_SYSTEM_PROMPT = f"""\
You assign a maturity level to one engineering skill.

Output exactly one JSON object:

  {{
    "maturity": "<one of: {_maturity_enum_block()}>",
    "reasoning": "<20-400 char explanation citing market signal>",
    "confidence": <float in [0.0, 1.0]>
  }}

Definitions (use the closest match):
  * well_known — broadly adopted, hiring-pipeline staple. Examples:
    Python, React, AWS, PostgreSQL, Kubernetes.
  * emerging  — gaining adoption, increasingly listed in JDs but not yet
    universal. Examples: Bun, Deno, htmx, LangChain (as of recent).
  * niche     — specialised tooling, narrow community, low-volume in JDs.
    Examples: Erlang/OTP, F#, Couchbase, Hadoop MapReduce.
  * deprecated — known-superseded by a successor and discouraged in
    new projects. Examples: AngularJS (1.x), jQuery (in greenfield),
    Internet Explorer 11, Python 2.

The reasoning field MUST cite a concrete market signal (JD volume,
GitHub trend, vendor-announced sunset, replaced by X) — not vague
sentiment.

No prose, no fences. JSON only.
"""


MATURITY_USER_TEMPLATE = """\
Skill:
{skill_block}

Emit one JSON Maturity object.
"""


# ── 7c: Context Keywords ──────────────────────────────────────────────────


CONTEXT_KEYWORDS_SYSTEM_PROMPT = """\
You produce a short list of distinctive co-occurring terms for one
engineering skill. These keywords help downstream extraction tag
adjacent JD text and surface related skills the catalog might miss.

Output exactly one JSON object:

  {
    "context_keywords": ["term", "term", ...]
  }

Rules:
  * Length: 8-15 keywords. Stop at 15.
  * "Distinctive" means: a term that appears NEAR this skill in JDs
    or technical writing more often than baseline. NOT generic
    workplace words ("teamwork", "communication") and NOT trivially
    derived ("the skill name itself", "Python developer" for Python).
  * Mix surface forms: tools, frameworks, paradigms, related concepts,
    common phrases. For "Python": ["Django", "Flask", "FastAPI",
    "asyncio", "pip", "venv", "REST", "API", "PEP 8", "type hints"].
  * No empty strings, no whitespace-only entries, no duplicates.

No prose, no fences. JSON only.
"""


CONTEXT_KEYWORDS_USER_TEMPLATE = """\
Skill:
{skill_block}

Emit one JSON ContextKeywords object.
"""


# ── 7d: Ambiguity ─────────────────────────────────────────────────────────


AMBIGUITY_SYSTEM_PROMPT = """\
You decide whether one engineering skill name is ambiguous in our
catalog — i.e., whether a JD mention could plausibly refer to a
DIFFERENT skill that's also in the catalog.

Output exactly one JSON object:

  {
    "ambiguity_flag": <bool>,
    "confused_with": [<list of skill_ids — non-empty iff ambiguity_flag=true>],
    "reasoning": "<20-400 char explanation>"
  }

Common ambiguity patterns:
  * Short / overloaded names: "Go" (language vs verb vs the board game),
    "R" (language vs other R-letter tooling), "C" (language vs grade).
  * Adjacent products with same vendor: "AWS Lambda" vs "Lambda
    Architecture".
  * Acronym collisions: "ML" (Machine Learning vs Markup Language).

Rules:
  * If ambiguity_flag=true, confused_with MUST list at least one specific
    skill_id this could be confused with. An empty list with the flag
    set is invalid.
  * If ambiguity_flag=false, confused_with should be empty.
  * Don't flag skills with rare-but-distinguishable contexts. The bar
    is "could a reasonable extractor confuse these in a typical JD?"

No prose, no fences. JSON only.
"""


AMBIGUITY_USER_TEMPLATE = """\
Skill:
{skill_block}

Emit one JSON Ambiguity object.
"""


# ── 7e: Versioning ────────────────────────────────────────────────────────


VERSIONING_SYSTEM_PROMPT = """\
You decide whether one engineering skill has meaningful versions and,
if so, name the current canonical version + common alias mappings.

Output exactly one JSON object:

  {
    "versioned": <bool>,
    "current_version": "<string, or null>",
    "version_aliases": {"<alias>": "<canonical version>"}
  }

Rules:
  * Versioned=true ONLY when the version meaningfully changes meaning
    (Python 2 vs 3, Java 8 vs 21, Angular 1.x vs 2+, Kubernetes pre/post
    1.0, .NET Framework vs .NET Core). For these, current_version is
    REQUIRED.
  * Versioned=false for concept skills, methodologies, soft skills, or
    tools where the version is essentially invisible to the consumer
    (e.g., REST, OAuth 2.0 — those are protocol versions baked into the
    skill name itself).
  * version_aliases maps surface forms to canonical version: e.g.,
    {"py3": "3", "Python 3.x": "3", "Python 3.10": "3.10"}. Empty dict
    is fine when no canonical mapping is needed.

No prose, no fences. JSON only.
"""


VERSIONING_USER_TEMPLATE = """\
Skill:
{skill_block}

Emit one JSON Versioning object.
"""


# ── shared formatter ──────────────────────────────────────────────────────


def format_skill_block(skill: dict) -> str:
    """Render the target skill's identity for the prompt. Mirrors the
    Stage 4/5/6 pattern — JSON of the skill's stable identity fields."""
    keep = {}
    for k in ("skill_id", "name", "type", "subtype", "primary_dimension"):
        if k in skill:
            keep[k] = skill[k]
    return json.dumps(keep, indent=2, ensure_ascii=False)
