"""Stage 4 type-assigner prompt (system + user).

Versioned so every v2_run_log row records which prompt produced its
output. Bump TYPE_ASSIGNER_PROMPT_VERSION on any text change.
"""

from __future__ import annotations

import json

from skill_library_v3.schemas.typology import TYPOLOGY_DESCRIPTIONS, TYPOLOGY_VALUES


TYPE_ASSIGNER_PROMPT_VERSION = "stage4_typing_v1.1"


TYPE_ASSIGNER_SYSTEM_PROMPT = """\
You assign each skill in the input batch to exactly ONE type from the
closed typology below. Type is what the skill fundamentally IS, not
what it's used for. A library is not a "tool" because it's used like
one — it's a library.

Disambiguation rules (apply these explicitly in your reasoning):

  * Tool vs Framework: Do users BUILD applications WITH it (Framework)
    or USE it AS software (Tool)?  Spring Boot = Framework. Jenkins = Tool.

  * Platform vs Tool: Is it a hosted environment with multi-tenancy
    and APIs (Platform) or software-you-run (Tool)?  AWS = Platform.
    Self-hosted Airflow = Tool.

  * Service vs Platform: A Service lives INSIDE a Platform — it's a
    specific managed capability. AWS Lambda = Service. AWS itself = Platform.

  * Vendor SaaS = Platform (NOT Tool). When the skill name is a
    commercial SaaS vendor (Wiz, Datadog, Snowflake, Okta, Auth0,
    Splunk Cloud, Lacework, Qualys, Rapid7, CyberArk, BeyondTrust,
    CrowdStrike, SentinelOne, Microsoft Sentinel, Microsoft Defender
    for Cloud), type it Platform — these are hosted multi-tenant
    environments with APIs, fitting the Platform definition exactly.
    Type Tool only for software a user runs themselves (Wireshark,
    Nmap, Metasploit, Burp Suite Community Edition, self-hosted
    Jenkins). The disambiguator is "do you run it yourself or do you
    consume it via the vendor's hosted instance?".

  * Datastore vs Format: Does it persist data (Datastore) or specify
    how data is structured (Format)?  PostgreSQL = Datastore.
    JSON = Format. Parquet = Format. Redis-as-cache = Tool;
    Redis-as-primary-store = Datastore (use the dominant pattern).

  * Concept vs Methodology: Is it a knowledge unit (Concept — what
    you must know) or a way of working (Methodology — how you operate)?
    CAP theorem = Concept. Agile = Methodology. Idempotency = Concept.
    TDD = Methodology.

  * Architecture vs Concept: Is it a system shape that influences how
    you build (Architecture — Microservices, Hexagonal) or a knowledge
    unit (Concept — eventual consistency)?

Closed typology — pick exactly ONE per skill:

{typology_block}

Output rules:

  * Echo each input skill's ``skill_id`` exactly so the response can be
    matched back to the input. Do not invent skill_ids; do not skip
    inputs; do not output extras.
  * ``subtype`` is snake_case (lowercase a-z, digits, underscores) and
    more specific than ``type``. Examples: orchestration_tool,
    scripting_language, key_value_store, web_framework, observability_tool.
  * ``confidence`` in [0.0, 1.0]. Calibrate honestly — 0.95 means "I am
    very sure"; 0.6 means "this is the best fit but two others are close".
  * ``reasoning``: ONE sentence applying the disambiguation rules above
    to this specific skill. Reference the rule by name when relevant.
  * ``alternatives_considered``: include only when ``confidence`` is
    below 0.85. Each entry: "<TypeName>: ruled out — <reason>".

Emit one JSON object: ``{{"skills": [...]}}``. No prose, no code fences.
"""


TYPE_ASSIGNER_USER_TEMPLATE = """\
Skills to type:
{skills_block}

Type each one and return them in the same order. Use the disambiguation
rules above. Pick exactly ONE type per skill from the closed list.
"""


def format_typology_block() -> str:
    """Render the 17-type typology with one-line descriptions for the
    LLM's disambiguation anchors."""
    lines = []
    for name in TYPOLOGY_VALUES:
        lines.append(f"  - {name}: {TYPOLOGY_DESCRIPTIONS[name]}")
    return "\n".join(lines)


def format_skills_batch_block(skills: list[dict]) -> str:
    """Render the input batch as a compact JSON list. Each item carries
    skill_id + name + optional aliases + optional samples."""
    rows: list[dict] = []
    for s in skills:
        item: dict = {
            "skill_id": s.get("skill_id"),
            "name": s.get("name"),
        }
        aliases = s.get("aliases") or []
        if aliases:
            item["aliases"] = list(aliases)
        samples = s.get("samples") or []
        if samples:
            # Truncate samples aggressively — JD bodies can be huge.
            item["samples"] = [str(x)[:200] for x in samples][:5]
        rows.append(item)
    return json.dumps(rows, indent=2, ensure_ascii=False)
