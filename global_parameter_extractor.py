"""Global Parameter Extractor – sends all candidate_blocks to LLM for the 7 global parameters.

Uses the full contextual tagging prompt with strict taxonomy enums.
"""

from __future__ import annotations

import os
import json
import logging
from openai import AsyncAzureOpenAI
from models import (
    ResumeBlock,
    GlobalParameters,
    ScaleAnchor,
    EnumExtraction,
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


_GLOBAL_PARAMS_SYSTEM_PROMPT = """\
You are an elite Data Normalization AI for a technical recruiting platform. Your sole objective is to extract 7 Global Parameters from a candidate's resume blocks, normalize them into strict standardized taxonomies, and output a validated JSON schema.

You will receive a JSON array of objects called `candidate_blocks`, each containing a `block_name` and `raw_text`.

CRITICAL GUARDRAILS & RULES:
1. EXHAUSTIVE MULTI-BLOCK SCAN: You MUST evaluate EVERY block provided in the input. A single parameter (e.g., scale_anchors) may appear in multiple blocks. If so, you must extract an object for EACH occurrence and append it to the parameter's array. Do not stop after finding the first match.
2. NO HALLUCINATION: If a parameter is not explicitly mentioned across any blocks, return an empty array `[]`. Do not guess.
3. PROVENANCE TRACING: For every extraction, you must strictly map the `block_name` it came from and provide a 3-6 word `raw_evidence` quote.
4. FINANCIAL METRIC BAN: Under NO circumstances should you extract financial metrics, revenue, cost savings, budget sizes, or ROI into the `scale_anchors`. Ignore all monetary figures.
5. NO MATH: Do not convert units. Extract the raw metric and raw unit separately.
6. THE ESCAPE HATCH: For enum-based parameters, if a valid engineering concept is present but DOES NOT fit the provided enums, output "Other: [Raw Text]" to prevent data loss.

NORMALIZATION DICTIONARY (STRICT TAXONOMY):

1. scale_anchors:
   - Extract non-financial system scale, data volume, or throughput. Separate number and unit.
   - Valid Units: RPS, RPM, TPS, MAU, DAU, GB, TB, PB, requests/day, etc.

2. codebase_lifecycle (Enum):
   - "0-to-1 / Greenfield"
   - "Migration / Modernization"
   - "Scaling / Optimization"
   - "Maintenance / Operations"
   - "Integration"

3. product_domain (Enum):
   - "Fintech / Payments", "E-commerce / Retail", "Healthcare / Healthtech", "Enterprise SaaS", "Data / Analytics Platform", "Consumer Social / Media", "Infrastructure / DevOps", "Cybersecurity / Identity", "AdTech / MarTech".

4. architectural_paradigm (Enum):
   - "Microservices", "Event-Driven / Async", "Serverless", "Monolithic", "Pub/Sub", "Batch Processing", "Streaming / Real-time", "Hexagonal / Clean Architecture", "Service-Oriented (SOA)".

5. compliance_exposure (Enum):
   - "PCI-DSS", "HIPAA", "SOC2", "GDPR", "FEDRAMP", "CCPA", "ISO-27001", "SOX", "ADA/Accessibility".

6. leadership_footprint:
   - Format: "[Verb] [Number] [Entity]". Default to "Led team" if no number exists.

7. cross_functional_area (Enum):
   - "Product Management" (Roadmaps, feature specs, Agile alignment)
   - "Design / UX" (Figma, accessibility, user research)
   - "Data Science / ML" (Model integration, data pipelines)
   - "Enterprise Clients / B2B" (Customer success, SLA negotiations)
   - "Risk / Legal / Compliance" (Audits, GDPR scrubbing)
   - "Sales / Marketing" (Martech, SEO, CRM integrations)
   - "Support / Operations" (Tier 3 escalations, incident response)

JSON OUTPUT SCHEMA:
Return ONLY the following JSON structure. Do not use ```json markdown.

{
  "reasoning_log": [
    "Step 1: Iterate through EVERY block in candidate_blocks sequentially.",
    "Step 2: For each block, extract all valid parameters, strictly filtering out financial metrics.",
    "Step 3: Map text to Enums (using 'Other: [Text]' if needed) and append to the respective arrays with accurate provenance."
  ],
  "global_parameters": {
    "scale_anchors": [
      {"extracted_metric": "String", "extracted_unit": "String", "provenance": "String", "raw_evidence": "String"}
    ],
    "codebase_lifecycle": [{"normalized_value": "String", "provenance": "String", "raw_evidence": "String"}],
    "product_domain": [{"normalized_value": "String", "provenance": "String", "raw_evidence": "String"}],
    "architectural_paradigm": [{"normalized_value": "String", "provenance": "String", "raw_evidence": "String"}],
    "compliance_exposure": [{"normalized_value": "String", "provenance": "String", "raw_evidence": "String"}],
    "leadership_footprint": [{"normalized_value": "String", "provenance": "String", "raw_evidence": "String"}],
    "cross_functional_area": [{"normalized_value": "String", "provenance": "String", "raw_evidence": "String"}]
  }
}
"""


async def extract_global_parameters(
    blocks: list[ResumeBlock],
) -> tuple[GlobalParameters, list[str]]:
    """Send all blocks to LLM and extract the 7 global parameters.

    Returns (GlobalParameters, reasoning_log).
    """
    client = _get_client()
    if client is None:
        logger.warning(
            "No Azure OpenAI key – returning empty global parameters")
        return GlobalParameters(), ["LLM unavailable – skipped global parameter extraction"]

    candidate_blocks = [
        {"block_name": b.block_name, "raw_text": b.raw_text} for b in blocks
    ]

    try:
        response = await client.chat.completions.create(
            model=_AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _GLOBAL_PARAMS_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(candidate_blocks)},
            ],
            max_completion_tokens=10000,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        logger.info(
            "Global params LLM response: finish_reason=%s, content_length=%d, "
            "reasoning_tokens=%s",
            choice.finish_reason,
            len(content),
            getattr(response.usage.completion_tokens_details,
                    "reasoning_tokens", "N/A")
            if response.usage and response.usage.completion_tokens_details else "N/A",
        )

        if not content.strip():
            logger.warning("Global params LLM returned empty content")
            return GlobalParameters(), ["LLM returned empty response"]

        # Strip markdown fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0]

        parsed = json.loads(content)
        gp = parsed.get("global_parameters", {})
        reasoning = parsed.get("reasoning_log", [])

        return (
            GlobalParameters(
                scale_anchors=[ScaleAnchor(**a)
                               for a in gp.get("scale_anchors", [])],
                codebase_lifecycle=[
                    EnumExtraction(**e) for e in gp.get("codebase_lifecycle", [])
                ],
                product_domain=[
                    EnumExtraction(**e) for e in gp.get("product_domain", [])
                ],
                architectural_paradigm=[
                    EnumExtraction(**e) for e in gp.get("architectural_paradigm", [])
                ],
                compliance_exposure=[
                    EnumExtraction(**e) for e in gp.get("compliance_exposure", [])
                ],
                leadership_footprint=[
                    EnumExtraction(**e) for e in gp.get("leadership_footprint", [])
                ],
                cross_functional_area=[
                    EnumExtraction(**e) for e in gp.get("cross_functional_area", [])
                ],
            ),
            reasoning,
        )
    except Exception as e:
        logger.exception("Global parameter extraction failed: %s", e)
        return GlobalParameters(), [f"Extraction error: {e}"]
