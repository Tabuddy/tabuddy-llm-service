"""Tool registry for the Skill Generator's function-calling loop.

Three tools the Skill Generator may call:

- ``web_search(query, count=5)`` — live search via shared dispatcher
- ``scrape_page(url)`` — single-URL scrape via shared dispatcher
- ``submit_skills(dim_id, skills, notes)`` — **terminator**; called once.
"""

from __future__ import annotations

import json
import logging

from skill_library_v2.tools.shared_tools import (
    SCHEMA_SCRAPE_PAGE,
    SCHEMA_WEB_SEARCH,
    dispatch_scrape_page,
    dispatch_web_search,
)

logger = logging.getLogger(__name__)

TERMINATOR_TOOL = "submit_skills"

SCHEMA_SUBMIT_SKILLS: dict = {
    "type": "function",
    "function": {
        "name": TERMINATOR_TOOL,
        "description": (
            "FINALIZER. Call this exactly once when your skill list for the "
            "dimension is complete. The arguments must match the "
            "SkillGeneratorOutput schema. Each skill MUST reference a "
            "(category_slug, sub_category_slug) pair from the Categorizer's "
            "proposals — anything else fails validation. After this call, "
            "the tool loop exits."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "dim_id": {"type": "string"},
                "skills": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 80,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "display_name", "slug",
                            "category_slug", "sub_category_slug",
                            "skill_nature",
                        ],
                        "properties": {
                            "display_name": {"type": "string"},
                            "slug": {
                                "type": "string",
                                "description": (
                                    "lowercase snake_case; no `_and_`; "
                                    "max 5 tokens; noun-first."
                                ),
                            },
                            "category_slug": {
                                "type": "string",
                                "description": (
                                    "Must match a category_slug from the "
                                    "Categorizer's proposals."
                                ),
                            },
                            "sub_category_slug": {
                                "type": "string",
                                "description": (
                                    "Must match a sub_category_slug from the "
                                    "Categorizer's proposals."
                                ),
                            },
                            "skill_nature": {
                                "type": "string",
                                "enum": ["TOOL", "CONCEPT", "PRACTICE",
                                         "PLATFORM", "LANGUAGE", "CREDENTIAL"],
                            },
                            "aliases": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                            },
                        },
                    },
                },
                "notes": {
                    "type": "string",
                    "description": "Optional free-form rationale.",
                    "default": "",
                },
            },
            "required": ["dim_id", "skills"],
        },
    },
}


# Web search/scrape deliberately excluded — same rationale as in
# categorizer_tools.py. The Skill Generator relies on the LLM's internal
# knowledge plus the Planner-injected web hints in role state. Smoke runs
# (gpt-5-mini and gpt-5.4-mini) hit 134% of baseline volume without per-dim
# search, so the cost was not buying us coverage.
TOOL_SCHEMAS: list[dict] = [
    SCHEMA_SUBMIT_SKILLS,
]


async def dispatch_tool(name: str, args: dict) -> str:
    """Run one non-terminator skill-generator tool."""
    if name == "web_search":
        return await dispatch_web_search(args)
    if name == "scrape_page":
        return await dispatch_scrape_page(args)
    return json.dumps({"error": f"unknown tool {name!r}"})


__all__ = ["TOOL_SCHEMAS", "TERMINATOR_TOOL", "dispatch_tool"]
