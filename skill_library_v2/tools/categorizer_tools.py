"""Tool registry for the Categorizer agent.

Three tools the Categorizer may call:

- ``web_search(query, count=5)`` — live search via shared dispatcher
- ``scrape_page(url)`` — single-URL scrape via shared dispatcher
- ``submit_categories(dim_id, proposals, notes)`` — **terminator**; must be
  called exactly once.
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

TERMINATOR_TOOL = "submit_categories"

SCHEMA_SUBMIT_CATEGORIES: dict = {
    "type": "function",
    "function": {
        "name": TERMINATOR_TOOL,
        "description": (
            "FINALIZER. Call exactly once when your category list for the "
            "dimension is complete. The arguments must match the "
            "CategorizerOutput schema. After this call, the tool loop exits."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "dim_id": {"type": "string"},
                "proposals": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "category_slug", "category_display",
                            "sub_category_slug", "sub_category_display",
                            "rationale",
                        ],
                        "properties": {
                            "category_slug": {
                                "type": "string",
                                "description": "lowercase snake_case",
                            },
                            "category_display": {"type": "string"},
                            "sub_category_slug": {
                                "type": "string",
                                "description": "lowercase snake_case",
                            },
                            "sub_category_display": {"type": "string"},
                            "rationale": {"type": "string"},
                            "expected_skill_count_hint": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 80,
                                "default": 10,
                            },
                        },
                    },
                },
                "notes": {"type": "string", "default": ""},
            },
            "required": ["dim_id", "proposals"],
        },
    },
}


# Web search/scrape deliberately excluded — they were burning Firecrawl
# credits at ~50-80 calls/role with no proportional quality gain. The Planner
# already grounds the role in current web context (gather_web_hints) before
# the Categorizer runs; per-dim search by the Categorizer is over-eager.
# Keep the dispatch fns importable in case we re-enable later behind a flag.
TOOL_SCHEMAS: list[dict] = [
    SCHEMA_SUBMIT_CATEGORIES,
]


async def dispatch_tool(name: str, args: dict) -> str:
    """Run one non-terminator categorizer tool and return its tool-message string."""
    if name == "web_search":
        return await dispatch_web_search(args)
    if name == "scrape_page":
        return await dispatch_scrape_page(args)
    return json.dumps({"error": f"unknown tool {name!r}"})


__all__ = ["TOOL_SCHEMAS", "TERMINATOR_TOOL", "dispatch_tool"]
