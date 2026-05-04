"""Shared tool schemas + dispatchers used by both the Categorizer and the
Skill Generator agents.

Defines:
  - SCHEMA_WEB_SEARCH / dispatch_web_search
  - SCHEMA_SCRAPE_PAGE / dispatch_scrape_page

Each agent composes its own ``TOOL_SCHEMAS`` list by importing these schemas
plus its own terminator schema, and routes through its own ``dispatch_tool``
which delegates the shared tools here.
"""

from __future__ import annotations

import json

from skill_library_v2.tools.content_extractor import extract_signal
from skill_library_v2.tools.firecrawl_scrape import scrape as _scrape
from skill_library_v2.tools.web_search import search as _search


SCHEMA_WEB_SEARCH: dict = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the live web for current lists of skills, frameworks, "
            "tools, or products relevant to a specific role/dimension. Use "
            "when your training knowledge may be stale. Returns up to "
            "`count` results; each has a title, URL, and short description."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {"type": "string"},
                "count": {
                    "type": "integer", "minimum": 1, "maximum": 10, "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

SCHEMA_SCRAPE_PAGE: dict = {
    "type": "function",
    "function": {
        "name": "scrape_page",
        "description": (
            "Fetch and filter one page's content. Returns headings, bullet "
            "lists, and paragraphs containing skill-relevant trigger words, "
            "truncated to ~2500 chars."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
        },
    },
}


async def dispatch_web_search(args: dict) -> str:
    query = str(args.get("query") or "").strip()
    count = int(args.get("count") or 5)
    if not query:
        return json.dumps({"error": "empty query"})
    results = await _search(query, count=max(1, min(10, count)))
    payload = [
        {"title": r.title, "url": r.url, "description": r.description}
        for r in results
    ]
    return json.dumps({"results": payload}, ensure_ascii=False)


async def dispatch_scrape_page(args: dict) -> str:
    url = str(args.get("url") or "").strip()
    if not url:
        return json.dumps({"error": "empty url"})
    markdown = await _scrape(url)
    if not markdown:
        return json.dumps({"error": "scrape returned no content", "url": url})
    filtered = extract_signal(markdown, max_chars=2500)
    return json.dumps(
        {"url": url, "content": filtered or "", "chars": len(filtered or "")},
        ensure_ascii=False,
    )


__all__ = [
    "SCHEMA_WEB_SEARCH",
    "SCHEMA_SCRAPE_PAGE",
    "dispatch_web_search",
    "dispatch_scrape_page",
]
