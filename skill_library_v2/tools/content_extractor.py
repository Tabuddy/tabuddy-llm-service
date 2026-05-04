"""Extractive signal filter for scraped page content.

Pure function — no I/O, no LLM, no external state. Takes a markdown document
(as returned by Firecrawl scrape) and keeps only the lines that carry skill
signal: section headings, bullet / numbered list items, and paragraphs
containing role-relevant trigger words. Truncates to ``max_chars`` at a line
boundary.

Called by ``tools.web_search.fetch_extracts`` for each scraped page; the
filtered text is fed into the Planner prompt below the snippet citation
index.
"""

from __future__ import annotations

import re

# Paragraphs that contain these substrings carry the signal we want. Matched
# case-insensitively. Kept short and role-agnostic; role-specific tuning
# belongs in Phase 2.
_TRIGGERS: tuple[str, ...] = (
    "skill",
    "responsibilit",
    "requirement",
    "technolog",
    "stack",
    "framework",
    "language",
    "database",
    "protocol",
    "tool",
    "experience with",
    "proficient in",
    "knowledge of",
)

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_BULLET_RE = re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+\S")
_TRIGGER_RE = re.compile(
    "|".join(re.escape(t) for t in _TRIGGERS),
    flags=re.IGNORECASE,
)
# Bullets whose only content is a markdown link to an in-page anchor are
# table-of-contents noise (e.g., `- [Backend engineer](#backend-engineer)`)
# that survives the trigger filter and eats the per-page char budget.
_ANCHOR_LINK_BULLET_RE = re.compile(
    r"^\s{0,3}[-*+]\s+\[[^\]]+\]\(#[^)]*\)\s*$"
)


def _strip_pre_heading(markdown: str) -> str:
    """Drop every line before the first markdown heading.

    Firecrawl's ``onlyMainContent`` is a structural heuristic; it keeps
    inline-rendered chrome (cookie banners, nav strips, newsletter boxes)
    that sits in the DOM above the real content. Anchoring extraction at
    the first ``# `` / ``## `` / ... heading throws that chrome away
    cheaply. If no heading exists the whole document is returned unchanged.
    """
    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        if _HEADING_RE.match(line):
            return "\n".join(lines[i:])
    return markdown


def _is_signal_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _HEADING_RE.match(line):
        return True
    if _ANCHOR_LINK_BULLET_RE.match(line):
        return False
    if _BULLET_RE.match(line):
        return True
    if _TRIGGER_RE.search(stripped):
        return True
    return False


def extract_signal(markdown: str, max_chars: int = 2500) -> str:
    """Return the subset of ``markdown`` that is likely skill-relevant.

    Kept lines: section headings, bullet / numbered list items, and any
    paragraph containing a trigger word. Consecutive kept lines are grouped;
    runs of dropped content collapse to a single blank line for readability.
    Output is truncated to ``max_chars`` at the nearest preceding line break.

    Returns the empty string if no lines matched — callers are expected to
    drop the extract in that case.
    """
    if not markdown:
        return ""

    markdown = _strip_pre_heading(markdown)

    kept: list[str] = []
    last_kept = False
    for raw in markdown.splitlines():
        if _is_signal_line(raw):
            kept.append(raw.rstrip())
            last_kept = True
            continue
        # Insert a single blank line between signal groups, not a flood.
        if last_kept:
            kept.append("")
            last_kept = False

    # Drop trailing blank if we just appended one.
    while kept and not kept[-1]:
        kept.pop()

    if not kept:
        return ""

    joined = "\n".join(kept)
    if len(joined) <= max_chars:
        return joined

    # Truncate at the last full line that fits under the budget.
    cut = joined.rfind("\n", 0, max_chars)
    if cut == -1:
        return joined[:max_chars]
    return joined[:cut]
