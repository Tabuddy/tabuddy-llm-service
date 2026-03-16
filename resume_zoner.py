"""Resume Zoner – splits raw resume text into structured blocks.

Deterministic section detection using common resume heading patterns.
Then sub-splits experience sections into per-role blocks using date patterns.
Outputs: list of ResumeBlock objects.
"""

from __future__ import annotations

import re
import logging
from models import ResumeBlock

logger = logging.getLogger(__name__)

# Patterns that signal the start of a resume section (case-insensitive).
# Matched against heading text AFTER stripping parenthetical content.
_SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)^(professional\s+)?summary|^profile|^about\s+me|^objective", "summary"),
    (
        r"(?i)^(work\s+)?experience|^employment|^professional\s+experience"
        r"|^career\s+history|^work\s+history",
        "experience",
    ),
    (r"(?i)^projects?|^key\s+projects?|^personal\s+projects?|^side\s+projects?", "project"),
    (
        r"(?i)^(technical\s+)?skills|^core\s+competenc|^competenc"
        r"|^technologies|^tools\s*(&|and)|^tech\s+stack",
        "skills_dump",
    ),
    (r"(?i)^education|^academic|^qualifications?|^degrees?", "education"),
    (
        r"(?i)^certific|^awards?|^honors?|^achievements?"
        r"|^publications?|^interests?|^extracurric|^volunteer",
        "other",
    ),
]

# Date range pattern used to detect role boundaries inside experience sections.
_DATE_RANGE_RE = re.compile(
    r"(?:"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"['\u2019]?\s*\d{2,4}"
    r"|(?:19|20)\d{2}"
    r")"
    r"\s*[-–—to]+\s*"
    r"(?:"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"['\u2019]?\s*\d{2,4}"
    r"|(?:19|20)\d{2}"
    r"|Present|Current|Ongoing|Now|Till\s+Date"
    r")",
    re.IGNORECASE,
)


def _strip_parens(text: str) -> str:
    """Remove parenthetical content for heading classification."""
    return re.sub(r"\s*\(.*?\)", "", text).strip()


def _classify_heading(line: str) -> str | None:
    """Return block_type if the line looks like a section heading, else None."""
    stripped = line.strip().rstrip(":").strip()
    if not stripped or len(stripped) > 100:
        return None
    # Strip parenthetical content before matching (e.g. "WORK EXPERIENCE (9 years)")
    clean = _strip_parens(stripped)
    if not clean:
        return None
    for pattern, btype in _SECTION_PATTERNS:
        if re.search(pattern, clean):
            return btype
    return None


def _is_heading_line(line: str) -> bool:
    """Heuristic: headings are short, may be ALL-CAPS or end with colon."""
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 100:
        return False

    # Strip parenthetical content before case check
    no_parens = _strip_parens(stripped)
    if not no_parens:
        return False

    # ALL CAPS check on the non-parenthetical part
    alpha_part = re.sub(r"[^a-zA-Z\s]", "", no_parens).strip()
    if alpha_part and len(alpha_part) >= 3 and alpha_part == alpha_part.upper():
        return True

    # Ends with colon
    if stripped.endswith(":"):
        return True

    # Title-case with section keyword match (fallback)
    if _classify_heading(stripped) is not None and len(stripped) < 50:
        return True

    return False


def _sub_split_experience(text: str) -> list[tuple[str, str]]:
    """Sub-split an experience section into per-role blocks using date ranges.

    Returns list of (role_label, text) tuples.
    Each date range line is assumed to mark a role boundary.
    """
    lines = text.split("\n")
    date_line_indices = [i for i, line in enumerate(
        lines) if _DATE_RANGE_RE.search(line)]

    if not date_line_indices:
        return [("role", text)]

    # For each date-range line, walk backward to find the start of that role block.
    # Roles are typically preceded by a blank line, or by company/title lines.
    role_starts: list[int] = []
    for di in date_line_indices:
        start = di
        # Walk back up to 4 lines to find the role header start
        for j in range(di - 1, max(0, di - 5) - 1, -1):
            if not lines[j].strip():
                start = j + 1
                break
            start = j
        role_starts.append(start)

    # Deduplicate and sort
    role_starts = sorted(set(role_starts))

    sub_blocks: list[tuple[str, str]] = []

    # If there's content before the first role, include it
    if role_starts[0] > 0:
        pre_text = "\n".join(lines[: role_starts[0]]).strip()
        if pre_text:
            sub_blocks.append(("preamble", pre_text))

    for idx, start in enumerate(role_starts):
        end = role_starts[idx + 1] if idx + \
            1 < len(role_starts) else len(lines)
        block_text = "\n".join(lines[start:end]).strip()
        if block_text:
            sub_blocks.append(("role", block_text))

    return sub_blocks


def zone_resume(raw_text: str) -> list[ResumeBlock]:
    """Split resume text into structured blocks."""
    lines = raw_text.split("\n")
    # (heading, block_type, lines)
    sections: list[tuple[str, str, list[str]]] = []

    current_heading = "Header"
    current_type = "other"
    current_lines: list[str] = []

    for line in lines:
        btype = _classify_heading(line)
        if btype and _is_heading_line(line):
            # Save previous section
            if current_lines:
                sections.append((current_heading, current_type, current_lines))
            current_heading = line.strip().rstrip(":").strip()
            current_type = btype
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_lines:
        sections.append((current_heading, current_type, current_lines))

    # Build ResumeBlock objects with proper naming
    blocks: list[ResumeBlock] = []
    experience_counter = 0
    project_counter = 0

    for heading, btype, sec_lines in sections:
        text = "\n".join(sec_lines).strip()
        if not text:
            continue

        if btype == "experience":
            # Sub-split experience section into per-role blocks
            sub_blocks = _sub_split_experience(text)
            for label, sub_text in sub_blocks:
                if label == "preamble":
                    blocks.append(
                        ResumeBlock(
                            block_name="Experience_Preamble",
                            raw_text=sub_text,
                            block_type="experience",
                        )
                    )
                else:
                    blocks.append(
                        ResumeBlock(
                            block_name=f"Experience_{experience_counter}",
                            raw_text=sub_text,
                            block_type="experience",
                        )
                    )
                    experience_counter += 1
        elif btype == "project":
            # Sub-split project section into per-project blocks
            sub_blocks = _sub_split_experience(text)
            for label, sub_text in sub_blocks:
                if label == "preamble":
                    blocks.append(
                        ResumeBlock(
                            block_name="Project_Preamble",
                            raw_text=sub_text,
                            block_type="project",
                        )
                    )
                else:
                    blocks.append(
                        ResumeBlock(
                            block_name=f"Project_{project_counter}",
                            raw_text=sub_text,
                            block_type="project",
                        )
                    )
                    project_counter += 1
            # If no sub-splits happened (no date ranges), treat as single project
            if not sub_blocks:
                blocks.append(
                    ResumeBlock(
                        block_name=f"Project_{project_counter}",
                        raw_text=text,
                        block_type="project",
                    )
                )
                project_counter += 1
        elif btype == "summary":
            blocks.append(
                ResumeBlock(block_name="Summary",
                            raw_text=text, block_type="summary")
            )
        elif btype == "skills_dump":
            blocks.append(
                ResumeBlock(
                    block_name="Skills_Dump", raw_text=text, block_type="skills_dump"
                )
            )
        elif btype == "education":
            blocks.append(
                ResumeBlock(
                    block_name="Education", raw_text=text, block_type="education"
                )
            )
        else:
            blocks.append(
                ResumeBlock(
                    block_name=heading or "Other", raw_text=text, block_type="other"
                )
            )

    # If no sections were detected, treat the whole thing as one block
    if not blocks:
        blocks.append(
            ResumeBlock(
                block_name="Full_Resume",
                raw_text=raw_text.strip(),
                block_type="other",
            )
        )

    # Post-process: if no summary block exists, try to extract one from the
    # Header block. Many resumes have a short profile/summary paragraph right
    # after the name and title, before any named section.
    has_summary = any(b.block_type == "summary" for b in blocks)
    if not has_summary:
        blocks = _extract_implicit_summary(blocks)

    return blocks


def _is_contact_line(line: str) -> bool:
    """Return True if the line looks like contact info (email, phone, URL)."""
    s = line.strip()
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", s):
        return True
    if re.search(r"\+?\d[\d\s\-().]{7,}", s):
        return True
    if re.search(r"https?://|linkedin\.com|github\.com|www\.", s, re.IGNORECASE):
        return True
    return False


def _extract_implicit_summary(blocks: list[ResumeBlock]) -> list[ResumeBlock]:
    """If the Header/other block contains paragraph-like sentences, split them
    out as a Summary block. Resumes often have an unlabelled profile blurb
    right after the name line(s)."""
    header_idx: int | None = None
    for i, b in enumerate(blocks):
        if b.block_type == "other" and b.block_name in ("Header", "Other"):
            header_idx = i
            break
    if header_idx is None:
        return blocks

    header = blocks[header_idx]
    lines = header.raw_text.split("\n")
    summary_lines: list[str] = []
    remaining_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            remaining_lines.append(line)
            continue
        # Skip contact info and very short lines (names, titles)
        if _is_contact_line(stripped):
            remaining_lines.append(line)
            continue
        word_count = len(stripped.split())
        # Directly qualifies: long sentence-like line
        if word_count >= 8 and len(stripped) >= 40:
            summary_lines.append(stripped)
        else:
            remaining_lines.append(line)

    # Fallback: if no individual long lines found, try grouping consecutive
    # non-heading, non-contact lines to see if they form a paragraph.
    if not summary_lines:
        candidate_group: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or _is_contact_line(stripped):
                if candidate_group:
                    joined = " ".join(candidate_group)
                    if len(joined.split()) >= 8 and len(joined) >= 40:
                        summary_lines = candidate_group
                        break
                    candidate_group = []
                continue
            # Skip ALL-CAPS short headings and very short tokens
            alpha = re.sub(r"[^a-zA-Z\s]", "", stripped).strip()
            if alpha and alpha == alpha.upper() and len(stripped.split()) <= 5:
                if candidate_group:
                    joined = " ".join(candidate_group)
                    if len(joined.split()) >= 8 and len(joined) >= 40:
                        summary_lines = candidate_group
                        break
                    candidate_group = []
                continue
            candidate_group.append(stripped)
        # Check last group
        if not summary_lines and candidate_group:
            joined = " ".join(candidate_group)
            if len(joined.split()) >= 8 and len(joined) >= 40:
                summary_lines = candidate_group

        # Rebuild remaining_lines excluding the summary lines we found
        if summary_lines:
            summary_set = set(summary_lines)
            remaining_lines = [
                l for l in lines if l.strip() not in summary_set]

    if not summary_lines:
        return blocks

    summary_text = "\n".join(summary_lines).strip()
    remaining_text = "\n".join(remaining_lines).strip()

    new_blocks = list(blocks)
    # Update the header block to only contain the non-summary lines
    if remaining_text:
        new_blocks[header_idx] = ResumeBlock(
            block_name=header.block_name,
            raw_text=remaining_text,
            block_type="other",
        )
    else:
        new_blocks.pop(header_idx)

    # Insert the summary block right after the header (or at position 0)
    insert_pos = header_idx + 1 if remaining_text else header_idx
    new_blocks.insert(
        insert_pos,
        ResumeBlock(block_name="Summary", raw_text=summary_text,
                    block_type="summary"),
    )

    return new_blocks
