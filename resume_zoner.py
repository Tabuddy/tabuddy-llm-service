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
# Project pattern is anchored to end of line so job titles like "Project Leader"
# or "Project Manager" are not mistaken for the Projects section.
_SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)^(professional\s+)?summary|^profile|^about\s+me|^objective", "summary"),
    (r"(?i)^project\s+experience", "project"),
    (
        r"(?i)^(work\s+)?experience|^employment|^professional\s+experience"
        r"|^career\s+history|^work\s+history"
        r"|^organizational\s+experiences?$",
        "experience",
    ),
    (
        r"(?i)(?:^key\s+projects?|^personal\s+projects?|^side\s+projects?|^projects?)\s*:?\s*$",
        "project",
    ),
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
# Supports: "Jan 2024 – Present", "02/2024 – Present", "07/2023 – 02/2024"
_DATE_RANGE_RE = re.compile(
    r"(?:"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"['\u2019]?\s*\d{2,4}"
    r"|(?:19|20)\d{2}"
    r"|\d{1,2}/\d{2,4}"  # numeric e.g. 02/2024, 07/2023
    r")"
    r"\s*[-–—to]+\s*"
    r"(?:"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"['\u2019]?\s*\d{2,4}"
    r"|(?:19|20)\d{2}"
    r"|\d{1,2}/\d{2,4}"
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


# Words that typically appear in job/role titles (not company names)
_ROLE_TITLE_WORDS_RE = re.compile(
    r"\b(engineer|developer|manager|lead|architect|analyst|consultant|"
    r"specialist|coordinator|director|designer|software|system|principal|"
    r"senior|staff|associate|head|officer)\b",
    re.IGNORECASE,
)


def _looks_like_role_title(line: str) -> bool:
    """True if the line looks like a role/title (e.g. 'Lead Software Developer')."""
    stripped = line.strip()
    if not stripped or _DATE_RANGE_RE.search(stripped):
        return False
    # "Role | date" is always a role line
    if "|" in stripped:
        return True
    return bool(_ROLE_TITLE_WORDS_RE.search(stripped))


def _sub_split_experience(text: str) -> list[tuple[str, str]]:
    """Sub-split an experience section into per-company blocks.

    Splits only when a date line is preceded by a company name (no "|", no role-title
    words), so we get one block per company with all roles and bullets for that company.
    """
    lines = text.split("\n")
    date_line_indices = [i for i, line in enumerate(lines) if _DATE_RANGE_RE.search(line)]

    if not date_line_indices:
        return [("role", text)]

    # New block only when the line before this date line looks like a company name:
    # no "|" (so not "Role | date") and no role-title keywords.
    company_start_indices: list[int] = []
    for di in date_line_indices:
        if di == 0:
            company_start_indices.append(0)
            continue
        prev_line = lines[di - 1].strip()
        if not prev_line:
            company_start_indices.append(di)
            continue
        # Same company if prev line is "Role | date", a role title, or bullet content
        if "|" in prev_line or _looks_like_role_title(prev_line):
            continue
        if prev_line.startswith("●") or prev_line.startswith("*") or prev_line.startswith("-"):
            continue
        # Prev line looks like company name – new block starts at that line (don't walk back further)
        company_start_indices.append(di - 1)

    if not company_start_indices:
        return [("role", text)]

    role_starts = sorted(set(company_start_indices))

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


# Project title pattern: "Project Name – Company" or "Project Name (context) – Company"
_PROJECT_TITLE_RE = re.compile(r".+\s+[–\-]\s+\S.+", re.IGNORECASE)


def _sub_split_projects(text: str) -> list[tuple[str, str]]:
    """Sub-split a project section into per-project blocks.

    Splits on lines that look like project titles (e.g. "EagleApps (Boston College) – KPMG").
    Returns list of (label, text) tuples; label is "preamble" or "role" (one per project).
    """
    lines = text.split("\n")
    # Find lines that look like project titles: contain " – " or " - " and are not bullets
    title_indices: list[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("●") or stripped.startswith("*"):
            continue
        # Project title: has dash/em-dash with text on both sides, and not a date range
        if _PROJECT_TITLE_RE.search(stripped) and not _DATE_RANGE_RE.search(stripped):
            title_indices.append(i)

    if not title_indices:
        # Fallback: split by date-range lines that look like project headers (no bullets)
        date_indices: list[int] = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("●") or stripped.startswith("*") or stripped.startswith("-"):
                continue
            if _DATE_RANGE_RE.search(stripped):
                date_indices.append(i)
        if not date_indices:
            return [("role", text)]
        sub_blocks: list[tuple[str, str]] = []
        for idx, start in enumerate(date_indices):
            end = date_indices[idx + 1] if idx + 1 < len(date_indices) else len(lines)
            block_text = "\n".join(lines[start:end]).strip()
            if block_text:
                sub_blocks.append(("role", block_text))
        return sub_blocks if sub_blocks else [("role", text)]

    sub_blocks: list[tuple[str, str]] = []
    if title_indices[0] > 0:
        pre = "\n".join(lines[: title_indices[0]]).strip()
        if pre:
            sub_blocks.append(("preamble", pre))

    for idx, start in enumerate(title_indices):
        end = title_indices[idx + 1] if idx + 1 < len(title_indices) else len(lines)
        block_text = "\n".join(lines[start:end]).strip()
        if block_text:
            sub_blocks.append(("role", block_text))

    return sub_blocks if sub_blocks else [("role", text)]


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
            # Sub-split project section by project title lines (e.g. "EagleApps – KPMG")
            sub_blocks = _sub_split_projects(text)
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
