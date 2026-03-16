"""Candidate detail extractor – pulls contact info from resume text using regex."""

from __future__ import annotations

import re
import logging
from models import CandidateDetails

logger = logging.getLogger(__name__)

# ── Regex patterns ──
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-.]?)?"  # country code
    r"\(?\d{2,4}\)?[\s\-.]?"     # area code
    r"\d{3,4}[\s\-.]?"           # first part
    r"\d{3,4}"                   # second part
)
_LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_\-]+/?", re.IGNORECASE
)
_GITHUB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9_\-]+/?", re.IGNORECASE
)


def extract_candidate_details(raw_text: str) -> CandidateDetails:
    """Extract candidate contact details from resume text using regex heuristics."""
    # Use only the first ~30 lines (header area) for name/contact info
    header_lines = raw_text.split("\n")[:30]
    header_text = "\n".join(header_lines)

    email = _first_match(_EMAIL_RE, header_text)
    phone = _first_match(_PHONE_RE, header_text)
    linkedin = _first_match(_LINKEDIN_RE, raw_text)  # can appear anywhere
    github = _first_match(_GITHUB_RE, raw_text)

    name = _extract_name(header_lines)

    return CandidateDetails(
        name=name,
        email=email,
        phone=phone,
        linkedin_url=linkedin,
        github_url=github,
    )


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    m = pattern.search(text)
    return m.group(0).strip() if m else None


def _extract_name(header_lines: list[str]) -> str | None:
    """Heuristic: the first non-empty line that isn't an email/phone/url is likely the name."""
    for line in header_lines[:5]:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip lines that are clearly not names
        if _EMAIL_RE.search(stripped):
            continue
        if _PHONE_RE.fullmatch(stripped):
            continue
        if "linkedin.com" in stripped.lower() or "github.com" in stripped.lower():
            continue
        if "http" in stripped.lower():
            continue
        # Likely a name if it's short-ish and mostly alpha
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in stripped) / max(
            len(stripped), 1
        )
        if alpha_ratio > 0.7 and len(stripped) < 60:
            return stripped
    return None
