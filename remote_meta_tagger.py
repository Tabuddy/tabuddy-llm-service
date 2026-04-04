"""HTTP client for the external meta-tagging API.

Calls https://meta-tag-api.tabuddy.co/parse-resume-hybrid to parse resumes
into structured ResumeTaggingResponse, bypassing local LLM-heavy pipeline.
"""

from __future__ import annotations

import os
import logging

import httpx
from models import ResumeTaggingResponse

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://meta-tag-api.tabuddy.co/parse-resume-hybrid"


async def parse_resume_via_api(
    file_bytes: bytes,
    filename: str = "resume.pdf",
    api_url: str | None = None,
) -> ResumeTaggingResponse:
    """Send a resume file to the external meta-tagging API.

    Args:
        file_bytes: Raw bytes of the resume file (PDF/DOCX/TXT).
        filename: Original filename (used to infer format).
        api_url: Override URL; falls back to env var or default.

    Returns:
        ResumeTaggingResponse with full structured meta-tags.

    Raises:
        httpx.HTTPError: On network or server error.
        ValueError: On invalid JSON response.
    """
    url = api_url or os.getenv("META_TAG_API_URL", _DEFAULT_URL)

    ext = os.path.splitext(filename)[1] or ".pdf"
    # Determine MIME type from extension
    mime_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".txt": "text/plain",
    }
    content_type = mime_map.get(ext.lower(), "application/octet-stream")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            url,
            files={
                "file": (filename, file_bytes, content_type),
            },
        )
        resp.raise_for_status()
        data = resp.json()

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object from meta-tag API, got {type(data).__name__}")

    return ResumeTaggingResponse.model_validate(data)
