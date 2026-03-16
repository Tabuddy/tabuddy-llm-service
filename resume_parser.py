"""Resume file parser – extracts raw text from PDF and DOCX uploads."""

from __future__ import annotations

import io
import logging
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)


async def extract_text(file: UploadFile) -> str:
    """Read an uploaded file and return its full plain text."""
    content = await file.read()
    filename = (file.filename or "").lower()

    if filename.endswith(".pdf"):
        return _parse_pdf(content)
    elif filename.endswith(".docx"):
        return _parse_docx(content)
    elif filename.endswith(".txt"):
        return content.decode("utf-8", errors="replace")
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a .pdf, .docx, or .txt resume.",
        )


def _parse_pdf(data: bytes) -> str:
    import pdfplumber

    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    full = "\n".join(text_parts).strip()
    if not full:
        raise HTTPException(
            status_code=422, detail="Could not extract text from PDF.")
    return full


def _parse_docx(data: bytes) -> str:
    import docx

    doc = docx.Document(io.BytesIO(data))
    text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
    full = "\n".join(text_parts).strip()
    if not full:
        raise HTTPException(
            status_code=422, detail="Could not extract text from DOCX."
        )
    return full
