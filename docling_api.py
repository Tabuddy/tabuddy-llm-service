"""Docling conversion routes (mounted on main app at /docling/*).

Install Docling yourself (same venv as this project is fine):
    uv pip install docling

Optional extras (OCR, VLM, etc.):
    uv pip install "docling[rapidocr]"

When you run the main service (e.g. `uv run fastapi dev`), Swagger at :8000/docs
includes:
    GET  /docling/health
    POST /docling/convert

Optional standalone server on another port:
    uv run uvicorn docling_api:app --reload --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel


class ConvertResponse(BaseModel):
    markdown: str
    filename: str


class HealthResponse(BaseModel):
    ok: bool
    docling_available: bool
    message: str


def _try_import_docling():
    try:
        from docling.document_converter import DocumentConverter  # noqa: F401

        return True, None
    except ImportError as e:
        return False, str(e)


_DOC_AVAILABLE, _DOC_IMPORT_ERROR = _try_import_docling()

docling_router = APIRouter(
    prefix="/docling",
    tags=["docling"],
)


@docling_router.get("/health", response_model=HealthResponse)
async def health():
    if _DOC_AVAILABLE:
        return HealthResponse(
            ok=True,
            docling_available=True,
            message="Docling is importable.",
        )
    return HealthResponse(
        ok=False,
        docling_available=False,
        message=f"Docling not installed or import failed: {_DOC_IMPORT_ERROR}",
    )


@docling_router.post("/convert", response_model=ConvertResponse)
async def convert_file(file: UploadFile = File(...)):
    """Upload a file; Docling converts it and returns Markdown.

    Docling supports PDF, DOCX, and more — see Docling docs. This endpoint
    saves the upload to a temp file and passes the path to DocumentConverter.
    """
    if not _DOC_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Docling is not available. Install with: pip install docling. Import error: {_DOC_IMPORT_ERROR}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    suffix = Path(file.filename or "document").suffix or ".pdf"
    # Keep suffix so Docling can infer format when needed
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        from docling_client import convert_path_best_effort

        result = convert_path_best_effort(tmp_path)
        if result is None or result.document is None:
            raise HTTPException(
                status_code=503,
                detail="Docling could not convert this document (out of memory or unsupported layout). "
                "Try a smaller PDF, export as PDF from Word, or use /parse-resume without hybrid.",
            )

        doc = result.document
        markdown = None

        # 1) Section-aware output using Docling chunking + contextual serialization.
        #    This avoids mixing two-column layouts line-by-line and instead keeps
        #    whole sections (SUMMARY/EXPERIENCE/PROJECT/...) together.
        try:
            from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

            chunker = HierarchicalChunker()
            chunks = list(chunker.chunk(dl_doc=doc))

            section_keywords = [
                "SUMMARY",
                "EXPERIENCE",
                "PROJECT",
                "EDUCATION",
                "SKILLS",
                "STRENGTHS",
                "INDUSTRY",
                "INTERESTS",
                "LANGUAGES",
            ]

            def _clean_text(s: str) -> str:
                # Remove private-use glyphs that can break terminals.
                return s.replace("\u0000", "").strip().translate(
                    {i: None for i in range(0xE000, 0xF8FF + 1)}
                )

            def _detect_section(ctx: str) -> str | None:
                # Use word-like matching to avoid accidental substring hits.
                import re

                u = ctx.upper()
                for kw in section_keywords:
                    pat = rf"\b{re.escape(kw)}\b"
                    if re.search(pat, u):
                        return kw
                return None

            current_section: str | None = None
            blocks: list[str] = []
            buffer: list[str] = []
            prelude: list[str] = []

            def _dedupe_lines(block_text: str) -> str:
                # Remove repeated header lines that show up multiple times due to
                # docling contextualization over chunks.
                lines = [ln.strip() for ln in block_text.splitlines() if ln.strip()]
                out: list[str] = []
                recent: list[str] = []
                for ln in lines:
                    ln_norm = " ".join(ln.split())
                    if len(ln_norm) < 2 and not ln_norm.startswith("-"):
                        continue
                    if out and ln_norm == out[-1]:
                        continue
                    if ln_norm in recent:
                        continue
                    out.append(ln_norm)
                    recent.append(ln_norm)
                    if len(recent) > 12:
                        recent.pop(0)
                return "\n".join(out).strip()

            for ch in chunks:
                ctx = chunker.contextualize(ch)
                ctx_clean = _clean_text(ctx)
                if not ctx_clean:
                    continue

                sec = _detect_section(ctx_clean)
                if sec is not None and sec != current_section:
                    # If this is the first detected section, flush any pre-section
                    # text (typically candidate name/contact) as a separate block.
                    if current_section is None and prelude:
                        prelude_text = _dedupe_lines("\n\n".join(prelude))
                        if prelude_text:
                            blocks.append("RESUME HEADER\n" + prelude_text)
                        prelude = []

                    # Close previous section block.
                    if buffer:
                        blocks.append(_dedupe_lines("\n\n".join(buffer)))
                        buffer = []
                    current_section = sec

                # Keep any leading text before we detect the first section.
                if current_section is None:
                    prelude.append(ctx_clean)
                    continue

                # If this chunk doesn't explicitly contain a heading, it belongs
                # to the current section.
                buffer.append(ctx_clean)

            if buffer:
                blocks.append(_dedupe_lines("\n\n".join(buffer)))
            if not blocks and prelude:
                prelude_text = _dedupe_lines("\n\n".join(prelude))
                if prelude_text:
                    blocks.append("RESUME HEADER\n" + prelude_text)

            if blocks:
                markdown = "\n\n".join(blocks)
        except Exception:
            markdown = None

        # 2) Fallback: bbox-based left/right column grouping.
        if markdown is None:
            try:
                exported = doc.export_to_dict()
                texts = exported.get("texts") or []
                pages = exported.get("pages") or {}

                def _page_size(pn: int) -> dict:
                    # `pages` uses string keys in exported dicts (e.g. "1")
                    page = pages.get(str(pn)) or pages.get(pn) or {}
                    return page.get("size") or {}

                def _page_height(pn: int) -> float:
                    size = _page_size(pn)
                    h = size.get("height")
                    return float(h) if h else 842.0

                def _page_width(pn: int) -> float:
                    size = _page_size(pn)
                    w = size.get("width")
                    return float(w) if w else 595.0

                segments: list[tuple[int, float, float, float, float, str]] = []
                # (page_no, x_center, y_top, x_left, x_right, text)
                for t in texts:
                    bbox = t.get("bbox") or {}
                    if not bbox:
                        continue

                    l = bbox.get("l")
                    r = bbox.get("r")
                    y_top = bbox.get("t")
                    if l is None or r is None or y_top is None:
                        continue

                    text = (t.get("text") or "").strip()
                    if not text:
                        continue

                    page_no = t.get("page_no")
                    if page_no is None:
                        # Fallback: assume single page
                        page_no = 1

                    try:
                        page_no_int = int(page_no)
                        l_f = float(l)
                        r_f = float(r)
                        x_center = (l_f + r_f) / 2.0
                        y_top_f = float(y_top)
                    except Exception:
                        continue

                    segments.append((page_no_int, x_center, y_top_f, l_f, r_f, text))

                if len(segments) >= 2:
                    pass
                # Group segments per page so the column split is computed per-page.
                segs_by_page: dict[int, list[tuple[int, float, float, float, float, str]]] = {}
                for s in segments:
                    segs_by_page.setdefault(s[0], []).append(s)

                def _build_lines(page_no: int, col_segments: list[tuple[int, float, float, float, float, str]]) -> list[str]:
                    # Sort top-to-bottom, then left-to-right for tokens in the same line.
                    col_segments_sorted = sorted(col_segments, key=lambda s: (s[2], s[1]))
                    tol = _page_height(page_no) * 0.015  # smaller tolerance to avoid merging lines

                    out_lines: list[str] = []
                    current_y: float | None = None
                    current_line: list[tuple[float, str]] = []

                    for _, x_center, y_top, _, _, text in col_segments_sorted:
                        if current_y is None:
                            current_y = y_top
                            current_line = [(x_center, text)]
                            continue

                        if abs(y_top - current_y) > tol:
                            # flush previous line
                            if current_line:
                                out_lines.append(" ".join(t for _, t in sorted(current_line)))
                            current_line = [(x_center, text)]
                            current_y = y_top
                        else:
                            current_line.append((x_center, text))

                    if current_line:
                        out_lines.append(" ".join(t for _, t in sorted(current_line)))

                    cleaned: list[str] = []
                    for ln in out_lines:
                        s2 = ln.strip()
                        if not s2 or len(s2) < 2:
                            continue
                        cleaned.append(s2)
                    return cleaned

                output_blocks: list[str] = []

                for pn in sorted(segs_by_page.keys()):
                    page_segs = segs_by_page[pn]
                    page_w = _page_width(pn)

                    # Treat wide items as "spanning": headers/contact info that span both columns.
                    # We exclude them from determining the split boundary.
                    spans = [s for s in page_segs if (s[4] - s[3]) > (0.75 * page_w)]
                    candidates = [s for s in page_segs if s not in spans]

                    xs = sorted([s[1] for s in candidates if s[1] is not None])
                    split = None
                    if len(xs) >= 2:
                        # Split at the largest gap between consecutive x-centers.
                        gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
                        gaps.sort(reverse=True)
                        gap_val, i = gaps[0]
                        # If the largest gap is too small, fall back to the median.
                        if gap_val > (0.03 * page_w):
                            split = (xs[i] + xs[i + 1]) / 2.0
                        else:
                            split = (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2.0
                    else:
                        split = (min([s[1] for s in page_segs]) + max([s[1] for s in page_segs])) / 2.0

                    left = [s for s in candidates if s[1] <= split]
                    right = [s for s in candidates if s[1] > split]

                    spanning_lines = _build_lines(pn, spans) if spans else []
                    left_lines = _build_lines(pn, left) if left else []
                    right_lines = _build_lines(pn, right) if right else []

                    if spanning_lines:
                        output_blocks.append("\n".join(spanning_lines))
                    if left_lines:
                        output_blocks.append("\n".join(left_lines))
                    if right_lines:
                        output_blocks.append("\n".join(right_lines))

                if output_blocks:
                    markdown = "\n\n".join(output_blocks)
            except Exception:
                markdown = None

        if markdown is None:
            markdown = doc.export_to_markdown()

        return ConvertResponse(markdown=markdown, filename=file.filename or "upload")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Docling conversion failed: {e}",
        ) from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Standalone app (optional: uvicorn docling_api:app --port 8001)
app = FastAPI(
    title="Docling conversion service",
    description="Convert PDFs (and other Docling-supported formats) to Markdown.",
    version="0.1.0",
)
app.include_router(docling_router)
