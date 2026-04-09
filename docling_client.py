"""Shared Docling DocumentConverter setup tuned to avoid std::bad_alloc / OOM on large PDFs.

Uses lower page raster scale, batch size 1, and optional fallbacks (no OCR, text-only backend).
Override with env: DOCLING_IMAGES_SCALE, DOCLING_OCR_BATCH_SIZE, DOCLING_LAYOUT_BATCH_SIZE,
DOCLING_TABLE_BATCH_SIZE, DOCLING_QUEUE_MAX_SIZE.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)


def _pdf_pipeline_options(
    *,
    do_ocr: bool,
    images_scale: float | None = None,
    force_backend_text: bool = False,
    do_table_structure: bool | None = None,
    device: str = "auto",
):
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    scale = images_scale if images_scale is not None else float(
        os.environ.get("DOCLING_IMAGES_SCALE", "0.72")
    )
    opts = PdfPipelineOptions(
        do_ocr=do_ocr,
        images_scale=scale,
        ocr_batch_size=int(os.environ.get("DOCLING_OCR_BATCH_SIZE", "1")),
        layout_batch_size=int(os.environ.get("DOCLING_LAYOUT_BATCH_SIZE", "1")),
        table_batch_size=int(os.environ.get("DOCLING_TABLE_BATCH_SIZE", "1")),
        queue_max_size=int(os.environ.get("DOCLING_QUEUE_MAX_SIZE", "8")),
        force_backend_text=force_backend_text,
        accelerator_options=AcceleratorOptions(device=device),
    )
    if do_table_structure is not None:
        opts.do_table_structure = do_table_structure
    return opts


def create_document_converter(
    *,
    do_ocr: bool = True,
    images_scale: float | None = None,
    force_backend_text: bool = False,
    do_table_structure: bool | None = None,
    device: str = "auto",
) -> DocumentConverter:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    po = _pdf_pipeline_options(
        do_ocr=do_ocr,
        images_scale=images_scale,
        force_backend_text=force_backend_text,
        do_table_structure=do_table_structure,
        device=device,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=po),
        }
    )


def convert_path_to_markdown(path: str | Path, *, device: str = "auto") -> str:
    """Run Docling on a file path; return markdown or empty string.

    Tries progressively cheaper modes if conversion fails (OOM / preprocess errors).
    """
    from docling.datamodel.base_models import ConversionStatus
    from docling.document_converter import DocumentConverter

    path = Path(path)
    suffix = path.suffix.lower()

    # Non-PDF: use default converter once (pipeline options are PDF-specific).
    if suffix != ".pdf":
        try:
            result = DocumentConverter().convert(str(path), raises_on_error=False)
            if result.document and result.status in (
                ConversionStatus.SUCCESS,
                ConversionStatus.PARTIAL_SUCCESS,
            ):
                return result.document.export_to_markdown()
            logger.warning(
                "Docling non-PDF failed: %s status=%s errors=%s",
                path.name,
                result.status,
                getattr(result, "errors", None),
            )
        except Exception as e:
            logger.warning("Docling non-PDF raised: %s", e)
        return ""

    attempts: list[tuple[str, dict]] = [
        (
            "low_mem_ocr",
            {"do_ocr": True, "images_scale": None, "force_backend_text": False, "do_table_structure": None},
        ),
        (
            "no_ocr",
            {"do_ocr": False, "images_scale": None, "force_backend_text": False, "do_table_structure": None},
        ),
        (
            "text_only",
            {
                "do_ocr": False,
                "images_scale": 0.55,
                "force_backend_text": True,
                "do_table_structure": False,
            },
        ),
    ]

    last_detail: str | None = None

    for label, kwargs in attempts:
        try:
            conv = create_document_converter(**kwargs, device=device)
            result = conv.convert(str(path), raises_on_error=False)
            st = result.status
            if result.document is None:
                last_detail = f"{label}: no document (status={st})"
                logger.warning("Docling %s: %s", label, last_detail)
                continue
            if st in (
                ConversionStatus.SUCCESS,
                ConversionStatus.PARTIAL_SUCCESS,
            ):
                return result.document.export_to_markdown()
            errs = getattr(result, "errors", None)
            last_detail = f"{label}: status={st} errors={errs}"
            logger.warning("Docling %s failed: %s", label, last_detail)
        except Exception as e:
            last_detail = f"{label}: {e}"
            logger.warning("Docling %s raised: %s", label, e)
            continue

    if last_detail:
        logger.warning("Docling gave up on %s: %s", path.name, last_detail)
    return ""


def convert_path_best_effort(path: str | Path, *, device: str = "auto"):
    """Like convert_path_to_markdown but returns the last successful ConversionResult (or None).

    Used by /docling/convert when post-processing needs `result.document` (chunker, etc.).
    """
    from docling.datamodel.base_models import ConversionStatus
    from docling.document_converter import DocumentConverter

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix != ".pdf":
        try:
            result = DocumentConverter().convert(str(path), raises_on_error=False)
            if result.document and result.status in (
                ConversionStatus.SUCCESS,
                ConversionStatus.PARTIAL_SUCCESS,
            ):
                return result
        except Exception as e:
            logger.warning("Docling non-PDF raised: %s", e)
        return None

    attempts: list[tuple[str, dict]] = [
        (
            "low_mem_ocr",
            {"do_ocr": True, "images_scale": None, "force_backend_text": False, "do_table_structure": None},
        ),
        (
            "no_ocr",
            {"do_ocr": False, "images_scale": None, "force_backend_text": False, "do_table_structure": None},
        ),
        (
            "text_only",
            {
                "do_ocr": False,
                "images_scale": 0.55,
                "force_backend_text": True,
                "do_table_structure": False,
            },
        ),
    ]

    for label, kwargs in attempts:
        try:
            conv = create_document_converter(**kwargs, device=device)
            result = conv.convert(str(path), raises_on_error=False)
            st = result.status
            if result.document is None:
                logger.warning("Docling %s: no document (status=%s)", label, st)
                continue
            if st in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
                return result
            logger.warning(
                "Docling %s: status=%s errors=%s",
                label,
                st,
                getattr(result, "errors", None),
            )
        except Exception as e:
            logger.warning("Docling %s raised: %s", label, e)

    return None
