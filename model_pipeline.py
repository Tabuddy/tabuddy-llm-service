"""HTTP endpoints for the Model Pipeline Management dashboard.

Provides API endpoints to:
  - View / clear low-confidence logs
  - Trigger background training
  - Hot-reload models after training
  - Upload / download models from Azure Blob Storage
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import setfit_classifier as clf
from model_azure import download_models_from_azure, upload_models_to_azure

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["model-pipeline"])

_BASE_DIR = Path(__file__).parent
_LOW_CONF_LOG = _BASE_DIR / "low_confidence_log.jsonl"

_MODEL_PATHS = {
    "tier1": _BASE_DIR / "setfit_models" / "tier1_router",
    "tier2": _BASE_DIR / "setfit_models" / "tier2_digital",
    "tier3": _BASE_DIR / "setfit_models" / "tier3_app_eng",
}

# ── Training State (global, shared between endpoints) ─────────────────────────
_training_state: dict[str, Any] = {
    "status": "idle",           # idle | training | completed | failed
    "started_at": None,
    "finished_at": None,
    "message": None,
    "layers_trained": [],
}
_lock = threading.Lock()

# ── Azure Sync Timestamps ─────────────────────────────────────────────────────
_last_azure_download: str | None = None
_last_azure_upload: str | None = None


# ── Tier metadata map ─────────────────────────────────────────────────────────
_TIER_META = {
    "tier1": {"display": "Tier 1 Router", "labels": ["Digital", "Physical", "Service"]},
    "tier2": {"display": "Tier 2 Digital", "labels": [
        "App_Engineering", "Data_Intelligence", "Infra_Cloud",
        "Product_Design", "Cyber_Security",
    ]},
    "tier3": {"display": "Tier 3 App Engineering", "labels": [
        "Stack_Java", "Stack_Python", "Stack_Node", "Stack_React",
        "Stack_ReactNative", "Stack_Angular", "Stack_iOS", "Stack_Android",
    ]},
}


# ── API Endpoint Models ───────────────────────────────────────────────────────
class ClearLogResponse(BaseModel):
    message: str
    entries_removed: int


class TrainingStatusResponse(BaseModel):
    status: str
    started_at: str | None
    finished_at: str | None
    message: str | None
    layers_trained: list[str]


class ModelStateItem(BaseModel):
    tier: str
    display: str
    exists: bool
    size_mb: float | None
    modified: str | None
    label_count: int
    labels: list[str]
    loaded: bool


class ModelStatusResponse(BaseModel):
    models: list[ModelStateItem]
    models_loaded: bool
    loaded_tiers: list[str]


class AzureActionResult(BaseModel):
    success: bool
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_training_background():
    """Background worker that runs all three tiers then hot-reloads."""
    global _training_state
    from setfit_trainer import TIER1_DATA, TIER2_DIGITAL_DATA, TIER3_APP_ENG_DATA, train_layer

    _datasets = {
        "tier1": TIER1_DATA,
        "tier2": TIER2_DIGITAL_DATA,
        "tier3": TIER3_APP_ENG_DATA,
    }

    with _lock:
        _training_state = {
            "status": "training",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "message": None,
            "layers_trained": [],
        }

    layers_trained: list[str] = []
    error_msg: str | None = None

    # Train each layer sequentially (CPU-bound, SetFit trainer is synchronous)
    for tier_key in ["tier1", "tier2", "tier3"]:
        data = _datasets[tier_key]
        output_path = _MODEL_PATHS[tier_key]
        try:
            logger.info("🚀 Training %s (%d examples)", tier_key, len(data))
            train_layer(tier_key.upper(), data, output_path)
            layers_trained.append(tier_key)
            with _lock:
                _training_state["layers_trained"] = list(layers_trained)
            logger.info("✅ Completed %s", tier_key)
        except Exception as exc:
            error_msg = f"Failed to train {tier_key}: {exc}"
            logger.exception(error_msg)
            break

    # After training, upload to Azure and hot-reload
    if not error_msg:
        # Upload to Azure
        try:
            result = upload_models_to_azure()
            if result.get("success"):
                global _last_azure_upload
                _last_azure_upload = datetime.now(timezone.utc).isoformat()
                logger.info("✅ Models uploaded to Azure (%d files)", result.get("uploaded", 0))
            else:
                logger.warning("⚠️ Azure upload skipped: %s", result.get("error"))
        except Exception as _e:
            logger.warning("⚠️ Azure upload error (non-fatal): %s", _e)

        # Hot-reload models in memory
        try:
            reload_results = clf.reload_models()
            logger.info("✅ Hot-reload results: %s", reload_results)
        except Exception as _e:
            error_msg = f"Hot-reload failed: {_e}"
            logger.exception(error_msg)

        # After successful training and reload, clear low-confidence logs to start fresh
        if not error_msg:
            try:
                with open(_LOW_CONF_LOG, 'w', encoding='utf-8') as f:
                    f.truncate(0)
                logger.info("Cleared low-confidence logs after training completion")
            except Exception as e:
                logger.warning("Could not clear low-confidence logs: %s", e)

    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        _training_state["status"] = "completed" if not error_msg else "failed"
        _training_state["finished_at"] = now
        _training_state["message"] = error_msg
        _training_state["layers_trained"] = layers_trained


def _get_model_status() -> list[dict]:
    """Return metadata for each tier model on disk and in memory."""
    results: list[dict] = []
    for tier_key, meta in _TIER_META.items():
        path = _MODEL_PATHS[tier_key]
        exists = path.exists()
        size_mb = None
        modified = None
        if exists:
            # Estimate model size from model.safetensors (largest file)
            sts = path / "model.safetensors"
            if sts.exists():
                size_mb = round(sts.stat().st_size / (1024 * 1024), 1)
            modified = datetime.fromtimestamp(path.stat().st_mtime).isoformat()

        results.append({
            "tier": tier_key,
            "display": meta["display"],
            "exists": exists,
            "size_mb": size_mb,
            "modified": modified,
            "label_count": len(meta["labels"]),
            "labels": meta["labels"],
            "loaded": clf._models.get(tier_key) is not None,
        })
    return results


# ── HTML Page ─────────────────────────────────────────────────────────────────

from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

_templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))


@router.get("/models-pipeline", response_class=HTMLResponse)
async def pipeline_page(request: Request):
    return _templates.TemplateResponse("pipeline.html", {
        "request": request,
    })


# ── API: Training Status ──────────────────────────────────────────────────────

@router.get("/api/training/status", response_model=TrainingStatusResponse)
async def training_status():
    with _lock:
        state = dict(_training_state)
    return state


@router.post("/api/training/start")
async def training_start():
    """Trigger background training. Returns immediately. Poll /status for progress."""
    with _lock:
        if _training_state["status"] == "training":
            raise HTTPException(status_code=409, detail="Training already in progress")

    thread = threading.Thread(target=_run_training_background, daemon=True)
    thread.start()
    return {"message": "Training started in background"}


# ── API: Low Confidence Logs ──────────────────────────────────────────────────

@router.get("/api/training/logs")
async def training_logs(
    layer: str | None = Query(None, description="Filter by tier name"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    per_page: int = Query(25, ge=1, le=100, description="Entries per page"),
    sort: str = Query('desc', regex='^(asc|desc)$', description="Sort order: 'asc' for oldest first, 'desc' for newest first")
):
    """Read low_confidence_log.jsonl and return paginated, sorted entries."""
    if not _LOW_CONF_LOG.exists():
        return JSONResponse(content={"entries": [], "count": 0, "page": page, "per_page": per_page, "total_pages": 0})

    entries: list[dict] = []
    try:
        with open(_LOW_CONF_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        if layer is None or entry.get("layer") == layer:
                            entries.append(entry)
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass

    total = len(entries)

    # Sort by order: log file is chronological (oldest first). For 'desc', reverse to newest first.
    if sort == 'desc':
        entries.reverse()

    # Paginate
    start = (page - 1) * per_page
    end = start + per_page
    paged_entries = entries[start:end]

    total_pages = (total + per_page - 1) // per_page if total else 0

    return JSONResponse(content={
        "entries": paged_entries,
        "count": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages
    })


@router.post("/api/training/clear-logs", response_model=ClearLogResponse)
async def clear_logs():
    """Truncate the low-confidence log file."""
    if not _LOW_CONF_LOG.exists():
        return ClearLogResponse(message="Log file does not exist", entries_removed=0)

    count = _LOW_CONF_LOG.stat().st_size
    # Count lines
    removed = 0
    try:
        with open(_LOW_CONF_LOG, "r", encoding="utf-8") as f:
            removed = sum(1 for line in f if line.strip())
    except Exception:
        pass

    _LOW_CONF_LOG.write_text("", encoding="utf-8")
    return ClearLogResponse(
        message=f"Cleared {removed} log entries",
        entries_removed=removed,
    )


# ── API: Model Status ─────────────────────────────────────────────────────────

@router.get("/api/models/status", response_model=ModelStatusResponse)
async def models_status():
    status_items = _get_model_status()
    return ModelStatusResponse(
        models=status_items,
        models_loaded=clf.MODELS_LOADED,
        loaded_tiers=clf.get_loaded_tiers(),
    )


# ── API: Azure Sync ───────────────────────────────────────────────────────────

@router.post("/api/azure/download")
async def azure_download():
    """Download latest models from Azure Blob and hot-reload."""
    try:
        download_models_from_azure()
        global _last_azure_download
        _last_azure_download = datetime.now(timezone.utc).isoformat()

        # Hot-reload after download
        reload_results = clf.reload_models()
        logger.info("✅ Downloaded & hot-reloaded: %s", reload_results)
        return AzureActionResult(success=True, message="Models downloaded and reloaded")
    except Exception as e:
        logger.exception("Azure download failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/azure/upload")
async def azure_upload():
    """Upload current local models to Azure Blob."""
    try:
        result = upload_models_to_azure()
        if result.get("success"):
            global _last_azure_upload
            _last_azure_upload = datetime.now(timezone.utc).isoformat()
            return AzureActionResult(
                success=True,
                message=f"Uploaded {result.get('uploaded', 0)} files to Azure",
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Upload failed"))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Azure upload failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
