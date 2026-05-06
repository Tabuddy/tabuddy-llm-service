"""FastAPI router for the /generate-skills admin UI + JSON APIs.

Stage 0 of skill_library_v3 — admin types a role, the router upserts it,
spawns the charter run, and returns 202 with a run_id. The UI polls until
the run reaches awaiting_review, then the admin clicks Pass or Regenerate.

Mirrors :mod:`canonical_skill_api` for shape and async wrapping idiom: own
``Jinja2Templates`` instance, sync DB calls wrapped with ``asyncio.to_thread``.

Routes:
    GET  /generate-skills                                → HTML page
    GET  /api/generate-skills/roles                      → role catalog
    GET  /api/generate-skills/runs                       → recent runs (last 20)
    POST /api/generate-skills/runs                       → 202 + run_id, spawns task
    GET  /api/generate-skills/runs/{run_id}              → status + charter
    POST /api/generate-skills/runs/{run_id}/approve      → mark approved
    POST /api/generate-skills/runs/{run_id}/regenerate   → reject + spawn new run
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from skill_library_v3.db import repository as repo
from skill_library_v3.runner import (
    create_initial_run,
    regenerate_run,
    run_stage_0,
    run_stage_1,
    run_stage_1_regenerate,
    run_stage_2,
    run_stage_2_regenerate,
    run_stage_3,
    run_stage_3_regenerate,
    run_stage_4,
    run_stage_6,
    run_stage_6_regenerate,
    run_stage_7,
    run_stage_7_regenerate,
    run_stage_8,
)

logger = logging.getLogger(__name__)

generate_skills_router = APIRouter(tags=["generate-skills"])
_templates = Jinja2Templates(directory="templates")


# ── Request bodies ─────────────────────────────────────────────────────────


class CreateRunRequest(BaseModel):
    role_name: str = Field(min_length=2, max_length=160)


class ReviewBody(BaseModel):
    reviewed_by: str | None = Field(default=None, max_length=120)
    notes: str | None = Field(default=None, max_length=4000)


# ── HTML shell ─────────────────────────────────────────────────────────────


@generate_skills_router.get("/generate-skills", response_class=HTMLResponse)
async def generate_skills_page(request: Request):
    return _templates.TemplateResponse(request, "generate_skills.html", {})


# ── JSON APIs ──────────────────────────────────────────────────────────────


@generate_skills_router.get("/api/generate-skills/roles")
async def api_list_roles() -> dict[str, Any]:
    items = await asyncio.to_thread(repo.list_roles)
    return {"items": items, "total": len(items)}


@generate_skills_router.get("/api/generate-skills/runs")
async def api_list_recent_runs(limit: int = 20) -> dict[str, Any]:
    items = await asyncio.to_thread(repo.list_recent_runs, max(1, min(int(limit), 100)))
    return {"items": items, "total": len(items)}


@generate_skills_router.post("/api/generate-skills/runs", status_code=202)
async def api_create_run(body: CreateRunRequest) -> dict[str, Any]:
    try:
        result = await asyncio.to_thread(create_initial_run, body.role_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    run_id = result["run_id"]
    asyncio.create_task(run_stage_0(run_id))
    return {
        "run_id": str(run_id),
        "role_slug": result["role_slug"],
        "status": "pending",
    }


@generate_skills_router.get("/api/generate-skills/runs/{run_id}")
async def api_get_run(run_id: uuid.UUID) -> dict[str, Any]:
    detail = await asyncio.to_thread(repo.get_run_with_review, run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return detail


@generate_skills_router.post("/api/generate-skills/runs/{run_id}/approve")
async def api_approve_run(run_id: uuid.UUID, body: ReviewBody) -> dict[str, Any]:
    try:
        result = await asyncio.to_thread(
            repo.approve_run,
            run_id,
            reviewed_by=body.reviewed_by,
            notes=body.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Cascade chain:
    #   stage0_charter approve  -> fire run_stage_1 (role card)
    #   stage1_anchor  approve  -> fire run_stage_2 (dimensions, headless)
    #   stage3_recon   approve  -> fire run_stage_4 (skill typing, headless)
    #
    # Stage 2 has no approve surface (headless); Stage 3 is auto-fired
    # from inside run_stage_2's success path. Stage 4 is auto-fired from
    # run_stage_3's success path AND from here for the human-approve
    # path that flips an awaiting_review Stage 3 to approved.
    # Cascade chain on human-approve. Most stages auto-approve in the
    # autonomy-tightened policy and trigger the next stage from inside
    # the runner; this block handles the awaiting_review → approve path
    # where the admin clicked Pass.
    #   stage0_charter approve     -> fire run_stage_1 (role card)
    #   stage1_anchor  approve     -> fire run_stage_2 (dimensions)
    #   stage3_recon   approve     -> fire run_stage_4 (skill typing)
    #   stage6_containment approve -> fire run_stage_7 (enrichment)
    #   stage7_enrichment approve  -> fire run_stage_8 (catalog load)
    #   stage8_load approve        -> terminal (catalog populated)
    pv = (result.get("prompt_version") or "")
    if pv.startswith("stage0_charter_"):
        asyncio.create_task(run_stage_1(run_id))
    elif pv.startswith("stage1_anchor_"):
        asyncio.create_task(run_stage_2(result["role_slug"]))
    elif pv.startswith("stage3_recon_"):
        asyncio.create_task(run_stage_4(result["role_slug"]))
    elif pv.startswith("stage6_containment_"):
        asyncio.create_task(run_stage_7(result["role_slug"]))
    elif pv.startswith("stage7_enrichment_"):
        asyncio.create_task(run_stage_8(result["role_slug"]))
    return result


@generate_skills_router.post("/api/generate-skills/runs/{run_id}/regenerate")
async def api_regenerate_run(run_id: uuid.UUID, body: ReviewBody) -> dict[str, Any]:
    try:
        result = await asyncio.to_thread(
            regenerate_run,
            run_id,
            reviewed_by=body.reviewed_by,
            notes=body.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    new_id = result["run_id"]
    prompt_version = result.get("prompt_version") or ""
    if prompt_version.startswith("stage1_anchor_"):
        asyncio.create_task(run_stage_1_regenerate(new_id))
    elif prompt_version.startswith("stage2_dim_gen_"):
        asyncio.create_task(run_stage_2_regenerate(new_id))
    elif prompt_version.startswith("stage3_recon_"):
        asyncio.create_task(run_stage_3_regenerate(new_id))
    elif prompt_version.startswith("stage6_containment_"):
        asyncio.create_task(run_stage_6_regenerate(new_id))
    elif prompt_version.startswith("stage7_enrichment_"):
        asyncio.create_task(run_stage_7_regenerate(new_id))
    elif prompt_version.startswith("stage8_load_"):
        asyncio.create_task(run_stage_8(result["role_slug"]))
    else:
        asyncio.create_task(run_stage_0(new_id))
    return {"run_id": str(new_id), "status": "pending", "prompt_version": prompt_version}
