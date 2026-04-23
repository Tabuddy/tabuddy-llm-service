"""LangGraph state shape passed between nodes.

Fields are **append-only** — each node adds what it produces but never
overwrites earlier fields. This preserves the full provenance trail per
tabuddy_v2_architecture.md §2 and §13.

Phase 1 uses `role_id`, `role_display`, `run_id`, `prompt_version`,
`model_snapshot`, `dimensions`, `planner_reasoning`, and `planner_web_hints`.
Later-phase fields are declared here so the shape is stable when Generator,
Critics, and Edge pipeline drop in.
"""

from __future__ import annotations

from typing import Any, TypedDict

from skill_library_v2.schemas.role import DimensionSlice, WebHint
from skill_library_v2.schemas.skill import SkillDraft


class ErrorRecord(TypedDict, total=False):
    node: str
    message: str
    retryable: bool


class ReviewItem(TypedDict, total=False):
    item_type: str       # 'dimension_flagged' | 'skill_proposed_new' | ...
    payload: dict[str, Any]
    reason: str


class PlanGraphState(TypedDict, total=False):
    # ── Seeded at entry ──
    role_id: str
    role_display: str
    run_id: str
    prompt_version: str
    model_snapshot: str

    # ── Filled by Planner (Phase 1) ──
    role_archetype: str
    dimensions: list[DimensionSlice]
    planner_reasoning: str
    planner_web_hints: list[WebHint]

    # ── Filled by later phases (declared for shape stability) ──
    retrieval_packs: dict[str, Any]       # keyed by dimension_id
    generated: dict[str, list[SkillDraft]]
    review_queue: list[ReviewItem]
    errors: list[ErrorRecord]
