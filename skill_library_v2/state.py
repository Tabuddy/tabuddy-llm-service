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

from operator import add
from typing import Annotated, Any, Literal, TypedDict

from skill_library_v2.schemas.category import CategorizerOutput
from skill_library_v2.schemas.role import DimensionSlice, PageExtract, WebHint
from skill_library_v2.schemas.skill import Correction, SkillDraft


# ─── Reducers ───────────────────────────────────────────────────────────────
# LangGraph applies these when multiple parallel Sends write to the same
# state key. Right-hand side wins on per-key conflict; this matches the
# "each dim worker owns its own dim_id entry" mental model.

def merge_dim_lists(
    left: dict[str, list[SkillDraft]] | None,
    right: dict[str, list[SkillDraft]] | None,
) -> dict[str, list[SkillDraft]]:
    merged: dict[str, list[SkillDraft]] = dict(left or {})
    for k, v in (right or {}).items():
        merged[k] = v
    return merged


def merge_by_key(left: dict | None, right: dict | None) -> dict:
    merged: dict = dict(left or {})
    if right:
        merged.update(right)
    return merged


def merge_correction_lists(
    left: dict[str, list[Correction]] | None,
    right: dict[str, list[Correction]] | None,
) -> dict[str, list[Correction]]:
    merged: dict[str, list[Correction]] = dict(left or {})
    for k, v in (right or {}).items():
        merged[k] = v
    return merged


DimStatus = Literal["pending", "generating", "criticizing", "done", "failed"]


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
    planner_page_extracts: list[PageExtract]

    # ── Filled by Generator/Critic/Validator (Phase 2) ──
    # All three dicts are keyed by dimension_id. The reducers merge
    # per-key writes from parallel Send workers.
    generated: Annotated[dict[str, list[SkillDraft]], merge_dim_lists]
    dim_status: Annotated[dict[str, DimStatus], merge_by_key]
    dim_categories: Annotated[dict[str, CategorizerOutput], merge_by_key]
    dim_retry_count: Annotated[dict[str, int], merge_by_key]
    dim_critic_corrections: Annotated[
        dict[str, list[Correction]], merge_correction_lists
    ]
    dim_critic_verdict: Annotated[
        dict[str, Literal["approve", "revise", "reject"]], merge_by_key
    ]
    dim_generator_notes: Annotated[dict[str, str], merge_by_key]

    # ── Reserved for later phases (retrieval/grounding not used in v1 loop) ──
    retrieval_packs: dict[str, Any]
    review_queue: Annotated[list[ReviewItem], add]
    errors: Annotated[list[ErrorRecord], add]
