"""LangGraph `StateGraph` construction.

Phase 2 topology (current):

    __start__ → plan_role ──┐
                            │  (conditional, Send API per-dim)
                            ▼
                      process_dim  (one worker per dim; generator →
                            │      critic → validator internally, ≤3 retries)
                            ▼
                      join_dimensions → end_phase_2 → END

The Planner produces dimensions; ``_fan_out`` issues one ``Send`` per
dimension into ``process_dim``; that worker runs the full v1 loop
(Generator → Critic → deterministic Validator) internally with bounded
retries, then writes a single state patch for its dim_id.
``join_dimensions`` is the fan-in barrier; ``end_phase_2`` logs the
summary.

The retry loop lives inside ``process_dim`` rather than as graph edges
because LangGraph's Send API threads payload data only to the *first*
node in a chain — subsequent nodes see global state without the
per-worker ``dim_id``. Keeping the loop in-node sidesteps this cleanly.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from skill_library_v2.agents.dim_worker import process_dim
from skill_library_v2.agents.planner import plan_role
from skill_library_v2.schemas.role import DimensionSlice
from skill_library_v2.state import PlanGraphState

logger = logging.getLogger(__name__)


def _fan_out(state: PlanGraphState) -> list[Send]:
    """Emit one ``Send`` per dimension into the generate_skills worker.

    The payload carries the per-dim context (``dim_id``, ``dim``) plus the
    role-level fields each worker needs — LangGraph *merges* the Send
    payload with the existing state, so parent-state keys remain visible.
    """
    role_id = state.get("role_id") or ""
    role_display = state.get("role_display") or ""
    role_archetype = state.get("role_archetype") or ""
    dims_raw = state.get("dimensions") or []

    # Normalize dims into dicts so the payload serializes cleanly.
    dim_dicts: list[dict] = []
    for d in dims_raw:
        if isinstance(d, DimensionSlice):
            dim_dicts.append(d.model_dump())
        elif isinstance(d, dict):
            dim_dicts.append(d)
        else:
            # Pydantic v1-style .dict() fallback
            dim_dicts.append(d.dict() if hasattr(d, "dict") else d)

    role_hints = state.get("planner_web_hints") or []
    sends: list[Send] = []
    for d in dim_dicts:
        sends.append(Send(
            "process_dim",
            {
                "dim_id": d["dimension_id"],
                "dim": d,
                # Carry the role-level context the worker reads.
                "role_id": role_id,
                "role_display": role_display,
                "role_archetype": role_archetype,
                "dimensions": dim_dicts,
                "planner_web_hints": role_hints,
            },
        ))
    logger.info("[fan_out] emitting %d Send(s) for role %s", len(sends), role_id)
    return sends


async def _join_dimensions(state: PlanGraphState) -> dict[str, Any]:
    """Barrier — runs once after all per-dim Sends have settled."""
    generated = state.get("generated") or {}
    statuses = state.get("dim_status") or {}
    done = sum(1 for v in statuses.values() if v == "done")
    failed = sum(1 for v in statuses.values() if v == "failed")
    total_skills = sum(len(v) for v in generated.values())
    logger.info(
        "[join_dimensions] role=%s: %d done / %d failed / %d dims; %d total skills",
        state.get("role_id"),
        done, failed, len(statuses), total_skills,
    )
    return {}


async def _end_phase_2(state: PlanGraphState) -> dict[str, Any]:
    """Terminal stub for Phase 2 — logs a summary and returns empty patch."""
    dims = state.get("dimensions") or []
    generated = state.get("generated") or {}
    statuses = state.get("dim_status") or {}
    logger.info(
        "[end_phase_2] run %s role %s: %d dims planned, %d generated, "
        "%d failed, %d flagged.",
        state.get("run_id"),
        state.get("role_id"),
        len(dims),
        sum(1 for v in statuses.values() if v == "done"),
        sum(1 for v in statuses.values() if v == "failed"),
        len(state.get("review_queue") or []),
    )
    return {}


def build_graph():
    """Compile the Phase 2 LangGraph StateGraph."""
    g: StateGraph = StateGraph(PlanGraphState)

    g.add_node("plan_role", plan_role)
    g.add_node("process_dim", process_dim)
    g.add_node("join_dimensions", _join_dimensions)
    g.add_node("end_phase_2", _end_phase_2)

    g.set_entry_point("plan_role")

    # plan_role -> per-dim fan-out via Send API
    g.add_conditional_edges("plan_role", _fan_out, ["process_dim"])

    g.add_edge("process_dim", "join_dimensions")
    g.add_edge("join_dimensions", "end_phase_2")
    g.add_edge("end_phase_2", END)

    return g.compile()
