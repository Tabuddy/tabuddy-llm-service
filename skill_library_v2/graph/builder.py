"""LangGraph `StateGraph` construction.

Phase 1 topology:

    __start__ → plan_role → end_phase_1 → END

``end_phase_1`` is a stub that simply logs the Planner output; Phase 2 will
replace it with ``fan_out_dimensions`` → per-dimension
(``dimension_router`` → ``retrieval_service`` → ``generator`` →
``evidence_validators`` → ``critic_stack``) as described in
tabuddy_v2_architecture.md §13.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from skill_library_v2.agents.planner import plan_role
from skill_library_v2.state import PlanGraphState

logger = logging.getLogger(__name__)


async def _end_phase_1(state: PlanGraphState) -> dict[str, Any]:
    """Terminal stub for Phase 1.

    Logs a summary so CLI runs print useful output even before persistence
    is wired in by the caller. Returns an empty patch.
    """
    dims = state.get("dimensions") or []
    logger.info(
        "[end_phase_1] run %s for role %s: %d dimensions (%d flagged).",
        state.get("run_id"),
        state.get("role_id"),
        len(dims),
        len(state.get("review_queue") or []),
    )
    return {}


def build_graph():
    """Compile the Phase 1 LangGraph StateGraph."""
    g: StateGraph = StateGraph(PlanGraphState)
    g.add_node("plan_role", plan_role)
    g.add_node("end_phase_1", _end_phase_1)

    g.set_entry_point("plan_role")
    g.add_edge("plan_role", "end_phase_1")
    g.add_edge("end_phase_1", END)

    return g.compile()
