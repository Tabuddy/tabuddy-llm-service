"""Manual runner for the Phase 1 pipeline.

Usage:

    uv run python -m skill_library_v2.cli --role "Backend Engineer"
    uv run python -m skill_library_v2.cli --role "Data Scientist" --role-id data_scientist
    uv run python -m skill_library_v2.cli --role "SRE" --no-persist

The CLI builds the LangGraph, seeds :class:`PlanGraphState` with the role
and a fresh ``run_id``, invokes the graph, prints the :class:`PlannerOutput`
pretty-printed, and (unless ``--no-persist``) writes to ``v2_run_log`` and
``v2_review_queue``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from uuid import UUID, uuid4

from skill_library_v2.graph.builder import build_graph
from skill_library_v2.schemas.role import RoleInput
from skill_library_v2.state import PlanGraphState
from skill_library_v2.tools.web_search import close_clients as close_web_clients

logger = logging.getLogger("skill_library_v2.cli")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    return _SLUG_RE.sub("_", text.strip().lower()).strip("_")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s : %(message)s",
    )


def _json_default(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


async def _run(role_display: str, role_id: str | None, persist: bool) -> int:
    role = RoleInput(
        role_id=role_id or _slugify(role_display),
        role_display=role_display,
    )
    run_id = uuid4()

    initial_state: PlanGraphState = {
        "run_id": str(run_id),
        "role_id": role.role_id,
        "role_display": role.role_display,
    }

    graph = build_graph()
    logger.info("Starting graph run %s for role %r", run_id, role.role_display)

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Graph run failed: %s", exc)
        if persist:
            try:
                from skill_library_v2.db.connection import mark_run_complete
                await mark_run_complete(run_id, status="failed", error_message=str(exc))
            except Exception:
                logger.exception("Also failed to mark run as failed in v2_run_log.")
        return 1

    dimensions = final_state.get("dimensions") or []
    review_queue = final_state.get("review_queue") or []

    planner_output = {
        "role_archetype": final_state.get("role_archetype"),
        "dimensions": [d.model_dump() for d in dimensions],
        "reasoning": final_state.get("planner_reasoning"),
        "flagged_for_review": [
            item["payload"]["dimension_id"]
            for item in review_queue
            if item.get("item_type") == "dimension_flagged"
        ],
    }

    print(json.dumps(
        {
            "run_id": str(run_id),
            "role_id": role.role_id,
            "role_display": role.role_display,
            "prompt_version": final_state.get("prompt_version"),
            "model_snapshot": final_state.get("model_snapshot"),
            "planner_output": planner_output,
            "web_hints": [h.model_dump() for h in final_state.get("planner_web_hints") or []],
        },
        indent=2,
        default=_json_default,
    ))

    if persist:
        try:
            from skill_library_v2.db.connection import (
                enqueue_review,
                mark_run_complete,
                record_run,
            )
            await record_run(
                run_id=run_id,
                role_id=role.role_id,
                role_display=role.role_display,
                prompt_version=final_state.get("prompt_version") or "unknown",
                model_snapshot=final_state.get("model_snapshot") or "unknown",
                planner_output=planner_output,
                planner_reasoning=final_state.get("planner_reasoning") or "",
                status="planned",
            )
            if review_queue:
                n = await enqueue_review(run_id, review_queue)
                logger.info("Enqueued %d review items.", n)
            await mark_run_complete(run_id, status="complete")
            logger.info("Persisted run %s to v2_run_log.", run_id)
        except Exception as exc:
            logger.exception("Persistence failed: %s", exc)
            return 2

    return 0


async def _main_async(role_display: str, role_id: str | None, persist: bool) -> int:
    try:
        return await _run(role_display=role_display, role_id=role_id, persist=persist)
    finally:
        # Release event-loop-bound resources before the loop closes.
        try:
            await close_web_clients()
        except Exception:
            logger.debug("close_web_clients failed", exc_info=True)
        if persist:
            try:
                from skill_library_v2.db.connection import close_pool
                await close_pool()
            except Exception:
                logger.debug("close_pool failed", exc_info=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m skill_library_v2.cli",
        description="Run the Phase 1 skill_library_v2 pipeline for a single role.",
    )
    parser.add_argument("--role", required=True, help='Role display name, e.g. "Backend Engineer".')
    parser.add_argument("--role-id", default=None, help="Optional role slug; defaults to a slug of --role.")
    parser.add_argument("--no-persist", action="store_true", help="Skip Postgres writes.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging.")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    return asyncio.run(
        _main_async(
            role_display=args.role,
            role_id=args.role_id,
            persist=not args.no_persist,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
