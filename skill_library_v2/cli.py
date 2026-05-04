"""Single-role pipeline runner with a consolidated human-readable summary.

This is the one-command way to exercise the full pipeline end-to-end
(Planner + per-dimension Generator + Critic + deterministic Validator)
for a single role.

Usage::

    uv run python -m skill_library_v2.cli --role "Backend Engineer"
    uv run python -m skill_library_v2.cli --role "Data Scientist" --format json
    uv run python -m skill_library_v2.cli --role "SRE" --save evals/runs/my_run/
    uv run python -m skill_library_v2.cli --role "Backend Engineer" --persist

By default:
  - Prints a consolidated text summary to stdout (role, archetype, each
    dimension with its skills, per-dim status + critic verdict, and a
    footer with totals).
  - Skips DB persistence (``--persist`` opts in; needs PG_DSN + schema
    applied).
  - Does not save JSON to disk (``--save DIR`` opts in).

Pass ``--format json`` to print the full raw output dict instead of the
summary (useful for piping into ``jq`` or feeding the eval scorer).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
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


# ─── Result-dict builder ────────────────────────────────────────────────────

def _build_result(
    *,
    run_id: UUID,
    role: RoleInput,
    final_state: PlanGraphState,
) -> dict:
    dims = final_state.get("dimensions") or []
    review_queue = final_state.get("review_queue") or []
    planner_output = {
        "role_archetype": final_state.get("role_archetype"),
        "dimensions": [d.model_dump() for d in dims],
        "reasoning": final_state.get("planner_reasoning"),
        "flagged_for_review": [
            item["payload"]["dimension_id"]
            for item in review_queue
            if item.get("item_type") == "dimension_flagged"
        ],
    }

    generated = final_state.get("generated") or {}
    dim_status = final_state.get("dim_status") or {}
    dim_retry_count = final_state.get("dim_retry_count") or {}
    dim_critic_verdict = final_state.get("dim_critic_verdict") or {}
    dim_critic_corrections = final_state.get("dim_critic_corrections") or {}
    dim_generator_notes = final_state.get("dim_generator_notes") or {}

    generator_output_by_dim: dict[str, dict] = {}
    for dim_id, skill_list in generated.items():
        generator_output_by_dim[dim_id] = {
            "skills": [s.model_dump() for s in skill_list],
            "skill_count": len(skill_list),
            "status": dim_status.get(dim_id),
            "retry_count": dim_retry_count.get(dim_id, 0),
            "critic_verdict": dim_critic_verdict.get(dim_id),
            "critic_corrections": [
                (c.model_dump() if hasattr(c, "model_dump") else c)
                for c in (dim_critic_corrections.get(dim_id) or [])
            ],
            "generator_notes": dim_generator_notes.get(dim_id, ""),
        }

    return {
        "run_id": str(run_id),
        "role_id": role.role_id,
        "role_display": role.role_display,
        "prompt_version": final_state.get("prompt_version"),
        "model_snapshot": final_state.get("model_snapshot"),
        "planner_output": planner_output,
        "generator_output_by_dim": generator_output_by_dim,
        "web_hints": [
            h.model_dump() for h in final_state.get("planner_web_hints") or []
        ],
        "page_extracts": [
            e.model_dump() for e in final_state.get("planner_page_extracts") or []
        ],
        "errors": final_state.get("errors") or [],
    }


# ─── Human-readable consolidated summary ────────────────────────────────────

def _format_summary(result: dict) -> str:
    lines: list[str] = []
    role_display = result["role_display"]
    role_id = result["role_id"]
    run_id = result["run_id"]
    prompt_version = result.get("prompt_version") or "?"
    model_snapshot = result.get("model_snapshot") or "?"

    po = result["planner_output"]
    archetype = po.get("role_archetype") or "(not set)"
    dims = po.get("dimensions") or []
    flagged = po.get("flagged_for_review") or []

    gen_by_dim: dict[str, dict] = result.get("generator_output_by_dim") or {}
    errors = result.get("errors") or []

    # ── Header ──
    lines.append("=" * 78)
    lines.append(f"ROLE: {role_display}   (role_id: {role_id})")
    lines.append(f"RUN:  {run_id}")
    lines.append(f"PROMPT_VERSION: {prompt_version}    MODEL: {model_snapshot}")
    lines.append("=" * 78)

    # ── Planner summary ──
    lines.append("")
    lines.append(f"PLANNER   archetype: {archetype}")
    lines.append(f"          dimensions: {len(dims)}" +
                 (f"    flagged: {flagged}" if flagged else ""))
    lines.append("")

    # ── Per-dim breakdown ──
    passed = 0
    failed = 0
    total_skills = 0

    for i, dim in enumerate(dims, start=1):
        dim_id = dim["dimension_id"]
        dim_name = dim["dimension_name"]
        expected_cats = ", ".join(dim.get("expected_skill_kinds") or [])
        difficulty = dim.get("difficulty_hint", "?")

        gen = gen_by_dim.get(dim_id)
        if gen is None:
            # Generator hasn't run or produced nothing for this dim.
            status_tag = "— no generator output —"
            verdict = "?"
            retries = 0
            skill_count = 0
            skills = []
            corrections = []
            notes = ""
        else:
            status = gen.get("status") or "?"
            verdict = gen.get("critic_verdict") or "?"
            retries = gen.get("retry_count", 0)
            skill_count = gen.get("skill_count", 0)
            skills = gen.get("skills") or []
            corrections = gen.get("critic_corrections") or []
            notes = gen.get("generator_notes") or ""
            if status == "done":
                status_tag = "[PASS]"
                passed += 1
            elif status == "failed":
                status_tag = "[FAIL]"
                failed += 1
            else:
                status_tag = f"[{status}]"
            total_skills += skill_count

        lines.append("─" * 78)
        lines.append(
            f"[{i:>2}/{len(dims)}] {dim_name}   ({dim_id})"
        )
        lines.append(
            f"        category-hints: {expected_cats or '(none)'}"
            f"    difficulty: {difficulty}"
        )
        lines.append(
            f"        {status_tag}  skills={skill_count}  "
            f"critic={verdict}  retries={retries}"
        )

        if skills:
            for s in skills:
                aliases = s.get("aliases") or []
                alias_str = f"  aliases={aliases}" if aliases else ""
                cat = s.get("category", "?")
                nature = s.get("skill_nature", "?")
                lines.append(
                    f"           - {s['slug']:<36s} "
                    f"\"{s.get('display_name','?')}\"  "
                    f"[{cat}/{nature}]{alias_str}"
                )

        if corrections:
            lines.append("        critic-corrections:")
            for c in corrections:
                lines.append(
                    f"           · {c.get('slug','?')}  [{c.get('issue_type','?')}]  "
                    f"{c.get('suggested_fix','?')}"
                )

        if notes and notes.strip():
            lines.append(f"        generator-notes: {notes.strip()[:200]}")

    # ── Footer ──
    lines.append("=" * 78)
    lines.append(
        f"SUMMARY: {passed}/{len(dims)} dims PASS, {failed} FAIL, "
        f"{total_skills} total skills"
    )
    if errors:
        lines.append(f"ERRORS: {len(errors)}")
        for e in errors[:5]:
            lines.append(f"   - [{e.get('node','?')}] {e.get('message','?')[:150]}")
    lines.append("=" * 78)

    return "\n".join(lines)


# ─── Main runner ────────────────────────────────────────────────────────────

async def _run(
    role_display: str,
    role_id: str | None,
    persist: bool,
    fmt: str,
    save_dir: Path | None,
) -> int:
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

    result = _build_result(run_id=run_id, role=role, final_state=final_state)

    # ── Print ──
    if fmt == "json":
        print(json.dumps(result, indent=2, default=_json_default, ensure_ascii=False))
    else:
        print(_format_summary(result))

    # ── Save ──
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{role.role_id}.json"
        out_path.write_text(
            json.dumps(result, indent=2, default=_json_default, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Wrote %s", out_path)

    # ── Persist (optional DB write) ──
    if persist:
        try:
            from skill_library_v2.db.connection import (
                enqueue_review,
                mark_run_complete,
                record_run,
            )
            review_queue = final_state.get("review_queue") or []
            await record_run(
                run_id=run_id,
                role_id=role.role_id,
                role_display=role.role_display,
                prompt_version=final_state.get("prompt_version") or "unknown",
                model_snapshot=final_state.get("model_snapshot") or "unknown",
                planner_output=result["planner_output"],
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


async def _main_async(
    role_display: str,
    role_id: str | None,
    persist: bool,
    fmt: str,
    save_dir: Path | None,
) -> int:
    try:
        return await _run(
            role_display=role_display,
            role_id=role_id,
            persist=persist,
            fmt=fmt,
            save_dir=save_dir,
        )
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
        description=(
            "Run the full skill_library_v2 pipeline (Planner + Generator + "
            "Critic + Validator) for a single role and print a consolidated "
            "summary."
        ),
    )
    parser.add_argument(
        "--role",
        required=True,
        help='Role display name, e.g. "Backend Engineer".',
    )
    parser.add_argument(
        "--role-id",
        default=None,
        help="Optional role slug; defaults to a slug of --role.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help=(
            "Output format. 'text' (default) prints a human-readable "
            "consolidated summary; 'json' prints the full raw result dict."
        ),
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Also save the full result as <DIR>/<role_id>.json. Useful for "
            "piping into the eval scorer."
        ),
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help=(
            "Write the Planner output to v2_run_log (requires PG_DSN + "
            "schema). Default: off — runs are ephemeral."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug-level logging.",
    )
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    return asyncio.run(
        _main_async(
            role_display=args.role,
            role_id=args.role_id,
            persist=args.persist,
            fmt=args.format,
            save_dir=args.save,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
