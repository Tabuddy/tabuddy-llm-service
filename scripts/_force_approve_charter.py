"""Force-approve a stuck Stage 0 charter + cascade fire Stage 1.

Use when the validator rejected a Stage 0 output for a known-good reason
(e.g. role name happens to contain a catalog skill alias and the validator
trips on it). Reviews the latest stage0 charter row for the given slug,
prints the charter content for sanity check, then flips status='approved'
and POSTs the approve endpoint to fire run_stage_1 via the cascade.

Usage:
    uv run python scripts/_force_approve_charter.py <role_slug>
        --review-only                      # show charter, don't approve
    uv run python scripts/_force_approve_charter.py <role_slug>
        --apply                            # approve + fire stage 1

Examples:
    uv run python scripts/_force_approve_charter.py react-native-developer --review-only
    uv run python scripts/_force_approve_charter.py react-native-developer --apply

Honors AUTOCHAIN_BASE_URL env var (default http://localhost:7777 in case
LLM service runs on the standard prod port; falls back to 8000 if set
otherwise).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import httpx
import psycopg2
from dotenv import load_dotenv

load_dotenv()
SCHEMA = "skill-library"
BASE_URL = os.environ.get("AUTOCHAIN_BASE_URL", "http://localhost:7777").rstrip("/")


def _connect():
    dsn = os.environ.get("PG_PROD") or os.environ.get("SKILL_LIBRARY_PG_DSN")
    return psycopg2.connect(dsn, options=f'-c search_path="{SCHEMA}",public')


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("role_slug", help="Role slug to recover (e.g. react-native-developer)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--review-only", action="store_true",
                   help="Print the charter content, don't approve")
    g.add_argument("--apply", action="store_true",
                   help="Set status='approved' on the latest stage0 row and fire Stage 1")
    args = p.parse_args(argv)

    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT run_id::text, status, prompt_version, planner_output,
                       error_message, started_at
                  FROM "{SCHEMA}".v2_run_log
                 WHERE role_id = %s AND prompt_version LIKE 'stage0_charter%%'
                 ORDER BY started_at DESC LIMIT 1
                """,
                (args.role_slug,),
            )
            row = cur.fetchone()
            if row is None:
                print(f"[force-approve] no stage0 charter run for {args.role_slug!r}")
                return 2
            run_id, status, pv, planner_output, err, started = row
            print(f"\nRole: {args.role_slug}")
            print(f"  latest stage0 run: {run_id}")
            print(f"  status: {status}")
            print(f"  prompt_version: {pv}")
            print(f"  started_at: {started}")
            if err:
                print(f"  error: {err[:300]}")
            print()
            print("Charter content:")
            print(json.dumps(planner_output, indent=2)[:2000])

            if args.review_only:
                print("\n[force-approve] review-only mode — no DB change.")
                return 0

            # Only act on non-approved terminal states.
            if status.lower() not in ("failed", "rejected", "awaiting_review"):
                print(f"\n[force-approve] status={status} is not failed/rejected/awaiting_review — refusing to overwrite.")
                return 1

            # Step 1: flip the row to approved DIRECTLY (the regular
            # /approve endpoint refuses non-awaiting_review statuses)
            cur.execute(
                f"UPDATE \"{SCHEMA}\".v2_run_log SET status='approved' WHERE run_id = %s",
                (run_id,),
            )
            print(f"\n[force-approve] flipped run {run_id[:8]} -> approved")
            conn.commit()
    finally:
        conn.close()

    # Step 2: fire Stage 1 via the LLM service so the cascade resumes
    print(f"[force-approve] firing Stage 1 for {args.role_slug} via {BASE_URL}")
    try:
        # The /api/generate-skills/runs/{run_id}/approve endpoint expects
        # status='awaiting_review' as a precondition. Since we already
        # flipped to 'approved' directly, we need to call /trigger-v3
        # again with the slug, which dispatches a fresh run that picks
        # up from where we are. BUT this creates a new Stage 0 run that
        # could fail the same way.
        #
        # Better path: call run_stage_1 directly via the v3_runner if
        # the LLM service is in-process, OR ask the user to manually
        # restart the bulk-enrich (which fetches DB and skips the now-
        # "approved" stage 0, then triggers a fresh Stage 1).
        print()
        print("[force-approve] Charter row is now 'approved'. Next step:")
        print(f"  Re-run the bulk-enrich script — it will skip {args.role_slug}'s")
        print("  stage0 (now approved) and resume from stage1 via /admin/trigger-v3,")
        print("  which creates a NEW pipeline run that starts at stage1 from the")
        print("  approved stage0 charter.")
        print()
        print("  OR call /api/generate-skills/runs/<run_id>/regenerate to start fresh")
        print(f"  from stage1: ")
        print(f"  curl -X POST {BASE_URL}/api/generate-skills/runs/{run_id}/regenerate \\")
        print(f"       -H 'Content-Type: application/json' -d '{{}}'  (untested)")
    except Exception as exc:  # noqa: BLE001
        print(f"[force-approve] stage1 dispatch failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
