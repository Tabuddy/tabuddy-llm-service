"""Cleanup failed / rejected v3 runs for the 7 target families so the bulk
enrichment can start clean.

Deletes rows from `v2_run_log` where:
  - role_id is one of the 7 target slugs OR a child of one
  - status is in ('failed', 'rejected')
  - the role's `role_kras` is still empty (we don't want to drop history
    of a role that already enriched on a later attempt)

Idempotent. Default --dry-run, --apply to execute.

Usage:
    uv run python scripts/_cleanup_v3_failed_runs.py            # dry-run
    uv run python scripts/_cleanup_v3_failed_runs.py --apply    # delete
"""
from __future__ import annotations

import argparse
import os
import sys

import psycopg2
from dotenv import load_dotenv

load_dotenv()
SCHEMA = "skill-library"

TARGET_PARENT_IDS = [1, 2, 7, 11, 15, 25, 75]


def _connect():
    dsn = os.environ.get("PG_PROD") or os.environ.get("SKILL_LIBRARY_PG_DSN")
    if not dsn:
        print("[cleanup] no DSN", file=sys.stderr)
        sys.exit(2)
    return psycopg2.connect(dsn, options=f'-c search_path="{SCHEMA}",public')


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="Execute the delete. Default is dry-run.")
    args = p.parse_args(argv)

    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                # Build the set of target slugs from the target parent IDs + their children
                cur.execute(
                    f"""
                    SELECT r.slug,
                           (SELECT COUNT(*) FROM "{SCHEMA}".role_kras WHERE role_id = r.id) AS n_kras
                      FROM "{SCHEMA}".roles r
                     WHERE r.id = ANY(%s) OR r.parent_role_id = ANY(%s)
                    """,
                    (TARGET_PARENT_IDS, TARGET_PARENT_IDS),
                )
                target_slugs_shell = [s for s, n in cur.fetchall() if n < 3]
                print(f"[cleanup] {len(target_slugs_shell)} shell target slugs in scope")

                if not target_slugs_shell:
                    print("[cleanup] no shell targets — nothing to clean.")
                    return 0

                # Preview rows that would be deleted
                cur.execute(
                    f"""
                    SELECT role_id, prompt_version, status, started_at, error_message
                      FROM "{SCHEMA}".v2_run_log
                     WHERE role_id = ANY(%s)
                       AND LOWER(status) IN ('failed','rejected')
                     ORDER BY role_id, started_at DESC
                    """,
                    (target_slugs_shell,),
                )
                rows = cur.fetchall()
                print(f"[cleanup] found {len(rows)} failed/rejected rows to delete:")
                for role_id, pv, status, started, err in rows:
                    err_short = (err or "").split("\n")[0][:80]
                    print(f"  {role_id:<32} {pv:<28} {status:<10} {started.strftime('%H:%M:%S')}  {err_short}")

                if not args.apply:
                    print("\n[cleanup] DRY-RUN. Pass --apply to delete.")
                    return 0

                # Execute the delete
                cur.execute(
                    f"""
                    DELETE FROM "{SCHEMA}".v2_run_log
                     WHERE role_id = ANY(%s)
                       AND LOWER(status) IN ('failed','rejected')
                    """,
                    (target_slugs_shell,),
                )
                deleted = cur.rowcount
                conn.commit()
                print(f"\n[cleanup] deleted {deleted} rows.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
