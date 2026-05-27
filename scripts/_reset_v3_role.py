"""Full v3 reset for a single role — wipes all v2_run_log rows so the
next /admin/trigger-v3 starts a clean Stage 0 cascade.

Use when a role got stuck mid-cascade (e.g. partial Stage 0/1/2 approved
but Stage 3/4/5/6/7/8 failed or never reached terminal) and the bulk
enrichment script needs to retry it from scratch.

The cleanup script `_cleanup_v3_failed_runs.py` only removes
failed/rejected rows. This script removes EVERYTHING for one specific
slug, including any approved intermediate stages that didn't cascade
to a complete catalog load (e.g. Flutter Developer with approved
Stages 0/1/2/3 but Stage 4 schema bug halted it).

Preserves:
  - The `roles` row itself
  - `role_aliases`, `role_dimensions`, `role_kras` (these only get
    populated by Stage 5/7/8 which never ran in the half-stuck case;
    if any are present, this script refuses to delete by default —
    use --force to override)
  - Any `dimensions` rows in the catalog (those are reused across
    roles; the next run will MERGE its dim_gen output against them)

Usage:
    uv run python scripts/_reset_v3_role.py <role_slug>
    uv run python scripts/_reset_v3_role.py <role_slug> --apply
    uv run python scripts/_reset_v3_role.py <role_slug> --apply --force
"""
from __future__ import annotations

import argparse
import os
import sys

import psycopg2
from dotenv import load_dotenv

load_dotenv()
SCHEMA = "skill-library"


def _connect():
    dsn = os.environ.get("PG_PROD") or os.environ.get("SKILL_LIBRARY_PG_DSN")
    if not dsn:
        print("[reset] no DSN", file=sys.stderr)
        sys.exit(2)
    return psycopg2.connect(dsn, options=f'-c search_path="{SCHEMA}",public')


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("role_slug", help="Slug of the role to reset (e.g. flutter-developer)")
    p.add_argument("--apply", action="store_true",
                   help="Execute the delete. Default is dry-run.")
    p.add_argument("--force", action="store_true",
                   help="Reset even if role already has role_dimensions/role_kras "
                        "(use only when you understand the impact — these will be "
                        "left orphaned and may need separate cleanup).")
    args = p.parse_args(argv)

    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                # Step 1: resolve role
                cur.execute(
                    f"SELECT id, display_name FROM \"{SCHEMA}\".roles WHERE slug = %s",
                    (args.role_slug,),
                )
                row = cur.fetchone()
                if row is None:
                    print(f"[reset] role slug {args.role_slug!r} not found")
                    return 2
                role_id, display = row
                print(f"Role: {display} (id={role_id}, slug={args.role_slug})")

                # Step 2: confirm catalog-level data is empty (unless --force)
                cur.execute(
                    f"SELECT COUNT(*) FROM \"{SCHEMA}\".role_dimensions WHERE role_id = %s",
                    (role_id,),
                )
                n_dims = cur.fetchone()[0]
                cur.execute(
                    f"SELECT COUNT(*) FROM \"{SCHEMA}\".role_kras WHERE role_id = %s",
                    (role_id,),
                )
                n_kras = cur.fetchone()[0]
                print(f"  role_dimensions: {n_dims}")
                print(f"  role_kras:       {n_kras}")

                if (n_dims or n_kras) and not args.force:
                    print(f"\n[reset] Role has catalog-level data already.")
                    print(f"  Refusing to delete v2_run_log rows because that would")
                    print(f"  orphan the role_dimensions / role_kras references.")
                    print(f"  Pass --force to override (and clean up manually after).")
                    return 1

                # Step 3: preview v2_run_log rows
                cur.execute(
                    f"""
                    SELECT run_id::text, prompt_version, status, started_at
                      FROM "{SCHEMA}".v2_run_log
                     WHERE role_id = %s
                     ORDER BY started_at ASC
                    """,
                    (args.role_slug,),
                )
                runs = cur.fetchall()
                print(f"\n  v2_run_log rows to delete: {len(runs)}")
                for r in runs:
                    print(f"    {r[0][:8]}  {r[1]:<28} {r[2]:<18} {r[3].strftime('%m-%d %H:%M')}")

                if not args.apply:
                    print("\n[reset] DRY-RUN. Pass --apply to delete.")
                    return 0

                # Step 4: delete v2_run_log rows (own transaction)
                cur.execute(
                    f"DELETE FROM \"{SCHEMA}\".v2_run_log WHERE role_id = %s",
                    (args.role_slug,),
                )
                deleted = cur.rowcount
                conn.commit()
                print(f"\n[reset] deleted {deleted} rows from v2_run_log.")

        # Step 5: separate transaction so v2_review_queue failures don't
        # roll back the v2_run_log delete above.
        with conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        f"DELETE FROM \"{SCHEMA}\".v2_review_queue "
                        f"WHERE run_id IN ("
                        f"  SELECT run_id FROM \"{SCHEMA}\".v2_run_log "
                        f"   WHERE role_id = %s)",
                        (args.role_slug,),
                    )
                    cleared = cur.rowcount
                    if cleared:
                        print(f"[reset] cleared {cleared} v2_review_queue rows.")
                    conn.commit()
                except (psycopg2.errors.UndefinedTable, psycopg2.errors.UndefinedColumn):
                    conn.rollback()
                    # v2_review_queue may not exist / have a different schema — ignore.
        print(f"\n[reset] {args.role_slug}: ready for fresh /admin/trigger-v3.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
