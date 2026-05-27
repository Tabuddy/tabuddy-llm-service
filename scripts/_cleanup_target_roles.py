"""One-shot catalog cleanup for the 7 client-facing target roles.

Phases (each runs only with --apply):
  A. Merge duplicate parents (id=418 -> 1 Backend, id=419 -> 15 Fullstack):
     - Repoint chosen_role_id on jd_pipeline_runs (none today, but safe).
     - Union aliases into the canonical row (skip dups).
     - Delete the duplicate row (cascade handles leftover FKs).
  B. Dedupe Web Developer child pairs (5 pairs of `<slug>-developer` vs
     `<slug>-dev`). Keep the `-dev` slug (matches §C worked examples in
     domain_role_classifier_prompt.md). Repoint any FK refs + union aliases.
  C. Insert Ionic Developer under Hybrid Mobile Developer (id=11).

Idempotent — safe to re-run. Default dry-run prints what would change.
"""
from __future__ import annotations

import argparse
import os
import sys

import psycopg2
from dotenv import load_dotenv

load_dotenv()
SCHEMA = "skill-library"

# (canonical_id, dup_id, label)
PARENT_MERGES: list[tuple[int, int, str]] = [
    (1, 418, "Backend Developer"),
    (15, 419, "Fullstack Developer"),
]

# (keep_slug, drop_slug)
WEBDEV_DEDUPE: list[tuple[str, str]] = [
    ("drupal-dev",    "drupal-developer"),
    ("wordpress-dev", "wordpress-developer"),
    ("magento-dev",   "magento-developer"),
    ("shopify-dev",   "shopify-developer"),
    ("sitecore-dev",  "sitecore-developer"),
]

IONIC_INSERT = {
    "slug": "ionic-developer",
    "display_name": "Ionic Developer",
    "domain": "Software Engineering",
    "parent_role_id": 11,  # Hybrid Mobile Developer
}


def _connect():
    dsn = os.environ.get("PG_PROD") or os.environ.get("SKILL_LIBRARY_PG_DSN")
    if not dsn:
        print("[cleanup] no DSN", file=sys.stderr)
        sys.exit(2)
    return psycopg2.connect(dsn, options=f'-c search_path="{SCHEMA}",public')


def merge_role(cur, canonical_id: int, dup_id: int, label: str, apply_: bool) -> dict:
    """Merge dup_id into canonical_id. Returns counts dict."""
    counts = {"aliases_moved": 0, "aliases_skipped_dup": 0, "fk_repointed": 0, "deleted": 0}
    # 1. Count + repoint jd_pipeline_runs FK refs
    cur.execute(
        f'SELECT COUNT(*) FROM "{SCHEMA}".jd_pipeline_runs WHERE chosen_role_id = %s',
        (dup_id,),
    )
    n_runs = cur.fetchone()[0]
    counts["fk_repointed"] = n_runs
    if apply_ and n_runs > 0:
        cur.execute(
            f'UPDATE "{SCHEMA}".jd_pipeline_runs SET chosen_role_id = %s '
            f"WHERE chosen_role_id = %s",
            (canonical_id, dup_id),
        )

    # 2. Move aliases not already present on canonical
    cur.execute(
        f"""
        SELECT alias_text FROM "{SCHEMA}".role_aliases
         WHERE role_id = %s
           AND LOWER(alias_text) NOT IN (
               SELECT LOWER(alias_text) FROM "{SCHEMA}".role_aliases WHERE role_id = %s
           )
        """,
        (dup_id, canonical_id),
    )
    to_move = [r[0] for r in cur.fetchall()]
    counts["aliases_moved"] = len(to_move)
    cur.execute(
        f'SELECT COUNT(*) FROM "{SCHEMA}".role_aliases WHERE role_id = %s',
        (dup_id,),
    )
    total_dup_aliases = cur.fetchone()[0]
    counts["aliases_skipped_dup"] = total_dup_aliases - len(to_move)

    if apply_ and to_move:
        cur.execute(
            f"""
            UPDATE "{SCHEMA}".role_aliases
               SET role_id = %s
             WHERE role_id = %s
               AND LOWER(alias_text) NOT IN (
                   SELECT LOWER(alias_text) FROM "{SCHEMA}".role_aliases WHERE role_id = %s
               )
            """,
            (canonical_id, dup_id, canonical_id),
        )

    # 3. Delete the duplicate row (CASCADE handles leftover role_aliases that
    # were already present on canonical — we drop them, not the canonical copy)
    if apply_:
        cur.execute(
            f'DELETE FROM "{SCHEMA}".role_aliases WHERE role_id = %s',
            (dup_id,),
        )
        cur.execute(f'DELETE FROM "{SCHEMA}".roles WHERE id = %s', (dup_id,))
        counts["deleted"] = 1
    return counts


def dedupe_webdev_pair(cur, keep_slug: str, drop_slug: str, apply_: bool) -> dict:
    counts = {"keep_id": None, "drop_id": None, "aliases_moved": 0, "fk_repointed": 0, "deleted": 0}
    cur.execute(
        f'SELECT id FROM "{SCHEMA}".roles WHERE slug = %s', (keep_slug,)
    )
    row = cur.fetchone()
    if not row:
        print(f"  [WARN] keep slug {keep_slug!r} not found, skipping pair")
        return counts
    keep_id = int(row[0])
    cur.execute(
        f'SELECT id FROM "{SCHEMA}".roles WHERE slug = %s', (drop_slug,)
    )
    row = cur.fetchone()
    if not row:
        return counts  # idempotent — nothing to drop
    drop_id = int(row[0])
    counts["keep_id"], counts["drop_id"] = keep_id, drop_id

    # Repoint jd_pipeline_runs
    cur.execute(
        f'SELECT COUNT(*) FROM "{SCHEMA}".jd_pipeline_runs WHERE chosen_role_id = %s',
        (drop_id,),
    )
    counts["fk_repointed"] = cur.fetchone()[0]
    if apply_ and counts["fk_repointed"] > 0:
        cur.execute(
            f'UPDATE "{SCHEMA}".jd_pipeline_runs SET chosen_role_id = %s WHERE chosen_role_id = %s',
            (keep_id, drop_id),
        )

    # Move aliases
    cur.execute(
        f"""
        SELECT alias_text FROM "{SCHEMA}".role_aliases
         WHERE role_id = %s
           AND LOWER(alias_text) NOT IN (
               SELECT LOWER(alias_text) FROM "{SCHEMA}".role_aliases WHERE role_id = %s
           )
        """,
        (drop_id, keep_id),
    )
    counts["aliases_moved"] = len(cur.fetchall())
    if apply_ and counts["aliases_moved"] > 0:
        cur.execute(
            f"""
            UPDATE "{SCHEMA}".role_aliases
               SET role_id = %s
             WHERE role_id = %s
               AND LOWER(alias_text) NOT IN (
                   SELECT LOWER(alias_text) FROM "{SCHEMA}".role_aliases WHERE role_id = %s
               )
            """,
            (keep_id, drop_id, keep_id),
        )

    # Delete drop side
    if apply_:
        cur.execute(
            f'DELETE FROM "{SCHEMA}".role_aliases WHERE role_id = %s', (drop_id,)
        )
        cur.execute(f'DELETE FROM "{SCHEMA}".roles WHERE id = %s', (drop_id,))
        counts["deleted"] = 1
    return counts


def insert_ionic(cur, apply_: bool) -> dict:
    counts = {"exists": False, "inserted": False, "id": None}
    cur.execute(
        f'SELECT id FROM "{SCHEMA}".roles WHERE slug = %s',
        (IONIC_INSERT["slug"],),
    )
    row = cur.fetchone()
    if row:
        counts["exists"] = True
        counts["id"] = int(row[0])
        return counts
    if apply_:
        cur.execute(
            f"""
            INSERT INTO "{SCHEMA}".roles
                (slug, display_name, domain, parent_role_id, is_branchable, source)
            VALUES (%s, %s, %s, %s, FALSE, 'MANUAL_CURATION'::"{SCHEMA}".entity_source)
            RETURNING id
            """,
            (
                IONIC_INSERT["slug"],
                IONIC_INSERT["display_name"],
                IONIC_INSERT["domain"],
                IONIC_INSERT["parent_role_id"],
            ),
        )
        counts["id"] = int(cur.fetchone()[0])
        counts["inserted"] = True
    return counts


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="Execute the changes. Default is dry-run.")
    args = p.parse_args(argv)

    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                print("=" * 60)
                print("Phase A: merge duplicate parents")
                print("=" * 60)
                for canonical_id, dup_id, label in PARENT_MERGES:
                    print(f"\n  {label}: {dup_id} -> {canonical_id}")
                    c = merge_role(cur, canonical_id, dup_id, label, args.apply)
                    print(f"    fk_repointed={c['fk_repointed']} "
                          f"aliases_moved={c['aliases_moved']} "
                          f"aliases_skipped(dup)={c['aliases_skipped_dup']} "
                          f"deleted={c['deleted']}")

                print()
                print("=" * 60)
                print("Phase B: dedupe Web Developer child pairs")
                print("=" * 60)
                for keep, drop in WEBDEV_DEDUPE:
                    print(f"\n  {keep} <- {drop}")
                    c = dedupe_webdev_pair(cur, keep, drop, args.apply)
                    print(f"    keep_id={c['keep_id']} drop_id={c['drop_id']} "
                          f"fk_repointed={c['fk_repointed']} "
                          f"aliases_moved={c['aliases_moved']} "
                          f"deleted={c['deleted']}")

                print()
                print("=" * 60)
                print("Phase C: insert Ionic Developer under Hybrid Mobile (id=11)")
                print("=" * 60)
                c = insert_ionic(cur, args.apply)
                if c["exists"]:
                    print(f"  already exists (id={c['id']}), skipping")
                else:
                    print(f"  {'inserted' if c['inserted'] else 'WOULD INSERT'} id={c['id']}")
        if args.apply:
            print("\n[cleanup] committed.")
        else:
            print("\n[cleanup] DRY-RUN — pass --apply to execute.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
