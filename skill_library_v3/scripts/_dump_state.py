"""Quick DB sanity dump — used as a smoke test, not part of the runtime."""
from __future__ import annotations

from skill_library_v3.db.connection import connect


def main() -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, slug, display_name, role_archetype FROM roles ORDER BY id")
            print("--- roles ---")
            for r in cur.fetchall():
                arche = (r[3] or "")[:80] + ("…" if r[3] and len(r[3]) > 80 else "")
                print(f"  #{r[0]} {r[1]:<24} {r[2]:<28} archetype={arche!r}")

            cur.execute(
                """
                SELECT role_id, status, prompt_version, started_at
                  FROM v2_run_log
                 WHERE prompt_version LIKE 'stage0_charter_%'
                 ORDER BY started_at
                """
            )
            print("\n--- v2_run_log (stage 0) ---")
            for r in cur.fetchall():
                print(f"  {r[0]:<24} {r[1]:<18} {r[2]:<24} {r[3]}")

            cur.execute(
                """
                SELECT id, run_id, item_type, resolved, resolved_by
                  FROM v2_review_queue
                 WHERE item_type = 'charter_review'
                 ORDER BY id
                """
            )
            print("\n--- v2_review_queue (charter_review) ---")
            for r in cur.fetchall():
                print(f"  #{r[0]} run={r[1]} resolved={r[3]} by={r[4]!r}")


if __name__ == "__main__":
    main()
