"""Purge awaiting_review / rejected run-log rows + remove roles whose
pipeline never reached Stage 8 approved.

Keeper rule: a role is kept iff it has at least one ``stage8_load_*``
v2_run_log row with status='approved'. Everything else (azure-cloud-
engineer, backend-engineer, mlops-engineer in the current dev DB) gets
its rows deleted — review_queue, run_log, and the roles row (CASCADEs
to role_aliases / role_dimensions).

For keeper roles the run history is preserved AS-IS except that the
awaiting_review / rejected status entries are dropped (their work is
either superseded or never landed). ``failed`` runs are kept for
forensic inspection.
"""

from __future__ import annotations

import logging

from skill_library_v3.db.connection import connect

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cleanup_dev_runs")


def main() -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            # 1. Identify keepers: roles with at least one approved Stage 8 run.
            cur.execute("""
                SELECT DISTINCT role_id
                  FROM v2_run_log
                 WHERE prompt_version LIKE %s AND status = 'approved'
            """, ("stage8_load_%",))
            keepers = sorted({r[0] for r in cur.fetchall()})
            logger.info("keeper roles: %s", keepers)

            # 2. All roles currently in the table.
            cur.execute("SELECT slug FROM roles ORDER BY slug")
            all_roles = [r[0] for r in cur.fetchall()]
            non_keepers = [s for s in all_roles if s not in keepers]
            logger.info("non-keeper roles to remove: %s", non_keepers)

            # 3. Purge review_queue rows for awaiting/rejected runs OR
            #    runs of non-keeper roles.
            cur.execute("""
                DELETE FROM v2_review_queue
                 WHERE run_id IN (
                     SELECT run_id FROM v2_run_log
                      WHERE status IN ('awaiting_review', 'rejected')
                         OR role_id = ANY(%s)
                 )
            """, (non_keepers,))
            logger.info("v2_review_queue rows deleted: %d", cur.rowcount or 0)

            # 4. Purge run_log rows in awaiting/rejected status OR
            #    belonging to non-keepers (any status).
            cur.execute("""
                DELETE FROM v2_run_log
                 WHERE status IN ('awaiting_review', 'rejected')
                    OR role_id = ANY(%s)
            """, (non_keepers,))
            logger.info("v2_run_log rows deleted: %d", cur.rowcount or 0)

            # 5. Drop the non-keeper roles. CASCADE drops their
            #    role_aliases + role_dimensions edges.
            if non_keepers:
                cur.execute("DELETE FROM roles WHERE slug = ANY(%s)", (non_keepers,))
                logger.info("roles deleted: %d", cur.rowcount or 0)
        conn.commit()

    # 6. Post-cleanup snapshot.
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT slug FROM roles ORDER BY slug")
            print()
            print("remaining roles:", [r[0] for r in cur.fetchall()])
            cur.execute("""
                SELECT prompt_version, status, count(*)
                  FROM v2_run_log
                 GROUP BY prompt_version, status
                 ORDER BY prompt_version, status
            """)
            print("\nremaining run-log distribution:")
            for r in cur.fetchall(): print(" ", r)


if __name__ == "__main__":
    main()
