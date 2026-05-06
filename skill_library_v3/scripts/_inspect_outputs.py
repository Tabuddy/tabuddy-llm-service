"""Print the latest approved charter and role card per role, plus run state."""
from __future__ import annotations

import json

from skill_library_v3.db.connection import connect


def main() -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role_id, prompt_version, status, started_at, completed_at,
                       error_message, planner_output
                  FROM v2_run_log
                 WHERE prompt_version LIKE 'stage%_%'
                 ORDER BY role_id, prompt_version, started_at
                """
            )
            for r in cur.fetchall():
                role_id, pv, status, started, completed, err, output = r
                print(f"=== {role_id} | {pv} | status={status}")
                print(f"    started={started}  completed={completed}")
                if err:
                    print(f"    error={err[:400]}")
                if output:
                    body = output if isinstance(output, dict) else json.loads(output)
                    if isinstance(body, dict):
                        body = {k: v for k, v in body.items() if not k.startswith("_")}
                    # For Stage 2 runs, summarize dim count + names rather than
                    # printing the full body (which can be 20 dims of detail).
                    if pv.startswith("stage2_dim_gen_") and isinstance(body, dict):
                        dims = body.get("candidate_dimensions") or []
                        print(f"    {len(dims)} candidate dimension(s):")
                        for d in dims:
                            if not isinstance(d, dict):
                                continue
                            n_exemplars = len(d.get("exemplar_skills") or [])
                            n_overlaps = len(d.get("overlap_flags") or [])
                            print(
                                f"      - {d.get('tentative_id', '?')} "
                                f"{d.get('name', '?')!r}  "
                                f"({n_exemplars} exemplars, "
                                f"{n_overlaps} overlap flags)"
                            )
                    else:
                        print(json.dumps(body, indent=2, ensure_ascii=False))
                print()


if __name__ == "__main__":
    main()
