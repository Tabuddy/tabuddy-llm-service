"""Pure-function lifecycle decider for Stage 0 (charter).

Stage 0's charter has these fields (per ``schemas/charter.py``):
  * role_id, role_name
  * definition (str)
  * in_scope, out_of_scope (lists of items)
  * adjacent_roles (list of {role, boundary_note})
  * always_load_skills (list of skill names)

Autonomy rule: a charter that has a non-empty definition AND populated
in_scope, out_of_scope, and adjacent_roles is a confident output and
auto-approves. A stub charter (any of those empty) gates at
awaiting_review for human eyes.

  * Any error in validator_log -> 'failed' (terminal — fix via regenerate)
  * Charter structurally thin -> 'awaiting_review'
  * Otherwise -> 'approved' (auto)

Warnings are advisory and never gate.
"""

from __future__ import annotations

from typing import Literal


CharterStatus = Literal["approved", "awaiting_review", "failed"]


_MIN_IN_SCOPE = 1
_MIN_OUT_OF_SCOPE = 1
_MIN_ADJACENT_ROLES = 1


def decide_charter_status(
    *,
    charter: dict,
    validator_log: dict,
) -> CharterStatus:
    """Pick the lifecycle status for a freshly-generated Stage 0 charter."""
    if validator_log.get("errors"):
        return "failed"

    in_scope = charter.get("in_scope") or []
    out_of_scope = charter.get("out_of_scope") or []
    adjacent = charter.get("adjacent_roles") or []
    definition = (charter.get("definition") or "").strip()

    if (
        not definition
        or len(in_scope) < _MIN_IN_SCOPE
        or len(out_of_scope) < _MIN_OUT_OF_SCOPE
        or len(adjacent) < _MIN_ADJACENT_ROLES
    ):
        return "awaiting_review"

    return "approved"
