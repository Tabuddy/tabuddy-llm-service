"""Pure-function lifecycle decider for Stage 3 (reconciliation).

Extracted from the runner so we can TDD it cleanly without spinning up
the LLM, embedder, or DB layers. The runner imports this and uses the
returned string as the v2_run_log.status to write.
"""

from __future__ import annotations

from typing import Iterable, Literal

from skill_library_v3.schemas.reconciliation import ReconciliationDecision


ReconStatus = Literal["approved", "awaiting_review", "failed"]


def decide_recon_status(
    *,
    decisions: Iterable[ReconciliationDecision],
    validator_log: dict,
) -> ReconStatus:
    """Decide the Stage 3 lifecycle status given 3b's decisions and 3c's
    post-validator output.

    Rules:
      * Any error in ``validator_log`` -> ``"failed"`` (terminal — fix
        via regenerate). Warnings are advisory and don't change status.
      * Any MERGE or SPLIT decision -> ``"awaiting_review"`` (per
        blueprint, human review required for these).
      * Otherwise (no pairs at all, or all KEEP_SEPARATE) ->
        ``"approved"`` (auto, no human gate needed).
    """
    if validator_log.get("errors"):
        return "failed"

    for d in decisions:
        if d.decision in ("MERGE", "SPLIT"):
            return "awaiting_review"

    return "approved"
