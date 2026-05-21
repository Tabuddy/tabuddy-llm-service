"""Pure-function lifecycle decider for Stage 3 (reconciliation).

Routes a Stage 3 run to one of three terminal statuses based on the
decisions the LLM made on each flagged pair PLUS the pair's kind
(intra_role vs cross_role).

Policy:

  * Validator errors                → "failed"   (terminal; fix via regenerate)
  * Any SPLIT decision              → "awaiting_review"
      (SPLITs are rare and replace 1 dim with 2+ new dims — always review)
  * Any cross_role MERGE            → "awaiting_review"
      (could collapse distinct catalog roles' dims, polluting the catalog)
  * Unknown pair_kind + MERGE       → "awaiting_review"
      (conservative default — when blast radius is unclear, ask an admin)
  * Otherwise (no pairs, or only KEEP_SEPARATE, or only intra_role MERGEs)
                                     → "approved"
      (intra_role MERGEs are the LLM cleaning up its own role's dims; no
      cross-role catalog impact, safe to auto-apply)

The intra-role auto-approve was added for the Phase 2 Excel-import batch.
Without it every new role landed in awaiting_review because the LLM's
dim-gen output for a fresh role typically produces 1-3 pairs of related-
but-redundant dimensions (e.g. "Pega Rule Authoring" + "Pega Decisioning"
that the reconciler then merges into one). Forcing manual approval for
that LLM self-cleanup is gate friction without integrity benefit.

Extracted from the runner so we can TDD it cleanly without spinning up
the LLM, embedder, or DB layers.
"""

from __future__ import annotations

from typing import Iterable, Literal

from skill_library_v3.schemas.reconciliation import ReconciliationDecision


ReconStatus = Literal["approved", "awaiting_review", "failed"]


def decide_recon_status(
    *,
    pair_decisions: list[tuple[dict, ReconciliationDecision]] | None = None,
    decisions: Iterable[ReconciliationDecision] | None = None,
    validator_log: dict,
) -> ReconStatus:
    """Decide the Stage 3 lifecycle status.

    Prefer ``pair_decisions`` (new caller signature) — that's the only way to
    distinguish intra_role MERGEs (safe) from cross_role MERGEs (risky).

    ``decisions`` is the legacy kwarg kept for backwards compat with callers
    that haven't been migrated yet; under that path we conservatively
    escalate every MERGE/SPLIT (the original behavior).
    """
    if validator_log.get("errors"):
        return "failed"

    # New path: pair_kind-aware routing.
    if pair_decisions is not None:
        for pair, d in pair_decisions:
            if d.decision == "SPLIT":
                # SPLITs always escalate regardless of pair_kind — they
                # replace 1 dim with 2+ new dims and are rare enough to
                # warrant a human look.
                return "awaiting_review"
            if d.decision == "MERGE":
                kind = (pair or {}).get("kind")
                if kind != "intra_role":
                    # Unknown kind or explicit cross_role → escalate.
                    return "awaiting_review"
                # intra_role MERGE → safe to auto-approve, continue loop.
        return "approved"

    # Legacy path: no pair context → conservative escalation on any
    # MERGE/SPLIT (the pre-Phase-2 behavior).
    if decisions is not None:
        for d in decisions:
            if d.decision in ("MERGE", "SPLIT"):
                return "awaiting_review"
        return "approved"

    # No decisions and no pair_decisions provided → treat as "no work".
    return "approved"
