"""Pydantic schemas for the Stage 3b LLM reconciler.

The reconciler takes one flagged dim pair (output of sub-stage 3a) and
returns one ReconciliationDecision: MERGE, SPLIT, or KEEP_SEPARATE.

Cross-field validation enforces that decisions and their associated
fields stay in sync — the LLM occasionally picks a decision and forgets
to fill the required fields, or fills the wrong ones. We refuse to parse
those rather than letting Stage 3c apply a malformed decision.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


DecisionKind = Literal["MERGE", "SPLIT", "KEEP_SEPARATE"]


class SplitDimension(BaseModel):
    """One of the new dims a SPLIT decision proposes."""

    name: str = Field(min_length=3, max_length=80)
    # 600 chars: the LLM's merge/split descriptions sometimes spell out
    # multi-clause scope statements ("Security practices for application
    # security, web app security, and compliance program controls.")
    # that legitimately exceed the original 400-char ceiling. We retain
    # an upper bound to avoid prompt-bloat on downstream stages.
    description: str = Field(min_length=10, max_length=600)


class ReconciliationDecision(BaseModel):
    """One decision per flagged pair.

    Field requirements by decision:
      * MERGE          -> merge_into_name + merge_into_description required;
                           split_into forbidden
      * SPLIT          -> split_into required (>=2 items);
                           merge_into_* forbidden
      * KEEP_SEPARATE  -> both merge and split fields forbidden
    """

    decision: DecisionKind
    # 1500 chars (was 800, originally 500). The reconciler's reasoning
    # sometimes spells out the boundary case in detail across multiple
    # clauses — "Dim A is automated testing... Dim B handles validation
    # and exception/error-flow design. These are adjacent but distinct
    # skills: one verifies behavior via tests, the other hardens runtime
    # behavior via validation..." A 1500-char ceiling covers the worst
    # legitimate cases we've seen in prod (~900-1100 chars) without
    # rejecting valid decisions. Downstream stages don't consume this
    # field — it's audit-only.
    reasoning: str = Field(min_length=20, max_length=1500)

    # Required for MERGE only:
    merge_into_name: str | None = Field(default=None, max_length=80)
    # 600 chars (was 400) — see note on SplitDimension.description.
    merge_into_description: str | None = Field(default=None, max_length=600)

    # Required for SPLIT only:
    split_into: list[SplitDimension] | None = None

    @model_validator(mode="after")
    def _validate_decision_fields(self) -> "ReconciliationDecision":
        if self.decision == "MERGE":
            if not self.merge_into_name or not (self.merge_into_name or "").strip():
                raise ValueError("MERGE decision requires merge_into_name")
            if not self.merge_into_description or not (self.merge_into_description or "").strip():
                raise ValueError("MERGE decision requires merge_into_description")
            if self.split_into:
                raise ValueError("MERGE decision must not include split_into")

        elif self.decision == "SPLIT":
            if not self.split_into:
                raise ValueError("SPLIT decision requires split_into")
            if len(self.split_into) < 2:
                raise ValueError(
                    "SPLIT decision requires at least 2 replacement dims; "
                    "a single replacement is a rename, not a split"
                )
            if self.merge_into_name or self.merge_into_description:
                raise ValueError(
                    "SPLIT decision must not include merge_into_name / "
                    "merge_into_description"
                )

        else:  # KEEP_SEPARATE
            if (
                self.merge_into_name
                or self.merge_into_description
                or self.split_into
            ):
                raise ValueError(
                    "KEEP_SEPARATE decision must not include merge or split fields"
                )

        return self
