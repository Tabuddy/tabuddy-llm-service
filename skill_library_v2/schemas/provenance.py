"""Provenance TypedDict per tabuddy_v2_architecture.md §2.

Every entity and edge carries one of these. Fields are append-only — each
pipeline stage adds what it knows; nothing gets overwritten. The computed
confidence (not stored) is a pure function of this record.

Phase 1 populates only a subset (Planner/run metadata). Phase 2+ fills the
rest as Generator, Critics, Validators, and the Market Signal subsystem
come online.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, TypedDict


class Provenance(TypedDict, total=False):
    # ── Origin ──
    created_by_agent: str            # e.g., "planner_v1.0", "generator_v2.0"
    model_snapshot: str              # pinned model id per §11
    prompt_version: str
    vocabulary_version: str

    # ── Grounding (entities) ──
    retrieval_source: Literal["catalog_hit", "proposed_new"]
    retrieval_hits: int
    evidence_urls: list[str]

    # ── Critic trail ──
    taxonomy_critic: Literal["approve", "reject", "uncertain"]
    duplicate_critic_score: float
    existence_critic: Literal["verified", "unverified", "n/a"]
    semantic_critic: Literal["approve", "reject", "uncertain"]

    # ── Validator trail ──
    validators_passed: list[str]
    validators_failed: list[str]

    # ── Market signal (entities) ──
    demand_score: float | None
    demand_trend: Literal["rising", "stable", "declining", "unknown"]

    # ── Corroboration (edges) ──
    corroborating_roles: list[str]
    pmi_score: float | None
    resume_transition_count: int | None

    # ── Human review ──
    human_reviewed: bool
    human_reviewer_id: str | None
    review_timestamp: datetime | None
