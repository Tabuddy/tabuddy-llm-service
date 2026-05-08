"""Orchestrates a single Stage 0 charter run end-to-end.

Lifecycle: pending → running → awaiting_review → approved | rejected,
with `failed` as a terminal error state.

The HTTP layer is a thin shell — it spawns ``run_stage_0`` as a fire-and-forget
asyncio task and returns 202 immediately. The UI polls the run status while
this function does its work.
"""

from __future__ import annotations

import logging
import traceback
import uuid

from llm_client import GENERATION_MODEL
from skill_library_v3.agents.charter import Stage0CharterAgent
from skill_library_v3.agents.containment import Stage6ContainmentAgent
from skill_library_v3.agents.dimension_gen import Stage2DimensionGeneratorAgent
from skill_library_v3.agents.dimension_validators import run_dimension_validators
from skill_library_v3.agents.enrichment import (
    Stage7AmbiguityAgent,
    Stage7ContextKeywordsAgent,
    Stage7MaturityAgent,
    Stage7VendorLicenseAgent,
    Stage7VersioningAgent,
)
from skill_library_v3.agents.placer import Stage5PlacerAgent
from skill_library_v3.agents.reconciler import Stage3ReconcilerAgent
from skill_library_v3.agents.role_anchor import Stage1RoleAnchorAgent
from skill_library_v3.agents.role_card_validators import run_role_card_validators
from skill_library_v3.agents.type_assigner import Stage4TypeAssignerAgent
from skill_library_v3.agents.validators import run_validators
from skill_library_v3.canonical_loader import load_catalog_for_role
from skill_library_v3.catalog_transform import build_catalog_payload
from skill_library_v3.charter_status import decide_charter_status
from skill_library_v3.confidence_routing import route_by_confidence
from skill_library_v3.db import repository as repo
from skill_library_v3.enrichment_status import decide_enrichment_status
from skill_library_v3.enrichment_validators import (
    validate_ambiguity_in_catalog,
    validate_maturity_distribution,
    validate_vendor_recognized,
)
from skill_library_v3.dim_apply import apply_decisions
from skill_library_v3.dim_embedder import make_default_embedder
from skill_library_v3.dim_overlap_pairs import compute_overlap_pairs
from skill_library_v3.dim_post_validate import (
    validate_exemplar_coverage,
    validate_no_near_duplicates,
)
from skill_library_v3.placement_validators import (
    cross_skill_consistency,
    embedding_cross_check,
    secondary_dim_audit,
    tool_standard_cross_mismatch,
    type_dim_consistency,
)
from skill_library_v3.prompts.charter import CHARTER_PROMPT_VERSION
from skill_library_v3.prompts.containment import CONTAINMENT_PROMPT_VERSION
from skill_library_v3.prompts.dimension_gen import DIM_GEN_PROMPT_VERSION
from skill_library_v3.prompts.enrichment import ENRICHMENT_PROMPT_VERSION
from skill_library_v3.prompts.placer import PLACER_PROMPT_VERSION
from skill_library_v3.prompts.reconciler import RECONCILER_PROMPT_VERSION
from skill_library_v3.prompts.role_card import ROLE_CARD_PROMPT_VERSION
from skill_library_v3.prompts.type_assigner import TYPE_ASSIGNER_PROMPT_VERSION
from skill_library_v3.recon_status import decide_recon_status
from skill_library_v3.schemas.enrichment import SkillEnrichment
from skill_library_v3.relationship_validators import (
    architecture_has_no_parents,
    library_has_lang_or_framework_parent,
    service_has_platform_parent,
    suppress_symmetry,
    validate_dag_no_cycles,
)
from skill_library_v3.schemas.placement import PlacedSkill
from skill_library_v3.schemas.typology import TypedSkill
from skill_library_v3.skill_aggregation import aggregate_skills, batch_skills
from skill_library_v3.skill_similarity import top_k_similar_skills

logger = logging.getLogger(__name__)


async def run_stage_0(run_id: uuid.UUID) -> None:
    """Execute Stage 0 for ``run_id``. Always terminal — sets the run to
    awaiting_review on success or failed on exception. Never raises."""
    import asyncio

    try:
        await asyncio.to_thread(repo.transition_run, run_id, status="running")

        # Pull everything we need — role record, adjacency, JD samples,
        # alias lookup — in one pass on a worker thread.
        ctx = await asyncio.to_thread(_load_run_context, run_id)
        if ctx is None:
            await asyncio.to_thread(
                repo.fail_run,
                run_id,
                error_message="run not found at execution time",
            )
            return

        agent = Stage0CharterAgent()
        charter = await agent.generate(
            role_slug=ctx["role_slug"],
            role_name=ctx["role_display"],
            adjacent_roles=ctx["adjacent_roles"],
            jd_samples=ctx["jd_samples"],
        )

        validator_log = run_validators(
            charter,
            approved_role_names=ctx["approved_role_names"],
            alias_lookup=ctx["alias_lookup"],
        )
        charter_dict = charter.model_dump()
        final_status = decide_charter_status(
            charter=charter_dict, validator_log=validator_log,
        )

        await asyncio.to_thread(
            repo.complete_charter_run,
            run_id,
            charter=charter_dict,
            validator_log=validator_log,
            final_status=final_status,
        )
        logger.info(
            "[v3 run %s] charter for %s: status=%s (%d warning(s), %d error(s))",
            run_id, ctx["role_display"], final_status,
            len(validator_log.get("warnings", [])),
            len(validator_log.get("errors", [])),
        )
        if final_status == "approved":
            asyncio.create_task(run_stage_1(run_id))
    except Exception as exc:  # noqa: BLE001 — runner is the last line of defense
        tb = traceback.format_exc()
        logger.exception("[v3 run %s] failed", run_id)
        try:
            await asyncio.to_thread(
                repo.fail_run,
                run_id,
                error_message=f"{type(exc).__name__}: {exc}\n\n{tb}",
            )
        except Exception:  # noqa: BLE001
            logger.exception("[v3 run %s] could not even mark failed", run_id)


def _load_run_context(run_id: uuid.UUID) -> dict | None:
    """Pull every read-side input the agent + validators need. One worker
    thread, one round of DB connections — keeps the async event loop free."""
    run = repo.get_run_with_review(run_id)
    if run is None:
        return None
    role_slug = run["role_slug"]
    return {
        "role_slug": role_slug,
        "role_display": run["role_display"],
        "adjacent_roles": repo.get_approved_adjacent_roles(exclude_slug=role_slug),
        "jd_samples": repo.get_jd_samples_for_role(role_slug, limit=5),
        "approved_role_names": repo.get_approved_role_names(),
        "alias_lookup": repo.get_alias_lookup_set(),
    }


# ── Public helpers used by the FastAPI router ──────────────────────────────


def create_initial_run(role_name: str) -> dict:
    """Upsert the role and create a pending run row. Returns
    ``{run_id, role_slug}``. The caller is expected to ``asyncio.create_task``
    the resulting ``run_stage_0(run_id)``."""
    role = repo.upsert_role(role_name)
    run_id = repo.create_pending_run(
        role_slug=role["slug"],
        role_display=role["display_name"],
        prompt_version=CHARTER_PROMPT_VERSION,
        model_snapshot=GENERATION_MODEL,
    )
    return {"run_id": run_id, "role_slug": role["slug"]}


def regenerate_run(
    prior_run_id: uuid.UUID,
    *,
    reviewed_by: str | None,
    notes: str | None,
) -> dict:
    """Reject ``prior_run_id``, create a sibling pending run.

    Returns ``{"run_id": <uuid>, "prompt_version": <str>}`` so the FastAPI
    handler can dispatch to the right stage runner without a second DB hit.
    """
    return repo.reject_and_create_regenerate(
        prior_run_id,
        reviewed_by=reviewed_by,
        notes=notes,
    )


# ── Stage 1 runner ─────────────────────────────────────────────────────────


async def run_stage_1(charter_run_id: uuid.UUID) -> None:
    """Generate a role card from an approved Stage 0 charter run.

    Lifecycle:
      pending  → running  → approved          (validators pass — auto)
      pending  → running  → awaiting_review   (validators flag errors — UI)
      pending  → running  → failed            (LLM exception)

    Always terminal — never raises out of this function.
    """
    import asyncio

    stage1_run_id: uuid.UUID | None = None
    try:
        # Resolve the charter run + load its approved planner_output. We do
        # this synchronously on a worker thread; the LLM call is the only
        # slow piece and runs on the event loop.
        ctx = await asyncio.to_thread(_load_stage_1_context, charter_run_id)
        if ctx is None:
            logger.warning(
                "[v3 stage1] charter run %s not found or not approved; skipping",
                charter_run_id,
            )
            return

        stage1_run_id = await asyncio.to_thread(
            repo.create_pending_run,
            role_slug=ctx["role_slug"],
            role_display=ctx["role_display"],
            prompt_version=ROLE_CARD_PROMPT_VERSION,
            model_snapshot=GENERATION_MODEL,
            parent_run_id=charter_run_id,  # stashed in payload for lineage
        )

        await asyncio.to_thread(repo.transition_run, stage1_run_id, status="running")

        agent = Stage1RoleAnchorAgent()
        role_card = await agent.generate(
            role_slug=ctx["role_slug"],
            role_name=ctx["role_display"],
            charter=ctx["charter"],
        )

        validator_log = run_role_card_validators(role_card, charter=ctx["charter"])

        final_status = await asyncio.to_thread(
            repo.complete_role_card_run,
            stage1_run_id,
            role_card=role_card.model_dump(),
            validator_log=validator_log,
        )
        logger.info(
            "[v3 stage1 run %s] role card for %s: status=%s, "
            "%d warning(s), %d error(s)",
            stage1_run_id,
            ctx["role_display"],
            final_status,
            len(validator_log.get("warnings", [])),
            len(validator_log.get("errors", [])),
        )

        # Cascade: fire Stage 2 when Stage 1 auto-approves. Human-approve
        # path is handled in the FastAPI router (api_approve_run).
        if final_status == "approved":
            asyncio.create_task(run_stage_2(ctx["role_slug"]))
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("[v3 stage1] run for charter %s failed", charter_run_id)
        if stage1_run_id is not None:
            try:
                await asyncio.to_thread(
                    repo.fail_run,
                    stage1_run_id,
                    error_message=f"{type(exc).__name__}: {exc}\n\n{tb}",
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[v3 stage1 run %s] could not even mark failed", stage1_run_id
                )


def _load_stage_1_context(charter_run_id: uuid.UUID) -> dict | None:
    """Fetch the charter run + verify it's an approved Stage 0 row, and
    extract the charter body (planner_output minus internal underscore-prefixed
    keys). Returns ``None`` when the run isn't a valid Stage 1 input."""
    run = repo.get_run_with_review(charter_run_id)
    if run is None:
        return None
    if run["status"] != "approved":
        return None
    if not (run.get("prompt_version") or "").startswith("stage0_charter_"):
        return None
    charter = run.get("charter") or {}
    return {
        "role_slug": run["role_slug"],
        "role_display": run["role_display"],
        "charter": charter,
    }


def enqueue_stage_1_for(charter_run_id: uuid.UUID) -> None:
    """Helper for FastAPI handlers and the backfill script: spawn a Stage 1
    run as a fire-and-forget asyncio task. No-op if the calling code isn't
    inside a running event loop (the backfill script handles its own loop)."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — caller is responsible for awaiting run_stage_1
        # directly (the backfill script does this).
        return
    loop.create_task(run_stage_1(charter_run_id))


# ── Stage 1 regenerate ─────────────────────────────────────────────────────


async def run_stage_1_regenerate(new_run_id: uuid.UUID) -> None:
    """Drive a freshly-created Stage 1 pending run that was inserted by
    ``repo.reject_and_create_regenerate``. The pending row already has the
    right prompt_version + role + parent_run_id stash; this just walks it
    through running → approved/awaiting_review/failed.
    """
    import asyncio

    try:
        ctx = await asyncio.to_thread(_load_stage_1_regenerate_context, new_run_id)
        if ctx is None:
            await asyncio.to_thread(
                repo.fail_run,
                new_run_id,
                error_message="Stage 1 regenerate: pending run not found or invalid",
            )
            return

        await asyncio.to_thread(repo.transition_run, new_run_id, status="running")

        agent = Stage1RoleAnchorAgent()
        role_card = await agent.generate(
            role_slug=ctx["role_slug"],
            role_name=ctx["role_display"],
            charter=ctx["charter"],
        )
        validator_log = run_role_card_validators(role_card, charter=ctx["charter"])

        final_status = await asyncio.to_thread(
            repo.complete_role_card_run,
            new_run_id,
            role_card=role_card.model_dump(),
            validator_log=validator_log,
        )
        logger.info(
            "[v3 stage1 regen %s] role card for %s: status=%s",
            new_run_id, ctx["role_display"], final_status,
        )
        if final_status == "approved":
            asyncio.create_task(run_stage_2(ctx["role_slug"]))
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("[v3 stage1 regen %s] failed", new_run_id)
        try:
            await asyncio.to_thread(
                repo.fail_run,
                new_run_id,
                error_message=f"{type(exc).__name__}: {exc}\n\n{tb}",
            )
        except Exception:  # noqa: BLE001
            logger.exception("[v3 stage1 regen %s] could not even mark failed", new_run_id)


def _load_stage_1_regenerate_context(new_run_id: uuid.UUID) -> dict | None:
    """Resolve the charter to feed a regenerated Stage 1 run.

    The simpler-than-walking-lineage path: load the latest approved charter
    for the same role. Stage 1 should always read "the current truth" — if
    the charter has been re-approved since the prior Stage 1 attempt, we
    want the new charter, not the old one.
    """
    new_run = repo.get_run_with_review(new_run_id)
    if new_run is None:
        return None
    charter = repo.get_latest_approved_charter_for_role(new_run["role_slug"])
    if charter is None:
        return None
    return {
        "role_slug": new_run["role_slug"],
        "role_display": new_run["role_display"],
        "charter": charter,
    }


# ── Stage 2 runner (headless) ──────────────────────────────────────────────


async def run_stage_2(role_slug: str) -> None:
    """Generate the candidate dimension list for ``role_slug`` from its
    latest approved role card.

    Lifecycle (headless — NO awaiting_review state):
      pending → running → approved (validators pass)
                       → failed   (validators error or LLM exception)

    Always terminal — never raises out of this function.
    """
    import asyncio

    stage2_run_id: uuid.UUID | None = None
    try:
        ctx = await asyncio.to_thread(_load_stage_2_context, role_slug)
        if ctx is None:
            logger.warning(
                "[v3 stage2] role %r has no approved role card; skipping",
                role_slug,
            )
            return

        # Resolve the role display name from the role card so the Stage 2
        # prompt and v2_run_log row use the canonical name.
        role_display = (
            ctx["role_card"].get("canonical_name")
            or ctx["role_display"]
            or role_slug
        )

        stage2_run_id = await asyncio.to_thread(
            repo.create_pending_run,
            role_slug=role_slug,
            role_display=role_display,
            prompt_version=DIM_GEN_PROMPT_VERSION,
            model_snapshot=GENERATION_MODEL,
            parent_run_id=ctx["role_card_run_id"],
        )

        await asyncio.to_thread(repo.transition_run, stage2_run_id, status="running")

        agent = Stage2DimensionGeneratorAgent()
        dim_list = await agent.generate(
            role_slug=role_slug,
            role_name=role_display,
            role_card=ctx["role_card"],
            always_load_skills=ctx["always_load_skills"],
            adjacent_dimensions_by_role=ctx["adjacent_dimensions_by_role"],
        )

        validator_log = run_dimension_validators(
            dim_list,
            always_load_skills=ctx["always_load_skills"],
            adjacent_dimensions_by_role=ctx["adjacent_dimensions_by_role"],
        )

        # Strip Pydantic wrapper down to plain dim list for storage.
        dimensions_payload = [
            d.model_dump() for d in dim_list.candidate_dimensions
        ]

        final_status = await asyncio.to_thread(
            repo.complete_dimensions_run,
            stage2_run_id,
            dimensions=dimensions_payload,
            validator_log=validator_log,
        )
        logger.info(
            "[v3 stage2 run %s] dimensions for %s: status=%s, "
            "%d dim(s), %d warning(s), %d error(s)",
            stage2_run_id,
            role_display,
            final_status,
            len(dimensions_payload),
            len(validator_log.get("warnings", [])),
            len(validator_log.get("errors", [])),
        )
        # Cascade: fire Stage 3 (reconciliation) when Stage 2 lands
        # 'approved'. Headless runs don't have a human-Pass surface;
        # the cascade is the only auto-trigger path for Stage 3.
        if final_status == "approved":
            asyncio.create_task(run_stage_3(role_slug))
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("[v3 stage2] run for role %r failed", role_slug)
        if stage2_run_id is not None:
            try:
                await asyncio.to_thread(
                    repo.fail_run,
                    stage2_run_id,
                    error_message=f"{type(exc).__name__}: {exc}\n\n{tb}",
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[v3 stage2 run %s] could not even mark failed",
                    stage2_run_id,
                )


def _load_stage_2_context(role_slug: str) -> dict | None:
    """Pull every Stage 2 input synchronously: latest approved role card,
    its underlying run_id, the always_load_skills field from the card,
    and adjacent roles' existing Stage 2 dimensions for overlap context.

    Returns ``None`` when no approved role card exists for the role —
    Stage 2 is meaningless without one.
    """
    role_card = repo.get_role_card_for_role(role_slug)
    if role_card is None:
        return None
    role_card_run = repo.get_latest_role_card_run_for_role(role_slug)
    if role_card_run is None:
        return None
    adjacent_role_names = role_card.get("adjacent_roles") or []
    adjacent_dimensions_by_role = repo.get_adjacent_role_dimensions(
        adjacent_role_names
    )
    return {
        "role_card": role_card,
        "role_card_run_id": uuid.UUID(role_card_run["run_id"]),
        "role_display": role_card_run.get("role_slug") or role_slug,
        "always_load_skills": role_card.get("always_load_skills") or [],
        "adjacent_dimensions_by_role": adjacent_dimensions_by_role,
    }


async def run_stage_2_regenerate(new_run_id: uuid.UUID) -> None:
    """Drive a freshly-created Stage 2 pending run that was inserted by
    ``repo.reject_and_create_regenerate``. Same shape as ``run_stage_2``
    but starts from the pending row rather than creating a new one.

    Used for the manual "Regenerate dimensions" button in the UI.
    """
    import asyncio

    try:
        meta = await asyncio.to_thread(repo.get_run_with_review, new_run_id)
        if meta is None:
            await asyncio.to_thread(
                repo.fail_run,
                new_run_id,
                error_message="Stage 2 regenerate: pending run not found",
            )
            return
        role_slug = meta["role_slug"]

        ctx = await asyncio.to_thread(_load_stage_2_context, role_slug)
        if ctx is None:
            await asyncio.to_thread(
                repo.fail_run,
                new_run_id,
                error_message=(
                    "Stage 2 regenerate: role has no approved role card; "
                    "regenerate Stage 1 first"
                ),
            )
            return

        await asyncio.to_thread(repo.transition_run, new_run_id, status="running")

        agent = Stage2DimensionGeneratorAgent()
        dim_list = await agent.generate(
            role_slug=role_slug,
            role_name=ctx["role_card"].get("canonical_name") or role_slug,
            role_card=ctx["role_card"],
            always_load_skills=ctx["always_load_skills"],
            adjacent_dimensions_by_role=ctx["adjacent_dimensions_by_role"],
        )
        validator_log = run_dimension_validators(
            dim_list,
            always_load_skills=ctx["always_load_skills"],
            adjacent_dimensions_by_role=ctx["adjacent_dimensions_by_role"],
        )
        dimensions_payload = [d.model_dump() for d in dim_list.candidate_dimensions]

        final_status = await asyncio.to_thread(
            repo.complete_dimensions_run,
            new_run_id,
            dimensions=dimensions_payload,
            validator_log=validator_log,
        )
        logger.info(
            "[v3 stage2 regen %s] dimensions for %s: status=%s, %d dim(s)",
            new_run_id, role_slug, final_status, len(dimensions_payload),
        )
        if final_status == "approved":
            asyncio.create_task(run_stage_3(role_slug))
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("[v3 stage2 regen %s] failed", new_run_id)
        try:
            await asyncio.to_thread(
                repo.fail_run,
                new_run_id,
                error_message=f"{type(exc).__name__}: {exc}\n\n{tb}",
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "[v3 stage2 regen %s] could not even mark failed", new_run_id
            )


# ── Stage 3 runner (reconciliation) ────────────────────────────────────────


async def run_stage_3(role_slug: str) -> None:
    """Reconcile a role's Stage 2 candidate dimensions.

    Lifecycle:
      pending → running → approved          (no flagged pairs OR all
                                              KEEP_SEPARATE; auto)
                       → awaiting_review   (any MERGE / SPLIT decision —
                                              human review required per
                                              blueprint)
                       → failed            (post-validator errors or
                                              LLM exception)

    Always terminal — never raises out of this function.
    """
    import asyncio

    stage3_run_id: uuid.UUID | None = None
    try:
        ctx = await asyncio.to_thread(_load_stage_3_context, role_slug)
        if ctx is None:
            logger.warning(
                "[v3 stage3] role %r has no approved Stage 2 dims; skipping",
                role_slug,
            )
            return

        role_display = ctx["role_display"]

        stage3_run_id = await asyncio.to_thread(
            repo.create_pending_run,
            role_slug=role_slug,
            role_display=role_display,
            prompt_version=RECONCILER_PROMPT_VERSION,
            model_snapshot=GENERATION_MODEL,
            parent_run_id=ctx["dimensions_run_id"],
        )
        await asyncio.to_thread(repo.transition_run, stage3_run_id, status="running")

        # Sub-stage 3a — flag overlapping pairs.
        embedder = make_default_embedder()
        flagged_pairs = await compute_overlap_pairs(
            role_slug=role_slug,
            role_display=role_display,
            role_dimensions=ctx["candidate_dimensions"],
            adjacent_dimensions_by_role=ctx["adjacent_dimensions_by_role"],
            embedder=embedder,
        )

        # Sub-stage 3b — LLM reconciliation per pair.
        decisions: list = []
        pair_decision_audit: list[dict] = []
        if flagged_pairs:
            agent = Stage3ReconcilerAgent()
            for pair in flagged_pairs:
                a_dim = _resolve_dim_for_pair_side(pair, "a", ctx)
                b_dim = _resolve_dim_for_pair_side(pair, "b", ctx)
                if a_dim is None or b_dim is None:
                    # Skip pair if one side can't be resolved (defensive).
                    continue
                decision = await agent.reconcile(
                    a_role=pair["a_role"],
                    a_dim=a_dim,
                    b_role=pair["b_role"],
                    b_dim=b_dim,
                    similarity=pair["similarity"],
                )
                decisions.append(decision)
                pair_decision_audit.append(
                    {"pair": pair, "decision": decision.model_dump()}
                )

        # Sub-stage 3c — apply decisions deterministically.
        apply_result = apply_decisions(
            candidate_dims=ctx["candidate_dimensions"],
            pair_decisions=list(zip(flagged_pairs, decisions)),
        )

        # Post-validators.
        validator_errors: list[dict] = []
        validator_warnings: list[dict] = []
        validator_errors.extend(
            validate_exemplar_coverage(
                originals=ctx["candidate_dimensions"], result=apply_result,
            )
        )
        near_dup_errors = await validate_no_near_duplicates(
            locked_dimensions=apply_result.locked_dimensions, embedder=embedder,
        )
        # Treat near-duplicate findings as warnings rather than hard errors —
        # they're a "human should look at this" signal, not a code-bug
        # signal like exemplar coverage failure.
        for err in near_dup_errors:
            err = dict(err)
            err["level"] = "warning"
            validator_warnings.append(err)
        validator_log = {"errors": validator_errors, "warnings": validator_warnings}

        final_status = decide_recon_status(
            decisions=decisions, validator_log=validator_log,
        )

        payload = {
            "candidate_dimensions": ctx["candidate_dimensions"],
            "flagged_pairs": flagged_pairs,
            "decisions": pair_decision_audit,
            "locked_dimensions": apply_result.locked_dimensions,
            "merge_log": apply_result.merge_log,
            "split_log": apply_result.split_log,
            "unassigned_exemplars": apply_result.unassigned_exemplars,
        }

        await asyncio.to_thread(
            repo.complete_recon_run,
            stage3_run_id,
            payload=payload,
            validator_log=validator_log,
            final_status=final_status,
        )
        logger.info(
            "[v3 stage3 run %s] reconciliation for %s: status=%s "
            "(%d pairs, %d merge, %d split, %d locked)",
            stage3_run_id,
            role_display,
            final_status,
            len(flagged_pairs),
            len(apply_result.merge_log),
            len(apply_result.split_log),
            len(apply_result.locked_dimensions),
        )
        # Cascade: fire Stage 4 (skill typing) when Stage 3 auto-approves.
        # Human-approve path (awaiting_review -> approved) is handled in
        # the FastAPI router's api_approve_run.
        if final_status == "approved":
            asyncio.create_task(run_stage_4(role_slug))
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("[v3 stage3] run for role %r failed", role_slug)
        if stage3_run_id is not None:
            try:
                await asyncio.to_thread(
                    repo.fail_run,
                    stage3_run_id,
                    error_message=f"{type(exc).__name__}: {exc}\n\n{tb}",
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[v3 stage3 run %s] could not even mark failed",
                    stage3_run_id,
                )


def _resolve_dim_for_pair_side(
    pair: dict, side: str, ctx: dict,
) -> dict | None:
    """Look up the full dim dict for one side of a flagged pair.

    Pairs only carry tentative_id + name + role; the reconciler needs
    description + in_scope + out_of_scope + exemplars too. We index by
    (role, tentative_id) since two roles can both have d_init_01.
    """
    role = pair[f"{side}_role"]
    tid = pair[f"{side}_tentative_id"]
    if role == ctx["role_display"]:
        return ctx["candidate_dims_by_id"].get(tid)
    adjacent = ctx["adjacent_dims_by_role_then_id"].get(role) or {}
    return adjacent.get(tid)


def _load_stage_3_context(role_slug: str) -> dict | None:
    """Pull every Stage 3 input synchronously: latest approved Stage 2
    candidate dims for the focal role, the role card (for display name +
    adjacent role list), adjacent roles' approved Stage 2 dims, plus
    by-id lookups so the runner can resolve full dim dicts when a pair
    references a tentative_id.
    """
    role_card = repo.get_role_card_for_role(role_slug)
    if role_card is None:
        return None
    candidate_dimensions = repo.get_latest_dimensions_for_role(role_slug)
    if not candidate_dimensions:
        return None

    role_display = role_card.get("canonical_name") or role_slug
    adjacent_role_names = role_card.get("adjacent_roles") or []
    adjacent_dims_by_role = repo.get_adjacent_role_dimensions(adjacent_role_names)

    # Resolve the parent (Stage 2) run id for lineage tracking.
    s2_run = repo.get_latest_role_card_run_for_role(role_slug)
    parent_run_id = uuid.UUID(s2_run["run_id"]) if s2_run else None

    candidate_dims_by_id = {
        d["tentative_id"]: d for d in candidate_dimensions if "tentative_id" in d
    }
    adjacent_dims_by_role_then_id: dict[str, dict[str, dict]] = {}
    for role_name, dims in adjacent_dims_by_role.items():
        adjacent_dims_by_role_then_id[role_name] = {
            d["tentative_id"]: d for d in dims if "tentative_id" in d
        }

    return {
        "role_display": role_display,
        "candidate_dimensions": candidate_dimensions,
        "candidate_dims_by_id": candidate_dims_by_id,
        "adjacent_dimensions_by_role": adjacent_dims_by_role,
        "adjacent_dims_by_role_then_id": adjacent_dims_by_role_then_id,
        "dimensions_run_id": parent_run_id,
    }


async def run_stage_3_regenerate(new_run_id: uuid.UUID) -> None:
    """Drive a freshly-created Stage 3 pending run inserted by
    ``repo.reject_and_create_regenerate``. Same shape as ``run_stage_3``
    but starts from a pending row already in the DB instead of creating
    one. Used for the manual Pass/Regenerate flow on Stage 3 reviews."""
    import asyncio

    try:
        meta = await asyncio.to_thread(repo.get_run_with_review, new_run_id)
        if meta is None:
            await asyncio.to_thread(
                repo.fail_run,
                new_run_id,
                error_message="Stage 3 regenerate: pending run not found",
            )
            return
        role_slug = meta["role_slug"]
        ctx = await asyncio.to_thread(_load_stage_3_context, role_slug)
        if ctx is None:
            await asyncio.to_thread(
                repo.fail_run,
                new_run_id,
                error_message=(
                    "Stage 3 regenerate: role missing role card or Stage 2 "
                    "dimensions; rerun upstream stages first"
                ),
            )
            return

        await asyncio.to_thread(repo.transition_run, new_run_id, status="running")

        embedder = make_default_embedder()
        flagged_pairs = await compute_overlap_pairs(
            role_slug=role_slug,
            role_display=ctx["role_display"],
            role_dimensions=ctx["candidate_dimensions"],
            adjacent_dimensions_by_role=ctx["adjacent_dimensions_by_role"],
            embedder=embedder,
        )

        decisions: list = []
        pair_decision_audit: list[dict] = []
        if flagged_pairs:
            agent = Stage3ReconcilerAgent()
            for pair in flagged_pairs:
                a_dim = _resolve_dim_for_pair_side(pair, "a", ctx)
                b_dim = _resolve_dim_for_pair_side(pair, "b", ctx)
                if a_dim is None or b_dim is None:
                    continue
                decision = await agent.reconcile(
                    a_role=pair["a_role"],
                    a_dim=a_dim,
                    b_role=pair["b_role"],
                    b_dim=b_dim,
                    similarity=pair["similarity"],
                )
                decisions.append(decision)
                pair_decision_audit.append(
                    {"pair": pair, "decision": decision.model_dump()}
                )

        apply_result = apply_decisions(
            candidate_dims=ctx["candidate_dimensions"],
            pair_decisions=list(zip(flagged_pairs, decisions)),
        )

        validator_errors: list[dict] = []
        validator_warnings: list[dict] = []
        validator_errors.extend(
            validate_exemplar_coverage(
                originals=ctx["candidate_dimensions"], result=apply_result,
            )
        )
        for err in await validate_no_near_duplicates(
            locked_dimensions=apply_result.locked_dimensions, embedder=embedder,
        ):
            warn = dict(err)
            warn["level"] = "warning"
            validator_warnings.append(warn)
        validator_log = {"errors": validator_errors, "warnings": validator_warnings}

        final_status = decide_recon_status(
            decisions=decisions, validator_log=validator_log,
        )
        payload = {
            "candidate_dimensions": ctx["candidate_dimensions"],
            "flagged_pairs": flagged_pairs,
            "decisions": pair_decision_audit,
            "locked_dimensions": apply_result.locked_dimensions,
            "merge_log": apply_result.merge_log,
            "split_log": apply_result.split_log,
            "unassigned_exemplars": apply_result.unassigned_exemplars,
        }
        await asyncio.to_thread(
            repo.complete_recon_run,
            new_run_id,
            payload=payload,
            validator_log=validator_log,
            final_status=final_status,
        )
        logger.info(
            "[v3 stage3 regen %s] reconciliation for %s: status=%s",
            new_run_id, role_slug, final_status,
        )
        if final_status == "approved":
            asyncio.create_task(run_stage_4(role_slug))
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("[v3 stage3 regen %s] failed", new_run_id)
        try:
            await asyncio.to_thread(
                repo.fail_run,
                new_run_id,
                error_message=f"{type(exc).__name__}: {exc}\n\n{tb}",
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "[v3 stage3 regen %s] could not even mark failed", new_run_id,
            )


# ── Stage 4 runner (skill typing — headless) ──────────────────────────────


async def run_stage_4(role_slug: str) -> None:
    """Type every skill from this role's approved Stage 3 output.

    Lifecycle (headless — no awaiting_review state, no review_queue row):
      pending → running → approved (typing succeeded; routing buckets
                                      surface in the UI for spot-check)
                       → failed   (LLM exception or empty input)

    Stage 4 is per-role for now (consistent with the existing v3
    cascade pattern). Catalog-wide deduping comes in a future iteration.

    Always terminal — never raises out of this function.
    """
    import asyncio

    stage4_run_id: uuid.UUID | None = None
    try:
        # Resolve inputs.
        role_card = await asyncio.to_thread(repo.get_role_card_for_role, role_slug)
        if role_card is None:
            logger.warning(
                "[v3 stage4] role %r has no approved role card; skipping",
                role_slug,
            )
            return
        role_display = role_card.get("canonical_name") or role_slug

        artifacts = await asyncio.to_thread(repo.get_stage3_artifacts_for_role, role_slug)
        if artifacts is None:
            logger.warning(
                "[v3 stage4] role %r has no approved Stage 3 artifacts; skipping",
                role_slug,
            )
            return

        # Aggregate this role's skill universe (locked-dim exemplars +
        # unassigned exemplars from SPLITs).
        aggregated = aggregate_skills(
            [
                {
                    "role_display": role_display,
                    "locked_dimensions": artifacts["locked_dimensions"],
                    "unassigned_exemplars": artifacts["unassigned_exemplars"],
                }
            ]
        )
        if not aggregated:
            logger.info(
                "[v3 stage4] role %r has no skills to type; skipping",
                role_slug,
            )
            return

        # Resolve the Stage 3 run id for lineage.
        with __import__("contextlib").nullcontext():
            pass
        # We don't need a special lookup — create_pending_run accepts
        # parent_run_id=None and the stash carries no Stage-3 link.
        stage4_run_id = await asyncio.to_thread(
            repo.create_pending_run,
            role_slug=role_slug,
            role_display=role_display,
            prompt_version=TYPE_ASSIGNER_PROMPT_VERSION,
            model_snapshot=GENERATION_MODEL,
            parent_run_id=None,
        )
        await asyncio.to_thread(repo.transition_run, stage4_run_id, status="running")

        # Batch + LLM call.
        agent = Stage4TypeAssignerAgent()
        batches = batch_skills(aggregated, batch_size=50)
        all_typed: list = []
        for batch_idx, batch in enumerate(batches, start=1):
            # Strip provenance fields the LLM doesn't need (roles,
            # unassigned_from) — keep skill_id + name.
            llm_input = [
                {"skill_id": s["skill_id"], "name": s["name"]} for s in batch
            ]
            typed = await agent.assign_types(skills=llm_input)
            logger.info(
                "[v3 stage4 run %s] batch %d/%d: typed %d skills",
                stage4_run_id, batch_idx, len(batches), len(typed),
            )
            all_typed.extend(typed)

        # Confidence routing — bucket sizes for the UI viewer.
        routing = route_by_confidence(all_typed)
        routing_summary = {
            "auto_approved": [s.skill_id for s in routing["auto_approved"]],
            "sample_review": [s.skill_id for s in routing["sample_review"]],
            "full_review":   [s.skill_id for s in routing["full_review"]],
        }

        validator_log = {"warnings": [], "errors": []}
        # Soft validator: every input skill should have one typing back.
        typed_ids = {s.skill_id for s in all_typed}
        missing = [s["skill_id"] for s in aggregated if s["skill_id"] not in typed_ids]
        if missing:
            validator_log["warnings"].append(
                {
                    "code": "skills_missing_from_typing",
                    "message": (
                        f"{len(missing)} skill(s) sent to LLM but not "
                        f"returned: {missing[:5]}{'...' if len(missing) > 5 else ''}"
                    ),
                }
            )

        final_status = "approved"  # No errors -> approved (headless)
        if validator_log.get("errors"):
            final_status = "failed"

        await asyncio.to_thread(
            repo.complete_typing_run,
            stage4_run_id,
            typed_skills=[s.model_dump() for s in all_typed],
            routing=routing_summary,
            validator_log=validator_log,
            final_status=final_status,
        )
        logger.info(
            "[v3 stage4 run %s] typing for %s: status=%s "
            "(%d typed, auto=%d sample=%d full=%d)",
            stage4_run_id,
            role_display,
            final_status,
            len(all_typed),
            len(routing["auto_approved"]),
            len(routing["sample_review"]),
            len(routing["full_review"]),
        )
        # Cascade: fire Stage 5 (skill placement) when Stage 4 succeeds.
        # Stage 4 is headless — no human approve path — so this is the
        # only trigger.
        if final_status == "approved":
            asyncio.create_task(run_stage_5(role_slug))
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("[v3 stage4] run for role %r failed", role_slug)
        if stage4_run_id is not None:
            try:
                await asyncio.to_thread(
                    repo.fail_run,
                    stage4_run_id,
                    error_message=f"{type(exc).__name__}: {exc}\n\n{tb}",
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[v3 stage4 run %s] could not even mark failed",
                    stage4_run_id,
                )


# ── Stage 5 runner (skill placement — headless) ───────────────────────────


async def run_stage_5(role_slug: str) -> None:
    """Place every typed skill into the role's locked dimensions.

    Lifecycle (headless — no awaiting_review state, no review_queue row):
      pending → running → approved (placement succeeded; warnings surface
                                      in the UI for spot-check)
                       → failed   (LLM exception or empty input)

    Always terminal — never raises out of this function.
    """
    import asyncio

    stage5_run_id: uuid.UUID | None = None
    try:
        # Resolve inputs.
        role_card = await asyncio.to_thread(repo.get_role_card_for_role, role_slug)
        if role_card is None:
            logger.warning(
                "[v3 stage5] role %r has no approved role card; skipping",
                role_slug,
            )
            return
        role_display = role_card.get("canonical_name") or role_slug

        # Stage 3 locked dims — needed as the placement target.
        s3 = await asyncio.to_thread(repo.get_stage3_artifacts_for_role, role_slug)
        if s3 is None or not s3.get("locked_dimensions"):
            logger.warning(
                "[v3 stage5] role %r has no Stage 3 locked dims; skipping",
                role_slug,
            )
            return
        locked_dims = s3["locked_dimensions"]

        # Stage 4 typed skills — what we're placing.
        typed_dicts = await asyncio.to_thread(
            repo.get_latest_typed_skills_for_role, role_slug
        )
        if not typed_dicts:
            logger.warning(
                "[v3 stage5] role %r has no Stage 4 typed skills; skipping",
                role_slug,
            )
            return
        typed_skills = [TypedSkill.model_validate(d) for d in typed_dicts]

        stage5_run_id = await asyncio.to_thread(
            repo.create_pending_run,
            role_slug=role_slug,
            role_display=role_display,
            prompt_version=PLACER_PROMPT_VERSION,
            model_snapshot=GENERATION_MODEL,
            parent_run_id=None,
        )
        await asyncio.to_thread(repo.transition_run, stage5_run_id, status="running")

        # LLM placement — one call for the whole batch.
        agent = Stage5PlacerAgent()
        placed_skills = await agent.place_skills(
            typed_skills=typed_skills,
            role_card=role_card,
            locked_dimensions=locked_dims,
        )

        # Deterministic validators — run all four; aggregate warnings.
        embedder = make_default_embedder()
        dims_by_id = {d["tentative_id"]: d for d in locked_dims}
        validator_warnings: list[dict] = []
        validator_errors: list[dict] = []
        typed_by_id = {t.skill_id: t for t in typed_skills}

        # Check 1: embedding cross-check — per skill.
        for placed in placed_skills:
            findings = await embedding_cross_check(
                skill=placed, dims=locked_dims, embedder=embedder,
            )
            for f in findings:
                if f.get("level") == "error":
                    validator_errors.append(f)
                else:
                    validator_warnings.append(f)

        # Check 2: type-dim consistency — per skill.
        for placed in placed_skills:
            typed = typed_by_id.get(placed.skill_id)
            if typed is None:
                continue
            for f in type_dim_consistency(
                typed=typed, placed=placed, dims_by_id=dims_by_id,
            ):
                if f.get("level") == "error":
                    validator_errors.append(f)
                else:
                    validator_warnings.append(f)

        # Check 2b: Tool ↔ Standard cross-mismatch (egregious-only).
        for placed in placed_skills:
            typed = typed_by_id.get(placed.skill_id)
            if typed is None:
                continue
            for f in tool_standard_cross_mismatch(
                typed=typed, placed=placed, dims_by_id=dims_by_id,
            ):
                if f.get("level") == "error":
                    validator_errors.append(f)
                else:
                    validator_warnings.append(f)

        # Check 3: cross-skill consistency — across the batch.
        for f in cross_skill_consistency(typed=typed_skills, placed=placed_skills):
            (validator_errors if f.get("level") == "error" else validator_warnings).append(f)

        # Check 4: secondary-dim audit — per skill. Stage 6 will populate
        # parent_skill_dims; for now this is empty so no findings.
        parent_skill_dims_by_skill_id: dict[str, list[str]] = {}
        for placed in placed_skills:
            for f in secondary_dim_audit(
                placed=placed,
                parent_skill_dims_by_skill_id=parent_skill_dims_by_skill_id,
            ):
                (validator_errors if f.get("level") == "error" else validator_warnings).append(f)

        validator_log = {"errors": validator_errors, "warnings": validator_warnings}

        # Confidence routing — buckets surface in the UI for spot-check.
        # Reuse the same boundaries as Stage 4 (>0.9 auto, 0.7-0.9 sample,
        # <0.7 full). PlacedSkill.placement_confidence drives the bucket.
        from skill_library_v3.confidence_routing import (
            HIGH_BAND_BOTTOM, LOW_BAND_TOP,
        )
        auto_ids: list[str] = []
        sample_ids: list[str] = []
        full_ids: list[str] = []
        for p in placed_skills:
            if p.placement_confidence < LOW_BAND_TOP:
                full_ids.append(p.skill_id)
            elif p.placement_confidence > HIGH_BAND_BOTTOM:
                auto_ids.append(p.skill_id)
            else:
                sample_ids.append(p.skill_id)
        # Embedding mismatches and cross-skill outliers force review
        # regardless of confidence — promote them to full_review.
        force_review_ids = {
            f["skill_id"]
            for f in (*validator_warnings, *validator_errors)
            if f.get("code") in {"embedding_mismatch", "cross_skill_outlier"}
            and f.get("skill_id")
        }
        for sid in force_review_ids:
            if sid in auto_ids:
                auto_ids.remove(sid)
            if sid in sample_ids:
                sample_ids.remove(sid)
            if sid not in full_ids:
                full_ids.append(sid)

        routing = {
            "auto_approved": auto_ids,
            "sample_review": sample_ids,
            "full_review": full_ids,
        }

        final_status = "failed" if validator_errors else "approved"

        await asyncio.to_thread(
            repo.complete_placement_run,
            stage5_run_id,
            placed_skills=[p.model_dump() for p in placed_skills],
            routing=routing,
            validator_log=validator_log,
            final_status=final_status,
        )
        logger.info(
            "[v3 stage5 run %s] placement for %s: status=%s "
            "(%d placed, auto=%d sample=%d full=%d, %d warning, %d error)",
            stage5_run_id,
            role_display,
            final_status,
            len(placed_skills),
            len(auto_ids), len(sample_ids), len(full_ids),
            len(validator_warnings), len(validator_errors),
        )
        # Cascade: fire Stage 6 (containment) when Stage 5 succeeds.
        if final_status == "approved":
            asyncio.create_task(run_stage_6(role_slug))
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("[v3 stage5] run for role %r failed", role_slug)
        if stage5_run_id is not None:
            try:
                await asyncio.to_thread(
                    repo.fail_run,
                    stage5_run_id,
                    error_message=f"{type(exc).__name__}: {exc}\n\n{tb}",
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[v3 stage5 run %s] could not even mark failed",
                    stage5_run_id,
                )


# ── Stage 6 runner (containment resolution) ───────────────────────────────


async def run_stage_6(role_slug: str) -> None:
    """Resolve relationships for every typed+placed skill in this role.

    Lifecycle:
      pending → running → approved          (no errors)
                       → awaiting_review   (cycles or type-rule errors)
                       → failed            (LLM exception)

    For each skill:
      1. Build top-K similar candidates from the role's other skills.
      2. Call Stage6ContainmentAgent.resolve(typed, placed, candidates).
      3. Aggregate one SkillRelationships per skill.

    After all skills:
      4. Run all five validators (DAG, three type-rules, suppress sym).
      5. errors -> awaiting_review; clean -> approved.
    """
    import asyncio

    stage6_run_id: uuid.UUID | None = None
    try:
        # Inputs.
        role_card = await asyncio.to_thread(repo.get_role_card_for_role, role_slug)
        if role_card is None:
            logger.warning(
                "[v3 stage6] role %r has no approved role card; skipping",
                role_slug,
            )
            return
        role_display = role_card.get("canonical_name") or role_slug

        typed_dicts = await asyncio.to_thread(
            repo.get_latest_typed_skills_for_role, role_slug
        )
        if not typed_dicts:
            logger.warning(
                "[v3 stage6] role %r has no Stage 4 typed skills; skipping",
                role_slug,
            )
            return
        placed_dicts = await asyncio.to_thread(
            repo.get_latest_placed_skills_for_role, role_slug
        )
        if not placed_dicts:
            logger.warning(
                "[v3 stage6] role %r has no Stage 5 placed skills; skipping",
                role_slug,
            )
            return

        typed_skills = [TypedSkill.model_validate(d) for d in typed_dicts]
        placed_skills = [PlacedSkill.model_validate(d) for d in placed_dicts]
        placed_by_id = {p.skill_id: p for p in placed_skills}

        stage6_run_id = await asyncio.to_thread(
            repo.create_pending_run,
            role_slug=role_slug,
            role_display=role_display,
            prompt_version=CONTAINMENT_PROMPT_VERSION,
            model_snapshot=GENERATION_MODEL,
            parent_run_id=None,
        )
        await asyncio.to_thread(repo.transition_run, stage6_run_id, status="running")

        # Build the candidate-skill list for similarity lookups.
        candidates_pool = [
            {"skill_id": t.skill_id, "name": t.name} for t in typed_skills
        ]
        embedder = make_default_embedder()
        agent = Stage6ContainmentAgent()
        relationships: list = []
        for typed in typed_skills:
            placed = placed_by_id.get(typed.skill_id)
            if placed is None:
                # Skill typed but not placed — defensive skip.
                continue
            target = {"skill_id": typed.skill_id, "name": typed.name}
            top_k = await top_k_similar_skills(
                target=target, candidates=candidates_pool,
                embedder=embedder, k=20,
            )
            rel = await agent.resolve(
                typed=typed, placed=placed, candidates=top_k,
            )
            relationships.append(rel)

        # Run all five validators.
        validator_errors: list[dict] = []
        validator_warnings: list[dict] = []
        for f in validate_dag_no_cycles(relationships):
            (validator_errors if f.get("level") == "error" else validator_warnings).append(f)
        for f in service_has_platform_parent(
            typed_skills=typed_skills, relationships=relationships,
        ):
            (validator_errors if f.get("level") == "error" else validator_warnings).append(f)
        for f in library_has_lang_or_framework_parent(
            typed_skills=typed_skills, relationships=relationships,
        ):
            (validator_errors if f.get("level") == "error" else validator_warnings).append(f)
        for f in architecture_has_no_parents(
            typed_skills=typed_skills, relationships=relationships,
        ):
            (validator_errors if f.get("level") == "error" else validator_warnings).append(f)
        for f in suppress_symmetry(relationships=relationships):
            (validator_errors if f.get("level") == "error" else validator_warnings).append(f)
        validator_log = {"errors": validator_errors, "warnings": validator_warnings}

        # Errors require human gate; otherwise auto-approve.
        if validator_errors:
            final_status = "awaiting_review"
        else:
            final_status = "approved"

        await asyncio.to_thread(
            repo.complete_containment_run,
            stage6_run_id,
            relationships=[r.model_dump() for r in relationships],
            validator_log=validator_log,
            final_status=final_status,
        )
        logger.info(
            "[v3 stage6 run %s] containment for %s: status=%s "
            "(%d skills, %d warning, %d error)",
            stage6_run_id, role_display, final_status,
            len(relationships), len(validator_warnings), len(validator_errors),
        )
        if final_status == "approved":
            asyncio.create_task(run_stage_7(role_slug))
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("[v3 stage6] run for role %r failed", role_slug)
        if stage6_run_id is not None:
            try:
                await asyncio.to_thread(
                    repo.fail_run,
                    stage6_run_id,
                    error_message=f"{type(exc).__name__}: {exc}\n\n{tb}",
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[v3 stage6 run %s] could not even mark failed",
                    stage6_run_id,
                )


async def run_stage_6_regenerate(new_run_id: uuid.UUID) -> None:
    """Drive a freshly-created Stage 6 pending run inserted by
    ``repo.reject_and_create_regenerate``. Same shape as ``run_stage_6``
    but starts from a pending row already in the DB."""
    import asyncio

    try:
        meta = await asyncio.to_thread(repo.get_run_with_review, new_run_id)
        if meta is None:
            await asyncio.to_thread(
                repo.fail_run,
                new_run_id,
                error_message="Stage 6 regenerate: pending run not found",
            )
            return
        role_slug = meta["role_slug"]
        role_card = await asyncio.to_thread(repo.get_role_card_for_role, role_slug)
        if role_card is None:
            await asyncio.to_thread(
                repo.fail_run,
                new_run_id,
                error_message="Stage 6 regenerate: role card missing",
            )
            return
        role_display = role_card.get("canonical_name") or role_slug

        typed_dicts = await asyncio.to_thread(
            repo.get_latest_typed_skills_for_role, role_slug
        )
        placed_dicts = await asyncio.to_thread(
            repo.get_latest_placed_skills_for_role, role_slug
        )
        if not typed_dicts or not placed_dicts:
            await asyncio.to_thread(
                repo.fail_run,
                new_run_id,
                error_message="Stage 6 regenerate: missing typed or placed skills",
            )
            return

        typed_skills = [TypedSkill.model_validate(d) for d in typed_dicts]
        placed_skills = [PlacedSkill.model_validate(d) for d in placed_dicts]
        placed_by_id = {p.skill_id: p for p in placed_skills}

        await asyncio.to_thread(repo.transition_run, new_run_id, status="running")

        candidates_pool = [
            {"skill_id": t.skill_id, "name": t.name} for t in typed_skills
        ]
        embedder = make_default_embedder()
        agent = Stage6ContainmentAgent()
        relationships: list = []
        for typed in typed_skills:
            placed = placed_by_id.get(typed.skill_id)
            if placed is None:
                continue
            target = {"skill_id": typed.skill_id, "name": typed.name}
            top_k = await top_k_similar_skills(
                target=target, candidates=candidates_pool,
                embedder=embedder, k=20,
            )
            rel = await agent.resolve(typed=typed, placed=placed, candidates=top_k)
            relationships.append(rel)

        validator_errors: list[dict] = []
        validator_warnings: list[dict] = []
        for f in validate_dag_no_cycles(relationships):
            (validator_errors if f.get("level") == "error" else validator_warnings).append(f)
        for f in service_has_platform_parent(typed_skills=typed_skills, relationships=relationships):
            (validator_errors if f.get("level") == "error" else validator_warnings).append(f)
        for f in library_has_lang_or_framework_parent(typed_skills=typed_skills, relationships=relationships):
            (validator_errors if f.get("level") == "error" else validator_warnings).append(f)
        for f in architecture_has_no_parents(typed_skills=typed_skills, relationships=relationships):
            (validator_errors if f.get("level") == "error" else validator_warnings).append(f)
        for f in suppress_symmetry(relationships=relationships):
            (validator_errors if f.get("level") == "error" else validator_warnings).append(f)
        validator_log = {"errors": validator_errors, "warnings": validator_warnings}

        final_status = "awaiting_review" if validator_errors else "approved"
        await asyncio.to_thread(
            repo.complete_containment_run,
            new_run_id,
            relationships=[r.model_dump() for r in relationships],
            validator_log=validator_log,
            final_status=final_status,
        )
        logger.info(
            "[v3 stage6 regen %s] containment for %s: status=%s",
            new_run_id, role_slug, final_status,
        )
        if final_status == "approved":
            asyncio.create_task(run_stage_7(role_slug))
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("[v3 stage6 regen %s] failed", new_run_id)
        try:
            await asyncio.to_thread(
                repo.fail_run,
                new_run_id,
                error_message=f"{type(exc).__name__}: {exc}\n\n{tb}",
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "[v3 stage6 regen %s] could not even mark failed", new_run_id,
            )


# ── Stage 7 — enrichment ────────────────────────────────────────────────


async def _enrich_one_skill(skill: dict) -> SkillEnrichment | Exception:
    """Run all five enrichment agents on one skill in parallel.

    Per-agent failures don't poison the others — ``return_exceptions=True``
    means a single agent's exception is returned in-line, the runner
    aggregates whichever agents succeeded into the validator_log as a
    warning, and the skill is dropped from the enrichment list (to avoid
    persisting a half-filled record).
    """
    import asyncio

    agents = (
        Stage7VendorLicenseAgent(),
        Stage7MaturityAgent(),
        Stage7ContextKeywordsAgent(),
        Stage7AmbiguityAgent(),
        Stage7VersioningAgent(),
    )
    results = await asyncio.gather(
        agents[0].enrich(skill=skill),
        agents[1].enrich(skill=skill),
        agents[2].enrich(skill=skill),
        agents[3].enrich(skill=skill),
        agents[4].enrich(skill=skill),
        return_exceptions=True,
    )
    for r in results:
        if isinstance(r, Exception):
            return r
    vendor_license, maturity, ctx_kw, ambiguity, versioning = results
    return SkillEnrichment(
        skill_id=skill["skill_id"],
        vendor_license=vendor_license,
        maturity=maturity,
        context_keywords=ctx_kw,
        ambiguity=ambiguity,
        versioning=versioning,
    )


async def _execute_enrichment(
    *, run_id: uuid.UUID, role_slug: str, role_display: str,
    typed_dicts: list[dict],
) -> None:
    """Shared body for ``run_stage_7`` and ``run_stage_7_regenerate``.

    Already-validated inputs in: typed skills with skill_id+name+type+
    subtype+primary_dimension. Caller owns the run row's pending → running
    transition path before invoking us.
    """
    import asyncio

    enrichments: list[SkillEnrichment] = []
    failures: list[dict] = []
    for typed in typed_dicts:
        skill_id = typed.get("skill_id")
        if not skill_id:
            continue
        result = await _enrich_one_skill(typed)
        if isinstance(result, Exception):
            failures.append({
                "code": "agent_exception",
                "level": "warning",
                "skill_id": skill_id,
                "message": f"{type(result).__name__}: {result}",
            })
            continue
        enrichments.append(result)

    catalog_skill_ids = {t["skill_id"] for t in typed_dicts if t.get("skill_id")}
    warnings = list(failures)
    warnings.extend(validate_ambiguity_in_catalog(enrichments, catalog_skill_ids))
    warnings.extend(validate_vendor_recognized(enrichments, set()))
    warnings.extend(validate_maturity_distribution(enrichments))
    validator_log: dict = {"errors": [], "warnings": warnings}

    final_status = decide_enrichment_status(
        enrichments=enrichments, validator_log=validator_log,
    )
    await asyncio.to_thread(
        repo.complete_enrichment_run,
        run_id,
        enrichments=[e.model_dump() for e in enrichments],
        validator_log=validator_log,
        final_status=final_status,
    )
    logger.info(
        "[v3 stage7 run %s] enrichment for %s: status=%s "
        "(%d skills, %d warning, %d failure)",
        run_id, role_display, final_status,
        len(enrichments), len(warnings) - len(failures), len(failures),
    )
    if final_status == "approved":
        asyncio.create_task(run_stage_8(role_slug))


async def run_stage_7(role_slug: str) -> None:
    """Enrich every typed skill of a role with vendor/license, maturity,
    context keywords, ambiguity, versioning. Five agents per skill run
    in parallel via ``asyncio.gather``.

    Lifecycle:
      pending → running → approved          (no flags)
                       → awaiting_review   (any ambiguity or deprecated)
                       → failed            (LLM exception in shared path)
    """
    import asyncio

    stage7_run_id: uuid.UUID | None = None
    try:
        role_card = await asyncio.to_thread(repo.get_role_card_for_role, role_slug)
        if role_card is None:
            logger.warning(
                "[v3 stage7] role %r has no approved role card; skipping",
                role_slug,
            )
            return
        role_display = role_card.get("canonical_name") or role_slug

        typed_dicts = await asyncio.to_thread(
            repo.get_latest_typed_skills_for_role, role_slug
        )
        if not typed_dicts:
            logger.warning(
                "[v3 stage7] role %r has no Stage 4 typed skills; skipping",
                role_slug,
            )
            return

        stage7_run_id = await asyncio.to_thread(
            repo.create_pending_run,
            role_slug=role_slug,
            role_display=role_display,
            prompt_version=ENRICHMENT_PROMPT_VERSION,
            model_snapshot=GENERATION_MODEL,
            parent_run_id=None,
        )
        await asyncio.to_thread(repo.transition_run, stage7_run_id, status="running")

        await _execute_enrichment(
            run_id=stage7_run_id,
            role_slug=role_slug,
            role_display=role_display,
            typed_dicts=typed_dicts,
        )
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("[v3 stage7] run for role %r failed", role_slug)
        if stage7_run_id is not None:
            try:
                await asyncio.to_thread(
                    repo.fail_run,
                    stage7_run_id,
                    error_message=f"{type(exc).__name__}: {exc}\n\n{tb}",
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[v3 stage7 run %s] could not even mark failed",
                    stage7_run_id,
                )


async def run_stage_7_regenerate(new_run_id: uuid.UUID) -> None:
    """Drive a freshly-created Stage 7 pending run from
    ``repo.reject_and_create_regenerate``."""
    import asyncio

    try:
        meta = await asyncio.to_thread(repo.get_run_with_review, new_run_id)
        if meta is None:
            await asyncio.to_thread(
                repo.fail_run,
                new_run_id,
                error_message="Stage 7 regenerate: pending run not found",
            )
            return
        role_slug = meta["role_slug"]
        role_card = await asyncio.to_thread(repo.get_role_card_for_role, role_slug)
        if role_card is None:
            await asyncio.to_thread(
                repo.fail_run,
                new_run_id,
                error_message="Stage 7 regenerate: role card missing",
            )
            return
        role_display = role_card.get("canonical_name") or role_slug

        typed_dicts = await asyncio.to_thread(
            repo.get_latest_typed_skills_for_role, role_slug
        )
        if not typed_dicts:
            await asyncio.to_thread(
                repo.fail_run,
                new_run_id,
                error_message="Stage 7 regenerate: missing typed skills",
            )
            return

        await asyncio.to_thread(repo.transition_run, new_run_id, status="running")
        await _execute_enrichment(
            run_id=new_run_id,
            role_slug=role_slug,
            role_display=role_display,
            typed_dicts=typed_dicts,
        )
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("[v3 stage7 regen %s] failed", new_run_id)
        try:
            await asyncio.to_thread(
                repo.fail_run,
                new_run_id,
                error_message=f"{type(exc).__name__}: {exc}\n\n{tb}",
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "[v3 stage7 regen %s] could not even mark failed", new_run_id,
            )


# ── Stage 8 — catalog load ──────────────────────────────────────────────


_STAGE8_PROMPT_VERSION = "stage8_load_v1.0"


async def run_stage_8(role_slug: str) -> None:
    """Build a CatalogPayload from the role's approved Stage 1-7 outputs
    and persist it into the canonical relational tables.

    Lifecycle:
      pending → running → approved   (load completed without errors)
                       → failed     (load raised an exception)

    No human gate by default — this is the "make the library queryable"
    stage. If something goes wrong, the JSON outputs on stages 1-7 are
    still on disk and can be replayed once the issue is fixed.
    """
    import asyncio

    stage8_run_id: uuid.UUID | None = None
    try:
        role_card = await asyncio.to_thread(repo.get_role_card_for_role, role_slug)
        if role_card is None:
            logger.warning(
                "[v3 stage8] role %r has no approved role card; skipping",
                role_slug,
            )
            return
        role_display = role_card.get("canonical_name") or role_slug

        # Pull all approved Stage 1-7 outputs in parallel.
        dimensions = await asyncio.to_thread(
            repo.get_latest_dimensions_for_role, role_slug
        )
        typed = await asyncio.to_thread(
            repo.get_latest_typed_skills_for_role, role_slug
        )
        placed = await asyncio.to_thread(
            repo.get_latest_placed_skills_for_role, role_slug
        )
        relationships = await asyncio.to_thread(
            repo.get_latest_containment_for_role, role_slug
        )
        enrichments = await asyncio.to_thread(
            repo.get_latest_enrichment_for_role, role_slug
        )
        if not all([dimensions, typed, placed, relationships, enrichments]):
            logger.warning(
                "[v3 stage8] role %r missing prior-stage outputs; skipping. "
                "dim=%s typed=%s placed=%s rel=%s enrich=%s",
                role_slug,
                bool(dimensions), bool(typed), bool(placed),
                bool(relationships), bool(enrichments),
            )
            return

        # Stage 3 produces locked_dimensions; Stage 2's get_latest_dimensions
        # returns the candidate_dimensions list.  We want the locked version
        # for the loader (post-reconciliation). Pull from Stage 3 directly.
        recon_dims = await asyncio.to_thread(
            _get_latest_locked_dimensions, role_slug
        )
        locked = recon_dims or dimensions

        stage8_run_id = await asyncio.to_thread(
            repo.create_pending_run,
            role_slug=role_slug,
            role_display=role_display,
            prompt_version=_STAGE8_PROMPT_VERSION,
            model_snapshot="deterministic",
            parent_run_id=None,
        )
        await asyncio.to_thread(repo.transition_run, stage8_run_id, status="running")

        payload = build_catalog_payload(
            role_slug=role_slug,
            role_card=role_card,
            locked_dimensions=locked,
            typed_skills=typed,
            placed_skills=placed,
            relationships=relationships,
            enrichments=enrichments,
        )
        load_result = await load_catalog_for_role(payload)

        validator_log = {"errors": load_result.errors, "warnings": []}
        final_status = "failed" if load_result.errors else "approved"

        await asyncio.to_thread(
            repo.complete_load_run,
            stage8_run_id,
            load_summary=load_result.to_dict(),
            validator_log=validator_log,
            final_status=final_status,
        )
        logger.info(
            "[v3 stage8 run %s] catalog load for %s: status=%s "
            "(skills=%d aliases=%d cats=%d sub_cats=%d dims=%d "
            "rel=%d tags=%d)",
            stage8_run_id, role_display, final_status,
            load_result.skills_written, load_result.aliases_written,
            load_result.categories_written, load_result.sub_categories_written,
            load_result.dimensions_written, load_result.relationships_written,
            load_result.tags_written,
        )
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.exception("[v3 stage8] run for role %r failed", role_slug)
        if stage8_run_id is not None:
            try:
                await asyncio.to_thread(
                    repo.fail_run,
                    stage8_run_id,
                    error_message=f"{type(exc).__name__}: {exc}\n\n{tb}",
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[v3 stage8 run %s] could not even mark failed",
                    stage8_run_id,
                )


def _get_latest_locked_dimensions(role_slug: str) -> list[dict] | None:
    """Pull the most recent approved Stage 3 reconciliation's
    ``locked_dimensions`` for this role. Stage 8 wants the post-recon
    list (after merges/splits) rather than the raw Stage 2 candidates.
    Pass the LIKE pattern as a parameter so psycopg2 doesn't interpret
    its '%' as another placeholder marker.
    """
    from skill_library_v3.db.connection import connect

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT planner_output
                  FROM v2_run_log
                 WHERE role_id = %s
                   AND prompt_version LIKE %s
                   AND status = 'approved'
                 ORDER BY completed_at DESC
                 LIMIT 1
                """,
                (role_slug, "stage3_recon_%"),
            )
            row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    body = row[0]
    if not isinstance(body, dict):
        return None
    locked = body.get("locked_dimensions")
    return locked if isinstance(locked, list) else None
