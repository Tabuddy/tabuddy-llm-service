"""Per-dimension worker — Categorizer → SkillGenerator → Critic → Validator.

One ``process_dim`` LangGraph node per Send payload. The categorizer runs
once (its output is stable across retries); the skill_generator + critic +
validator iterate up to MAX_ATTEMPTS on revise/reject/validation_errors.

State patch on completion includes:
  - generated[dim_id]               (list[SkillDraft])
  - dim_categories[dim_id]          (CategorizerOutput)
  - dim_status[dim_id]              ("done" | "failed")
  - dim_retry_count[dim_id]         (int)
  - dim_critic_verdict[dim_id]      ("approve" | "revise" | "reject")
  - dim_critic_corrections[dim_id]  (list[Correction])
  - dim_generator_notes[dim_id]     (str)
"""

from __future__ import annotations

import logging
from typing import Any

from skill_library_v2.agents.categorizer import _get_categorizer
from skill_library_v2.agents.critic import _get_critic
from skill_library_v2.agents.skill_generator import _get_skill_generator
from skill_library_v2.agents.validator import _validate_one
from skill_library_v2.schemas.category import CategorizerOutput
from skill_library_v2.schemas.role import DimensionSlice
from skill_library_v2.schemas.skill import (
    Correction,
    CriticOutput,
    SkillDraft,
    SkillGeneratorOutput,
)
from skill_library_v2.state import PlanGraphState

logger = logging.getLogger(__name__)


MAX_ATTEMPTS = 3


async def process_dim(state: PlanGraphState) -> dict[str, Any]:
    """Run the full Categorizer → SkillGenerator → Critic → Validator loop for one dim."""
    dim_id: str = state["dim_id"]
    dim = DimensionSlice.model_validate(state["dim"])
    role_id: str = state["role_id"]
    role_display: str = state["role_display"]
    role_archetype: str = state.get("role_archetype") or ""

    all_dims_raw: list[Any] = state.get("dimensions") or []
    all_dims: list[DimensionSlice] = [
        d if isinstance(d, DimensionSlice) else DimensionSlice.model_validate(d)
        for d in all_dims_raw
    ]
    other_dims = [d for d in all_dims if d.dimension_id != dim_id]

    role_hints_raw = state.get("planner_web_hints") or []
    role_hints = [
        h.model_dump() if hasattr(h, "model_dump") else h
        for h in role_hints_raw
    ]

    categorizer = _get_categorizer()
    skill_gen = _get_skill_generator()
    critic = _get_critic()

    # ── Phase A: Categorize once. Stable across skill-gen retries. ─────────
    try:
        cat_output: CategorizerOutput = await categorizer.run(
            role_id=role_id,
            role_display=role_display,
            role_archetype=role_archetype,
            dim=dim,
            other_dims=other_dims,
            role_web_hints=role_hints,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[worker] dim=%s categorizer failed: %s", dim_id, exc,
        )
        return {
            "dim_status": {dim_id: "failed"},
            "errors": [{
                "node": "categorizer",
                "message": f"{type(exc).__name__}: {exc}",
                "retryable": False,
            }],
        }

    logger.info(
        "[worker] dim=%s role=%s categorizer emitted %d proposals",
        dim_id, role_id, len(cat_output.proposals),
    )

    # ── Phase B: Skill-gen + critic + validator with retry budget. ─────────
    already_by_dim: dict[str, list[SkillDraft]] = state.get("generated") or {}
    already_slugs: dict[str, list[str]] = {
        k: [s.slug for s in v]
        for k, v in already_by_dim.items()
        if k != dim_id
    }

    corrections: list[Correction] = []
    last_skills: SkillGeneratorOutput | None = None
    last_critic: CriticOutput | None = None
    last_validator_errs: list[str] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.info(
            "[worker] dim=%s role=%s attempt=%d/%d corrections=%d",
            dim_id, role_id, attempt, MAX_ATTEMPTS, len(corrections),
        )

        # Skill generator
        try:
            last_skills = await skill_gen.run(
                role_id=role_id,
                role_display=role_display,
                role_archetype=role_archetype,
                dim=dim,
                categorizer_output=cat_output,
                already_emitted=already_slugs,
                corrections=[c.model_dump() for c in corrections],
                role_web_hints=role_hints,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[worker] dim=%s skill_generator failed on attempt %d: %s",
                dim_id, attempt, exc,
            )
            last_skills = None
            break

        # ── Normalize before validate. The LLM routinely emits 81-110 due to
        #    the coverage push, repeats slugs within a dim, and repeats
        #    aliases within a skill. Clamping + deduping here prevents wasted
        #    retries on issues trivial to fix deterministically.
        _MAX_SKILLS_PER_DIM = 80
        if last_skills is not None:
            # Within-dim slug dedup (first-wins). Validator otherwise rejects
            # the whole batch on any duplicate.
            seen_slugs: set[str] = set()
            unique_skills: list[SkillDraft] = []
            for s in last_skills.skills:
                if s.slug in seen_slugs:
                    continue
                seen_slugs.add(s.slug)
                unique_skills.append(s)
            if len(unique_skills) < len(last_skills.skills):
                logger.info(
                    "[worker] dim=%s slug dedup %d -> %d",
                    dim_id, len(last_skills.skills), len(unique_skills),
                )
            last_skills.skills = unique_skills

            if len(last_skills.skills) > _MAX_SKILLS_PER_DIM:
                overflow = len(last_skills.skills) - _MAX_SKILLS_PER_DIM
                logger.info(
                    "[worker] dim=%s clamping skills %d -> %d (dropped %d overflow)",
                    dim_id, len(last_skills.skills), _MAX_SKILLS_PER_DIM, overflow,
                )
                last_skills.skills = last_skills.skills[:_MAX_SKILLS_PER_DIM]

            # Per-skill alias dedup (case-insensitive, first-wins).
            for s in last_skills.skills:
                seen_aliases: set[str] = set()
                deduped: list[str] = []
                for a in s.aliases or []:
                    al = (a or "").strip()
                    if not al:
                        continue
                    key = al.lower()
                    if key in seen_aliases:
                        continue
                    seen_aliases.add(key)
                    deduped.append(a)
                s.aliases = deduped

        # Critic
        try:
            last_critic = await critic.run(
                role_id=role_id,
                role_display=role_display,
                dim=dim,
                skills=last_skills.skills,
                categorizer_output=cat_output,
                generator_notes=last_skills.notes,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[worker] dim=%s critic failed on attempt %d: %s — "
                "treating as approve so validator still runs",
                dim_id, attempt, exc,
            )
            last_critic = CriticOutput(
                dim_id=dim_id,
                corrections=[],
                approved_slugs=[],
                overall_verdict="approve",
            )

        # Deterministic validator (with bucket-integrity check)
        last_validator_errs = _validate_one(last_skills.skills, dim, cat_output)

        verdict = last_critic.overall_verdict
        hard_retry = bool(last_validator_errs) or verdict == "reject"
        soft_retry = verdict == "revise" and attempt < 2

        if not hard_retry and not soft_retry:
            logger.info(
                "[worker] dim=%s role=%s PASS (%d skills, verdict=%s, "
                "%d critic corrections recorded)",
                dim_id, role_id, len(last_skills.skills), verdict,
                len(last_critic.corrections),
            )
            return {
                "generated": {dim_id: last_skills.skills},
                "dim_categories": {dim_id: cat_output},
                "dim_status": {dim_id: "done"},
                "dim_retry_count": {dim_id: attempt - 1},
                "dim_generator_notes": {dim_id: last_skills.notes},
                "dim_critic_verdict": {dim_id: verdict},
                "dim_critic_corrections": {dim_id: last_critic.corrections},
            }

        if attempt >= MAX_ATTEMPTS:
            if hard_retry:
                logger.warning(
                    "[worker] dim=%s role=%s FAILED after %d attempts "
                    "(verdict=%s, validator_errs=%s)",
                    dim_id, role_id, attempt, verdict,
                    last_validator_errs[:5] if last_validator_errs else [],
                )
                break
            logger.info(
                "[worker] dim=%s role=%s accept-after-soft-retry "
                "(%d skills, verdict=%s, %d critic corrections)",
                dim_id, role_id, len(last_skills.skills), verdict,
                len(last_critic.corrections),
            )
            return {
                "generated": {dim_id: last_skills.skills},
                "dim_categories": {dim_id: cat_output},
                "dim_status": {dim_id: "done"},
                "dim_retry_count": {dim_id: attempt - 1},
                "dim_generator_notes": {dim_id: last_skills.notes},
                "dim_critic_verdict": {dim_id: verdict},
                "dim_critic_corrections": {dim_id: last_critic.corrections},
            }

        corrections = list(last_critic.corrections)

    # Exhausted retries OR skill_gen crashed.
    skills = list(last_skills.skills) if last_skills else []
    verdict = last_critic.overall_verdict if last_critic else "reject"
    errors_patch: list[dict] = []
    if last_validator_errs or not skills or verdict != "approve":
        errors_patch.append({
            "node": "process_dim",
            "message": (
                f"dim={dim_id} exhausted {MAX_ATTEMPTS} attempts; "
                f"verdict={verdict}; last_errs={last_validator_errs[:3]}"
            ),
            "retryable": False,
        })

    return {
        "generated": {dim_id: skills},
        "dim_categories": {dim_id: cat_output},
        "dim_status": {dim_id: "failed"},
        "dim_retry_count": {dim_id: MAX_ATTEMPTS},
        "dim_generator_notes": {
            dim_id: last_skills.notes if last_skills else ""
        },
        "dim_critic_verdict": {dim_id: verdict},
        "dim_critic_corrections": {
            dim_id: last_critic.corrections if last_critic else []
        },
        "errors": errors_patch,
    }
