"""Stage 2.5a — coverage-matrix critic agent.

Single reasoning-tier LLM call per role. Takes (role_card,
current_dim_set, coverage_matrix), returns:

    {"covered": [axis_slug, ...],
     "missing": [axis_slug, ...],
     "proposed_dims": [{axis_slug, name, description, in_scope,
                        exemplar_skills, out_of_scope}, ...]}

The deterministic guard :func:`post_process_critic_response` runs AFTER
the LLM call. It enforces two rules that we don't ask the prompt to
enforce (because rejecting hallucinations is cheaper deterministically
than re-prompting):

  * ``len(exemplar_skills) < 4`` -> drop proposal (anti-abstraction).
  * ``axis_slug not in matrix`` -> drop proposal (off-checklist).

Surviving proposals get tagged with ``tentative_id = "d_critic_NN"`` and
``_origin = "stage_2_5_critic"`` so the Stage 2.5 orchestrator and
downstream stages can audit critic-sourced dims.
"""

from __future__ import annotations

from skill_library_v3.coverage_matrices import CoverageItem


def post_process_critic_response(
    raw: dict,
    *,
    matrix: list[CoverageItem],
) -> dict:
    """Apply the deterministic guards to a critic LLM response.

    Returns a new dict with the same shape as ``raw`` plus a ``dropped``
    key listing why each rejected proposal was dropped. Does not mutate
    ``raw``.
    """
    allowed_slugs = {item.slug for item in matrix}
    proposals = list(raw.get("proposed_dims") or [])

    surviving: list[dict] = []
    dropped: list[dict] = []
    next_seq = 1

    for proposal in proposals:
        exemplars = proposal.get("exemplar_skills") or []
        if len(exemplars) < 4:
            dropped.append(
                {
                    "axis_slug": proposal.get("axis_slug"),
                    "reason": "too_few_exemplars",
                    "exemplar_count": len(exemplars),
                }
            )
            continue

        axis_slug = proposal.get("axis_slug")
        if axis_slug not in allowed_slugs:
            dropped.append(
                {
                    "axis_slug": axis_slug,
                    "reason": "off_checklist_axis",
                }
            )
            continue

        tagged = dict(proposal)
        tagged["tentative_id"] = f"d_critic_{next_seq:02d}"
        tagged["_origin"] = "stage_2_5_critic"
        surviving.append(tagged)
        next_seq += 1

    return {
        "covered": list(raw.get("covered") or []),
        "missing": list(raw.get("missing") or []),
        "proposed_dims": surviving,
        "dropped": dropped,
    }


# ── Agent wrapper ──────────────────────────────────────────────────────────

try:  # pragma: no cover — exercised only when LLM clients are configured.
    from skill_library_v2.agents.base import BaseLLMAgent

    from skill_library_v3.prompts.dim_critic import (
        DIM_CRITIC_PROMPT_VERSION,
        DIM_CRITIC_SYSTEM_PROMPT,
        DIM_CRITIC_USER_TEMPLATE,
        format_dim_summary_block,
        format_coverage_matrix_block,
    )
    from skill_library_v3.schemas.dim_critic import CriticOutput

    class Stage25CriticAgent(BaseLLMAgent):
        """Coverage-matrix critic. One reasoning-tier call per role."""

        tier = "reasoning"

        def __init__(self) -> None:
            super().__init__(
                agent_name="stage_2_5_dim_critic",
                prompt_version=DIM_CRITIC_PROMPT_VERSION,
            )

        async def critique(
            self,
            *,
            role_name: str,
            role_card: dict,
            current_dimensions: list[dict],
            matrix: list[CoverageItem],
        ) -> dict:
            """Run one LLM call and return the post-processed response."""
            user_prompt = DIM_CRITIC_USER_TEMPLATE.format(
                role_name=role_name,
                role_definition=role_card.get("definition") or "",
                dim_summary_block=format_dim_summary_block(current_dimensions),
                coverage_matrix_block=format_coverage_matrix_block(matrix),
            )
            raw = await self.call_json(
                system_prompt=DIM_CRITIC_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=CriticOutput,
                temperature=0.2,
            )
            if hasattr(raw, "model_dump"):
                raw = raw.model_dump()
            return post_process_critic_response(raw, matrix=matrix)

except ImportError:
    # BaseLLMAgent or the critic prompt module isn't available. The
    # post_process_critic_response function above is still importable;
    # the runner-integration path is what wires the full agent.
    pass
