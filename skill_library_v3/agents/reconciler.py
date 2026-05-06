"""Stage 3b reconciler agent — decides MERGE / SPLIT / KEEP_SEPARATE for
one flagged dim pair from sub-stage 3a.

Subclasses :class:`skill_library_v2.agents.base.BaseLLMAgent` to inherit
JSON parse + corrective retry + tier client selection. Tests subclass
this agent and override ``call_json`` to verify prompt construction
without hitting Azure (see ``tests/test_reconciler_agent.py``).
"""

from __future__ import annotations

from skill_library_v2.agents.base import BaseLLMAgent
from skill_library_v3.prompts.reconciler import (
    RECONCILER_PROMPT_VERSION,
    RECONCILER_SYSTEM_PROMPT,
    RECONCILER_USER_TEMPLATE,
    format_exemplars,
)
from skill_library_v3.schemas.reconciliation import ReconciliationDecision


class Stage3ReconcilerAgent(BaseLLMAgent):
    """Decides one ReconciliationDecision per flagged dim pair.

    tier=generation -> gpt-5.4-mini (consistent with the rest of v3 per
    project memory). The decision is reasoning-shaped (compare two
    definitions, pick a category) but the generation tier handles it
    fine and keeps tier choice consistent across stages.
    """

    tier = "generation"

    def __init__(self) -> None:
        super().__init__(
            agent_name="stage3_reconciler",
            prompt_version=RECONCILER_PROMPT_VERSION,
        )

    async def reconcile(
        self,
        *,
        a_role: str,
        a_dim: dict,
        b_role: str,
        b_dim: dict,
        similarity: float,
    ) -> ReconciliationDecision:
        kind = "intra_role" if a_role == b_role else "cross_role"
        user_prompt = RECONCILER_USER_TEMPLATE.format(
            similarity=similarity,
            kind=kind,
            a_role=a_role,
            a_id=a_dim.get("tentative_id", ""),
            a_name=a_dim.get("name", ""),
            a_description=a_dim.get("description", ""),
            a_in_scope=a_dim.get("in_scope", ""),
            a_out_of_scope=a_dim.get("out_of_scope", ""),
            a_exemplars=format_exemplars(a_dim.get("exemplar_skills")),
            b_role=b_role,
            b_id=b_dim.get("tentative_id", ""),
            b_name=b_dim.get("name", ""),
            b_description=b_dim.get("description", ""),
            b_in_scope=b_dim.get("in_scope", ""),
            b_out_of_scope=b_dim.get("out_of_scope", ""),
            b_exemplars=format_exemplars(b_dim.get("exemplar_skills")),
        )
        return await self.call_json(
            system_prompt=RECONCILER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=ReconciliationDecision,
            temperature=0.1,
        )
