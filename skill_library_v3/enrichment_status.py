"""Pure-function lifecycle decider for Stage 7 (enrichment).

Autonomy-tightened rules — only gate when the catalog has a real signal
that needs human input. Single-skill ambiguity is common (the LLM is
thorough about flagging anything that *could* collide); only treat it
as a real gate when the LLM listed multiple candidates it could be
confused with, or when more than 10% of skills in the role were flagged.

Rules:
  * Any error in validator_log -> 'failed' (terminal — fix via regenerate)
  * >= 1 enrichment with maturity='deprecated' -> 'awaiting_review'
    (real "should we keep this skill" decision)
  * Any single skill with len(confused_with) >= 2 -> 'awaiting_review'
    (LLM is unsure about an actual collision)
  * Aggregate ambiguity rate > 10% of skills -> 'awaiting_review'
  * Otherwise -> 'approved' (auto)

Warnings in validator_log are advisory and never change status.
"""

from __future__ import annotations

from typing import Iterable, Literal

from skill_library_v3.schemas.enrichment import SkillEnrichment


EnrichmentStatus = Literal["approved", "awaiting_review", "failed"]


_AMBIGUITY_RATE_THRESHOLD = 0.10
_MULTI_CONFUSED_THRESHOLD = 2


def decide_enrichment_status(
    *,
    enrichments: Iterable[SkillEnrichment],
    validator_log: dict,
) -> EnrichmentStatus:
    if validator_log.get("errors"):
        return "failed"

    enrichments_list = list(enrichments)
    total = len(enrichments_list)
    if total == 0:
        return "approved"

    flagged = 0
    for e in enrichments_list:
        if e.maturity.maturity == "deprecated":
            return "awaiting_review"
        if e.ambiguity.ambiguity_flag:
            flagged += 1
            if len(e.ambiguity.confused_with) >= _MULTI_CONFUSED_THRESHOLD:
                return "awaiting_review"

    if (flagged / total) > _AMBIGUITY_RATE_THRESHOLD:
        return "awaiting_review"
    return "approved"
