"""Aggregator – assembles all pipeline outputs into the final response.

Builds the trajectory-based global_skill_index: each skill gets an array
of traces showing every block it appears in, with action_verb, context,
co-dependencies, and metrics.
"""

from __future__ import annotations

from collections import defaultdict
from models import (
    CandidateDetails,
    BlockTagResult,
    GlobalParameters,
    ContextMetaTags,
    GlobalSkillEntry,
    SkillTrace,
    ResumeTaggingResponse,
    ZonedBlockPreview,
)


def build_global_skill_index(
    block_results: list[BlockTagResult],
) -> list[GlobalSkillEntry]:
    """Build trajectory-based skill index from per-block results.

    Each skill maps to a list of SkillTrace objects, one per block occurrence.
    Sorted by number of traces (most-used skills first).
    """
    skill_traces: dict[str, list[SkillTrace]] = defaultdict(list)

    for br in block_results:
        for entry in br.skills:
            key = entry.skill.strip().lower()
            skill_traces[key].append(
                SkillTrace(
                    provenance=br.block_name,
                    action_verb=entry.action_verb,
                    context=entry.context,
                    co_dependent_skills=entry.co_dependent_skills,
                    metric=entry.metric,
                )
            )

    # Keep original casing from first occurrence
    canonical_names: dict[str, str] = {}
    for br in block_results:
        for entry in br.skills:
            key = entry.skill.strip().lower()
            if key not in canonical_names:
                canonical_names[key] = entry.skill

    return sorted(
        [
            GlobalSkillEntry(
                skill=canonical_names.get(key, key),
                traces=traces,
            )
            for key, traces in skill_traces.items()
        ],
        key=lambda e: len(e.traces),
        reverse=True,
    )


def aggregate(
    candidate: CandidateDetails,
    block_results: list[BlockTagResult],
    global_params: GlobalParameters,
    context_meta_tags: ContextMetaTags,
    reasoning_log: list[str],
    parsed_text: str = "",
    zoned_blocks: list | None = None,
) -> ResumeTaggingResponse:
    """Assemble the final response."""
    zoned_preview = []
    if zoned_blocks:
        zoned_preview = [
            ZonedBlockPreview(
                block_name=b.block_name,
                block_type=b.block_type,
                raw_text=b.raw_text,
            )
            for b in zoned_blocks
        ]
    return ResumeTaggingResponse(
        candidate=candidate,
        context_meta_tags=context_meta_tags,
        global_skill_index=build_global_skill_index(block_results),
        blocks=block_results,
        global_parameters=global_params,
        reasoning_log=reasoning_log,
        parsed_text=parsed_text,
        zoned_blocks_preview=zoned_preview,
    )
