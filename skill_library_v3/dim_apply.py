"""Stage 3 sub-stage 3c: apply reconciliation decisions to produce the
final locked dimension list.

Pure function. No I/O, no LLM, no embeddings — just deterministic
transformation of (candidate_dims, decisions) -> (locked, merge_log,
split_log, unassigned_exemplars).

Algorithm:

  Walk decisions in order. Track which original tentative_ids have
  been "consumed" by a previous decision. For each pair:
    * KEEP_SEPARATE -> no-op; both originals stay
    * MERGE         -> consume both; emit one new merged dim that
                       pools both originals' in_scope, out_of_scope,
                       exemplar_skills (deduplicated, order-preserved)
    * SPLIT         -> consume both; emit the LLM's split_into dims
                       (each starts empty on exemplars; pool the
                       originals' exemplars into unassigned_exemplars
                       for Stage 4/5 to redistribute)

  After processing decisions, append every untouched candidate.

  Conflicts (a tentative_id referenced by two decisions): first-wins.
  The second decision is silently skipped.

  Defensive: a decision referencing an unknown tentative_id is silently
  skipped — guards against stale 3a results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from skill_library_v3.schemas.reconciliation import ReconciliationDecision


@dataclass
class ApplyResult:
    """Output of apply_decisions."""

    locked_dimensions: list[dict] = field(default_factory=list)
    merge_log: list[dict] = field(default_factory=list)
    split_log: list[dict] = field(default_factory=list)
    unassigned_exemplars: list[str] = field(default_factory=list)


def _split_csv(text: str | None) -> list[str]:
    """Parse a comma-delimited string into a clean list. Empty / None
    becomes []."""
    if not text:
        return []
    return [p.strip() for p in text.split(",") if p.strip()]


def _join_unique_csv(*values: str | None) -> str:
    """Join multiple comma-delimited strings into one, dedup
    case-insensitively, preserve order of first occurrence."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        for item in _split_csv(v):
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
    return ", ".join(out)


def _dedup_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        if not s or not s.strip():
            continue
        key = s.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def apply_decisions(
    *,
    candidate_dims: list[dict],
    pair_decisions: list[tuple[dict, ReconciliationDecision]],
) -> ApplyResult:
    """Apply 3b's reconciliation decisions to 3a's candidate dims.

    ``pair_decisions`` is an ordered list of (pair_dict, decision)
    tuples. Each pair_dict carries ``a_tentative_id`` and
    ``b_tentative_id`` keys (the same shape produced by 3a's
    ``compute_overlap_pairs``).
    """
    by_id: dict[str, dict] = {d["tentative_id"]: d for d in candidate_dims}
    consumed: set[str] = set()
    result = ApplyResult()
    next_merge_seq = 1
    next_split_seq = 1

    for pair, decision in pair_decisions:
        a_id = pair.get("a_tentative_id")
        b_id = pair.get("b_tentative_id")

        # Defensive: skip stale or malformed pairs.
        if a_id not in by_id or b_id not in by_id:
            continue
        # First-wins: a dim consumed by an earlier decision can't be
        # touched again. The second decision is silently skipped (its
        # reasoning still lives in the pair_decisions list for audit).
        if a_id in consumed or b_id in consumed:
            continue

        if decision.decision == "KEEP_SEPARATE":
            # Both dims stay as-is in the untouched-fall-through pass below.
            continue

        a = by_id[a_id]
        b = by_id[b_id]

        if decision.decision == "MERGE":
            merged_id = f"d_merge_{next_merge_seq:02d}"
            next_merge_seq += 1
            merged_dim = {
                "tentative_id": merged_id,
                "name": decision.merge_into_name or "",
                "description": decision.merge_into_description or "",
                "in_scope": _join_unique_csv(a.get("in_scope"), b.get("in_scope")),
                "out_of_scope": _join_unique_csv(
                    a.get("out_of_scope"), b.get("out_of_scope")
                ),
                "exemplar_skills": _dedup_preserve_order(
                    [*a.get("exemplar_skills", []), *b.get("exemplar_skills", [])]
                ),
                "overlap_flags": [],
            }
            result.locked_dimensions.append(merged_dim)
            result.merge_log.append(
                {
                    "merged_from": [a_id, b_id],
                    "into": merged_id,
                    "into_name": merged_dim["name"],
                    "reasoning": decision.reasoning,
                }
            )
            consumed.update({a_id, b_id})

        elif decision.decision == "SPLIT":
            split_seq = next_split_seq
            next_split_seq += 1
            new_ids: list[str] = []
            new_names: list[str] = []
            for i, sd in enumerate(decision.split_into or [], start=1):
                split_id = f"d_split_{split_seq:02d}_{i:02d}"
                new_dim = {
                    "tentative_id": split_id,
                    "name": sd.name,
                    "description": sd.description,
                    "in_scope": "",
                    "out_of_scope": "",
                    "exemplar_skills": [],
                    "overlap_flags": [],
                }
                result.locked_dimensions.append(new_dim)
                new_ids.append(split_id)
                new_names.append(sd.name)
            result.split_log.append(
                {
                    "split_from": [a_id, b_id],
                    "into": new_ids,
                    "into_names": new_names,
                    "reasoning": decision.reasoning,
                }
            )
            # Pool the originals' exemplars for later reassignment.
            result.unassigned_exemplars.extend(
                _dedup_preserve_order(
                    [*a.get("exemplar_skills", []), *b.get("exemplar_skills", [])]
                )
            )
            consumed.update({a_id, b_id})

    # Append untouched candidates in their original order.
    for d in candidate_dims:
        if d["tentative_id"] in consumed:
            continue
        result.locked_dimensions.append(dict(d))

    # Final dedup of unassigned_exemplars (in case multiple SPLITs pooled
    # the same skill name).
    result.unassigned_exemplars = _dedup_preserve_order(result.unassigned_exemplars)
    return result
