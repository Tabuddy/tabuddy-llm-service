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


# v1.4 cluster-gate thresholds (applied to centrality-weighted exemplar
# overlap). Below the lower bound the MERGE is hard-overridden to
# KEEP_SEPARATE; between the bounds the merge proceeds but the merged
# dim is tagged for human review.
CLUSTER_KEEP_SEPARATE_THRESHOLD: float = 0.20
CLUSTER_MERGE_CONFIRM_THRESHOLD: float = 0.50


@dataclass
class ApplyResult:
    """Output of apply_decisions."""

    locked_dimensions: list[dict] = field(default_factory=list)
    merge_log: list[dict] = field(default_factory=list)
    split_log: list[dict] = field(default_factory=list)
    unassigned_exemplars: list[str] = field(default_factory=list)
    cascading_merge_warnings: list[dict] = field(default_factory=list)
    cluster_gate_overrides: list[dict] = field(default_factory=list)
    borderline_merges: list[dict] = field(default_factory=list)


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


# ── Step 1 (v1.4): exemplar-cluster overlap ────────────────────────────────


def _normalize_exemplar(s: str) -> str:
    """Lowercase + collapse whitespace. Used for cluster-overlap matching
    only — does not change what gets stored in locked_dimensions."""
    return " ".join((s or "").lower().split())


def compute_exemplar_cluster_overlap(
    dim_a: dict,
    dim_b: dict,
    *,
    centrality_weighting: bool = True,
) -> float:
    """``|A ∩ B| / min(|A|, |B|)`` over the two dims' ``exemplar_skills``
    only (not ``in_scope`` tokens).

    Used as the v1.4 hard-override gate before Stage 3c lands a MERGE.
    The token-Jaccard rule in ``dim_overlap_pairs.py`` is for *detection*
    (combined in_scope+exemplar tokens, broad recall); this is for the
    *decision* (exemplar-only, exact cluster signal).

    If ``centrality_weighting=True`` (default), each exemplar contributes
    weight ``1/(rank+1)`` where ``rank`` is its 0-based position in its
    own dim's exemplar list. Sharing a most-defining (rank 0) exemplar
    in both dims scores far higher than sharing a peripheral (rank 4+)
    exemplar — the v1.3 d_init_02 ↔ d_init_13 merge (both listed Kubeflow
    at rank 4) is the motivating failure.

    Matching is case-insensitive and whitespace-tolerant.

    Returns 0.0 if either dim has no exemplars.
    """
    a_raw = list(dim_a.get("exemplar_skills") or [])
    b_raw = list(dim_b.get("exemplar_skills") or [])
    if not a_raw or not b_raw:
        return 0.0

    a_norm = [_normalize_exemplar(s) for s in a_raw]
    b_norm = [_normalize_exemplar(s) for s in b_raw]
    a_set = set(a_norm)
    b_set = set(b_norm)

    if not centrality_weighting:
        intersection = a_set & b_set
        denom = min(len(a_set), len(b_set))
        return len(intersection) / denom if denom else 0.0

    a_weight = {tok: 1.0 / (i + 1) for i, tok in enumerate(a_norm) if tok not in {x for x in a_norm[:i]}}
    b_weight = {tok: 1.0 / (i + 1) for i, tok in enumerate(b_norm) if tok not in {x for x in b_norm[:i]}}

    shared = set(a_weight) & set(b_weight)
    weighted_intersect = sum(min(a_weight[t], b_weight[t]) for t in shared)

    weighted_size_a = sum(1.0 / (i + 1) for i in range(len(a_weight)))
    weighted_size_b = sum(1.0 / (i + 1) for i in range(len(b_weight)))
    denom = min(weighted_size_a, weighted_size_b)
    return weighted_intersect / denom if denom else 0.0


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
            # Surface as a warning so reviewers can see the missed
            # decision instead of it being invisibly dropped.
            if decision.decision in ("MERGE", "SPLIT"):
                reused = [x for x in (a_id, b_id) if x in consumed]
                result.cascading_merge_warnings.append(
                    {
                        "code": "cascading_merge",
                        "decision": decision.decision,
                        "pair": [a_id, b_id],
                        "reused_source_ids": reused,
                        "reasoning": decision.reasoning,
                        "message": (
                            f"{decision.decision} for pair {a_id} ↔ {b_id} was "
                            f"skipped because source id(s) {reused!r} were "
                            "already consumed by an earlier decision"
                        ),
                    }
                )
            continue

        if decision.decision == "KEEP_SEPARATE":
            # Both dims stay as-is in the untouched-fall-through pass below.
            continue

        a = by_id[a_id]
        b = by_id[b_id]

        if decision.decision == "MERGE":
            # v1.4 cluster gate. Compute exemplar-only overlap BEFORE the
            # strict-union build. Below 0.20 the LLM's MERGE is overridden
            # to KEEP_SEPARATE (clusters are disjoint regardless of how
            # similar the descriptions sound). 0.20-0.50 proceeds but is
            # flagged for review. >=0.50 proceeds unchanged.
            cluster_overlap = compute_exemplar_cluster_overlap(a, b)
            if cluster_overlap < CLUSTER_KEEP_SEPARATE_THRESHOLD:
                result.cluster_gate_overrides.append(
                    {
                        "code": "cluster_gate_override",
                        "pair": [a_id, b_id],
                        "exemplar_overlap": cluster_overlap,
                        "llm_decision": "MERGE",
                        "final_decision": "KEEP_SEPARATE",
                        "reasoning": decision.reasoning,
                        "message": (
                            f"Exemplar overlap {cluster_overlap:.2f} < "
                            f"{CLUSTER_KEEP_SEPARATE_THRESHOLD:.2f} — clusters "
                            f"disjoint; LLM MERGE overridden to KEEP_SEPARATE"
                        ),
                    }
                )
                # Skip merge: both sources fall through to untouched pass.
                continue

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
                "_merge_provenance": {
                    "source_dim_ids": [a_id, b_id],
                    "in_scope_origin": "strict_union",
                    "exemplar_origin": "strict_union",
                },
            }
            if cluster_overlap < CLUSTER_MERGE_CONFIRM_THRESHOLD:
                merged_dim["_borderline_merge"] = {
                    "exemplar_overlap": cluster_overlap,
                    "requires_review": True,
                }
                result.borderline_merges.append(
                    {
                        "code": "borderline_merge",
                        "pair": [a_id, b_id],
                        "merged_id": merged_id,
                        "exemplar_overlap": cluster_overlap,
                        "reasoning": decision.reasoning,
                        "message": (
                            f"Exemplar overlap {cluster_overlap:.2f} in "
                            f"borderline range [{CLUSTER_KEEP_SEPARATE_THRESHOLD:.2f}, "
                            f"{CLUSTER_MERGE_CONFIRM_THRESHOLD:.2f}); merged dim "
                            f"flagged for human review"
                        ),
                    }
                )
            result.locked_dimensions.append(merged_dim)
            result.merge_log.append(
                {
                    "merged_from": [a_id, b_id],
                    "into": merged_id,
                    "into_name": merged_dim["name"],
                    "a_dim_id": a_id,
                    "a_name": pair.get("a_name") or a.get("name") or "",
                    "a_role": pair.get("a_role") or "",
                    "b_dim_id": b_id,
                    "b_name": pair.get("b_name") or b.get("name") or "",
                    "b_role": pair.get("b_role") or "",
                    "pair_kind": pair.get("kind") or "",
                    "similarity": pair.get("similarity"),
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
                    "a_dim_id": a_id,
                    "a_name": pair.get("a_name") or a.get("name") or "",
                    "a_role": pair.get("a_role") or "",
                    "b_dim_id": b_id,
                    "b_name": pair.get("b_name") or b.get("name") or "",
                    "b_role": pair.get("b_role") or "",
                    "pair_kind": pair.get("kind") or "",
                    "similarity": pair.get("similarity"),
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
