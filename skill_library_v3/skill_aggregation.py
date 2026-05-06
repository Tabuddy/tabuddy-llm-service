"""Stage 4 input prep: collect every unique skill across roles.

Stage 2's exemplar_skills + Stage 3's unassigned_exemplars together form
the universe of skills the catalog needs to type. This module pulls them
into a single deduplicated list, with role provenance recorded per skill
so Stage 5 (placement) can later read which roles each skill belongs to.

Pure function — no DB, no LLM, no embeddings. Stage 4's runner calls
this once at the start of a typing run.
"""

from __future__ import annotations

import re

# Slug helper: lowercase, non-alnum runs collapsed to ``-``, trimmed.
_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    s = _SLUG_NON_ALNUM.sub("-", name.strip().lower())
    return s.strip("-") or "unknown"


def aggregate_skills(role_inputs: list[dict]) -> list[dict]:
    """Aggregate skills across multiple roles' Stage 3 outputs.

    ``role_inputs`` items: ``{role_display, locked_dimensions,
    unassigned_exemplars}``. Each locked_dimension carries
    ``exemplar_skills`` (list of skill names).

    Returns a list of dicts (one per unique skill, deduplicated
    case-insensitively):
      * ``skill_id``        — stable lowercase slug of the canonical name
      * ``name``            — canonical display name (first casing seen)
      * ``roles``           — roles whose locked dims claimed this skill
      * ``unassigned_from`` — roles that pooled this skill via SPLIT
                              decisions (no locked dim claimed it there)

    Order of output: order of first occurrence in the input traversal
    (deterministic).
    """
    by_lc: dict[str, dict] = {}

    for role_input in role_inputs:
        role_display = role_input.get("role_display", "")

        # Walk locked-dim exemplars first.
        for dim in role_input.get("locked_dimensions") or []:
            for raw in dim.get("exemplar_skills") or []:
                if not isinstance(raw, str):
                    continue
                name = raw.strip()
                if not name:
                    continue
                key = name.lower()
                entry = by_lc.get(key)
                if entry is None:
                    by_lc[key] = {
                        "skill_id": _slugify(name),
                        "name": name,
                        "roles": [role_display] if role_display else [],
                        "unassigned_from": [],
                    }
                else:
                    if role_display and role_display not in entry["roles"]:
                        entry["roles"].append(role_display)

        # Then unassigned exemplars (from SPLIT decisions). These belong
        # to the role's universe but no locked dim claimed them.
        for raw in role_input.get("unassigned_exemplars") or []:
            if not isinstance(raw, str):
                continue
            name = raw.strip()
            if not name:
                continue
            key = name.lower()
            entry = by_lc.get(key)
            if entry is None:
                by_lc[key] = {
                    "skill_id": _slugify(name),
                    "name": name,
                    "roles": [],
                    "unassigned_from": [role_display] if role_display else [],
                }
            else:
                if role_display and role_display not in entry["unassigned_from"]:
                    entry["unassigned_from"].append(role_display)

    return list(by_lc.values())


def batch_skills(skills: list[dict], *, batch_size: int = 50) -> list[list[dict]]:
    """Split ``skills`` into batches of size ``batch_size``.

    Order is preserved — batches are contiguous slices, not random
    partitions. Empty input yields an empty list, not a list with one
    empty batch.

    Default ``batch_size`` is 50 per the blueprint ("batched 50-100 per
    call"). The smaller end of that range gives cheaper failures and
    smaller per-call payloads.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")
    return [skills[i : i + batch_size] for i in range(0, len(skills), batch_size)]
