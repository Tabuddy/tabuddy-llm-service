"""Stage 7 enrichment validators (catalog-aware soft checks).

Pydantic catches schema-level issues at parse time. These runtime
validators add catalog-aware soft checks:

  * validate_ambiguity_in_catalog — confused_with skill_ids exist in
                                     the catalog
  * validate_vendor_recognized    — vendor is in a known vendor set
                                     (case-insensitive)
  * validate_maturity_distribution — sanity check on maturity mix
                                     across the role's skill set

All findings are warnings. The lifecycle decider (separate module)
promotes status to ``awaiting_review`` based on the LLM's flags
(ambiguity_flag=true / maturity=deprecated), not on validator output.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from skill_library_v3.schemas.enrichment import SkillEnrichment


def validate_ambiguity_in_catalog(
    enrichments: Iterable[SkillEnrichment],
    catalog_skill_ids: set[str],
) -> list[dict]:
    """For each enrichment with ambiguity_flag=True, every entry in
    confused_with must be a skill_id in the catalog. Unknown targets
    are warnings (the LLM may have invented a skill id)."""
    out: list[dict] = []
    for e in enrichments:
        if not e.ambiguity.ambiguity_flag:
            continue
        for target in e.ambiguity.confused_with:
            if target in catalog_skill_ids:
                continue
            out.append(
                {
                    "level": "warning",
                    "code": "ambiguity_target_not_in_catalog",
                    "skill_id": e.skill_id,
                    "unknown_target": target,
                    "message": (
                        f"{e.skill_id!r} ambiguity flagged with confused_with="
                        f"{target!r} but {target!r} is not in the catalog"
                    ),
                }
            )
    return out


def validate_vendor_recognized(
    enrichments: Iterable[SkillEnrichment],
    known_vendors: set[str],
) -> list[dict]:
    """Soft check: vendor names should match the known set
    (case-insensitively). Null vendor (Concept / SoftSkill) skips the
    check. A non-empty unrecognized vendor is a warning — could be a
    real new vendor OR a hallucinated name."""
    known_lc = {v.strip().lower() for v in known_vendors if v and v.strip()}
    out: list[dict] = []
    for e in enrichments:
        v = e.vendor_license.vendor
        if v is None or not v.strip():
            continue
        if v.strip().lower() in known_lc:
            continue
        out.append(
            {
                "level": "warning",
                "code": "vendor_not_in_known_set",
                "skill_id": e.skill_id,
                "vendor": v,
                "message": (
                    f"{e.skill_id!r} vendor={v!r} not in known vendor set; "
                    "could be a new vendor or LLM hallucination"
                ),
            }
        )
    return out


# Threshold knobs — adjust if calibration on real runs reveals these
# bands are too sensitive or too loose. Documented here so operations
# can find them.
_DEPRECATED_OVER_THRESHOLD = 0.5  # >50% deprecated -> warn


def validate_maturity_distribution(
    enrichments: list[SkillEnrichment],
) -> list[dict]:
    """Sanity-check the role's overall maturity mix.

    Two heuristics, both warnings:
      * >50% deprecated: LLM is being overzealous OR the role's skill
        set is genuinely sunset (rare; surface for human eyeball)
      * 0% well_known: highly suspicious; almost every tech role has
        at least one well-known skill (Git, SQL, etc.)

    Empty input -> no findings.
    """
    if not enrichments:
        return []
    counts = Counter(e.maturity.maturity for e in enrichments)
    total = sum(counts.values())
    out: list[dict] = []
    deprecated_share = counts.get("deprecated", 0) / total
    if deprecated_share > _DEPRECATED_OVER_THRESHOLD:
        out.append(
            {
                "level": "warning",
                "code": "too_many_deprecated",
                "deprecated_share": deprecated_share,
                "total_skills": total,
                "deprecated_count": counts.get("deprecated", 0),
                "message": (
                    f"{counts.get('deprecated', 0)}/{total} skills "
                    f"({deprecated_share:.0%}) flagged deprecated; LLM may "
                    "be over-classifying"
                ),
            }
        )
    if counts.get("well_known", 0) == 0:
        out.append(
            {
                "level": "warning",
                "code": "no_well_known_skills",
                "total_skills": total,
                "message": (
                    f"none of the {total} enriched skills are flagged "
                    "well_known; suspicious — most tech roles have at "
                    "least Git/SQL/etc."
                ),
            }
        )
    return out
