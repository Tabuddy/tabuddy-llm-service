"""Stage 4 confidence-bucket routing.

Per blueprint:
  * confidence < 0.7   -> full_review     (every skill must be reviewed)
  * confidence 0.7-0.9 -> sample_review   (admin samples ~10%)
  * confidence > 0.9   -> auto_approved   (admin samples ~2% spot-check)

The ~10% / ~2% sampling is a UI/admin policy decision, not part of this
function. ``route_by_confidence`` just buckets deterministically; the
admin (or a future scheduler) decides what fraction of each bucket to
review.

Pure function — no I/O, no LLM, no DB.
"""

from __future__ import annotations

from skill_library_v3.schemas.typology import TypedSkill


# Bucket boundaries are inclusive on the sample band per blueprint:
#   "Confidence 0.7-0.9" -> sample (both endpoints inclusive)
LOW_BAND_TOP = 0.7      # below this -> full_review
HIGH_BAND_BOTTOM = 0.9  # above this -> auto_approved


def route_by_confidence(skills: list[TypedSkill]) -> dict[str, list[TypedSkill]]:
    """Partition ``skills`` into the three review buckets per their
    confidence score. Returns a dict with keys ``auto_approved``,
    ``sample_review``, ``full_review``."""
    auto: list[TypedSkill] = []
    sample: list[TypedSkill] = []
    full: list[TypedSkill] = []
    for s in skills:
        if s.confidence < LOW_BAND_TOP:
            full.append(s)
        elif s.confidence > HIGH_BAND_BOTTOM:
            auto.append(s)
        else:
            sample.append(s)
    return {
        "auto_approved": auto,
        "sample_review": sample,
        "full_review": full,
    }
