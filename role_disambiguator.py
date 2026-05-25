"""Overlapping-role disambiguator — 3-tier deterministic tie-breaker.

Used by:
  - `jd_classifier.stage4_route_decision` Branch 1 (multi-alias-tie) BEFORE
    invoking LLM2.
  - `domain_classifier.classify_domain_role` §B sanity-check after the LLM
    picks a role in a domain.

Resolution order:
  Tier A — KRA-distinctness     (skipped if any candidate has < KRA_MIN_COUNT
                                  KRAs OR < SKILL_MIN_COUNT canonical skills;
                                  see "shell-row" handling)
  Tier B — Title-exact-match    (candidate display_name == nano_role or first
                                  nano_aliases entry, case-insensitive)
  Tier C — Skill-context overlap (canonical-skill set intersection between JD
                                  and candidate's dimension_skills)

Returns `(winner_dict, tier_label)` on a confident pick, or `None` when all
three tiers tie — the caller falls through to LLM2.

The module is import-light: it pulls KRAs + canonical-skill names from
Postgres in a single helper (`load_candidate_enrichment`) which is also used
to enrich the LLM2 prompt (Tier D fallback) so we don't double-fetch.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

from jd_similarity_matcher import _pg_connect, _qs

logger = logging.getLogger(__name__)


# ── Tuning constants ─────────────────────────────────────────────────────────

KRA_MIN_COUNT: int = 3
SKILL_MIN_COUNT: int = 10
KRA_MARGIN_MIN: float = 0.05
SKILL_MARGIN_MIN: int = 2

TOP_KRAS_FOR_PROMPT: int = 5
TOP_SKILLS_FOR_PROMPT: int = 10


# ── Title normalisation ──────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")


def _normalise_title(text: str) -> str:
    """Lower-case + collapse whitespace + strip punctuation/seniority noise.

    Matches "DevOps Engineer", "  devops engineer  ", "DevOps  Engineer." as
    the same string. Does NOT strip seniority prefixes — that's the role
    generalization layer upstream's job.
    """
    if not text:
        return ""
    s = text.strip().lower()
    s = _WHITESPACE_RE.sub(" ", s)
    s = s.rstrip(".,;:")
    return s


# ── DB enrichment fetcher ────────────────────────────────────────────────────


def load_candidate_enrichment(role_ids: Iterable[int]) -> dict[int, dict[str, Any]]:
    """Return per-role enrichment: KRA texts + canonical-skill display_names.

    Shape:
      {
        <role_id>: {
          "n_kras":    int,            # total KRA count (for shell-row skip)
          "n_skills":  int,            # total canonical-skill count
          "top_kras":  list[str],      # up to TOP_KRAS_FOR_PROMPT KRA texts
          "top_skills": list[str],     # up to TOP_SKILLS_FOR_PROMPT names
          "skill_set": set[str],       # lower-cased, full set (used by Tier C)
        }
      }

    Roles with no rows in either table get zeros + empty collections — the
    caller treats those as "shell rows" and skips Tier A for them.
    """
    ids = [int(r) for r in role_ids if r is not None]
    if not ids:
        return {}

    qs = _qs()
    out: dict[int, dict[str, Any]] = {
        rid: {
            "n_kras": 0,
            "n_skills": 0,
            "top_kras": [],
            "top_skills": [],
            "skill_set": set(),
        }
        for rid in ids
    }

    try:
        conn = _pg_connect()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[disambiguator] DB connect failed: %s", exc)
        return out

    try:
        with conn:
            with conn.cursor() as cur:
                # KRAs — ordered by position so top_kras is stable and meaningful.
                cur.execute(
                    f"""
                    SELECT role_id, kra_text
                      FROM {qs}.role_kras
                     WHERE role_id = ANY(%s)
                     ORDER BY role_id, position
                    """,
                    (ids,),
                )
                for rid, text in cur.fetchall():
                    rid = int(rid)
                    if rid not in out:
                        continue
                    out[rid]["n_kras"] += 1
                    if len(out[rid]["top_kras"]) < TOP_KRAS_FOR_PROMPT:
                        out[rid]["top_kras"].append(str(text))

                # Canonical skills via role_dimensions → dimension_skills.
                cur.execute(
                    f"""
                    SELECT rd.role_id, cs.display_name
                      FROM {qs}.role_dimensions rd
                      JOIN {qs}.dimension_skills ds ON ds.dimension_id = rd.dimension_id
                      JOIN {qs}.canonical_skills cs ON cs.id = ds.skill_id
                     WHERE rd.role_id = ANY(%s) AND cs.display_name IS NOT NULL
                     GROUP BY rd.role_id, cs.display_name
                     ORDER BY rd.role_id, cs.display_name
                    """,
                    (ids,),
                )
                for rid, name in cur.fetchall():
                    rid = int(rid)
                    if rid not in out:
                        continue
                    name_str = str(name)
                    out[rid]["n_skills"] += 1
                    out[rid]["skill_set"].add(name_str.lower())
                    if len(out[rid]["top_skills"]) < TOP_SKILLS_FOR_PROMPT:
                        out[rid]["top_skills"].append(name_str)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[disambiguator] enrichment fetch failed: %s", exc)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    return out


# ── Individual tiers ─────────────────────────────────────────────────────────


def _tier_a_kra_distinctness(
    candidates: list[dict[str, Any]],
    enrichment: dict[int, dict[str, Any]],
    kra_scores: dict[int, float] | None,
) -> dict[str, Any] | None:
    """Return the candidate with the best Stage 3c KRA score, IF every
    candidate is fully enriched AND the winner's margin clears
    `KRA_MARGIN_MIN`. Returns None when Tier A must be skipped or ties.

    `kra_scores` is the already-computed `kra_match_roles` mapping from
    Stage 3 (role_id → score). Passing it in lets us reuse the embeddings
    Stage 3c already paid for instead of re-embedding here.
    """
    if not candidates:
        return None

    # Shell-row skip — any candidate below either floor disables Tier A.
    for c in candidates:
        rid = c.get("role_id")
        info = enrichment.get(rid, {}) if rid is not None else {}
        if info.get("n_kras", 0) < KRA_MIN_COUNT:
            logger.debug(
                "[disambiguator] Tier A SKIPPED: candidate %s has n_kras=%d (< %d)",
                c.get("slug"), info.get("n_kras", 0), KRA_MIN_COUNT,
            )
            return None
        if info.get("n_skills", 0) < SKILL_MIN_COUNT:
            logger.debug(
                "[disambiguator] Tier A SKIPPED: candidate %s has n_skills=%d (< %d)",
                c.get("slug"), info.get("n_skills", 0), SKILL_MIN_COUNT,
            )
            return None

    if not kra_scores:
        # No KRA signal from Stage 3 — can't rank deterministically. Skip Tier A.
        logger.debug("[disambiguator] Tier A SKIPPED: no kra_scores provided")
        return None

    scored = sorted(
        (
            (float(kra_scores.get(c["role_id"], 0.0)), c)
            for c in candidates if c.get("role_id") is not None
        ),
        key=lambda t: t[0],
        reverse=True,
    )
    if len(scored) < 2:
        return scored[0][1] if scored else None
    top_score, top_cand = scored[0]
    second_score, _ = scored[1]
    margin = top_score - second_score
    if margin < KRA_MARGIN_MIN:
        logger.debug(
            "[disambiguator] Tier A INCONCLUSIVE: margin=%.4f < %.4f", margin, KRA_MARGIN_MIN,
        )
        return None
    logger.info(
        "[disambiguator] Tier A WINNER: %s (score=%.3f, margin=%.3f)",
        top_cand.get("slug"), top_score, margin,
    )
    return top_cand


def _tier_b_title_exact_match(
    candidates: list[dict[str, Any]],
    nano_role: str,
    nano_aliases: list[str] | None,
) -> dict[str, Any] | None:
    """Return the candidate whose display_name equals nano_role or the first
    nano_aliases entry (normalised). Preference order: nano_role wins over
    aliases; aliases follow input order. Returns None when no candidate
    matches OR more than one matches at the highest-priority key.
    """
    if not candidates:
        return None

    norm_role = _normalise_title(nano_role)
    norm_aliases = [_normalise_title(a) for a in (nano_aliases or []) if a]

    targets: list[str] = []
    if norm_role:
        targets.append(norm_role)
    for a in norm_aliases:
        if a and a not in targets:
            targets.append(a)

    if not targets:
        return None

    for tgt in targets:
        matches = [
            c for c in candidates
            if _normalise_title(c.get("display_name", "")) == tgt
        ]
        if len(matches) == 1:
            logger.info(
                "[disambiguator] Tier B WINNER: %s (display_name == %r)",
                matches[0].get("slug"), tgt,
            )
            return matches[0]
        if len(matches) > 1:
            # Two candidates share the same display_name (shouldn't happen
            # after the dedup pass, but be defensive). Don't pick a coin-flip.
            logger.debug(
                "[disambiguator] Tier B INCONCLUSIVE: %d candidates match %r",
                len(matches), tgt,
            )
            return None
    return None


def _tier_c_skill_overlap(
    candidates: list[dict[str, Any]],
    enrichment: dict[int, dict[str, Any]],
    jd_skills: list[str] | None,
) -> dict[str, Any] | None:
    """Pick the candidate with the largest canonical-skill overlap against
    `jd_skills` (Stage-2 extracted JD skills), IF the winning margin is at
    least `SKILL_MARGIN_MIN`. Returns None when JD has no skills or the
    contest is too close.
    """
    if not candidates or not jd_skills:
        return None

    jd_set = {s.lower() for s in jd_skills if s and isinstance(s, str)}
    if not jd_set:
        return None

    scored: list[tuple[int, dict[str, Any]]] = []
    for c in candidates:
        rid = c.get("role_id")
        skill_set = enrichment.get(rid, {}).get("skill_set", set()) if rid else set()
        overlap = len(jd_set & skill_set) if skill_set else 0
        scored.append((overlap, c))

    scored.sort(key=lambda t: t[0], reverse=True)
    if not scored:
        return None
    top_overlap, top_cand = scored[0]
    if top_overlap == 0:
        return None
    second_overlap = scored[1][0] if len(scored) > 1 else 0
    margin = top_overlap - second_overlap
    if margin < SKILL_MARGIN_MIN:
        logger.debug(
            "[disambiguator] Tier C INCONCLUSIVE: top=%d second=%d margin=%d < %d",
            top_overlap, second_overlap, margin, SKILL_MARGIN_MIN,
        )
        return None
    logger.info(
        "[disambiguator] Tier C WINNER: %s (overlap=%d, margin=%d)",
        top_cand.get("slug"), top_overlap, margin,
    )
    return top_cand


# ── Public entrypoint ────────────────────────────────────────────────────────


def disambiguate_overlap(
    candidates: list[dict[str, Any]],
    *,
    nano_role: str = "",
    nano_aliases: list[str] | None = None,
    jd_skills: list[str] | None = None,
    kra_scores: dict[int, float] | None = None,
    enrichment: dict[int, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str] | None:
    """Resolve a set of overlapping candidates via Tiers A → B → C.

    Args:
      candidates: list of dicts, each with required keys `role_id`, `slug`,
        `display_name`. Extra keys are passed through to the winner unchanged.
      nano_role: the JD title (nano-extracted, generalized). Tier B compares
        candidate display_names against this.
      nano_aliases: aliases nano emitted alongside the role title.
      jd_skills: canonical-skill display_names extracted from the JD body
        at Stage 2 (`skills_for_stage3` in main.py).
      kra_scores: role_id → Stage 3c KRA score. Reuses Stage 3c output to
        avoid re-embedding. When None, Tier A is skipped.
      enrichment: pre-loaded enrichment dict (output of
        `load_candidate_enrichment`). When None, loads it on demand.

    Returns: (winner_dict, "TIER_A_KRA" | "TIER_B_TITLE" | "TIER_C_SKILL")
             on success, or None when all three tiers were inconclusive.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        # Trivially the winner — pick it with TIER_B_TITLE so callers can
        # treat single-candidate cohorts uniformly.
        return candidates[0], "TIER_B_TITLE"

    if enrichment is None:
        role_ids = [c["role_id"] for c in candidates if c.get("role_id") is not None]
        enrichment = load_candidate_enrichment(role_ids)

    # Tier A
    winner = _tier_a_kra_distinctness(candidates, enrichment, kra_scores)
    if winner is not None:
        return winner, "TIER_A_KRA"

    # Tier B
    winner = _tier_b_title_exact_match(candidates, nano_role, nano_aliases or [])
    if winner is not None:
        return winner, "TIER_B_TITLE"

    # Tier C
    winner = _tier_c_skill_overlap(candidates, enrichment, jd_skills or [])
    if winner is not None:
        return winner, "TIER_C_SKILL"

    logger.debug(
        "[disambiguator] all tiers inconclusive over %d candidates; "
        "caller should fall through to LLM2",
        len(candidates),
    )
    return None
