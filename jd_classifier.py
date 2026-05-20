"""Stage 4 (Decision Routing) + Stage 5 (Post-Classification Updates) for the
JD role-classifier pipeline.

Stage 4 reads the three signal lists from :class:`Stage3Result` and routes the
JD into one of six cases A–F. Cases D and F may fire an LLM2 tie-breaker.
Stage 5 applies the consequences: rolling-mean centroid update + audit rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

from jd_similarity_matcher import RoleSignal, Stage3Result


# ── Tunable constants ────────────────────────────────────────────────────────

# With Stage 3c using MAX-per-role aggregation, real KRA top scores range
# from ~0.35 (weak alignment) to ~0.55 (very strong). KRA_MIN_SCORE acts as
# the "the JD's R&R aligns with this role's KRAs at all" floor; below it,
# no classification is attempted.
KRA_MIN_SCORE: float = 0.40
KRA_MARGIN_DEFAULT: float = 0.05
KRA_MARGIN_ON_COLLISION: float = 0.08
KRA_TIE_BAND: float = 0.02
SKILL_TIE_BAND: float = 0.05
LLM2_MIN_CONFIDENCE: float = 0.70

# Gap #6 bypass: when alias is an exact match and skill agrees with it at
# moderate confidence, skip KRA gating entirely. KRA is noisy because R&R
# blocks rarely overlap precisely with stored KRAs even after Stage 3c uses
# MAX aggregation. This bypass prevents valid classifications from being
# queued when the JD title + skill list make the role unambiguous.
ALIAS_EXACT_MIN_SCORE: float = 0.95
SKILL_WITNESS_MIN_SCORE: float = 0.50

# Auto-trigger gates for the v3 skill-library pipeline (Stage 0 -> Stage 8).
# Only fire when all 3 signals are weak (truly unknown role).
NEW_ROLE_SKILL_CEILING: float = 0.30
# Belt-and-suspenders floor for new-role detection. Stage 1 role resolution
# (resolve_stage1_role) is the primary signal — this only catches the case
# where stage1_resolved was not computed for some reason.
NEW_ROLE_ALIAS_FLOOR: float = 0.85
# Candidate-KRA threshold: when a classified role's top KRA similarity is
# below this, treat the JD R&R as a novel responsibility for that role and
# enqueue it for admin review.
NEW_KRA_THRESHOLD: float = 0.40

# Embedding-similarity threshold for Stage 1 role generalization. When the
# verbatim title doesn't exact-match any canonical name/alias, embed it and
# fall back to the closest role_alias. At >= this threshold we treat as
# the same canonical role (prevents catalog explosion from seniority/
# framework variants like "Senior Backend Engineer", "Sr. GCP Data Engineer").
# Tuned to 0.75 after batch validation: many real-world variant titles
# (e.g., "Sr. GCP Data Engineer" vs "Data Engineer") embed in the 0.75-0.85
# range; 0.85 was too strict.
STAGE1_EMBEDDING_THRESHOLD: float = 0.70


# ── Data models ──────────────────────────────────────────────────────────────

CaseLiteral = Literal["A", "B", "C", "D", "E", "F"]


@dataclass
class Stage4Decision:
    case: CaseLiteral
    chosen_role: RoleSignal | None
    confidence: float
    llm2_fired: bool
    llm2_reasoning: str | None
    alias_collision_detected: bool
    queued: bool
    reasoning: str


@dataclass
class Llm2Result:
    chosen_role_slug: str
    confidence: float
    reasoning: str


Llm2Fn = Callable[..., Awaitable[Llm2Result]]


@dataclass
class Stage1RoleResolution:
    """Result of resolving Stage 1's verbatim role title against the canonical
    catalog. None when neither exact match nor embedding-similarity fallback
    finds a role — that None is THE signal that this is a brand-new role.

    match_kind:
      * 'name'      — exact case-insensitive match on roles.display_name
      * 'alias'     — exact case-insensitive match on role_aliases.alias_lower
      * 'embedding' — semantic match (cosine >= STAGE1_EMBEDDING_THRESHOLD)
                      against role_aliases.alias_embedding; used to generalize
                      seniority/framework variants like 'Senior Backend
                      Engineer' -> 'Backend Engineer' without firing v3.
    """

    role_id: int
    slug: str
    display_name: str
    match_kind: Literal["name", "alias", "embedding"]
    similarity: float | None = None


def _embed_role_text(role_text: str) -> list[float] | None:
    """Embed a Stage 1 role title via the shared Azure embedding client.

    Pulled out so tests can monkeypatch the import target without touching
    the actual embedding client.
    """
    from skill_matcher import _azure_embed_sync
    vecs = _azure_embed_sync([role_text])
    if not vecs or not vecs[0]:
        return None
    return list(vecs[0])


def resolve_stage1_role(
    conn,
    role_text: str,
    *,
    schema: str,
    embed_fn: Callable[[str], list[float] | None] | None = None,
) -> Stage1RoleResolution | None:
    """Resolve Stage 1's verbatim role title against the canonical catalog.

    Resolution order:
      1. Exact case-insensitive match on ``roles.display_name``
      2. Exact case-insensitive match on ``role_aliases.alias_lower``
      3. Embedding fallback: embed ``role_text`` and find the role with the
         highest cosine similarity against any of its alias_embeddings.
         Accept the match only when similarity >= STAGE1_EMBEDDING_THRESHOLD.

    Step 3 prevents catalog explosion: 'Senior Backend Engineer' embeds at
    cosine ~0.92 against 'Backend Developer' / 'Backend Engineer' aliases,
    so it resolves to backend-engineer rather than triggering v3.

    Deliberately avoids the trigram fallback that Stage 3b uses (that
    fallback returns substring noise like 'AI Engineer' -> 'ar-vr-engineer'
    0.71 on the shared 'engineer' token).
    """
    text = (role_text or "").strip().lower()
    if not text:
        return None
    qs = '"' + schema.replace('"', '""') + '"'

    with conn.cursor() as cur:
        # 1) exact display_name
        cur.execute(
            f"""
            SELECT r.id, r.slug, r.display_name
              FROM {qs}.roles r
             WHERE LOWER(r.display_name) = %s
             LIMIT 1
            """,
            (text,),
        )
        row = cur.fetchone()
        if row:
            return Stage1RoleResolution(
                role_id=int(row[0]), slug=str(row[1]),
                display_name=str(row[2]), match_kind="name",
                similarity=1.0,
            )

        # 2) exact alias match
        cur.execute(
            f"""
            SELECT r.id, r.slug, r.display_name
              FROM {qs}.role_aliases ra
              JOIN {qs}.roles r ON r.id = ra.role_id
             WHERE ra.alias_lower = %s
             LIMIT 1
            """,
            (text,),
        )
        row = cur.fetchone()
        if row:
            return Stage1RoleResolution(
                role_id=int(row[0]), slug=str(row[1]),
                display_name=str(row[2]), match_kind="alias",
                similarity=1.0,
            )

        # 3) embedding fallback (generalization)
        embed = embed_fn or _embed_role_text
        vec = embed(role_text)
        if not vec:
            return None
        vec_str = "[" + ",".join(map(str, vec)) + "]"
        cur.execute(
            f"""
            SELECT r.id, r.slug, r.display_name,
                   MAX(1 - (ra.alias_embedding <=> %s::{qs}.vector(1536))) AS sim
              FROM {qs}.role_aliases ra
              JOIN {qs}.roles r ON r.id = ra.role_id
             WHERE ra.alias_embedding IS NOT NULL
             GROUP BY r.id, r.slug, r.display_name
             ORDER BY sim DESC
             LIMIT 1
            """,
            (vec_str,),
        )
        row = cur.fetchone()
        if row and float(row[3]) >= STAGE1_EMBEDDING_THRESHOLD:
            return Stage1RoleResolution(
                role_id=int(row[0]), slug=str(row[1]),
                display_name=str(row[2]), match_kind="embedding",
                similarity=float(row[3]),
            )

    return None


@dataclass
class Stage5Updates:
    centroid_updated: bool
    centroid_n_after: int | None
    queue_entry_id: int | None
    collision_log_id: int | None


# ── Stage 4: Decision Routing ────────────────────────────────────────────────

async def stage4_route_decision(
    s3: Stage3Result,
    r_and_r_text: str,
    role_name_input: str,
    *,
    stage1_resolved: Stage1RoleResolution | None = None,
    llm2_fn: Llm2Fn | None = None,
) -> Stage4Decision:
    """Route a Stage 3 result into one of the 6 cases.

    When ``stage1_resolved`` is provided (Stage 1's verbatim role title
    canonically resolves), it is the FIRST discriminator — Stage 4 will
    short-circuit to Case A or Case D using that resolution rather than
    fall through to Stage 3b's trigram-noisy alias scores.

    Pure-logic Cases A/B/C/E never call llm2_fn. Cases D/F call it with the
    candidate roles + R&R text and return its choice (or queue if its
    confidence is below LLM2_MIN_CONFIDENCE).
    """
    skill = s3.skill_match_roles
    alias = s3.alias_match_roles
    kra = s3.kra_match_roles

    top_skill = skill[0] if skill else None
    top_alias = alias[0] if alias else None
    alias_collision = len({a.role_id for a in alias}) > 1

    # ── Stage-1-priority branch (NEW) ─────────────────────────────────────────
    # Stage 1 already extracted a verbatim role title and we exact-matched it
    # against the canonical catalog. If that match exists, it is the highest-
    # signal source we have — overrides the trigram alias noise that Stage 3b
    # can produce on shared substrings like "engineer".
    if stage1_resolved is not None:
        s1_role = RoleSignal(
            role_id=stage1_resolved.role_id,
            slug=stage1_resolved.slug,
            display_name=stage1_resolved.display_name,
            score=1.0,
            signal_type="stage1_match",
        )
        matching_kra = next(
            (r for r in kra if r.role_id == stage1_resolved.role_id),
            None,
        )
        if matching_kra is not None and matching_kra.score >= KRA_MIN_SCORE:
            return Stage4Decision(
                case="A",
                chosen_role=s1_role,
                confidence=matching_kra.score,
                llm2_fired=False,
                llm2_reasoning=None,
                alias_collision_detected=False,
                queued=False,
                reasoning=(
                    f"Stage 1 title '{stage1_resolved.display_name}' "
                    f"({stage1_resolved.match_kind} match"
                    + (f", sim {stage1_resolved.similarity:.2f}" if stage1_resolved.similarity is not None else "")
                    + f"); KRA agrees ({matching_kra.score:.2f})"
                ),
            )
        if kra and kra[0].role_id != stage1_resolved.role_id:
            # Stage 1 names role X, KRA tops role Y → LLM2 tie-break with
            # BOTH candidates so the title-suggested role is seen.
            return await _resolve_via_llm2(
                case="D",
                tied_candidates=[s1_role, kra[0]],
                r_and_r_text=r_and_r_text,
                llm2_fn=llm2_fn,
                alias_collision=True,
                queue_reason="llm2_unsure",
            )
        # KRA list is empty or none of it matches Stage 1's role — classify
        # by Stage 1 anyway with a confidence haircut.
        return Stage4Decision(
            case="A",
            chosen_role=s1_role,
            confidence=0.80,
            llm2_fired=False,
            llm2_reasoning=None,
            alias_collision_detected=False,
            queued=False,
            reasoning=(
                f"Stage 1 title '{stage1_resolved.display_name}' "
                f"({stage1_resolved.match_kind} match"
                + (f", sim {stage1_resolved.similarity:.2f}" if stage1_resolved.similarity is not None else "")
                + "); KRA inconclusive"
            ),
        )

    # ── Trigram-noise filter (NEW) ────────────────────────────────────────────
    # When Stage 1's title did NOT canonically resolve AND the fuzzy alias
    # signal points at a DIFFERENT role than the KRA top, Stage 3b is most
    # likely returning substring noise (e.g., "AI Engineer" trigram-matches
    # "ar-vr-engineer" 0.71 because both contain "engineer", while the real
    # nearest role by KRA is ai-compliance-officer). Treat alias as absent.
    #
    # If the fuzzy alias top AGREES with KRA top, it is corroborating
    # evidence — keep it so consensus checks (Cases A/C) can still fire.
    if (
        stage1_resolved is None
        and top_alias is not None
        and top_alias.score < ALIAS_EXACT_MIN_SCORE
        and kra
        and top_alias.role_id != kra[0].role_id
    ):
        alias = []
        top_alias = None
        alias_collision = False

    # ── Alias + skill consensus bypass (gap #6) ───────────────────────────────
    # When KRA is too weak / too noisy to gate on (would route to Case E),
    # BUT the JD title is an exact alias hit AND skills independently agree,
    # classify confidently. Lets clear classifications through when KRA is
    # the only weak signal.
    second_kra = kra[1].score if len(kra) > 1 else 0.0
    top_kra_score = kra[0].score if kra else 0.0
    kra_margin = top_kra_score - second_kra
    required_margin = (
        KRA_MARGIN_ON_COLLISION if alias_collision else KRA_MARGIN_DEFAULT
    )
    kra_would_queue = (
        not kra
        or top_kra_score < KRA_MIN_SCORE
        or kra_margin < required_margin
    )
    alias_exact_and_skill_agrees = (
        top_alias is not None
        and top_alias.score >= ALIAS_EXACT_MIN_SCORE
        and top_skill is not None
        and top_skill.role_id == top_alias.role_id
        and top_skill.score >= SKILL_WITNESS_MIN_SCORE
    )
    if kra_would_queue and alias_exact_and_skill_agrees:
        return Stage4Decision(
            case="A",
            chosen_role=top_alias,
            confidence=top_alias.score * 0.9,
            llm2_fired=False,
            llm2_reasoning=None,
            alias_collision_detected=False,
            queued=False,
            reasoning=(
                f"Alias exact ({top_alias.score:.2f}) + skill agrees "
                f"({top_skill.score:.2f}) on {top_alias.slug}; "
                f"KRA weak/noisy -> bypass"
            ),
        )

    # ── Pre-margin convergence (fix A): when signals agree, classify before
    # margin gate has a chance to queue. ─────────────────────────────────────
    skill_tied = (
        len(skill) >= 2
        and (skill[0].score - skill[1].score) < SKILL_TIE_BAND
    )

    # Case A early: all 3 top point at the same role (and skill not tied).
    if (
        kra
        and top_kra_score >= KRA_MIN_SCORE
        and top_skill is not None
        and top_alias is not None
        and not skill_tied
        and top_skill.role_id == top_alias.role_id == kra[0].role_id
    ):
        return Stage4Decision(
            case="A",
            chosen_role=kra[0],
            confidence=kra[0].score,
            llm2_fired=False,
            llm2_reasoning=None,
            alias_collision_detected=False,
            queued=False,
            reasoning=f"All 3 signals top-rank {kra[0].slug}",
        )

    # Case C early: skill tied + alias and KRA agree on the same role.
    if (
        kra
        and top_kra_score >= KRA_MIN_SCORE
        and skill_tied
        and top_alias is not None
        and top_alias.role_id == kra[0].role_id
    ):
        return Stage4Decision(
            case="C",
            chosen_role=kra[0],
            confidence=kra[0].score,
            llm2_fired=False,
            llm2_reasoning=None,
            alias_collision_detected=False,
            queued=False,
            reasoning=(
                f"Skill scores tied ({skill[0].score:.2f} vs {skill[1].score:.2f}); "
                f"alias+KRA agree on {kra[0].slug}"
            ),
        )

    # Case D priority (fix B): alias is an exact hit but disagrees with KRA
    # top. Force LLM2 with [alias_role, kra_top] so the alias-suggested role
    # is in the candidate set — even when KRA top-2 happen to be tied.
    if (
        kra
        and top_kra_score >= KRA_MIN_SCORE
        and top_alias is not None
        and top_alias.score >= ALIAS_EXACT_MIN_SCORE
        and top_alias.role_id != kra[0].role_id
    ):
        return await _resolve_via_llm2(
            case="D",
            tied_candidates=[top_alias, kra[0]],
            r_and_r_text=r_and_r_text,
            llm2_fn=llm2_fn,
            alias_collision=True,
            queue_reason="llm2_unsure",
        )

    # ── Hard gates first (Case E) ─────────────────────────────────────────────
    if not kra:
        return _queue("E", reason="no_kra_signal",
                      detail="no KRA results")
    top_kra = kra[0]
    if top_kra.score < KRA_MIN_SCORE:
        # Skill_top fallback: KRA is noisy but skill profile may clearly
        # identify a canonical role. Generalize "designation" titles (e.g.,
        # 'Applications Development Programmer Analyst' has skill_top=
        # backend-engineer 0.27) into the skill_top role rather than queue
        # → prevents v3 from firing for corporate-jargon titles.
        if (
            stage1_resolved is None
            and top_skill is not None
            and top_skill.score >= 0.20
        ):
            return Stage4Decision(
                case="B",
                chosen_role=top_skill,
                confidence=top_skill.score,
                llm2_fired=False,
                llm2_reasoning=None,
                alias_collision_detected=False,
                queued=False,
                reasoning=(
                    f"Stage 1 title '{role_name_input}' is unmapped "
                    f"(designation?); KRA inconclusive "
                    f"({top_kra.score:.2f}). Skill profile points at "
                    f"{top_skill.slug} ({top_skill.score:.2f}) - generalize."
                ),
            )
        return _queue("E", reason="low_kra",
                      detail=f"top KRA {top_kra.score:.2f} < {KRA_MIN_SCORE}")

    second_kra = kra[1].score if len(kra) > 1 else 0.0
    margin = top_kra.score - second_kra
    required_margin = (
        KRA_MARGIN_ON_COLLISION if alias_collision else KRA_MARGIN_DEFAULT
    )
    if margin < KRA_TIE_BAND:
        # Case F: top KRA candidates are statistically indistinguishable.
        # Fire LLM2 tie-breaker over the tied cohort.
        tied_candidates = _tied_cohort(kra, KRA_TIE_BAND)
        return await _resolve_via_llm2(
            case="F",
            tied_candidates=tied_candidates,
            r_and_r_text=r_and_r_text,
            llm2_fn=llm2_fn,
            alias_collision=False,
            queue_reason="llm2_unsure",
        )
    if margin < required_margin:
        # When the JD title doesn't canonically resolve (stage1_resolved=None),
        # the small_margin gate would otherwise queue. Instead classify into
        # the KRA top — the JD will get a best-effort answer for downstream
        # consumers, and v3 will independently materialize the real role.
        if stage1_resolved is None and top_kra_score >= KRA_MIN_SCORE:
            # When skill_top has meaningful confidence AND disagrees with
            # kra_top, route through LLM2 rather than blindly pick kra_top.
            # Prevents "Senior Java Backend Developer" → android-engineer
            # (KRA top wrong because Java is over-associated with android in
            # the canonical_skills catalog).
            #
            # Gate: max(0.20 absolute, 50% of kra_top.score). Combines a
            # noise floor with a relative-strength check so it adapts to
            # the catalog's actual scoring distribution.
            skill_vs_kra_gate = max(0.20, 0.5 * top_kra_score)
            if (
                top_skill is not None
                and top_skill.score >= skill_vs_kra_gate
                and top_skill.role_id != kra[0].role_id
            ):
                return await _resolve_via_llm2(
                    case="D",
                    tied_candidates=[top_skill, kra[0]],
                    r_and_r_text=r_and_r_text,
                    llm2_fn=llm2_fn,
                    alias_collision=False,
                    queue_reason="llm2_unsure",
                )
            return Stage4Decision(
                case="B",
                chosen_role=kra[0],
                confidence=kra[0].score,
                llm2_fired=False,
                llm2_reasoning=None,
                alias_collision_detected=True,
                queued=False,
                reasoning=(
                    f"Stage 1 title '{role_name_input}' not in catalog; "
                    f"KRA top-2 within margin -> classify into nearest "
                    f"neighbor {kra[0].slug} ({kra[0].score:.2f})"
                ),
            )
        # Skill_top fallback (same generalization rule as the low_kra gate):
        # if stage1 unresolved AND skill_top points at an existing role with
        # decent confidence, classify by skill rather than queue + fire v3.
        if (
            stage1_resolved is None
            and top_skill is not None
            and top_skill.score >= 0.20
        ):
            return Stage4Decision(
                case="B",
                chosen_role=top_skill,
                confidence=top_skill.score,
                llm2_fired=False,
                llm2_reasoning=None,
                alias_collision_detected=False,
                queued=False,
                reasoning=(
                    f"Stage 1 title '{role_name_input}' is unmapped; "
                    f"KRA margin too small. Skill profile points at "
                    f"{top_skill.slug} ({top_skill.score:.2f}) - generalize."
                ),
            )
        return _queue("E", reason="small_margin",
                      detail=f"KRA margin {margin:.2f} < {required_margin}")

    # ── Signal convergence ────────────────────────────────────────────────────
    skill_tied = (
        len(skill) >= 2
        and (skill[0].score - skill[1].score) < SKILL_TIE_BAND
    )

    # Case C: skill scores tied (cannot discriminate) but alias+KRA agree
    # → classify by KRA. Must check this *before* Case A because skill_top
    # may coincide with the tied-cohort winner.
    if (
        skill_tied
        and top_alias is not None
        and top_alias.role_id == top_kra.role_id
    ):
        return Stage4Decision(
            case="C",
            chosen_role=top_kra,
            confidence=top_kra.score,
            llm2_fired=False,
            llm2_reasoning=None,
            alias_collision_detected=False,
            queued=False,
            reasoning=(
                f"Skill scores tied "
                f"(top {skill[0].score:.2f} vs {skill[1].score:.2f}); "
                f"alias+KRA agree on {top_kra.slug}"
            ),
        )

    # Case A: all 3 agree (and skill not tied)
    if (
        top_skill is not None
        and top_alias is not None
        and top_skill.role_id == top_alias.role_id == top_kra.role_id
    ):
        return Stage4Decision(
            case="A",
            chosen_role=top_kra,
            confidence=top_kra.score,
            llm2_fired=False,
            llm2_reasoning=None,
            alias_collision_detected=False,
            queued=False,
            reasoning=f"All 3 signals top-rank {top_kra.slug}",
        )

    # Case B: skill+KRA agree, alias mismatches or absent
    if (
        top_skill is not None
        and top_skill.role_id == top_kra.role_id
        and (top_alias is None or top_alias.role_id != top_kra.role_id)
    ):
        alias_detail = (
            "alias missed entirely" if top_alias is None
            else f"alias->{top_alias.slug}"
        )
        return Stage4Decision(
            case="B",
            chosen_role=top_kra,
            confidence=top_kra.score * 0.95,  # small penalty for alias gap
            llm2_fired=False,
            llm2_reasoning=None,
            alias_collision_detected=True,
            queued=False,
            reasoning=f"Skill+KRA agree on {top_kra.slug}; {alias_detail}",
        )

    # Case D: alias and KRA point at different roles, skill does NOT witness
    # KRA → LLM2 tie-breaker between the two candidates.
    if top_alias is not None and top_alias.role_id != top_kra.role_id:
        return await _resolve_via_llm2(
            case="D",
            tied_candidates=[top_alias, top_kra],
            r_and_r_text=r_and_r_text,
            llm2_fn=llm2_fn,
            alias_collision=True,
            queue_reason="llm2_unsure",
        )

    # Case D' (skill-vs-KRA disagreement) — when there's no usable alias
    # signal AND skill_top points at a different role than kra_top with
    # meaningful confidence, fire LLM2 with [skill_top, kra_top] so the
    # skill-suggested role appears in the candidate set. Previously this
    # case fell through to the "top KRA stands" fallback and silently
    # picked KRA top, which routinely misclassified JDs like "Senior Java
    # Backend Developer" (skill=backend, kra=android because of Java).
    #
    # Gate: max(0.20 absolute, 50% of kra_top.score). See Case B' branch
    # above for the same logic — adapts to sparse-catalog scoring.
    skill_vs_kra_gate = max(0.20, 0.5 * kra[0].score)
    if (
        top_skill is not None
        and top_skill.score >= skill_vs_kra_gate
        and top_skill.role_id != kra[0].role_id
    ):
        return await _resolve_via_llm2(
            case="D",
            tied_candidates=[top_skill, kra[0]],
            r_and_r_text=r_and_r_text,
            llm2_fn=llm2_fn,
            alias_collision=False,
            queue_reason="llm2_unsure",
        )

    # Fallback: top KRA stands but no other signal supports it. Treat as
    # weak Case A (auto-classify with a confidence haircut).
    top_kra = kra[0]
    return Stage4Decision(
        case="A",
        chosen_role=top_kra,
        confidence=top_kra.score * 0.85,
        llm2_fired=False,
        llm2_reasoning=None,
        alias_collision_detected=False,
        queued=False,
        reasoning=f"Top KRA {top_kra.slug} stands; no contradicting signal",
    )


def _tied_cohort(kra: list[RoleSignal], band: float) -> list[RoleSignal]:
    """Return all KRA results whose score is within ``band`` of the top score."""
    if not kra:
        return []
    top = kra[0].score
    return [r for r in kra if (top - r.score) < band]


# ── Production LLM2 wrapper (real Azure call) ────────────────────────────────

_LLM2_SYSTEM_PROMPT = (
    "You break ties between candidate roles for a job description. You are "
    "given the responsibilities section of the JD and 2–3 candidate roles. "
    "Pick the one role whose day-to-day work best matches the JD's "
    "responsibilities. Return JSON with exactly these keys: "
    '"chosen_role_slug" (one of the candidate slugs verbatim), '
    '"confidence" (float 0.0-1.0), "reasoning" (one sentence).'
)


async def llm2_resolve_role(
    r_and_r_text: str,
    candidates: list[RoleSignal],
    *,
    cost_acc=None,
) -> Llm2Result:
    """Default production LLM2: o4-mini reasoning tier, JSON-mode response.

    Falls back to a queue-inducing low-confidence result when Azure key is
    not configured or the call fails — defensive: never let LLM2 raise.
    """
    import json
    import logging

    from llm_client import REASONING_MODEL, get_reasoning_client

    logger = logging.getLogger(__name__)
    client = get_reasoning_client()
    if client is None:
        return Llm2Result(
            chosen_role_slug=candidates[0].slug if candidates else "",
            confidence=0.0,
            reasoning="LLM2 unavailable: no reasoning client",
        )

    candidate_block = "\n".join(
        f"- {c.slug} ({c.display_name})" for c in candidates
    )
    user_msg = (
        f"JD responsibilities:\n{(r_and_r_text or '')[:6000]}\n\n"
        f"Candidates:\n{candidate_block}\n\n"
        "Pick exactly one."
    )

    try:
        kwargs = dict(
            model=REASONING_MODEL,
            messages=[
                {"role": "system", "content": _LLM2_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            timeout=30,
        )
        # o4-mini rejects temperature; default 1.0 is fine.
        resp = await client.chat.completions.create(**kwargs)
        if cost_acc is not None:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                cost_acc.add(
                    REASONING_MODEL,
                    int(getattr(usage, "prompt_tokens", 0) or 0),
                    int(getattr(usage, "completion_tokens", 0) or 0),
                )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        return Llm2Result(
            chosen_role_slug=str(data.get("chosen_role_slug", "")).strip(),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=str(data.get("reasoning", ""))[:500],
        )
    except Exception as exc:
        logger.warning("[llm2] call failed: %s", exc)
        return Llm2Result(
            chosen_role_slug=candidates[0].slug if candidates else "",
            confidence=0.0,
            reasoning=f"LLM2 error: {exc}",
        )


async def _resolve_via_llm2(
    *,
    case: CaseLiteral,
    tied_candidates: list[RoleSignal],
    r_and_r_text: str,
    llm2_fn: Llm2Fn | None,
    alias_collision: bool,
    queue_reason: str,
) -> Stage4Decision:
    fn = llm2_fn or llm2_resolve_role
    result = await fn(r_and_r_text=r_and_r_text, candidates=tied_candidates)

    if result.confidence < LLM2_MIN_CONFIDENCE:
        return Stage4Decision(
            case=case,
            chosen_role=None,
            confidence=result.confidence,
            llm2_fired=True,
            llm2_reasoning=result.reasoning,
            alias_collision_detected=alias_collision,
            queued=True,
            reasoning=(
                f"LLM2 unsure (confidence {result.confidence:.2f} < "
                f"{LLM2_MIN_CONFIDENCE}); queueing"
            ),
        )

    # LLM2 chose a slug — find the matching RoleSignal from the candidates so
    # we don't synthesize fake role_id values downstream.
    chosen = next(
        (c for c in tied_candidates if c.slug == result.chosen_role_slug),
        None,
    )
    if chosen is None:
        return _queue(case, reason="llm2_hallucinated_slug",
                      detail=f"LLM2 returned {result.chosen_role_slug!r}")

    return Stage4Decision(
        case=case,
        chosen_role=chosen,
        confidence=result.confidence,
        llm2_fired=True,
        llm2_reasoning=result.reasoning,
        alias_collision_detected=alias_collision,
        queued=False,
        reasoning=f"LLM2 picked {chosen.slug} (confidence {result.confidence:.2f})",
    )


# ── Stage 5: DB write helpers ────────────────────────────────────────────────

def _vec_to_str(emb: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in emb) + "]"


def _parse_vec(raw) -> list[float]:
    """Parse pgvector output (str like '[0.1,0.2]') into a list of floats."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    s = str(raw).strip().lstrip("[").rstrip("]")
    return [float(p) for p in s.split(",") if p.strip()]


def log_alias_collision(
    conn,
    *,
    jd_run_id: str,
    decision: "Stage4Decision",
    role_name_input: str,
    s3: Stage3Result,
) -> int | None:
    """Insert one row into alias_collision_log and return its id."""
    alias_top = s3.alias_match_roles[0].role_id if s3.alias_match_roles else None
    kra_top = s3.kra_match_roles[0].role_id if s3.kra_match_roles else None
    skill_top = s3.skill_match_roles[0].role_id if s3.skill_match_roles else None
    resolution = decision.chosen_role.role_id if decision.chosen_role else None

    sql = """
        INSERT INTO alias_collision_log
            (jd_run_id, case_type, role_name_input,
             alias_top_role_id, kra_top_role_id, skill_top_role_id,
             resolution_role_id, llm2_reasoning)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    params = (
        jd_run_id, decision.case, role_name_input,
        alias_top, kra_top, skill_top,
        resolution, decision.llm2_reasoning,
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return int(row[0]) if row else None


def enqueue_candidate(
    conn,
    *,
    jd_run_id: str,
    decision: "Stage4Decision",
    role_name_input: str,
    r_and_r_text: str,
    s3: Stage3Result,
    reason: str,
) -> int | None:
    """Insert one row into new_role_candidate_queue and return its id."""
    top_kra_score = (
        float(s3.kra_match_roles[0].score) if s3.kra_match_roles else None
    )
    if len(s3.kra_match_roles) >= 2:
        margin = float(s3.kra_match_roles[0].score - s3.kra_match_roles[1].score)
    else:
        margin = None
    candidate_ids = [r.role_id for r in s3.kra_match_roles[:5]] or None

    sql = """
        INSERT INTO new_role_candidate_queue
            (jd_run_id, case_type, role_name_input, r_and_r_text,
             top_kra_score, kra_margin, candidate_role_ids, reason, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending')
        RETURNING id
    """
    params = (
        jd_run_id, decision.case, role_name_input, r_and_r_text,
        top_kra_score, margin, candidate_ids, reason,
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return int(row[0]) if row else None


def detect_unknown_skills(
    input_skills: list[dict],
    matched_lower: set[str],
) -> list[dict]:
    """Return JD-input skills whose lowercased name didn't match the
    canonical catalog. ``input_skills`` items shape: ``{skill_name, is_primary}``.
    """
    return [
        s for s in input_skills
        if s.get("skill_name", "").strip().lower() not in matched_lower
    ]


def enqueue_skill_candidates(
    conn,
    *,
    jd_run_id: str,
    role_id: int,
    unknown_skills: list[dict],
) -> list[int]:
    """Insert one row per unknown skill, return list of new ids."""
    if not unknown_skills:
        return []

    sql = """
        INSERT INTO new_skill_candidate_queue
            (jd_run_id, role_id, skill_name, is_primary, status)
        VALUES (%s, %s, %s, %s, 'pending')
        ON CONFLICT (jd_run_id, role_id, skill_name) DO NOTHING
        RETURNING id
    """
    ids: list[int] = []
    with conn.cursor() as cur:
        for s in unknown_skills:
            cur.execute(
                sql,
                (
                    jd_run_id, role_id,
                    s.get("skill_name", "")[:200],
                    bool(s.get("is_primary", False)),
                ),
            )
            row = cur.fetchone()
            if row:
                ids.append(int(row[0]))
    return ids


def enqueue_kra_candidate(
    conn,
    *,
    jd_run_id: str,
    role_id: int,
    r_and_r_text: str,
    best_kra_similarity: float | None,
) -> int | None:
    """Capture an R&R block as a candidate KRA for a confidently classified
    role. Returns the new id."""
    sql = """
        INSERT INTO new_kra_candidate_queue
            (jd_run_id, role_id, r_and_r_text, best_kra_similarity, status)
        VALUES (%s, %s, %s, %s, 'pending')
        RETURNING id
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (jd_run_id, role_id, r_and_r_text,
             float(best_kra_similarity) if best_kra_similarity is not None else None),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None


def query_matched_skill_names(
    conn,
    skill_names: list[str],
    *,
    schema: str,
) -> set[str]:
    """Return lowercased skill names that exist in the canonical catalog
    (either as canonical_skills.display_name or skill_aliases.alias_lower)."""
    if not skill_names:
        return set()
    lower = [s.strip().lower() for s in skill_names if s.strip()]
    if not lower:
        return set()
    qs = '"' + schema.replace('"', '""') + '"'
    sql = f"""
        SELECT LOWER(display_name) FROM {qs}.canonical_skills
         WHERE LOWER(display_name) = ANY(%s)
        UNION
        SELECT alias_lower FROM {qs}.skill_aliases
         WHERE alias_lower = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (lower, lower))
        return {row[0] for row in cur.fetchall()}


def is_absolutely_new_role(
    decision: "Stage4Decision",
    s3: Stage3Result,
    *,
    stage1_resolved: Stage1RoleResolution | None = None,
) -> bool:
    """A role is 'absolutely new' when the JD title doesn't canonically
    resolve AND skill signal is weak.

    Priority:
      1. If ``stage1_resolved`` is provided and not None → role IS in
         catalog (matched by display_name or curated alias) → not new.
      2. Else if skill_top is decent (>= NEW_ROLE_SKILL_CEILING) → assume
         the JD's skill profile points at an existing role.
      3. Else if alias_top crosses NEW_ROLE_ALIAS_FLOOR (default 0.85, only
         exact-ish hits) → role exists.
      4. Otherwise → new role candidate.

    The Stage 3b alias score is intentionally NOT consulted before
    ``stage1_resolved``. Trigram noise (e.g., "AI Engineer" → "ar-vr-
    engineer" 0.71) used to block v3 trigger here; now stage1_resolved=None
    is the authoritative "not in catalog" signal.
    """
    if stage1_resolved is not None:
        return False
    # Safety net: if Stage 4 confidently classified into an existing canonical
    # role (Case A/B/C/D-resolved/F-resolved), that role is already in the
    # catalog by definition — don't fire v3. This protects against
    # threshold-tuning issues with the embedding fallback in
    # resolve_stage1_role (a fuzzy variant title like "Mid Level IOS
    # Developer" might score 0.83 against ios-engineer's aliases, just below
    # STAGE1_EMBEDDING_THRESHOLD=0.85, yet Stage 4 still found the right
    # canonical match via signal convergence).
    if decision.chosen_role is not None and not decision.queued:
        return False
    # An exact-ish alias hit (>= NEW_ROLE_ALIAS_FLOOR = 0.85) is a fallback
    # "exists in catalog" signal for callers that didn't compute
    # stage1_resolved. Trigram noise (typically < 0.80) does NOT count.
    if (
        s3.alias_match_roles
        and s3.alias_match_roles[0].score >= NEW_ROLE_ALIAS_FLOOR
    ):
        return False
    # Intentionally no skill_top check: new-titled JDs (AI Engineer,
    # MLOps Engineer, etc.) typically share many skills with existing roles
    # but are themselves distinct roles worth materializing in the catalog.
    return True


def stage5_apply_updates(
    conn,
    *,
    jd_run_id: str,
    decision: "Stage4Decision",
    r_and_r_embedding: list[float] | None,
    r_and_r_text: str,
    role_name_input: str,
    s3: Stage3Result,
) -> Stage5Updates:
    """Apply consequences of the Stage 4 decision: centroid update + audit
    writes. Caller owns the transaction (conn.commit() at the end).
    """
    out = Stage5Updates(
        centroid_updated=False, centroid_n_after=None,
        queue_entry_id=None, collision_log_id=None,
    )

    # 1. Queue first so the row exists when the audit log references it.
    if decision.queued:
        reason = (
            "llm2_unsure" if decision.llm2_fired
            else "low_kra"
        )
        out.queue_entry_id = enqueue_candidate(
            conn, jd_run_id=jd_run_id, decision=decision,
            role_name_input=role_name_input, r_and_r_text=r_and_r_text,
            s3=s3, reason=reason,
        )

    # 2. Alias-collision audit for Cases B and D (whenever flagged).
    if decision.alias_collision_detected:
        out.collision_log_id = log_alias_collision(
            conn, jd_run_id=jd_run_id, decision=decision,
            role_name_input=role_name_input, s3=s3,
        )

    # 3. Centroid update — only for confident classifications.
    if (
        decision.chosen_role is not None
        and not decision.queued
        and r_and_r_embedding
    ):
        n_after = update_kra_centroid(
            conn,
            role_id=decision.chosen_role.role_id,
            embedding=r_and_r_embedding,
            jd_run_id=jd_run_id,
        )
        out.centroid_updated = True
        out.centroid_n_after = n_after

    return out


def update_kra_centroid(
    conn,
    *,
    role_id: int,
    embedding: list[float],
    jd_run_id: str,
) -> int:
    """Apply the D-1 rolling-mean update and persist. Returns new n."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT centroid_embedding, n_jds_classified "
            "FROM role_kra_centroids WHERE role_id = %s FOR UPDATE",
            (role_id,),
        )
        row = cur.fetchone()

        if row is None:
            new_centroid, new_n = rolling_centroid(None, 0, embedding)
            cur.execute(
                "INSERT INTO role_kra_centroids "
                "(role_id, centroid_embedding, n_jds_classified, last_jd_run_id) "
                "VALUES (%s, %s, %s, %s)",
                (role_id, _vec_to_str(new_centroid), new_n, jd_run_id),
            )
            return new_n

        old_centroid = _parse_vec(row[0])
        n = int(row[1])
        new_centroid, new_n = rolling_centroid(old_centroid, n, embedding)
        cur.execute(
            "UPDATE role_kra_centroids "
            "SET centroid_embedding = %s, n_jds_classified = %s, "
            "    last_jd_run_id = %s, updated_at = NOW() "
            "WHERE role_id = %s",
            (_vec_to_str(new_centroid), new_n, jd_run_id, role_id),
        )
        return new_n


# ── Stage 5: rolling-mean centroid math ──────────────────────────────────────

def rolling_centroid(
    old: list[float] | None,
    n: int,
    new_emb: list[float],
) -> tuple[list[float], int]:
    """D-1 rolling mean: new_centroid = old + (new_emb - old) / (n + 1).

    First update (old=None, n=0) seeds the centroid with new_emb.
    """
    if old is None:
        return list(new_emb), 1

    if len(old) != len(new_emb):
        raise ValueError(
            f"embedding dim mismatch: centroid={len(old)} new={len(new_emb)}"
        )

    next_n = n + 1
    inv = 1.0 / next_n
    return [o + (v - o) * inv for o, v in zip(old, new_emb)], next_n


def _queue(case: CaseLiteral, *, reason: str, detail: str) -> Stage4Decision:
    return Stage4Decision(
        case=case,
        chosen_role=None,
        confidence=0.0,
        llm2_fired=False,
        llm2_reasoning=None,
        alias_collision_detected=False,
        queued=True,
        reasoning=f"{reason}: {detail}",
    )
