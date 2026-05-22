"""Stage 4 (Decision Routing) + Stage 5 (Post-Classification Updates) for the
JD role-classifier pipeline.

Stage 4 reads the three signal lists from :class:`Stage3Result` and routes the
JD into one of six cases A–F. Cases D and F may fire an LLM2 tie-breaker.
Stage 5 applies the consequences: rolling-mean centroid update + audit rows.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

from excel_role_resolver import (
    resolve_via_excel as _resolve_via_excel,
    upsert_excel_role as _upsert_excel_role,
)
from jd_similarity_matcher import RoleSignal, Stage3Result
from skill_library_v3.db.repository import slugify as _slugify_canonical

_logger = logging.getLogger(__name__)


# ── Tunable constants ────────────────────────────────────────────────────────

# Stage 3c now uses mean-of-top-3 per-sentence scoring (not MAX).
# Observed score ranges with descriptive KRA texts + mean-of-top-3:
#   ~0.40 (weak / wrong role)  →  ~0.65 (strong / correct role)
# KRA_MIN_SCORE: floor below which no classification is attempted.
# KRA_MARGIN_DEFAULT: minimum gap between #1 and #2 role scores required
#   to classify without LLM2; mean-of-top-3 naturally widens margins so
#   0.05 remains a reasonable floor.
# KRA_TIE_BAND: scores within this band are treated as a statistical tie → Case F.
KRA_MIN_SCORE: float = 0.45
KRA_MARGIN_DEFAULT: float = 0.05
KRA_MARGIN_ON_COLLISION: float = 0.08
KRA_TIE_BAND: float = 0.02
SKILL_TIE_BAND: float = 0.05
LLM2_MIN_CONFIDENCE: float = 0.70

# Gap #6 bypass: when alias is an exact match and skill agrees with it at
# moderate confidence, skip KRA gating entirely. Prevents valid classifications
# from being queued when the JD title + skill list make the role unambiguous.
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

CaseLiteral = Literal["A", "B", "C", "D", "E", "F", "NEW", "EXCEL_NEW"]


def _slugify_for_new_role(text: str) -> str:
    # Mirrors skill_library_v3.db.repository.slugify so the UI preview slug
    # matches what v3's upsert_role will actually persist.
    out: list[str] = []
    last_dash = False
    for ch in (text or "").strip().lower():
        if ch.isalnum():
            out.append(ch)
            last_dash = False
        elif ch in (" ", "-", "_", "\\", ".", ",", "/", "+"):
            if not last_dash and out:
                out.append("-")
                last_dash = True
    return "".join(out).strip("-") or "unknown"


async def _attempt_excel_classification(
    *,
    jd_title: str,
    r_and_r_text: str,
    r_and_r_embedding: list[float] | None,
    role_name_input: str,
    cost_acc=None,
) -> "Stage4Decision | None":
    """Try to classify the JD into a generalized Excel-taxonomy role before
    synthesizing a brand-new one. Returns a Stage4Decision with
    case='EXCEL_NEW' when the resolver finds a high-confidence match AND the
    shell-row upsert succeeds. Returns None otherwise so the caller falls
    through to the existing Case NEW synth path.

    The returned decision still sets `is_new_role=True` and
    `new_role_display_name` to the *Excel canonical* role name (NOT the
    verbatim JD title), so main.py's downstream v3 dispatch enriches the
    correct catalog row in the background.
    """
    if not r_and_r_embedding:
        return None
    try:
        result = await _resolve_via_excel(
            jd_title=jd_title,
            r_and_r_text=r_and_r_text,
            r_and_r_embedding=r_and_r_embedding,
            cost_acc=cost_acc,
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning("[stage4/excel] resolver raised: %s", exc)
        return None

    if result.matched_entry is None:
        return None

    entry = result.matched_entry
    role_id = _upsert_excel_role(entry)
    if role_id is None:
        _logger.warning(
            "[stage4/excel] upsert returned None for role=%r — falling through to synth",
            entry.role,
        )
        return None

    slug = _slugify_canonical(entry.role)
    chosen = RoleSignal(
        role_id=role_id,
        slug=slug,
        display_name=entry.role,
        score=result.confidence,
        signal_type="excel_match",
    )
    return Stage4Decision(
        case="EXCEL_NEW",
        chosen_role=chosen,
        confidence=result.confidence,
        llm2_fired=False,
        llm2_reasoning=None,
        alias_collision_detected=False,
        queued=False,
        reasoning=(
            f"JD title '{role_name_input}' not in catalog; Excel taxonomy "
            f"matched '{entry.role}' (confidence {result.confidence:.2f}): "
            f"{result.reasoning}"
        ),
        is_new_role=True,
        new_role_display_name=entry.role,
        new_role_slug=slug,
    )


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
    is_new_role: bool = False
    new_role_display_name: str | None = None
    new_role_slug: str | None = None


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

_ALIAS_MULTI_TIE_THRESHOLD: int = 3


def _alias_multi_tie(alias: list[RoleSignal]) -> list[RoleSignal] | None:
    """Detect the "multiple roles claim the same JD title at 1.0" pattern.

    The Stage 3b alias table over-includes generic titles ("Software Engineer"
    is listed as an alias for backend-engineer, ar-vr-engineer, and
    game-developer simultaneously). When the same JD title hits 3+ catalog
    roles all at score 1.0, the alias signal is ambiguous noise — NOT
    authority. Returns the tied cohort (so callers can fire LLM2 over it),
    or None when there's no multi-tie.
    """
    if len(alias) < _ALIAS_MULTI_TIE_THRESHOLD:
        return None
    top_score = alias[0].score
    if top_score < ALIAS_EXACT_MIN_SCORE:
        return None
    tied = [a for a in alias if a.score >= ALIAS_EXACT_MIN_SCORE]
    if len(tied) < _ALIAS_MULTI_TIE_THRESHOLD:
        return None
    return tied


def _dedupe_by_role_id(*signals: RoleSignal | None) -> list[RoleSignal]:
    """Build a deduplicated candidate list preserving order; drops None."""
    seen: set[int | None] = set()
    out: list[RoleSignal] = []
    for s in signals:
        if s is None:
            continue
        if s.role_id in seen:
            continue
        seen.add(s.role_id)
        out.append(s)
    return out


async def stage4_route_decision(
    s3: Stage3Result,
    r_and_r_text: str,
    role_name_input: str,
    *,
    stage1_resolved: Stage1RoleResolution | None = None,
    llm2_fn: Llm2Fn | None = None,
) -> Stage4Decision:
    """Body-driven Stage 4 routing — skill + KRA are the primary decision
    signals; the verbatim JD title is advisory-only.

    Decision tree (top-down):

      1. Multi-alias-tie (3+ roles share the JD title at score 1.0) → fire
         LLM2 over the tied alias cohort. The shared title can't itself
         disambiguate; let LLM2 read the JD R&R and pick.

      2. Skill+KRA convergence (top_skill.role_id == top_kra.role_id AND
         kra_top.score >= KRA_MIN_SCORE) → Case A. Alias becomes advisory
         (audited if it disagrees).

      3. Skill-confident + KRA-noisy (top_skill.score >= 0.50 AND KRA
         margin tight) → Case B skill-led.

      4. KRA-only signal (kra_top.score >= KRA_MIN_SCORE AND skill_top
         absent/weak < 0.20). If KRA top-2 within KRA_TIE_BAND, fire LLM2
         over the tied KRA cohort. Otherwise Case B KRA-led.

      5. Single alias hit (alias_top.score >= ALIAS_EXACT_MIN_SCORE,
         non-multi-tie) AND candidates differ → fire LLM2 with [alias,
         kra_top, skill_top] dedup'd. Alias is a candidate, not the answer.

      6. No usable signal (KRA + skill both weak) → Excel intercept; if
         Excel also returns None, synth Case NEW.

      7. Fallback → LLM2 over [kra_top, skill_top] dedup'd, or queue.

    The Stage-1 verbatim role title is intentionally NOT consulted as a
    decision driver. It's still computed (for audit logs / Excel intercept
    JD title) but doesn't gate any branch.

    Pure-logic branches (cases 2, 3, 4-non-tied) never call llm2_fn. Cases
    1, 4-tied, 5, 7 fire LLM2.
    """
    # stage1_resolved is intentionally ignored in routing (body-driven mode).
    # We still accept the kwarg for backwards compatibility with callers.
    _ = stage1_resolved
    skill = s3.skill_match_roles
    alias = s3.alias_match_roles
    kra = s3.kra_match_roles

    top_skill = skill[0] if skill else None
    top_alias = alias[0] if alias else None
    top_kra = kra[0] if kra else None
    top_kra_score = top_kra.score if top_kra else 0.0
    second_kra_score = kra[1].score if len(kra) > 1 else 0.0
    kra_margin = top_kra_score - second_kra_score

    r_and_r_embedding = getattr(s3, "r_and_r_embedding", None)

    # ── Branch 1: multi-alias-tie → LLM2 disambiguation ───────────────────────
    # 3+ catalog roles share the JD title at score 1.0 — the title is
    # ambiguous noise. Let LLM2 read the JD body and pick from the cohort.
    tied_alias_cohort = _alias_multi_tie(alias)
    if tied_alias_cohort is not None:
        return await _resolve_via_llm2(
            case="D",
            tied_candidates=tied_alias_cohort,
            r_and_r_text=r_and_r_text,
            llm2_fn=llm2_fn,
            alias_collision=True,
            queue_reason="multi_alias_tie",
        )

    # ── Branch 2: skill+KRA convergence → Case A ──────────────────────────────
    # When skill_top and kra_top agree AND KRA has a meaningful signal,
    # classify directly. Alias is logged but doesn't gate.
    if (
        top_skill is not None
        and top_kra is not None
        and top_skill.role_id == top_kra.role_id
        and top_kra_score >= KRA_MIN_SCORE
    ):
        alias_note = ""
        if top_alias is not None:
            if top_alias.role_id == top_kra.role_id:
                alias_note = f"; alias agrees ({top_alias.slug})"
            else:
                alias_note = f"; alias differs (advisory: {top_alias.slug})"
        return Stage4Decision(
            case="A",
            chosen_role=top_kra,
            confidence=top_kra_score,
            llm2_fired=False,
            llm2_reasoning=None,
            alias_collision_detected=(
                top_alias is not None
                and top_alias.role_id != top_kra.role_id
            ),
            queued=False,
            reasoning=(
                f"Skill+KRA converge on {top_kra.slug} "
                f"({top_skill.score:.2f}/{top_kra_score:.2f}){alias_note}"
            ),
        )

    # ── Branch 3: skill-confident + KRA-noisy → Case B skill-led ──────────────
    # When skill_top has high witness confidence AND KRA top-2 are tight,
    # trust the skill profile. Disambiguate via LLM2 if skill and KRA top
    # disagree (let LLM2 pick from both).
    if (
        top_skill is not None
        and top_skill.score >= SKILL_WITNESS_MIN_SCORE
        and top_kra is not None
        and kra_margin < KRA_MARGIN_DEFAULT
        and top_skill.role_id != top_kra.role_id
    ):
        return await _resolve_via_llm2(
            case="D",
            tied_candidates=_dedupe_by_role_id(top_skill, top_kra),
            r_and_r_text=r_and_r_text,
            llm2_fn=llm2_fn,
            alias_collision=False,
            queue_reason="skill_vs_noisy_kra",
        )

    # ── Branch 4: KRA-only signal (skill absent/weak) ─────────────────────────
    # No usable skill signal — rely on KRA. If KRA top-2 are within tie
    # band, fire LLM2 over the tied KRA cohort. Otherwise classify by KRA.
    skill_weak = top_skill is None or top_skill.score < 0.20
    if skill_weak and top_kra is not None and top_kra_score >= KRA_MIN_SCORE:
        if kra_margin < KRA_TIE_BAND:
            tied_kra = _tied_cohort(kra, KRA_TIE_BAND)
            return await _resolve_via_llm2(
                case="F",
                tied_candidates=tied_kra,
                r_and_r_text=r_and_r_text,
                llm2_fn=llm2_fn,
                alias_collision=False,
                queue_reason="kra_tie",
            )
        return Stage4Decision(
            case="B",
            chosen_role=top_kra,
            confidence=top_kra_score,
            llm2_fired=False,
            llm2_reasoning=None,
            alias_collision_detected=(
                top_alias is not None
                and top_alias.role_id != top_kra.role_id
            ),
            queued=False,
            reasoning=(
                f"KRA-led classification on {top_kra.slug} "
                f"(skill weak, KRA {top_kra_score:.2f})"
            ),
        )

    # ── Branch 5: single alias hit + signals diverge → LLM2 ───────────────────
    # Alias is exact (single role at 1.0, not multi-tie). When signals
    # diverge across alias/KRA/skill, treat alias as a candidate (not the
    # answer) and let LLM2 pick from the dedup'd cohort.
    if (
        top_alias is not None
        and top_alias.score >= ALIAS_EXACT_MIN_SCORE
    ):
        candidates = _dedupe_by_role_id(top_alias, top_kra, top_skill)
        # If they all agree on one role: trivial Case A.
        if len(candidates) == 1:
            chosen = candidates[0]
            return Stage4Decision(
                case="A",
                chosen_role=chosen,
                confidence=max(chosen.score, top_kra_score),
                llm2_fired=False,
                llm2_reasoning=None,
                alias_collision_detected=False,
                queued=False,
                reasoning=(
                    f"All signals agree on {chosen.slug}"
                ),
            )
        # Multiple candidates — LLM2 disambiguates from JD body.
        return await _resolve_via_llm2(
            case="D",
            tied_candidates=candidates,
            r_and_r_text=r_and_r_text,
            llm2_fn=llm2_fn,
            alias_collision=True,
            queue_reason="alias_vs_body",
        )

    # ── Branch 6: no usable signal → Excel intercept → Case NEW synth ─────────
    # Either KRA is too weak (< 0.45) and skill is also weak, OR everything
    # is empty. Try the Excel taxonomy first; fall back to synth Case NEW.
    excel_decision = await _attempt_excel_classification(
        jd_title=role_name_input,
        r_and_r_text=r_and_r_text,
        r_and_r_embedding=r_and_r_embedding,
        role_name_input=role_name_input,
        cost_acc=None,
    )
    if excel_decision is not None:
        return excel_decision

    if role_name_input:
        new_slug = _slugify_for_new_role(role_name_input)
        synth_role = RoleSignal(
            role_id=None,
            slug=new_slug,
            display_name=role_name_input,
            score=0.0,
            signal_type="new_role_synth",
        )
        return Stage4Decision(
            case="NEW",
            chosen_role=synth_role,
            confidence=0.0,
            llm2_fired=False,
            llm2_reasoning=None,
            alias_collision_detected=False,
            queued=False,
            reasoning=(
                f"No catalog signal for '{role_name_input}': "
                f"KRA top={top_kra_score:.2f} (<{KRA_MIN_SCORE}), "
                f"skill top={(top_skill.score if top_skill else 0):.2f} (<0.20). "
                f"Finalizing as new role; v3 enriches in background."
            ),
            is_new_role=True,
            new_role_display_name=role_name_input,
            new_role_slug=new_slug,
        )

    return _queue(
        "E",
        reason="no_signal",
        detail=(
            f"KRA top={top_kra_score:.2f}, skill top="
            f"{(top_skill.score if top_skill else 0):.2f}, "
            f"alias_top={(top_alias.score if top_alias else 0):.2f}"
        ),
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
