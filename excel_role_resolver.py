"""Runtime Excel-taxonomy resolver for Stage 4 Case NEW intercept.

When a JD's role title doesn't resolve against the catalog, this module:
  1. Cosine-ranks the Excel taxonomy (174 generalized industry roles) against
     the JD's R&R embedding (reusing Stage 2's output, no extra Azure call).
  2. Asks gpt-4o-mini to pick the best canonical match from the top-15
     candidates, or return empty when no candidate fits.
  3. If confidence >= EXCEL_MIN_CONFIDENCE, upserts a shell `roles` row +
     `role_aliases` from the Excel data and returns a Stage4Decision with
     case='EXCEL_NEW'. The catalog grows; v3 can still fire in background
     for KRA/dimension enrichment, but it's now optional rather than gating.

If the resolver returns no match (LLM-proposed novel role or low confidence),
the caller (jd_classifier.stage4_route_decision) falls through to the
existing Case NEW synth path unchanged.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from excel_taxonomy_loader import ExcelRoleEntry, load_taxonomy
from jd_similarity_matcher import _cosine, _pg_connect, _qs
from skill_library_v3.db.repository import slugify

logger = logging.getLogger(__name__)


EXCEL_MIN_CONFIDENCE: float = 0.70
EXCEL_TOP_K_CANDIDATES: int = 15


def _load_taxonomy_cached() -> tuple[ExcelRoleEntry, ...]:
    """Indirection so tests can monkeypatch the taxonomy without touching the
    lru_cache on `load_taxonomy`."""
    return load_taxonomy()


@dataclass(frozen=True)
class ExcelMatchResult:
    matched_entry: ExcelRoleEntry | None
    confidence: float
    reasoning: str
    candidates_considered: tuple[str, ...]


_SYSTEM_PROMPT = (
    "You classify a job description into a canonical role from a fixed list "
    "of candidates. The candidates are generalized industry-standard role "
    "names shown in the format `- <role_name> [<domain>] — <rationale> "
    "(aliases: ...)`. Pick the candidate whose responsibilities best match "
    'the JD, OR set "matched_role_name" to "" if no candidate fits. Return '
    'JSON with these keys exactly: "matched_role_name" (the role_name part '
    "ONLY, exactly as it appears before the `[<domain>]` bracket — do NOT "
    'include the bracket or anything after it; OR ""), "confidence" '
    '(float 0.0-1.0), "reasoning" (one sentence).'
)


def _build_user_prompt(
    *,
    jd_title: str,
    r_and_r_text: str,
    candidates: tuple[ExcelRoleEntry, ...],
) -> str:
    cand_block = "\n".join(
        f"- {c.role} [{c.domain}] — {c.rationale} "
        f"(aliases: {', '.join(c.aliases) if c.aliases else 'none'})"
        for c in candidates
    )
    return (
        f"JD title: {jd_title}\n\n"
        f"JD responsibilities:\n{(r_and_r_text or '')[:3000]}\n\n"
        f"Candidate roles:\n{cand_block}\n\n"
        "Pick the best match VERBATIM from the candidate list above, or "
        'return "" for matched_role_name if no candidate fits.'
    )


async def _call_resolver_llm(prompt: str, *, cost_acc=None) -> dict:
    """Fast-tier (gpt-4o-mini) JSON-mode call. Returns the parsed payload or
    a safe-empty dict on any failure. Wires `cost_acc` if provided."""
    from llm_client import FAST_MODEL, get_fast_client

    client = get_fast_client()
    if client is None:
        return {
            "matched_role_name": "",
            "confidence": 0.0,
            "reasoning": "fast client unavailable",
        }
    try:
        resp = await client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=20,
        )
        if cost_acc is not None:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                cost_acc.add(
                    FAST_MODEL,
                    int(getattr(usage, "prompt_tokens", 0) or 0),
                    int(getattr(usage, "completion_tokens", 0) or 0),
                )
        raw = resp.choices[0].message.content or "{}"
        return json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[excel-resolver] LLM call failed: %s", exc)
        return {
            "matched_role_name": "",
            "confidence": 0.0,
            "reasoning": f"LLM error: {exc}",
        }


async def resolve_via_excel(
    *,
    jd_title: str,
    r_and_r_text: str,
    r_and_r_embedding: list[float] | None,
    cost_acc=None,
) -> ExcelMatchResult:
    """Cosine-narrow Excel taxonomy to top-15 → fast-tier LLM picks one or
    returns empty. Hallucinated names (not in candidate list) are rejected.
    Returns matched_entry=None when no high-confidence match is found, so
    the caller can fall through to the existing Case NEW synth path."""
    taxonomy = _load_taxonomy_cached()
    candidates = rank_excel_candidates(
        r_and_r_embedding, taxonomy, top_n=EXCEL_TOP_K_CANDIDATES,
    )
    if not candidates:
        return ExcelMatchResult(
            matched_entry=None,
            confidence=0.0,
            reasoning="no candidates (taxonomy empty or no embedding)",
            candidates_considered=(),
        )

    prompt = _build_user_prompt(
        jd_title=jd_title, r_and_r_text=r_and_r_text, candidates=candidates,
    )
    data = await _call_resolver_llm(prompt, cost_acc=cost_acc)
    matched_name = str(data.get("matched_role_name", "")).strip()
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    reasoning = str(data.get("reasoning", ""))[:500]

    # Defensive: strip the `[<domain>]` suffix if the LLM included it.
    if "[" in matched_name:
        matched_name = matched_name.split("[", 1)[0].strip()

    # Hallucination guard: matched name MUST appear in our top-K candidate
    # list. Also try a case-insensitive fallback before giving up.
    by_name = {c.role: c for c in candidates}
    matched_entry = by_name.get(matched_name)
    if matched_entry is None and matched_name:
        by_name_lower = {c.role.lower(): c for c in candidates}
        matched_entry = by_name_lower.get(matched_name.lower())

    # Gate: valid match AND confidence above threshold
    if matched_entry is None or confidence < EXCEL_MIN_CONFIDENCE:
        return ExcelMatchResult(
            matched_entry=None,
            confidence=confidence,
            reasoning=reasoning or "no match",
            candidates_considered=tuple(c.role for c in candidates),
        )
    return ExcelMatchResult(
        matched_entry=matched_entry,
        confidence=confidence,
        reasoning=reasoning,
        candidates_considered=tuple(c.role for c in candidates),
    )


# ── Catalog persistence ─────────────────────────────────────────────────────


def _domain_to_archetype(domain: str) -> str:
    """Map Excel Domain into the role_archetype free-text label used elsewhere."""
    d = (domain or "").lower()
    if "software" in d:
        return "Engineering"
    if "cloud" in d or "devops" in d or "platform" in d:
        return "DevOps"
    if "data" in d or "analytic" in d:
        return "Data"
    if "security" in d:
        return "Security"
    if "test" in d or "qa" in d or "quality" in d:
        return "QA"
    return "Engineering"


def upsert_excel_role(entry: ExcelRoleEntry) -> int | None:
    """Insert a shell `roles` row + its aliases. Idempotent via ON CONFLICT
    DO NOTHING. When the role already exists, looks it up by slug.

    Returns the catalog role_id, or None on DB error (caller falls through
    to existing Case NEW synth path without persisting)."""
    slug = slugify(entry.role)
    archetype = _domain_to_archetype(entry.domain)
    qs = _qs()

    try:
        conn = _pg_connect()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[excel-resolver] DB connect failed: %s", exc)
        return None

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {qs}.roles (slug, display_name, role_archetype, source)
                    VALUES (%s, %s, %s, 'MANUAL_CURATION'::{qs}.entity_source)
                    ON CONFLICT (slug) DO NOTHING
                    RETURNING id
                    """,
                    (slug, entry.role, archetype),
                )
                row = cur.fetchone()
                if row is None:
                    # Already existed — look it up by slug
                    cur.execute(
                        f"SELECT id FROM {qs}.roles WHERE slug = %s",
                        (slug,),
                    )
                    row = cur.fetchone()
                if row is None:
                    logger.warning(
                        "[excel-resolver] upsert produced no role_id for slug=%s", slug,
                    )
                    return None
                role_id = int(row[0])

                # Insert aliases idempotently
                for alias in entry.aliases:
                    alias_text = alias.strip()[:200]
                    if not alias_text:
                        continue
                    cur.execute(
                        f"""
                        INSERT INTO {qs}.role_aliases
                            (role_id, alias_text, alias_type, match_strategy, is_primary)
                        VALUES (%s, %s,
                                'COLLOQUIAL'::{qs}.alias_type,
                                'CASE_INSENSITIVE'::{qs}.match_strategy,
                                FALSE)
                        ON CONFLICT (role_id, alias_text) DO NOTHING
                        """,
                        (role_id, alias_text),
                    )
                return role_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("[excel-resolver] upsert raised: %s", exc)
        return None
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def role_has_kras(role_id: int) -> bool:
    """True when at least one `role_kras` row exists for this role (v3 has
    already enriched it). Used by main.py to skip duplicate v3 dispatch on
    repeat Excel matches.

    Fail-closed: returns True on any DB error so the caller doesn't fire a
    duplicate background v3 task."""
    qs = _qs()
    try:
        conn = _pg_connect()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[excel-resolver] role_has_kras DB connect failed: %s", exc)
        return True

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM {qs}.role_kras WHERE role_id = %s LIMIT 1",
                    (role_id,),
                )
                row = cur.fetchone()
                return bool(row and int(row[0]) > 0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[excel-resolver] role_has_kras error: %s", exc)
        return True  # fail-closed
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def rank_excel_candidates(
    jd_embedding: list[float] | None,
    taxonomy: tuple[ExcelRoleEntry, ...],
    *,
    top_n: int = 15,
) -> tuple[ExcelRoleEntry, ...]:
    """Cosine-rank Excel rows against the JD R&R embedding. Returns the top_n
    entries in descending similarity order. Empty tuple when either input is
    degenerate (None embedding or empty taxonomy)."""
    if not jd_embedding or not taxonomy:
        return ()
    scored = [
        (_cosine(jd_embedding, list(e.embedding)), e)
        for e in taxonomy
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    return tuple(e for _, e in scored[:top_n])
