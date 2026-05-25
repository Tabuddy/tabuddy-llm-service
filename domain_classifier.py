"""Two-/three-step LLM domain-then-role classifier.

Fires after Stage 3b alias-trust (Branch 1.5) didn't claim the JD. Three calls
to `gpt-5.4-mini` (the SKILL_MODEL tier, JSON-mode):

  §A  pick 1-of-21 family labels (18 tech + 3 non-tech buckets)
  §B  pick 1-of-N roles in that family from the DB + extract matched
      skills / dimensions / KRAs from the JD body
  §C  pick 1-of-M sub-roles — ONLY when the §B pick has is_branchable=True

Returns `None` (caller falls through to Stage 3a/3c + branches 2-7) when:
  - §A is_tech is False, family isn't in ALLOWED_TECH_FAMILIES, or returns "fail"
  - in-domain DB candidates list is empty
  - §B returns role_slug="" or a slug not in candidates
  - §B confidence < DOMAIN_ROLE_MIN_CONFIDENCE
Sub-role failures degrade gracefully (parent stays, sub_role=None).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jd_similarity_matcher import _pg_connect, _qs

logger = logging.getLogger(__name__)


# 18 tech families — must match the §A label list AND roles.domain values.
# CRM (Non-Salesforce) and ERP — Other Platforms have specific punctuation
# the LLM must reproduce verbatim.
ALLOWED_TECH_FAMILIES: tuple[str, ...] = (
    "AI / ML",
    "CRM (Non-Salesforce)",
    "Cloud Engineering",
    "Data Engineering & Analytics",
    "Database Specialists",
    "DevOps & Platform",
    "ERP — Other Platforms",
    "Emerging Tech",
    "Hardware Engineering",
    "ITSM & Automation",
    "Infrastructure & Networking",
    "Other Specialty",
    "SAP",
    "Salesforce",
    "Security",
    "Software Engineering",
    "Tech-Adjacent",
    "Testing & Quality",
)

DOMAIN_ROLE_MIN_CONFIDENCE: float = 0.75
SUB_ROLE_MIN_CONFIDENCE: float = 0.70
MAX_RNR_CHARS_FOR_PROMPT: int = 2500

_LIST_CAP: int = 20  # defensive cap on matched_skills/dimensions/kras list lengths


@dataclass(frozen=True)
class SubRolePick:
    role_id: int
    slug: str
    display_name: str
    confidence: float
    reasoning: str


@dataclass(frozen=True)
class DomainClassificationResult:
    domain: str
    role_id: int
    role_slug: str
    role_display_name: str
    confidence: float
    reasoning: str
    family_evidence: list[str]
    matched_skills: list[str]
    matched_dimensions: list[str]
    matched_kras: list[str]
    sub_role: SubRolePick | None


# ───────────────────────── prompt loading ─────────────────────────

_PROMPT_PATH = Path(__file__).resolve().parent / "domain_role_classifier_prompt.md"


def _load_prompts() -> tuple[str, str, str]:
    """Split the markdown into (§A, §B, §C) system-prompt strings."""
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    # `## §B — STEP 2 SYSTEM PROMPT` and `## §C — STEP 3 SYSTEM PROMPT` are the
    # split markers. Everything between §A header and §B header is §A; etc.
    parts_b = text.split("## §B — STEP 2 SYSTEM PROMPT", 1)
    if len(parts_b) != 2:
        raise RuntimeError("§B header not found in domain_role_classifier_prompt.md")
    step1_block = parts_b[0]
    rest = parts_b[1]
    parts_c = rest.split("## §C — STEP 3 SYSTEM PROMPT", 1)
    if len(parts_c) != 2:
        raise RuntimeError("§C header not found in domain_role_classifier_prompt.md")
    step2_block, step3_block = parts_c[0], parts_c[1]
    # Trim leading "## §A — STEP 1 SYSTEM PROMPT (family pick)" from step1 if present.
    a_header = "## §A — STEP 1 SYSTEM PROMPT"
    if a_header in step1_block:
        step1_block = step1_block.split(a_header, 1)[1]
    return step1_block.strip(), step2_block.strip(), step3_block.strip()


# ───────────────────────── DB helpers ─────────────────────────

def _fetch_roles_in_domain(domain: str) -> list[dict[str, Any]]:
    """Return [{id, slug, display_name, aliases, is_branchable}] for `domain`."""
    qs = _qs()
    try:
        conn = _pg_connect()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[domain-classifier] DB connect failed: %s", exc)
        return []
    try:
        with conn:
            with conn.cursor() as cur:
                # Exclude leaf children — those should only be reachable via
                # §C (sub-role pick) after §B picks their branchable parent.
                cur.execute(
                    f"""
                    SELECT r.id, r.slug, r.display_name, r.is_branchable,
                           COALESCE(
                             ARRAY_AGG(ra.alias_text ORDER BY ra.alias_text)
                               FILTER (WHERE ra.alias_text IS NOT NULL),
                             '{{}}'::text[]) AS aliases
                      FROM {qs}.roles r
                      LEFT JOIN {qs}.role_aliases ra ON ra.role_id = r.id
                     WHERE r.domain = %s AND r.parent_role_id IS NULL
                     GROUP BY r.id, r.slug, r.display_name, r.is_branchable
                     ORDER BY r.display_name
                    """,
                    (domain,),
                )
                return [
                    {
                        "id": int(row[0]),
                        "slug": row[1],
                        "display_name": row[2],
                        "is_branchable": bool(row[3]),
                        "aliases": list(row[4] or []),
                    }
                    for row in cur.fetchall()
                ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[domain-classifier] _fetch_roles_in_domain failed: %s", exc)
        return []
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def _fetch_subroles(parent_role_id: int) -> list[dict[str, Any]]:
    """Return children of `parent_role_id` for sub-role step 3."""
    qs = _qs()
    try:
        conn = _pg_connect()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[domain-classifier] DB connect failed: %s", exc)
        return []
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, slug, display_name, role_archetype
                      FROM {qs}.roles
                     WHERE parent_role_id = %s
                     ORDER BY display_name
                    """,
                    (parent_role_id,),
                )
                return [
                    {
                        "id": int(row[0]),
                        "slug": row[1],
                        "display_name": row[2],
                        "role_archetype": row[3] or "",
                    }
                    for row in cur.fetchall()
                ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[domain-classifier] _fetch_subroles failed: %s", exc)
        return []
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


# ───────────────────────── LLM helpers ─────────────────────────


def _accrue_usage(cost_acc, resp, model: str) -> None:
    if cost_acc is None:
        return
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    try:
        cost_acc.add(
            model,
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )
    except Exception:  # noqa: BLE001
        pass


async def _llm_pick_family(
    *,
    jd_title: str,
    r_and_r_text: str,
    nano_role: str,
    nano_aliases: list[str],
    system_prompt: str,
    cost_acc=None,
) -> dict[str, Any]:
    from llm_client import SKILL_MODEL, get_fast_client
    client = get_fast_client()
    if client is None:
        return {"stage0_result": "fail", "reason": "no_client"}
    user_prompt = (
        f"JD title: {jd_title}\n"
        f"Nano-extracted role: {nano_role}\n"
        f"Nano aliases: {', '.join(nano_aliases) if nano_aliases else 'none'}\n\n"
        f"JD responsibilities:\n{(r_and_r_text or '')[:MAX_RNR_CHARS_FOR_PROMPT]}"
    )
    try:
        resp = await client.chat.completions.create(
            model=SKILL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            timeout=20,
        )
        _accrue_usage(cost_acc, resp, SKILL_MODEL)
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[domain-classifier] step1 failed: %s", exc)
        return {"stage0_result": "fail", "reason": f"error: {exc}"}


async def _llm_pick_role_in_domain(
    *,
    jd_title: str,
    r_and_r_text: str,
    candidates: list[dict[str, Any]],
    system_prompt: str,
    cost_acc=None,
) -> dict[str, Any]:
    from llm_client import SKILL_MODEL, get_fast_client
    client = get_fast_client()
    if client is None:
        return {"role_slug": "", "confidence": 0.0, "reasoning": "no client"}
    cand_lines = [
        "- {slug}: {name} (aliases: {al})".format(
            slug=c["slug"],
            name=c["display_name"],
            al=", ".join(c["aliases"]) if c["aliases"] else "none",
        )
        for c in candidates
    ]
    user_prompt = (
        f"JD title: {jd_title}\n\n"
        f"JD responsibilities:\n{(r_and_r_text or '')[:MAX_RNR_CHARS_FOR_PROMPT]}\n\n"
        f"Candidate roles ({len(candidates)} total):\n" + "\n".join(cand_lines)
    )
    try:
        resp = await client.chat.completions.create(
            model=SKILL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            timeout=20,
        )
        _accrue_usage(cost_acc, resp, SKILL_MODEL)
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[domain-classifier] step2 failed: %s", exc)
        return {"role_slug": "", "confidence": 0.0, "reasoning": f"error: {exc}"}


async def _llm_pick_subrole(
    *,
    jd_title: str,
    r_and_r_text: str,
    parent_name: str,
    children: list[dict[str, Any]],
    system_prompt: str,
    cost_acc=None,
) -> dict[str, Any]:
    from llm_client import SKILL_MODEL, get_fast_client
    client = get_fast_client()
    if client is None:
        return {"sub_role_slug": "", "confidence": 0.0, "reasoning": "no client"}
    cand_lines = [
        f"- {c['slug']}: {c['display_name']}" for c in children
    ]
    user_prompt = (
        f"Parent role: {parent_name}\n"
        f"JD title: {jd_title}\n\n"
        f"JD responsibilities:\n{(r_and_r_text or '')[:MAX_RNR_CHARS_FOR_PROMPT]}\n\n"
        f"Child candidates ({len(children)} total):\n" + "\n".join(cand_lines)
    )
    try:
        resp = await client.chat.completions.create(
            model=SKILL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            timeout=20,
        )
        _accrue_usage(cost_acc, resp, SKILL_MODEL)
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[domain-classifier] step3 failed: %s", exc)
        return {"sub_role_slug": "", "confidence": 0.0, "reasoning": f"error: {exc}"}


# ───────────────────────── coercion helper ─────────────────────────


def _coerce_str_list(value: Any) -> list[str]:
    """Coerce a JSON value into a clean list[str]: drop non-strings,
    strip whitespace, drop empties, dedupe case-insensitively, cap length."""
    if not isinstance(value, list):
        return []
    seen_lower: set[str] = set()
    out: list[str] = []
    for v in value:
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s:
            continue
        key = s.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        out.append(s)
        if len(out) >= _LIST_CAP:
            break
    return out


# ───────────────────────── §B sanity-check helpers ─────────────────────────


def _normalise(text: str) -> str:
    return " ".join((text or "").strip().lower().split()).rstrip(".,;:")


async def _maybe_override_via_disambiguator(
    *,
    chosen: dict[str, Any],
    candidates: list[dict[str, Any]],
    nano_role: str,
    nano_aliases: list[str],
) -> dict[str, Any] | None:
    """Borderline-band sanity check for §B picks.

    Build a mini-cohort containing the LLM's pick plus every in-domain
    candidate whose display_name matches `nano_role` or any `nano_aliases`
    entry. Run the shared disambiguator; only return a candidate when it
    DIFFERS from the LLM pick AND won via TIER_B_TITLE (strongest signal).
    Returns None when the disambiguator agrees with the LLM or is
    inconclusive — the caller then keeps the LLM pick.
    """
    from role_disambiguator import disambiguate_overlap, load_candidate_enrichment

    targets = {_normalise(nano_role)} | {
        _normalise(a) for a in nano_aliases if a
    }
    targets.discard("")
    if not targets:
        return None

    cohort: list[dict[str, Any]] = [chosen]
    chosen_slug = chosen.get("slug")
    for c in candidates:
        if c.get("slug") == chosen_slug:
            continue
        if _normalise(c.get("display_name", "")) in targets:
            cohort.append(c)
    if len(cohort) < 2:
        return None  # no alternative candidate matches nano titles

    # Disambiguator expects `role_id` key; §B candidates have `id` — map.
    cohort_for_disambig: list[dict[str, Any]] = [
        {**c, "role_id": int(c["id"])} for c in cohort
    ]
    role_ids = [c["role_id"] for c in cohort_for_disambig]
    enrichment = load_candidate_enrichment(role_ids)
    result = disambiguate_overlap(
        cohort_for_disambig,
        nano_role=nano_role,
        nano_aliases=nano_aliases,
        jd_skills=[],  # §B doesn't have JD-extracted skills in scope
        kra_scores=None,  # no Stage 3c output here; Tier A skipped anyway
        enrichment=enrichment,
    )
    if result is None:
        return None
    winner_dict, tier = result
    if tier != "TIER_B_TITLE":
        # Only TIER_B is strong enough to override an LLM pick at this stage.
        return None
    if winner_dict.get("slug") == chosen_slug:
        return None  # disambiguator agrees with LLM; nothing to do
    # Return the original §B candidate (with `id` key) — not the
    # role_id-augmented dict — so downstream code stays unchanged.
    return next(
        (c for c in candidates if c.get("slug") == winner_dict.get("slug")),
        None,
    )


# ───────────────────────── entrypoint ─────────────────────────


async def classify_domain_role(
    *,
    jd_title: str,
    r_and_r_text: str,
    nano_role: str,
    nano_aliases: list[str],
    cost_acc=None,
) -> DomainClassificationResult | None:
    """Top-level: §A → §B (→ §C if branchable). See module docstring for gates."""
    step1_prompt, step2_prompt, step3_prompt = _load_prompts()

    # ── §A: family pick
    step1 = await _llm_pick_family(
        jd_title=jd_title, r_and_r_text=r_and_r_text,
        nano_role=nano_role, nano_aliases=nano_aliases,
        system_prompt=step1_prompt, cost_acc=cost_acc,
    )
    if step1.get("stage0_result") != "pass":
        return None
    family = str(step1.get("family", "")).strip()
    if family not in ALLOWED_TECH_FAMILIES:
        logger.info(
            "[domain-classifier] step1 family=%r not in ALLOWED_TECH_FAMILIES → fallthrough",
            family,
        )
        return None
    if not bool(step1.get("is_tech", False)):
        logger.info("[domain-classifier] step1 is_tech=False → fallthrough")
        return None

    # ── §B: role-in-family + skills/dimensions/KRAs
    candidates = _fetch_roles_in_domain(family)
    if not candidates:
        logger.info("[domain-classifier] no DB candidates in domain=%r → fallthrough", family)
        return None
    step2 = await _llm_pick_role_in_domain(
        jd_title=jd_title, r_and_r_text=r_and_r_text,
        candidates=candidates, system_prompt=step2_prompt, cost_acc=cost_acc,
    )
    role_slug = str(step2.get("role_slug", "")).strip()
    try:
        step2_conf = float(step2.get("confidence", 0.0))
    except (TypeError, ValueError):
        step2_conf = 0.0
    if not role_slug:
        return None
    chosen = next((c for c in candidates if c["slug"] == role_slug), None)
    if chosen is None:
        logger.info(
            "[domain-classifier] step2 hallucinated slug=%r → fallthrough", role_slug,
        )
        return None
    if step2_conf < DOMAIN_ROLE_MIN_CONFIDENCE:
        return None

    # ── §B sanity check: title-exact-match disambiguator (borderline only) ──
    # When the LLM's confidence is in the borderline band (0.70–0.85), build a
    # mini-cohort of "chosen + any in-domain candidate whose display_name
    # matches nano_role or nano_aliases" and run the deterministic
    # disambiguator over it. The override only fires when the disambiguator
    # picks a different role via TIER_B_TITLE (strongest deterministic signal)
    # — protects against §B picking a generic peer when one candidate's
    # display_name IS the JD's literal title.
    if 0.70 <= step2_conf < 0.85:
        try:
            override = await _maybe_override_via_disambiguator(
                chosen=chosen,
                candidates=candidates,
                nano_role=nano_role,
                nano_aliases=nano_aliases or [],
            )
            if override is not None:
                logger.info(
                    "[domain-classifier] §B override: %r → %r (TIER_B_TITLE)",
                    chosen["slug"], override["slug"],
                )
                chosen = override
        except Exception as exc:  # noqa: BLE001
            logger.warning("[domain-classifier] §B disambiguator failed: %s", exc)

    matched_skills = _coerce_str_list(step2.get("matched_skills"))
    matched_dimensions = _coerce_str_list(step2.get("matched_dimensions"))
    matched_kras = _coerce_str_list(step2.get("matched_kras"))
    reasoning = str(step2.get("reasoning", ""))[:500]
    family_evidence = _coerce_str_list(step1.get("evidence_anchors"))

    # ── §C: sub-role (only when chosen parent is_branchable)
    sub_role: SubRolePick | None = None
    if chosen.get("is_branchable"):
        children = _fetch_subroles(int(chosen["id"]))
        if children:
            step3 = await _llm_pick_subrole(
                jd_title=jd_title, r_and_r_text=r_and_r_text,
                parent_name=chosen["display_name"], children=children,
                system_prompt=step3_prompt, cost_acc=cost_acc,
            )
            sub_slug = str(step3.get("sub_role_slug", "")).strip()
            try:
                sub_conf = float(step3.get("confidence", 0.0))
            except (TypeError, ValueError):
                sub_conf = 0.0
            child = next((c for c in children if c["slug"] == sub_slug), None) \
                if sub_slug else None
            if child is not None and sub_conf >= SUB_ROLE_MIN_CONFIDENCE:
                sub_role = SubRolePick(
                    role_id=int(child["id"]),
                    slug=child["slug"],
                    display_name=child["display_name"],
                    confidence=sub_conf,
                    reasoning=str(step3.get("reasoning", ""))[:500],
                )

    return DomainClassificationResult(
        domain=family,
        role_id=int(chosen["id"]),
        role_slug=chosen["slug"],
        role_display_name=chosen["display_name"],
        confidence=step2_conf,
        reasoning=reasoning,
        family_evidence=family_evidence,
        matched_skills=matched_skills,
        matched_dimensions=matched_dimensions,
        matched_kras=matched_kras,
        sub_role=sub_role,
    )
