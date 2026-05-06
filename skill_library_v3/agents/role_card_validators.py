"""Deterministic post-LLM checks for Stage 1 role cards (v1.2).

The auto-approve gate fires when the partition has zero ``errors``.
Warnings surface in the UI but do not block. Same dict shape as Stage 0's
charter validators so the existing renderer handles either uniformly.

v1.2 changes:
  * primary_responsibilities and common_artifacts are now semicolon-
    delimited STRINGS (not list[str]). Validators split before counting.
  * Two new fields validated: adjacent_roles, always_load_skills (both
    list[str]). Carry-forward semantics: role card may add to charter's
    sets but must not drop from them.
  * aliases must include at least one abbreviation when non-empty
    (heuristic warning: token of length <= 4 chars).
"""

from __future__ import annotations

from skill_library_v3.schemas.role_card import FAMILY_VALUES, RoleCard


_RESPONSIBILITY_VERBS: frozenset[str] = frozenset(
    {
        "design", "build", "implement", "operate", "maintain", "develop",
        "manage", "monitor", "deliver", "collaborate", "ensure", "review",
        "automate", "investigate", "lead", "own", "ship", "drive", "create",
        "configure", "deploy", "optimize", "analyze", "audit", "support",
        "troubleshoot", "document", "integrate", "evaluate", "research",
        "architect", "scale", "secure", "validate", "tune", "diagnose",
        "coordinate", "facilitate", "model", "instrument", "harden",
    }
)


def _split_semis(s: str) -> list[str]:
    """Split a semicolon-delimited string and trim whitespace per item.
    Empty items are dropped (handles trailing ``;`` and ``; ;`` typos)."""
    return [p.strip() for p in (s or "").split(";") if p.strip()]


def definition_length(card: RoleCard) -> list[dict]:
    if len(card.definition) > 200:
        return [
            {
                "level": "error",
                "code": "definition_too_long",
                "location": "definition",
                "message": f"definition is {len(card.definition)} chars — must be <= 200.",
            }
        ]
    return []


def responsibilities_count(card: RoleCard) -> list[dict]:
    items = _split_semis(card.primary_responsibilities)
    n = len(items)
    if n < 5 or n > 9:
        return [
            {
                "level": "error",
                "code": "responsibilities_count_out_of_range",
                "location": "primary_responsibilities",
                "message": (
                    f"got {n} semicolon-delimited responsibilities; must be 5-9 "
                    f"(e.g. 'API design and implementation; business logic; …')."
                ),
            }
        ]
    return []


def artifacts_count(card: RoleCard) -> list[dict]:
    items = _split_semis(card.common_artifacts)
    n = len(items)
    if n < 3 or n > 10:
        return [
            {
                "level": "error",
                "code": "artifacts_count_out_of_range",
                "location": "common_artifacts",
                "message": (
                    f"got {n} semicolon-delimited artifacts; must be 3-10."
                ),
            }
        ]
    return []


def family_in_enum(card: RoleCard) -> list[dict]:
    if card.family not in FAMILY_VALUES:
        return [
            {
                "level": "error",
                "code": "family_unknown",
                "location": "family",
                "message": f"family={card.family!r} is not in the closed enum.",
            }
        ]
    return []


def aliases_distinct_from_canonical(card: RoleCard) -> list[dict]:
    canonical_lc = card.canonical_name.strip().lower()
    out: list[dict] = []
    for i, alias in enumerate(card.aliases):
        if alias.strip().lower() == canonical_lc:
            out.append(
                {
                    "level": "warning",
                    "code": "alias_matches_canonical",
                    "location": f"aliases[{i}]",
                    "message": f"alias {alias!r} duplicates canonical_name; drop it.",
                }
            )
    return out


def aliases_include_abbreviation(card: RoleCard) -> list[dict]:
    """Heuristic: at least one alias should be a short-form abbreviation
    (length <= 4 chars). Most engineering roles have one (BE, DE, FE, ML,
    QA, SRE). Warning, not error — some roles legitimately have none."""
    if not card.aliases:
        return []
    has_abbrev = any(len(a.strip()) <= 4 and a.strip() for a in card.aliases)
    if has_abbrev:
        return []
    return [
        {
            "level": "warning",
            "code": "aliases_missing_abbreviation",
            "location": "aliases",
            "message": (
                "aliases list is non-empty but contains no short-form abbreviation; "
                "consider adding the field's common short form (BE, DE, FE, etc.) "
                "if one exists."
            ),
        }
    ]


def responsibility_phrasing(card: RoleCard) -> list[dict]:
    """Each responsibility (split on ;) should lead with a verb.
    Heuristic-only, warnings only — phrasing nudge, not a hard rule.
    Note: noun-led responsibilities like "API design and implementation"
    are common in the v1.2 spec, so we only warn on phrases that don't
    look noun-or-verb-led at all."""
    items = _split_semis(card.primary_responsibilities)
    out: list[dict] = []
    for i, item in enumerate(items):
        first = (item.split() or [""])[0].lower().rstrip(",.;:")
        # strip simple -ing / -s forms so "designing" / "designs" still match
        stem = first
        for suffix in ("ing", "es", "s"):
            if stem.endswith(suffix):
                candidate = stem[: -len(suffix)]
                if candidate in _RESPONSIBILITY_VERBS:
                    stem = candidate
                    break
        # Accept verb-led OR noun-led (allow specific responsibility phrases
        # like "API design and implementation"). Only warn on items that
        # start with adverbs / vague qualifiers.
        weak_starts = {"general", "overall", "various", "multiple", "some", "the", "a"}
        if first in weak_starts:
            out.append(
                {
                    "level": "warning",
                    "code": "responsibility_phrasing",
                    "location": f"primary_responsibilities[item {i}]",
                    "message": (
                        f"phrase starts with weak qualifier {first!r}; tighten."
                    ),
                }
            )
    return out


def seniority_distinct(card: RoleCard) -> list[dict]:
    bands = {
        "junior": card.seniority_signals.junior.strip().lower(),
        "mid": card.seniority_signals.mid.strip().lower(),
        "senior": card.seniority_signals.senior.strip().lower(),
        "staff": card.seniority_signals.staff.strip().lower(),
    }
    out: list[dict] = []
    seen: dict[str, str] = {}
    for level, text in bands.items():
        if text in seen:
            out.append(
                {
                    "level": "warning",
                    "code": "seniority_duplicate",
                    "location": f"seniority_signals.{level}",
                    "message": (
                        f"identical to seniority_signals.{seen[text]}; should differentiate."
                    ),
                }
            )
        else:
            seen[text] = level
    return out


def adjacent_roles_subset_of_charter(
    card: RoleCard, charter: dict | None
) -> list[dict]:
    """The role card's adjacent_roles list must be a SUPERSET of the
    charter's adjacent_roles set (carry-forward). Adding canonical
    adjacent roles is fine; dropping any that the charter listed is a
    warning."""
    if not charter:
        return []
    charter_adjacents_raw = charter.get("adjacent_roles") or []
    charter_names: set[str] = set()
    for item in charter_adjacents_raw:
        if isinstance(item, dict):
            name = (item.get("role") or "").strip().lower()
        else:
            name = str(item).strip().lower()
        if name:
            charter_names.add(name)
    card_names = {a.strip().lower() for a in card.adjacent_roles if a.strip()}
    missing = sorted(charter_names - card_names)
    if not missing:
        return []
    return [
        {
            "level": "warning",
            "code": "adjacent_roles_dropped_from_charter",
            "location": "adjacent_roles",
            "message": (
                f"role card dropped charter-listed adjacent roles: {missing}. "
                "Carry-forward rule: card may add but not drop."
            ),
        }
    ]


def always_load_includes_charter_set(
    card: RoleCard, charter: dict | None
) -> list[dict]:
    """Same superset rule for always_load_skills."""
    if not charter:
        return []
    charter_skills = {
        s.strip().lower()
        for s in (charter.get("always_load_skills") or [])
        if isinstance(s, str) and s.strip()
    }
    card_skills = {s.strip().lower() for s in card.always_load_skills if s.strip()}
    missing = sorted(charter_skills - card_skills)
    if not missing:
        return []
    return [
        {
            "level": "warning",
            "code": "always_load_dropped_from_charter",
            "location": "always_load_skills",
            "message": (
                f"role card dropped charter-listed always-load skills: {missing}. "
                "Carry-forward rule: card may add but not drop."
            ),
        }
    ]


def adjacent_roles_count(card: RoleCard) -> list[dict]:
    n = len(card.adjacent_roles)
    if n < 3:
        return [
            {
                "level": "warning",
                "code": "adjacent_roles_too_few",
                "location": "adjacent_roles",
                "message": (
                    f"only {n} adjacent role(s); aim for 5-15 to give downstream "
                    "stages enough boundary signal."
                ),
            }
        ]
    return []


def always_load_count(card: RoleCard) -> list[dict]:
    n = len(card.always_load_skills)
    if n < 3:
        return [
            {
                "level": "warning",
                "code": "always_load_too_few",
                "location": "always_load_skills",
                "message": (
                    f"only {n} always-load skill(s); aim for 3-10 universal "
                    "cross-cutting items (Git, SQL, Code Review, Unit Testing, etc.)."
                ),
            }
        ]
    return []


def run_role_card_validators(
    card: RoleCard,
    *,
    charter: dict | None = None,
) -> dict:
    """Run every check; partition results into warnings/errors.

    ``charter`` (optional): the source charter from Stage 0 used for the
    superset-carry-forward checks. When None, those checks are skipped
    silently (callers like backfill scripts that don't have the charter
    handy can omit it).
    """
    findings: list[dict] = []
    findings += definition_length(card)
    findings += responsibilities_count(card)
    findings += artifacts_count(card)
    findings += family_in_enum(card)
    findings += aliases_distinct_from_canonical(card)
    findings += aliases_include_abbreviation(card)
    findings += responsibility_phrasing(card)
    findings += seniority_distinct(card)
    findings += adjacent_roles_count(card)
    findings += always_load_count(card)
    findings += adjacent_roles_subset_of_charter(card, charter)
    findings += always_load_includes_charter_set(card, charter)

    warnings = [f for f in findings if f["level"] == "warning"]
    errors = [f for f in findings if f["level"] == "error"]
    return {"warnings": warnings, "errors": errors}
