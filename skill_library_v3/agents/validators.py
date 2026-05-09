"""Deterministic post-LLM checks for Stage 0 charters.

These do NOT block writing. Each check returns a list of warning/error dicts
that get serialised into the review_queue payload's ``validator_log`` field
so the admin sees them next to the charter when deciding pass/regenerate.

Distinguishing warnings from errors:
  * **error**: structural / catalog-integrity violation that the admin will
    almost certainly want to regenerate (skill name in scope, definition
    way over length).
  * **warning**: heuristic flag — adjacency or owned_by references a role
    that isn't in the catalog yet. This is expected during catalog
    bootstrap (e.g. DevOps charter pointing at a not-yet-created MLOps).
"""

from __future__ import annotations

import re
from typing import Iterable

from skill_library_v3.schemas.charter import CharterOutput


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def _scope_strings(
    charter: CharterOutput,
) -> Iterable[tuple[str, str, frozenset[str]]]:
    """Yield ``(location_label, text, owner_tokens)`` for every place a skill
    name might sneak into the charter.

    ``owner_tokens`` is non-empty only for ``out_of_scope[i].task`` rows —
    it carries the lowercased word-tokens of the same item's ``owned_by``
    field so the alias scanner can exempt platform/role words that the LLM
    naturally repeats when describing a boundary (e.g. an out-of-scope task
    "Operate Azure infrastructure" with owned_by "Azure Cloud Engineer").
    """
    empty: frozenset[str] = frozenset()
    yield "definition", charter.definition, empty
    for i, item in enumerate(charter.in_scope):
        yield f"in_scope[{i}]", item, empty
    for i, item in enumerate(charter.out_of_scope):
        owner_tokens = frozenset(_tokenize(item.owned_by or ""))
        yield f"out_of_scope[{i}].task", item.task, owner_tokens


def adjacent_roles_in_catalog(
    charter: CharterOutput,
    approved_role_names: set[str],
) -> list[dict]:
    """Each adjacent_roles[].role must match an approved-charter role."""
    out: list[dict] = []
    for i, ar in enumerate(charter.adjacent_roles):
        if _normalize(ar.role) not in approved_role_names:
            out.append(
                {
                    "level": "warning",
                    "code": "adjacent_role_unknown",
                    "location": f"adjacent_roles[{i}].role",
                    "message": (
                        f"Adjacent role {ar.role!r} has no approved charter yet — "
                        "add that role to the catalog or remove this entry."
                    ),
                }
            )
    return out


def owned_by_in_catalog(
    charter: CharterOutput,
    approved_role_names: set[str],
) -> list[dict]:
    """Each out_of_scope[].owned_by must match an approved-charter role."""
    out: list[dict] = []
    for i, item in enumerate(charter.out_of_scope):
        if _normalize(item.owned_by) not in approved_role_names:
            out.append(
                {
                    "level": "warning",
                    "code": "owned_by_unknown",
                    "location": f"out_of_scope[{i}].owned_by",
                    "message": (
                        f"out-of-scope owner {item.owned_by!r} is not yet in the catalog — "
                        "this is expected if you plan to add that role next."
                    ),
                }
            )
    return out


# Word-boundary match. Splitting on \W+ avoids false positives like "kafkaesque"
# matching the "kafka" alias, while still catching multi-word skills like
# "google cloud" because the alias_lower set already lowercases them.
_WORD_RE = re.compile(r"\W+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _WORD_RE.split(text.lower()) if t]


# Generic-noun stop-list. These words are real canonical skills in the catalog,
# but they double as everyday English nouns/verbs that legitimately appear in
# responsibility prose. Flagging them as "skill leaks" produced 8/23 false
# positives in the first prod cohort (servicenow-developer, manual-tester,
# data-analyst, etc.). The repository-layer type filter handles the
# CONCEPT/METHODOLOGY/PATTERN/PRACTICE bucket; this set is the safety net for
# concrete-nature collisions plus a defense-in-depth duplicate of the most
# common dual-use soft-nature words.
_GENERIC_NOUN_STOPLIST: frozenset[str] = frozenset({
    # Concrete-nature collisions — the canonical skill is a real tool but the
    # alias word is a generic English noun that responsibility text reuses.
    "move",            # Move (Aptos/Sui smart-contract language)
    "combine",         # Combine (SwiftUI reactive framework)
    "flow",            # Flow (Facebook static type checker)
    "activities",      # Activities (Android Activity framework)
    "notifications",   # Notifications (cloud service)
    "dashboards",      # dashboards (catalog TOOL)
    "task",            # Task (C#/.NET Task framework — collides with English noun)
    "cost management", # Cost Management (Azure CLOUD_SERVICE — but also a
                       # generic responsibility area like "FinOps cost
                       # management"; multi-word stop-list entry).
    "make",            # Make (GNU Make build TOOL — collides with English
                       # verb. Prod 2026-05-09: Virtualization Engineer
                       # charter failed on 'Make low-level kernel patches').
    "business rules",  # Business Rules (FRAMEWORK — Business Rules Engine).
                       # The phrase is a generic responsibility area
                       # ('implement business rules'). Prod 2026-05-09:
                       # Backend Developer charter failed.
    # Defense-in-depth for soft-nature words. These are filtered out at the
    # repository layer (get_alias_lookup_set), but listing them here keeps
    # the validator safe when alias_lookup is constructed by callers that
    # bypass that filter — tests, fresh-DB bootstrap, ad-hoc scripts.
    "automation",
    "alerting",
    "logging",
    "monitoring",
    "messaging",
    "caching",
    "tracing",
    "indexing",
    "queries",
})


def no_skills_in_scope(
    charter: CharterOutput,
    alias_lookup: set[str],
) -> list[dict]:
    """Flag skill aliases appearing in scope text. Walks the alias set looking
    for whole-token or whole-phrase hits in each scope string. Empty
    ``alias_lookup`` (fresh DB, no skills loaded yet) is a no-op.

    Three exemptions stack on top of the raw alias set:

    1. **Role-name tokens.** For "Azure Cloud Engineer" the word "azure" in
       the definition is the role's identity, not a skill leak. Strip
       role-name tokens (and the full role name) from the alias set.

    2. **Generic-noun stop-list** (``_GENERIC_NOUN_STOPLIST``). Some catalog
       skill names are everyday English nouns ("notifications", "move",
       "automation"). These produced ~8/23 false positives in the first
       prod cohort; the stop-list pre-emptively drops them.

    3. **Per-task owner tokens.** For ``out_of_scope[i].task`` only,
       aliases that are tokens of the same item's ``owned_by`` field are
       exempt — a charter that says "Operate Azure infrastructure"
       owned_by "Azure Cloud Engineer" is naming the boundary, not
       leaking a tool.
    """
    if not alias_lookup:
        return []
    role_name_tokens = set(_tokenize(charter.role_name or ""))
    role_name_lower = (charter.role_name or "").strip().lower()
    effective_aliases = {
        a for a in alias_lookup
        if a
        and a != role_name_lower
        and a not in _GENERIC_NOUN_STOPLIST
        and (" " in a or a not in role_name_tokens)
    }
    if not effective_aliases:
        return []
    out: list[dict] = []
    for location, text, owner_tokens in _scope_strings(charter):
        if not text:
            continue
        tokens = set(_tokenize(text))
        lower_text = " " + text.lower() + " "
        for alias in effective_aliases:
            # Per-task owner-token exemption (only out_of_scope items have
            # non-empty owner_tokens).
            if alias in owner_tokens:
                continue
            if " " in alias:
                # Multi-word: substring check with surrounding whitespace
                # to keep "node js" from matching "Node.js" but allow real
                # phrase hits like "google cloud".
                if f" {alias} " in lower_text:
                    out.append(
                        {
                            "level": "error",
                            "code": "skill_in_scope",
                            "location": location,
                            "message": (
                                f"Skill alias {alias!r} found in {location} — "
                                "scope statements must be responsibilities, not skills."
                            ),
                        }
                    )
                    break
            else:
                if alias in tokens:
                    out.append(
                        {
                            "level": "error",
                            "code": "skill_in_scope",
                            "location": location,
                            "message": (
                                f"Skill alias {alias!r} found in {location} — "
                                "scope statements must be responsibilities, not skills."
                            ),
                        }
                    )
                    break
    return out


def role_name_not_alias_of_existing_role(
    charter: CharterOutput,
    *,
    approved_role_aliases: dict[str, str],
) -> list[dict]:
    """Refuse a Stage 0 charter when its ``role_name`` is already an
    alias of some other approved role.

    Catches the duplicate-role pattern where an admin types a name that
    aliases an existing role (e.g., ``Backend Developer`` when
    ``Backend Engineer`` already lists it as an alias). The catalog
    should not end up with two near-duplicate roles.

    ``approved_role_aliases`` maps every recognized role name (canonical
    + alias, lowercased) to the canonical role's slug. The check fires
    only when the slug differs from the charter's own ``role_id`` —
    re-running an existing role's charter is allowed.
    """
    if not approved_role_aliases:
        return []
    name_lower = (charter.role_name or "").strip().lower()
    canonical_slug = approved_role_aliases.get(name_lower)
    if canonical_slug is None:
        return []
    if canonical_slug == charter.role_id:
        return []
    return [
        {
            "level": "error",
            "code": "role_name_aliases_existing_role",
            "location": "role_name",
            "canonical_role_id": canonical_slug,
            "message": (
                f"role_name {charter.role_name!r} is already an alias of "
                f"existing approved role {canonical_slug!r}. "
                "Use the existing role instead of creating a near-duplicate. "
                "If this role is genuinely distinct, change the canonical_name "
                "to disambiguate (e.g., add a level or specialty qualifier)."
            ),
        }
    ]


def length_constraints(charter: CharterOutput) -> list[dict]:
    """Length sanity checks beyond what Pydantic already enforces."""
    out: list[dict] = []
    for i, item in enumerate(charter.in_scope):
        words = item.split()
        if len(words) < 3:
            out.append(
                {
                    "level": "warning",
                    "code": "in_scope_too_short",
                    "location": f"in_scope[{i}]",
                    "message": f"Item is only {len(words)} word(s); aim for 5-25 words.",
                }
            )
        elif len(words) > 30:
            out.append(
                {
                    "level": "warning",
                    "code": "in_scope_too_long",
                    "location": f"in_scope[{i}]",
                    "message": f"Item is {len(words)} words; tighten toward 5-25.",
                }
            )
    return out


def run_validators(
    charter: CharterOutput,
    *,
    approved_role_names: set[str],
    alias_lookup: set[str],
    approved_role_aliases: dict[str, str] | None = None,
) -> dict:
    """Run every check, partition results into warnings/errors, return the
    validator_log dict the runner serialises.

    ``approved_role_aliases`` (optional, default empty) maps every
    recognized role name + alias (lowercased) to its canonical role slug
    — used by the role-name alias-collision check. Older callers that
    pass only ``approved_role_names`` and ``alias_lookup`` still work;
    they just skip the new check.
    """
    findings: list[dict] = []
    findings += adjacent_roles_in_catalog(charter, approved_role_names)
    findings += owned_by_in_catalog(charter, approved_role_names)
    findings += no_skills_in_scope(charter, alias_lookup)
    findings += length_constraints(charter)
    findings += role_name_not_alias_of_existing_role(
        charter,
        approved_role_aliases=approved_role_aliases or {},
    )

    warnings = [f for f in findings if f["level"] == "warning"]
    errors = [f for f in findings if f["level"] == "error"]
    return {"warnings": warnings, "errors": errors}
