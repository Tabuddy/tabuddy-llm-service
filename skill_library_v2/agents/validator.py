"""Deterministic skill Validator.

Pure code — no LLM. Called inline by ``dim_worker.process_dim`` after the
Critic; if any error is reported, the dim worker triggers a retry.

Checks performed:
  - Skill count within [_MIN_SKILLS, _MAX_SKILLS]
  - No duplicate slugs within the batch
  - Slug shape: lowercase snake_case, no connector tokens, ≤5 tokens
  - skill_nature is one of the allowed enum values
  - display_name length [1, 80], no markdown chars, not all-caps for ≥9 chars
  - No duplicate aliases within a single skill
  - **Bucket integrity**: every (category_slug, sub_category_slug) on a skill
    must appear in the Categorizer's proposals for this dim.
"""

from __future__ import annotations

import logging
import re

from skill_library_v2.schemas.category import CategorizerOutput
from skill_library_v2.schemas.role import DimensionSlice
from skill_library_v2.schemas.skill import SkillDraft

logger = logging.getLogger(__name__)


_MIN_SKILLS = 8
_MAX_SKILLS = 80

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
# Belt-and-suspenders sentence-shape detectors. Most connectors deliberately
# excluded — they appear in canonical industry terms:
#   _to_ : time_to_first_byte, raw_to_curated_layers, order_to_cash
#   _of_ : language_of_page, segregation_of_duties
#   _for_: cdk_for_terraform, oauth_2_0_for_native_apps, pingone_for_customers,
#          proof_key_for_code_exchange, openid_for_verifiable_X
#   _with_: code_with_me, sign_in_with_(google|apple|x|linkedin|github)
#   _and_: compare_and_swap, store_and_forward
# Only _or_ and _the_ remain — neither appears in observed real product names,
# both are strong sentence-shape signals. The token-cap below + the Critic +
# the bucket-integrity check are the actual quality bars.
_CONNECTOR_TOKENS = ("_or_", "_the_")
# Allow * (algorithm names: A*, D*, RRT*, K*) — # is already permitted (C#).
# Still ban backticks, tildes, brackets, braces, and angle brackets — those
# are pure-markdown noise and don't appear in real skill display names.
_MARKDOWN_BAD = re.compile(r"[`~\[\]<>{}]")
_ALLOWED_NATURES = {
    "TOOL", "CONCEPT", "PRACTICE", "PLATFORM", "LANGUAGE", "CREDENTIAL",
    "LIBRARY", "FRAMEWORK", "RUNTIME", "PROTOCOL", "STANDARD",
    "METHODOLOGY", "CLOUD_SERVICE",
    "PATTERN", "API", "ALGORITHM",
}


def _validate_one(
    skills: list[SkillDraft],
    dim: DimensionSlice,
    categorizer_output: CategorizerOutput,
) -> list[str]:
    """Return a list of human-readable error strings. Empty = pass."""
    errs: list[str] = []

    if not (_MIN_SKILLS <= len(skills) <= _MAX_SKILLS):
        errs.append(
            f"skill count {len(skills)} outside [{_MIN_SKILLS},{_MAX_SKILLS}]"
        )

    seen_slugs: dict[str, int] = {}
    for s in skills:
        seen_slugs[s.slug] = seen_slugs.get(s.slug, 0) + 1
    dups = [sl for sl, n in seen_slugs.items() if n > 1]
    if dups:
        errs.append(f"duplicate slugs: {dups}")

    valid_pairs: set[tuple[str, str]] = {
        (p.category_slug, p.sub_category_slug)
        for p in categorizer_output.proposals
    }

    for s in skills:
        if not _SLUG_RE.match(s.slug):
            errs.append(f"slug {s.slug!r} not lowercase snake_case")
        for tok in _CONNECTOR_TOKENS:
            if tok in s.slug:
                errs.append(f"slug {s.slug!r} uses connector {tok!r}")
                break
        if s.slug.count("_") >= 6:
            errs.append(f"slug {s.slug!r} has >6 tokens")

        pair = (s.category_slug, s.sub_category_slug)
        if pair not in valid_pairs:
            errs.append(
                f"skill {s.slug!r} references unknown bucket "
                f"({s.category_slug!r}, {s.sub_category_slug!r}); "
                f"categorizer proposed: {sorted(valid_pairs)}"
            )

        if s.skill_nature not in _ALLOWED_NATURES:
            errs.append(
                f"skill {s.slug!r} has unknown skill_nature {s.skill_nature!r}"
            )

        name = (s.display_name or "").strip()
        if len(name) < 1 or len(name) > 80:
            errs.append(f"skill {s.slug!r} display_name length out of [1,80]")
        elif _MARKDOWN_BAD.search(name):
            errs.append(f"skill {s.slug!r} display_name has markdown chars")
        # All-caps check intentionally removed — file/spec names like
        # CODEOWNERS, DOCKERFILE, MAKEFILE, JENKINSFILE are legitimate display
        # names. The Critic catches genuine "PYTHON should be Python" cases.

        seen_aliases: set[str] = set()
        for a in s.aliases or []:
            al = (a or "").strip()
            if not al:
                continue
            al_lower = al.lower()
            if al_lower in seen_aliases:
                errs.append(f"skill {s.slug!r} has duplicate alias {a!r}")
            seen_aliases.add(al_lower)

    return errs
