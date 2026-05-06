"""One-shot backfill: regenerate Stage 1 role cards.

Two modes:

* DEFAULT — fill role cards for every approved charter that has no
  approved role card yet. The charter pipeline shipped before role
  anchoring did, so any role approved during the Stage 0-only window
  needs catch-up.

* ``--include-stale`` — also re-run roles whose latest approved role
  card was produced by an OLDER prompt version (anything other than the
  current ``ROLE_CARD_PROMPT_VERSION``). Use this after the prompt
  bumps so existing roles get re-anchored under the new schema.

Usage:
    uv run python -m skill_library_v3.scripts.backfill_role_cards
    uv run python -m skill_library_v3.scripts.backfill_role_cards --include-stale
    uv run python -m skill_library_v3.scripts.backfill_role_cards --dry-run

Idempotent — re-runs are no-ops once each role has an approved role
card produced under the current prompt version. Runs Stage 1
sequentially (not in parallel) so a backfill doesn't saturate the LLM.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid

from skill_library_v3.db import repository as repo
from skill_library_v3.prompts.role_card import ROLE_CARD_PROMPT_VERSION
from skill_library_v3.runner import run_stage_1

logger = logging.getLogger(__name__)


def _collect_targets(include_stale: bool) -> list[dict]:
    """Combine unanchored charters and (optionally) stale role cards into
    a deduplicated list keyed by role_slug. Each target item carries the
    charter_run_id we'll feed into ``run_stage_1``."""
    unanchored = repo.find_unanchored_approved_charters()
    targets: dict[str, dict] = {item["role_slug"]: item for item in unanchored}

    if include_stale:
        stale = repo.find_stale_role_cards(ROLE_CARD_PROMPT_VERSION)
        # For stale roles we need the latest approved charter run_id to
        # feed into run_stage_1. The role-card-stale entry only has the
        # role_card_run_id; use a separate lookup to get the charter.
        for item in stale:
            slug = item["role_slug"]
            if slug in targets:
                continue
            charter = _latest_approved_charter_run_id(slug)
            if charter is None:
                logger.warning(
                    "[v3 backfill] role %r has stale role card (%s) but no "
                    "approved charter — skipping; resolve via UI first",
                    item["role_display"],
                    item["current_prompt_version"],
                )
                continue
            targets[slug] = {
                "role_slug": slug,
                "role_display": item["role_display"],
                "charter_run_id": charter,
                "stale_from": item["current_prompt_version"],
            }
    return list(targets.values())


def _latest_approved_charter_run_id(role_slug: str) -> str | None:
    """Helper used by --include-stale to resolve a charter run_id we can
    pass to run_stage_1. Falls back to None if the role has no approved
    charter (shouldn't happen for an established role, but guard anyway)."""
    from skill_library_v3.db.connection import connect

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id FROM v2_run_log
                 WHERE role_id = %s
                   AND prompt_version LIKE %s
                   AND status = 'approved'
                 ORDER BY completed_at DESC
                 LIMIT 1
                """,
                (role_slug, repo.CHARTER_PROMPT_VERSION_PREFIX + "%"),
            )
            row = cur.fetchone()
            return str(row[0]) if row else None


async def _backfill(dry_run: bool, include_stale: bool) -> int:
    targets = _collect_targets(include_stale)
    if not targets:
        logger.info(
            "[v3 backfill] all roles up-to-date under prompt version %s.",
            ROLE_CARD_PROMPT_VERSION,
        )
        return 0

    logger.info("[v3 backfill] found %d role(s) to (re)generate:", len(targets))
    for item in targets:
        suffix = (
            f" (stale from {item.get('stale_from')!r})"
            if item.get("stale_from") else ""
        )
        logger.info("  - %s (charter run %s)%s",
                    item["role_display"], item["charter_run_id"], suffix)

    if dry_run:
        logger.info("[v3 backfill] --dry-run: not generating role cards.")
        return 0

    failed = 0
    for i, item in enumerate(targets, start=1):
        charter_run_id = uuid.UUID(item["charter_run_id"])
        logger.info(
            "[v3 backfill] (%d/%d) generating role card for %s...",
            i, len(targets), item["role_display"],
        )
        try:
            await run_stage_1(charter_run_id)
        except Exception:  # noqa: BLE001
            failed += 1
            logger.exception(
                "[v3 backfill] run_stage_1 for %s raised; continuing",
                item["role_display"],
            )
    logger.info("[v3 backfill] done. %d role(s) processed, %d failed.",
                len(targets), failed)
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be generated without calling the LLM.",
    )
    parser.add_argument(
        "--include-stale",
        action="store_true",
        help=(
            "Also regenerate role cards whose prompt_version is older than "
            f"{ROLE_CARD_PROMPT_VERSION}."
        ),
    )
    args = parser.parse_args(argv)
    return asyncio.run(_backfill(
        dry_run=args.dry_run,
        include_stale=args.include_stale,
    ))


if __name__ == "__main__":
    sys.exit(main())
