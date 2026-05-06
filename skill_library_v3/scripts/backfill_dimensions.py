"""One-shot backfill: generate Stage 2 dimensions for every approved
role card that has no current Stage 2 row.

Stage 2 normally cascades automatically when Stage 1 is approved (auto
or human). This script catches up roles approved before Stage 2 shipped,
or roles whose latest Stage 2 row was produced under an older prompt
version than the current ``DIM_GEN_PROMPT_VERSION``.

Stage 2 is fully headless — there's no review queue and no UI affordance
for manual regenerate, so this script is the only recovery path for
validator-failed Stage 2 runs. Each call re-runs every role that doesn't
have a current approved Stage 2 row.

Usage:
    uv run python -m skill_library_v3.scripts.backfill_dimensions
    uv run python -m skill_library_v3.scripts.backfill_dimensions --dry-run

Idempotent. Sequential (one role at a time) so the LLM rate limit isn't
saturated.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from skill_library_v3.db import repository as repo
from skill_library_v3.prompts.dimension_gen import DIM_GEN_PROMPT_VERSION
from skill_library_v3.runner import run_stage_2

logger = logging.getLogger(__name__)


async def _backfill(dry_run: bool) -> int:
    targets = repo.find_roles_needing_dimensions(DIM_GEN_PROMPT_VERSION)
    if not targets:
        logger.info(
            "[v3 backfill-dims] all approved role cards have current Stage 2 "
            "rows under prompt %s.",
            DIM_GEN_PROMPT_VERSION,
        )
        return 0

    logger.info("[v3 backfill-dims] found %d role(s) needing Stage 2:", len(targets))
    for item in targets:
        logger.info(
            "  - %s (role card run %s)",
            item["role_display"], item["role_card_run_id"],
        )

    if dry_run:
        logger.info("[v3 backfill-dims] --dry-run: not generating dimensions.")
        return 0

    failed = 0
    for i, item in enumerate(targets, start=1):
        logger.info(
            "[v3 backfill-dims] (%d/%d) generating dimensions for %s...",
            i, len(targets), item["role_display"],
        )
        try:
            await run_stage_2(item["role_slug"])
        except Exception:  # noqa: BLE001
            failed += 1
            logger.exception(
                "[v3 backfill-dims] run_stage_2 for %s raised; continuing",
                item["role_display"],
            )
    logger.info(
        "[v3 backfill-dims] done. %d role(s) processed, %d failed.",
        len(targets), failed,
    )
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be generated without calling the LLM.",
    )
    args = parser.parse_args(argv)
    return asyncio.run(_backfill(dry_run=args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
