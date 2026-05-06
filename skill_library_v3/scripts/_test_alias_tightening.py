"""One-shot test: regenerate DevOps Engineer's role card with the v1.1 prompt
and print a before/after diff so we can see whether the alias clause fix
actually disambiguates DevOps from SRE / Platform Engineer / Build & Release.

Runs offline — no uvicorn required. Targets the same skill-library DB the
running app uses, so the new run lands in v2_run_log alongside everything
else for the UI to pick up.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid

from skill_library_v3.db import repository as repo
from skill_library_v3.runner import run_stage_1_regenerate

logger = logging.getLogger(__name__)


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    role_slug = "devops-engineer"
    prior = repo.get_latest_role_card_run_for_role(role_slug)
    if prior is None:
        print(f"[test] no Stage 1 row found for {role_slug}; nothing to compare against.")
        return 1

    print(f"[test] BEFORE — latest Stage 1 for {role_slug}:")
    print(f"  run_id      = {prior['run_id']}")
    print(f"  status      = {prior['status']}")
    print(f"  reviewed_by = {prior['reviewed_by']}")
    print(f"  aliases (old): {(prior.get('role_card') or {}).get('aliases')}")
    print()

    # Supersede the old approved row, create a new pending Stage 1 run
    # linked to it. Allowed by the relaxed status check we put in earlier.
    print("[test] superseding old approved role card...")
    result = await asyncio.to_thread(
        repo.reject_and_create_regenerate,
        uuid.UUID(prior["run_id"]),
        reviewed_by="alias-prompt-test",
        notes="testing v1.1 prompt — should drop SRE / Platform / Build&Release as aliases",
    )
    new_run_id: uuid.UUID = result["run_id"]
    print(f"[test] new pending run id: {new_run_id}")
    print(f"[test] running Stage 1 (v1.1 prompt)...")

    # Drive the new pending row through the regenerate runner. This calls
    # the LLM with the freshly-tightened prompt.
    await run_stage_1_regenerate(new_run_id)

    new_run = repo.get_latest_role_card_run_for_role(role_slug)
    if new_run is None or new_run["run_id"] != str(new_run_id):
        print("[test] could not find the new run after regenerate")
        return 2

    print()
    print(f"[test] AFTER — new Stage 1 for {role_slug}:")
    print(f"  run_id      = {new_run['run_id']}")
    print(f"  status      = {new_run['status']}")
    print(f"  reviewed_by = {new_run['reviewed_by']}")

    new_card = new_run.get("role_card") or {}
    print(f"  aliases (new): {new_card.get('aliases')}")
    print()
    print("[test] new role card (full):")
    print(json.dumps(new_card, indent=2, ensure_ascii=False))

    # Quick automated check on the alias list against the negative examples
    # the prompt now lists explicitly.
    forbidden = {
        "site reliability engineer",
        "platform engineer",
        "build and release engineer",
        "production engineer",
        "devsecops engineer",
    }
    leaks = [a for a in (new_card.get("aliases") or []) if a.strip().lower() in forbidden]
    if leaks:
        print(f"[test] STILL LEAKING distinct roles into aliases: {leaks}")
        return 3
    print("[test] no forbidden sibling roles found in aliases")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
