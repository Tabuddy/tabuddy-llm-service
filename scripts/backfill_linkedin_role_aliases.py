#!/usr/bin/env python3
"""
Backfill role_aliases from ``linkedin_roles`` staging rows.

Reads ``linkedin_roles`` (id, slug, display_name). Dedup deletes staging rows
when the title already exists in ``role_aliases``. Aliases are stored on the
canonical ``roles`` row (find or create on persist).

Phase 1 (default): read + LLM + log WOULD_* only — persist blocks commented.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env", override=True)

from skill_library_repository import SkillLibraryRepository
from skill_library_v2.agents.base import BaseLLMAgent
from skill_library_v3.catalog_transform import _classify_role_alias
from skill_library_v3.prompts.role_card import (
    ROLE_ALIAS_BACKFILL_PROMPT_VERSION,
    ROLE_ALIAS_BACKFILL_SYSTEM_PROMPT,
    ROLE_ALIAS_BACKFILL_USER_TEMPLATE,
)
from skill_library_v3.schemas.role_card import RoleAliasOnlyOutput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("backfill_linkedin_role_aliases")

_MAX_ALIAS_LEN = 200


class _LinkedInRoleAliasAgent(BaseLLMAgent):
    """Same generation tier as v3 Stage1RoleAnchorAgent (gpt-5.4-mini)."""

    tier = "generation"

    def __init__(self) -> None:
        super().__init__(
            agent_name="linkedin_role_alias_backfill",
            prompt_version=ROLE_ALIAS_BACKFILL_PROMPT_VERSION,
        )


def _normalize_key(text: str) -> str:
    return (text or "").strip().lower()


def _build_alias_rows(
    display_name: str,
    canonical_name: str,
    llm_aliases: list[str],
) -> list[dict]:
    """Mirror v3 catalog_transform role_aliases: CANONICAL primary + classified aliases."""
    primary = (canonical_name or display_name or "").strip()
    if not primary:
        return []

    rows: list[dict] = [
        {
            "alias_text": primary[:_MAX_ALIAS_LEN],
            "alias_type": "CANONICAL",
            "is_primary": True,
        },
    ]
    seen = {_normalize_key(primary)}
    for raw in llm_aliases:
        text = (raw or "").strip()[:_MAX_ALIAS_LEN]
        if not text or _normalize_key(text) in seen:
            continue
        seen.add(_normalize_key(text))
        rows.append(
            {
                "alias_text": text,
                "alias_type": _classify_role_alias(text),
                "is_primary": False,
            }
        )
    return rows


_agent: _LinkedInRoleAliasAgent | None = None


def _get_agent() -> _LinkedInRoleAliasAgent:
    global _agent
    if _agent is None:
        _agent = _LinkedInRoleAliasAgent()
    return _agent


async def _llm_generate_role_aliases(
    display_name: str,
    role_slug: str,
) -> tuple[str, list[str]]:
    """v3 Stage 1 alias rules + generation model; title-only (no charter)."""
    agent = _get_agent()
    result: RoleAliasOnlyOutput = await agent.call_json(
        system_prompt=ROLE_ALIAS_BACKFILL_SYSTEM_PROMPT,
        user_prompt=ROLE_ALIAS_BACKFILL_USER_TEMPLATE.format(
            role_name=display_name,
            role_slug=role_slug or "",
        ),
        schema=RoleAliasOnlyOutput,
        temperature=0.2,
    )
    canonical = (result.canonical_name or display_name).strip()
    if not canonical:
        canonical = display_name
    aliases = [str(a).strip() for a in (result.aliases or []) if str(a).strip()]
    return canonical, aliases


def _persist_delete_staging(repo: SkillLibraryRepository, linkedin_role_id: int) -> None:
    # --- PERSIST (uncomment after testing) ---
    # deleted = repo.delete_linkedin_roles_row(linkedin_role_id)
    # logger.info("DELETED linkedin_roles id=%s ok=%s", linkedin_role_id, deleted)
    pass


def _persist_catalog_and_aliases(
    repo: SkillLibraryRepository,
    *,
    slug: str,
    display_name: str,
    catalog_role_id: int | None,
    rows: list[dict],
) -> int | None:
    """Return catalog ``roles.id`` after persist (or None if skipped)."""
    # --- PERSIST (uncomment after testing) ---
    # catalog = repo.resolve_catalog_role_for_linkedin(
    #     slug=slug,
    #     display_name=display_name,
    #     create_if_missing=True,
    # )
    # rid = int(catalog["id"])
    # from skill_matcher import _azure_embed_sync
    # texts = [r["alias_text"] for r in rows]
    # vecs = _azure_embed_sync(texts) or []
    # embeddings = {
    #     r["alias_text"]: (list(vecs[i]) if i < len(vecs) and vecs[i] else None)
    #     for i, r in enumerate(rows)
    # }
    # n = repo.insert_role_aliases(rid, rows, embeddings=embeddings)
    # logger.info(
    #     "INSERTED %d role_aliases on roles.id=%s (from linkedin_roles)",
    #     n, rid,
    # )
    # return rid
    return catalog_role_id


async def run_backfill(
    *,
    limit: int | None,
    force: bool,
) -> dict[str, int]:
    repo = SkillLibraryRepository()
    roles = repo.list_linkedin_roles(limit=limit)

    alias_lower = repo.load_all_role_alias_lower()
    logger.info(
        "Loaded %d linkedin_roles rows, %d existing role_aliases (read-only)",
        len(roles),
        len(alias_lower),
    )

    stats = {
        "processed": 0,
        "would_delete_staging": 0,
        "would_create_catalog_role": 0,
        "would_insert": 0,
        "skipped": 0,
        "llm_errors": 0,
    }

    for row in roles:
        stats["processed"] += 1
        linkedin_id = int(row["id"])
        display_name = str(row.get("display_name") or "").strip()
        slug = str(row.get("slug") or "").strip()
        key = _normalize_key(display_name)

        if not key:
            logger.warning("SKIP empty display_name linkedin_roles.id=%s", linkedin_id)
            stats["skipped"] += 1
            continue

        # Dedup: lower(display_name) vs role_aliases.alias_lower (case-insensitive).
        if key in alias_lower:
            logger.info(
                "WOULD_DELETE linkedin_roles id=%s slug=%s display_name=%r "
                "(matched alias_lower=%r)",
                linkedin_id,
                slug,
                display_name,
                key,
            )
            stats["would_delete_staging"] += 1
            _persist_delete_staging(repo, linkedin_id)
            continue

        catalog = repo.find_role_by_identity(slug=slug, display_name=display_name)
        catalog_role_id = int(catalog["id"]) if catalog else None

        if catalog_role_id is not None and repo.role_has_aliases(catalog_role_id) and not force:
            logger.info(
                "SKIP already_has_aliases linkedin_roles.id=%s roles.id=%s display_name=%r",
                linkedin_id,
                catalog_role_id,
                display_name,
            )
            stats["skipped"] += 1
            continue

        if catalog_role_id is None:
            logger.info(
                "WOULD_CREATE_ROLE slug=%r display_name=%r (for role_aliases FK)",
                slug,
                display_name,
            )
            stats["would_create_catalog_role"] += 1

        try:
            canonical_name, llm_aliases = await _llm_generate_role_aliases(
                display_name,
                slug,
            )
        except Exception as exc:
            logger.error(
                "LLM failed linkedin_roles.id=%s display_name=%r: %s",
                linkedin_id,
                display_name,
                exc,
            )
            stats["llm_errors"] += 1
            continue

        alias_rows = _build_alias_rows(display_name, canonical_name, llm_aliases)
        target_role_id = catalog_role_id if catalog_role_id is not None else 0
        for r in alias_rows:
            logger.info(
                "WOULD_INSERT roles.id=%s (linkedin_roles.id=%s) alias_text=%r "
                "alias_type=%s is_primary=%s",
                target_role_id or "(new)",
                linkedin_id,
                r["alias_text"],
                r["alias_type"],
                r["is_primary"],
            )
            stats["would_insert"] += 1
            alias_lower.add(_normalize_key(r["alias_text"]))

        _persist_catalog_and_aliases(
            repo,
            slug=slug,
            display_name=display_name,
            catalog_role_id=catalog_role_id,
            rows=alias_rows,
        )

    logger.info(
        "DONE processed=%d would_delete_staging=%d would_create_catalog_role=%d "
        "would_insert=%d skipped=%d llm_errors=%d",
        stats["processed"],
        stats["would_delete_staging"],
        stats["would_create_catalog_role"],
        stats["would_insert"],
        stats["skipped"],
        stats["llm_errors"],
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate role_aliases from linkedin_roles (log-only phase 1).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max staging rows")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run LLM even if canonical roles.id already has aliases",
    )
    args = parser.parse_args()
    asyncio.run(run_backfill(limit=args.limit, force=args.force))


if __name__ == "__main__":
    main()
