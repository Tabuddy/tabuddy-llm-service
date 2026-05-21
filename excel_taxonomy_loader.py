"""Load the pre-built Excel taxonomy embedding cache.

The cache is built once by `scripts/build_excel_taxonomy.py` and read at
process startup. If the cache file is missing or corrupted, this module
returns an empty tuple so the runtime resolver degrades gracefully — the
JD pipeline falls back to existing Case NEW synth + v3 behavior.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).resolve().parent / "excel_taxonomy.cache.json"


@dataclass(frozen=True)
class ExcelRoleEntry:
    role: str
    domain: str
    type: str
    rationale: str
    aliases: tuple[str, ...]
    embedding: tuple[float, ...]


@lru_cache(maxsize=1)
def load_taxonomy() -> tuple[ExcelRoleEntry, ...]:
    """Return all Excel taxonomy entries with their 1536-d embeddings.

    Cached per-process (lru_cache); call `load_taxonomy.cache_clear()` in
    tests when monkeypatching `_CACHE_PATH`. Returns an empty tuple on any
    read/parse failure so the resolver no-ops safely.
    """
    if not _CACHE_PATH.exists():
        logger.warning(
            "Excel taxonomy cache missing at %s — resolver will no-op. "
            "Rebuild with: uv run python -m scripts.build_excel_taxonomy",
            _CACHE_PATH,
        )
        return ()
    try:
        payload = json.loads(_CACHE_PATH.read_text())
        entries = tuple(
            ExcelRoleEntry(
                role=e["role"],
                domain=e["domain"],
                type=e["type"],
                rationale=e["rationale"],
                aliases=tuple(e["aliases"]),
                embedding=tuple(e["embedding"]),
            )
            for e in payload["entries"]
        )
        logger.info("loaded %d Excel taxonomy entries from %s", len(entries), _CACHE_PATH)
        return entries
    except (KeyError, json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "Excel taxonomy cache corrupted (%s) — resolver will no-op", exc,
        )
        return ()
