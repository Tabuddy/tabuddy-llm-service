"""Build the Excel taxonomy embedding cache for the runtime role resolver.

Reads the Revised sheet of role_classification_v2_24Apr (1).xlsx, embeds each
row's (role + domain + rationale + aliases) blob with text-embedding-3-small,
and writes the result to excel_taxonomy.cache.json. The runtime resolver
loads that file once per process via excel_taxonomy_loader.load_taxonomy().

Usage:
    uv run python -m scripts.build_excel_taxonomy
    uv run python -m scripts.build_excel_taxonomy --xlsx other.xlsx --out other.json
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExcelRow:
    row_index: int
    role: str
    domain: str
    type: str
    rationale: str
    aliases: list[str]
    embed_text: str


def _parse_aliases(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    return [a.strip() for a in str(raw).split(",") if a.strip()]


def extract_rows(df: pd.DataFrame) -> list[ExcelRow]:
    """Parse the Revised-sheet DataFrame into ExcelRow records. Skips rows
    whose Role cell is blank."""
    out: list[ExcelRow] = []
    for _, row in df.iterrows():
        raw_role = row.get("Role")
        if raw_role is None or (isinstance(raw_role, float) and pd.isna(raw_role)):
            continue
        role = str(raw_role).strip()
        if not role:
            continue
        rationale = str(row.get("Rationale") or "").strip()
        domain = str(row.get("Domain") or "").strip()
        type_ = str(row.get("Type") or "").strip()
        aliases = _parse_aliases(row.get("Roles Collapsed Into This"))
        embed_text = (
            f"{role}\n"
            f"Domain: {domain}\n"
            f"Rationale: {rationale}\n"
            f"Aliases: {', '.join(aliases) if aliases else '(none)'}"
        )
        row_idx_raw = row.get("#")
        try:
            row_index = int(row_idx_raw) if row_idx_raw is not None and not pd.isna(row_idx_raw) else 0
        except (TypeError, ValueError):
            row_index = 0
        out.append(ExcelRow(
            row_index=row_index,
            role=role,
            domain=domain,
            type=type_,
            rationale=rationale,
            aliases=aliases,
            embed_text=embed_text,
        ))
    return out


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via the shared Azure embedding client. Returns
    1536-d vectors. Failed slots return []. Monkeypatched in tests."""
    from skill_matcher import _azure_embed_sync

    vecs = _azure_embed_sync(texts)
    return [list(v) if v else [] for v in vecs]


def build_taxonomy(df: pd.DataFrame, cache_path: Path, *, batch_size: int = 50) -> None:
    """Extract rows, embed each in batches, write the JSON cache file.
    Idempotent — overwrites cache_path."""
    rows = extract_rows(df)
    embeddings: list[list[float]] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        chunk_vecs = _embed_batch([r.embed_text for r in chunk])
        embeddings.extend(chunk_vecs)
        logger.info("embedded %d/%d rows", min(i + batch_size, len(rows)), len(rows))

    payload = {
        "version": 1,
        "entries": [
            {**asdict(r), "embedding": emb}
            for r, emb in zip(rows, embeddings)
        ],
    }
    cache_path.write_text(json.dumps(payload))
    logger.info("wrote %s (%d entries)", cache_path, len(payload["entries"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", default="role_classification_v2_24Apr (1).xlsx")
    parser.add_argument("--sheet", default="Revised")
    parser.add_argument("--out", default="excel_taxonomy.cache.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = pd.read_excel(args.xlsx, sheet_name=args.sheet)
    build_taxonomy(df, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
