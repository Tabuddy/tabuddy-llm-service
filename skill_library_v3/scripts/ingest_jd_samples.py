"""Download a Kaggle JD dataset via kagglehub and load it into ``jd_samples``.

Default dataset: ``adityarajsrv/job-descriptions-2025-tech-and-non-tech-roles``
(2025 roles, includes emerging AI roles). Override with --dataset.

Auth: kagglehub picks up KAGGLE_USERNAME/KAGGLE_KEY env vars or
~/.kaggle/kaggle.json automatically. First run without creds fails with a
link to https://www.kaggle.com/docs/api.

Usage:
    # First run: just inspect the columns of the downloaded CSV.
    uv run python -m skill_library_v3.scripts.ingest_jd_samples --inspect-only

    # Subsequent runs: ingest with explicit column mapping (or auto-detect).
    uv run python -m skill_library_v3.scripts.ingest_jd_samples \\
        --title-col 'Job Title' --body-col 'Job Description'

Role-matching: rapidfuzz token_set_ratio between each JD's title and every
role currently in ``roles``. Best match >= 80 wins; rows below threshold are
inserted with ``role_slug=NULL`` and can be matched later as new roles get
added (rerun the script with --backfill-only).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from rapidfuzz import fuzz

from skill_library_v3.db import repository as repo
from skill_library_v3.db.connection import connect

logger = logging.getLogger(__name__)


DEFAULT_DATASET = "adityarajsrv/job-descriptions-2025-tech-and-non-tech-roles"
DEFAULT_MATCH_THRESHOLD = 80.0

TITLE_CANDIDATES = [
    "Job Title", "job_title", "title", "Title", "role", "Role",
    "position", "Position", "JobTitle",
]
BODY_CANDIDATES = [
    "Job Description", "job_description", "description", "Description",
    "body", "Body", "JobDescription", "details",
]


def _download(dataset: str) -> Path:
    """Download via kagglehub. Imported lazily so the rest of the module is
    usable without kagglehub installed (e.g. for --backfill-only)."""
    try:
        import kagglehub  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is not installed. Run `uv add kagglehub` first, or "
            "supply --csv to skip the download."
        ) from exc

    logger.info("[v3 ingest] downloading dataset %s via kagglehub...", dataset)
    path = kagglehub.dataset_download(dataset)
    return Path(path)


def _find_csv(directory: Path) -> Path:
    csvs = sorted(directory.glob("**/*.csv"))
    if not csvs:
        raise SystemExit(f"no CSV files under {directory}")
    if len(csvs) > 1:
        logger.info("[v3 ingest] multiple CSVs found, using the first: %s", csvs[0].name)
    return csvs[0]


def _read_csv_streaming(path: Path):
    """Yield (row_index, dict) tuples. Tries utf-8 then latin-1 as fallback."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as fh:
                reader = csv.DictReader(fh)
                fieldnames = reader.fieldnames or []
                yield ("__fields__", fieldnames)
                for i, row in enumerate(reader):
                    yield (i, row)
            return
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"could not decode {path} with utf-8 or latin-1")


def _autodetect(fieldnames: list[str], candidates: list[str]) -> str | None:
    fields_lower = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c in fieldnames:
            return c
        if c.lower() in fields_lower:
            return fields_lower[c.lower()]
    return None


def _match_role(title: str, role_index: list[tuple[str, str]]) -> tuple[str | None, float]:
    """Return ``(slug, score)`` for the best fuzzy match, or ``(None, score)``
    if below threshold."""
    if not role_index or not title:
        return None, 0.0
    best_slug, best_score = None, 0.0
    for slug, name in role_index:
        score = float(fuzz.token_set_ratio(title, name))
        if score > best_score:
            best_score, best_slug = score, slug
    if best_score < DEFAULT_MATCH_THRESHOLD:
        return None, best_score
    return best_slug, best_score


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Kaggle slug, e.g. adityarajsrv/job-descriptions-2025-tech-and-non-tech-roles")
    parser.add_argument("--csv", default=None,
                        help="Skip kagglehub and use a local CSV path.")
    parser.add_argument("--title-col", default=None,
                        help="Column name for the job title.")
    parser.add_argument("--body-col", default=None,
                        help="Column name for the job description body.")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Download/open and print columns + first row, then exit.")
    parser.add_argument("--max-rows", type=int, default=20000,
                        help="Hard cap on rows ingested (default 20k).")
    parser.add_argument("--backfill-only", action="store_true",
                        help="Skip ingestion; re-fuzzy-match existing jd_samples rows whose role_slug is NULL.")
    args = parser.parse_args(argv)

    if args.backfill_only:
        return _backfill(args)

    if args.csv:
        csv_path = Path(args.csv).expanduser()
        if not csv_path.is_file():
            logger.error("--csv path not found: %s", csv_path)
            return 1
    else:
        directory = _download(args.dataset)
        csv_path = _find_csv(directory)
    logger.info("[v3 ingest] reading %s", csv_path)

    iterator = _read_csv_streaming(csv_path)
    _, fieldnames = next(iterator)
    logger.info("[v3 ingest] columns: %s", fieldnames)

    if args.inspect_only:
        # Show one sample row.
        for i, row in iterator:
            if i == 0:
                logger.info("[v3 ingest] first row: %s", {k: (v or "")[:120] for k, v in row.items()})
                break
        return 0

    title_col = args.title_col or _autodetect(fieldnames, TITLE_CANDIDATES)
    body_col = args.body_col or _autodetect(fieldnames, BODY_CANDIDATES)
    if not title_col or not body_col:
        logger.error(
            "Could not auto-detect title/body columns. Available columns: %s\n"
            "Pass --title-col and --body-col explicitly.",
            fieldnames,
        )
        return 2
    logger.info("[v3 ingest] using title_col=%r, body_col=%r", title_col, body_col)

    # Snapshot the role catalog before ingest. Empty catalog is fine — every
    # row goes in with role_slug=NULL and gets matched on a later --backfill-only run.
    role_index = _load_role_index()
    logger.info("[v3 ingest] role catalog: %d role(s)", len(role_index))

    batch: list[dict] = []
    matched = 0
    inserted = 0
    seen = 0
    BATCH_SIZE = 500

    for i, row in iterator:
        seen += 1
        if seen > args.max_rows:
            break
        title = (row.get(title_col) or "").strip()
        body = (row.get(body_col) or "").strip()
        if not title or not body:
            continue
        slug, score = _match_role(title, role_index)
        if slug is not None:
            matched += 1
        batch.append(
            {
                "role_slug": slug,
                "source_dataset": args.dataset if not args.csv else f"local:{csv_path.name}",
                "source_row_id": str(i),
                "title": title,
                "body": body,
                "match_score": score,
            }
        )
        if len(batch) >= BATCH_SIZE:
            inserted += repo.insert_jd_samples(batch)
            batch.clear()

    if batch:
        inserted += repo.insert_jd_samples(batch)

    logger.info(
        "[v3 ingest] saw %d row(s), inserted %d, matched to a role: %d",
        seen, inserted, matched,
    )
    return 0


def _load_role_index() -> list[tuple[str, str]]:
    return [(r["slug"], r["display_name"]) for r in repo.list_roles()]


def _backfill(args: argparse.Namespace) -> int:
    """Re-match existing unmatched rows. Useful after adding new roles."""
    role_index = _load_role_index()
    if not role_index:
        logger.error("[v3 ingest] role catalog is empty; nothing to match against.")
        return 1

    logger.info("[v3 ingest] backfilling unmatched jd_samples against %d role(s)", len(role_index))
    rematched = 0
    cursor_id = 0
    BATCH = 500
    with connect() as conn:
        with conn.cursor() as cur:
            while True:
                cur.execute(
                    """
                    SELECT id, title FROM jd_samples
                     WHERE role_slug IS NULL AND id > %s
                     ORDER BY id
                     LIMIT %s
                    """,
                    (cursor_id, BATCH),
                )
                rows = cur.fetchall()
                if not rows:
                    break
                updates: list[tuple[str, float, int]] = []
                for row_id, title in rows:
                    slug, score = _match_role(title or "", role_index)
                    if slug is not None:
                        updates.append((slug, score, row_id))
                    cursor_id = row_id
                if updates:
                    cur.executemany(
                        "UPDATE jd_samples SET role_slug = %s, match_score = %s WHERE id = %s",
                        updates,
                    )
                    rematched += len(updates)
            conn.commit()
    logger.info("[v3 ingest] backfilled %d row(s)", rematched)
    return 0


if __name__ == "__main__":
    sys.exit(main())
