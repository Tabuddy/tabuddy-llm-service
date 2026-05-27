"""Ingest distinct LinkedIn ``position_title`` values into a staging table.

Source: OpenSearch ``profiles`` index. Each profile has an ``experiences[]``
array; we mine ``experiences[].position_title`` from every profile.
Switching from Postgres-source to OpenSearch avoids the high-offset cost
of paginating deep into ``linkedin.profile_experiences`` (offset 10M+
rows is slow even with a PK index).

Target: ``skill-library.linkedin_roles`` (NOT the production ``roles``
table). Once cleaning has been verified on the staging data, a separate
step will copy approved rows into ``roles`` with source='LINKEDIN'.

Pipeline:
  1. ``helpers.scan(index="profiles", size=10000)`` — OpenSearch scroll
     in 10K-doc pages (the cluster's max_result_window cap). One scroll
     page = one flush batch by default (--batch-size 10000). Each
     profile contributes 0..N position_titles (one per experience).
     With --limit N, stops after N profiles.
  2. Run each raw title through ``clean_position_title`` — strips
     company/location ("X at Google", "Y @ Acme"), seniority/level
     prefixes, employment-type suffixes, parenthetical/bracketed
     asides, and roman/arabic numerals. VP/AVP/Director are stripped
     wherever they appear; if a title is ONLY those tokens, it's
     dropped. "Manager"-containing titles preserve their seniority
     prefix ("Senior Manager" stays "Senior Manager"). "Associate"
     always strips. Bare "Manager"/"Developer"/"Engineering" plus
     other single-word generics are dropped.
  3. Slugify with hash disambiguation on collisions (every distinct
     cleaned display_name gets its own row).
  4. Bulk INSERT into ``linkedin_roles`` (slug UNIQUE; ON CONFLICT
     DO NOTHING — slugs already in the table are skipped, so re-runs
     are cheap and idempotent).

Recommended run order:

    # 1. Apply the staging table migration (idempotent).
    psql "$PG_PROD" -f db/schema_linkedin_roles_staging.sql

    # 2. Dry-run on first 10K profiles (one batch) for a cleaning preview.
    python scripts/import_linkedin_position_titles.py --dry-run --limit 10000

    # 3. First 50K profiles for real (5 batches).
    python scripts/import_linkedin_position_titles.py --limit 50000

    # 4. Full ingest — every profile, every experience.
    python scripts/import_linkedin_position_titles.py

Env vars:
    OPENSEARCH_URL         — OpenSearch endpoint (default: prod)
    OPENSEARCH_INDEX       — defaults to 'profiles'
    OPENSEARCH_USERNAME / OPENSEARCH_PASSWORD
    TARGET_DSN             — write DSN (defaults to PG_PROD)
    TARGET_SCHEMA          — defaults to 'skill-library'
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
from typing import Iterator
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("linkedin_position_titles")

# Suppress opensearchpy + urllib3 HTTP chatter (scroll POSTs, connection
# pool messages). opensearchpy ≥2.x renamed its loggers from
# "opensearchpy.*" to "opensearch.*"; suppress both so we never get the
# per-scroll `POST /_search/scroll [status:200 request:0.7s]` lines.
for _noisy in (
    "opensearch", "opensearch.trace", "opensearch.connection",
    "opensearch.connection.http_urllib3",
    "opensearchpy", "opensearchpy.trace", "opensearchpy.connection",
    "opensearchpy.connection.http_urllib3",
    "urllib3", "urllib3.connectionpool",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ── Noise vocabulary ─────────────────────────────────────────────────────────
# Standalone tokens (after splitting on whitespace + punctuation) that are
# stripped regardless of position. Compared lowercase, trailing '.' removed.

SENIORITY_TOKENS: set[str] = {
    "senior", "sr", "snr",
    "junior", "jr", "jnr",
    "lead", "principal", "staff", "chief", "head",
    "assistant", "asst",
    "intern", "trainee", "fresher", "apprentice",
    "entry", "entrylevel", "mid", "midlevel", "midsenior",
}

# Always stripped regardless of the Manager exception. "Associate Sales
# Manager" should become "Sales Manager" — the Manager exception preserves
# seniority prefixes, but the user specifically flagged "associate" as noise
# that should always be removed.
ALWAYS_STRIP_NOISE: set[str] = {"associate", "assoc"}

LEVEL_TOKENS: set[str] = {
    # Roman numerals (lower-cased) — only stripped when standalone token
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    # Bare arabic digits
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    # Level / grade markers
    "l1", "l2", "l3", "l4", "l5", "l6", "l7",
    "level", "grade", "tier", "band",
}

EMPLOYMENT_TYPE_TOKENS: set[str] = {
    "fulltime", "parttime", "ft", "pt", "fte",
    "contract", "contractor", "consultant", "consulting",
    "freelance", "freelancer",
    "temp", "temporary", "permanent", "perm",
    "remote", "hybrid", "onsite", "wfh",
    "contractbased", "contractor",
}

# Anything matching this entire (lowercased) string after cleaning is dropped.
# Exec titles (CEO/CTO/...) are intentionally NOT here — they're legitimate
# position titles and survive into the catalog (smart_title preserves casing).
JUNK_EXACT: set[str] = {
    "n/a", "na", "none", "unknown", "self employed", "self-employed",
    "selfemployed", "freelancer", "various",
}

# VP / AVP / Director: always stripped wherever they appear. If a title is
# ONLY these tokens (bare "VP", "Director"), kept ends up empty and the row
# is dropped. In combinations ("VP of Engineering", "Director of Sales")
# only the token itself goes; the rest survives.
STRIP_OR_DROP_TOKENS: set[str] = {"vp", "avp", "director"}

# Stopwords stripped from BOTH ends of the cleaned token list. Handles
# "VP of Engineering" → "Engineering" (leading "of" gone) and "Java Trainer
# And" → "Java Trainer" (trailing conjunction gone).
EDGE_STOPWORDS: set[str] = {
    "of", "the", "for", "to", "in", "on", "at", "by", "with", "from", "as",
    "and", "or", "but", "&",
    "a", "an",
}

# Cleaned titles that exactly match one of these are dropped entirely —
# too generic to be useful in a role catalog. "Senior Manager" / "Software
# Developer" / "Software Engineering" still pass; only the bare word is
# rejected.
DROP_IF_EXACT: set[str] = {
    "manager", "developer", "engineering",
    "operations", "sales", "marketing", "design",
    "consultant", "trainer", "specialist", "analyst",
}

# Pre-tokenization normalizer: insert a space after "Sr.", "Jr.", "Snr.",
# "Jnr.", "Assoc.", "Asst." when stuck to the next word ("Sr.Automation"
# → "Sr. Automation") so the abbreviation tokenizes as its own token and
# gets stripped by SENIORITY_TOKENS / ALWAYS_STRIP_NOISE. Leaves ".NET",
# "Node.js" alone because they don't match the alternation.
SENIORITY_ABBREV_DOT_RE = re.compile(
    r"(?i)\b(sr|jr|snr|jnr|assoc|asst)\.(?=\S)"
)

# Split on " at " or "@" — keep LHS (job title), drop RHS (company/location).
SPLIT_AT_RE = re.compile(r"\s+at\s+|\s*@\s*", re.IGNORECASE)

# Trailing separators that usually introduce a company / location.
SPLIT_SEP_RE = re.compile(r"\s*[|/]\s*|\s+[-–—]\s+|\s*•\s*")

# Parenthetical / bracketed asides (location, employment type, etc.).
PAREN_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}")

# Tokenizer keeps "C++", "C#", ".NET", "Node.js" intact.
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9.+#/&]*|\d+")

# Slug builder (mirrors skill_library_v3.db.repository.slugify behavior).
SLUG_KEEP_RE = re.compile(r"[A-Za-z0-9]+")


def clean_position_title(raw: str) -> str | None:
    """Return a cleaned, title-cased role name, or None if the input is junk.

    Order matters: split company/location off FIRST, then strip parentheticals,
    then token-level noise removal. That way "Senior SWE at Lead Bank" doesn't
    accidentally lose "Lead" from the company side before it's split off.
    """
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None

    # 1. Drop everything after " at " / "@" (company side).
    s = SPLIT_AT_RE.split(s, maxsplit=1)[0]

    # 2. Drop parenthetical / bracketed asides.
    s = PAREN_RE.sub(" ", s)

    # 3. Drop everything after a strong separator ("|", " - ", " — ", " / ").
    s = SPLIT_SEP_RE.split(s, maxsplit=1)[0]

    # 4. Drop trailing comma-separated suffixes ("Engineer, New York").
    s = s.split(",", 1)[0]

    # 5. Normalize stuck-together seniority abbreviations: "Sr.Automation"
    #    → "Sr. Automation", so "Sr" tokenizes as its own (strippable) token.
    s = SENIORITY_ABBREV_DOT_RE.sub(r"\1. ", s)

    # 6. Tokenize and drop noise tokens.
    tokens = TOKEN_RE.findall(s)

    # "Manager exception": when Manager appears anywhere in the title,
    # preserve seniority/level prefixes that would otherwise be stripped —
    # "Senior Manager" is a real role distinct from "Manager", and we
    # shouldn't collapse them. ALWAYS_STRIP_NOISE ("associate", "assoc")
    # is the exception-to-the-exception and goes regardless.
    has_manager = any(tok.lower().rstrip(".") == "manager" for tok in tokens)

    kept: list[str] = []
    for tok in tokens:
        norm = tok.lower().rstrip(".")
        # VP / AVP / Director: always stripped. Bare rows fall through to the
        # `if not kept: return None` below and are dropped.
        if norm in STRIP_OR_DROP_TOKENS:
            continue
        # Associate / Assoc: always stripped, even when Manager is present.
        if norm in ALWAYS_STRIP_NOISE:
            continue
        if norm in EMPLOYMENT_TYPE_TOKENS:
            continue
        # Bare arabic digit as standalone token → skip (handled in LEVEL_TOKENS
        # but only for 0–10; this catches "12", "II/III" remnants etc.).
        if norm.isdigit():
            continue
        if not has_manager:
            if norm in SENIORITY_TOKENS:
                continue
            if norm in LEVEL_TOKENS:
                continue
        kept.append(tok)

    # 7. Strip edge stopwords from BOTH ends. Handles "VP of Engineering"
    #    (leading "of") and "Java Trainer And" (trailing conjunction).
    while kept and kept[0].lower().rstrip(".") in EDGE_STOPWORDS:
        kept.pop(0)
    while kept and kept[-1].lower().rstrip(".") in EDGE_STOPWORDS:
        kept.pop()

    if not kept:
        return None

    cleaned = " ".join(kept)
    cleaned = re.sub(r"\s+", " ", cleaned)
    # 8. Strip whitespace, punctuation, digits from both ends — defensive
    #    final pass per spec ("strip space, special char, numbers before
    #    and after the role").
    cleaned = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", cleaned)

    if len(cleaned) < 2 or not re.search(r"[A-Za-z]", cleaned):
        return None

    low = cleaned.lower()
    if low in JUNK_EXACT:
        return None
    # "Manager" / "Developer" alone are too generic to catalog as roles —
    # the user wants only compound forms ("Sales Manager", "Software
    # Developer") to survive.
    if low in DROP_IF_EXACT:
        return None

    # Title-case but preserve common all-caps acronyms / tech tokens.
    return _smart_title(cleaned)


_KEEP_UPPER = {"CEO", "CTO", "COO", "CFO", "CMO", "CIO", "VP", "SVP", "EVP",
               "QA", "QC", "UX", "UI", "AI", "ML", "NLP", "BI", "HR", "IT",
               "SRE", "SEO", "SEM", "CRM", "ERP", "PM", "TPM", "EM"}


def _smart_title(s: str) -> str:
    out: list[str] = []
    for w in s.split(" "):
        if not w:
            continue
        up = w.upper()
        if up in _KEEP_UPPER:
            out.append(up)
        elif "+" in w or "#" in w or "." in w:
            # Tech tokens like C++, C#, .NET, Node.js — keep author's casing.
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:].lower())
    return " ".join(out)


def slugify(text: str) -> str:
    """Lowercase-with-dashes slug. Matches skill_library_v3.db.repository."""
    parts = SLUG_KEEP_RE.findall((text or "").lower())
    return "-".join(parts) or "unknown"


# ── DB plumbing ──────────────────────────────────────────────────────────────


def _build_opensearch_client() -> OpenSearch:
    """Build the OpenSearch client from env. Mirrors relationship-mapping.py."""
    url = os.getenv(
        "OPENSEARCH_URL",
        os.getenv("OPENSEARCH_NODE_URL", "https://135.235.196.207:9200"),
    )
    username = os.getenv("OPENSEARCH_USERNAME", os.getenv("OPENSEARCH_USER", "admin"))
    password = os.getenv(
        "OPENSEARCH_PASSWORD",
        os.getenv("OPENSEARCH_PASS", "Link3diN$c6ap3rOp3nS3a6ch"),
    )
    timeout = int(os.getenv("OPENSEARCH_TIMEOUT_SECONDS", "300"))

    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    host = parsed.hostname or url
    port = parsed.port or (443 if scheme == "https" else 80)
    return OpenSearch(
        hosts=[{"host": host, "port": port, "scheme": scheme}],
        http_auth=(username, password),
        use_ssl=scheme == "https",
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=timeout,
        max_retries=1,
        retry_on_timeout=False,
    )


# Hard cap on the OpenSearch scroll page size — the prod cluster sets
# index.max_result_window=10000, which scroll inherits. We can't fetch
# pages larger than this regardless of how we tune --batch-size; the
# script's "batch" is a DB-flush boundary, not a single scroll page.
OS_SCROLL_PAGE = 10_000


def _stream_title_batches(
    client: OpenSearch,
    index: str,
    batch_size: int = 10_000,
    limit: int | None = None,
    start_id: int | None = None,
) -> Iterator[tuple[int, str, str, list[str]]]:
    """Yield (batch_index, first_profile_id, last_profile_id, raw_titles)
    per batch. The caller flushes each batch to Postgres before the next
    batch arrives, so memory stays bounded and progress is durable.

    Two distinct concepts live here:
      * OpenSearch scroll PAGE: ``size=OS_SCROLL_PAGE`` (10K, cluster cap).
      * Our flush BATCH: ``batch_size`` (default 10K). One flush batch =
        ceil(batch_size / OS_SCROLL_PAGE) scroll pages.

    Resume / partition semantics:
      * `start_id`: a range filter ``{profile_id >= N}`` pushed into the
        OpenSearch query. True O(log N) seek — no skip-and-discard cost.
        Pair with `--limit` to process [start_id, start_id+limit-ish)
        windows. To resume, note the last `profile_id` logged by the
        previous run and pass it as the next `--start-id`.
      * `limit`: stop after N profiles have been PROCESSED.
    """
    if start_id is not None:
        body: dict = {
            "query": {"range": {"profile_id": {"gte": start_id}}},
            "_source": ["experiences.position_title"],
            "sort": [{"profile_id": "asc"}],
        }
    else:
        body = {
            "query": {"match_all": {}},
            "_source": ["experiences.position_title"],
        }

    scroll_size = min(batch_size, OS_SCROLL_PAGE)

    profiles_processed = 0
    batch_idx = 0
    batch_titles: list[str] = []
    batch_first_id: str | None = None
    last_processed_id: str | None = None

    # preserve_order=True only when start_id is given so that sort order
    # is respected (cleaner resume). Without start_id, _doc order is faster.
    preserve_order = start_id is not None

    for hit in helpers.scan(
        client=client,
        index=index,
        query=body,
        size=scroll_size,
        preserve_order=preserve_order,
        clear_scroll=True,
        scroll="10m",
    ):
        profile_id = str(hit.get("_id"))
        if batch_first_id is None:
            batch_first_id = profile_id

        src = hit.get("_source") or {}
        for exp in src.get("experiences") or []:
            title = (exp or {}).get("position_title")
            if title:
                batch_titles.append(title)

        profiles_processed += 1
        last_processed_id = profile_id

        if profiles_processed % batch_size == 0:
            batch_idx += 1
            yield (batch_idx, batch_first_id, last_processed_id, batch_titles)
            batch_titles = []
            batch_first_id = None

        if limit is not None and profiles_processed >= limit:
            break

    if batch_titles and batch_first_id is not None and last_processed_id is not None:
        batch_idx += 1
        yield (batch_idx, batch_first_id, last_processed_id, batch_titles)


def _fetch_existing_slugs(dsn: str, schema: str) -> set[str]:
    """Pull every existing slug from target.linkedin_roles so we can
    disambiguate against them, not just against this batch."""
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(f'SELECT slug FROM "{schema}".linkedin_roles')
            return {row[0] for row in cur.fetchall()}


def _disambiguate_slug(base: str, display_name: str, taken: set[str]) -> str:
    """If `base` is taken, append a 6-char content hash. Re-extend on the
    (vanishingly rare) double-collision."""
    if base not in taken:
        return base
    digest = hashlib.sha1(display_name.encode("utf-8")).hexdigest()
    for n in (6, 8, 12, 16):
        candidate = f"{base}-{digest[:n]}"
        if candidate not in taken:
            return candidate
    # Last resort — full digest. Statistically unreachable.
    return f"{base}-{digest}"


def _bulk_insert_linkedin_roles(
    dsn: str,
    schema: str,
    rows: list[tuple[str, str]],  # (slug, display_name)
    page_size: int = 5_000,
) -> int:
    """Insert (slug, display_name) into linkedin_roles. Slug is UNIQUE;
    ON CONFLICT DO NOTHING means rows already in the table are skipped.

    Returns the count of newly-inserted rows.
    """
    if not rows:
        return 0

    sql = (
        f'INSERT INTO "{schema}".linkedin_roles (slug, display_name) '
        f"VALUES %s "
        f"ON CONFLICT (slug) DO NOTHING"
    )

    inserted = 0
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            for start in range(0, len(rows), page_size):
                chunk = rows[start : start + page_size]
                psycopg2.extras.execute_values(cur, sql, chunk, page_size=page_size)
                inserted += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        conn.commit()
    return inserted


# ── Entry point ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Stream + clean + dedupe but skip the INSERT. Logs sample mappings.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help=(
            "Cap the number of PROFILES processed (not titles). With the "
            "10K default batch size, --limit 50000 means 5 batches."
        ),
    )
    p.add_argument(
        "--offset", dest="start_id", type=int, default=None,
        help=(
            "Resume / partition from this profile_id (inclusive). Pushed "
            "into the OpenSearch query as a range filter — true O(log N) "
            "seek, no skip-and-discard cost. To resume, copy the last "
            "profile_id from the previous run's batch log."
        ),
    )
    p.add_argument(
        "--batch-size", type=int, default=10_000,
        help="Flush batch size — INSERT every N profiles (default 10000).",
    )
    p.add_argument(
        "--sample", type=int, default=30,
        help="How many raw→cleaned examples to log (default 30).",
    )
    args = p.parse_args(argv)

    target_dsn = os.getenv("TARGET_DSN") or os.getenv("PG_PROD")
    target_schema = os.getenv("TARGET_SCHEMA", "skill-library")
    os_index = os.getenv("OPENSEARCH_INDEX", "profiles")

    if not target_dsn:
        logger.error("Set TARGET_DSN or PG_PROD for the write target.")
        return 2

    logger.info("source: OpenSearch index '%s'", os_index)
    logger.info("target: %s.linkedin_roles", target_schema)
    logger.info("batch size: %d profiles per flush", args.batch_size)
    if args.start_id is not None:
        logger.info("start_id: profile_id >= %d", args.start_id)
    if args.limit is not None:
        logger.info("limit: %d profiles", args.limit)
    if args.dry_run:
        logger.info("DRY RUN — no writes will be made")

    os_client = _build_opensearch_client()

    # Existing slugs from prior runs — used for in-stream dedup and
    # disambiguation. Cumulative `used_slugs` grows as we insert new
    # rows so future batches dedup against this run's own writes too.
    existing = _fetch_existing_slugs(target_dsn, target_schema)
    logger.info("target.linkedin_roles already has %d slugs", len(existing))

    used_slugs: set[str] = set(existing)
    # Cleaned display_names already enqueued/inserted this run — prevents
    # re-emitting the same role across batches.
    seen_names: set[str] = set()
    samples: list[tuple[str, str, str]] = []  # (raw, cleaned, slug)

    # Cumulative totals (logged at end).
    total_titles = 0
    total_dropped = 0
    total_existing = 0
    total_collisions = 0
    total_inserted = 0

    for batch_idx, first_id, last_id, raw_titles in _stream_title_batches(
        os_client, os_index,
        batch_size=args.batch_size, limit=args.limit, start_id=args.start_id,
    ):
        batch_new_rows: list[tuple[str, str]] = []
        b_dropped = 0
        b_existing = 0
        b_collisions = 0

        for raw in raw_titles:
            cleaned = clean_position_title(raw)
            if cleaned is None:
                b_dropped += 1
                continue
            if cleaned in seen_names:
                continue
            base_slug = slugify(cleaned)
            if base_slug == "unknown":
                b_dropped += 1
                continue
            if base_slug in used_slugs:
                # Already in linkedin_roles (from prior run or earlier
                # batch this run) — same display_name, same base slug.
                b_existing += 1
                seen_names.add(cleaned)
                continue
            slug = _disambiguate_slug(base_slug, cleaned, used_slugs)
            if slug != base_slug:
                b_collisions += 1
            used_slugs.add(slug)
            seen_names.add(cleaned)
            batch_new_rows.append((slug, cleaned))
            if len(samples) < args.sample:
                samples.append((raw, cleaned, slug))

        # Flush this batch before the next one begins.
        if batch_new_rows and not args.dry_run:
            inserted = _bulk_insert_linkedin_roles(
                target_dsn, target_schema, batch_new_rows
            )
        else:
            inserted = 0
        total_inserted += inserted

        total_titles += len(raw_titles)
        total_dropped += b_dropped
        total_existing += b_existing
        total_collisions += b_collisions

        logger.info(
            "  batch %d  profile_id %s → %s  | titles=%d  new=%d  dropped=%d  already_exists=%d  collisions=%d  inserted=%d",
            batch_idx, first_id, last_id, len(raw_titles), len(batch_new_rows),
            b_dropped, b_existing, b_collisions, inserted,
        )

    logger.info(
        "Done: titles=%d new=%d dropped=%d already_exists=%d collisions=%d inserted_total=%d",
        total_titles, len(seen_names) - total_existing,
        total_dropped, total_existing, total_collisions, total_inserted,
    )

    logger.info("Sample mappings (raw → cleaned → slug):")
    for raw, cleaned, slug in samples:
        logger.info("    %r  →  %r  →  %r", raw, cleaned, slug)

    if args.dry_run:
        logger.info("dry-run complete — no writes performed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
