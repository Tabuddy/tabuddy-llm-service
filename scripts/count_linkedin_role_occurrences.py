"""Backfill linkedin_roles.count from OpenSearch.

For each row in ``skill-library.linkedin_roles``, count how many
profiles in OpenSearch have at least one ``experiences[].position_title``
that contains the role's ``display_name`` as a phrase. Store the result
in ``linkedin_roles.count``. Used to filter out low-frequency junk roles
that survived the cleaning pass.

Why phrase-match instead of raw substring?
  ``match_phrase`` searches for the role as a sequence of tokens
  produced by the same analyzer that indexed the data. So:
    role "Software Engineer"     matches "Senior Software Engineer @ X"
    role "Software Engineer"     matches "Software Engineer III"
    role "Software Engineer"     does NOT match "Software Engineering"
  True character-substring (wildcard *X*) would be too slow at 4.5M
  roles × 70M docs. Phrase-match is what people usually want when they
  say "title contains this".

Performance:
  ~4.5M roles → naive 1-query-per-role = days. We batch via the
  OpenSearch ``_msearch`` API: --msearch-size queries (default 100)
  fit in a single HTTP request. UPDATEs are bulk-applied per batch
  via a single ``UPDATE … FROM (VALUES …)`` statement so we don't
  ping Postgres per row either.

Resumability:
  Rows with ``count IS NULL`` are processed; already-counted rows are
  skipped. Safe to Ctrl-C and re-run — work resumes mid-table.

Usage:
    # 1. Apply the migration first (adds the count column + index).
    psql "$PG_PROD" -f db/schema_linkedin_roles_count.sql

    # 2. Dry-run on first 50 roles — preview counts, no UPDATEs.
    python scripts/count_linkedin_role_occurrences.py --dry-run --limit 50

    # 3. Real backfill.
    python scripts/count_linkedin_role_occurrences.py

    # Resume mid-table by skipping already-counted rows (default) or
    # explicitly start from a specific linkedin_roles.id:
    python scripts/count_linkedin_role_occurrences.py --start-id 1000000

    # Recount everything (ignore the resume filter):
    python scripts/count_linkedin_role_occurrences.py --recount

Env vars (same as import_linkedin_position_titles.py):
    OPENSEARCH_URL / OPENSEARCH_INDEX / OPENSEARCH_USERNAME / OPENSEARCH_PASSWORD
    TARGET_DSN (defaults to PG_PROD) / TARGET_SCHEMA (defaults to 'skill-library')
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from typing import Iterator
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from opensearchpy.connection import Urllib3HttpConnection
from opensearchpy.exceptions import (
    ConnectionError as OSConnectionError,
    ConnectionTimeout as OSConnectionTimeout,
)

load_dotenv()


# Force urllib3 connection-pool size regardless of how the installed
# opensearchpy version exposes (or doesn't expose) maxsize on the main
# constructor. Subclass that hard-codes maxsize wins over kwarg-dropping.
class _PooledHttpConnection(Urllib3HttpConnection):
    """Hard-coded urllib3 pool size, robust to opensearch-py kwarg renames.

    v2.x exposes ``maxsize``. v3.x renamed it (or added new aliases like
    ``pool_maxsize``). Setting all known names plus reaching into the
    instantiated pool covers every release.
    """
    _POOL_SIZE = 32

    def __init__(self, *args, **kwargs):
        # Try every known kwarg name — extras are dropped silently by
        # the base class, named ones are honored where supported.
        for key in ("maxsize", "pool_maxsize", "connections_per_node"):
            existing = kwargs.get(key)
            if not existing or int(existing) < self._POOL_SIZE:
                kwargs[key] = self._POOL_SIZE
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            # Some kwarg name wasn't accepted — strip the aliases and retry.
            for key in ("pool_maxsize", "connections_per_node"):
                kwargs.pop(key, None)
            super().__init__(*args, **kwargs)

        # Belt-and-suspenders: if the underlying urllib3 pool exists and
        # is still sized at 1, resize its internal queue. The pool's
        # ``maxsize`` lives on a ``LifoQueue``.
        pool = getattr(self, "pool", None)
        if pool is not None and getattr(pool, "pool", None) is not None:
            try:
                pool.pool.maxsize = self._POOL_SIZE
            except Exception:
                pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("count_linkedin_roles")

for _noisy in (
    "opensearch", "opensearch.trace", "opensearch.connection",
    "opensearch.connection.http_urllib3",
    "opensearchpy", "opensearchpy.trace", "opensearchpy.connection",
    "opensearchpy.connection.http_urllib3",
    "urllib3",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# urllib3 emits "Connection pool is full, discarding connection" at
# WARNING level whenever pool size < concurrent requests. We've tried
# bumping the pool via OpenSearch(maxsize=32) and a connection_class
# subclass, but older opensearchpy versions silently drop the kwarg.
# Until that's resolved the warning is cosmetic — the requests still
# succeed, just with TCP/TLS churn that costs ~10-30% throughput.
# Suppress it to keep the log readable.
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


# Exception types that are retryable. Everything else (4xx, query errors,
# data validation) bubbles up immediately — only network-level hiccups
# are worth a retry.
_RETRY_ERRORS: tuple[type[BaseException], ...] = (
    psycopg2.OperationalError,
    psycopg2.InterfaceError,  # "connection already closed" etc.
    OSConnectionError,
    OSConnectionTimeout,
)


def _retry(max_attempts: int = 4, base_delay: float = 1.0, factor: float = 4.0):
    """Exponential-backoff retry on transient network errors.

    Default schedule: attempt 1 immediate, then 1s / 4s / 16s before
    the final attempt. After ``max_attempts`` failures, re-raises.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except _RETRY_ERRORS as exc:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(
                            "%s gave up after %d attempts: %s",
                            fn.__name__, attempt, exc,
                        )
                        raise
                    delay = base_delay * (factor ** (attempt - 1))
                    logger.warning(
                        "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                        fn.__name__, attempt, max_attempts, exc, delay,
                    )
                    time.sleep(delay)
        return wrapper
    return decorator


# ── OpenSearch client (mirrors import_linkedin_position_titles.py) ──────────


def _build_opensearch_client() -> OpenSearch:
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
    # connection_class=_PooledHttpConnection forces urllib3 maxsize=32
    # regardless of opensearchpy version. Without this, the default
    # pool size of 1 causes "Connection pool is full" warnings + TCP
    # churn under concurrent workers.
    return OpenSearch(
        hosts=[{"host": host, "port": port, "scheme": scheme}],
        http_auth=(username, password),
        use_ssl=scheme == "https",
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=timeout,
        max_retries=2,
        retry_on_timeout=True,
        connection_class=_PooledHttpConnection,
    )


# ── DB plumbing ─────────────────────────────────────────────────────────────


# Whitelist for target columns — used directly in SQL so it MUST come
# from argparse choices, not user input. Both columns are BIGINT in
# linkedin_roles; ``count`` is pass-1 (nested_phrase), ``query_match_count``
# is pass-2 (nested_substring).
_TARGET_COLUMNS = ("count", "query_match_count")


@_retry()
def _fetch_roles_page(
    dsn: str, schema: str, page_size: int, cursor_id: int,
    recount: bool, min_count: int | None, target_column: str,
) -> list[tuple[int, str]]:
    """One keyset-paged page fetch — wrapped so a flaky Postgres
    connection doesn't kill the whole multi-hour run.

    Two independent filters combine:
      Selection:  --min-count N  → ``count >= N``  (which rows to consider)
      Resume:     ``{target_column} IS NULL``      (skip already-done rows;
                  bypassed by --recount)
    """
    assert target_column in _TARGET_COLUMNS, target_column  # SQL safety

    conds = ["id > %s"]
    params: list = [cursor_id]
    if min_count is not None:
        conds.append("count >= %s")
        params.append(min_count)
    if not recount:
        conds.append(f"{target_column} IS NULL")

    sql = (
        f'SELECT id, display_name '
        f'FROM "{schema}".linkedin_roles '
        f'WHERE {" AND ".join(conds)} '
        f'ORDER BY id LIMIT {int(page_size)}'
    )
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            return cur.fetchall()


def _stream_roles(
    dsn: str,
    schema: str,
    page_size: int,
    start_id: int | None,
    recount: bool,
    min_count: int | None,
    target_column: str,
) -> Iterator[tuple[int, str]]:
    """Yield (id, display_name) rows from linkedin_roles ordered by id.

    See ``_fetch_roles_page`` for row selection. Streams keyset-paged on
    id so we never load 4.5M rows into memory.
    """
    cursor_id = (start_id - 1) if start_id is not None else -1
    while True:
        rows = _fetch_roles_page(
            dsn, schema, page_size, cursor_id, recount, min_count, target_column,
        )
        if not rows:
            return
        for rid, name in rows:
            yield rid, name
        cursor_id = rows[-1][0]


@_retry()
def _bulk_update_counts(
    dsn: str,
    schema: str,
    counts: list[tuple[int, int]],  # (id, count)
    target_column: str,
) -> int:
    """UPDATE linkedin_roles.{target_column} for the given (id, count)
    pairs in a single round-trip using ``UPDATE … FROM (VALUES …)``."""
    if not counts:
        return 0
    assert target_column in _TARGET_COLUMNS, target_column  # SQL safety
    sql = (
        f'UPDATE "{schema}".linkedin_roles AS lr '
        f"SET {target_column} = data.cnt "
        f"FROM (VALUES %s) AS data(id, cnt) "
        f"WHERE lr.id = data.id"
    )
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, counts, page_size=len(counts))
            affected = cur.rowcount
        conn.commit()
    return affected if affected and affected > 0 else 0


# ── OpenSearch counting ─────────────────────────────────────────────────────


def _build_query(mode: str, name: str) -> dict:
    """Return the OpenSearch query body for a single role name.

    Modes:
      * ``match_phrase``    — phrase-match on the analyzed text field.
                              Works if experiences is NOT nested. WARNING:
                              the analyzer stems + drops stopwords, so
                              "IT Manager", "Managers", "Manage" all reduce
                              to the same token [manag] and get the same
                              count. Not what you usually want.
      * ``nested_phrase``   — match_phrase wrapped in a nested query.
                              Same stemming/stopword caveat applies.
      * ``nested_substring``— nested + wildcard on the ``.keyword`` subfield.
                              TRUE case-insensitive substring match against
                              the raw stored string. Slower but accurate —
                              "IT Manager" matches only titles literally
                              containing "it manager".
      * ``wildcard``        — wildcard at the top level (no nested wrap).
                              Mostly a probe fallback.
    """
    if mode == "match_phrase":
        return {"match_phrase": {"experiences.position_title": name}}
    if mode == "nested_phrase":
        return {
            "nested": {
                "path": "experiences",
                "query": {"match_phrase": {"experiences.position_title": name}},
            }
        }
    if mode in ("wildcard", "nested_substring"):
        # Escape user-supplied wildcard metacharacters in the name itself;
        # the surrounding *…* is what enables substring match.
        safe = name.replace("\\", "\\\\").replace("*", "\\*").replace("?", "\\?")
        wildcard_clause = {
            "wildcard": {
                # .keyword subfield is the unanalyzed (no stemming, no
                # stopword removal) raw string. Required for "literal
                # contains" semantics.
                "experiences.position_title.keyword": {
                    "value": f"*{safe}*",
                    "case_insensitive": True,
                }
            }
        }
        if mode == "wildcard":
            return wildcard_clause
        return {
            "nested": {
                "path": "experiences",
                "query": wildcard_clause,
            }
        }
    raise ValueError(f"unknown query mode: {mode!r}")


@_retry()
def _count_batch_via_msearch(
    client: OpenSearch,
    index: str,
    roles: list[tuple[int, str]],
    mode: str,
) -> dict[int, int]:
    """Run one ``_msearch`` request, one sub-query per role, using the
    given query mode. Returns ``{role_id: count}``.
    """
    if not roles:
        return {}

    body: list[dict] = []
    for _, name in roles:
        body.append({"index": index})
        body.append({
            "size": 0,
            "track_total_hits": True,
            "query": _build_query(mode, name),
        })

    resp = client.msearch(body=body)
    out: dict[int, int] = {}
    for (rid, _), sub in zip(roles, resp.get("responses", [])):
        if "error" in sub:
            logger.warning("msearch error for role_id=%d: %s", rid, sub["error"])
            out[rid] = 0
            continue
        total = ((sub.get("hits") or {}).get("total") or {}).get("value", 0)
        out[rid] = int(total)
    return out


# ── Per-batch worker (called from ThreadPoolExecutor) ───────────────────────


def _process_one_batch(
    batch: list[tuple[int, str]],
    *,
    client: OpenSearch,
    index: str,
    mode: str,
    target_dsn: str,
    target_schema: str,
    target_column: str,
    dry_run: bool,
) -> tuple[int, int, list[tuple[str, int]]]:
    """One unit of work: msearch + UPDATE for a single batch.

    Returns (n_processed, n_updated, [(name, count), …]) so the main
    thread can tally totals and pull samples without touching shared
    state from inside the worker.
    """
    if not batch:
        return 0, 0, []
    id_to_count = _count_batch_via_msearch(client, index, batch, mode)
    if dry_run:
        updated = 0
    else:
        updates = [(rid, id_to_count.get(rid, 0)) for rid, _ in batch]
        updated = _bulk_update_counts(
            target_dsn, target_schema, updates, target_column,
        )
    sample_pairs = [(name, id_to_count.get(rid, 0)) for rid, name in batch[:5]]
    return len(batch), updated, sample_pairs


# ── Probe: try multiple query modes against a fixed set of roles ────────────


def _probe(client: OpenSearch, index: str) -> None:
    """Diagnose which query mode actually finds matches. Runs each mode
    against a handful of common roles + queries the mapping for the
    `experiences.position_title` field. Prints a side-by-side count
    table so it's obvious which mode to use in production."""
    test_roles = [
        "Software Engineer", "UI Designer", "Doctor", "Sales Manager",
        "Project Manager", "Student", "Data Scientist", "Manager",
    ]

    logger.info("─── Mapping for experiences.position_title ───")
    try:
        mapping = client.indices.get_field_mapping(
            fields="experiences.position_title", index=index,
        )
        for idx_name, idx_body in mapping.items():
            for field_name, field_def in (idx_body.get("mappings") or {}).items():
                logger.info("  %s · %s : %s", idx_name, field_name, field_def)
    except Exception as exc:
        logger.warning("could not fetch mapping: %s", exc)

    modes = ["match_phrase", "nested_phrase", "nested_substring", "wildcard"]
    header = f"  {'role':<26} | " + " | ".join(f"{m:>17}" for m in modes)
    logger.info("─── Counts by query mode ───")
    logger.info(header)
    logger.info("  " + "-" * (26 + 4 + 17 * len(modes)))
    for role in test_roles:
        cells = []
        for mode in modes:
            try:
                resp = client.search(
                    index=index, body={
                        "size": 0,
                        "track_total_hits": True,
                        "query": _build_query(mode, role),
                    },
                )
                total = ((resp.get("hits") or {}).get("total") or {}).get("value", 0)
                cells.append(f"{total:>17,}")
            except Exception as exc:
                cells.append(f"{'ERR':>17}")
                logger.debug("  %s [%s] error: %s", role, mode, exc)
        logger.info("  %-26s | %s", role, " | ".join(cells))


# ── Entry point ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute counts but skip the UPDATE. Logs samples.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap total roles processed (debug / preview).",
    )
    p.add_argument(
        "--start-id", dest="start_id", type=int, default=None,
        help="Resume from linkedin_roles.id >= N (default: process all NULL counts).",
    )
    p.add_argument(
        "--recount", action="store_true",
        help="Re-process even rows that already have a count value.",
    )
    p.add_argument(
        "--min-count", dest="min_count", type=int, default=None,
        help=(
            "Selection filter: only consider rows where count >= N. "
            "Pass-2 use case: ``--min-count 10000 --target-column "
            "query_match_count --query-mode nested_substring`` runs the "
            "accurate substring query only on rows that already cleared "
            "a pass-1 frequency threshold."
        ),
    )
    p.add_argument(
        "--target-column",
        dest="target_column",
        choices=list(_TARGET_COLUMNS),
        default="count",
        help=(
            "Which column to write results to (and to use for the "
            "``IS NULL`` resume filter). 'count' is the pass-1 column "
            "(nested_phrase results). 'query_match_count' is the pass-2 "
            "column (nested_substring results)."
        ),
    )
    p.add_argument(
        "--msearch-size", type=int, default=None,
        help=(
            "Roles per _msearch request. Default depends on --query-mode: "
            "100 for fast phrase queries (match_phrase / nested_phrase), "
            "25 for slow wildcard queries (nested_substring / wildcard) "
            "to localize the blast radius if one common substring "
            "(*Manage*, *Sales*) stalls a batch. Pass explicitly to override."
        ),
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help=(
            "Concurrent msearch + UPDATE workers (default 4). Each worker "
            "runs one batch end-to-end in its own thread. 4 ≈ 4× speedup; "
            "crank to 8 if OpenSearch isn't sweating. 1 = serial."
        ),
    )
    p.add_argument(
        "--query-mode",
        choices=["match_phrase", "nested_phrase", "nested_substring", "wildcard"],
        default="nested_substring",
        help=(
            "Which OpenSearch query to use per role. Default "
            "'nested_substring' does a true case-insensitive substring "
            "match against the .keyword subfield, wrapped in a nested "
            "query (experiences is nested in the prod mapping). "
            "'nested_phrase' is faster but unreliable — its analyzer "
            "stems words and drops stopwords, so e.g. 'IT Manager' and "
            "'Manage' end up matching the same docs. Run --probe to see."
        ),
    )
    p.add_argument(
        "--probe", action="store_true",
        help=(
            "Diagnostic: print the field mapping for experiences.position_title "
            "and run all three query modes against a few common roles. Use this "
            "when counts are all 0 to figure out which --query-mode to use."
        ),
    )
    p.add_argument(
        "--db-page-size", type=int, default=10_000,
        help="Roles fetched from Postgres per page (default 10000).",
    )
    p.add_argument(
        "--sample", type=int, default=20,
        help="How many (role → count) examples to log (default 20).",
    )
    args = p.parse_args(argv)

    # Auto-tune msearch-size when not explicitly set: small batches for
    # wildcard-based modes (where one slow sub-query stalls a whole
    # msearch), larger batches for fast phrase modes.
    if args.msearch_size is None:
        args.msearch_size = (
            25 if args.query_mode in ("nested_substring", "wildcard") else 100
        )

    target_dsn = os.getenv("TARGET_DSN") or os.getenv("PG_PROD")
    target_schema = os.getenv("TARGET_SCHEMA", "skill-library")
    os_index = os.getenv("OPENSEARCH_INDEX", "profiles")

    if not target_dsn:
        logger.error("Set TARGET_DSN or PG_PROD.")
        return 2

    logger.info("source: OpenSearch index '%s'", os_index)
    logger.info(
        "target: %s.linkedin_roles.%s", target_schema, args.target_column,
    )
    if args.start_id is not None:
        logger.info("start-id: %d", args.start_id)
    if args.limit is not None:
        logger.info("limit: %d roles", args.limit)
    if args.min_count is not None:
        logger.info("min-count: re-processing rows where count >= %d", args.min_count)
    if args.recount:
        logger.info("recount: re-processing already-counted rows")
    if args.dry_run:
        logger.info("DRY RUN — no UPDATEs will be issued")
    logger.info("msearch size: %d roles per request", args.msearch_size)

    client = _build_opensearch_client()

    if args.probe:
        _probe(client, os_index)
        return 0

    logger.info("query mode: %s", args.query_mode)
    logger.info("workers: %d concurrent msearch/UPDATE", args.workers)

    run_started = time.monotonic()
    samples: list[tuple[str, int]] = []
    processed = 0
    updated_total = 0
    last_log_threshold = 0  # for progress log throttling

    # Effective flush size = msearch_size, but never larger than --limit.
    flush_size = args.msearch_size
    if args.limit is not None:
        flush_size = min(flush_size, args.limit)

    # Substring/wildcard modes have high per-batch latency variance — one
    # slow sub-query (*Software*, *Manager*) can stall a whole msearch for
    # a minute+. Log per-batch in those modes so the user sees heartbeat.
    # Phrase modes are fast (~10ms/msearch), so keep the 10× throttle.
    log_every_batch = args.query_mode in ("nested_substring", "wildcard")

    def _harvest(fut: Future) -> None:
        """Pull a completed future's result into the main-thread totals."""
        nonlocal processed, updated_total, last_log_threshold
        n_processed, n_updated, batch_samples = fut.result()
        processed += n_processed
        updated_total += n_updated
        for s in batch_samples:
            if len(samples) < args.sample:
                samples.append(s)
        if log_every_batch:
            logger.info(
                "  processed %d roles, updated %d (last batch: %d roles, sample: %s = %d)",
                processed, updated_total, n_processed,
                batch_samples[0][0] if batch_samples else "?",
                batch_samples[0][1] if batch_samples else 0,
            )
        elif processed - last_log_threshold >= flush_size * 10:
            last_log_threshold = processed - (processed % (flush_size * 10))
            logger.info(
                "  processed %d roles, updated %d", processed, updated_total,
            )

    # Iterate the role stream on the main thread; each batch is submitted
    # to the pool. We keep at most workers*2 batches in flight to bound
    # memory and avoid the queue running far ahead of the stream.
    role_iter = _stream_roles(
        target_dsn, target_schema,
        page_size=args.db_page_size,
        start_id=args.start_id,
        recount=args.recount,
        min_count=args.min_count,
        target_column=args.target_column,
    )

    pending: set[Future] = set()
    MAX_PENDING = max(1, args.workers * 2)
    batch: list[tuple[int, str]] = []
    submitted = 0  # batches submitted so far × flush_size (upper bound)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:

        def _submit(b: list[tuple[int, str]]) -> None:
            nonlocal submitted
            fut = pool.submit(
                _process_one_batch, b,
                client=client, index=os_index, mode=args.query_mode,
                target_dsn=target_dsn, target_schema=target_schema,
                target_column=args.target_column,
                dry_run=args.dry_run,
            )
            pending.add(fut)
            if submitted == 0:
                logger.info(
                    "  first batch submitted (%d roles) — waiting for OpenSearch…",
                    len(b),
                )
            submitted += len(b)

        for rid, name in role_iter:
            batch.append((rid, name))
            if len(batch) >= flush_size:
                _submit(batch)
                batch = []

                # Backpressure: wait for one to finish before queuing more.
                if len(pending) >= MAX_PENDING:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        _harvest(fut)

                # Respect --limit at submission time (might overshoot
                # slightly by up to (workers-1) batches in flight; fine).
                if args.limit is not None and submitted >= args.limit:
                    break

        # Tail partial batch.
        if batch and (args.limit is None or submitted < args.limit):
            _submit(batch)
            batch = []

        # Drain remaining in-flight work.
        for fut in as_completed(pending):
            _harvest(fut)

    elapsed = time.monotonic() - run_started
    rps = processed / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Done. processed=%d updated=%d elapsed=%.1fs throughput=%.1f roles/sec",
        processed, updated_total, elapsed, rps,
    )

    # Project against a typical remaining table size (4.5M) so the
    # dry-run timing answer "is this fast enough?" is on screen.
    if rps > 0:
        for target_remaining in (4_500_000,):
            secs = target_remaining / rps
            hrs = secs / 3600
            logger.info(
                "  → at this rate, %s remaining roles ≈ %.1f h (%.0f min)",
                f"{target_remaining:,}", hrs, secs / 60,
            )

    logger.info("Sample (role → count):")
    for name, cnt in samples:
        logger.info("    %8d  %s", cnt, name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
