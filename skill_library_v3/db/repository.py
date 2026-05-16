"""Sync psycopg2 CRUD over the v3 catalog tables.

All methods are blocking; FastAPI handlers wrap them with
``asyncio.to_thread(...)`` to avoid stalling the event loop. This mirrors the
existing :class:`SkillLibraryRepository` pattern.

The v3 pipeline maps Stage 0 charter state onto the existing schema:

* ``roles``            → role catalog (UPSERTed on first generate)
* ``v2_run_log``       → one row per Stage 0 generation attempt
                         (status: running / awaiting_review / approved / rejected / failed)
* ``v2_review_queue``  → human-review gate, item_type='charter_review'
* ``jd_samples``       → Kaggle JD ingest (additive aux table)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Iterable

import psycopg2
import psycopg2.extras

from skill_library_v3.db.connection import connect

logger = logging.getLogger(__name__)


CHARTER_PROMPT_VERSION_PREFIX = "stage0_charter_"
STAGE1_PROMPT_VERSION_PREFIX = "stage1_anchor_"
STAGE2_PROMPT_VERSION_PREFIX = "stage2_dim_gen_"
STAGE3_PROMPT_VERSION_PREFIX = "stage3_recon_"
STAGE4_PROMPT_VERSION_PREFIX = "stage4_typing_"
STAGE5_PROMPT_VERSION_PREFIX = "stage5_placement_"
STAGE6_PROMPT_VERSION_PREFIX = "stage6_containment_"
STAGE7_PROMPT_VERSION_PREFIX = "stage7_enrichment_"
STAGE8_PROMPT_VERSION_PREFIX = "stage8_load_"
CHARTER_REVIEW_ITEM_TYPE = "charter_review"
ROLE_CARD_REVIEW_ITEM_TYPE = "role_card_review"
RECONCILIATION_REVIEW_ITEM_TYPE = "reconciliation_review"
CONTAINMENT_REVIEW_ITEM_TYPE = "containment_review"
ENRICHMENT_REVIEW_ITEM_TYPE = "enrichment_review"
LOAD_REVIEW_ITEM_TYPE = "load_review"


def _item_type_for_prompt_version(prompt_version: str | None) -> str | None:
    """Map a v2_run_log.prompt_version to its v2_review_queue item_type.

    Returns None for headless stages (Stage 2 dim generation, Stage 4
    typing, Stage 5 placement). Stage 6 has a human-review path
    (cycles or type-rule violations need human attention) so it has
    its own item_type.
    """
    pv = prompt_version or ""
    if pv.startswith(STAGE1_PROMPT_VERSION_PREFIX):
        return ROLE_CARD_REVIEW_ITEM_TYPE
    if pv.startswith(STAGE2_PROMPT_VERSION_PREFIX):
        return None
    if pv.startswith(STAGE3_PROMPT_VERSION_PREFIX):
        return RECONCILIATION_REVIEW_ITEM_TYPE
    if pv.startswith(STAGE4_PROMPT_VERSION_PREFIX):
        return None
    if pv.startswith(STAGE5_PROMPT_VERSION_PREFIX):
        return None
    if pv.startswith(STAGE6_PROMPT_VERSION_PREFIX):
        return CONTAINMENT_REVIEW_ITEM_TYPE
    if pv.startswith(STAGE7_PROMPT_VERSION_PREFIX):
        return ENRICHMENT_REVIEW_ITEM_TYPE
    if pv.startswith(STAGE8_PROMPT_VERSION_PREFIX):
        return LOAD_REVIEW_ITEM_TYPE
    return CHARTER_REVIEW_ITEM_TYPE


# ── helpers ────────────────────────────────────────────────────────────────


def slugify(text: str) -> str:
    """Lowercase-with-dashes slug. Empty input yields ``"unknown"``."""
    out: list[str] = []
    last_dash = False
    for ch in (text or "").strip().lower():
        if ch.isalnum():
            out.append(ch)
            last_dash = False
        elif ch in (" ", "-", "_", "/", ".", ","):
            if not last_dash and out:
                out.append("-")
                last_dash = True
    return "".join(out).strip("-") or "unknown"


def _row_to_dict(cur, row) -> dict | None:
    if row is None:
        return None
    return dict(zip([c[0] for c in cur.description], row))


def _rows_to_dicts(cur, rows) -> list[dict]:
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in rows]


def _first_sentence(text: str | None, *, fallback: str = "") -> str:
    """Return the first sentence of ``text`` (up to first period/?/!), trimmed.

    Used to compress an approved charter's full ``definition`` into the brief
    paragraph the next role's prompt sees as adjacency context.
    """
    if not text:
        return fallback
    m = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return (m[0] if m else text.strip()).strip()


# ── role catalog ────────────────────────────────────────────────────────────


def upsert_role(role_name: str) -> dict:
    """Find or insert a role by display_name (case-insensitive). Returns the
    full row including ``id`` and ``slug``."""
    name = (role_name or "").strip()
    if not name:
        raise ValueError("role_name cannot be empty")
    slug = slugify(name)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, slug, display_name, role_archetype
                  FROM roles
                 WHERE lower(slug) = %s OR lower(display_name) = %s
                 LIMIT 1
                """,
                (slug, name.lower()),
            )
            existing = cur.fetchone()
            if existing:
                return _row_to_dict(cur, existing)

            cur.execute(
                """
                INSERT INTO roles (slug, display_name, source)
                VALUES (%s, %s, 'MANUAL_CURATION')
                RETURNING id, slug, display_name, role_archetype
                """,
                (slug, name),
            )
            row = cur.fetchone()
            cols = [c[0] for c in cur.description]
        conn.commit()
    return dict(zip(cols, row))


def list_roles() -> list[dict]:
    """All roles, with whether they have an approved charter yet. Used by the
    UI's role dropdown and recent-runs lane."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    r.id, r.slug, r.display_name, r.role_archetype,
                    EXISTS (
                        SELECT 1 FROM v2_run_log v
                         WHERE v.role_id = r.slug
                           AND v.status = 'approved'
                           AND v.prompt_version LIKE %s
                    ) AS has_approved_charter
                  FROM roles r
                 ORDER BY r.display_name
                """,
                (CHARTER_PROMPT_VERSION_PREFIX + "%",),
            )
            return _rows_to_dicts(cur, cur.fetchall())


def get_approved_adjacent_roles(*, exclude_slug: str) -> list[dict]:
    """Return one row per role that has a current approved charter (other than
    ``exclude_slug``). Used to build the adjacency block in the Stage 0 prompt.

    Each row carries ``slug``, ``display_name``, and ``brief`` (first sentence
    of the approved charter's ``definition``).
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (v.role_id)
                       r.slug              AS slug,
                       r.display_name      AS display_name,
                       v.planner_output->>'definition' AS definition
                  FROM v2_run_log v
                  JOIN roles r ON r.slug = v.role_id
                 WHERE v.status = 'approved'
                   AND v.prompt_version LIKE %s
                   AND v.role_id <> %s
                 ORDER BY v.role_id, v.completed_at DESC
                """,
                (CHARTER_PROMPT_VERSION_PREFIX + "%", exclude_slug),
            )
            rows = _rows_to_dicts(cur, cur.fetchall())

    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "slug": r["slug"],
                "display_name": r["display_name"],
                "brief": _first_sentence(r.get("definition"), fallback=r["display_name"]),
            }
        )
    return out


def get_approved_role_names() -> set[str]:
    """Lowercase display_names of every role with an approved charter.
    Used by the validator to verify ``adjacent_roles[].role`` and
    ``out_of_scope[].owned_by`` references resolve."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT lower(r.display_name) AS name
                  FROM v2_run_log v
                  JOIN roles r ON r.slug = v.role_id
                 WHERE v.status = 'approved'
                   AND v.prompt_version LIKE %s
                """,
                (CHARTER_PROMPT_VERSION_PREFIX + "%",),
            )
            return {row[0] for row in cur.fetchall() if row[0]}


def get_approved_role_aliases() -> dict[str, str]:
    """Map every recognized role name (canonical + alias, lowercased)
    to the canonical role slug. Used by the Stage 0 charter validator
    to refuse charters whose ``role_name`` aliases an already-approved
    role.

    Sources two layers:
      1. Canonical display_names of roles with an approved Stage 0 charter
         (``roles.display_name`` joined to ``v2_run_log``).
      2. Aliases lifted from each role's most recent approved Stage 1
         role card (``planner_output->>aliases``).

    Canonical wins when an alias collides with another role's canonical
    name — we only ``setdefault`` aliases on top of canonicals so an
    existing canonical entry is never shadowed.
    """
    out: dict[str, str] = {}
    with connect() as conn:
        with conn.cursor() as cur:
            # Layer 1 — canonical names of roles with an approved charter.
            cur.execute(
                """
                SELECT DISTINCT r.slug, lower(r.display_name)
                  FROM v2_run_log v
                  JOIN roles r ON r.slug = v.role_id
                 WHERE v.status = 'approved'
                   AND v.prompt_version LIKE %s
                """,
                (CHARTER_PROMPT_VERSION_PREFIX + "%",),
            )
            for slug, name in cur.fetchall():
                if name:
                    out[name] = slug
            # Layer 2 — aliases from approved Stage 1 role cards (latest
            # per role).
            cur.execute(
                """
                SELECT DISTINCT ON (role_id)
                       role_id, planner_output
                  FROM v2_run_log
                 WHERE status = 'approved'
                   AND prompt_version LIKE 'stage1_%%'
                 ORDER BY role_id, started_at DESC
                """
            )
            for slug, po in cur.fetchall():
                if po is None:
                    continue
                rc = po if isinstance(po, dict) else json.loads(po)
                aliases = rc.get("aliases") or []
                if not isinstance(aliases, list):
                    continue
                for alias in aliases:
                    if not isinstance(alias, str):
                        continue
                    a = alias.strip().lower()
                    if a:
                        out.setdefault(a, slug)
    return out


# ── stage runs ──────────────────────────────────────────────────────────────


def create_pending_run(
    *,
    role_slug: str,
    role_display: str,
    prompt_version: str,
    model_snapshot: str,
    parent_run_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """Insert a ``v2_run_log`` row in status='pending'. parent_run_id (for
    regenerate lineage) is stashed in the eventual review_queue payload, not
    in v2_run_log itself (no column for it on this branch)."""
    run_id = uuid.uuid4()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO v2_run_log
                    (run_id, role_id, role_display, prompt_version,
                     model_snapshot, planner_output, status)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, 'pending')
                """,
                (
                    str(run_id),
                    role_slug,
                    role_display,
                    prompt_version,
                    model_snapshot,
                    json.dumps(
                        {
                            "_parent_run_id": str(parent_run_id) if parent_run_id else None,
                        }
                    ),
                ),
            )
        conn.commit()
    return run_id


def transition_run(
    run_id: uuid.UUID,
    *,
    status: str,
    error_message: str | None = None,
) -> None:
    """Move a run between non-terminal states (pending → running, etc.)."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE v2_run_log
                   SET status        = %s,
                       error_message = COALESCE(%s, error_message),
                       started_at    = CASE
                                          WHEN %s = 'running' AND started_at IS NULL
                                          THEN NOW()
                                          ELSE started_at
                                       END
                 WHERE run_id = %s
                """,
                (status, error_message, status, str(run_id)),
            )
        conn.commit()


def complete_charter_run(
    run_id: uuid.UUID,
    *,
    charter: dict,
    validator_log: dict,
    final_status: str = "awaiting_review",
    planner_reasoning: str | None = None,
) -> str:
    """Stage 0 finished — write charter to planner_output, transition to
    ``final_status`` (``approved`` | ``awaiting_review`` | ``failed``).
    Review-queue row is inserted only on awaiting_review.

    For the autonomy path: a clean charter (no validator errors, all
    minimum fields populated) is set to ``approved`` and the runner
    cascades into Stage 1 directly. Charters with thin content or
    validator errors still flow through awaiting_review for human
    inspection.
    """
    if final_status not in {"approved", "awaiting_review", "failed"}:
        raise ValueError(f"unexpected final_status={final_status!r}")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role_id, planner_output FROM v2_run_log WHERE run_id = %s",
                (str(run_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"v2_run_log row {run_id} not found")
            role_slug = row[0]
            prior = row[1] or {}
            parent_run_id = prior.get("_parent_run_id") if isinstance(prior, dict) else None

            output = dict(charter)
            output["_parent_run_id"] = parent_run_id
            output["_validator_log"] = validator_log

            err_msg: str | None = None
            if final_status == "failed":
                errs = validator_log.get("errors", [])[:3]
                err_msg = "; ".join(
                    f"[{e.get('code', '?')}] {e.get('message', '?')}"
                    for e in errs
                ) or "Stage 0 charter failed validation"

            cur.execute(
                """
                UPDATE v2_run_log
                   SET planner_output    = %s::jsonb,
                       planner_reasoning = %s,
                       status            = %s,
                       completed_at      = NOW(),
                       error_message     = %s
                 WHERE run_id = %s
                """,
                (json.dumps(output), planner_reasoning, final_status,
                 err_msg, str(run_id)),
            )

            if final_status == "awaiting_review":
                review_payload = {
                    "role_slug": role_slug,
                    "charter": charter,
                    "validator_log": validator_log,
                    "parent_run_id": parent_run_id,
                }
                warn_count = len(validator_log.get("warnings", []))
                err_count = len(validator_log.get("errors", []))
                reason = (
                    f"Charter awaiting admin approval — "
                    f"{warn_count} validator warning(s), {err_count} error(s)"
                )
                cur.execute(
                    """
                    INSERT INTO v2_review_queue (run_id, item_type, payload, reason)
                    VALUES (%s, %s, %s::jsonb, %s)
                    """,
                    (
                        str(run_id),
                        CHARTER_REVIEW_ITEM_TYPE,
                        json.dumps(review_payload),
                        reason,
                    ),
                )
        conn.commit()
    return final_status


def complete_role_card_run(
    run_id: uuid.UUID,
    *,
    role_card: dict,
    validator_log: dict,
    planner_reasoning: str | None = None,
) -> str:
    """Stage 1 finished cleanly. Auto-approves when validator_log['errors'] is
    empty; otherwise transitions to awaiting_review and surfaces in the UI.

    Returns the resulting status string ('approved' | 'awaiting_review') so
    the runner can log it. The review-queue row is always inserted (even on
    auto-approve) for audit; it just lands as ``resolved=TRUE`` with
    ``resolved_by='auto'`` in that case.
    """
    has_errors = bool(validator_log.get("errors"))
    final_status = "awaiting_review" if has_errors else "approved"

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role_id, planner_output FROM v2_run_log WHERE run_id = %s",
                (str(run_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"v2_run_log row {run_id} not found")
            role_slug = row[0]
            prior = row[1] or {}
            parent_run_id = (
                prior.get("_parent_run_id") if isinstance(prior, dict) else None
            )

            output = dict(role_card)
            # Carry the lineage marker through so future audits can trace the
            # role card back to its trigger (Stage 0 charter on first run,
            # prior Stage 1 row on regenerate).
            output["_parent_run_id"] = parent_run_id

            cur.execute(
                """
                UPDATE v2_run_log
                   SET planner_output    = %s::jsonb,
                       planner_reasoning = %s,
                       status            = %s,
                       completed_at      = NOW(),
                       error_message     = NULL
                 WHERE run_id = %s
                """,
                (json.dumps(output), planner_reasoning, final_status, str(run_id)),
            )

            review_payload = {
                "role_slug": role_slug,
                "role_card": role_card,
                "validator_log": validator_log,
                "parent_run_id": parent_run_id,
            }
            warn_count = len(validator_log.get("warnings", []))
            err_count = len(validator_log.get("errors", []))
            if has_errors:
                reason = (
                    f"Role card awaiting admin approval — "
                    f"{err_count} validator error(s), {warn_count} warning(s)"
                )
                cur.execute(
                    """
                    INSERT INTO v2_review_queue
                        (run_id, item_type, payload, reason)
                    VALUES (%s, %s, %s::jsonb, %s)
                    """,
                    (
                        str(run_id),
                        ROLE_CARD_REVIEW_ITEM_TYPE,
                        json.dumps(review_payload),
                        reason,
                    ),
                )
            else:
                reason = (
                    f"Auto-approved on validators-pass — "
                    f"{warn_count} warning(s), 0 error(s)"
                )
                cur.execute(
                    """
                    INSERT INTO v2_review_queue
                        (run_id, item_type, payload, reason,
                         resolved, resolved_by, resolved_at)
                    VALUES (%s, %s, %s::jsonb, %s,
                            TRUE, 'auto', NOW())
                    """,
                    (
                        str(run_id),
                        ROLE_CARD_REVIEW_ITEM_TYPE,
                        json.dumps(review_payload),
                        reason,
                    ),
                )
        conn.commit()
    return final_status


def fail_run(run_id: uuid.UUID, *, error_message: str) -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE v2_run_log
                   SET status        = 'failed',
                       error_message = %s,
                       completed_at  = NOW()
                 WHERE run_id = %s
                """,
                (error_message[:8000], str(run_id)),
            )
        conn.commit()


def get_run_with_review(run_id: uuid.UUID) -> dict | None:
    """Combined run + review row for the polling endpoint.

    For Stage 0 charter runs we additionally embed ``role_card_run``: the
    most-recent Stage 1 row for the same role (any status). This lets the UI
    poll a single endpoint and pick up the auto-fired role card without a
    second round-trip.
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, role_id, role_display, prompt_version,
                       model_snapshot, planner_output, planner_reasoning,
                       status, error_message, started_at, completed_at
                  FROM v2_run_log
                 WHERE run_id = %s
                """,
                (str(run_id),),
            )
            run = _row_to_dict(cur, cur.fetchone())
            if run is None:
                return None

            cur.execute(
                """
                SELECT id, item_type, payload, reason, created_at,
                       resolved, resolved_by, resolved_at
                  FROM v2_review_queue
                 WHERE run_id = %s
                 ORDER BY created_at DESC
                 LIMIT 1
                """,
                (str(run_id),),
            )
            review = _row_to_dict(cur, cur.fetchone())

            role_card_run = None
            dimensions_run = None
            reconciliation_run = None
            typing_run = None
            placement_run = None
            containment_run = None
            enrichment_run = None
            load_run = None
            if (run["prompt_version"] or "").startswith(CHARTER_PROMPT_VERSION_PREFIX):
                role_card_run = _fetch_latest_role_card_run(cur, run["role_id"])
                dimensions_run = _fetch_latest_dimensions_run(cur, run["role_id"])
                reconciliation_run = _fetch_latest_reconciliation_run(cur, run["role_id"])
                typing_run = _fetch_latest_typing_run(cur, run["role_id"])
                placement_run = _fetch_latest_placement_run(cur, run["role_id"])
                containment_run = _fetch_latest_containment_run(cur, run["role_id"])
                enrichment_run = _fetch_latest_enrichment_run(cur, run["role_id"])
                load_run = _fetch_latest_load_run(cur, run["role_id"])

    charter = run.get("planner_output") or {}
    if isinstance(charter, dict):
        charter = {k: v for k, v in charter.items() if not k.startswith("_")}

    review_payload = (review or {}).get("payload") or {}
    if isinstance(review_payload, dict):
        validator_log = review_payload.get("validator_log") or {"warnings": [], "errors": []}
    else:
        validator_log = {"warnings": [], "errors": []}

    return {
        "run_id": str(run["run_id"]),
        "role_slug": run["role_id"],
        "role_display": run["role_display"],
        "status": run["status"],
        "error_message": run["error_message"],
        "prompt_version": run["prompt_version"],
        "model_snapshot": run["model_snapshot"],
        "started_at": run["started_at"].isoformat() if run["started_at"] else None,
        "completed_at": run["completed_at"].isoformat() if run["completed_at"] else None,
        "charter": charter if charter else None,
        "validator_log": validator_log,
        "review_resolved": bool(review["resolved"]) if review else False,
        "reviewed_by": (review or {}).get("resolved_by"),
        "reviewed_at": (
            review["resolved_at"].isoformat()
            if review and review.get("resolved_at")
            else None
        ),
        "role_card_run": role_card_run,
        "dimensions_run": dimensions_run,
        "reconciliation_run": reconciliation_run,
        "typing_run": typing_run,
        "placement_run": placement_run,
        "containment_run": containment_run,
        "enrichment_run": enrichment_run,
        "load_run": load_run,
    }


def _fetch_latest_load_run(cur, role_slug: str) -> dict | None:
    """Stage 8 polling-endpoint embed. Auto-approves when load succeeds;
    flips to ``failed`` on persistence errors. No human gate by default
    (the autonomy goal). Reviews row exists only if a downstream
    consumer flags an issue post-load."""
    cur.execute(
        """
        SELECT run_id, role_id, prompt_version, model_snapshot,
               planner_output, status, error_message,
               started_at, completed_at
          FROM v2_run_log
         WHERE role_id = %s
           AND prompt_version LIKE %s
         ORDER BY started_at DESC NULLS LAST
         LIMIT 1
        """,
        (role_slug, STAGE8_PROMPT_VERSION_PREFIX + "%"),
    )
    s8 = _row_to_dict(cur, cur.fetchone())
    if s8 is None:
        return None
    body = s8.get("planner_output") or {}
    if not isinstance(body, dict):
        body = {}
    cleaned = {k: v for k, v in body.items() if not k.startswith("_")}
    validator_log = body.get("_validator_log") or {"warnings": [], "errors": []}
    return {
        "run_id": str(s8["run_id"]),
        "role_slug": s8["role_id"],
        "status": s8["status"],
        "error_message": s8["error_message"],
        "prompt_version": s8["prompt_version"],
        "model_snapshot": s8["model_snapshot"],
        "started_at": s8["started_at"].isoformat() if s8["started_at"] else None,
        "completed_at": s8["completed_at"].isoformat() if s8["completed_at"] else None,
        "load_summary": cleaned.get("load_summary") or {},
        "validator_log": validator_log,
        "review_resolved": False,
        "reviewed_by": None,
        "reviewed_at": None,
    }


def _fetch_latest_enrichment_run(cur, role_slug: str) -> dict | None:
    """Stage 7 polling-endpoint embed. Has a human-review path
    (ambiguity_flag=true OR maturity=deprecated) so we look up the
    review_queue row."""
    cur.execute(
        """
        SELECT run_id, role_id, prompt_version, model_snapshot,
               planner_output, status, error_message,
               started_at, completed_at
          FROM v2_run_log
         WHERE role_id = %s
           AND prompt_version LIKE %s
         ORDER BY started_at DESC NULLS LAST
         LIMIT 1
        """,
        (role_slug, STAGE7_PROMPT_VERSION_PREFIX + "%"),
    )
    s7 = _row_to_dict(cur, cur.fetchone())
    if s7 is None:
        return None
    cur.execute(
        """
        SELECT payload, resolved, resolved_by, resolved_at
          FROM v2_review_queue
         WHERE run_id = %s
           AND item_type = %s
         ORDER BY created_at DESC
         LIMIT 1
        """,
        (str(s7["run_id"]), ENRICHMENT_REVIEW_ITEM_TYPE),
    )
    review = _row_to_dict(cur, cur.fetchone())
    body = s7.get("planner_output") or {}
    if not isinstance(body, dict):
        body = {}
    cleaned = {k: v for k, v in body.items() if not k.startswith("_")}
    validator_log = body.get("_validator_log") or {"warnings": [], "errors": []}
    return {
        "run_id": str(s7["run_id"]),
        "role_slug": s7["role_id"],
        "status": s7["status"],
        "error_message": s7["error_message"],
        "prompt_version": s7["prompt_version"],
        "model_snapshot": s7["model_snapshot"],
        "started_at": s7["started_at"].isoformat() if s7["started_at"] else None,
        "completed_at": s7["completed_at"].isoformat() if s7["completed_at"] else None,
        "enrichments": cleaned.get("enrichments") or [],
        "validator_log": validator_log,
        "review_resolved": bool(review["resolved"]) if review else False,
        "reviewed_by": (review or {}).get("resolved_by"),
        "reviewed_at": (
            review["resolved_at"].isoformat()
            if review and review.get("resolved_at") else None
        ),
    }


def _fetch_latest_containment_run(cur, role_slug: str) -> dict | None:
    """Stage 6 polling-endpoint embed. Has a human-review path
    (cycles, type-rule violations) so we DO look up the review_queue row."""
    cur.execute(
        """
        SELECT run_id, role_id, prompt_version, model_snapshot,
               planner_output, status, error_message,
               started_at, completed_at
          FROM v2_run_log
         WHERE role_id = %s
           AND prompt_version LIKE %s
         ORDER BY started_at DESC NULLS LAST
         LIMIT 1
        """,
        (role_slug, STAGE6_PROMPT_VERSION_PREFIX + "%"),
    )
    s6 = _row_to_dict(cur, cur.fetchone())
    if s6 is None:
        return None
    cur.execute(
        """
        SELECT payload, resolved, resolved_by, resolved_at
          FROM v2_review_queue
         WHERE run_id = %s
           AND item_type = %s
         ORDER BY created_at DESC
         LIMIT 1
        """,
        (str(s6["run_id"]), CONTAINMENT_REVIEW_ITEM_TYPE),
    )
    review = _row_to_dict(cur, cur.fetchone())
    body = s6.get("planner_output") or {}
    if not isinstance(body, dict):
        body = {}
    cleaned = {k: v for k, v in body.items() if not k.startswith("_")}
    validator_log = body.get("_validator_log") or {"warnings": [], "errors": []}
    return {
        "run_id": str(s6["run_id"]),
        "role_slug": s6["role_id"],
        "status": s6["status"],
        "error_message": s6["error_message"],
        "prompt_version": s6["prompt_version"],
        "model_snapshot": s6["model_snapshot"],
        "started_at": s6["started_at"].isoformat() if s6["started_at"] else None,
        "completed_at": s6["completed_at"].isoformat() if s6["completed_at"] else None,
        "relationships": cleaned.get("relationships") or [],
        "validator_log": validator_log,
        "review_resolved": bool(review["resolved"]) if review else False,
        "reviewed_by": (review or {}).get("resolved_by"),
        "reviewed_at": (
            review["resolved_at"].isoformat()
            if review and review.get("resolved_at") else None
        ),
    }


def _fetch_latest_placement_run(cur, role_slug: str) -> dict | None:
    """Stage 5 polling-endpoint embed. Headless lifecycle: only returns
    the v2_run_log row + body — no review_queue lookup needed."""
    cur.execute(
        """
        SELECT run_id, role_id, prompt_version, model_snapshot,
               planner_output, status, error_message,
               started_at, completed_at
          FROM v2_run_log
         WHERE role_id = %s
           AND prompt_version LIKE %s
         ORDER BY started_at DESC NULLS LAST
         LIMIT 1
        """,
        (role_slug, STAGE5_PROMPT_VERSION_PREFIX + "%"),
    )
    s5 = _row_to_dict(cur, cur.fetchone())
    if s5 is None:
        return None
    body = s5.get("planner_output") or {}
    if not isinstance(body, dict):
        body = {}
    cleaned = {k: v for k, v in body.items() if not k.startswith("_")}
    validator_log = body.get("_validator_log") or {"warnings": [], "errors": []}
    return {
        "run_id": str(s5["run_id"]),
        "role_slug": s5["role_id"],
        "status": s5["status"],
        "error_message": s5["error_message"],
        "prompt_version": s5["prompt_version"],
        "model_snapshot": s5["model_snapshot"],
        "started_at": s5["started_at"].isoformat() if s5["started_at"] else None,
        "completed_at": s5["completed_at"].isoformat() if s5["completed_at"] else None,
        "placed_skills": cleaned.get("placed_skills") or [],
        "routing": cleaned.get("routing") or {},
        "validator_log": validator_log,
    }


def _fetch_latest_typing_run(cur, role_slug: str) -> dict | None:
    """Fetch the latest Stage 4 typing run for ``role_slug`` for the
    polling endpoint. Stage 4 is headless (no review_queue), so this
    pulls only the v2_run_log row and the typed_skills body.

    Returns None when no Stage 4 row exists yet for the role.
    """
    cur.execute(
        """
        SELECT run_id, role_id, prompt_version, model_snapshot,
               planner_output, status, error_message,
               started_at, completed_at
          FROM v2_run_log
         WHERE role_id = %s
           AND prompt_version LIKE %s
         ORDER BY started_at DESC NULLS LAST
         LIMIT 1
        """,
        (role_slug, STAGE4_PROMPT_VERSION_PREFIX + "%"),
    )
    s4 = _row_to_dict(cur, cur.fetchone())
    if s4 is None:
        return None
    body = s4.get("planner_output") or {}
    if not isinstance(body, dict):
        body = {}
    cleaned = {k: v for k, v in body.items() if not k.startswith("_")}
    validator_log = body.get("_validator_log") or {"warnings": [], "errors": []}
    return {
        "run_id": str(s4["run_id"]),
        "role_slug": s4["role_id"],
        "status": s4["status"],
        "error_message": s4["error_message"],
        "prompt_version": s4["prompt_version"],
        "model_snapshot": s4["model_snapshot"],
        "started_at": s4["started_at"].isoformat() if s4["started_at"] else None,
        "completed_at": s4["completed_at"].isoformat() if s4["completed_at"] else None,
        "typed_skills": cleaned.get("typed_skills") or [],
        "routing": cleaned.get("routing") or {},
        "validator_log": validator_log,
    }


def _fetch_latest_reconciliation_run(cur, role_slug: str) -> dict | None:
    """Fetch the latest Stage 3 run for ``role_slug`` for the polling
    endpoint. Pulls the v2_run_log row + the open review_queue row when
    one exists. Returns None when no Stage 3 row exists yet."""
    cur.execute(
        """
        SELECT run_id, role_id, prompt_version, model_snapshot,
               planner_output, status, error_message,
               started_at, completed_at
          FROM v2_run_log
         WHERE role_id = %s
           AND prompt_version LIKE %s
         ORDER BY started_at DESC NULLS LAST
         LIMIT 1
        """,
        (role_slug, STAGE3_PROMPT_VERSION_PREFIX + "%"),
    )
    s3 = _row_to_dict(cur, cur.fetchone())
    if s3 is None:
        return None

    cur.execute(
        """
        SELECT payload, resolved, resolved_by, resolved_at
          FROM v2_review_queue
         WHERE run_id = %s
           AND item_type = %s
         ORDER BY created_at DESC
         LIMIT 1
        """,
        (str(s3["run_id"]), RECONCILIATION_REVIEW_ITEM_TYPE),
    )
    review = _row_to_dict(cur, cur.fetchone())

    body = s3.get("planner_output") or {}
    if not isinstance(body, dict):
        body = {}
    cleaned = {k: v for k, v in body.items() if not k.startswith("_")}
    validator_log = body.get("_validator_log") or {"warnings": [], "errors": []}

    return {
        "run_id": str(s3["run_id"]),
        "role_slug": s3["role_id"],
        "status": s3["status"],
        "error_message": s3["error_message"],
        "prompt_version": s3["prompt_version"],
        "model_snapshot": s3["model_snapshot"],
        "started_at": s3["started_at"].isoformat() if s3["started_at"] else None,
        "completed_at": s3["completed_at"].isoformat() if s3["completed_at"] else None,
        # Surface the meaningful reconciliation outputs (everything except
        # internal _-prefixed metadata).
        "flagged_pairs": cleaned.get("flagged_pairs") or [],
        "decisions": cleaned.get("decisions") or [],
        "locked_dimensions": cleaned.get("locked_dimensions") or [],
        "merge_log": cleaned.get("merge_log") or [],
        "split_log": cleaned.get("split_log") or [],
        "unassigned_exemplars": cleaned.get("unassigned_exemplars") or [],
        "validator_log": validator_log,
        "review_resolved": bool(review["resolved"]) if review else False,
        "reviewed_by": (review or {}).get("resolved_by"),
        "reviewed_at": (
            review["resolved_at"].isoformat()
            if review and review.get("resolved_at")
            else None
        ),
    }


def _fetch_latest_dimensions_run(cur, role_slug: str) -> dict | None:
    """Helper: fetch the latest Stage 2 run for ``role_slug``, packaged for
    the polling endpoint. No review_queue lookup — Stage 2 is headless.
    Returns None when no Stage 2 row exists yet (role hasn't been processed)
    or the latest row is in pending/running before completion."""
    cur.execute(
        """
        SELECT run_id, role_id, prompt_version, model_snapshot,
               planner_output, status, error_message,
               started_at, completed_at
          FROM v2_run_log
         WHERE role_id = %s
           AND prompt_version LIKE %s
         ORDER BY started_at DESC NULLS LAST
         LIMIT 1
        """,
        (role_slug, STAGE2_PROMPT_VERSION_PREFIX + "%"),
    )
    s2 = _row_to_dict(cur, cur.fetchone())
    if s2 is None:
        return None
    body = s2.get("planner_output") or {}
    if not isinstance(body, dict):
        body = {}
    dims_raw = body.get("candidate_dimensions") or []
    cleaned_dims: list[dict] = []
    for d in dims_raw:
        if isinstance(d, dict):
            cleaned_dims.append(
                {k: v for k, v in d.items() if not k.startswith("_")}
            )
    validator_log = body.get("_validator_log") or {"warnings": [], "errors": []}
    return {
        "run_id": str(s2["run_id"]),
        "role_slug": s2["role_id"],
        "status": s2["status"],
        "error_message": s2["error_message"],
        "prompt_version": s2["prompt_version"],
        "model_snapshot": s2["model_snapshot"],
        "started_at": s2["started_at"].isoformat() if s2["started_at"] else None,
        "completed_at": s2["completed_at"].isoformat() if s2["completed_at"] else None,
        "candidate_dimensions": cleaned_dims,
        "validator_log": validator_log,
    }


def _fetch_latest_role_card_run(cur, role_slug: str) -> dict | None:
    """Helper: fetch the latest Stage 1 run + its review row for ``role_slug``,
    packaged in the same shape the polling endpoint emits for the charter run.

    Called inside ``get_run_with_review`` so it shares the open cursor.
    Returns ``None`` when no Stage 1 row exists yet (i.e. role still
    pre-Stage-1) or when the role itself is unknown.
    """
    cur.execute(
        """
        SELECT run_id, role_id, role_display, prompt_version, model_snapshot,
               planner_output, planner_reasoning, status, error_message,
               started_at, completed_at
          FROM v2_run_log
         WHERE role_id = %s
           AND prompt_version LIKE %s
         ORDER BY started_at DESC NULLS LAST
         LIMIT 1
        """,
        (role_slug, STAGE1_PROMPT_VERSION_PREFIX + "%"),
    )
    s1 = _row_to_dict(cur, cur.fetchone())
    if s1 is None:
        return None

    cur.execute(
        """
        SELECT payload, resolved, resolved_by, resolved_at
          FROM v2_review_queue
         WHERE run_id = %s
           AND item_type = %s
         ORDER BY created_at DESC
         LIMIT 1
        """,
        (str(s1["run_id"]), ROLE_CARD_REVIEW_ITEM_TYPE),
    )
    s1_review = _row_to_dict(cur, cur.fetchone())

    role_card = s1.get("planner_output") or {}
    if isinstance(role_card, dict):
        role_card = {k: v for k, v in role_card.items() if not k.startswith("_")}

    payload = (s1_review or {}).get("payload") or {}
    validator_log = (
        payload.get("validator_log") if isinstance(payload, dict) else None
    ) or {"warnings": [], "errors": []}

    return {
        "run_id": str(s1["run_id"]),
        "role_slug": s1["role_id"],
        "status": s1["status"],
        "error_message": s1["error_message"],
        "prompt_version": s1["prompt_version"],
        "model_snapshot": s1["model_snapshot"],
        "started_at": s1["started_at"].isoformat() if s1["started_at"] else None,
        "completed_at": s1["completed_at"].isoformat() if s1["completed_at"] else None,
        "role_card": role_card if role_card else None,
        "validator_log": validator_log,
        "review_resolved": bool(s1_review["resolved"]) if s1_review else False,
        "reviewed_by": (s1_review or {}).get("resolved_by"),
        "reviewed_at": (
            s1_review["resolved_at"].isoformat()
            if s1_review and s1_review.get("resolved_at")
            else None
        ),
    }


def list_recent_runs(limit: int = 20) -> list[dict]:
    """Last N Stage 0 runs for the UI's recent-runs lane.

    Each row also carries ``stage8_complete`` — True when the same role has
    an approved Stage 8 (catalog-load) run, signalling the full pipeline
    landed in the canonical catalog. The UI shows this as a small dot next
    to the role name.
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT v.run_id, v.role_id, v.role_display, v.status,
                       v.started_at, v.completed_at,
                       EXISTS (
                           SELECT 1 FROM v2_run_log s8
                            WHERE s8.role_id = v.role_id
                              AND s8.prompt_version LIKE %s
                              AND s8.status = 'approved'
                       ) AS stage8_complete
                  FROM v2_run_log v
                 WHERE v.prompt_version LIKE %s
                 ORDER BY v.started_at DESC NULLS LAST
                 LIMIT %s
                """,
                (
                    STAGE8_PROMPT_VERSION_PREFIX + "%",
                    CHARTER_PROMPT_VERSION_PREFIX + "%",
                    int(limit),
                ),
            )
            rows = _rows_to_dicts(cur, cur.fetchall())
    for r in rows:
        r["run_id"] = str(r["run_id"])
        if r.get("started_at"):
            r["started_at"] = r["started_at"].isoformat()
        if r.get("completed_at"):
            r["completed_at"] = r["completed_at"].isoformat()
        r["stage8_complete"] = bool(r.get("stage8_complete"))
    return rows


# ── review actions (approve / regenerate) ───────────────────────────────────


def approve_run(
    run_id: uuid.UUID,
    *,
    reviewed_by: str | None,
    notes: str | None,
) -> dict:
    """Mark a run approved. Polymorphic on prompt_version: Stage 0 charter
    approvals also mirror the charter definition into ``roles.role_archetype``
    so existing canonical-schema readers see something useful; Stage 1 role-
    card approvals don't touch the role row.
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role_id, planner_output, status, prompt_version
                  FROM v2_run_log
                 WHERE run_id = %s
                """,
                (str(run_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"run {run_id} not found")
            role_slug, planner_output, status, prompt_version = row
            if status != "awaiting_review":
                raise ValueError(
                    f"run {run_id} is in status={status!r}; only awaiting_review can be approved"
                )

            item_type = _item_type_for_prompt_version(prompt_version)

            cur.execute(
                "UPDATE v2_run_log SET status = 'approved' WHERE run_id = %s",
                (str(run_id),),
            )

            # NOTE: jsonb_set(payload, path, to_jsonb(NULL::text)) returns SQL
            # NULL — and ``payload`` is NOT NULL. So we only rewrite the
            # ``notes`` key when caller actually supplied notes; otherwise we
            # leave payload untouched.
            cur.execute(
                """
                UPDATE v2_review_queue
                   SET resolved    = TRUE,
                       resolved_by = %s,
                       resolved_at = NOW(),
                       payload     = CASE
                                        WHEN %s::text IS NULL THEN payload
                                        ELSE jsonb_set(
                                                COALESCE(payload, '{}'::jsonb),
                                                '{notes}',
                                                to_jsonb(%s::text),
                                                true
                                             )
                                     END
                 WHERE run_id = %s
                   AND item_type = %s
                """,
                (reviewed_by, notes, notes, str(run_id), item_type),
            )

            # role_archetype mirror is charter-only — Stage 1 role cards
            # don't have a single ``definition`` field that maps cleanly.
            if (prompt_version or "").startswith(CHARTER_PROMPT_VERSION_PREFIX):
                charter = planner_output if isinstance(planner_output, dict) else {}
                definition = charter.get("definition") if isinstance(charter, dict) else None
                if definition:
                    cur.execute(
                        """
                        UPDATE roles
                           SET role_archetype = %s,
                               updated_at     = NOW()
                         WHERE slug = %s
                        """,
                        (definition, role_slug),
                    )
        conn.commit()
    return {
        "run_id": str(run_id),
        "role_slug": role_slug,
        "status": "approved",
        "prompt_version": prompt_version,
    }


def reject_and_create_regenerate(
    prior_run_id: uuid.UUID,
    *,
    reviewed_by: str | None,
    notes: str | None,
) -> dict:
    """Atomically: mark prior run rejected, resolve its review row, insert a
    new pending run linked via parent_run_id. The runner is invoked separately
    by the caller (FastAPI handler does ``asyncio.create_task``).

    Accepts EITHER ``awaiting_review`` (standard regenerate) or ``approved``
    (supersede an already-approved run). Supersede is what you want when the
    catalog has changed since the original approval — e.g. a new adjacent
    role was added and the existing charter no longer reflects the boundary.
    Already-rejected / failed / running runs are still refused.

    Returns ``{"run_id": <new>, "prompt_version": <stage>}`` so the caller
    can dispatch to the right stage runner without an extra round-trip.
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role_id, role_display, prompt_version, model_snapshot, status
                  FROM v2_run_log
                 WHERE run_id = %s
                """,
                (str(prior_run_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"run {prior_run_id} not found")
            role_slug, role_display, prompt_version, model_snapshot, status = row
            # Stage 2 is headless: runs land in 'approved' or 'failed', never
            # 'awaiting_review'. We allow 'failed' here so the UI's regenerate
            # button can recover from validator failures.
            if status not in {"awaiting_review", "approved", "failed"}:
                raise ValueError(
                    f"run {prior_run_id} is in status={status!r}; "
                    "only awaiting_review / approved / failed runs can be regenerated"
                )

            item_type = _item_type_for_prompt_version(prompt_version)

            cur.execute(
                "UPDATE v2_run_log SET status = 'rejected' WHERE run_id = %s",
                (str(prior_run_id),),
            )

            # Stage 2 has no review_queue rows (headless). Skip ALL queue
            # work when item_type is None — there's nothing to update or
            # supersede. Stage 0/1 retain the original queue handling.
            if item_type is None:
                pass  # headless stage — nothing to do for the queue
            elif status == "awaiting_review":
                # Standard regenerate path: turn the open review-queue row
                # into resolved with the regenerate reviewer's notes.
                # See approve_run for why the CASE is necessary — jsonb_set
                # with to_jsonb(NULL::text) returns SQL NULL and would zero
                # out payload.
                cur.execute(
                    """
                    UPDATE v2_review_queue
                       SET resolved    = TRUE,
                           resolved_by = %s,
                           resolved_at = NOW(),
                           payload     = CASE
                                            WHEN %s::text IS NULL THEN payload
                                            ELSE jsonb_set(
                                                    COALESCE(payload, '{}'::jsonb),
                                                    '{notes}',
                                                    to_jsonb(%s::text),
                                                    true
                                                 )
                                         END
                     WHERE run_id = %s
                       AND item_type = %s
                    """,
                    (reviewed_by, notes, notes, str(prior_run_id), item_type),
                )
            else:
                # Supersede path: prior_run was approved (or, for failed
                # runs, never reached the review queue at all). Don't
                # overwrite the original review-queue row when one exists
                # — that would erase the approval audit info. Instead,
                # INSERT a fresh "superseded" audit row so we have both:
                # the original approval AND the supersede event.
                supersede_payload = {
                    "supersede_of_run_id": str(prior_run_id),
                    "supersede_of_status": status,
                    "notes": notes,
                }
                supersede_reason = (
                    "superseded after validator failure"
                    if status == "failed"
                    else "superseded by regenerate"
                )
                cur.execute(
                    """
                    INSERT INTO v2_review_queue
                        (run_id, item_type, payload, reason,
                         resolved, resolved_by, resolved_at)
                    VALUES (%s, %s, %s::jsonb, %s,
                            TRUE, %s, NOW())
                    """,
                    (
                        str(prior_run_id),
                        item_type,
                        json.dumps(supersede_payload),
                        supersede_reason,
                        reviewed_by or "unknown",
                    ),
                )

            new_id = uuid.uuid4()
            cur.execute(
                """
                INSERT INTO v2_run_log
                    (run_id, role_id, role_display, prompt_version,
                     model_snapshot, planner_output, status)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, 'pending')
                """,
                (
                    str(new_id),
                    role_slug,
                    role_display,
                    prompt_version,
                    model_snapshot,
                    json.dumps({"_parent_run_id": str(prior_run_id)}),
                ),
            )
        conn.commit()
    return {"run_id": new_id, "prompt_version": prompt_version}


def recover_orphan_runs(*, stale_after_seconds: int = 600) -> int:
    """On uvicorn startup, sweep any v3 run still 'running' from a prior
    crashed process to 'failed'. Bounded by prompt_version prefix so this never
    touches non-v3 state. Covers every stage we've shipped (Stages 0-2 today).
    Returns the number of runs reaped.

    Note: the LIKE pattern goes through ``%s`` parameters (NOT inlined) so
    psycopg2 doesn't try to interpret the ``%`` in the pattern as another
    placeholder marker.
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE v2_run_log
                   SET status        = 'failed',
                       error_message = 'orphaned by uvicorn restart',
                       completed_at  = NOW()
                 WHERE status = 'running'
                   AND (   prompt_version LIKE %s
                        OR prompt_version LIKE %s
                        OR prompt_version LIKE %s
                        OR prompt_version LIKE %s
                        OR prompt_version LIKE %s
                        OR prompt_version LIKE %s
                        OR prompt_version LIKE %s
                        OR prompt_version LIKE %s
                        OR prompt_version LIKE %s)
                   AND started_at < NOW() - (%s || ' seconds')::interval
                """,
                (
                    CHARTER_PROMPT_VERSION_PREFIX + "%",
                    STAGE1_PROMPT_VERSION_PREFIX + "%",
                    STAGE2_PROMPT_VERSION_PREFIX + "%",
                    STAGE3_PROMPT_VERSION_PREFIX + "%",
                    STAGE4_PROMPT_VERSION_PREFIX + "%",
                    STAGE5_PROMPT_VERSION_PREFIX + "%",
                    STAGE6_PROMPT_VERSION_PREFIX + "%",
                    STAGE7_PROMPT_VERSION_PREFIX + "%",
                    STAGE8_PROMPT_VERSION_PREFIX + "%",
                    str(int(stale_after_seconds)),
                ),
            )
            reaped = cur.rowcount or 0
        conn.commit()
    if reaped:
        logger.warning("[v3] reaped %d orphan v3 run(s)", reaped)
    return reaped


# ── JD samples ──────────────────────────────────────────────────────────────


def get_jd_samples_for_role(role_slug: str, *, limit: int = 5) -> list[dict]:
    """Top-N JD samples whose ingest-time fuzzy match landed on this role."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT title, body, match_score
                  FROM jd_samples
                 WHERE role_slug = %s
                 ORDER BY match_score DESC NULLS LAST
                 LIMIT %s
                """,
                (role_slug, int(limit)),
            )
            return _rows_to_dicts(cur, cur.fetchall())


def insert_jd_samples(rows: Iterable[dict]) -> int:
    """Bulk insert JD rows. ``rows`` items: dict with keys
    ``role_slug, source_dataset, source_row_id, title, body, match_score``."""
    payload = [
        (
            r.get("role_slug"),
            r["source_dataset"],
            r.get("source_row_id"),
            r["title"],
            r["body"],
            r.get("match_score"),
        )
        for r in rows
        if r.get("title") and r.get("body")
    ]
    if not payload:
        return 0
    with connect() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO jd_samples
                    (role_slug, source_dataset, source_row_id, title, body, match_score)
                VALUES %s
                """,
                payload,
            )
            inserted = cur.rowcount or 0
        conn.commit()
    return inserted


# ── alias lookup (skill-name validator) ─────────────────────────────────────


def get_alias_lookup_set() -> set[str]:
    """Lowercased set of every concrete-nature skill alias + display name.

    Used by validators.no_skills_in_scope to detect when the LLM smuggled a
    real tool/library/platform name into a charter scope statement.

    The query joins ``skill_aliases`` and ``canonical_skills`` and excludes
    rows whose ``skill_nature`` is ``CONCEPT``, ``METHODOLOGY``, ``PATTERN``,
    or ``PRACTICE``. Those four natures describe responsibility-shaped
    skills — words like "automation", "disaster recovery", "alerting",
    "naming conventions" — that the LLM legitimately uses when describing
    what a role does. Treating them as skill leaks produced 14/23 false
    positives in the first prod cohort. The validator's local stop-list
    (``_GENERIC_NOUN_STOPLIST``) handles the residual concrete-nature
    English-collision cases that this filter doesn't catch.

    Falls back to an empty set when the DB is unavailable or the v3 schema
    has been bootstrapped but no skills have been loaded yet (a brand-new
    ``skill-library`` DB has zero rows in skill_aliases / canonical_skills).
    """
    try:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT lower(sa.alias_text)
                      FROM skill_aliases sa
                      JOIN canonical_skills cs ON cs.id = sa.skill_id
                     WHERE cs.skill_nature::text NOT IN
                           ('CONCEPT', 'METHODOLOGY', 'PATTERN', 'PRACTICE')
                    UNION
                    SELECT lower(cs.display_name)
                      FROM canonical_skills cs
                     WHERE cs.skill_nature::text NOT IN
                           ('CONCEPT', 'METHODOLOGY', 'PATTERN', 'PRACTICE')
                    """
                )
                return {row[0] for row in cur.fetchall() if row[0]}
    except psycopg2.Error as exc:
        logger.warning("[v3] alias lookup unavailable, validator running with empty set: %s", exc)
        return set()


# ── Stage 1 read-side helpers ──────────────────────────────────────────────


def get_latest_role_card_run_for_role(role_slug: str) -> dict | None:
    """Public wrapper around :func:`_fetch_latest_role_card_run` for callers
    outside ``get_run_with_review`` (e.g. backfill scripts, future Stage 2).
    Returns the latest Stage 1 row for ``role_slug`` regardless of status, or
    ``None`` when none exists yet."""
    with connect() as conn:
        with conn.cursor() as cur:
            return _fetch_latest_role_card_run(cur, role_slug)


def get_latest_approved_charter_for_role(role_slug: str) -> dict | None:
    """Latest *approved* charter body for ``role_slug``, with internal
    underscore-prefixed metadata stripped. Used by Stage 1 regenerate to
    refeed the LLM with the current truth even when the charter was
    re-approved between the rejected Stage 1 attempt and the regenerate."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT planner_output
                  FROM v2_run_log
                 WHERE role_id = %s
                   AND prompt_version LIKE %s
                   AND status = 'approved'
                 ORDER BY completed_at DESC
                 LIMIT 1
                """,
                (role_slug, CHARTER_PROMPT_VERSION_PREFIX + "%"),
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                return None
            data = row[0]
    if not isinstance(data, dict):
        return None
    return {k: v for k, v in data.items() if not k.startswith("_")}


def get_role_card_for_role(role_slug: str) -> dict | None:
    """Latest *approved* role card body (planner_output stripped of internal
    underscore-prefixed metadata) for ``role_slug``. This is the canonical
    Stage 1 artifact that downstream stages (Stage 2 onward) will read."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT planner_output
                  FROM v2_run_log
                 WHERE role_id = %s
                   AND prompt_version LIKE %s
                   AND status = 'approved'
                 ORDER BY completed_at DESC
                 LIMIT 1
                """,
                (role_slug, STAGE1_PROMPT_VERSION_PREFIX + "%"),
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                return None
            data = row[0]
    if not isinstance(data, dict):
        return None
    return {k: v for k, v in data.items() if not k.startswith("_")}


def find_unanchored_approved_charters() -> list[dict]:
    """Return the latest approved Stage 0 charter run for every role that
    does NOT yet have an approved Stage 1 row.

    Each item: ``{"role_slug", "role_display", "charter_run_id"}``. Used by
    the backfill script to fill role cards for roles approved before Stage 1
    shipped, and to recover from runs that failed mid-Stage-1.
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT latest_charter.role_slug,
                       latest_charter.role_display,
                       latest_charter.charter_run_id
                  FROM (
                        SELECT DISTINCT ON (role_id)
                               role_id    AS role_slug,
                               role_display,
                               run_id     AS charter_run_id
                          FROM v2_run_log
                         WHERE prompt_version LIKE %s
                           AND status = 'approved'
                         ORDER BY role_id, completed_at DESC
                       ) latest_charter
                 WHERE NOT EXISTS (
                        SELECT 1 FROM v2_run_log s1
                         WHERE s1.role_id = latest_charter.role_slug
                           AND s1.prompt_version LIKE %s
                           AND s1.status = 'approved'
                       )
                 ORDER BY latest_charter.role_slug
                """,
                (
                    CHARTER_PROMPT_VERSION_PREFIX + "%",
                    STAGE1_PROMPT_VERSION_PREFIX + "%",
                ),
            )
            rows = cur.fetchall()
    return [
        {
            "role_slug": r[0],
            "role_display": r[1],
            "charter_run_id": str(r[2]),
        }
        for r in rows
    ]


def find_stale_role_cards(target_prompt_version: str) -> list[dict]:
    """Return roles whose latest approved role card was produced by an
    older prompt version (anything other than ``target_prompt_version``).
    Used by the role-card backfill's ``--include-stale`` mode.

    Each item: ``{"role_slug", "role_display", "role_card_run_id",
    "current_prompt_version"}``."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (role_id)
                       role_id        AS role_slug,
                       role_display,
                       run_id         AS role_card_run_id,
                       prompt_version AS current_prompt_version
                  FROM v2_run_log
                 WHERE prompt_version LIKE %s
                   AND status = 'approved'
                 ORDER BY role_id, completed_at DESC
                """,
                (STAGE1_PROMPT_VERSION_PREFIX + "%",),
            )
            rows = cur.fetchall()
    return [
        {
            "role_slug": r[0],
            "role_display": r[1],
            "role_card_run_id": str(r[2]),
            "current_prompt_version": r[3],
        }
        for r in rows
        if r[3] != target_prompt_version
    ]


# ── Stage 2 helpers ────────────────────────────────────────────────────────


def complete_dimensions_run(
    run_id: uuid.UUID,
    *,
    dimensions: list[dict],
    validator_log: dict,
) -> str:
    """Stage 2 finished. Headless lifecycle:
      validators pass  -> status='approved', no review_queue insert
      validators fail  -> status='failed', error_message summary

    Returns the resulting status string. ``planner_output`` always carries
    the dimension list under ``candidate_dimensions`` plus the validator
    log under ``_validator_log`` for audit.
    """
    has_errors = bool(validator_log.get("errors"))
    final_status = "failed" if has_errors else "approved"

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT planner_output FROM v2_run_log WHERE run_id = %s",
                (str(run_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"v2_run_log row {run_id} not found")
            prior = row[0] or {}
            parent_run_id = (
                prior.get("_parent_run_id") if isinstance(prior, dict) else None
            )

            output = {
                "candidate_dimensions": dimensions,
                "_parent_run_id": parent_run_id,
                "_validator_log": validator_log,
            }

            if has_errors:
                # Concise summary of first 3 errors for quick triage.
                errs = validator_log.get("errors", [])[:3]
                err_msg = "; ".join(
                    f"[{e.get('code', '?')}] {e.get('message', '?')}"
                    for e in errs
                )
                if len(validator_log.get("errors", [])) > 3:
                    err_msg += f"; (+{len(validator_log['errors']) - 3} more)"
            else:
                err_msg = None

            cur.execute(
                """
                UPDATE v2_run_log
                   SET planner_output = %s::jsonb,
                       status         = %s,
                       completed_at   = NOW(),
                       error_message  = %s
                 WHERE run_id = %s
                """,
                (json.dumps(output), final_status, err_msg, str(run_id)),
            )
            # Per the headless design: NO v2_review_queue insert.
        conn.commit()
    return final_status


def get_latest_dimensions_for_role(role_slug: str) -> list[dict] | None:
    """Latest *approved* Stage 2 candidate_dimensions for the role, with
    internal underscore-prefixed metadata stripped. Returns the inner
    list (not the wrapper dict). None when no approved Stage 2 row exists
    yet."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT planner_output
                  FROM v2_run_log
                 WHERE role_id = %s
                   AND prompt_version LIKE %s
                   AND status = 'approved'
                 ORDER BY completed_at DESC
                 LIMIT 1
                """,
                (role_slug, STAGE2_PROMPT_VERSION_PREFIX + "%"),
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                return None
            data = row[0]
    if not isinstance(data, dict):
        return None
    dims = data.get("candidate_dimensions")
    if not isinstance(dims, list):
        return None
    # Strip any internal keys per-dim that may have leaked through.
    cleaned: list[dict] = []
    for d in dims:
        if isinstance(d, dict):
            cleaned.append({k: v for k, v in d.items() if not k.startswith("_")})
    return cleaned


def get_adjacent_role_dimensions(
    adjacent_role_names: list[str],
) -> dict[str, list[dict]]:
    """Map of {role_display_name: [candidate_dimensions]} for each adjacent
    role that has an approved Stage 2 row. Roles without Stage 2 yet are
    silently omitted. Used by Stage 2 prompt to give the LLM overlap
    context."""
    if not adjacent_role_names:
        return {}
    # Match by display_name (case-insensitive), since adjacent_roles in the
    # role card is a list of role display names, not slugs.
    placeholders = ",".join(["%s"] * len(adjacent_role_names))
    name_lower_list = [n.strip().lower() for n in adjacent_role_names]
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT ON (r.id)
                       r.display_name,
                       v.planner_output
                  FROM roles r
                  JOIN v2_run_log v ON v.role_id = r.slug
                 WHERE lower(r.display_name) IN ({placeholders})
                   AND v.prompt_version LIKE %s
                   AND v.status = 'approved'
                 ORDER BY r.id, v.completed_at DESC
                """,
                (*name_lower_list, STAGE2_PROMPT_VERSION_PREFIX + "%"),
            )
            rows = cur.fetchall()
    out: dict[str, list[dict]] = {}
    for display_name, planner_output in rows:
        if not isinstance(planner_output, dict):
            continue
        dims = planner_output.get("candidate_dimensions")
        if not isinstance(dims, list):
            continue
        cleaned = [
            {k: v for k, v in d.items() if not k.startswith("_")}
            for d in dims
            if isinstance(d, dict)
        ]
        if cleaned:
            out[display_name] = cleaned
    return out


def complete_recon_run(
    run_id: uuid.UUID,
    *,
    payload: dict,
    validator_log: dict,
    final_status: str,
) -> str:
    """Stage 3 finished. Polymorphic on ``final_status``:
      * "approved"        -> v2_run_log.status='approved', no queue row
      * "awaiting_review" -> v2_run_log.status='awaiting_review',
                              INSERT a v2_review_queue row so the UI
                              surfaces the merge/split decisions for
                              human gating
      * "failed"          -> v2_run_log.status='failed', error_message
                              summary, no queue row

    ``payload`` is the full reconciliation body (flagged_pairs,
    decisions, locked_dimensions, merge_log, split_log,
    unassigned_exemplars). ``_validator_log`` and ``_parent_run_id``
    are stitched in for audit and stripped on read by
    :func:`_fetch_latest_reconciliation_run`.
    """
    if final_status not in {"approved", "awaiting_review", "failed"}:
        raise ValueError(f"unexpected final_status={final_status!r}")

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role_id, planner_output FROM v2_run_log WHERE run_id = %s",
                (str(run_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"v2_run_log row {run_id} not found")
            role_slug = row[0]
            prior = row[1] or {}
            parent_run_id = (
                prior.get("_parent_run_id") if isinstance(prior, dict) else None
            )

            output = dict(payload)
            output["_parent_run_id"] = parent_run_id
            output["_validator_log"] = validator_log

            err_msg: str | None = None
            if final_status == "failed":
                errs = validator_log.get("errors", [])[:3]
                err_msg = "; ".join(
                    f"[{e.get('code', '?')}] {e.get('message', '?')}" for e in errs
                ) or "Stage 3 reconciliation failed"
                if len(validator_log.get("errors", [])) > 3:
                    err_msg += f"; (+{len(validator_log['errors']) - 3} more)"

            cur.execute(
                """
                UPDATE v2_run_log
                   SET planner_output = %s::jsonb,
                       status         = %s,
                       completed_at   = NOW(),
                       error_message  = %s
                 WHERE run_id = %s
                """,
                (json.dumps(output), final_status, err_msg, str(run_id)),
            )

            if final_status == "awaiting_review":
                merge_count = len(payload.get("merge_log") or [])
                split_count = len(payload.get("split_log") or [])
                reason = (
                    f"Reconciliation awaiting admin approval — "
                    f"{merge_count} MERGE, {split_count} SPLIT, "
                    f"{len(validator_log.get('warnings', []))} warning(s)"
                )
                review_payload = {
                    "role_slug": role_slug,
                    "merge_log": payload.get("merge_log") or [],
                    "split_log": payload.get("split_log") or [],
                    "validator_log": validator_log,
                    "parent_run_id": parent_run_id,
                }
                cur.execute(
                    """
                    INSERT INTO v2_review_queue
                        (run_id, item_type, payload, reason)
                    VALUES (%s, %s, %s::jsonb, %s)
                    """,
                    (
                        str(run_id),
                        RECONCILIATION_REVIEW_ITEM_TYPE,
                        json.dumps(review_payload),
                        reason,
                    ),
                )
        conn.commit()
    return final_status


def get_locked_dimensions_for_role(role_slug: str) -> list[dict] | None:
    """Latest *approved* Stage 3 locked_dimensions for the role, with
    internal _-prefixed metadata stripped. This is what later stages
    (Stage 4 onward) will read instead of the raw Stage 2 candidates.
    Returns ``None`` when the role has no approved Stage 3 row yet."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT planner_output
                  FROM v2_run_log
                 WHERE role_id = %s
                   AND prompt_version LIKE %s
                   AND status = 'approved'
                 ORDER BY completed_at DESC
                 LIMIT 1
                """,
                (role_slug, STAGE3_PROMPT_VERSION_PREFIX + "%"),
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                return None
            data = row[0]
    if not isinstance(data, dict):
        return None
    locked = data.get("locked_dimensions")
    if not isinstance(locked, list):
        return None
    return [
        {k: v for k, v in d.items() if isinstance(d, dict) and not k.startswith("_")}
        for d in locked
        if isinstance(d, dict)
    ]


def complete_typing_run(
    run_id: uuid.UUID,
    *,
    typed_skills: list[dict],
    routing: dict,
    validator_log: dict,
    final_status: str,
) -> str:
    """Stage 4 (typing) finished. Headless lifecycle:
      ``approved`` -> v2_run_log.status='approved', no review_queue row
      ``failed``   -> v2_run_log.status='failed', error_message summary

    ``routing`` is the bucketed result from ``route_by_confidence``
    (auto_approved / sample_review / full_review skill_ids). Stored as
    a small dict in planner_output so the UI can render bucket sizes
    without re-running the routing function client-side.
    """
    if final_status not in {"approved", "failed"}:
        raise ValueError(f"unexpected final_status={final_status!r}")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT planner_output FROM v2_run_log WHERE run_id = %s",
                (str(run_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"v2_run_log row {run_id} not found")
            prior = row[0] or {}
            parent_run_id = (
                prior.get("_parent_run_id") if isinstance(prior, dict) else None
            )
            output = {
                "typed_skills": typed_skills,
                "routing": routing,
                "_parent_run_id": parent_run_id,
                "_validator_log": validator_log,
            }
            err_msg: str | None = None
            if final_status == "failed":
                errs = validator_log.get("errors", [])[:3]
                err_msg = "; ".join(
                    f"[{e.get('code', '?')}] {e.get('message', '?')}" for e in errs
                ) or "Stage 4 typing failed"
            cur.execute(
                """
                UPDATE v2_run_log
                   SET planner_output = %s::jsonb,
                       status         = %s,
                       completed_at   = NOW(),
                       error_message  = %s
                 WHERE run_id = %s
                """,
                (json.dumps(output), final_status, err_msg, str(run_id)),
            )
            # Headless: NO v2_review_queue insert.
        conn.commit()
    return final_status


def get_latest_typed_skills_for_role(role_slug: str) -> list[dict] | None:
    """Latest *approved* Stage 4 typed_skills for the role. None when no
    approved Stage 4 row exists yet."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT planner_output
                  FROM v2_run_log
                 WHERE role_id = %s
                   AND prompt_version LIKE %s
                   AND status = 'approved'
                 ORDER BY completed_at DESC
                 LIMIT 1
                """,
                (role_slug, STAGE4_PROMPT_VERSION_PREFIX + "%"),
            )
            row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    body = row[0]
    if not isinstance(body, dict):
        return None
    skills = body.get("typed_skills")
    return skills if isinstance(skills, list) else None


def get_stage3_artifacts_for_role(role_slug: str) -> dict | None:
    """Pull the latest approved Stage 3 ``locked_dimensions`` AND
    ``unassigned_exemplars`` in one call. Stage 4 needs both to compute
    the role's full skill universe.

    Returns ``None`` when the role has no approved Stage 3 row yet.
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT planner_output
                  FROM v2_run_log
                 WHERE role_id = %s
                   AND prompt_version LIKE %s
                   AND status = 'approved'
                 ORDER BY completed_at DESC
                 LIMIT 1
                """,
                (role_slug, STAGE3_PROMPT_VERSION_PREFIX + "%"),
            )
            row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    body = row[0]
    if not isinstance(body, dict):
        return None
    return {
        "locked_dimensions": body.get("locked_dimensions") or [],
        "unassigned_exemplars": body.get("unassigned_exemplars") or [],
    }


def complete_containment_run(
    run_id: uuid.UUID,
    *,
    relationships: list[dict],
    validator_log: dict,
    final_status: str,
) -> str:
    """Stage 6 (containment) finished. Lifecycle:
      ``approved``         -> v2_run_log.status='approved', no queue row
      ``awaiting_review``  -> status='awaiting_review' + queue row
                                (cycles or type-rule errors need human gate)
      ``failed``           -> status='failed', error_message summary

    ``relationships`` is the list of SkillRelationships dicts the LLM
    produced (one per skill in the role's typed-skill set).
    """
    if final_status not in {"approved", "awaiting_review", "failed"}:
        raise ValueError(f"unexpected final_status={final_status!r}")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role_id, planner_output FROM v2_run_log WHERE run_id = %s",
                (str(run_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"v2_run_log row {run_id} not found")
            role_slug = row[0]
            prior = row[1] or {}
            parent_run_id = (
                prior.get("_parent_run_id") if isinstance(prior, dict) else None
            )
            output = {
                "relationships": relationships,
                "_parent_run_id": parent_run_id,
                "_validator_log": validator_log,
            }
            err_msg: str | None = None
            if final_status == "failed":
                errs = validator_log.get("errors", [])[:3]
                err_msg = "; ".join(
                    f"[{e.get('code', '?')}] {e.get('message', '?')}" for e in errs
                ) or "Stage 6 containment failed"
            cur.execute(
                """
                UPDATE v2_run_log
                   SET planner_output = %s::jsonb,
                       status         = %s,
                       completed_at   = NOW(),
                       error_message  = %s
                 WHERE run_id = %s
                """,
                (json.dumps(output), final_status, err_msg, str(run_id)),
            )
            if final_status == "awaiting_review":
                error_count = len(validator_log.get("errors", []))
                warn_count = len(validator_log.get("warnings", []))
                reason = (
                    f"Containment awaiting admin approval — "
                    f"{error_count} error(s), {warn_count} warning(s)"
                )
                review_payload = {
                    "role_slug": role_slug,
                    "validator_log": validator_log,
                    "parent_run_id": parent_run_id,
                }
                cur.execute(
                    """
                    INSERT INTO v2_review_queue
                        (run_id, item_type, payload, reason)
                    VALUES (%s, %s, %s::jsonb, %s)
                    """,
                    (
                        str(run_id),
                        CONTAINMENT_REVIEW_ITEM_TYPE,
                        json.dumps(review_payload),
                        reason,
                    ),
                )
        conn.commit()
    return final_status


def get_latest_containment_for_role(role_slug: str) -> list[dict] | None:
    """Latest *approved* Stage 6 relationships list for the role.
    None when no approved Stage 6 row exists yet."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT planner_output
                  FROM v2_run_log
                 WHERE role_id = %s
                   AND prompt_version LIKE %s
                   AND status = 'approved'
                 ORDER BY completed_at DESC
                 LIMIT 1
                """,
                (role_slug, STAGE6_PROMPT_VERSION_PREFIX + "%"),
            )
            row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    body = row[0]
    if not isinstance(body, dict):
        return None
    rels = body.get("relationships")
    return rels if isinstance(rels, list) else None


def complete_enrichment_run(
    run_id: uuid.UUID,
    *,
    enrichments: list[dict],
    validator_log: dict,
    final_status: str,
) -> str:
    """Stage 7 (enrichment) finished. Lifecycle:
      ``approved``         -> v2_run_log.status='approved', no queue row
      ``awaiting_review``  -> status='awaiting_review' + queue row
                                (ambiguity_flag=true OR maturity=deprecated)
      ``failed``           -> status='failed', error_message summary

    ``enrichments`` is the list of SkillEnrichment dicts — one per skill
    in the role's typed-skill set, merging the five agents' outputs.
    """
    if final_status not in {"approved", "awaiting_review", "failed"}:
        raise ValueError(f"unexpected final_status={final_status!r}")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role_id, planner_output FROM v2_run_log WHERE run_id = %s",
                (str(run_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"v2_run_log row {run_id} not found")
            role_slug = row[0]
            prior = row[1] or {}
            parent_run_id = (
                prior.get("_parent_run_id") if isinstance(prior, dict) else None
            )
            output = {
                "enrichments": enrichments,
                "_parent_run_id": parent_run_id,
                "_validator_log": validator_log,
            }
            err_msg: str | None = None
            if final_status == "failed":
                errs = validator_log.get("errors", [])[:3]
                err_msg = "; ".join(
                    f"[{e.get('code', '?')}] {e.get('message', '?')}" for e in errs
                ) or "Stage 7 enrichment failed"
            cur.execute(
                """
                UPDATE v2_run_log
                   SET planner_output = %s::jsonb,
                       status         = %s,
                       completed_at   = NOW(),
                       error_message  = %s
                 WHERE run_id = %s
                """,
                (json.dumps(output), final_status, err_msg, str(run_id)),
            )
            if final_status == "awaiting_review":
                ambiguous = [
                    e for e in enrichments
                    if (e.get("ambiguity") or {}).get("ambiguity_flag")
                ]
                deprecated = [
                    e for e in enrichments
                    if (e.get("maturity") or {}).get("maturity") == "deprecated"
                ]
                reason = (
                    f"Enrichment awaiting admin approval — "
                    f"{len(ambiguous)} ambiguous, {len(deprecated)} deprecated"
                )
                review_payload = {
                    "role_slug": role_slug,
                    "validator_log": validator_log,
                    "parent_run_id": parent_run_id,
                    "ambiguous_skill_ids": [e.get("skill_id") for e in ambiguous],
                    "deprecated_skill_ids": [e.get("skill_id") for e in deprecated],
                }
                cur.execute(
                    """
                    INSERT INTO v2_review_queue
                        (run_id, item_type, payload, reason)
                    VALUES (%s, %s, %s::jsonb, %s)
                    """,
                    (
                        str(run_id),
                        ENRICHMENT_REVIEW_ITEM_TYPE,
                        json.dumps(review_payload),
                        reason,
                    ),
                )
        conn.commit()
    return final_status


def complete_load_run(
    run_id: uuid.UUID,
    *,
    load_summary: dict,
    validator_log: dict,
    final_status: str,
) -> str:
    """Stage 8 (catalog load) finished. Lifecycle:
      ``approved`` -> v2_run_log.status='approved' (auto, default path)
      ``failed``   -> status='failed' with error summary
    """
    if final_status not in {"approved", "failed"}:
        raise ValueError(f"unexpected final_status={final_status!r}")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT planner_output FROM v2_run_log WHERE run_id = %s",
                (str(run_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"v2_run_log row {run_id} not found")
            prior = row[0] or {}
            parent_run_id = (
                prior.get("_parent_run_id") if isinstance(prior, dict) else None
            )
            output = {
                "load_summary": load_summary,
                "_parent_run_id": parent_run_id,
                "_validator_log": validator_log,
            }
            err_msg: str | None = None
            if final_status == "failed":
                errs = validator_log.get("errors", [])[:3]
                err_msg = "; ".join(
                    f"[{e.get('code', '?')}] {e.get('message', '?')}" for e in errs
                ) or "Stage 8 catalog load failed"
            cur.execute(
                """
                UPDATE v2_run_log
                   SET planner_output = %s::jsonb,
                       status         = %s,
                       completed_at   = NOW(),
                       error_message  = %s
                 WHERE run_id = %s
                """,
                (json.dumps(output), final_status, err_msg, str(run_id)),
            )
        conn.commit()
    return final_status


def query_kras_by_embedding(
    query_embedding: list[float],
    *,
    similarity_threshold: float = 0.65,
    max_results: int = 20,
) -> list[dict]:
    """Top-N KRA matches for a query embedding (pgvector cosine).

    Thin sync wrapper around the ``search_kras_by_embedding`` SQL function
    defined in ``db/v3_additions_kras.sql``. The role-classifier's
    Stage 3c calls this with one JD responsibility bundle at a time.

    Returns a list of dicts with keys: role_id, role_slug, role_display_name,
    kra_id, kra_text, source_field, similarity.
    """
    if not query_embedding:
        return []
    vec_literal = "[" + ",".join(f"{x:.7f}" for x in query_embedding) + "]"
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role_id, role_slug, role_display_name,
                       kra_id, kra_text, source_field, similarity
                  FROM search_kras_by_embedding(%s::vector, %s, %s)
                """,
                (vec_literal, similarity_threshold, max_results),
            )
            rows = cur.fetchall()
    return [
        {
            "role_id": r[0],
            "role_slug": r[1],
            "role_display_name": r[2],
            "kra_id": r[3],
            "kra_text": r[4],
            "source_field": r[5],
            "similarity": float(r[6]),
        }
        for r in rows
    ]


def get_latest_enrichment_for_role(role_slug: str) -> list[dict] | None:
    """Latest *approved* Stage 7 enrichment list for the role.
    None when no approved Stage 7 row exists yet."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT planner_output
                  FROM v2_run_log
                 WHERE role_id = %s
                   AND prompt_version LIKE %s
                   AND status = 'approved'
                 ORDER BY completed_at DESC
                 LIMIT 1
                """,
                (role_slug, STAGE7_PROMPT_VERSION_PREFIX + "%"),
            )
            row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    body = row[0]
    if not isinstance(body, dict):
        return None
    enr = body.get("enrichments")
    return enr if isinstance(enr, list) else None


def complete_placement_run(
    run_id: uuid.UUID,
    *,
    placed_skills: list[dict],
    routing: dict,
    validator_log: dict,
    final_status: str,
) -> str:
    """Stage 5 (placement) finished. Headless lifecycle:
      ``approved`` -> v2_run_log.status='approved', no review_queue row
      ``failed``   -> v2_run_log.status='failed', error_message summary
    """
    if final_status not in {"approved", "failed"}:
        raise ValueError(f"unexpected final_status={final_status!r}")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT planner_output FROM v2_run_log WHERE run_id = %s",
                (str(run_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"v2_run_log row {run_id} not found")
            prior = row[0] or {}
            parent_run_id = (
                prior.get("_parent_run_id") if isinstance(prior, dict) else None
            )
            output = {
                "placed_skills": placed_skills,
                "routing": routing,
                "_parent_run_id": parent_run_id,
                "_validator_log": validator_log,
            }
            err_msg: str | None = None
            if final_status == "failed":
                errs = validator_log.get("errors", [])[:3]
                err_msg = "; ".join(
                    f"[{e.get('code', '?')}] {e.get('message', '?')}" for e in errs
                ) or "Stage 5 placement failed"
            cur.execute(
                """
                UPDATE v2_run_log
                   SET planner_output = %s::jsonb,
                       status         = %s,
                       completed_at   = NOW(),
                       error_message  = %s
                 WHERE run_id = %s
                """,
                (json.dumps(output), final_status, err_msg, str(run_id)),
            )
            # Headless: no v2_review_queue row.
        conn.commit()
    return final_status


def get_latest_placed_skills_for_role(role_slug: str) -> list[dict] | None:
    """Latest *approved* Stage 5 placed_skills for the role. None when
    no approved Stage 5 row exists yet."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT planner_output
                  FROM v2_run_log
                 WHERE role_id = %s
                   AND prompt_version LIKE %s
                   AND status = 'approved'
                 ORDER BY completed_at DESC
                 LIMIT 1
                """,
                (role_slug, STAGE5_PROMPT_VERSION_PREFIX + "%"),
            )
            row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    body = row[0]
    if not isinstance(body, dict):
        return None
    placed = body.get("placed_skills")
    return placed if isinstance(placed, list) else None


def find_roles_needing_typing(target_prompt_version: str) -> list[dict]:
    """Roles whose latest approved Stage 3 has no current approved
    Stage 4 row (or has one under an older prompt version). Used by the
    Stage 4 backfill driver."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT latest_recon.role_slug,
                       latest_recon.role_display
                  FROM (
                        SELECT DISTINCT ON (role_id)
                               role_id      AS role_slug,
                               role_display
                          FROM v2_run_log
                         WHERE prompt_version LIKE %s
                           AND status = 'approved'
                         ORDER BY role_id, completed_at DESC
                       ) latest_recon
                 WHERE NOT EXISTS (
                        SELECT 1 FROM v2_run_log s4
                         WHERE s4.role_id = latest_recon.role_slug
                           AND s4.prompt_version = %s
                           AND s4.status = 'approved'
                       )
                 ORDER BY latest_recon.role_slug
                """,
                (STAGE3_PROMPT_VERSION_PREFIX + "%", target_prompt_version),
            )
            rows = cur.fetchall()
    return [{"role_slug": r[0], "role_display": r[1]} for r in rows]


def find_roles_needing_reconciliation(target_prompt_version: str) -> list[dict]:
    """Roles whose latest approved Stage 2 row has no current approved
    Stage 3 row (or has one produced under an older prompt version).
    Used by the Stage 3 backfill driver."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT latest_dims.role_slug,
                       latest_dims.role_display
                  FROM (
                        SELECT DISTINCT ON (role_id)
                               role_id      AS role_slug,
                               role_display
                          FROM v2_run_log
                         WHERE prompt_version LIKE %s
                           AND status = 'approved'
                         ORDER BY role_id, completed_at DESC
                       ) latest_dims
                 WHERE NOT EXISTS (
                        SELECT 1 FROM v2_run_log s3
                         WHERE s3.role_id = latest_dims.role_slug
                           AND s3.prompt_version = %s
                           AND s3.status = 'approved'
                       )
                 ORDER BY latest_dims.role_slug
                """,
                (STAGE2_PROMPT_VERSION_PREFIX + "%", target_prompt_version),
            )
            rows = cur.fetchall()
    return [
        {"role_slug": r[0], "role_display": r[1]}
        for r in rows
    ]


def find_roles_needing_dimensions(target_prompt_version: str) -> list[dict]:
    """Return roles whose latest approved role card has no approved Stage 2
    row, OR has an approved Stage 2 row whose prompt_version differs from
    ``target_prompt_version``. Used by backfill_dimensions.

    Each item: ``{"role_slug", "role_display", "role_card_run_id"}``."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT latest_card.role_slug,
                       latest_card.role_display,
                       latest_card.role_card_run_id
                  FROM (
                        SELECT DISTINCT ON (role_id)
                               role_id        AS role_slug,
                               role_display,
                               run_id         AS role_card_run_id
                          FROM v2_run_log
                         WHERE prompt_version LIKE %s
                           AND status = 'approved'
                         ORDER BY role_id, completed_at DESC
                       ) latest_card
                 WHERE NOT EXISTS (
                        SELECT 1 FROM v2_run_log s2
                         WHERE s2.role_id = latest_card.role_slug
                           AND s2.prompt_version = %s
                           AND s2.status = 'approved'
                       )
                 ORDER BY latest_card.role_slug
                """,
                (STAGE1_PROMPT_VERSION_PREFIX + "%", target_prompt_version),
            )
            rows = cur.fetchall()
    return [
        {
            "role_slug": r[0],
            "role_display": r[1],
            "role_card_run_id": str(r[2]),
        }
        for r in rows
    ]
