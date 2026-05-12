"""Repository for JD pipeline run history.

One row per end-to-end run of the current 3-step JD skill pipeline:

  1. ``POST /skills/extract-from-jd`` — LLM extracts ``final_skills`` from JD text;
     ``start_run`` inserts a row (``status=extract_from_jd_done``) with ``api1_response``.
  2. ``POST /skills/extract-details`` — alias + dimension catalogue + v3 new-skill
     orchestration; ``attach_api2`` stores ``api2_response`` (``status=extract_details_done``).
  3. ``POST /skills/final-role-output`` — persists links / new canonicals when enabled;
     ``attach_api3`` stores ``api3_response`` (``status=completed``) plus denormalized
     role and skill list fields.

``jd_run_artifacts`` records concrete inserts (aliases, canonicals, dimensions, links)
and *proposed* rows when writes are skipped.

"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from psycopg2 import sql

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

log = logging.getLogger(__name__)


def _coerce_uuid(value: str | UUID | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return str(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(UUID(text))
    except (ValueError, AttributeError, TypeError):
        return None


def _json_safe(value: Any) -> Any:
    """conversion of pydantic models / dataclasses to JSON-safe dicts."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return json.loads(json.dumps(value, default=str))


_VALID_ARTIFACT_KINDS = {
    "non_skill_added",
    "alias_added",
    "canonical_skill_added",
    "library_enrichment_backfilled",
    "dimension_created",
    "role_created",
    "dimension_skill_link",
    "role_dimension_link",
    "canonical_skill_proposed",
    "dimension_proposed",
    "dimension_skill_link_proposed",
    "role_dimension_link_proposed",
}


class JdPipelineRunRepository:
    """repository for JD pipeline run history.

    Public methods catch and log exceptions so that history failures never
    surface as user-facing errors. Read methods (`list_runs`, `get_run`)
    raise on failure since they are called from explicit history endpoints.
    """

    def __init__(
        self,
        *,
        database_url: str | None = None,
        schema: str | None = None,
    ) -> None:
        self.database_url = (
            database_url
            if database_url is not None
            else os.getenv("DATABASE_URL", "")
        )
        self.db_host = os.getenv("DB_HOST", "")
        self.db_port = int(os.getenv("DB_PORT", "5432"))
        self.db_name = os.getenv("DB_NAME", "")
        self.db_user = os.getenv("DB_USER", "")
        self.db_password = os.getenv("DB_PASSWORD", "")
        self.schema = (
            schema
            if schema is not None
            else os.getenv(
                "SKILL_LIBRARY_SCHEMA",
                os.getenv("DB_SCHEMA", "public"),
            )
        )
        self.runs_table = "jd_pipeline_runs"
        self.artifacts_table = "jd_run_artifacts"

        if not self.database_url and not (
            self.db_host and self.db_name and self.db_user and self.db_password
        ):
            raise ValueError(
                "Provide DATABASE_URL or DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD."
            )

    def _connect(self):
        if self.database_url:
            return psycopg2.connect(self.database_url)
        return psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
            sslmode=os.getenv("DB_SSLMODE", "require"),
            connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "30")),
        )

    # ── Schema bootstrap ────────────────────────────────────────────────────
    def ensure_schema(self) -> bool:
        """Create the history tables if they don't exist. Returns True on success."""
        runs_ddl = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {schema}.{runs} (
                id                    UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
                jd_text               TEXT         NOT NULL,
                status                TEXT         NOT NULL,
                api1_response         JSONB,
                api2_response         JSONB,
                api3_response         JSONB,
                chosen_role_display   TEXT,
                chosen_role_id        BIGINT,
                final_skills_count    INTEGER,
                final_skills          TEXT[],
                jd_role_hint_display  TEXT,
                error_message         TEXT,
                duration_ms           INTEGER,
                created_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                updated_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
            """
        ).format(
            schema=sql.Identifier(self.schema),
            runs=sql.Identifier(self.runs_table),
        )
        runs_idx_created = sql.SQL(
            "CREATE INDEX IF NOT EXISTS idx_jd_runs_created_at "
            "ON {schema}.{runs} (created_at DESC)"
        ).format(
            schema=sql.Identifier(self.schema),
            runs=sql.Identifier(self.runs_table),
        )
        runs_idx_status = sql.SQL(
            "CREATE INDEX IF NOT EXISTS idx_jd_runs_status "
            "ON {schema}.{runs} (status)"
        ).format(
            schema=sql.Identifier(self.schema),
            runs=sql.Identifier(self.runs_table),
        )

        artifacts_ddl = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {schema}.{artifacts} (
                id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                run_id          UUID         NOT NULL
                                  REFERENCES {schema}.{runs}(id) ON DELETE CASCADE,
                artifact_kind   TEXT         NOT NULL,
                artifact_id     BIGINT,
                artifact_text   TEXT,
                created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
            """
        ).format(
            schema=sql.Identifier(self.schema),
            artifacts=sql.Identifier(self.artifacts_table),
            runs=sql.Identifier(self.runs_table),
        )
        artifacts_idx_run = sql.SQL(
            "CREATE INDEX IF NOT EXISTS idx_jd_run_artifacts_run "
            "ON {schema}.{artifacts} (run_id)"
        ).format(
            schema=sql.Identifier(self.schema),
            artifacts=sql.Identifier(self.artifacts_table),
        )
        artifacts_idx_kind = sql.SQL(
            "CREATE INDEX IF NOT EXISTS idx_jd_run_artifacts_kind "
            "ON {schema}.{artifacts} (run_id, artifact_kind)"
        ).format(
            schema=sql.Identifier(self.schema),
            artifacts=sql.Identifier(self.artifacts_table),
        )

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
                    cur.execute(runs_ddl)
                    cur.execute(runs_idx_created)
                    cur.execute(runs_idx_status)
                    cur.execute(artifacts_ddl)
                    cur.execute(artifacts_idx_run)
                    cur.execute(artifacts_idx_kind)
                    for alter in (
                        "ALTER TABLE {schema}.{runs} ADD COLUMN IF NOT EXISTS llm_cost_api1_usd DOUBLE PRECISION",
                        "ALTER TABLE {schema}.{runs} ADD COLUMN IF NOT EXISTS llm_cost_api2_usd DOUBLE PRECISION",
                        "ALTER TABLE {schema}.{runs} ADD COLUMN IF NOT EXISTS llm_cost_api3_usd DOUBLE PRECISION",
                    ):
                        cur.execute(
                            sql.SQL(alter).format(
                                schema=sql.Identifier(self.schema),
                                runs=sql.Identifier(self.runs_table),
                            )
                        )
                conn.commit()
            log.info(
                "[JD pipeline / history] schema ensured for %s.%s + %s.%s",
                self.schema,
                self.runs_table,
                self.schema,
                self.artifacts_table,
            )
            return True
        except Exception as exc:
            log.warning(
                "[JD pipeline / history] ensure_schema failed (%s); "
                "history persistence will be a no-op",
                exc,
            )
            return False

    # ── Writes ────────────────────────────────────────────────
    def start_run(
        self,
        *,
        jd_text: str,
        api1_response: Any,
        jd_role_hint_display: str | None = None,
        llm_cost_usd: float | None = None,
    ) -> str | None:
        """Insert a new run row with the API 1 response. Returns run_id or None."""
        try:
            payload = json.dumps(_json_safe(api1_response), ensure_ascii=False)
            stmt = sql.SQL(
                """
                INSERT INTO {schema}.{runs}
                    (jd_text, status, api1_response, jd_role_hint_display, llm_cost_api1_usd)
                VALUES (%s, %s, %s::jsonb, %s, %s)
                RETURNING id
                """
            ).format(
                schema=sql.Identifier(self.schema),
                runs=sql.Identifier(self.runs_table),
            )
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        stmt,
                        (
                            jd_text,
                            "extract_from_jd_done",
                            payload,
                            jd_role_hint_display,
                            float(llm_cost_usd) if llm_cost_usd is not None else None,
                        ),
                    )
                    row = cur.fetchone()
                conn.commit()
            run_id = str(row[0]) if row else None
            if run_id:
                log.info("[JD pipeline / history] start_run created run_id=%s", run_id)
            return run_id
        except Exception as exc:
            log.warning("[JD pipeline / history] start_run failed: %s", exc)
            return None

    def attach_api2(
        self,
        run_id: str | UUID | None,
        api2_response: Any,
        *,
        llm_cost_usd: float | None = None,
    ) -> bool:
        """Update the row with API 2 response and bump status."""
        rid = _coerce_uuid(run_id)
        if not rid:
            return False
        try:
            payload = json.dumps(_json_safe(api2_response), ensure_ascii=False)
            stmt = sql.SQL(
                """
                UPDATE {schema}.{runs}
                   SET api2_response = %s::jsonb,
                       status = %s,
                       llm_cost_api2_usd = COALESCE(%s, llm_cost_api2_usd),
                       updated_at = NOW()
                 WHERE id = %s::uuid
                """
            ).format(
                schema=sql.Identifier(self.schema),
                runs=sql.Identifier(self.runs_table),
            )
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        stmt,
                        (
                            payload,
                            "extract_details_done",
                            float(llm_cost_usd) if llm_cost_usd is not None else None,
                            rid,
                        ),
                    )
                    rowcount = cur.rowcount
                conn.commit()
            if rowcount == 0:
                log.warning(
                    "[JD pipeline / history] attach_api2: run_id %s not found",
                    rid,
                )
                return False
            return True
        except Exception as exc:
            log.warning(
                "[JD pipeline / history] attach_api2 failed for run_id=%s: %s",
                rid,
                exc,
            )
            return False

    def attach_api3(
        self,
        run_id: str | UUID | None,
        api3_response: Any,
        *,
        chosen_role_display: str | None = None,
        chosen_role_id: int | None = None,
        final_skills: Iterable[str] | None = None,
        duration_ms: int | None = None,
        llm_cost_usd: float | None = None,
    ) -> bool:
        """Mark a run completed with the API 3 response and denormalized fields."""
        rid = _coerce_uuid(run_id)
        if not rid:
            return False
        try:
            payload = json.dumps(_json_safe(api3_response), ensure_ascii=False)
            skills_list = (
                [str(s) for s in final_skills if s is not None]
                if final_skills is not None
                else None
            )
            skills_count = len(skills_list) if skills_list is not None else None
            stmt = sql.SQL(
                """
                UPDATE {schema}.{runs}
                   SET api3_response = %s::jsonb,
                       status = %s,
                       chosen_role_display = COALESCE(%s, chosen_role_display),
                       chosen_role_id = COALESCE(%s, chosen_role_id),
                       final_skills = COALESCE(%s, final_skills),
                       final_skills_count = COALESCE(%s, final_skills_count),
                       duration_ms = COALESCE(%s, duration_ms),
                       llm_cost_api3_usd = COALESCE(%s, llm_cost_api3_usd),
                       updated_at = NOW()
                 WHERE id = %s::uuid
                """
            ).format(
                schema=sql.Identifier(self.schema),
                runs=sql.Identifier(self.runs_table),
            )
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        stmt,
                        (
                            payload,
                            "completed",
                            chosen_role_display,
                            chosen_role_id,
                            skills_list,
                            skills_count,
                            duration_ms,
                            float(llm_cost_usd) if llm_cost_usd is not None else None,
                            rid,
                        ),
                    )
                    rowcount = cur.rowcount
                conn.commit()
            if rowcount == 0:
                log.warning(
                    "[JD pipeline / history] attach_api3: run_id %s not found",
                    rid,
                )
                return False
            return True
        except Exception as exc:
            log.warning(
                "[JD pipeline / history] attach_api3 failed for run_id=%s: %s",
                rid,
                exc,
            )
            return False

    def get_run_llm_costs_usd(
        self, run_id: str | UUID | None
    ) -> tuple[float | None, float | None, float | None] | None:
        """Return stored per-API LLM costs for a run, or None if row/columns unavailable."""
        rid = _coerce_uuid(run_id)
        if not rid:
            return None
        stmt = sql.SQL(
            """
            SELECT llm_cost_api1_usd, llm_cost_api2_usd, llm_cost_api3_usd
              FROM {schema}.{runs}
             WHERE id = %s::uuid
            """
        ).format(
            schema=sql.Identifier(self.schema),
            runs=sql.Identifier(self.runs_table),
        )
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(stmt, (rid,))
                    row = cur.fetchone()
            if not row:
                return None
            def _f(v: Any) -> float | None:
                if v is None:
                    return None
                return float(v)

            return (_f(row[0]), _f(row[1]), _f(row[2]))
        except Exception:
            return None

    def mark_failed(
        self,
        run_id: str | UUID | None,
        error_message: str,
    ) -> bool:
        rid = _coerce_uuid(run_id)
        if not rid:
            return False
        try:
            stmt = sql.SQL(
                """
                UPDATE {schema}.{runs}
                   SET status = %s,
                       error_message = %s,
                       updated_at = NOW()
                 WHERE id = %s::uuid
                """
            ).format(
                schema=sql.Identifier(self.schema),
                runs=sql.Identifier(self.runs_table),
            )
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(stmt, ("failed", error_message[:8000], rid))
                conn.commit()
            return True
        except Exception as exc:
            log.warning(
                "[JD pipeline / history] mark_failed failed for run_id=%s: %s",
                rid,
                exc,
            )
            return False

    def record_artifact(
        self,
        run_id: str | UUID | None,
        kind: str,
        *,
        artifact_id: int | None = None,
        artifact_text: str | None = None,
    ) -> bool:
        rid = _coerce_uuid(run_id)
        if not rid:
            return False
        if kind not in _VALID_ARTIFACT_KINDS:
            log.warning(
                "[JD pipeline / history] unknown artifact_kind=%r — skipped", kind
            )
            return False
        try:
            stmt = sql.SQL(
                """
                INSERT INTO {schema}.{artifacts}
                    (run_id, artifact_kind, artifact_id, artifact_text)
                VALUES (%s::uuid, %s, %s, %s)
                """
            ).format(
                schema=sql.Identifier(self.schema),
                artifacts=sql.Identifier(self.artifacts_table),
            )
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        stmt,
                        (
                            rid,
                            kind,
                            int(artifact_id) if artifact_id is not None else None,
                            (str(artifact_text) if artifact_text is not None else None),
                        ),
                    )
                conn.commit()
            return True
        except Exception as exc:
            log.warning(
                "[JD pipeline / history] record_artifact failed (kind=%s): %s",
                kind,
                exc,
            )
            return False

    def record_artifacts_bulk(
        self,
        run_id: str | UUID | None,
        items: Iterable[dict[str, Any]],
    ) -> int:
        """Insert many artifacts in a single trip. Returns inserted count."""
        rid = _coerce_uuid(run_id)
        if not rid:
            return 0
        rows: list[tuple[str, str, int | None, str | None]] = []
        for it in items:
            kind = str(it.get("kind") or "")
            if kind not in _VALID_ARTIFACT_KINDS:
                continue
            aid = it.get("artifact_id")
            txt = it.get("artifact_text")
            rows.append(
                (
                    rid,
                    kind,
                    int(aid) if aid is not None else None,
                    str(txt) if txt is not None else None,
                )
            )
        if not rows:
            return 0
        try:
            stmt = sql.SQL(
                """
                INSERT INTO {schema}.{artifacts}
                    (run_id, artifact_kind, artifact_id, artifact_text)
                VALUES (%s::uuid, %s, %s, %s)
                """
            ).format(
                schema=sql.Identifier(self.schema),
                artifacts=sql.Identifier(self.artifacts_table),
            )
            with self._connect() as conn:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_batch(cur, stmt.as_string(conn), rows)
                conn.commit()
            return len(rows)
        except Exception as exc:
            log.warning(
                "[JD pipeline / history] record_artifacts_bulk failed: %s", exc
            )
            return 0

    # ── Reads ───────────────────────────────────────────────────────────────
    def list_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        where_sql = sql.SQL("")
        params: list[Any] = []
        if status:
            where_sql = sql.SQL("WHERE status = %s")
            params.append(status)

        stmt = sql.SQL(
            """
            SELECT id::text                     AS id,
                   created_at,
                   updated_at,
                   status,
                   chosen_role_display,
                   chosen_role_id,
                   final_skills_count,
                   final_skills,
                   jd_role_hint_display,
                   duration_ms,
                   error_message,
                   LEFT(jd_text, 240)           AS jd_text_preview,
                   LENGTH(jd_text)              AS jd_text_length
              FROM {schema}.{runs}
              {where}
              ORDER BY created_at DESC
              LIMIT %s OFFSET %s
            """
        ).format(
            schema=sql.Identifier(self.schema),
            runs=sql.Identifier(self.runs_table),
            where=where_sql,
        )
        params.extend([limit, offset])
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(stmt, tuple(params))
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str | UUID) -> dict[str, Any] | None:
        rid = _coerce_uuid(run_id)
        if not rid:
            return None
        run_stmt = sql.SQL(
            """
            SELECT id::text                AS id,
                   jd_text,
                   status,
                   api1_response,
                   api2_response,
                   api3_response,
                   chosen_role_display,
                   chosen_role_id,
                   final_skills_count,
                   final_skills,
                   jd_role_hint_display,
                   error_message,
                   duration_ms,
                   created_at,
                   updated_at
              FROM {schema}.{runs}
             WHERE id = %s::uuid
            """
        ).format(
            schema=sql.Identifier(self.schema),
            runs=sql.Identifier(self.runs_table),
        )
        artifacts_stmt = sql.SQL(
            """
            SELECT id,
                   artifact_kind,
                   artifact_id,
                   artifact_text,
                   created_at
              FROM {schema}.{artifacts}
             WHERE run_id = %s::uuid
             ORDER BY created_at ASC, id ASC
            """
        ).format(
            schema=sql.Identifier(self.schema),
            artifacts=sql.Identifier(self.artifacts_table),
        )
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(run_stmt, (rid,))
                run_row = cur.fetchone()
                if not run_row:
                    return None
                cur.execute(artifacts_stmt, (rid,))
                artifact_rows = cur.fetchall()
        result = dict(run_row)
        result["artifacts"] = [dict(r) for r in artifact_rows]
        return result
