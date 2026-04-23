-- ============================================================================
-- skill_library_v2 — additive schema for the LangGraph skill-library pipeline
-- Requires db/schema.sql to have been applied first (canonical_skills, etc.).
-- Idempotent: safe to re-run.
-- ============================================================================

-- ─── Run log ─────────────────────────────────────────────────────────────────
-- One row per graph invocation. Captures the Planner output and run status so
-- later phases (Generator, Critics) can attach their own rows/artifacts to the
-- run_id for a full provenance trail.

CREATE TABLE IF NOT EXISTS v2_run_log (
    run_id            UUID         PRIMARY KEY,
    role_id           VARCHAR(100) NOT NULL,
    role_display      VARCHAR(200) NOT NULL,
    prompt_version    VARCHAR(50)  NOT NULL,
    model_snapshot    VARCHAR(100) NOT NULL,
    planner_output    JSONB        NOT NULL,
    planner_reasoning TEXT,
    status            VARCHAR(30)  NOT NULL,   -- 'planned' | 'complete' | 'failed'
    error_message     TEXT,
    started_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    completed_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_v2_run_log_role ON v2_run_log (role_id);
CREATE INDEX IF NOT EXISTS idx_v2_run_log_status ON v2_run_log (status);


-- ─── Review queue ────────────────────────────────────────────────────────────
-- Anything the pipeline cannot auto-decide is enqueued here with provenance.
-- Unresolved-only index keeps human-review triage fast as the table grows.

CREATE TABLE IF NOT EXISTS v2_review_queue (
    id            BIGSERIAL     PRIMARY KEY,
    run_id        UUID          REFERENCES v2_run_log(run_id) ON DELETE CASCADE,
    item_type     VARCHAR(40)   NOT NULL,   -- 'dimension_flagged' | 'skill_proposed_new' | ...
    payload       JSONB         NOT NULL,
    reason        TEXT          NOT NULL,
    created_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    resolved      BOOLEAN       NOT NULL DEFAULT FALSE,
    resolved_by   VARCHAR(100),
    resolved_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_v2_review_queue_unresolved
    ON v2_review_queue (created_at) WHERE resolved = FALSE;
CREATE INDEX IF NOT EXISTS idx_v2_review_queue_run ON v2_review_queue (run_id);
