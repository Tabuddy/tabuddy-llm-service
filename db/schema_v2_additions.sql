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


-- ─── Dimension embeddings (cross-role dedup at Stage 8 load) ─────────────
-- Mirrors the name_embedding columns on categories / sub_categories /
-- canonical_skills. Stage 8's loader uses pgvector nearest-neighbor on this
-- column to detect when a newly-loaded role's dimension is semantically
-- equivalent to an existing dimension and link to the existing row instead
-- of inserting a duplicate.

ALTER TABLE dimensions
    ADD COLUMN IF NOT EXISTS name_embedding vector(1536);

CREATE INDEX IF NOT EXISTS idx_dimensions_name_embedding ON dimensions
    USING hnsw (name_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);


-- ─── Role aliases (mirror of skill_aliases for the role catalog) ─────────
-- Lets `SELECT role_id FROM role_aliases WHERE alias_lower = 'fe'` resolve
-- short forms ("FE", "SRE"), full names ("Front-end Developer"), and other
-- surface variants of a canonical role. Reuses the alias_type and
-- match_strategy enums already defined in db/schema.sql so the role + skill
-- catalogs stay symmetric on the consumer side.

CREATE TABLE IF NOT EXISTS role_aliases (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    role_id         BIGINT              NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    alias_text      VARCHAR(200)        NOT NULL,
    alias_type      alias_type          NOT NULL,
    match_strategy  match_strategy      NOT NULL DEFAULT 'CASE_INSENSITIVE',
    match_pattern   TEXT,
    is_primary      BOOLEAN             NOT NULL DEFAULT FALSE,
    region_affinity TEXT[],
    alias_lower     VARCHAR(200) GENERATED ALWAYS AS (LOWER(alias_text)) STORED,
    alias_embedding vector(1536),
    CONSTRAINT uq_role_alias UNIQUE (role_id, alias_text)
);
CREATE INDEX IF NOT EXISTS idx_role_aliases_lower ON role_aliases (alias_lower);
CREATE INDEX IF NOT EXISTS idx_role_aliases_trgm  ON role_aliases USING gin (alias_lower gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_role_aliases_embedding ON role_aliases
    USING hnsw (alias_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE UNIQUE INDEX IF NOT EXISTS idx_role_aliases_primary ON role_aliases (role_id) WHERE is_primary = TRUE;
CREATE INDEX IF NOT EXISTS idx_role_aliases_role_id ON role_aliases (role_id);


-- ─── Canonical skill enrichment columns (Stage 7 -> permanent storage) ───
-- Stage 7 produces vendor / license / year_introduced / maturity reasoning
-- per skill. Previously these lived only in v2_run_log.planner_output JSONB
-- so consumers couldn't query them. Promote to first-class columns on
-- canonical_skills so /api/canonical-skill/skills can surface them and
-- reports can filter by vendor, license, or year.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'license_type') THEN
        CREATE TYPE license_type AS ENUM (
            'proprietary', 'mit', 'apache_2', 'bsd', 'gpl_v2', 'gpl_v3',
            'lgpl', 'mpl', 'isc', 'unlicense', 'cc_by', 'cc_by_sa',
            'other_open', 'unknown'
        );
    END IF;
END$$;

ALTER TABLE canonical_skills
    ADD COLUMN IF NOT EXISTS vendor              VARCHAR(120),
    ADD COLUMN IF NOT EXISTS license             license_type,
    ADD COLUMN IF NOT EXISTS year_introduced     SMALLINT,
    ADD COLUMN IF NOT EXISTS maturity_reasoning  TEXT;
