-- ============================================================================
-- skill_library_v3 — additive DDL applied on top of the existing
-- db/schema.sql + db/schema_v2_additions.sql in the new `skill-library` DB.
-- Idempotent. Safe to re-run.
-- ============================================================================

-- ─── Performance index for "fetch latest approved charter per role" ─────────
-- Stage 0 produces one v2_run_log row per generation attempt; the most-recent
-- row with status='approved' AND prompt_version like 'stage0_charter_%' is the
-- current canonical charter for that role. The runner reads this when building
-- the adjacency block for subsequent roles' charter prompts.
CREATE INDEX IF NOT EXISTS idx_v2_run_log_approved_charter
    ON v2_run_log (role_id, completed_at DESC)
    WHERE status = 'approved';

-- Stage 1 partial index — narrower than the charter index because the
-- queries that read it always filter on prompt_version LIKE 'stage1_anchor_%'.
-- Used by get_role_card_for_role, _fetch_latest_role_card_run, and the
-- adjacency lookup that Stage 2+ will issue.
CREATE INDEX IF NOT EXISTS idx_v2_run_log_approved_role_card
    ON v2_run_log (role_id, completed_at DESC)
    WHERE status = 'approved' AND prompt_version LIKE 'stage1_anchor_%';


-- ─── JD samples auxiliary table (Kaggle ingest) ─────────────────────────────
-- Holds raw job descriptions imported from Kaggle (currently
-- adityarajsrv/job-descriptions-2025-tech-and-non-tech-roles). The Stage 0
-- charter prompt SELECTs LIMIT 5 of these per role at run time as optional
-- grounding. role_slug references roles.slug by convention (no FK so rows
-- can be ingested before any role exists, then matched lazily).
CREATE TABLE IF NOT EXISTS jd_samples (
    id              BIGSERIAL PRIMARY KEY,
    role_slug       VARCHAR(120),
    source_dataset  VARCHAR(120) NOT NULL,
    source_row_id   VARCHAR(200),
    title           TEXT NOT NULL,
    body            TEXT NOT NULL,
    match_score     REAL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jd_samples_role
    ON jd_samples (role_slug);

CREATE INDEX IF NOT EXISTS idx_jd_samples_score
    ON jd_samples (match_score)
    WHERE role_slug IS NOT NULL;
