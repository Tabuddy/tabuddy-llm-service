-- ============================================================================
-- Staging table for LinkedIn position_title ingest.
--
-- Holds cleaned, deduped role names mined from linkedin.profile_experiences.
-- Kept separate from `roles` until the cleaning logic has been verified;
-- once happy we'll copy approved rows into `roles` with source='LINKEDIN'.
--
-- Idempotent. Safe to re-run.
--
-- Apply on prod (skill-library schema):
--     PGOPTIONS='-c search_path="skill-library",dev,public' \
--       psql "$PG_PROD" -f db/schema_linkedin_roles_staging.sql
-- ============================================================================

CREATE TABLE IF NOT EXISTS linkedin_roles (
    id           BIGINT       GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    slug         VARCHAR(200) NOT NULL UNIQUE,
    display_name VARCHAR(255) NOT NULL,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Tear down the earlier raw_count column + its index if a prior version of
-- this migration created them. Plain CREATE TABLE IF NOT EXISTS won't change
-- an existing table's schema, so we do it explicitly here.
DROP INDEX  IF EXISTS idx_linkedin_roles_raw_count;
ALTER TABLE linkedin_roles DROP COLUMN IF EXISTS raw_count;
ALTER TABLE linkedin_roles DROP COLUMN IF EXISTS updated_at;
