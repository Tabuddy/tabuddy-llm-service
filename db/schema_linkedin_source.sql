-- ============================================================================
-- Add LINKEDIN provenance to the entity_source enum.
-- Idempotent. Safe to re-run.
--
-- Apply on prod:
--     PGOPTIONS='-c search_path="skill-library",dev,public' \
--       psql "$PG_PROD" -f db/schema_linkedin_source.sql
--
-- Apply locally:
--     psql "$SKILL_LIBRARY_PG_DSN" -f db/schema_linkedin_source.sql
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_enum e
          JOIN pg_type t ON t.oid = e.enumtypid
         WHERE t.typname = 'entity_source'
           AND e.enumlabel = 'LINKEDIN'
    ) THEN
        ALTER TYPE entity_source ADD VALUE 'LINKEDIN';
    END IF;
END
$$;
