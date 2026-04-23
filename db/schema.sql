-- ============================================================================
-- TAbuddy Canonical Skill Library — PostgreSQL Schema
-- Requires: PostgreSQL 15+ with pgvector extension
-- ============================================================================

-- Enable pgvector for embedding-based similarity search
CREATE EXTENSION IF NOT EXISTS vector;
-- Enable pg_trgm for fuzzy text search (trigram similarity)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ─── ENUM TYPES ──────────────────────────────────────────────────────────────

CREATE TYPE skill_category AS ENUM (
    'LANGUAGE', 'FRAMEWORK', 'LIBRARY', 'DATABASE',
    'CLOUD_PLATFORM', 'CLOUD_SERVICE', 'DEVOPS_TOOL',
    'ARCHITECTURE_CONCEPT', 'METHODOLOGY', 'PROTOCOL',
    'OPERATING_SYSTEM', 'SOFT_SKILL', 'DOMAIN_CONCEPT',
    'CERTIFICATION', 'BUILD_TOOL', 'TESTING_TOOL',
    'MONITORING_TOOL', 'SECURITY_TOOL', 'DATA_TOOL',
    'DESIGN_TOOL', 'COLLABORATION_TOOL', 'MESSAGE_BROKER',
    'CONTAINER_RUNTIME', 'ORCHESTRATION_TOOL', 'WEB_SERVER',
    'ORM', 'HARDWARE_TOOL', 'IDE_TOOL', 'VERSION_CONTROL'
);

CREATE TYPE skill_nature AS ENUM (
    'TOOL', 'CONCEPT', 'PRACTICE', 'PLATFORM', 'LANGUAGE', 'CREDENTIAL'
);

CREATE TYPE skill_volatility AS ENUM (
    'STABLE', 'EVOLVING', 'EMERGING', 'DEPRECATED'
);

CREATE TYPE skill_lifespan AS ENUM (
    'EVERGREEN', 'MULTI_YEAR', 'SHORT_LIVED'
);

CREATE TYPE version_strategy AS ENUM (
    'MERGE_TO_PARENT', 'SEPARATE_ENTITY', 'NOT_APPLICABLE'
);

CREATE TYPE entity_source AS ENUM (
    'MANUAL_CURATION', 'AUTOMATED_DISCOVERY', 'USER_SUBMISSION'
);

CREATE TYPE alias_type AS ENUM (
    'CANONICAL', 'ABBREVIATION', 'VERSION', 'MISSPELLING',
    'REGIONAL_VARIANT', 'DEPRECATED_NAME', 'COLLOQUIAL',
    'FULL_NAME', 'ACRONYM'
);

CREATE TYPE match_strategy AS ENUM (
    'EXACT', 'CASE_INSENSITIVE', 'BOUNDARY_AWARE', 'REGEX'
);

CREATE TYPE relationship_type AS ENUM (
    'TRANSFERS_TO', 'REQUIRES', 'COMPETES_WITH', 'CO_EVOLVES_WITH'
);

CREATE TYPE relationship_direction AS ENUM (
    'FORWARD', 'REVERSE', 'BIDIRECTIONAL'
);

-- ─── MAIN TABLES ─────────────────────────────────────────────────────────────

CREATE TABLE canonical_skills (
    id              UUID PRIMARY KEY,
    display_name    VARCHAR(100)        NOT NULL,
    slug            VARCHAR(120)        NOT NULL UNIQUE,
    category        skill_category      NOT NULL,
    sub_category    VARCHAR(100),
    parent_skill_id UUID                REFERENCES canonical_skills(id) ON DELETE SET NULL,
    depth           SMALLINT            NOT NULL DEFAULT 0 CHECK (depth BETWEEN 0 AND 4),
    skill_nature    skill_nature        NOT NULL,
    volatility      skill_volatility    NOT NULL DEFAULT 'STABLE',
    is_extractable  BOOLEAN             NOT NULL DEFAULT TRUE,
    is_also_category BOOLEAN            NOT NULL DEFAULT FALSE,
    typical_lifespan skill_lifespan     NOT NULL DEFAULT 'EVERGREEN',
    version_parent_id UUID              REFERENCES canonical_skills(id) ON DELETE SET NULL,
    version_tag     VARCHAR(50),
    version_strategy version_strategy   NOT NULL DEFAULT 'NOT_APPLICABLE',
    source          entity_source       NOT NULL DEFAULT 'MANUAL_CURATION',
    confidence      NUMERIC(3,2)        NOT NULL DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
    created_at      TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ,

    -- Vector embedding of the display_name for semantic similarity search.
    -- 1536 dimensions = OpenAI text-embedding-3-small; adjust if using a different model.
    name_embedding  vector(1536)
);

-- Full-text search index on display_name
CREATE INDEX idx_skills_display_name_trgm ON canonical_skills
    USING gin (display_name gin_trgm_ops);

-- Category filtering
CREATE INDEX idx_skills_category ON canonical_skills (category);

-- Hierarchy traversal
CREATE INDEX idx_skills_parent ON canonical_skills (parent_skill_id)
    WHERE parent_skill_id IS NOT NULL;

-- HNSW index for fast approximate nearest-neighbor on skill name embeddings
CREATE INDEX idx_skills_name_embedding ON canonical_skills
    USING hnsw (name_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);


-- ─── SKILL ALIASES ───────────────────────────────────────────────────────────

CREATE TABLE skill_aliases (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    skill_id        UUID                NOT NULL REFERENCES canonical_skills(id) ON DELETE CASCADE,
    alias_text      VARCHAR(200)        NOT NULL,
    alias_type      alias_type          NOT NULL,
    match_strategy  match_strategy      NOT NULL DEFAULT 'CASE_INSENSITIVE',
    match_pattern   TEXT,               -- regex pattern when match_strategy = 'REGEX'
    is_primary      BOOLEAN             NOT NULL DEFAULT FALSE,
    region_affinity TEXT[],             -- ISO country codes, e.g. {'IN','US'}

    -- Lowercase form for fast exact / case-insensitive lookups (approach 1)
    alias_lower     VARCHAR(200) GENERATED ALWAYS AS (LOWER(alias_text)) STORED,

    -- Vector embedding of alias_text for semantic similarity search (approach 2).
    -- Catches cases like "angular.js" → "Angular", "React.js" → "React" etc.
    alias_embedding vector(1536),

    CONSTRAINT uq_skill_alias UNIQUE (skill_id, alias_text)
);

-- ── Approach 1: Direct match indexes ──
-- Exact case-insensitive lookup via the stored generated column
CREATE INDEX idx_aliases_lower ON skill_aliases (alias_lower);

-- Trigram index for fuzzy text search (typo tolerance)
CREATE INDEX idx_aliases_trgm ON skill_aliases
    USING gin (alias_lower gin_trgm_ops);

-- ── Approach 2: Vector similarity index ──
-- HNSW index for fast cosine similarity on alias embeddings
CREATE INDEX idx_aliases_embedding ON skill_aliases
    USING hnsw (alias_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Only one primary alias per skill
CREATE UNIQUE INDEX idx_aliases_primary ON skill_aliases (skill_id)
    WHERE is_primary = TRUE;

-- Fast alias → skill resolution
CREATE INDEX idx_aliases_skill_id ON skill_aliases (skill_id);


-- ─── SKILL TAGS ──────────────────────────────────────────────────────────────

CREATE TABLE skill_tags (
    skill_id    UUID        NOT NULL REFERENCES canonical_skills(id) ON DELETE CASCADE,
    tag         VARCHAR(80) NOT NULL,
    PRIMARY KEY (skill_id, tag)
);

CREATE INDEX idx_tags_tag ON skill_tags (tag);


-- ─── SKILL RELATIONSHIPS ─────────────────────────────────────────────────────

CREATE TABLE skill_relationships (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_skill_id     UUID                    NOT NULL REFERENCES canonical_skills(id) ON DELETE CASCADE,
    target_skill_id     UUID                    NOT NULL REFERENCES canonical_skills(id) ON DELETE CASCADE,
    relationship_type   relationship_type       NOT NULL,
    direction           relationship_direction  NOT NULL DEFAULT 'FORWARD',
    base_confidence     NUMERIC(3,2)            NOT NULL DEFAULT 0.5 CHECK (base_confidence BETWEEN 0 AND 1),
    context_modifiers   JSONB,                  -- array of {context_type, context_value, confidence_adj}
    rationale           TEXT,

    CONSTRAINT uq_skill_relationship UNIQUE (source_skill_id, target_skill_id, relationship_type),
    CONSTRAINT no_self_relationship CHECK (source_skill_id != target_skill_id)
);

CREATE INDEX idx_rel_source ON skill_relationships (source_skill_id);
CREATE INDEX idx_rel_target ON skill_relationships (target_skill_id);
CREATE INDEX idx_rel_type   ON skill_relationships (relationship_type);


-- ============================================================================
-- SEARCH FUNCTIONS
-- ============================================================================

-- ─── Approach 1: Direct Match ────────────────────────────────────────────────
-- Exact case-insensitive lookup against all aliases in the DB.
-- Returns the canonical skill for each matched alias.

CREATE OR REPLACE FUNCTION search_skills_direct(
    query_text TEXT
)
RETURNS TABLE (
    skill_id        UUID,
    display_name    VARCHAR(100),
    category        skill_category,
    matched_alias   VARCHAR(200),
    alias_type      alias_type,
    match_type      TEXT          -- 'exact' or 'trigram'
)
LANGUAGE sql STABLE AS $$
    -- First try exact case-insensitive match
    SELECT
        cs.id,
        cs.display_name,
        cs.category,
        sa.alias_text,
        sa.alias_type,
        'exact'::TEXT
    FROM skill_aliases sa
    JOIN canonical_skills cs ON cs.id = sa.skill_id
    WHERE sa.alias_lower = LOWER(TRIM(query_text))

    UNION ALL

    -- Fallback: trigram similarity for typo tolerance (threshold 0.3)
    SELECT
        cs.id,
        cs.display_name,
        cs.category,
        sa.alias_text,
        sa.alias_type,
        'trigram'::TEXT
    FROM skill_aliases sa
    JOIN canonical_skills cs ON cs.id = sa.skill_id
    WHERE sa.alias_lower % LOWER(TRIM(query_text))
      AND sa.alias_lower != LOWER(TRIM(query_text))  -- exclude already-matched exact hits
    ORDER BY similarity(sa.alias_lower, LOWER(TRIM(query_text))) DESC
    LIMIT 10;
$$;


-- ─── Approach 2: Vector Embedding Similarity Search ──────────────────────────
-- Finds skills whose alias embeddings are closest to the query embedding.
-- Handles cases like "angular.js" matching "Angular", "React.js" → "React".
-- The caller must pre-compute the query embedding and pass it in.

CREATE OR REPLACE FUNCTION search_skills_by_embedding(
    query_embedding vector(1536),
    similarity_threshold FLOAT DEFAULT 0.82,
    max_results INT DEFAULT 10
)
RETURNS TABLE (
    skill_id        UUID,
    display_name    VARCHAR(100),
    category        skill_category,
    matched_alias   VARCHAR(200),
    similarity      FLOAT
)
LANGUAGE sql STABLE AS $$
    SELECT
        cs.id,
        cs.display_name,
        cs.category,
        sa.alias_text,
        1 - (sa.alias_embedding <=> query_embedding) AS similarity
    FROM skill_aliases sa
    JOIN canonical_skills cs ON cs.id = sa.skill_id
    WHERE sa.alias_embedding IS NOT NULL
    ORDER BY sa.alias_embedding <=> query_embedding
    LIMIT max_results;
$$;

-- Wrapper that filters by threshold after retrieval
CREATE OR REPLACE FUNCTION search_skills_by_embedding_filtered(
    query_embedding vector(1536),
    similarity_threshold FLOAT DEFAULT 0.82,
    max_results INT DEFAULT 10
)
RETURNS TABLE (
    skill_id        UUID,
    display_name    VARCHAR(100),
    category        skill_category,
    matched_alias   VARCHAR(200),
    similarity      FLOAT
)
LANGUAGE sql STABLE AS $$
    SELECT * FROM search_skills_by_embedding(query_embedding, similarity_threshold, max_results)
    WHERE similarity >= similarity_threshold;
$$;


-- ─── Combined Search: Direct + Vector Fallback ──────────────────────────────
-- Tries direct match first; if no exact hit, falls back to vector similarity.
-- This is the recommended entry point for skill detection.

CREATE OR REPLACE FUNCTION detect_skill(
    query_text TEXT,
    query_embedding vector(1536) DEFAULT NULL,
    similarity_threshold FLOAT DEFAULT 0.82
)
RETURNS TABLE (
    skill_id        UUID,
    display_name    VARCHAR(100),
    category        skill_category,
    matched_alias   VARCHAR(200),
    match_method    TEXT,          -- 'exact', 'trigram', or 'vector'
    score           FLOAT
)
LANGUAGE plpgsql STABLE AS $$
BEGIN
    -- Step 1: Try direct match (exact + trigram)
    RETURN QUERY
    SELECT
        d.skill_id,
        d.display_name,
        d.category,
        d.matched_alias,
        d.match_type AS match_method,
        CASE
            WHEN d.match_type = 'exact' THEN 1.0::FLOAT
            ELSE similarity(LOWER(d.matched_alias), LOWER(TRIM(query_text)))::FLOAT
        END AS score
    FROM search_skills_direct(query_text) d;

    -- If direct match found something, we're done
    IF FOUND THEN
        RETURN;
    END IF;

    -- Step 2: Fall back to vector similarity (if embedding provided)
    IF query_embedding IS NOT NULL THEN
        RETURN QUERY
        SELECT
            v.skill_id,
            v.display_name,
            v.category,
            v.matched_alias,
            'vector'::TEXT AS match_method,
            v.similarity AS score
        FROM search_skills_by_embedding_filtered(
            query_embedding, similarity_threshold, 5
        ) v;
    END IF;
END;
$$;


-- ─── Batch Skill Detection ──────────────────────────────────────────────────
-- Detect multiple skills at once from an array of raw skill strings.

CREATE OR REPLACE FUNCTION detect_skills_batch(
    raw_skills TEXT[]
)
RETURNS TABLE (
    raw_input       TEXT,
    skill_id        UUID,
    display_name    VARCHAR(100),
    category        skill_category,
    matched_alias   VARCHAR(200),
    match_method    TEXT,
    score           FLOAT
)
LANGUAGE sql STABLE AS $$
    SELECT
        rs.raw,
        d.skill_id,
        d.display_name,
        d.category,
        d.matched_alias,
        d.match_type AS match_method,
        CASE
            WHEN d.match_type = 'exact' THEN 1.0::FLOAT
            ELSE similarity(LOWER(d.matched_alias), LOWER(TRIM(rs.raw)))::FLOAT
        END AS score
    FROM UNNEST(raw_skills) AS rs(raw)
    CROSS JOIN LATERAL search_skills_direct(rs.raw) d;
$$;


-- ============================================================================
-- UTILITY VIEWS
-- ============================================================================

-- Quick lookup: all aliases flattened with their canonical skill
CREATE VIEW v_alias_lookup AS
SELECT
    sa.alias_lower,
    sa.alias_text,
    sa.alias_type,
    sa.match_strategy,
    cs.id AS skill_id,
    cs.display_name,
    cs.category,
    cs.sub_category
FROM skill_aliases sa
JOIN canonical_skills cs ON cs.id = sa.skill_id;

-- Registry stats view
CREATE VIEW v_registry_stats AS
SELECT
    (SELECT COUNT(*) FROM canonical_skills) AS total_skills,
    (SELECT COUNT(*) FROM skill_aliases) AS total_aliases,
    (SELECT COUNT(*) FROM skill_relationships) AS total_relationships,
    (SELECT COUNT(*) FROM skill_aliases WHERE alias_embedding IS NOT NULL) AS aliases_with_embeddings,
    (SELECT COUNT(*) FROM canonical_skills WHERE name_embedding IS NOT NULL) AS skills_with_embeddings;
