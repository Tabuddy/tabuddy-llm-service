-- ============================================================================
-- TAbuddy Canonical Skill Library — PostgreSQL Schema (v2: relational, ORM-managed)
-- Requires: PostgreSQL 15+ with pgvector and pg_trgm extensions
-- Managed by SQLAlchemy 2.0 + Alembic (see db/models.py, alembic/versions/).
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ─── Categories (replaces the old skill_category enum) ─────────────────────
CREATE TABLE categories (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    slug            VARCHAR(80)  NOT NULL UNIQUE,
    display_name    VARCHAR(120) NOT NULL,
    description     TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ,
    name_embedding  vector(1536)
);
CREATE INDEX idx_categories_name_embedding ON categories
    USING hnsw (name_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- ─── Sub-categories ────────────────────────────────────────────────────────
-- Sub-category slugs are unique GLOBALLY (not per-category) so the loader
-- can dedup with `ON CONFLICT (slug) DO NOTHING`.
CREATE TABLE sub_categories (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    category_id     BIGINT       NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    slug            VARCHAR(120) NOT NULL UNIQUE,
    display_name    VARCHAR(160) NOT NULL,
    description     TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ,
    name_embedding  vector(1536)
);
CREATE INDEX idx_sub_categories_category ON sub_categories (category_id);
CREATE INDEX idx_sub_categories_name_embedding ON sub_categories
    USING hnsw (name_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- ─── Other enums (small + stable) ──────────────────────────────────────────
CREATE TYPE skill_nature AS ENUM (
    'TOOL', 'CONCEPT', 'PRACTICE', 'PLATFORM', 'LANGUAGE', 'CREDENTIAL',
    'LIBRARY', 'FRAMEWORK', 'RUNTIME', 'PROTOCOL', 'STANDARD',
    'METHODOLOGY', 'CLOUD_SERVICE',
    'PATTERN', 'API', 'ALGORITHM'
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

-- ─── Canonical skills (now references categories/sub_categories by id) ─────
CREATE TABLE canonical_skills (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    slug              VARCHAR(120)        NOT NULL UNIQUE,
    display_name      VARCHAR(100)        NOT NULL,
    category_id       BIGINT              NOT NULL REFERENCES categories(id),
    sub_category_id   BIGINT              REFERENCES sub_categories(id),
    parent_skill_id   BIGINT              REFERENCES canonical_skills(id) ON DELETE SET NULL,
    depth             SMALLINT            NOT NULL DEFAULT 0 CHECK (depth BETWEEN 0 AND 4),
    skill_nature      skill_nature        NOT NULL,
    volatility        skill_volatility    NOT NULL DEFAULT 'STABLE',
    is_extractable    BOOLEAN             NOT NULL DEFAULT TRUE,
    is_also_category  BOOLEAN             NOT NULL DEFAULT FALSE,
    typical_lifespan  skill_lifespan      NOT NULL DEFAULT 'EVERGREEN',
    version_parent_id BIGINT              REFERENCES canonical_skills(id) ON DELETE SET NULL,
    version_tag       VARCHAR(50),
    version_strategy  version_strategy    NOT NULL DEFAULT 'NOT_APPLICABLE',
    source            entity_source       NOT NULL DEFAULT 'MANUAL_CURATION',
    confidence        NUMERIC(3,2)        NOT NULL DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
    created_at        TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ,
    name_embedding    vector(1536)
);
CREATE INDEX idx_skills_display_name_trgm ON canonical_skills USING gin (display_name gin_trgm_ops);
CREATE INDEX idx_skills_category ON canonical_skills (category_id);
CREATE INDEX idx_skills_sub_category ON canonical_skills (sub_category_id);
CREATE INDEX idx_skills_parent ON canonical_skills (parent_skill_id) WHERE parent_skill_id IS NOT NULL;
CREATE INDEX idx_skills_name_embedding ON canonical_skills
    USING hnsw (name_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- ─── Skill aliases ─────────────────────────────────────────────────────────
CREATE TABLE skill_aliases (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    skill_id        BIGINT              NOT NULL REFERENCES canonical_skills(id) ON DELETE CASCADE,
    alias_text      VARCHAR(200)        NOT NULL,
    alias_type      alias_type          NOT NULL,
    match_strategy  match_strategy      NOT NULL DEFAULT 'CASE_INSENSITIVE',
    match_pattern   TEXT,
    is_primary      BOOLEAN             NOT NULL DEFAULT FALSE,
    region_affinity TEXT[],
    alias_lower     VARCHAR(200) GENERATED ALWAYS AS (LOWER(alias_text)) STORED,
    alias_embedding vector(1536),
    CONSTRAINT uq_skill_alias UNIQUE (skill_id, alias_text)
);
CREATE INDEX idx_aliases_lower ON skill_aliases (alias_lower);
CREATE INDEX idx_aliases_trgm  ON skill_aliases USING gin (alias_lower gin_trgm_ops);
CREATE INDEX idx_aliases_embedding ON skill_aliases
    USING hnsw (alias_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE UNIQUE INDEX idx_aliases_primary ON skill_aliases (skill_id) WHERE is_primary = TRUE;
CREATE INDEX idx_aliases_skill_id ON skill_aliases (skill_id);

-- ─── Roles ─────────────────────────────────────────────────────────────────
CREATE TABLE roles (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    slug            VARCHAR(120)  NOT NULL UNIQUE,
    display_name    VARCHAR(160)  NOT NULL,
    role_archetype  TEXT,
    domain          VARCHAR(64)   NOT NULL DEFAULT 'Other',
    parent_role_id  BIGINT        NULL REFERENCES roles(id) ON DELETE SET NULL,
    is_branchable   BOOLEAN       NOT NULL DEFAULT FALSE,
    source          entity_source NOT NULL DEFAULT 'AUTOMATED_DISCOVERY',
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ
);
CREATE INDEX idx_roles_slug_trgm ON roles USING gin (slug gin_trgm_ops);
CREATE INDEX idx_roles_domain    ON roles (domain);
CREATE INDEX idx_roles_parent    ON roles (parent_role_id);

-- ─── Dimensions ────────────────────────────────────────────────────────────
CREATE TABLE dimensions (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    slug            VARCHAR(160)  NOT NULL UNIQUE,
    display_name    VARCHAR(200)  NOT NULL,
    rationale       TEXT,
    difficulty_hint VARCHAR(20)   NOT NULL DEFAULT 'well_known',
    source          entity_source NOT NULL DEFAULT 'AUTOMATED_DISCOVERY',
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ
);

-- ─── Role ↔ Dimension (many-to-many) ───────────────────────────────────────
CREATE TABLE role_dimensions (
    role_id      BIGINT NOT NULL REFERENCES roles(id)      ON DELETE CASCADE,
    dimension_id BIGINT NOT NULL REFERENCES dimensions(id) ON DELETE CASCADE,
    PRIMARY KEY (role_id, dimension_id)
);
CREATE INDEX idx_role_dimensions_dim ON role_dimensions (dimension_id);

-- ─── Dimension ↔ Skill (many-to-many) ──────────────────────────────────────
CREATE TABLE dimension_skills (
    dimension_id BIGINT NOT NULL REFERENCES dimensions(id)        ON DELETE CASCADE,
    skill_id     BIGINT NOT NULL REFERENCES canonical_skills(id)  ON DELETE CASCADE,
    PRIMARY KEY (dimension_id, skill_id)
);
CREATE INDEX idx_dimension_skills_skill ON dimension_skills (skill_id);

-- ─── Dimension ↔ (Category, SubCategory) — audit of categorizer output ─────
CREATE TABLE dimension_categories (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dimension_id    BIGINT NOT NULL REFERENCES dimensions(id)     ON DELETE CASCADE,
    category_id     BIGINT NOT NULL REFERENCES categories(id),
    sub_category_id BIGINT          REFERENCES sub_categories(id),
    rationale       TEXT,
    UNIQUE (dimension_id, category_id, sub_category_id)
);
CREATE INDEX idx_dim_cats_dim ON dimension_categories (dimension_id);

-- ─── Skill tags ────────────────────────────────────────────────────────────
CREATE TABLE skill_tags (
    skill_id BIGINT      NOT NULL REFERENCES canonical_skills(id) ON DELETE CASCADE,
    tag      VARCHAR(80) NOT NULL,
    PRIMARY KEY (skill_id, tag)
);
CREATE INDEX idx_tags_tag ON skill_tags (tag);

-- ─── Skill relationships ───────────────────────────────────────────────────
CREATE TABLE skill_relationships (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_skill_id     BIGINT                  NOT NULL REFERENCES canonical_skills(id) ON DELETE CASCADE,
    target_skill_id     BIGINT                  NOT NULL REFERENCES canonical_skills(id) ON DELETE CASCADE,
    relationship_type   relationship_type       NOT NULL,
    direction           relationship_direction  NOT NULL DEFAULT 'FORWARD',
    base_confidence     NUMERIC(3,2)            NOT NULL DEFAULT 0.5 CHECK (base_confidence BETWEEN 0 AND 1),
    context_modifiers   JSONB,
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

-- ─── Direct Match: exact case-insensitive + trigram fallback ──────────────
-- Implementation note: the trigram branch is defined in a CTE so its
-- ORDER BY can reference ``sa.alias_lower`` (only visible inside that
-- subquery) and LIMIT bounds only the fuzzy results, not the whole UNION.
CREATE OR REPLACE FUNCTION search_skills_direct(
    query_text TEXT
)
RETURNS TABLE (
    skill_id        BIGINT,
    display_name    VARCHAR(100),
    category_slug   VARCHAR(80),
    matched_alias   VARCHAR(200),
    alias_type      alias_type,
    match_type      TEXT
)
LANGUAGE sql STABLE AS $$
    WITH q AS (SELECT LOWER(TRIM(query_text)) AS qt),
    exact_hits AS (
        SELECT
            cs.id            AS skill_id,
            cs.display_name  AS display_name,
            c.slug           AS category_slug,
            sa.alias_text    AS matched_alias,
            sa.alias_type    AS alias_type,
            'exact'::TEXT    AS match_type
        FROM skill_aliases sa
        JOIN canonical_skills cs ON cs.id = sa.skill_id
        JOIN categories c        ON c.id  = cs.category_id
        CROSS JOIN q
        WHERE sa.alias_lower = q.qt
    ),
    trigram_hits AS (
        SELECT
            cs.id            AS skill_id,
            cs.display_name  AS display_name,
            c.slug           AS category_slug,
            sa.alias_text    AS matched_alias,
            sa.alias_type    AS alias_type,
            'trigram'::TEXT  AS match_type
        FROM skill_aliases sa
        JOIN canonical_skills cs ON cs.id = sa.skill_id
        JOIN categories c        ON c.id  = cs.category_id
        CROSS JOIN q
        WHERE sa.alias_lower % q.qt
          AND sa.alias_lower != q.qt
        ORDER BY similarity(sa.alias_lower, q.qt) DESC
        LIMIT 10
    )
    SELECT * FROM exact_hits
    UNION ALL
    SELECT * FROM trigram_hits;
$$;

-- ─── Vector Embedding Similarity ──────────────────────────────────────────
CREATE OR REPLACE FUNCTION search_skills_by_embedding(
    query_embedding vector(1536),
    similarity_threshold FLOAT DEFAULT 0.82,
    max_results INT DEFAULT 10
)
RETURNS TABLE (
    skill_id        BIGINT,
    display_name    VARCHAR(100),
    category_slug   VARCHAR(80),
    matched_alias   VARCHAR(200),
    similarity      FLOAT
)
LANGUAGE sql STABLE AS $$
    SELECT
        cs.id,
        cs.display_name,
        c.slug AS category_slug,
        sa.alias_text,
        1 - (sa.alias_embedding <=> query_embedding) AS similarity
    FROM skill_aliases sa
    JOIN canonical_skills cs ON cs.id = sa.skill_id
    JOIN categories c        ON c.id  = cs.category_id
    WHERE sa.alias_embedding IS NOT NULL
    ORDER BY sa.alias_embedding <=> query_embedding
    LIMIT max_results;
$$;

CREATE OR REPLACE FUNCTION search_skills_by_embedding_filtered(
    query_embedding vector(1536),
    similarity_threshold FLOAT DEFAULT 0.82,
    max_results INT DEFAULT 10
)
RETURNS TABLE (
    skill_id        BIGINT,
    display_name    VARCHAR(100),
    category_slug   VARCHAR(80),
    matched_alias   VARCHAR(200),
    similarity      FLOAT
)
LANGUAGE sql STABLE AS $$
    SELECT * FROM search_skills_by_embedding(query_embedding, similarity_threshold, max_results)
    WHERE similarity >= similarity_threshold;
$$;

-- ─── Combined: Direct + Vector Fallback ────────────────────────────────────
CREATE OR REPLACE FUNCTION detect_skill(
    query_text TEXT,
    query_embedding vector(1536) DEFAULT NULL,
    similarity_threshold FLOAT DEFAULT 0.82
)
RETURNS TABLE (
    skill_id        BIGINT,
    display_name    VARCHAR(100),
    category_slug   VARCHAR(80),
    matched_alias   VARCHAR(200),
    match_method    TEXT,
    score           FLOAT
)
LANGUAGE plpgsql STABLE AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.skill_id,
        d.display_name,
        d.category_slug,
        d.matched_alias,
        d.match_type AS match_method,
        CASE
            WHEN d.match_type = 'exact' THEN 1.0::FLOAT
            ELSE similarity(LOWER(d.matched_alias), LOWER(TRIM(query_text)))::FLOAT
        END AS score
    FROM search_skills_direct(query_text) d;

    IF FOUND THEN
        RETURN;
    END IF;

    IF query_embedding IS NOT NULL THEN
        RETURN QUERY
        SELECT
            v.skill_id,
            v.display_name,
            v.category_slug,
            v.matched_alias,
            'vector'::TEXT AS match_method,
            v.similarity AS score
        FROM search_skills_by_embedding_filtered(
            query_embedding, similarity_threshold, 5
        ) v;
    END IF;
END;
$$;

-- ─── Batch Skill Detection ────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION detect_skills_batch(
    raw_skills TEXT[]
)
RETURNS TABLE (
    raw_input       TEXT,
    skill_id        BIGINT,
    display_name    VARCHAR(100),
    category_slug   VARCHAR(80),
    matched_alias   VARCHAR(200),
    match_method    TEXT,
    score           FLOAT
)
LANGUAGE sql STABLE AS $$
    SELECT
        rs.raw,
        d.skill_id,
        d.display_name,
        d.category_slug,
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

CREATE VIEW v_alias_lookup AS
SELECT
    sa.alias_lower,
    sa.alias_text,
    sa.alias_type,
    sa.match_strategy,
    cs.id AS skill_id,
    cs.display_name,
    c.slug  AS category_slug,
    sc.slug AS sub_category_slug
FROM skill_aliases sa
JOIN canonical_skills cs ON cs.id = sa.skill_id
JOIN categories c        ON c.id  = cs.category_id
LEFT JOIN sub_categories sc ON sc.id = cs.sub_category_id;

CREATE VIEW v_registry_stats AS
SELECT
    (SELECT COUNT(*) FROM canonical_skills)    AS total_skills,
    (SELECT COUNT(*) FROM skill_aliases)       AS total_aliases,
    (SELECT COUNT(*) FROM skill_relationships) AS total_relationships,
    (SELECT COUNT(*) FROM categories)          AS total_categories,
    (SELECT COUNT(*) FROM sub_categories)      AS total_sub_categories,
    (SELECT COUNT(*) FROM roles)               AS total_roles,
    (SELECT COUNT(*) FROM dimensions)          AS total_dimensions,
    (SELECT COUNT(*) FROM skill_aliases     WHERE alias_embedding IS NOT NULL) AS aliases_with_embeddings,
    (SELECT COUNT(*) FROM canonical_skills  WHERE name_embedding  IS NOT NULL) AS skills_with_embeddings;

-- ============================================================================
-- JD PIPELINE HISTORY
-- One row per end-to-end run of the 3-API JD pipeline
-- (extract-from-jd → extract-details → final-role-output).
-- ============================================================================

CREATE TABLE jd_pipeline_runs (
    id                    UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    jd_text               TEXT         NOT NULL,
    status                TEXT         NOT NULL,
    api1_response         JSONB,
    api2_response         JSONB,
    api3_response         JSONB,
    chosen_role_display   TEXT,
    chosen_role_id        BIGINT       REFERENCES roles(id) ON DELETE SET NULL,
    final_skills_count    INTEGER,
    final_skills          TEXT[],
    jd_role_hint_display  TEXT,
    error_message         TEXT,
    duration_ms           INTEGER,
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_jd_runs_created_at ON jd_pipeline_runs (created_at DESC);
CREATE INDEX idx_jd_runs_status     ON jd_pipeline_runs (status);
CREATE INDEX idx_jd_runs_chosen_role ON jd_pipeline_runs (chosen_role_id) WHERE chosen_role_id IS NOT NULL;

CREATE TABLE jd_run_artifacts (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id          UUID         NOT NULL REFERENCES jd_pipeline_runs(id) ON DELETE CASCADE,
    artifact_kind   TEXT         NOT NULL,
    artifact_id     BIGINT,
    artifact_text   TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_jd_run_artifacts_run  ON jd_run_artifacts (run_id);
CREATE INDEX idx_jd_run_artifacts_kind ON jd_run_artifacts (run_id, artifact_kind);
