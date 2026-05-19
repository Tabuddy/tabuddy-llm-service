-- v3_additions_stage4_stage5.sql
-- Stage 4 (Decision Routing) + Stage 5 (Post-Classification Updates)
-- storage for the JD role-classifier pipeline.
--
-- IMPORTANT on prod: pgvector lives in the "skill-library" schema, not "dev".
-- Before running this file on prod, set:
--     SET search_path TO "skill-library", dev, public;
-- so the vector type and HNSW operator class resolve correctly. See the
-- prod_pgvector_schema memory entry for the full background.

-- 1. Per-role KRA centroid (D-1 rolling mean over R&R embeddings of
-- classified JDs). NULL until the first confident classification.
CREATE TABLE IF NOT EXISTS role_kra_centroids (
    role_id              BIGINT PRIMARY KEY REFERENCES roles(id) ON DELETE CASCADE,
    centroid_embedding   vector(1536),
    n_jds_classified     INTEGER NOT NULL DEFAULT 0,
    last_jd_run_id       UUID REFERENCES jd_pipeline_runs(id) ON DELETE SET NULL,
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_role_kra_centroids_embedding
    ON role_kra_centroids USING hnsw (centroid_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);


-- 2. Alias collision audit log — fires on Case B (skill+KRA agree, alias
-- mismatch) and Case D (LLM2 resolves alias-vs-KRA conflict).
CREATE TABLE IF NOT EXISTS alias_collision_log (
    id                   BIGSERIAL PRIMARY KEY,
    jd_run_id            UUID REFERENCES jd_pipeline_runs(id) ON DELETE CASCADE,
    case_type            CHAR(1) NOT NULL CHECK (case_type IN ('B', 'D')),
    role_name_input      TEXT NOT NULL,
    alias_top_role_id    BIGINT REFERENCES roles(id) ON DELETE SET NULL,
    kra_top_role_id      BIGINT REFERENCES roles(id) ON DELETE SET NULL,
    skill_top_role_id    BIGINT REFERENCES roles(id) ON DELETE SET NULL,
    resolution_role_id   BIGINT REFERENCES roles(id) ON DELETE SET NULL,
    llm2_reasoning       TEXT,
    detected_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alias_collision_log_run
    ON alias_collision_log(jd_run_id);
CREATE INDEX IF NOT EXISTS idx_alias_collision_log_kra
    ON alias_collision_log(kra_top_role_id);


-- 3. New-role candidate queue — Case E (low KRA / small margin), unresolved
-- Case D, unresolved Case F.
CREATE TABLE IF NOT EXISTS new_role_candidate_queue (
    id                   BIGSERIAL PRIMARY KEY,
    jd_run_id            UUID REFERENCES jd_pipeline_runs(id) ON DELETE CASCADE,
    case_type            CHAR(1) NOT NULL CHECK (case_type IN ('D', 'E', 'F')),
    role_name_input      TEXT NOT NULL,
    r_and_r_text         TEXT NOT NULL,
    top_kra_score        REAL,
    kra_margin           REAL,
    candidate_role_ids   BIGINT[],
    reason               TEXT NOT NULL,
    status               TEXT NOT NULL DEFAULT 'pending'
                              CHECK (status IN ('pending', 'promoted', 'rejected')),
    resolved_role_id     BIGINT REFERENCES roles(id) ON DELETE SET NULL,
    resolved_at          TIMESTAMPTZ,
    resolved_by          TEXT,
    notes                TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_new_role_queue_status
    ON new_role_candidate_queue(status, created_at);


-- 4. New-skill candidate queue — captures skills extracted from JDs that
-- did NOT match any canonical_skill / skill_alias, attached to the
-- confidently-classified role for admin promotion later.
CREATE TABLE IF NOT EXISTS new_skill_candidate_queue (
    id                   BIGSERIAL PRIMARY KEY,
    jd_run_id            UUID REFERENCES jd_pipeline_runs(id) ON DELETE CASCADE,
    role_id              BIGINT REFERENCES roles(id) ON DELETE SET NULL,
    skill_name           TEXT NOT NULL,
    is_primary           BOOLEAN NOT NULL DEFAULT false,
    status               TEXT NOT NULL DEFAULT 'pending'
                              CHECK (status IN ('pending', 'promoted', 'rejected')),
    resolved_canonical_id BIGINT REFERENCES canonical_skills(id) ON DELETE SET NULL,
    resolved_at          TIMESTAMPTZ,
    resolved_by          TEXT,
    notes                TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (jd_run_id, role_id, skill_name)
);

CREATE INDEX IF NOT EXISTS idx_new_skill_queue_status
    ON new_skill_candidate_queue(status, created_at);
CREATE INDEX IF NOT EXISTS idx_new_skill_queue_role
    ON new_skill_candidate_queue(role_id);


-- 5. New-KRA candidate queue — captures JD R&R blocks whose top KRA match
-- against the classified role was weak (< NEW_KRA_THRESHOLD), suggesting
-- the responsibility is novel for that role.
CREATE TABLE IF NOT EXISTS new_kra_candidate_queue (
    id                   BIGSERIAL PRIMARY KEY,
    jd_run_id            UUID REFERENCES jd_pipeline_runs(id) ON DELETE CASCADE,
    role_id              BIGINT NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    r_and_r_text         TEXT NOT NULL,
    best_kra_similarity  REAL,
    status               TEXT NOT NULL DEFAULT 'pending'
                              CHECK (status IN ('pending', 'promoted', 'rejected')),
    resolved_kra_id      BIGINT REFERENCES role_kras(id) ON DELETE SET NULL,
    resolved_at          TIMESTAMPTZ,
    resolved_by          TEXT,
    notes                TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_new_kra_queue_status
    ON new_kra_candidate_queue(status, created_at);
CREATE INDEX IF NOT EXISTS idx_new_kra_queue_role
    ON new_kra_candidate_queue(role_id);
