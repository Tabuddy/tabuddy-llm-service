-- dev.canonical_skills definition

-- Drop table

-- DROP TABLE dev.canonical_skills;

CREATE TABLE dev.canonical_skills (
	id int8 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 9223372036854775807 START 1 CACHE 1 NO CYCLE) NOT NULL,
	slug varchar(120) NOT NULL,
	display_name varchar(100) NOT NULL,
	category_id int8 NOT NULL,
	sub_category_id int8 NULL,
	parent_skill_id int8 NULL,
	"depth" int2 DEFAULT 0 NOT NULL,
	"skill_nature" dev."skill_nature" NOT NULL,
	volatility dev."skill_volatility" DEFAULT 'STABLE'::dev.skill_volatility NOT NULL,
	is_extractable bool DEFAULT true NOT NULL,
	is_also_category bool DEFAULT false NOT NULL,
	typical_lifespan dev."skill_lifespan" DEFAULT 'EVERGREEN'::dev.skill_lifespan NOT NULL,
	version_parent_id int8 NULL,
	version_tag varchar(50) NULL,
	"version_strategy" dev."version_strategy" DEFAULT 'NOT_APPLICABLE'::dev.version_strategy NOT NULL,
	"source" dev."entity_source" DEFAULT 'MANUAL_CURATION'::dev.entity_source NOT NULL,
	confidence numeric(3, 2) DEFAULT 1.0 NOT NULL,
	created_at timestamptz DEFAULT now() NOT NULL,
	updated_at timestamptz NULL,
	name_embedding dev.vector NULL,
	CONSTRAINT canonical_skills_confidence_check CHECK (((confidence >= (0)::numeric) AND (confidence <= (1)::numeric))),
	CONSTRAINT canonical_skills_depth_check CHECK (((depth >= 0) AND (depth <= 4))),
	CONSTRAINT canonical_skills_pkey PRIMARY KEY (id),
	CONSTRAINT canonical_skills_slug_key UNIQUE (slug),
	CONSTRAINT canonical_skills_category_id_fkey FOREIGN KEY (category_id) REFERENCES dev.categories(id),
	CONSTRAINT canonical_skills_parent_skill_id_fkey FOREIGN KEY (parent_skill_id) REFERENCES dev.canonical_skills(id) ON DELETE SET NULL,
	CONSTRAINT canonical_skills_sub_category_id_fkey FOREIGN KEY (sub_category_id) REFERENCES dev.sub_categories(id),
	CONSTRAINT canonical_skills_version_parent_id_fkey FOREIGN KEY (version_parent_id) REFERENCES dev.canonical_skills(id) ON DELETE SET NULL
);
CREATE INDEX idx_skills_category ON dev.canonical_skills USING btree (category_id);
CREATE INDEX idx_skills_display_name_trgm ON dev.canonical_skills USING gin (display_name gin_trgm_ops);
CREATE INDEX idx_skills_name_embedding ON dev.canonical_skills USING hnsw (name_embedding dev.vector_cosine_ops) WITH (m='16', ef_construction='64');
CREATE INDEX idx_skills_parent ON dev.canonical_skills USING btree (parent_skill_id) WHERE (parent_skill_id IS NOT NULL);
CREATE INDEX idx_skills_sub_category ON dev.canonical_skills USING btree (sub_category_id);

-- dev.categories definition

-- Drop table

-- DROP TABLE dev.categories;

CREATE TABLE dev.categories (
	id int8 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 9223372036854775807 START 1 CACHE 1 NO CYCLE) NOT NULL,
	slug varchar(80) NOT NULL,
	display_name varchar(120) NOT NULL,
	description text NULL,
	created_at timestamptz DEFAULT now() NOT NULL,
	updated_at timestamptz NULL,
	name_embedding dev.vector NULL,
	CONSTRAINT categories_pkey PRIMARY KEY (id),
	CONSTRAINT categories_slug_key UNIQUE (slug)
);
CREATE INDEX idx_categories_name_embedding ON dev.categories USING hnsw (name_embedding dev.vector_cosine_ops) WITH (m='16', ef_construction='64');

-- dev.dimension_categories definition

-- Drop table

-- DROP TABLE dev.dimension_categories;

CREATE TABLE dev.dimension_categories (
	id int8 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 9223372036854775807 START 1 CACHE 1 NO CYCLE) NOT NULL,
	dimension_id int8 NOT NULL,
	category_id int8 NOT NULL,
	sub_category_id int8 NULL,
	rationale text NULL,
	CONSTRAINT dimension_categories_dimension_id_category_id_sub_category__key UNIQUE (dimension_id, category_id, sub_category_id),
	CONSTRAINT dimension_categories_pkey PRIMARY KEY (id),
	CONSTRAINT dimension_categories_category_id_fkey FOREIGN KEY (category_id) REFERENCES dev.categories(id),
	CONSTRAINT dimension_categories_dimension_id_fkey FOREIGN KEY (dimension_id) REFERENCES dev.dimensions(id) ON DELETE CASCADE,
	CONSTRAINT dimension_categories_sub_category_id_fkey FOREIGN KEY (sub_category_id) REFERENCES dev.sub_categories(id)
);
CREATE INDEX idx_dim_cats_dim ON dev.dimension_categories USING btree (dimension_id);

-- dev.dimension_skills definition

-- Drop table

-- DROP TABLE dev.dimension_skills;

CREATE TABLE dev.dimension_skills (
	dimension_id int8 NOT NULL,
	skill_id int8 NOT NULL,
	CONSTRAINT dimension_skills_pkey PRIMARY KEY (dimension_id, skill_id),
	CONSTRAINT dimension_skills_dimension_id_fkey FOREIGN KEY (dimension_id) REFERENCES dev.dimensions(id) ON DELETE CASCADE,
	CONSTRAINT dimension_skills_skill_id_fkey FOREIGN KEY (skill_id) REFERENCES dev.canonical_skills(id) ON DELETE CASCADE
);
CREATE INDEX idx_dimension_skills_skill ON dev.dimension_skills USING btree (skill_id);

-- dev.dimensions definition

-- Drop table

-- DROP TABLE dev.dimensions;

CREATE TABLE dev.dimensions (
	id int8 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 9223372036854775807 START 1 CACHE 1 NO CYCLE) NOT NULL,
	slug varchar(160) NOT NULL,
	display_name varchar(200) NOT NULL,
	rationale text NULL,
	difficulty_hint varchar(20) DEFAULT 'well_known'::character varying NOT NULL,
	"source" dev."entity_source" DEFAULT 'AUTOMATED_DISCOVERY'::dev.entity_source NOT NULL,
	created_at timestamptz DEFAULT now() NOT NULL,
	updated_at timestamptz NULL,
	CONSTRAINT dimensions_pkey PRIMARY KEY (id),
	CONSTRAINT dimensions_slug_key UNIQUE (slug)
);

-- dev.non_skills definition

-- Drop table

-- DROP TABLE dev.non_skills;

CREATE TABLE dev.non_skills (
	id int8 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 9223372036854775807 START 1 CACHE 1 NO CYCLE) NOT NULL,
	"name" text NULL
);

-- dev.role_dimensions definition

-- Drop table

-- DROP TABLE dev.role_dimensions;

CREATE TABLE dev.role_dimensions (
	role_id int8 NOT NULL,
	dimension_id int8 NOT NULL,
	CONSTRAINT role_dimensions_pkey PRIMARY KEY (role_id, dimension_id),
	CONSTRAINT role_dimensions_dimension_id_fkey FOREIGN KEY (dimension_id) REFERENCES dev.dimensions(id) ON DELETE CASCADE,
	CONSTRAINT role_dimensions_role_id_fkey FOREIGN KEY (role_id) REFERENCES dev.roles(id) ON DELETE CASCADE
);
CREATE INDEX idx_role_dimensions_dim ON dev.role_dimensions USING btree (dimension_id);

-- dev.roles definition

-- Drop table

-- DROP TABLE dev.roles;

CREATE TABLE dev.roles (
	id int8 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 9223372036854775807 START 1 CACHE 1 NO CYCLE) NOT NULL,
	slug varchar(120) NOT NULL,
	display_name varchar(160) NOT NULL,
	role_archetype text NULL,
	"source" dev."entity_source" DEFAULT 'AUTOMATED_DISCOVERY'::dev.entity_source NOT NULL,
	created_at timestamptz DEFAULT now() NOT NULL,
	updated_at timestamptz NULL,
	CONSTRAINT roles_pkey PRIMARY KEY (id),
	CONSTRAINT roles_slug_key UNIQUE (slug)
);
CREATE INDEX idx_roles_slug_trgm ON dev.roles USING gin (slug gin_trgm_ops);

-- dev.skill_aliases definition

-- Drop table

-- DROP TABLE dev.skill_aliases;

CREATE TABLE dev.skill_aliases (
	id int8 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 9223372036854775807 START 1 CACHE 1 NO CYCLE) NOT NULL,
	skill_id int8 NOT NULL,
	alias_text varchar(200) NOT NULL,
	"alias_type" dev."alias_type" NOT NULL,
	"match_strategy" dev."match_strategy" DEFAULT 'CASE_INSENSITIVE'::dev.match_strategy NOT NULL,
	match_pattern text NULL,
	is_primary bool DEFAULT false NOT NULL,
	region_affinity _text NULL,
	alias_lower varchar(200) GENERATED ALWAYS AS (lower(alias_text::text)) STORED NULL,
	alias_embedding dev.vector NULL,
	CONSTRAINT skill_aliases_pkey PRIMARY KEY (id),
	CONSTRAINT uq_skill_alias UNIQUE (skill_id, alias_text),
	CONSTRAINT skill_aliases_skill_id_fkey FOREIGN KEY (skill_id) REFERENCES dev.canonical_skills(id) ON DELETE CASCADE
);
CREATE INDEX idx_aliases_embedding ON dev.skill_aliases USING hnsw (alias_embedding dev.vector_cosine_ops) WITH (m='16', ef_construction='64');
CREATE INDEX idx_aliases_lower ON dev.skill_aliases USING btree (alias_lower);
CREATE UNIQUE INDEX idx_aliases_primary ON dev.skill_aliases USING btree (skill_id) WHERE (is_primary = true);
CREATE INDEX idx_aliases_skill_id ON dev.skill_aliases USING btree (skill_id);
CREATE INDEX idx_aliases_trgm ON dev.skill_aliases USING gin (alias_lower gin_trgm_ops);

-- dev.skill_relationships definition

-- Drop table

-- DROP TABLE dev.skill_relationships;

CREATE TABLE dev.skill_relationships (
	id int8 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 9223372036854775807 START 1 CACHE 1 NO CYCLE) NOT NULL,
	source_skill_id int8 NOT NULL,
	target_skill_id int8 NOT NULL,
	"relationship_type" dev."relationship_type" NOT NULL,
	direction dev."relationship_direction" DEFAULT 'FORWARD'::dev.relationship_direction NOT NULL,
	base_confidence numeric(3, 2) DEFAULT 0.5 NOT NULL,
	context_modifiers jsonb NULL,
	rationale text NULL,
	CONSTRAINT no_self_relationship CHECK ((source_skill_id <> target_skill_id)),
	CONSTRAINT skill_relationships_base_confidence_check CHECK (((base_confidence >= (0)::numeric) AND (base_confidence <= (1)::numeric))),
	CONSTRAINT skill_relationships_pkey PRIMARY KEY (id),
	CONSTRAINT uq_skill_relationship UNIQUE (source_skill_id, target_skill_id, relationship_type),
	CONSTRAINT skill_relationships_source_skill_id_fkey FOREIGN KEY (source_skill_id) REFERENCES dev.canonical_skills(id) ON DELETE CASCADE,
	CONSTRAINT skill_relationships_target_skill_id_fkey FOREIGN KEY (target_skill_id) REFERENCES dev.canonical_skills(id) ON DELETE CASCADE
);
CREATE INDEX idx_rel_source ON dev.skill_relationships USING btree (source_skill_id);
CREATE INDEX idx_rel_target ON dev.skill_relationships USING btree (target_skill_id);
CREATE INDEX idx_rel_type ON dev.skill_relationships USING btree (relationship_type);

-- dev.skill_tags definition

-- Drop table

-- DROP TABLE dev.skill_tags;

CREATE TABLE dev.skill_tags (
	skill_id int8 NOT NULL,
	tag varchar(80) NOT NULL,
	CONSTRAINT skill_tags_pkey PRIMARY KEY (skill_id, tag),
	CONSTRAINT skill_tags_skill_id_fkey FOREIGN KEY (skill_id) REFERENCES dev.canonical_skills(id) ON DELETE CASCADE
);
CREATE INDEX idx_tags_tag ON dev.skill_tags USING btree (tag);

-- dev.sub_categories definition

-- Drop table

-- DROP TABLE dev.sub_categories;

CREATE TABLE dev.sub_categories (
	id int8 GENERATED ALWAYS AS IDENTITY( INCREMENT BY 1 MINVALUE 1 MAXVALUE 9223372036854775807 START 1 CACHE 1 NO CYCLE) NOT NULL,
	category_id int8 NOT NULL,
	slug varchar(120) NOT NULL,
	display_name varchar(160) NOT NULL,
	description text NULL,
	created_at timestamptz DEFAULT now() NOT NULL,
	updated_at timestamptz NULL,
	name_embedding dev.vector NULL,
	CONSTRAINT sub_categories_pkey PRIMARY KEY (id),
	CONSTRAINT sub_categories_slug_key UNIQUE (slug),
	CONSTRAINT sub_categories_category_id_fkey FOREIGN KEY (category_id) REFERENCES dev.categories(id) ON DELETE CASCADE
);
CREATE INDEX idx_sub_categories_category ON dev.sub_categories USING btree (category_id);
CREATE INDEX idx_sub_categories_name_embedding ON dev.sub_categories USING hnsw (name_embedding dev.vector_cosine_ops) WITH (m='16', ef_construction='64');
