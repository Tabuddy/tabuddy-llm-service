"""
Backfill script: generate and store 384-dim embeddings (all-MiniLM-L6-v2) for:
  - dev.canonical_skills  → name_embedding  (text: display_name)
  - dev.skill_aliases     → alias_embedding (text: alias_text)
  - dev.dimensions        → name_embedding  (text: display_name + rationale)
  - dev.roles             → name_embedding  (text: display_name + role_archetype)

Run after:
  DROP INDEX IF EXISTS dev.idx_skills_name_embedding;
  DROP INDEX IF EXISTS dev.idx_aliases_embedding;
  UPDATE dev.canonical_skills SET name_embedding = NULL;
  UPDATE dev.skill_aliases    SET alias_embedding = NULL;

Then after this script finishes, recreate indexes:
  CREATE INDEX idx_skills_name_embedding ON dev.canonical_skills
    USING hnsw (name_embedding dev.vector_cosine_ops) WITH (m=16, ef_construction=64);
  CREATE INDEX idx_aliases_embedding ON dev.skill_aliases
    USING hnsw (alias_embedding dev.vector_cosine_ops) WITH (m=16, ef_construction=64);
"""

import os
import logging
from dotenv import load_dotenv
import psycopg2
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── DB config ─────────────────────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST",     "135.235.196.207")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_NAME     = os.getenv("DB_NAME",     "postgres")
DB_USER     = os.getenv("DB_USER",     "linkedin_scraper")
DB_PASSWORD = os.getenv("DB_PASSWORD", os.getenv("LINKEDIN_PASSWORD", "L!nked!nS3r@p3R"))
DB_SSLMODE  = os.getenv("DB_SSLMODE",  "require")

EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE  = 256


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode=DB_SSLMODE,
        connect_timeout=30,
        options="-c search_path=dev,public",
    )


def _vec_str(vec) -> str:
    return "[" + ",".join(map(str, vec.tolist())) + "]"


def _build_text(primary: str, secondary: str | None) -> str:
    primary = (primary or "").strip()
    secondary = (secondary or "").strip()
    return (primary + " " + secondary).strip() if secondary else primary


# ── Canonical Skills ──────────────────────────────────────────────────────────

def backfill_canonical_skills(conn, model: SentenceTransformer) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, display_name
            FROM dev.canonical_skills
            ORDER BY id
        """)
        rows = cur.fetchall()

    total = len(rows)
    if total == 0:
        logger.info("canonical_skills: table is empty, nothing to backfill.")
        return

    logger.info("canonical_skills: %d rows to embed.", total)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        ids   = [r[0] for r in batch]
        texts = [_build_text(r[1], None) for r in batch]

        vectors = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)

        with conn.cursor() as cur:
            for row_id, vec in zip(ids, vectors):
                cur.execute(
                    "UPDATE dev.canonical_skills SET name_embedding = %s::dev.vector WHERE id = %s",
                    (_vec_str(vec), row_id),
                )
        conn.commit()

        done = min(batch_start + BATCH_SIZE, total)
        logger.info("canonical_skills: %d / %d done.", done, total)

    logger.info("canonical_skills: backfill complete.")


# ── Skill Aliases ─────────────────────────────────────────────────────────────

def backfill_skill_aliases(conn, model: SentenceTransformer) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, alias_text
            FROM dev.skill_aliases
            ORDER BY id
        """)
        rows = cur.fetchall()

    total = len(rows)
    if total == 0:
        logger.info("skill_aliases: table is empty, nothing to backfill.")
        return

    logger.info("skill_aliases: %d rows to embed.", total)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        ids   = [r[0] for r in batch]
        texts = [_build_text(r[1], None) for r in batch]

        vectors = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)

        with conn.cursor() as cur:
            for row_id, vec in zip(ids, vectors):
                cur.execute(
                    "UPDATE dev.skill_aliases SET alias_embedding = %s::dev.vector WHERE id = %s",
                    (_vec_str(vec), row_id),
                )
        conn.commit()

        done = min(batch_start + BATCH_SIZE, total)
        logger.info("skill_aliases: %d / %d done.", done, total)

    logger.info("skill_aliases: backfill complete.")


# ── Dimensions ────────────────────────────────────────────────────────────────

def backfill_dimensions(conn, model: SentenceTransformer) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, display_name, rationale
            FROM dev.dimensions
            ORDER BY id
        """)
        rows = cur.fetchall()

    total = len(rows)
    if total == 0:
        logger.info("dimensions: table is empty, nothing to backfill.")
        return

    logger.info("dimensions: %d rows to embed.", total)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        ids   = [r[0] for r in batch]
        texts = [_build_text(r[1], r[2]) for r in batch]

        vectors = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)

        with conn.cursor() as cur:
            for row_id, vec in zip(ids, vectors):
                cur.execute(
                    "UPDATE dev.dimensions SET name_embedding = %s::dev.vector WHERE id = %s",
                    (_vec_str(vec), row_id),
                )
        conn.commit()

        done = min(batch_start + BATCH_SIZE, total)
        logger.info("dimensions: %d / %d done.", done, total)

    logger.info("dimensions: backfill complete.")


# ── Roles ─────────────────────────────────────────────────────────────────────

def backfill_roles(conn, model: SentenceTransformer) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, display_name, role_archetype
            FROM dev.roles
            ORDER BY id
        """)
        rows = cur.fetchall()

    total = len(rows)
    if total == 0:
        logger.info("roles: table is empty, nothing to backfill.")
        return

    logger.info("roles: %d rows to embed.", total)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        ids   = [r[0] for r in batch]
        texts = [_build_text(r[1], r[2]) for r in batch]

        vectors = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)

        with conn.cursor() as cur:
            for row_id, vec in zip(ids, vectors):
                cur.execute(
                    "UPDATE dev.roles SET name_embedding = %s::dev.vector WHERE id = %s",
                    (_vec_str(vec), row_id),
                )
        conn.commit()

        done = min(batch_start + BATCH_SIZE, total)
        logger.info("roles: %d / %d done.", done, total)

    logger.info("roles: backfill complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)

    logger.info("Connecting to DB: %s@%s:%s/%s", DB_USER, DB_HOST, DB_PORT, DB_NAME)
    conn = _connect()

    try:
        backfill_canonical_skills(conn, model)
        backfill_skill_aliases(conn, model)
        backfill_dimensions(conn, model)
        backfill_roles(conn, model)
    finally:
        conn.close()

    logger.info("All done.")
    logger.info(
        "REMINDER: Recreate HNSW indexes now:\n"
        "  CREATE INDEX idx_skills_name_embedding ON dev.canonical_skills\n"
        "    USING hnsw (name_embedding dev.vector_cosine_ops) WITH (m=16, ef_construction=64);\n"
        "  CREATE INDEX idx_aliases_embedding ON dev.skill_aliases\n"
        "    USING hnsw (alias_embedding dev.vector_cosine_ops) WITH (m=16, ef_construction=64);"
    )
