"""Embed JD text and search similar rows in skill-library.jd_test_similarity(change the table name to where the jd embeddings are stored)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

_SKILL_SCHEMA = os.getenv("SKILL_LIBRARY_SCHEMA", "skill-library").strip()
_JD_TABLE = "jd_test_similarity"     # change the table name to where the jd embeddings are stored


def _quote_ident(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _pg_connect() -> psycopg2.extensions.connection:
    schema = _SKILL_SCHEMA
    database_url = os.getenv("DATABASE_URL", "") or os.getenv("SKILL_LIBRARY_PG_DSN", "")
    if database_url:
        return psycopg2.connect(database_url, options=f"-c search_path={schema},public")
    return psycopg2.connect(
        host=os.getenv("DB_HOST", ""),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", ""),
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASSWORD", ""),
        sslmode=os.getenv("DB_SSLMODE", "require"),
        connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "30")),
        options=f"-c search_path={schema},public",
    )


@dataclass
class SimilarJdMatch:
    id: str
    title: str | None
    similarity: float
    preview: str | None = None


def embed_jd_text(text: str) -> list[float] | None:
    """1536-d embedding for the given text."""
    from skill_matcher import _azure_embed_sync

    trimmed = (text or "").strip()
    if not trimmed:
        return None
    vecs = _azure_embed_sync([trimmed[:12_000]])
    if not vecs or not vecs[0]:
        return None
    return list(vecs[0])


def find_similar_jds(
    text: str,
    *,
    top_k: int = 3,
    threshold: float = 0.0,
) -> list[SimilarJdMatch]:
    """Embed text and return top cosine matches from jd_test_similarity."""
    vec = embed_jd_text(text)
    if vec is None:
        raise RuntimeError("Embedding failed — check AZURE_OPEN_AI_KEY")

    q_schema = _quote_ident(_SKILL_SCHEMA)
    table = f"{q_schema}.{_JD_TABLE}"
    vector = f"{q_schema}.vector(1536)"
    vec_str = "[" + ",".join(map(str, vec)) + "]"

    sql = f"""
        SELECT id::text,
               title,
               LEFT(body, 300) AS preview,
               ROUND((1 - (body_embedding <=> %s::{vector}))::numeric, 4) AS similarity
          FROM {table}
         WHERE body_embedding IS NOT NULL
           AND 1 - (body_embedding <=> %s::{vector}) >= %s
         ORDER BY body_embedding <=> %s::{vector}
         LIMIT %s
    """

    conn = _pg_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vec_str, vec_str, threshold, vec_str, top_k))
                cols = [c[0] for c in cur.description]
                rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()

    return [
        SimilarJdMatch(
            id=str(r["id"]),
            title=r.get("title"),
            similarity=float(r["similarity"]),
            preview=r.get("preview"),
        )
        for r in rows
    ]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    hits = find_similar_jds(args.text, top_k=args.top_k, threshold=args.threshold)
    print(
        json.dumps(
            [
                {
                    "id": h.id,
                    "title": h.title,
                    "similarity": h.similarity,
                    "preview": h.preview,
                }
                for h in hits
            ],
            indent=2,
        )
    )
