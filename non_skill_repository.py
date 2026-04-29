from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)


def _to_strings(words: Iterable[str]) -> list[str]:
    return [str(w) for w in words if w is not None]


def _to_lower(words: Iterable[str]) -> list[str]:
    return [str(w).lower() for w in words if w is not None]


class NonSkillRepository:
    def __init__(self) -> None:
        self.database_url = os.getenv("DATABASE_URL", "")
        self.db_host = os.getenv("DB_HOST", "")
        self.db_port = int(os.getenv("DB_PORT", "5432"))
        self.db_name = os.getenv("DB_NAME", "")
        self.db_user = os.getenv("DB_USER", "")
        self.db_password = os.getenv("DB_PASSWORD", "")

        self.schema = os.getenv("DB_SCHEMA", "dev")
        self.table = os.getenv("DB_NON_SKILLS_TABLE", "non_skills")

        if not self.database_url and not (
            self.db_host and self.db_name and self.db_user and self.db_password
        ):
            raise ValueError(
                "Provide DATABASE_URL or DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD."
            )

    def _connect(self):
        if self.database_url:
            return psycopg2.connect(self.database_url)
        return psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
            sslmode=os.getenv("DB_SSLMODE", "require"),
            connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "30")),
        )

        # Returns a set of known non-skills
    def get_known_non_skills(self, words: Iterable[str]) -> set[str]:
        values = _to_lower(words)
        if not values:
            return set()

        query = sql.SQL(
            "SELECT lower(name) FROM {}.{} WHERE lower(name) = ANY(%s)"
        ).format(sql.Identifier(self.schema), sql.Identifier(self.table))

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (values,))
                return {row[0] for row in cur.fetchall()}

    # Filters out known non-skills from a list of words
    def filter_non_skills(self, words: Iterable[str]) -> list[str]:
        original = _to_strings(words)
        if not original:
            return []

        known = self.get_known_non_skills(original)
        return [w for w in original if w.lower() not in known]

    # Adds new non-skills to the database
    def add_non_skills(self, words: Iterable[str]) -> list[str]:
        values = _to_lower(words)
        if not values:
            return []

        insert_query = sql.SQL(
            """
            INSERT INTO {}.{} (name)
            SELECT v.name
            FROM unnest(%s::text[]) AS v(name)
            WHERE NOT EXISTS (
                SELECT 1 FROM {}.{} t WHERE lower(t.name) = lower(v.name)
            )
            """
        ).format(
            sql.Identifier(self.schema),
            sql.Identifier(self.table),
            sql.Identifier(self.schema),
            sql.Identifier(self.table),
        )

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(insert_query, (values,))
            conn.commit()
        return values

    def remove_non_skills(self, words: Iterable[str]) -> int:
        values = _to_lower(words)
        if not values:
            return 0

        delete_query = sql.SQL(
            "DELETE FROM {}.{} WHERE lower(name) = ANY(%s)"
        ).format(sql.Identifier(self.schema), sql.Identifier(self.table))

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(delete_query, (values,))
                deleted = cur.rowcount or 0
            conn.commit()
        return int(deleted)
