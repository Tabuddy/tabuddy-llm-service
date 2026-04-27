import os
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional
from urllib.parse import urlparse

import psycopg2
import psycopg2.pool
from opensearchpy import OpenSearch


@dataclass
class Config:
    pg_host: str     = os.getenv("DB_HOST",     "135.235.196.207")
    pg_port: int     = int(os.getenv("DB_PORT", "5432"))
    pg_database: str = os.getenv("DB_NAME",     "postgres")
    pg_user: str     = os.getenv("DB_USER",     "linkedin_scraper")
    pg_password: str = os.getenv("DB_PASSWORD", os.getenv("LINKEDIN_PASSWORD", "L!nked!nS3r@p3R"))
    opensearch_url: str = os.getenv(
        "OPENSEARCH_URL", os.getenv("OPENSEARCH_NODE_URL", "https://135.235.196.207:9200")
    )
    opensearch_index: str    = os.getenv("OPENSEARCH_SKILLS_INDEX", "canonical_skills")
    opensearch_username: str = os.getenv("OPENSEARCH_USERNAME", os.getenv("OPENSEARCH_USER", "admin"))
    opensearch_password: str = os.getenv(
        "OPENSEARCH_PASSWORD", os.getenv("OPENSEARCH_PASS", "Link3diN$c6ap3rOp3nS3a6ch")
    )
    fuzzy_threshold: float  = 0.45
    vector_threshold: float = 0.75
    embed_model_name: str   = "all-MiniLM-L6-v2"


@dataclass
class SkillMatch:
    input_skill: str
    match_type: str
    canonical_id: Optional[str] = None
    canonical_name: Optional[str] = None
    canonical_slug: Optional[str] = None
    category: Optional[str] = None
    confidence: float = 0.0

    def matched(self) -> bool:
        return self.match_type != "unmatched"

    def to_dict(self) -> dict:
        return {
            "input_skill":    self.input_skill,
            "match_type":     self.match_type,
            "canonical_id":   self.canonical_id,
            "canonical_name": self.canonical_name,
            "canonical_slug": self.canonical_slug,
            "category":       self.category,
            "confidence":     self.confidence,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def process_jd(jd_text: str, config: Config = None) -> dict:
    """
    Full pipeline: JD text → SkillNer extraction → canonical DB matching (exact/fuzzy/vector)
                   + spaCy noun/propn/OOV extraction with unknown-word filtering.

    Returns:
        {
            "skills":        [str, ...]  # canonical_name strings, confidence >= 0.60, deduped
            "unknown_words": [str, ...]  # spaCy words not in canonical DB or extracted skill list
        }
    """
    _load_skillner()
    spacy_words = _extract_spacy_words(jd_text)
    extracted   = _extract_skills_from_jd(jd_text)

    with SkillMatcher(config) as matcher:
        results       = matcher.match_many(extracted)
        db_known      = matcher.known_terms()

    skills = list(dict.fromkeys(
        r.canonical_name
        for r in results
        if r.matched() and r.confidence >= 0.60
    ))

    extracted_lower = {s.lower() for s in extracted}
    unknown_words = [
        w for w in spacy_words
        if w.lower() not in db_known and w.lower() not in extracted_lower
    ]

    return {"skills": skills, "unknown_words": unknown_words}


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — MODEL LOADING
# Lazy-loaded globals so importing this module does not trigger heavy downloads.
# ─────────────────────────────────────────────────────────────────────────────

_nlp             = None
_skill_extractor = None

_AMBIGUOUS_SKILLS = ["Go", "R", "C", "C++", "C#", "Scala", "Rust", "Swift", "Julia"]
_AMBIGUOUS_PATTERN = re.compile(
    r"(?:experience in|proficient in|knowledge of|expertise in|skilled in|"
    r"using|with|in|and|or)\s+(" + "|".join(re.escape(s) for s in _AMBIGUOUS_SKILLS) + r")\b",
    re.IGNORECASE,
)


def _load_skillner():
    global _nlp, _skill_extractor
    if _skill_extractor is None:
        import spacy
        from spacy.matcher import PhraseMatcher
        from skillNer.general_params import SKILL_DB
        from skillNer.skill_extractor_class import SkillExtractor
        _nlp = spacy.load("en_core_web_lg")
        _skill_extractor = SkillExtractor(_nlp, SKILL_DB, PhraseMatcher)
    return _skill_extractor


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — SPACY NOISE FILTER
# Keeps only meaningful tokens: nouns, proper nouns, and OOV (unknown) words.
# Strips stop words, punctuation, whitespace, and tokens shorter than 2 chars.
# OOV catches new/niche tech terms (e.g. "LangChain", "Mojo") not in spaCy's
# 685k-word vocabulary of en_core_web_lg.
# ─────────────────────────────────────────────────────────────────────────────

def _extract_spacy_words(jd_text: str) -> list[str]:
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_lg")
    doc   = _nlp(jd_text)
    words = set()
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        text = token.text.strip()
        if len(text) < 2 or token.like_num:
            continue
        if token.pos_ in ("NOUN", "PROPN") or token.is_oov:
            words.add(text)
    return sorted(words)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — SKILLNER EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _recover_ambiguous(jd_text: str) -> set[str]:
    found = set()
    for match in _AMBIGUOUS_PATTERN.finditer(jd_text):
        skill = match.group(1)
        canonical = next(
            (s for s in _AMBIGUOUS_SKILLS if s.lower() == skill.lower()), skill.title()
        )
        found.add(canonical)
    return found


def _extract_skills_from_jd(jd_text: str) -> list[str]:
    extractor   = _load_skillner()
    annotations = extractor.annotate(jd_text)
    skills: set[str] = set()
    for m in annotations["results"]["full_matches"]:
        skills.add(m["doc_node_value"].strip().title())
    for m in annotations["results"]["ngram_scored"]:
        if m["score"] >= 1:
            skills.add(m["doc_node_value"].strip().title())
    skills.update(_recover_ambiguous(jd_text))
    return list(skills)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — THREE-STAGE MATCHER
# Stage 1: Exact   → PostgreSQL alias_lower (indexed B-tree)
# Stage 2: Fuzzy   → PostgreSQL pg_trgm  |  Python difflib fallback
# Stage 3: Vector  → OpenSearch k-NN cosine on name_embedding
# ─────────────────────────────────────────────────────────────────────────────

class SkillMatcher:
    def __init__(self, config: Config = None):
        self.config  = config or Config()
        self._pg_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self._os_client: Optional[OpenSearch] = None
        self._embed_model = None
        self._pg_trgm: Optional[bool] = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._pg_pool:
            self._pg_pool.closeall()

    def _pg(self) -> psycopg2.pool.ThreadedConnectionPool:
        if self._pg_pool is None:
            self._pg_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1, maxconn=5,
                host=self.config.pg_host,
                port=self.config.pg_port,
                dbname=self.config.pg_database,
                user=self.config.pg_user,
                password=self.config.pg_password,
                sslmode="require",
                connect_timeout=30,
            )
        return self._pg_pool

    def _os(self) -> OpenSearch:
        if self._os_client is None:
            parsed = urlparse(self.config.opensearch_url)
            self._os_client = OpenSearch(
                hosts=[{"host": parsed.hostname, "port": parsed.port or 9200}],
                http_auth=(self.config.opensearch_username, self.config.opensearch_password),
                use_ssl=parsed.scheme == "https",
                verify_certs=False,
                ssl_show_warn=False,
            )
        return self._os_client

    def _embedder(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(self.config.embed_model_name)
        return self._embed_model

    def _batch_exact(self, skills: list[str]) -> dict[str, SkillMatch]:
        pool = self._pg()
        conn = pool.getconn()
        results: dict[str, SkillMatch] = {}
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT lower(input_skill),
                           cs.id::text,
                           cs.display_name,
                           cs.slug,
                           cs.category::text
                    FROM unnest(%s::text[]) AS input_skill
                    JOIN dev.skill_aliases sa ON sa.alias_lower = lower(input_skill)
                    JOIN dev.canonical_skills cs ON cs.id = sa.skill_id
                """, (skills,))
                for input_lower, cid, display_name, slug, category in cur.fetchall():
                    results[input_lower] = SkillMatch(
                        input_skill=input_lower,
                        match_type="exact",
                        canonical_id=cid,
                        canonical_name=display_name,
                        canonical_slug=slug,
                        category=category,
                        confidence=1.0,
                    )
        finally:
            pool.putconn(conn)
        return results

    def _batch_fuzzy(self, skills: list[str]) -> dict[str, SkillMatch]:
        pool = self._pg()
        conn = pool.getconn()
        results: dict[str, SkillMatch] = {}
        try:
            if self._pg_trgm is not False:
                try:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT q.skill,
                                   cs.id::text,
                                   cs.display_name,
                                   cs.slug,
                                   cs.category::text,
                                   best.sim
                            FROM unnest(%s::text[]) AS q(skill)
                            CROSS JOIN LATERAL (
                                SELECT sa.skill_id,
                                       MAX(similarity(sa.alias_lower::text, lower(q.skill)::text)) AS sim
                                FROM dev.skill_aliases sa
                                WHERE similarity(sa.alias_lower::text, lower(q.skill)::text) > %s
                                GROUP BY sa.skill_id
                                ORDER BY sim DESC
                                LIMIT 1
                            ) AS best
                            JOIN dev.canonical_skills cs ON cs.id = best.skill_id
                        """, (skills, self.config.fuzzy_threshold))
                        for skill, cid, display_name, slug, category, sim in cur.fetchall():
                            results[skill.lower()] = SkillMatch(
                                input_skill=skill,
                                match_type="fuzzy",
                                canonical_id=cid,
                                canonical_name=display_name,
                                canonical_slug=slug,
                                category=category,
                                confidence=round(float(sim), 4),
                            )
                    self._pg_trgm = True
                except psycopg2.Error as e:
                    if e.pgcode == "42883":
                        conn.rollback()
                        self._pg_trgm = False
                        results = self._batch_fuzzy_python(skills, conn)
                    else:
                        raise
            else:
                results = self._batch_fuzzy_python(skills, conn)
        finally:
            pool.putconn(conn)
        return results

    def _batch_fuzzy_python(self, skills: list[str], conn) -> dict[str, SkillMatch]:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT sa.alias_lower, sa.skill_id::text,
                       cs.display_name, cs.slug, cs.category::text
                FROM dev.skill_aliases sa
                JOIN dev.canonical_skills cs ON cs.id = sa.skill_id
            """)
            all_aliases = cur.fetchall()
        results: dict[str, SkillMatch] = {}
        for skill in skills:
            skill_lower = skill.lower()
            best_sim, best_row = 0.0, None
            for alias_lower, skill_id, display_name, slug, category in all_aliases:
                sim = SequenceMatcher(None, skill_lower, alias_lower).ratio()
                if sim > best_sim:
                    best_sim, best_row = sim, (skill_id, display_name, slug, category)
            if best_row and best_sim > self.config.fuzzy_threshold:
                results[skill_lower] = SkillMatch(
                    input_skill=skill,
                    match_type="fuzzy",
                    canonical_id=best_row[0],
                    canonical_name=best_row[1],
                    canonical_slug=best_row[2],
                    category=best_row[3],
                    confidence=round(best_sim, 4),
                )
        return results

    def _batch_vector(self, skills: list[str]) -> dict[str, SkillMatch]:
        if not skills:
            return {}
        vectors = self._embedder().encode(skills, batch_size=64, show_progress_bar=False)
        msearch_body = []
        for vec in vectors:
            msearch_body.append({"index": self.config.opensearch_index})
            msearch_body.append({
                "size": 1,
                "query": {"knn": {"name_embedding": {"vector": vec.tolist(), "k": 1}}},
                "_source": ["id", "display_name", "slug", "category"],
            })
        responses = self._os().msearch(body=msearch_body)["responses"]
        results: dict[str, SkillMatch] = {}
        for skill, resp in zip(skills, responses):
            hits = resp.get("hits", {}).get("hits", [])
            if not hits:
                continue
            hit   = hits[0]
            score = float(hit["_score"])
            if score < self.config.vector_threshold:
                continue
            src = hit["_source"]
            results[skill.lower()] = SkillMatch(
                input_skill=skill,
                match_type="vector",
                canonical_id=src.get("id"),
                canonical_name=src.get("display_name"),
                canonical_slug=src.get("slug"),
                category=src.get("category"),
                confidence=round(score, 4),
            )
        return results

    def known_terms(self) -> set[str]:
        pool = self._pg()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT alias_lower FROM dev.skill_aliases
                    UNION
                    SELECT lower(display_name) FROM dev.canonical_skills
                """)
                return {row[0] for row in cur.fetchall()}
        finally:
            pool.putconn(conn)

    def match_many(self, skills: list[str]) -> list[SkillMatch]:
        if not skills:
            return []
        resolved: dict[str, SkillMatch] = {}
        pending = list(skills)

        exact = self._batch_exact(pending)
        resolved.update(exact)
        pending = [s for s in pending if s.lower() not in resolved]

        if pending:
            fuzzy = self._batch_fuzzy(pending)
            resolved.update(fuzzy)
            pending = [s for s in pending if s.lower() not in resolved]

        if pending:
            vector = self._batch_vector(pending)
            resolved.update(vector)

        output: list[SkillMatch] = []
        for skill in skills:
            m = resolved.get(skill.lower())
            if m:
                m.input_skill = skill
                output.append(m)
            else:
                output.append(SkillMatch(input_skill=skill, match_type="unmatched"))
        return output




jd = """
Job Title: Sr. Java Backend Developer

Requirements:
- Java 8, 17 and Spring Boot
- RestAPI, XML and JSON
- Maven, Gitlab, Jenkins, Kubernetes
- JUnit, Mockito
- Oracle, PLSQL
- Docker, Agile and Scrum
- Experience with amazon web service is a plus
- Familiarity with angularx or ReactJS for frontend collaboration
- Good understanding of k8s deployments
"""

result = process_jd(jd)
print(json.dumps(result, indent=2))
