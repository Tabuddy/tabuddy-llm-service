"""Stage 4 typology — the closed 17-type list every skill is assigned to.

Per blueprint: Language, Library, Framework, Tool, Platform, Service,
Runtime, Datastore, Protocol, Standard, Format, Concept, Methodology,
Architecture, Domain, SoftSkill, Certification.

This typology is v3-specific and intentionally separate from the
existing ``skill_nature`` enum in ``db/schema.sql``. v3 stores
``TypedSkill`` output as JSONB on a v2_run_log row — Stage 8 / final-load
will eventually map these to skill_nature when writing to canonical_skills.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TYPOLOGY_VALUES: tuple[str, ...] = (
    "Language",
    "Library",
    "Framework",
    "Tool",
    "Platform",
    "Service",
    "Runtime",
    "Datastore",
    "Protocol",
    "Standard",
    "Format",
    "Concept",
    "Methodology",
    "Architecture",
    "Domain",
    "SoftSkill",
    "Certification",
)


SkillType = Literal[
    "Language",
    "Library",
    "Framework",
    "Tool",
    "Platform",
    "Service",
    "Runtime",
    "Datastore",
    "Protocol",
    "Standard",
    "Format",
    "Concept",
    "Methodology",
    "Architecture",
    "Domain",
    "SoftSkill",
    "Certification",
]


# Short description per type — rendered into the Stage 4 prompt as
# disambiguation anchors. Examples are deliberately concrete and follow
# the blueprint's disambiguation rules (Tool vs Framework: build with vs
# use as software; Platform vs Tool: hosted multi-tenant vs run-yourself).
TYPOLOGY_DESCRIPTIONS: dict[str, str] = {
    "Language":      "a programming, query, markup, or data-definition language. Examples: Python, Java, SQL, Bash, HTML.",
    "Library":       "a code package consumed by application code (you import it). Examples: NumPy, Hibernate, jQuery, requests.",
    "Framework":     "a structured codebase you build applications inside (it calls your code). Examples: Spring Boot, Django, React, Rails.",
    "Tool":          "software you operate as a user (you run it; it doesn't host you). Examples: Docker, Apache Airflow (self-hosted), Jenkins, Wireshark.",
    "Platform":      "a hosted, multi-tenant environment with APIs and managed services. Examples: AWS, Azure, GCP, Salesforce, Snowflake.",
    "Service":       "a hosted offering within a platform — a specific managed capability. Examples: AWS Lambda, Azure Cosmos DB, Google BigQuery, Stripe API.",
    "Runtime":       "an execution environment for code. Examples: Node.js, JVM, .NET CLR, CPython, Wasmtime.",
    "Datastore":     "a system that persists data. Examples: PostgreSQL, MongoDB, Cassandra, Redis (when used as primary), Elasticsearch.",
    "Protocol":      "a communication standard between systems. Examples: HTTP/2, gRPC, MQTT, OAuth 2.0, OpenID Connect.",
    "Standard":      "an industry- or body-defined specification. Examples: ISO 27001, GDPR, OWASP Top 10, PCI DSS.",
    "Format":        "a data structure or wire format. Examples: JSON, Avro, Protocol Buffers, Parquet, CSV.",
    "Concept":       "a named knowledge unit (theory, principle, pattern). Examples: CAP theorem, eventual consistency, idempotency, microservices.",
    "Methodology":   "a way of working / process. Examples: Agile, Scrum, TDD, Kanban, GitFlow.",
    "Architecture":  "a system-shape pattern. Examples: Microservices, Event-Driven Architecture, Hexagonal Architecture, CQRS.",
    "Domain":        "a vertical / industry / problem-space body of knowledge. Examples: Healthcare, FinTech, Ad Tech, Embedded Systems.",
    "SoftSkill":     "a non-technical interpersonal capability. Examples: Communication, Mentorship, Stakeholder Management, Code Review.",
    "Certification": "a formal third-party credential. Examples: AWS Certified Solutions Architect, CISSP, CKAD, PMP.",
}


_SUBTYPE_PATTERN = r"^[a-z][a-z0-9_]*$"


class TypedSkill(BaseModel):
    """Stage 4 output — one entry per skill the LLM has typed."""

    skill_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    type: SkillType
    subtype: str = Field(min_length=2, max_length=60, pattern=_SUBTYPE_PATTERN)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=20, max_length=400)
    alternatives_considered: list[str] = Field(default_factory=list, max_length=8)


class TypedSkillBatch(BaseModel):
    """Wrapper so ``call_json`` can validate the LLM's batch response.

    OpenAI's JSON-object response_format requires a top-level object, not
    a bare list — wrapping the typed skills under ``skills`` lets the LLM
    return ``{"skills": [...]}`` cleanly.
    """

    skills: list[TypedSkill] = Field(default_factory=list)
