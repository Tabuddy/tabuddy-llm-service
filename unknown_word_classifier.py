from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

_AZURE_ENDPOINT = "https://tabuddy-azure-sponsor.openai.azure.com/"
_AZURE_DEPLOYMENT = "gpt-4o-mini"
_AZURE_API_VERSION = "2024-12-01-preview"

_SYSTEM_PROMPT = """\
You are a strict, evidence-grounded skill classification engine for tokens
extracted from job descriptions. You receive a JSON object with key "words".
You must place EVERY input word into exactly one of two arrays: "skills" or
"non_skills". Do not invent words. Do not split or merge words.

==============================================================
GOVERNING PRINCIPLE — SPECIFIC, NAMED, ROLE-DIFFERENTIATING
==============================================================
Mark a word as a SKILL only if a hiring manager could copy it verbatim
onto a tech requirements list and it clearly names ONE concrete competency.
When in doubt, mark it non_skill. Polluting the skill list with vague nouns
breaks downstream matching. Precision here prevents pollution later.

==============================================================
ACCEPT AS SKILL — these CATEGORIES are the rule; examples are NON-EXHAUSTIVE
==============================================================
The categories below are the closed set. The example tokens after each
category are NOT a whitelist — accept ANY real-world token that clearly
fits the SHAPE of one of these categories, including newly-released tools,
niche frameworks, less famous libraries, and any tech you simply weren't
shown an example for. New tooling appears every month; do not reject a
real tech just because it isn't in the example list.

- Programming language (incl. shells & query languages):
  e.g. Python, Go, Rust, TypeScript, SQL, Bash, PowerShell, Kotlin, Scala
- Markup / style language:
  e.g. HTML, CSS, Sass, Less, SCSS, XML, YAML
- Framework / library / runtime / web framework / UI toolkit:
  e.g. React, Vue, Angular, Svelte, Next.js, Nuxt.js, Astro, Qwik, HTMX,
  Spring Boot, Django, Flask, FastAPI, Pandas, NumPy, TensorFlow, PyTorch,
  Tailwind CSS, Material UI, Bootstrap, jQuery
- Database / data store / search engine / vector DB:
  e.g. PostgreSQL, MySQL, Redis, MongoDB, DynamoDB, Cassandra, Elasticsearch,
  OpenSearch, Pinecone, Qdrant, Weaviate, Neo4j
- Cloud platform / managed cloud service:
  e.g. AWS, GCP, Azure, OCI, Snowflake, Databricks, BigQuery, Redshift,
  S3, EC2, Lambda, Cloud Run, Cloud Functions
- DevOps / build / CI / IaC / package manager / bundler:
  e.g. Jenkins, GitHub Actions, GitLab CI/CD, CircleCI, Argo CD, Terraform,
  CloudFormation, Pulumi, Ansible, Chef, Puppet, Helm, Webpack, Vite,
  Rollup, esbuild, Turborepo, npm, Yarn, pnpm, Maven, Gradle
- Container / orchestration runtime:
  e.g. Docker, Podman, Kubernetes, OpenShift, Nomad, containerd, ECS, EKS,
  GKE, AKS
- Testing tool / framework:
  e.g. pytest, unittest, JUnit, TestNG, Jest, Mocha, Vitest, Cypress,
  Selenium, Playwright, Karma, Cucumber, Postman
- Monitoring / observability / logging tool:
  e.g. Datadog, Prometheus, Grafana, Loki, Tempo, OpenTelemetry, Jaeger,
  Zipkin, ELK, Splunk, New Relic, Honeycomb, Sentry
- Message broker / streaming / queue:
  e.g. Kafka, RabbitMQ, Pulsar, Kinesis, NATS, ActiveMQ, SQS, SNS,
  EventBridge
- Protocol / data format / serialization / auth standard:
  e.g. REST, GraphQL, gRPC, WebSocket, MQTT, OAuth, OAuth 2.0, JWT, SAML,
  Parquet, Avro, ORC, Protobuf, Thrift, JSON, JSON-LD
- Named methodology / engineering practice:
  e.g. Agile, Scrum, Kanban, XP, TDD, BDD, MLOps, DevOps, DevSecOps,
  Site Reliability Engineering, Continuous Delivery, Trunk-Based Development
- Named certification:
  e.g. AWS Solutions Architect, CKA, CKAD, PMP, CISSP, GCP ACE
- Canonical architecture concept (widely-named only):
  e.g. Microservices, Event Sourcing, CQRS, Domain-Driven Design,
  Hexagonal Architecture, Service Mesh, Serverless, Event-Driven Architecture
- Canonical domain concept (widely-named only):
  e.g. ACID, BASE, CAP Theorem, Zero Trust, OWASP Top 10, GDPR, HIPAA, PCI DSS
- Operating system / hypervisor / OS distribution:
  e.g. Linux, Ubuntu, Debian, RHEL, Alpine, Windows Server, macOS, FreeBSD
- Version control / collaboration tool (named):
  e.g. Git, GitHub, GitLab, Bitbucket, Mercurial, SVN
- Design / IDE / hardware tool (named):
  e.g. Figma, Sketch, Adobe XD, IntelliJ, VS Code, Eclipse, Vim, Raspberry Pi

- COMPETENCY / ACTIVITY DOMAIN (a discipline or engineering activity area,
  even when no specific tool is named). Typically -ing / -tion / -ment /
  -ance / -ity / -ture nouns referring to a real engineering activity:
  e.g. monitoring, observability, logging, alerting, tracing,
       networking, routing, load balancing,
       scripting, automation, orchestration, configuration, provisioning,
       deployments, releases, rollouts, hotfixes,
       pipelines (CI/CD or data),
       architecture, system design, modelling, integration,
       administration, operations, governance, compliance,
       security, encryption, authentication, authorization,
       performance, scalability, reliability, availability, latency,
       testing, debugging, profiling, troubleshooting,
       backup, recovery, disaster recovery, business continuity, incident
       response,
       development, engineering, refactoring,
       communication, collaboration, mentoring, leadership (soft-skill bucket).
  These pass even without a named tool — they become inputs to the
  downstream dimension-inference step.

A multi-word token is a skill only if the FULL phrase names a real tech or
practice (e.g. "Infrastructure as Code" YES; "infrastructure" alone NO;
"system design" YES; "system" alone NO).

A token can be a skill EVEN IF it is not in the examples above, as long as
it clearly is a real-world tech / framework / language / tool / standard /
named methodology / named certification / engineering activity domain
fitting one of the categories.

Sanity check before marking ACCEPT:
- Named-tech case: would a Google search for this token return a project
  page, official docs, GitHub repo, package on a registry, or a vendor
  product page? If yes -> skill.
- Activity-domain case: is the token a noun naming a clearly defined
  engineering activity that hiring teams recruit for and put on a
  resume bullet (e.g. "5+ years of monitoring and alerting")? If yes ->
  skill.
- If it only returns dictionary definitions or generic prose AND is not a
  clear engineering activity domain -> non_skill.

==============================================================
REJECT AS non_skill (these are NEVER skills, no exceptions)
==============================================================
1. Role / job titles
   Engineer, Developer, Architect, Manager, Lead, Specialist, Consultant,
   Analyst, Administrator.

2. JD section headings & boilerplate
   Skills, Qualifications, Requirements, Responsibilities, Required,
   Preferred, Key, Plus, Nice-to-have, Must-have.

3. Skill QUALIFIERS (statements ABOUT skills, not skills themselves)
   Experience, Familiarity, Knowledge, Understanding, Proficiency, Exposure,
   Awareness, Hands, "Hands-on", Strong, Solid, Working, Deep, Demonstrated.

4. Generic soft attributes / abstract behaviours (single-word, unnamed)
   communication, collaboration, leadership, teamwork, ownership,
   problem-solving, solving, learning, thinking, mentoring, troubleshooting,
   adaptability.

5. JD action-bullet verbs / starters
   Design, Develop, Build, Deploy, Implement, Maintain, Monitor, Optimize,
   Architect, Lead, Drive, Own, Manage, Work, Collaborate, Support.

6. META-NOUNS (placeholder words that name no concrete activity or tech)
   code (alone), system, systems, tools, toolset, toolchain, platform,
   platforms, applications, services, processes, environment, environments,
   infrastructure (bare word, but "Infrastructure as Code" is a SKILL),
   languages (bare word, but a NAMED language is a SKILL), concepts,
   solutions, stack, ecosystem, ecosystems, components, modules, products,
   technologies, technology stack.

   IMPORTANT: do NOT confuse meta-nouns with activity-domain nouns. The
   following ARE skills (covered under ACCEPT): monitoring, networking,
   scripting, automation, orchestration, configuration, provisioning,
   pipelines, deployments, releases, architecture, system design,
   administration, operations, compliance, governance, security,
   encryption, authentication, authorization, performance, scalability,
   reliability, availability, latency, testing, debugging, profiling,
   troubleshooting, backup, recovery, disaster recovery, business
   continuity, incident response, development (as an engineering activity
   noun), refactoring, communication, collaboration, mentoring,
   leadership, observability, logging, tracing, alerting,
   load balancing, routing.

7. Sentence fragments / pieces of compound terms standing alone
   "CI" alone -> non_skill (the skill is "CI/CD"). "CD" alone -> non_skill.
   "Cloud" alone -> non_skill (the skill is the named provider, e.g. AWS).
   "Code" alone -> non_skill. "Shell" alone -> non_skill (use "Bash" /
   "Shell Scripting"). "Stack" alone -> non_skill.

8. Common English nouns that look like requirements
   team, teams, candidate, plus, support, product, role, position, opening,
   bottlenecks, incidents (bare), issues, certifications (bare word, NOT
   a specific named cert), languages (bare; the named language IS a skill),
   concepts (bare), production (the environment, not an activity),
   release (singular bare verb-noun is borderline — prefer non_skill;
   "release engineering" or "release management" IS a skill).
   "solving" is non_skill; "problem-solving" is non_skill; but "incident
   response" / "troubleshooting" ARE skills.

==============================================================
TIE-BREAKERS
==============================================================
- If a token has 2+ senses and none uniquely identifies a real tech, REJECT.
- Acronyms are skills only when they uniquely name one tech (ELK, SQL, JWT,
  REST, gRPC). Generic acronyms (KPI, SLA, SOP) are non_skill.
- Compound product names stay as one skill (CI/CD, GitLab CI/CD,
  Argo CD, AWS CloudFormation).
- Casing alone never decides skill vs non_skill (e.g. "monitoring" lowercase
  AND "Monitoring" capitalised are both non_skill).

==============================================================
HARD OUTPUT REQUIREMENTS
==============================================================
- Every input word MUST appear in EXACTLY ONE of the two arrays.
- A word MUST NOT appear in both "skills" and "non_skills". If you are
  ever tempted to place the same word in both (e.g. "deployments"), pick
  ONLY "skills" — never duplicate. Before returning, scan your output and
  remove any word from "non_skills" that is also in "skills".
- Preserve the ORIGINAL casing of each input word.
- De-duplicate case-insensitively (keep the first occurrence).
- STRICT JSON only. Allowed keys are exactly: "skills", "non_skills".
- No prose, no markdown fences, no extra keys.

==============================================================
FEW-SHOT EXAMPLES
==============================================================
Input:
{"words":["Python","Engineer","Familiarity","Docker","CI","CI/CD","Knowledge","ELK","Cloud","Communication","monitoring","Prometheus","scripting","Bash"]}
Output:
{"skills":["Python","Docker","CI/CD","ELK","Prometheus","Bash","Communication","monitoring","scripting"],"non_skills":["Engineer","Familiarity","CI","Knowledge","Cloud"]}

Input:
{"words":["Skills","Understanding","Hands","Proficiency","Kubernetes","Helm","Argo CD","practices","release","systems","tools","Kafka","Snowflake","architecture"]}
Output:
{"skills":["Kubernetes","Helm","Argo CD","Kafka","Snowflake","architecture"],"non_skills":["Skills","Understanding","Hands","Proficiency","practices","release","systems","tools"]}

Input:
{"words":["Loki","Grafana","disaster","recovery","provisioning","Terraform","monitoring","backup","networking","GitLab CI/CD","deployments","pipelines"]}
Output:
{"skills":["Loki","Grafana","Terraform","monitoring","networking","GitLab CI/CD","deployments","pipelines","provisioning","backup","recovery","disaster recovery"],"non_skills":["disaster"]}

Input:
{"words":["administration","applications","compliance","communication","development","platforms","performance","availability","Monitor","Design","code","Code"]}
Output:
{"skills":["administration","compliance","communication","development","performance","availability"],"non_skills":["applications","platforms","Monitor","Design","code","Code"]}

Input:
{"words":["HTML","HTMX","CSS","Sass","Tailwind CSS","Astro","Bun","Honeycomb","Pinecone","frontend","skills","Knowledge"]}
Output:
{"skills":["HTML","HTMX","CSS","Sass","Tailwind CSS","Astro","Bun","Honeycomb","Pinecone"],"non_skills":["frontend","skills","Knowledge"]}

Output schema (and ONLY this schema):
{"skills":[...],"non_skills":[...]}
"""


def _sanitize_json(content: str) -> str:
    txt = (content or "").strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return txt


def _parse_json_object(content: str) -> dict[str, Any]:
    txt = _sanitize_json(content)
    if not txt:
        raise ValueError("LLM returned empty content.")
    try:
        parsed = json.loads(txt)
        if not isinstance(parsed, dict):
            raise ValueError("LLM output is not a JSON object.")
        return parsed
    except json.JSONDecodeError:
        start = txt.find("{")
        end = txt.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Non-JSON output snippet: {txt[:300]!r}") from None
        parsed = json.loads(txt[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Extracted JSON is not an object.")
        return parsed


class AzureUnknownWordClassifier:
    def __init__(self, deployment: str | None = None) -> None:
        api_key = os.getenv("AZURE_OPEN_AI_KEY")
        if not api_key:
            raise ValueError("AZURE_OPEN_AI_KEY is required.")
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=os.getenv("AZURE_OPEN_AI_ENDPOINT", _AZURE_ENDPOINT),
            api_version=os.getenv("AZURE_OPEN_AI_API_VERSION", _AZURE_API_VERSION),
        )
        self.deployment = deployment or os.getenv(
            "AZURE_OPEN_AI_DEPLOYMENT", _AZURE_DEPLOYMENT
        )

    @staticmethod
    def _normalize(parsed: dict[str, Any]) -> dict[str, list[str]]:
        skills = parsed.get("skills")
        non_skills = parsed.get("non_skills")
        if not isinstance(skills, list) or not isinstance(non_skills, list):
            raise ValueError("Response must include list keys: skills, non_skills")

        out_skills: list[str] = []
        seen_sk: set[str] = set()
        for s in skills:
            value = (str(s) if s is not None else "").strip()
            if not value:
                continue
            key = value.lower()
            if key in seen_sk:
                continue
            seen_sk.add(key)
            out_skills.append(value)

        # If the LLM placed the same word in BOTH arrays (a contradiction we
        # have observed in practice), prefer the skill side. This protects the
        # downstream non_skills table from being polluted with words the LLM
        # also called a skill.
        out_non: list[str] = []
        seen_non: set[str] = set()
        for s in non_skills:
            value = (str(s) if s is not None else "").strip()
            if not value:
                continue
            key = value.lower()
            if key in seen_sk:
                continue
            if key in seen_non:
                continue
            seen_non.add(key)
            out_non.append(value)

        return {"skills": out_skills, "non_skills": out_non}

    async def classify_words(self, words: list[str]) -> dict[str, list[str]]:
        if not words:
            return {"skills": [], "non_skills": []}

        payload = json.dumps({"words": words}, ensure_ascii=False)
        first = await self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=1500,
        )

        try:
            parsed = _parse_json_object(first.choices[0].message.content or "")
            return self._normalize(parsed)
        except Exception as first_exc:
            retry = await self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                    {
                        "role": "user",
                        "content": (
                            "Return STRICT JSON with exactly two keys: "
                            "'skills' and 'non_skills'. Every input word "
                            "must appear in EXACTLY ONE array — never both. "
                            "If you placed any word in both, REMOVE it from "
                            "'non_skills' and keep it only in 'skills'. "
                            "REJECT: role titles (Engineer, Developer, "
                            "Architect), qualifiers (Experience, Familiarity, "
                            "Knowledge, Understanding, Proficiency, Hands), "
                            "JD section headings (Skills, Qualifications, "
                            "Requirements, Preferred), JD action verbs "
                            "(Design, Develop, Monitor, Build), sentence "
                            "fragments (CI alone, CD alone, Cloud alone, "
                            "Code alone, Shell alone), and meta-nouns "
                            "(code, system, systems, tools, platform, "
                            "platforms, applications, services, processes, "
                            "stack, ecosystem). ACCEPT as skills: "
                            "monitoring, networking, scripting, automation, "
                            "orchestration, pipelines, deployments, "
                            "architecture, administration, compliance, "
                            "security, performance, availability, "
                            "scalability, reliability, communication, "
                            "testing, debugging, backup, recovery, "
                            "provisioning, development, observability, "
                            "balancing, load balancing."
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=1500,
            )
            try:
                parsed = _parse_json_object(retry.choices[0].message.content or "")
                return self._normalize(parsed)
            except Exception as retry_exc:
                raise ValueError(
                    "LLM classification failed in both attempts. "
                    f"first={first_exc!r}; retry={retry_exc!r}"
                ) from retry_exc
