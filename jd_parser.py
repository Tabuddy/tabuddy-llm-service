"""JD Parser — converts raw Job Description text into structured meta-tag atoms.

Pipeline:
  1. SetFit 3-tier classification of the JD
  2. Single LLM call to extract structured atoms using the JD atom prompt
  3. Deterministic normalization into JDProfile
"""

from __future__ import annotations

import json
import logging
import os
import re
from openai import AsyncAzureOpenAI

from ranking_models import (
    JDAtom,
    JDCapabilityAtom,
    JDProfile,
    JDScaleAtom,
    LayerPrediction,
    LayeredClassification,
    TierClassification,
    TierPrediction,
)
import setfit_classifier as clf
import prompt_rescue

logger = logging.getLogger(__name__)

_AZURE_ENDPOINT = "https://tabuddy-azure-sponsor.openai.azure.com/"
_AZURE_DEPLOYMENT = "gpt-4o-mini"
_AZURE_API_VERSION = "2024-12-01-preview"

_client: AsyncAzureOpenAI | None = None


def _get_client() -> AsyncAzureOpenAI | None:
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("AZURE_OPEN_AI_KEY")
    if not api_key:
        return None
    _client = AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=_AZURE_ENDPOINT,
        api_version=_AZURE_API_VERSION,
    )
    return _client


# ── JD Atom Extraction Prompt ──────────────────────────────────────────────────
_JD_ATOM_PROMPT = """\
You are an Expert Data Architect and Recruitment Domain Specialist.
Your task is to parse a Job Description into contextual JSON atoms for algorithmic resume ranking.

The purpose is to convert an unstructured Job Description into a set of high-signal JSON atoms that represent:
role intent, contextual capability, scale, impact, and operating constraints.

Extract ONLY high-confidence signals explicitly stated or strongly implied.
Do NOT fabricate values, create placeholders, or invent missing information.

OUTPUT RULES:
• Output must be a JSON array of atom objects.
• Include only atoms that were extracted (evidence-backed).
• Each atom MUST have: atom_type, raw_text, normalized_entity.
• Capability atoms must also have: capability_type, ownership_level, intensity_level, classification (PRIMARY/SECONDARY), bundle_id.
• Scale atoms add: value.
• Do NOT wrap output in markdown fences.

CONTEXTUAL EXTRACTION GUIDELINES (be as precise as the JD allows):

1. ROLE INTENT — What is this hire fundamentally for?
   Capture: role_family (e.g., "Backend Engineering"), role_objective (the core mission),
   role_scope (individual contributor vs lead/manager), delivery_mode.

2. CAPABILITY ATOMS — What must this person actually DO?
   For each distinct responsibility, create a core_capability or secondary_capability atom.
   - Extract the actual technology/domain, not generic phrases.
   - Distinguish between building applications (app engineering) vs managing infrastructure.
   - Assign bundle_ids (B1, B2...) per responsibility cluster.

3. SCALE & IMPACT — What magnitude of work is expected?
   - scale_team: team size; scale_volume: traffic/users/data scale
   - impact_measurable_outcome: expected business impact
   - If JD mentions "startup", "fast-paced" → capture as intensity_level.

4. EXPERIENCE — Be precise about months vs years.
   - "6+ months" → capture verbatim; months = fractional years (6 months = 0.5)
   - "0-1 years" or "fresher" → min_years = 0.0, expected_seniority = "fresher"
   - Internships/apprenticeships often specify months — capture those.

5. CONTEXT & CONSTRAINTS — Working environment signals.
   - industry_vertical / industry_sub_vertical
   - process_methodology, environment_operating_environment
   - compliance_context, collaboration patterns

Bundle_id: assign "B1", "B2"... per responsibility block. Global atoms have no bundle_id.
classification: PRIMARY for core drivers, SECONDARY for supporting capabilities.
ownership_level: "Primary Owner" | "Shared Owner" | "Support"
intensity_level: "High" | "Medium" | "Low"

Return ONLY the JSON array. Example:
[
  {"atom_type": "role_family", "raw_text": "Senior Software Engineer", "normalized_entity": "Backend Engineering"},
  {"atom_type": "role_objective", "raw_text": "build and scale our checkout microservice", "normalized_entity": "Checkout Platform Development"},
  {"atom_type": "core_capability", "raw_text": "design and implement RESTful APIs in Python", "normalized_entity": "REST API Development",
   "capability_type": "Technical", "ownership_level": "Primary Owner", "intensity_level": "High",
   "classification": "PRIMARY", "bundle_id": "B1"},
  {"atom_type": "scale_team", "raw_text": "team of 12 engineers", "normalized_entity": "Engineering Team", "value": "12", "bundle_id": "B1"}
]
"""


def _build_default_tier_classification() -> TierClassification:
    return TierClassification(
        tier1=TierPrediction(label="Unknown", score=0.0),
        hierarchy_path="Unknown",
        final_label="Unknown",
        low_confidence=True,
    )


# ── LLM Cross-Check using Prompt Rescue ───────────────────────────────────────────────
async def _crosscheck_tier_classification(
    jd_text: str,
    setfit_result: TierClassification,
) -> TierClassification:
    """Cross-check SetFit tier classification using prompt rescue engine.

    Fires when SetFit confidence is below the pillar rescue threshold.
    Uses the structured context store for higher-quality corrections.
    """
    # Trust very high confidence predictions
    if (setfit_result.tier1.score > 0.92
            and setfit_result.tier2
            and setfit_result.tier2.score > 0.92):
        return setfit_result

    # Build SetFit scores dict for the rescue engine
    # (We use tier2 score as the primary signal since that's where misclassifications occur)
    t2_score = setfit_result.tier2.score if setfit_result.tier2 else setfit_result.tier1.score
    if not prompt_rescue.needs_rescue("pillar", t2_score):
        return setfit_result

    # Map current tier labels to pillar names for the scores dict
    _TIER2_TO_PILLAR = {
        "App_Engineering": "Application Engineering",
        "Data_Intelligence": "Data & Intelligence",
        "Infra_Cloud": "Infrastructure & Cloud",
        "Product_Design": "Product & Design",
        "Cyber_Security": "Cyber Security",
    }
    setfit_scores: dict[str, float] = {}
    if setfit_result.tier2:
        setfit_scores[_TIER2_TO_PILLAR.get(
            setfit_result.tier2.label, setfit_result.tier2.label)] = t2_score

    rescue_result = await prompt_rescue.rescue_classification(
        stage="pillar",
        setfit_scores=setfit_scores,
        text=jd_text,
    )

    if rescue_result.get("confidence_source") != "llm_rescue":
        return setfit_result

    # Map pillar name back to tier2 label
    _PILLAR_TO_TIER2 = {v: k for k, v in _TIER2_TO_PILLAR.items()}
    rescued_pillar = rescue_result["value"]
    corrected_t2 = _PILLAR_TO_TIER2.get(rescued_pillar)

    if not corrected_t2 or corrected_t2 == (setfit_result.tier2.label if setfit_result.tier2 else ""):
        return setfit_result

    logger.info(
        "⚡ Prompt rescue overrides JD classification: %s → %s",
        setfit_result.hierarchy_path, corrected_t2,
    )
    corrected = TierClassification(
        tier1=TierPrediction(label="Digital", score=0.90),
        tier2=TierPrediction(label=corrected_t2, score=0.85),
        hierarchy_path=f"Digital > {corrected_t2}",
        final_label=corrected_t2,
        low_confidence=False,
    )
    return corrected


async def _build_jd_layered_classification(
    jd_text: str,
    tier_class: TierClassification,
) -> LayeredClassification:
    """Build a LayeredClassification for a JD using the SetFit result + prompt rescue.

    JDs typically describe a single target role, so compatible_layers is limited
    to cases where the JD explicitly describes multi-stack requirements.
    """
    # Use the setfit_classifier's classify_as_layered for the base mapping
    lc = clf.classify_as_layered(jd_text[:2000])

    # Override the pillar/layer if the cross-checked tier_class differs
    _TIER2_TO_PILLAR = {
        "App_Engineering": "Application Engineering",
        "Data_Intelligence": "Data & Intelligence",
        "Infra_Cloud": "Infrastructure & Cloud",
        "Product_Design": "Product & Design",
        "Cyber_Security": "Cyber Security",
    }
    if tier_class.tier2 and tier_class.tier2.label in _TIER2_TO_PILLAR:
        corrected_pillar = _TIER2_TO_PILLAR[tier_class.tier2.label]
        if corrected_pillar != lc.pillar.label:
            lc.pillar = LayerPrediction(
                label=corrected_pillar,
                score=tier_class.tier2.score,
                confidence_source="llm_rescue",
            )
            # Rebuild hierarchy_path
            parts = [corrected_pillar]
            if lc.layer:
                parts.append(lc.layer.label)
            if lc.activity:
                parts.append(lc.activity.label)
            lc.hierarchy_path = " > ".join(parts)

    # Trigger layer-level rescue if the layer confidence is low
    if lc.layer and prompt_rescue.needs_rescue("layer", lc.layer.score):
        layer_scores: dict[str, float] = {lc.layer.label: lc.layer.score}
        rescue = await prompt_rescue.rescue_classification(
            stage="layer",
            setfit_scores=layer_scores,
            text=jd_text,
            resolved_context={"pillar": lc.pillar.label},
        )
        if rescue.get("confidence_source") == "llm_rescue":
            rescued_layer = rescue["value"]
            lc.layer = LayerPrediction(
                label=rescued_layer, score=rescue["score"], confidence_source="llm_rescue"
            )
            # Update hierarchy_path
            parts = [lc.pillar.label, rescued_layer]
            if lc.activity:
                parts.append(lc.activity.label)
            lc.hierarchy_path = " > ".join(parts)

    return lc


# ── Tech Keyword Extraction ────────────────────────────────────────────────────
_COMMON_TECH_TERMS = {
    # Frontend
    "react", "reactjs", "react.js", "vue", "vue.js", "angular", "svelte",
    "next.js", "nextjs", "html", "css", "tailwind", "sass", "bootstrap",
    "jquery", "webpack", "vite", "babel",
    # Backend & Fullstack
    "node", "nodejs", "node.js", "express", "fastapi", "django", "flask",
    "spring boot", "spring", "java", "python", "golang", "rust", "ruby",
    "php", "swift", "kotlin", "csharp", "c#", ".net", "asp.net",
    "microservices", "restful", "rest", "grpc", "graphql", "websocket",
    "serverless", "lambd", "api", "backend", "frontend", "fullstack",
    "typescript", "javascript", "es6", "babel",
    # Mobile
    "ios", "android", "react native", "flutter", "swift", "kotlin",
    "xcode", "android studio", "mobile", "app",
    # DevOps & Cloud
    "docker", "kubernetes", "k8s", "docker-compose", "helm",
    "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
    "aws", "azure", "gcp", "google cloud", "cloud", "iaas", "paas", "saas",
    "ci/cd", "pipelines", "infrastructure", "orchestration", "monitoring",
    "observability", "logging", "tracing", "prometheus", "grafana",
    "elk", "elastic", "stack", "kibana", "datadog", "new relic",
    # Databases
    "postgresql", "postgres", "mysql", "mariadb", "mongodb", "redis",
    "cassandra", "mssql", "sql server", "oracle", "elasticsearch",
    "dynamodb", "cosmosdb", "firestore", "neo4j", "graph database",
    "rdbms", "nosql", "sql", "database", "db", "data storage",
    # Data Engineering & Analytics
    "big data", "hadoop", "spark", "databricks", "airflow", "luigi", "prefect",
    "dbt", "data pipeline", "etl", "elt", "data warehouse", "data lake",
    "data mesh", "data fabric", "kafka", "confluent", "pulsar", "rabbitmq",
    "sqoop", "flink", "kinesis", "stream processing", "batch processing",
    "pandas", "numpy", "scipy", "scikit-learn", "sklearn", "tensorflow",
    "pytorch", "keras", "mlflow", "kubeflow", "feast", "hopsworks",
    "feature store", "model deployment", "model serving",
    "knowledge graph", "rdf", "owl", "sparql", "graphql", "triple store",
    "vector", "embeddings", "vector database", "pinecone", "weaviate", "milvus",
    "qdrant", "pgvector", "ann", "approximate nearest neighbor",
    "hdfs", "s3", "azure blob", "cloud storage", "object storage",
    "data ingestion", "data extraction", "data transformation", "data loading",
    "data quality", "data governance", "data catalog", "metadata",
    # Testing & QA
    "selenium", "cypress", "playwright", "jest", "mocha", "junit",
    "pytest", "testng", "cucumber", "bdd", "tdd", "integration test",
    "unit test", "e2e", "end-to-end", "mocking", "stubbing",
    # Version Control & Collaboration
    "git", "github", "gitlab", "bitbucket", "azure devops", "jira",
    "confluence", "slack", "microsoft teams", "agile", "scrum", "kanban",
    "waterfall", "devops", "sre", "site reliability", "incident response",
    # Security & Compliance
    "oauth", "jwt", "saml", "ldap", "kerberos", "encryption", "TLS",
    "SSL", "HTTPS", "firewall", "vpc", "subnet", "vpn", "rbac", "abac",
    "pci", "hipaa", "gdpr", "compliance", "audit", "penetration test",
    "vulnerability", "scanning", " secrets management", "key vault",
}


def _extract_experience_requirements(jd_text: str) -> tuple[float | None, float | None, str | None]:
    """Extract years/months of experience and seniority level from JD text.

    Returns (min_years, max_years, seniority_label) for use in scoring penalties.
    Supports both year-based ("5+ years") and month-based ("6 months") patterns.
    Month-based requirements are converted to fractional years (e.g., 6 months = 0.5).
    """
    text = jd_text.lower()
    min_years: float | None = None
    max_years: float | None = None
    seniority: str | None = None

    # Seniority keywords (order matters — check most specific first)
    seniority_map = [
        (("principal engineer", "principle", "staff engineer", "staff "), "principal"),
        (("engineering manager", "tech lead", "team lead"), "senior"),
        (("senior",), "senior"),
        (("lead",), "senior"),
        (("sde-3", "sde 3", "sde3"), "senior"),
        (("sde-2", "sde 2", "sde2", "mid.level", "mid level", "intermediate"), "mid"),
        (("sde-1", "sde 1", "sde1", "junior", "software engineer i"), "junior"),
        (("entry.level", "fresher", "graduate"), "fresher"),
        (("intern",), "fresher"),
    ]
    for patterns, label in seniority_map:
        if any(re.search(p, text) for p in patterns):
            seniority = label
            break

    # === YEAR-BASED PATTERNS ===
    # Order: range → plus → specific → generic (first match wins)

    # Pattern: "5-10 years" / "3 to 5 years" / "5–10 years"
    year_range = re.search(
        r"(\d+(?:\.\d+)?)\s*[-\u2013\u2014to]+\s*(\d+(?:\.\d+)?)\s*years?", text
    )
    if year_range:
        min_years = float(year_range.group(1))
        max_years = float(year_range.group(2))
    else:
        # Pattern: "7+ years"
        plus_match = re.search(r"(\d+(?:\.\d+)?)\s*\+\s*years?", text)
        if plus_match:
            min_years = float(plus_match.group(1))
        else:
            # Pattern: "minimum X years" / "at least X years" / "X years minimum"
            for pat in [
                r"(?:minimum|at least)\s+(\d+(?:\.\d+)?)\s*\+?\s*year",
                r"(\d+(?:\.\d+)?)\s*years?\s*(?:minimum|required)",
                r"(?:minimum|at least)\s+(\d+(?:\.\d+)?)\s*year",
            ]:
                m = re.search(pat, text)
                if m:
                    min_years = float(m.group(1))
                    break

    # === MONTH-BASED PATTERNS ===
    # (run before the generic "X years required" patterns to avoid conflicts)

    # Pattern: "0-12 months", "6-18 months"
    months_range = re.search(
        r"(\d+(?:\.\d+)?)\s*[-\u2013\u2014to]+\s*(\d+(?:\.\d+)?)\s*months?", text
    )
    if months_range:
        min_years = float(months_range.group(1)) / 12.0
        max_years = float(months_range.group(2)) / 12.0
    else:
        # Pattern: "6+ months"
        months_plus = re.search(r"(\d+(?:\.\d+)?)\s*\+\s*months?", text)
        if months_plus:
            min_years = float(months_plus.group(1)) / 12.0
        else:
            for pat in [
                r"(?:minimum|at least)\s+(\d+(?:\.\d+)?)\s*months?",
                r"(\d+(?:\.\d+)?)\s*months?\s*(?:minimum|experience)",
            ]:
                m = re.search(pat, text)
                if m:
                    min_years = float(m.group(1)) / 12.0
                    break

    # Pattern: "0-1 years" (treat as 0 min)
    zero_range = re.search(r"0\s*[-\u2013\u2014to]+\s*\d+\s*years?", text)
    if zero_range:
        if min_years is None or min_years > 0:
            min_years = 0.0

    # Pattern: "no experience required", "fresher welcome", "freshers can apply"
    for pat in [
        r"no\s+experience\s+(?:required|needed)",
        r"fresher[s]?\s+(?:welcome|can\s+apply|encourage)",
        r"recent\s+gradua",
    ]:
        if re.search(pat, text):
            if min_years is None or min_years > 0:
                min_years = 0.0
            seniority = "fresher"

    # Infer from seniority label if no numeric found
    if min_years is None and seniority:
        seniority_levels = {
            "fresher": 0.0,
            "junior": 1.0,
            "mid": 3.0,
            "senior": 5.0,
            "principal": 10.0,
        }
        min_years = seniority_levels.get(seniority)

    return min_years, max_years, seniority


def _extract_tech_from_jd(jd_text: str) -> list[str]:
    """Extract individual technology keywords from raw JD text."""
    text_lower = jd_text.lower()
    found: set[str] = set()
    for term in sorted(_COMMON_TECH_TERMS, key=len, reverse=True):
        if term in text_lower:
            normed = re.sub(r"[^a-z0-9\s]", "", term).strip()
            if normed and len(normed) > 1:
                found.add(normed)
    return sorted(found)


def _extract_atoms_from_response(raw_json: list) -> tuple[
    list[JDCapabilityAtom],  # core
    list[JDCapabilityAtom],  # secondary
    list[JDCapabilityAtom],  # adjacent
    list[JDScaleAtom],        # scale
    list[JDAtom],             # impact
    list[JDAtom],             # stakeholder
    list[JDAtom],             # process
    list[JDAtom],             # environment
    list[JDAtom],             # certification
    list[str],                # compliance_context
    dict,                     # global scalars
]:
    """Categorize raw extracted atoms into typed lists."""
    core: list[JDCapabilityAtom] = []
    secondary: list[JDCapabilityAtom] = []
    adjacent: list[JDCapabilityAtom] = []
    scale: list[JDScaleAtom] = []
    impact: list[JDAtom] = []
    stakeholder: list[JDAtom] = []
    process: list[JDAtom] = []
    environment: list[JDAtom] = []
    certification: list[JDAtom] = []
    compliance: list[str] = []
    globals_: dict = {}

    _GLOBAL_TYPES = {
        "role_family", "role_objective", "role_scope", "delivery_mode",
        "industry_vertical", "industry_sub_vertical", "business_model",
    }
    _CAP_TYPES = {"core_capability",
                  "secondary_capability", "adjacent_capability"}
    _SCALE_TYPES = {
        "scale_geographic", "scale_volume", "scale_revenue",
        "scale_infrastructure", "scale_team",
    }
    _IMPACT_TYPES = {
        "impact_business_type", "impact_measurable_outcome",
        "impact_kpi_ownership", "impact_value_creation_mode",
        "impact_lifecycle_stage",
    }

    for atom in raw_json:
        if not isinstance(atom, dict):
            continue
        atype = atom.get("atom_type", "")
        raw = atom.get("raw_text", "")
        normed = atom.get("normalized_entity", "")
        bid = atom.get("bundle_id")

        if atype in _GLOBAL_TYPES:
            globals_[atype] = normed
            continue

        if atype == "compliance_context":
            compliance.append(normed)
            continue

        if atype == "certification_context":
            certification.append(JDAtom(
                atom_type=atype, raw_text=raw,
                normalized_entity=normed, bundle_id=bid,
            ))
            continue

        if atype in _CAP_TYPES:
            cap = JDCapabilityAtom(
                atom_type=atype, raw_text=raw, normalized_entity=normed,
                bundle_id=bid,
                capability_type=atom.get("capability_type"),
                capability_subtype=atom.get("capability_subtype"),
                ownership_level=atom.get("ownership_level", "Support"),
                intensity_level=atom.get("intensity_level", "Medium"),
                classification=atom.get("classification", "SECONDARY"),
                motion=atom.get("motion"),
            )
            if atype == "core_capability":
                core.append(cap)
            elif atype == "secondary_capability":
                secondary.append(cap)
            else:
                adjacent.append(cap)
            continue

        if atype in _SCALE_TYPES:
            scale.append(JDScaleAtom(
                atom_type=atype, raw_text=raw,
                normalized_entity=normed, bundle_id=bid,
                value=atom.get("value"),
            ))
            continue

        if atype in _IMPACT_TYPES:
            impact.append(JDAtom(
                atom_type=atype, raw_text=raw,
                normalized_entity=normed, bundle_id=bid,
            ))
            continue

        if atype == "stakeholder_context":
            stakeholder.append(JDAtom(
                atom_type=atype, raw_text=raw,
                normalized_entity=normed, bundle_id=bid,
            ))
            continue

        if atype == "process_methodology":
            process.append(JDAtom(
                atom_type=atype, raw_text=raw,
                normalized_entity=normed, bundle_id=bid,
            ))
            continue

        if atype.startswith("environment_"):
            environment.append(JDAtom(
                atom_type=atype, raw_text=raw,
                normalized_entity=normed, bundle_id=bid,
            ))

    return (
        core, secondary, adjacent, scale, impact,
        stakeholder, process, environment, certification,
        compliance, globals_,
    )


def _build_normalized_skill_set(
    core: list[JDCapabilityAtom],
    secondary: list[JDCapabilityAtom],
    adjacent: list[JDCapabilityAtom],
    process: list[JDAtom],
    certification: list[JDAtom],
    environment: list[JDAtom],
) -> list[str]:
    """Flatten all normalized entities into a searchable skill set."""
    skills: set[str] = set()
    for cap in core + secondary + adjacent:
        if cap.normalized_entity:
            skills.add(cap.normalized_entity.lower())
    for a in process + certification + environment:
        if a.normalized_entity:
            skills.add(a.normalized_entity.lower())
    return sorted(skills)


async def parse_jd(jd_text: str) -> JDProfile:
    """Parse a raw Job Description into structured JDProfile.

    Steps:
      1. SetFit 3-tier classify the JD text
      2. LLM call to extract atoms
      3. Normalize and assemble JDProfile
    """
    # Step 1: SetFit classification
    if clf.models_available():
        tier_class = clf.classify_text(jd_text[:2000])
    else:
        logger.warning(
            "SetFit models not loaded — using fallback classification")
        tier_class = _build_default_tier_classification()

    # Step 1b: Prompt rescue cross-check (replaces the old LLM cross-check)
    tier_class = await _crosscheck_tier_classification(jd_text, tier_class)

    # Step 1c: Build layered classification for new 4-layer taxonomy
    layered_class = await _build_jd_layered_classification(jd_text, tier_class)

    # Step 2: LLM atom extraction
    client = _get_client()
    raw_atoms: list = []

    if client:
        try:
            response = await client.chat.completions.create(
                model=_AZURE_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": _JD_ATOM_PROMPT},
                    {"role": "user", "content": jd_text},
                ],
                max_completion_tokens=6000,
            )
            content = (response.choices[0].message.content or "").strip()
            # Strip code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
            raw_atoms = json.loads(content)
            if not isinstance(raw_atoms, list):
                raw_atoms = []
        except Exception as e:
            logger.exception("JD LLM extraction failed: %s", e)
            raw_atoms = []
    else:
        logger.warning("No Azure OpenAI key — JD will have no atoms extracted")

    # Step 3: Categorize and normalize
    (
        core, secondary, adjacent, scale, impact,
        stakeholder, process, environment, certification,
        compliance, globals_,
    ) = _extract_atoms_from_response(raw_atoms)

    required_skills = _build_normalized_skill_set(
        core, secondary, adjacent, process, certification, environment,
    )
    required_tech = _extract_tech_from_jd(jd_text)
    exp_min, exp_max, exp_seniority = _extract_experience_requirements(jd_text)

    # Extract raw experience phrase for downstream reference
    exp_text = None
    for pat in [
        r"(\d+\+?\s*years?[^\n]*)",
        r"(\d+\+?\s*months?[^\n]*)",
        r"(experience\s*:\s*[^\n]+)",
    ]:
        m = re.search(pat, jd_text, re.IGNORECASE)
        if m:
            exp_text = m.group(1).strip()[:100]
            break

    return JDProfile(
        raw_jd=jd_text,
        tier_classification=tier_class,
        layered_classification=layered_class,
        role_family=globals_.get("role_family"),
        role_objective=globals_.get("role_objective"),
        role_scope=globals_.get("role_scope"),
        delivery_mode=globals_.get("delivery_mode"),
        industry_vertical=globals_.get("industry_vertical"),
        industry_sub_vertical=globals_.get("industry_sub_vertical"),
        business_model=globals_.get("business_model"),
        compliance_context=list(set(compliance)),
        required_years_min=exp_min,
        required_years_max=exp_max,
        expected_seniority=exp_seniority,
        required_experience_text=exp_text,
        core_capabilities=core,
        secondary_capabilities=secondary,
        adjacent_capabilities=adjacent,
        scale_atoms=scale,
        impact_atoms=impact,
        stakeholder_atoms=stakeholder,
        process_methodology_atoms=process,
        environment_atoms=environment,
        certification_atoms=certification,
        required_skills_normalized=required_skills,
        required_tech_normalized=required_tech,
    )
