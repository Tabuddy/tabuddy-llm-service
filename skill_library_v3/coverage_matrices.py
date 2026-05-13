"""v1.4 Stage 2.5a — coverage matrices per (family, canonical_name).

Each matrix is a checklist of axes the critic must verify the Stage 2
generator covered. Curated for the families where we've seen coverage
gaps (Data & ML, Security) and the families our role catalog spans
heavily (Software Engineering, Infrastructure, QA, ERP). Unknown
families fall back to the generic 5-bucket framework.

Anchors per CoverageItem ground the critic prompt: the LLM sees
"Workflow Orchestration (Airflow, Dagster, Prefect, Argo Workflows)"
rather than "Workflow Orchestration", which sharply reduces the chance
of an empty-axis proposal that ``post_process_critic_response`` would
drop downstream anyway.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoverageItem:
    slug: str
    display: str
    anchors: tuple[str, ...] = ()


# ── Curated matrices ───────────────────────────────────────────────────────


_MLOPS_MATRIX: list[CoverageItem] = [
    CoverageItem("orchestration", "Workflow Orchestration",
                 ("Airflow", "Dagster", "Prefect", "Argo Workflows")),
    CoverageItem("serving", "Model Serving",
                 ("KServe", "BentoML", "Triton Inference Server", "TorchServe")),
    CoverageItem("registry", "Model Registry",
                 ("MLflow Model Registry", "SageMaker Model Registry",
                  "Vertex AI Model Registry")),
    CoverageItem("tracking", "Experiment Tracking",
                 ("MLflow Tracking", "Weights & Biases", "Neptune", "Comet")),
    CoverageItem("feature_store", "Feature Store",
                 ("Feast", "Tecton", "SageMaker Feature Store",
                  "Vertex AI Feature Store")),
    CoverageItem("monitoring", "Model Monitoring",
                 ("Evidently", "Arize", "Fiddler", "WhyLabs")),
    CoverageItem("distributed_training", "Distributed Training",
                 ("Ray Train", "DeepSpeed", "PyTorch DDP", "FSDP",
                  "Megatron-LM")),
    CoverageItem("k8s_runtime", "Kubernetes & GPU Runtime for ML",
                 ("KServe", "Kubeflow Trainer", "NVIDIA GPU Operator",
                  "Volcano")),
    CoverageItem("llm_ops", "LLM Operations",
                 ("Pinecone", "Weaviate", "vLLM", "LangSmith", "LangChain")),
    CoverageItem("governance", "AI Governance & Model Security",
                 ("NIST AI RMF", "EU AI Act", "Hugging Face safetensors",
                  "Model cards")),
    CoverageItem("cloud_platforms", "Cloud Platforms",
                 ("AWS", "GCP", "Azure")),
    CoverageItem("ci_cd_for_ml", "CI/CD for ML",
                 ("GitHub Actions", "GitLab CI", "Jenkins", "Argo CD")),
    CoverageItem("iac", "Infrastructure as Code",
                 ("Terraform", "Pulumi", "AWS CloudFormation")),
    CoverageItem("hardware", "Accelerators & Hardware",
                 ("NVIDIA GPUs", "TPUs", "CUDA", "AMD ROCm")),
    CoverageItem("versioning", "Model & Data Versioning",
                 ("DVC", "lakeFS", "Delta Lake", "Pachyderm")),
]


_CYBERSEC_MATRIX: list[CoverageItem] = [
    CoverageItem("network_security", "Network Security",
                 ("Firewalls", "VPN", "Zero Trust Network Access",
                  "Network Segmentation")),
    CoverageItem("endpoint_security", "Endpoint Security & EDR",
                 ("CrowdStrike Falcon", "SentinelOne", "Microsoft Defender")),
    CoverageItem("siem", "SIEM & Detection Engineering",
                 ("Splunk", "Microsoft Sentinel", "Elastic Security",
                  "Sumo Logic")),
    CoverageItem("cnapp", "Cloud Security Posture (CNAPP)",
                 ("Wiz", "Prisma Cloud", "Lacework", "Orca Security")),
    CoverageItem("iam", "Identity & Access Management",
                 ("Okta", "Azure AD", "CyberArk", "BeyondTrust")),
    CoverageItem("vuln_management", "Vulnerability Management",
                 ("Qualys", "Rapid7 InsightVM", "Tenable Nessus")),
    CoverageItem("pentesting", "Penetration Testing & Offensive Security",
                 ("Burp Suite", "Metasploit", "Cobalt Strike", "Nmap")),
    CoverageItem("appsec", "Application Security",
                 ("OWASP Top 10", "SAST", "DAST", "Veracode", "Snyk")),
    CoverageItem("cloud_platforms", "Cloud Platforms",
                 ("AWS", "GCP", "Azure")),
    CoverageItem("compliance", "Compliance & Frameworks",
                 ("NIST", "ISO 27001", "SOC 2", "PCI DSS", "GDPR")),
    CoverageItem("cryptography", "Cryptography & PKI",
                 ("TLS", "PKI", "HSM", "Key Management")),
    CoverageItem("incident_response", "Incident Response & Forensics",
                 ("DFIR", "Volatility", "Autopsy")),
    CoverageItem("threat_intel", "Threat Intelligence",
                 ("MITRE ATT&CK", "STIX/TAXII", "ThreatConnect")),
    CoverageItem("ot_iot", "OT/IoT Security",
                 ("Claroty", "Nozomi Networks", "Dragos")),
]


_BACKEND_MATRIX: list[CoverageItem] = [
    CoverageItem("languages", "Programming Languages",
                 ("Java", "Python", "Go", "TypeScript", "C#", "Kotlin")),
    CoverageItem("frameworks", "Web & Application Frameworks",
                 ("Spring Boot", "Django", "FastAPI", "Express", "Rails")),
    CoverageItem("api_design", "API Design",
                 ("REST", "GraphQL", "gRPC", "OpenAPI")),
    CoverageItem("relational_db", "Relational Databases",
                 ("PostgreSQL", "MySQL", "SQL Server")),
    CoverageItem("nosql_db", "NoSQL Databases",
                 ("MongoDB", "Cassandra", "DynamoDB", "Redis")),
    CoverageItem("caching", "Caching Systems",
                 ("Redis", "Memcached")),
    CoverageItem("message_queues", "Message Queues & Event Streaming",
                 ("Apache Kafka", "RabbitMQ", "AWS SQS", "NATS")),
    CoverageItem("authn_authz", "Authentication & Authorization",
                 ("OAuth 2.0", "OIDC", "JWT", "SAML", "RBAC")),
    CoverageItem("microservices", "Microservices & Service Architecture",
                 ("API Gateway", "Service Mesh", "Service Discovery",
                  "Circuit Breaker")),
    CoverageItem("containerization", "Containerization",
                 ("Docker", "Kubernetes", "Helm")),
    CoverageItem("testing", "Testing",
                 ("JUnit", "pytest", "Pact", "TDD")),
    CoverageItem("observability", "Observability & Monitoring",
                 ("OpenTelemetry", "Prometheus", "Datadog", "Jaeger")),
    CoverageItem("cloud_platforms", "Cloud Platforms",
                 ("AWS", "GCP", "Azure")),
]


_FRONTEND_MATRIX: list[CoverageItem] = [
    CoverageItem("languages", "Languages & Compilers",
                 ("TypeScript", "JavaScript", "WebAssembly")),
    CoverageItem("frameworks", "UI Frameworks & Libraries",
                 ("React", "Vue", "Angular", "Svelte")),
    CoverageItem("meta_frameworks", "Meta-Frameworks & SSR",
                 ("Next.js", "Nuxt", "Remix", "Astro")),
    CoverageItem("styling", "Styling & Design Systems",
                 ("Tailwind CSS", "CSS Modules", "Styled Components")),
    CoverageItem("state_management", "State Management",
                 ("Redux", "Zustand", "Jotai", "TanStack Query")),
    CoverageItem("testing", "Testing",
                 ("Jest", "Vitest", "Playwright", "Cypress",
                  "Testing Library")),
    CoverageItem("build_tools", "Build & Bundling Tools",
                 ("Vite", "Webpack", "esbuild", "Turbopack")),
    CoverageItem("accessibility", "Accessibility & Standards",
                 ("WCAG", "ARIA", "axe-core")),
    CoverageItem("performance", "Performance & Core Web Vitals",
                 ("Lighthouse", "LCP", "CLS", "INP")),
]


_AI_ENGINEER_MATRIX: list[CoverageItem] = [
    CoverageItem("llm_apis", "LLM Provider APIs",
                 ("OpenAI", "Anthropic", "Google Gemini", "Mistral")),
    CoverageItem("frameworks", "LLM Orchestration Frameworks",
                 ("LangChain", "LlamaIndex", "DSPy", "Haystack")),
    CoverageItem("rag", "Retrieval Augmented Generation",
                 ("Pinecone", "Weaviate", "Qdrant", "Chroma", "Milvus")),
    CoverageItem("evaluation", "LLM Evaluation & Observability",
                 ("LangSmith", "Helicone", "LangFuse", "Promptfoo")),
    CoverageItem("serving", "LLM Serving",
                 ("vLLM", "TGI", "llama.cpp", "Ollama")),
    CoverageItem("prompt_management", "Prompt Engineering & Management",
                 ("PromptLayer", "Humanloop")),
    CoverageItem("cloud_platforms", "Cloud Platforms",
                 ("AWS", "GCP", "Azure")),
    CoverageItem("fine_tuning", "Fine-Tuning & Adaptation",
                 ("LoRA", "QLoRA", "PEFT", "SFT")),
]


_DEVOPS_MATRIX: list[CoverageItem] = [
    CoverageItem("iac", "Infrastructure as Code",
                 ("Terraform", "Pulumi", "AWS CloudFormation")),
    CoverageItem("ci_cd", "CI/CD",
                 ("GitHub Actions", "GitLab CI", "Jenkins", "Argo CD")),
    CoverageItem("containerization", "Containerization & Orchestration",
                 ("Docker", "Kubernetes", "Helm")),
    CoverageItem("observability", "Observability",
                 ("Prometheus", "Grafana", "OpenTelemetry", "Datadog")),
    CoverageItem("cloud_platforms", "Cloud Platforms",
                 ("AWS", "GCP", "Azure")),
    CoverageItem("config_management", "Configuration Management",
                 ("Ansible", "Chef", "Puppet")),
    CoverageItem("secrets", "Secrets Management",
                 ("HashiCorp Vault", "AWS Secrets Manager", "1Password")),
    CoverageItem("service_mesh", "Service Mesh & Networking",
                 ("Istio", "Linkerd", "Consul")),
]


_QA_MATRIX: list[CoverageItem] = [
    CoverageItem("ui_automation", "UI Test Automation",
                 ("Selenium", "Playwright", "Cypress", "WebdriverIO")),
    CoverageItem("api_testing", "API Testing",
                 ("Postman", "REST Assured", "Karate")),
    CoverageItem("load_testing", "Load & Performance Testing",
                 ("JMeter", "k6", "Gatling", "Locust")),
    CoverageItem("mobile_testing", "Mobile Test Automation",
                 ("Appium", "Espresso", "XCUITest")),
    CoverageItem("test_management", "Test Management & Reporting",
                 ("TestRail", "Zephyr", "Xray")),
    CoverageItem("frameworks", "Languages & Frameworks for Tests",
                 ("Java + TestNG", "Python + pytest", "JavaScript + Mocha")),
]


_ERP_MATRIX: list[CoverageItem] = [
    CoverageItem("platforms", "ERP / ITSM Platforms",
                 ("ServiceNow", "SAP", "Oracle", "Workday")),
    CoverageItem("modules", "Functional Modules",
                 ("ITSM", "HRIS", "Finance", "CRM")),
    CoverageItem("scripting", "Platform Scripting & Customization",
                 ("Glide Scripting", "ABAP", "Apex")),
    CoverageItem("integration", "Integration Patterns",
                 ("REST APIs", "SOAP", "iPaaS", "Workato")),
]


# ── Lookup tables ──────────────────────────────────────────────────────────


COVERAGE_MATRICES: dict[str, list[CoverageItem]] = {
    "Data & ML / MLOps Engineer": _MLOPS_MATRIX,
    "Data & ML / ML Engineer": _MLOPS_MATRIX,
    "Data & ML / Machine Learning Engineer": _MLOPS_MATRIX,
    "Security / Cybersecurity Engineer": _CYBERSEC_MATRIX,
    "Software Engineering / Backend Engineer": _BACKEND_MATRIX,
    "Software Engineering / Full-stack Developer": _BACKEND_MATRIX,
    "Software Engineering / Frontend Engineer": _FRONTEND_MATRIX,
    "Software Engineering / AI Engineer": _AI_ENGINEER_MATRIX,
    "Infrastructure & Platform / DevOps Engineer": _DEVOPS_MATRIX,
    "Infrastructure & Platform / Site Reliability Engineer": _DEVOPS_MATRIX,
    "QA & SDET / Automation Tester": _QA_MATRIX,
    "QA & SDET / SDET": _QA_MATRIX,
    "ERP & Enterprise / ServiceNow Developer": _ERP_MATRIX,
}


_FALLBACK_MATRIX: list[CoverageItem] = [
    CoverageItem("languages", "Programming Languages & DSLs"),
    CoverageItem("frameworks", "Frameworks & Libraries"),
    CoverageItem("vendor_products", "Vendor Product Families"),
    CoverageItem("concepts", "Concepts, Patterns & Practices"),
    CoverageItem("standards", "Standards, Protocols & Compliance"),
]


def pick_matrix(*, family: str, canonical_name: str) -> list[CoverageItem]:
    """Return the curated matrix for ``(family, canonical_name)``, or
    the generic fallback if no curated matrix exists.

    Matching is exact on the composite key ``f"{family} / {canonical_name}"``.
    Family-only fallback isn't useful here — a "Data & ML" role with an
    unknown sub-archetype could be a Data Engineer or a research-focused
    ML scientist, and their coverage matrices differ.
    """
    key = f"{family} / {canonical_name}"
    return COVERAGE_MATRICES.get(key, _FALLBACK_MATRIX)
