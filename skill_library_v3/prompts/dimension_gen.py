"""Stage 2 dimension-generation prompt (system + user).

Versioned so every ``v2_run_log`` row records which prompt produced its
``planner_output``. Bump DIM_GEN_PROMPT_VERSION on any text change.
The ``stage2_`` prefix is what the runner + repository use to discriminate
Stage 2 runs from Stages 0/1 in the shared v2_run_log table.
"""

from __future__ import annotations

import json

DIM_GEN_PROMPT_VERSION = "stage2_dim_gen_v1.4"


DIM_GEN_SYSTEM_PROMPT = """\
You are the Stage 2 Dimension Generator for a canonical skills-catalog
pipeline. Given a role card (the dense compression of an approved role
charter), produce 12-20 candidate dimensions covering the role's skill
surface.

A "dimension" is a coherent skill cluster a practitioner of this role
needs. Examples across role families:
  * Backend Engineer: Programming Languages, Web & Application Frameworks,
    API Design & Patterns, Relational Databases, Authentication &
    Authorization, Microservices & Service Architecture.
  * Cybersecurity Engineer: Network Security, Endpoint Security & EDR
    Products, SIEM Products & Detection Engineering, Vulnerability
    Management Products, Cloud Security Tools (CNAPP/CSPM/CWPP),
    Cryptography Primitives, Web Application Vulnerability Concepts.
  * Data Engineer: Cloud Data Warehouses, Data Pipeline Orchestrators,
    Stream Processing, ETL/ELT Tools, Data Quality & Lineage.

Hard rules:

  * Generate EXACTLY 12 to 20 dimensions. No fewer, no more.
  * Intentionally allow a few overlaps and over-splits — Stage 3
    reconciliation will merge/split as needed. Better to surface a
    duplicate than to silently miss a skill cluster.
  * No "Miscellaneous", "Other", "General", "Misc" dimensions. If you
    can't name 5+ concrete exemplar skills under a dimension, drop it
    or fold it into a sibling dimension.
  * Skills listed in always_load_skills (Git, SQL, Code Review, Unit
    Testing, etc.) MUST NOT be turned into their own dimensions. Those
    are cross-cutting universal items handled separately. Do not name
    a dimension "Git" or "SQL" or "Unit Testing".
  * Avoid marketing / buzzword names ("Cloud-Native Excellence",
    "Modern Engineering Practices"). Name dimensions by their substance
    (Containerization, Software Design Patterns & Architecture).

Per-dimension fields:

  * tentative_id: 'd_init_NN' where NN is two digits zero-padded
    (d_init_01, d_init_02, ..., d_init_19, d_init_20). Unique within
    the role.
  * name: 3-8 word noun-phrase. Specific, not generic ("Pipeline
    Orchestration", not "Tools").
  * description: 1-3 sentences (<=400 chars). What the dimension covers
    and why it's a coherent cluster for this role.
  * in_scope: comma-delimited specifics — concrete tools, frameworks,
    libraries, or sub-topics that belong in this dim. Example for
    Relational Databases: "PostgreSQL, MySQL, SQL Server, Oracle,
    MariaDB". 10-400 chars.
  * out_of_scope: comma-delimited specifics naming what is excluded and
    why (which other dim or which adjacent role owns it). Example:
    "Analytical warehouses (those go to Data Engineer's dim);
    DBA-owned operational tasks". 10-400 chars.
  * exemplar_skills: 3-15 concrete skill names that anchor this
    dimension. These are CONCRETE skills (PostgreSQL, Spring Boot,
    OAuth 2.0), not concepts.
  * overlap_flags: zero or more flags pointing at potential overlap.
    Three forms:
      - {"with_dim_name": "<name>", "reason": "..."} — overlap with
        another dim YOU just generated for THIS role (the LLM knows the
        names you used)
      - {"with_dim_id": "<id>", "reason": "..."} — overlap with an
        adjacent role's existing canonical-store dim id
      - {"with_role": "<role>", "reason": "..."} — overlap with an
        adjacent role's territory generally
    Use overlap_flags WHENEVER an adjacent role has a dim that would
    plausibly contain the same skill. Stage 3 needs these to reconcile.

Coverage discipline — five-bucket framework. Every role's skill surface
decomposes into some mixture of these five buckets. Use them as a
checklist before finalising:

  1. **Languages & DSLs** — what the role authors code in. Programming
     languages, query languages, IaC DSLs, smart-contract languages,
     detection-rule languages, shader languages — whichever apply.
  2. **Frameworks & Libraries** — runtime libraries the role depends on
     (web frameworks, ORMs, ML libs, smart-contract frameworks, UI
     toolkits, agentic frameworks).
  3. **Vendor Product Families** — concrete commercial / open-source
     product categories. Each product family with 3+ named offerings the
     role engages with deserves its own dimension: SIEM products, EDR
     products, CNAPP/CSPM tools, IAM products, vulnerability scanners,
     MLOps platforms, cloud data warehouses, BI tools, container
     security, CI/CD platforms, orchestrators, observability stacks.
  4. **Concepts, Patterns & Practices** — methodology and design
     knowledge (distributed systems concepts, threat modelling, OWASP
     Top 10, design patterns, MLOps practices, network protocols).
  5. **Standards, Protocols & Compliance** — formal specs and compliance
     frameworks (TLS / IPSec / DNSSEC, OAuth / OIDC / SAML, PCI-DSS /
     HIPAA / SOC 2 / ISO 27001) — kept as their own dim when the role
     is regulated.

Anti-pattern — DO NOT collapse a vendor product family into a one-token
mention inside a concepts dim. If a role legitimately uses Splunk AND
Microsoft Sentinel AND Elastic Security, those belong in a `SIEM
Products & Detection Engineering` dim, not as the word "SIEM" buried in
the in_scope of a `Security Monitoring` concepts dim. The same rule
applies to EDR products, CNAPP tools, IAM products, MLOps platforms,
data warehouses, etc.: a category with 3+ named offerings the role
engages with is its own dimension, separate from any concepts dim that
governs the same area.

Vendor pairing rule (CRITICAL for Stage 6 containment to succeed). When
a Vendor Product Family dim names commercial SaaS products, the
exemplar_skills list MUST include the vendor itself as a STANDALONE
exemplar — not just a sub-capability name. Examples of right vs wrong:

  * RIGHT: ["Wiz", "Wiz Cloud Security Posture Management", "Lacework",
    "Lacework Polygraph", "Prisma Cloud", "Prisma Cloud Compute"]
  * WRONG: ["wiz-posture-review", "lacework-detections",
    "prisma-cloud-policies"]
    — these compound names omit the vendor itself, so when Stage 4
    types them as Service, Stage 6 has no Platform parent to attach
    them to (they are orphans).

The same rule applies to AWS / Azure / GCP (the platforms must appear
as standalone exemplars alongside any of their named services like
"AWS KMS", "Azure Key Vault"), to identity vendors (Okta, Auth0 are
standalone Platforms), and to chain platforms (Ethereum, Solana are
standalone Platforms alongside chain-specific services).

Don't force-fit irrelevant axes — if the role doesn't touch a bucket,
omit it. A pure-concepts role legitimately has fewer Vendor Product
Families dims; a heavy-tooling role legitimately has more. Sweep the
five buckets before finalising.

Anti-overlap discriminator. If a vendor product (e.g., Vertex AI,
SageMaker, Azure ML, MLflow, Kafka, Redis, Splunk, Microsoft Sentinel)
plausibly fits TWO of your candidate dims, pick ONE primary dim that
owns it and put it in the OTHER dim's out_of_scope as
"<Product> (belongs to <other dim name>)". NEVER list the same product
in in_scope of two dims you generate. Stage 3 reconciliation should
not have to choose which of your two dims owns Vertex AI — the
disambiguation is yours, made up-front, costing only a few tokens of
out_of_scope rather than a full LLM merge call. The same rule applies
to in_scope and exemplar_skills: a product name appears in at most
one dim's in_scope and at most one dim's exemplar_skills across the
role.

Exemplar ordering. List each dim's ``exemplar_skills`` in defining-first
order — the most central / canonical examples of the dim FIRST, peripheral
examples LAST. Stage 3's cluster gate weights shared exemplars by
inverse position (1/(rank+1)), so two dims sharing their #1 exemplar
score far higher than two dims sharing a peripheral exemplar. Getting
the order right tightens borderline merge decisions; getting it wrong
either collapses dims that shouldn't merge or keeps dims separate that
should.

Emit a single JSON object matching the schema. No prose, no code fences.
"""


DIM_GEN_USER_TEMPLATE = """\
Role card for {role_name} ({role_slug}):

{role_card_block}

Always-load skills (do NOT make these dimensions):
{always_load_block}

{family_hints_block}{adjacent_dim_block}

Produce the JSON dimension list now. Remember: 12-20 dimensions, every
dimension has 3+ exemplar_skills, flag overlaps with adjacent roles
when present.
"""


# ── formatters ─────────────────────────────────────────────────────────────


def format_role_card_block(role_card: dict) -> str:
    """Render the role card as a compact block. v1.2 role cards have
    primary_responsibilities + common_artifacts as semicolon-strings; v1.0
    has them as lists. Handle both for backwards compatibility during the
    v1.0 → v1.2 transition."""

    def _to_str(field) -> str:
        if isinstance(field, list):
            return "; ".join(field)
        return str(field or "")

    parts = [
        f"canonical_name: {role_card.get('canonical_name', '')}",
        f"family: {role_card.get('family', '')}",
        f"definition: {role_card.get('definition', '')}",
        f"primary_responsibilities: {_to_str(role_card.get('primary_responsibilities'))}",
        f"common_artifacts: {_to_str(role_card.get('common_artifacts'))}",
        f"distinguishing_tasks: {role_card.get('distinguishing_tasks', '')}",
    ]
    aliases = role_card.get("aliases") or []
    if aliases:
        parts.append(f"aliases: {', '.join(aliases)}")
    seniority = role_card.get("seniority_signals") or {}
    if isinstance(seniority, dict) and any(seniority.values()):
        sen_lines = [
            f"  - {level}: {seniority.get(level, '')}"
            for level in ("junior", "mid", "senior", "staff")
            if seniority.get(level)
        ]
        parts.append("seniority_signals:\n" + "\n".join(sen_lines))
    return "\n".join(parts)


def format_always_load_block(skills: list[str]) -> str:
    if not skills:
        return "(none — no cross-cutting universals to exclude)"
    return ", ".join(skills)


# ── family-aware axis hints ────────────────────────────────────────────────


# Family → narrative axis-hints paragraph. The LLM sees these as
# *suggestions to consider* — not a must-have list. The 12-20 dim cap
# forces it to pick the highest-value axes for the specific role.
#
# Software Engineering is split by canonical_name (sub-archetype) because
# Backend / Frontend / Mobile / AI Engineer / Blockchain Developer share a
# family but need wildly different axis hints. Backend / full-stack →
# empty hints (the system prompt's universal layer already covers them).
#
# Order of fall-through inside Software Engineering:
#   1. Blockchain (canonical contains "blockchain")
#   2. Mobile (canonical contains "ios" / "android" / "mobile")
#   3. AI Engineer (canonical contains "ai engineer")
#   4. Frontend (canonical contains "frontend")
#   5. Default → empty hints
_SECURITY_HINTS = (
    "Sub-archetype axis hints for Security roles — consider these "
    "vendor-product and concept dimensions before finalising:\n"
    "  - Network Security (firewalls, IDS/IPS, NDR, VPN, ZTNA; network "
    "protocols TLS/IPSec/DNS-security)\n"
    "  - Endpoint Security & EDR Products (CrowdStrike, SentinelOne, "
    "Microsoft Defender, Carbon Black, Cortex XDR)\n"
    "  - SIEM Products & Detection Engineering Languages (Splunk, "
    "Microsoft Sentinel, QRadar, Elastic Security, Sumo Logic + SPL, "
    "KQL, Lucene, Sigma)\n"
    "  - Vulnerability Management Products (Nessus, Qualys, Rapid7, "
    "OpenVAS, Burp Suite)\n"
    "  - Cloud Security Tools — CNAPP/CSPM/CWPP (Wiz, Prisma Cloud, "
    "Lacework, Aqua, Sysdig, Orca + GuardDuty, Defender for Cloud, "
    "Security Command Center)\n"
    "  - IAM Products (Okta, Auth0, Azure AD, PingIdentity, ForgeRock + "
    "PAM products: CyberArk, BeyondTrust)\n"
    "  - Container & Kubernetes Security (Trivy, Clair, Falco, Sysdig "
    "Runtime, OPA, Kyverno)\n"
    "  - Cryptography Primitives (hashing, symmetric/asymmetric, "
    "PKI/CA/CRL/OCSP, digital signatures, TLS internals)\n"
    "  - Forensics & Malware Analysis (Volatility, IDA Pro, Ghidra, "
    "YARA, Sigma)\n"
    "  - Web Application Vulnerability Concepts (OWASP Top 10 — XSS, "
    "CSRF, SQLi, IDOR, SSRF; WAF, RASP)\n"
    "  - Email Security (DMARC, DKIM, SPF, Proofpoint, Mimecast)\n"
    "  - Data Security & DLP (Microsoft Purview, Symantec DLP, "
    "Forcepoint; classification, tokenisation)\n"
    "  - Pentesting & Red-Team Tools (Burp Suite, Metasploit, Kali, "
    "BloodHound, Cobalt Strike, Nmap, sqlmap)\n"
    "  - Compliance Frameworks (PCI-DSS, HIPAA, SOC 2, ISO 27001, "
    "NIST CSF, NIST 800-53, GDPR, FedRAMP)\n"
    "  - Cloud Platforms (AWS, Azure, GCP) — include this so cross-"
    "cutting Service-typed skills (KMS, GuardDuty, Defender for Cloud, "
    "Security Command Center) have a Platform parent to attach to in "
    "Stage 6 containment\n"
    "\n"
    "Vendor emission for security: every vendor named above must appear "
    "as a STANDALONE exemplar_skill in the dim that uses it — not just "
    "as a substring of a compound name. Required Platform-typed vendor "
    "exemplars when their products are in_scope: AWS, Azure, GCP, Okta, "
    "Auth0, Microsoft, Wiz, Qualys, Rapid7, Lacework, Prisma Cloud, "
    "CyberArk, BeyondTrust, CrowdStrike, SentinelOne, Splunk, Microsoft "
    "Sentinel. Pair each vendor with any of its named sub-services as "
    "additional exemplars (e.g., AWS + AWS KMS + AWS GuardDuty, or Wiz "
    "+ Wiz Cloud Security Posture Management).\n"
)

_BLOCKCHAIN_HINTS = (
    "Sub-archetype axis hints for Blockchain Developer — consider these "
    "platform, primitive, and tooling dimensions:\n"
    "  - L1/L2 Chain Platforms (Ethereum, Solana, Avalanche, Polygon, "
    "Sui, Aptos, Arbitrum, Optimism)\n"
    "  - Smart Contract Languages (Solidity, Rust, Move, Vyper, Cairo)\n"
    "  - Smart Contract Frameworks (Hardhat, Foundry, Anchor, Truffle, "
    "OpenZeppelin Contracts, CosmWasm)\n"
    "  - Wallet Ecosystems (MetaMask, Phantom, WalletConnect, "
    "Coinbase Wallet)\n"
    "  - Web3 Client Libraries (ethers.js, web3.js, viem, web3.py)\n"
    "  - On-Chain Indexing Infrastructure (Alchemy, Infura, QuickNode, "
    "The Graph, subgraphs)\n"
    "  - DeFi Primitives (DEX/AMM, lending, staking, yield farming, "
    "derivatives, perps)\n"
    "  - NFT Ecosystem (ERC-721/1155 standards, marketplaces, "
    "metadata, royalties)\n"
    "  - Cross-Chain Bridges Products (LayerZero, Wormhole, Axelar, "
    "Chainlink CCIP)\n"
    "  - DAO Tooling (Snapshot, Tally, Aragon, Gnosis Safe)\n"
    "  - Smart Contract Audit Tooling (Slither, Mythril, Echidna, "
    "Certora, Foundry fuzzing)\n"
    "  - Cryptographic Primitives & Key Management (ECDSA, Schnorr, "
    "BLS, hashing, Merkle proofs, HSMs)\n"
    "  - Token Standards (ERC-20, ERC-721, ERC-1155, ERC-4626, SPL)\n"
)

_MOBILE_HINTS = (
    "Sub-archetype axis hints for Mobile roles — consider these "
    "platform-specific dimensions:\n"
    "  - Native Languages (Swift, Kotlin, Objective-C, Java)\n"
    "  - UI Frameworks (SwiftUI, UIKit, Jetpack Compose, Android Views)\n"
    "  - Cross-Platform Frameworks (React Native, Flutter)\n"
    "  - Build Systems (Xcode, Gradle, fastlane)\n"
    "  - Mobile Storage & Persistence (Core Data, Room, SQLite, Realm)\n"
    "  - Mobile Networking (URLSession, OkHttp, Retrofit, Alamofire)\n"
    "  - Mobile Auth & Keychain (Keychain Services, Android Keystore, "
    "biometrics)\n"
    "  - Background Services & Lifecycle (WorkManager, BGTaskScheduler)\n"
    "  - Push Notifications (APNs, FCM)\n"
    "  - Mobile Testing (XCTest, Espresso, UI Automator)\n"
    "  - App Distribution (App Store Connect, Google Play Console, "
    "TestFlight, internal CI for mobile)\n"
    "  - Mobile Observability (Firebase Crashlytics, Sentry, "
    "Bugsnag)\n"
)

_AI_ENGINEER_HINTS = (
    "Sub-archetype axis hints for AI Engineer — consider these "
    "LLM-application dimensions:\n"
    "  - LLM Provider APIs (OpenAI, Anthropic, Azure OpenAI, "
    "Bedrock, Vertex AI)\n"
    "  - Prompt Engineering Patterns (few-shot, CoT, ReAct, "
    "structured output)\n"
    "  - RAG Architectures (retrieval, reranking, chunking, "
    "hybrid search)\n"
    "  - Vector Databases (Pinecone, Weaviate, Qdrant, Chroma, "
    "pgvector, Milvus)\n"
    "  - Embedding Models (OpenAI embeddings, Cohere, BGE, E5)\n"
    "  - LLM Orchestration Frameworks (LangChain, LlamaIndex, DSPy, "
    "Semantic Kernel)\n"
    "  - Agentic Frameworks (LangGraph, CrewAI, AutoGen, Letta)\n"
    "  - Evaluation Tooling (Ragas, Promptfoo, DeepEval, TruLens)\n"
    "  - Fine-tuning Pipelines (LoRA/QLoRA, PEFT, Unsloth, Together)\n"
    "  - LLM Observability (LangSmith, Langfuse, Phoenix, Helicone)\n"
)

_FRONTEND_HINTS = (
    "Sub-archetype axis hints for Frontend Engineer — consider these "
    "browser-app dimensions:\n"
    "  - UI Frameworks (React, Vue, Angular, Svelte, Solid)\n"
    "  - State Management (Redux, Zustand, MobX, Pinia, Recoil)\n"
    "  - CSS Architecture (Tailwind, CSS-in-JS, CSS Modules, BEM, "
    "Sass)\n"
    "  - Build Tooling (Vite, Webpack, Turbopack, esbuild, Rollup)\n"
    "  - Browser APIs (Fetch, WebSockets, Service Workers, "
    "IndexedDB)\n"
    "  - Accessibility Standards (WCAG, ARIA, screen-reader testing)\n"
    "  - Component Libraries (Material UI, shadcn/ui, Chakra, "
    "Ant Design)\n"
    "  - Testing (Jest, Vitest, Testing Library, Playwright, Cypress)\n"
    "  - Bundling & Performance (code-splitting, lazy-loading, "
    "Lighthouse, Core Web Vitals)\n"
    "  - Mobile-Web (responsive design, PWA, mobile-first patterns)\n"
)

_DATA_ML_HINTS = (
    "Family axis hints for Data & ML roles — consider these data-platform "
    "and ML dimensions:\n"
    "  - OLTP & OLAP Data Stores (split when both apply)\n"
    "  - Cloud Data Warehouses (Snowflake, BigQuery, Redshift, "
    "Synapse, Databricks SQL)\n"
    "  - Data Pipeline Orchestrators (Airflow, Dagster, Prefect, "
    "Argo Workflows)\n"
    "  - ETL/ELT Tools (dbt, Fivetran, Informatica, Matillion, "
    "Airbyte)\n"
    "  - Stream Processing (Apache Flink, Spark Streaming, "
    "Kafka Streams, ksqlDB)\n"
    "  - ML Frameworks (PyTorch, TensorFlow, JAX, scikit-learn, "
    "XGBoost, LightGBM)\n"
    "  - Feature Stores (Feast, Tecton, SageMaker Feature Store)\n"
    "  - Model Serving (TorchServe, TFServing, Triton, BentoML, "
    "Ray Serve)\n"
    "  - MLOps Platforms (Vertex AI, SageMaker, Azure ML, MLflow, "
    "Kubeflow, Weights & Biases) — end-to-end managed ML platforms; "
    "distinct from cloud compute runtimes (the latter belong in a "
    "Cloud Compute Runtimes dim, not here)\n"
    "  - Experiment Tracking (MLflow, Weights & Biases, Neptune, "
    "Comet) — keep separate from Feature Stores; MLflow Tracking is "
    "an experiment artifact, not a feature store concern\n"
    "  - BI & Visualization (Tableau, Power BI, Looker, Metabase, "
    "Superset)\n"
    "  - Data Quality & Lineage (Great Expectations, Monte Carlo, "
    "OpenLineage, dbt tests)\n"
    # 2026 axes — first-class for MLOps / AI Engineer roles
    "  - LLM Operations (vector databases — Pinecone, Weaviate, "
    "Qdrant, Milvus, Chroma; LLM serving — vLLM, TGI, llama.cpp; "
    "LLM evaluation & observability — LangSmith, Langfuse, Helicone; "
    "prompt management — PromptLayer, Humanloop; LLM orchestration — "
    "LangChain, LlamaIndex, DSPy)\n"
    "  - Distributed Training (Ray Train, Horovod, DeepSpeed, FSDP, "
    "PyTorch DDP, Megatron-LM)\n"
    "  - Kubernetes for ML (KServe, Kubeflow Trainer, NVIDIA GPU "
    "Operator, NVIDIA Device Plugin, Volcano scheduler)\n"
    "  - Model & Data Versioning (DVC, lakeFS, Delta Lake, "
    "Pachyderm) — distinct from MLflow Model Registry; this covers "
    "data + model artifact versioning at the filesystem layer\n"
    "  - AI Governance & Model Security (Model cards, NIST AI RMF, "
    "EU AI Act readiness, model supply chain security, "
    "Hugging Face safetensors)\n"
    "\n"
    "MLOps / AI Engineer sub-archetype note: for MLOps and AI Engineer "
    "roles in 2026, the LLM Operations and Distributed Training axes "
    "are first-class — most production ML systems include one or "
    "both. Do not skip them as niche.\n"
)

_STORAGE_ENGINEER_HINTS = (
    "Sub-archetype axis hints for Storage Engineer — the vendor surface "
    "is wholly distinct from generic DevOps and deserves dedicated "
    "product-family dimensions:\n"
    "  - Enterprise Storage Vendor Products (NetApp ONTAP, Pure Storage "
    "FlashArray/FlashBlade, Dell EMC PowerStore/PowerMax/Unity, HPE "
    "3PAR/Primera/Alletra, Hitachi VSP, IBM FlashSystem)\n"
    "  - Software-Defined / Object Storage (Ceph, MinIO, OpenStack "
    "Swift, VMware vSAN, GlusterFS)\n"
    "  - Cloud Storage Services (AWS S3 / EBS / EFS / FSx, Azure Blob / "
    "Files / Disks, Google Cloud Storage / Persistent Disk / Filestore)\n"
    "  - Backup & Data Protection Software (Veeam, Commvault, Veritas "
    "NetBackup, Rubrik, Cohesity, Dell Avamar / NetWorker)\n"
    "  - Disaster Recovery & Replication Products (Zerto, Azure Site "
    "Recovery, AWS DRS, EMC SRDF, NetApp SnapMirror)\n"
    "  - Storage Networking & Fabric (Fibre Channel switches — Brocade, "
    "Cisco MDS; iSCSI, NVMe-oF, FCoE, multipathing, zoning)\n"
    "  - Storage Protocols & Filesystems (NFS, SMB/CIFS, NVMe, ZFS, "
    "BTRFS, XFS, ext4, GPFS, Lustre)\n"
    "  - Storage Management & Orchestration (NetApp Active IQ / "
    "OnCommand, Pure1, Dell Unisphere, vSphere Storage)\n"
    "  - Capacity & Performance Analytics (capacity forecasting, IOPS "
    "modelling, latency budgets, telemetry pipelines)\n"
    "  - Data Lifecycle & Archive (lifecycle policies, tiering, WORM, "
    "immutability, archive transitions, AWS Glacier / Azure Archive)\n"
    "  - Storage Security (encryption at rest, KMIP, key management "
    "integration, tenant isolation, secure erase)\n"
    "  - Provisioning Automation (storage provisioning APIs, Ansible / "
    "Terraform storage providers, vendor REST APIs)\n"
)


_INFRA_PLATFORM_HINTS = (
    "Family axis hints for Infrastructure & Platform roles — consider "
    "these platform-tooling dimensions:\n"
    "  - IaC (Terraform, Pulumi, CloudFormation, Bicep, CDK)\n"
    "  - Configuration Management (Ansible, Chef, Puppet, SaltStack)\n"
    "  - Container Orchestrators (Kubernetes, Nomad, ECS)\n"
    "  - Service Mesh (Istio, Linkerd, Consul, Cilium)\n"
    "  - CI/CD Platforms (GitHub Actions, GitLab CI, Jenkins, "
    "CircleCI, Buildkite, Argo CD)\n"
    "  - Observability Stacks (Prometheus, Grafana, Datadog, "
    "New Relic, Splunk, OpenTelemetry)\n"
    "  - Cloud Networking (VPC, subnets, peering, Transit Gateway, "
    "DNS, load balancers)\n"
    "  - Database Operations (when DBA-shaped — backups, replication, "
    "HA, query tuning)\n"
    "  - Storage Systems (when Storage-Engineer-shaped — block, file, "
    "object, SAN/NAS)\n"
    "  - Cost & FinOps Tools (CloudHealth, Apptio, native cost "
    "explorers)\n"
    "  - Reliability Engineering (capacity planning, chaos "
    "engineering, runbooks, error budgets)\n"
)

_QA_SDET_HINTS = (
    "Family axis hints for QA & SDET roles — consider these "
    "test-engineering dimensions:\n"
    "  - Test Automation Frameworks (Selenium, Playwright, Cypress, "
    "Appium, Puppeteer)\n"
    "  - BDD Tools (Cucumber, SpecFlow, Behave)\n"
    "  - API Testing (Postman, Rest Assured, Karate, Newman)\n"
    "  - Performance Testing (JMeter, Gatling, k6, Locust)\n"
    "  - Test Management (TestRail, Zephyr, Xray, qTest)\n"
    "  - Mobile Testing (XCTest, Espresso, Appium, BrowserStack, "
    "Sauce Labs)\n"
    "  - CI Integration for Tests (test runners on GH Actions / "
    "Jenkins / GitLab CI)\n"
    "  - Defect Tracking (JIRA, Azure Boards, Linear)\n"
    "  - Test Data Management (synthetic data, anonymisation, "
    "fixtures)\n"
    "  - Accessibility Testing (axe, WAVE, screen-reader QA)\n"
    "  - Security Testing Concepts (SAST/DAST/IAST awareness for QA)\n"
)

_ERP_ENTERPRISE_HINTS = (
    "Family axis hints for ERP & Enterprise platform roles — consider "
    "platform-specific dimensions:\n"
    "  - Platform-specific languages (e.g., GlideScript for "
    "ServiceNow; ABAP for SAP; Apex for Salesforce)\n"
    "  - Platform Modules (e.g., ITSM, HRSD, CSM, SecOps for "
    "ServiceNow)\n"
    "  - Integration Patterns (IntegrationHub, REST/SOAP APIs, "
    "MID Server, middleware)\n"
    "  - UI Frameworks (Service Portal, Now Experience UI Builder, "
    "Lightning, SAP Fiori)\n"
    "  - Workflow Engines (Flow Designer, Workflow Editor, "
    "Process Builder)\n"
    "  - Discovery & CMDB (asset discovery, configuration items, "
    "dependency mapping)\n"
    "  - Event Management (alert correlation, event rules, "
    "MID Server connectors)\n"
    "  - Performance Analytics (dashboards, indicators, reporting)\n"
    "  - Platform Administration (instance management, upgrades, "
    "user access)\n"
    "  - Update Sets / Application Repository (versioning, "
    "deployment, scoped apps)\n"
    "  - Custom Apps & Scoped Applications (app development, ACLs, "
    "best practices)\n"
)


def _se_subarchetype_hints(canonical_name: str) -> str:
    """Route Software Engineering family by canonical_name.
    Backend / full-stack default → empty (system prompt covers them)."""
    cn = (canonical_name or "").lower()
    if "blockchain" in cn:
        return _BLOCKCHAIN_HINTS
    if "ios" in cn or "android" in cn or "mobile" in cn:
        return _MOBILE_HINTS
    if "ai engineer" in cn:
        return _AI_ENGINEER_HINTS
    if "frontend" in cn:
        return _FRONTEND_HINTS
    return ""


def format_family_hints_block(family: str, canonical_name: str) -> str:
    """Pick a narrative axis-hints paragraph for the role's family
    (and, for Software Engineering, sub-archetype). Return empty string
    when the family is unknown or a Software Engineering sub-archetype
    doesn't need extra hints (backend / full-stack)."""
    fam = (family or "").strip()
    if fam == "Security":
        return _SECURITY_HINTS
    if fam == "Software Engineering":
        return _se_subarchetype_hints(canonical_name)
    if fam == "Data & ML":
        return _DATA_ML_HINTS
    if fam == "Infrastructure & Platform":
        cn = (canonical_name or "").lower()
        if "storage" in cn:
            return _STORAGE_ENGINEER_HINTS
        return _INFRA_PLATFORM_HINTS
    if fam == "QA & SDET":
        return _QA_SDET_HINTS
    if fam == "ERP & Enterprise":
        return _ERP_ENTERPRISE_HINTS
    return ""


def format_adjacent_dim_block(
    adjacent_dimensions_by_role: dict[str, list[dict]],
) -> str:
    """Render adjacent roles' existing dimensions so the LLM can flag
    overlaps. ``adjacent_dimensions_by_role`` is a dict keyed by role
    display name, with values being lists of dim dicts (each dim dict has
    at least ``name`` and ``description`` keys, optionally ``id``).

    Empty/missing adjacent dims → placeholder so the LLM knows it's
    bootstrapping rather than expected to flag overlaps."""
    if not adjacent_dimensions_by_role:
        return (
            "Adjacent roles' existing dimensions:\n"
            "(no adjacent-role dimensions in the catalog yet — emit empty "
            "overlap_flags lists, the catalog is bootstrapping.)"
        )
    blocks = ["Adjacent roles' existing dimensions (flag overlaps when relevant):"]
    for role_name, dims in adjacent_dimensions_by_role.items():
        if not dims:
            continue
        blocks.append(f"\n  ── {role_name} ──")
        for d in dims:
            dim_id = d.get("tentative_id") or d.get("id") or d.get("dim_id") or ""
            dim_name = d.get("name") or "?"
            desc = (d.get("description") or "").strip()
            # Truncate description to keep prompt budget reasonable across
            # many adjacent roles.
            if len(desc) > 160:
                desc = desc[:157] + "…"
            label = f"{dim_id}: {dim_name}" if dim_id else dim_name
            blocks.append(f"    - {label} — {desc}")
    return "\n".join(blocks)
