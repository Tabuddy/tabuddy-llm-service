"""
Rewrite all primary_responsibility KRA texts with descriptive sentences
and re-embed them using Azure text-embedding-3-small.

Run: python reembed_kras.py
"""
from __future__ import annotations
import os, json, time
import psycopg2
from dotenv import load_dotenv
load_dotenv()

from llm_client import EMBEDDING_MODEL, get_embedding_sync_client

# ── New KRA texts: descriptive 1-2 sentence per responsibility ──────────────
# Keyed by (role display_name, position). Must match exact count in DB.

NEW_KRAS: dict[str, list[str]] = {
    "Backend Engineer": [
        "Implements server-side business logic, REST API endpoints, and microservice handlers using Java, Python, Go, or Node.js to process application requests and enforce business rules.",
        "Designs and maintains RESTful and GraphQL API interfaces with proper authentication, versioning, rate limiting, and documentation for frontend and third-party consumers.",
        "Writes database access logic including SQL queries, ORM mappings, stored procedures, and migration scripts for relational databases like PostgreSQL and MySQL.",
        "Implements request validation, structured error handling, and input sanitization across backend services to ensure predictable and secure API behavior.",
        "Integrates with third-party services, payment gateways, messaging queues like Kafka or RabbitMQ, and internal microservices via HTTP and event-driven patterns.",
        "Identifies and resolves backend performance bottlenecks through query optimization, indexing strategies, connection pooling, and distributed caching with Redis.",
        "Adds structured logging, metrics, distributed tracing, and alerting to improve system observability and support production debugging.",
        "Configures Docker containers, deployment descriptors, environment variables, and CI/CD pipeline stages for backend service releases.",
        "Investigates and resolves production incidents, API bugs, and service degradation through root cause analysis, hotfixes, and post-mortems.",
    ],
    "Frontend Engineer": [
        "Builds responsive user interfaces and interactive web components using React, Vue, or Angular with TypeScript, HTML5, and modern CSS for browser-based applications.",
        "Implements client-side state management using Redux, Zustand, Vuex, or Context API to handle complex application data flows and user session state.",
        "Integrates frontend views with RESTful or GraphQL backend APIs, handling async data fetching, loading states, error boundaries, and cache invalidation.",
        "Ensures WCAG accessibility compliance by implementing semantic HTML, ARIA attributes, keyboard navigation, and screen reader compatibility.",
        "Optimizes web application performance through code splitting, lazy loading, tree shaking, bundle size reduction, and Core Web Vitals improvements.",
        "Implements form validation, user input sanitization, field-level error messaging, and multi-step form flows in client-side applications.",
        "Maintains reusable component libraries, design system tokens, and shared UI patterns across the application to ensure visual and behavioral consistency.",
        "Collaborates with UX designers to refine interaction details, animations, responsive breakpoints, and micro-interaction behavior.",
        "Supports production releases by diagnosing cross-browser compatibility issues, investigating UI regressions, and resolving frontend performance degradation.",
    ],
    "Full Stack Engineer": [
        "Implements complete product features end-to-end from database schema design through backend API to frontend UI using JavaScript, TypeScript, Python, or Ruby on Rails.",
        "Builds and integrates client-side React or Vue components with server-side Node.js or Django APIs, managing bidirectional data flow across frontend and backend layers.",
        "Handles HTTP request and response cycles, session and token management, cookie handling, and state synchronization between client and server.",
        "Writes application logic across the presentation layer, service layer, and data access layer of web applications following MVC or clean architecture patterns.",
        "Designs and queries relational databases like PostgreSQL and document stores like MongoDB, writing migrations, indexes, and optimized queries.",
        "Debugs full-stack issues that span frontend rendering, API behavior, database queries, and infrastructure configuration to identify root causes.",
        "Optimizes application performance from database query efficiency through API response latency to frontend rendering speed and bundle size.",
        "Works closely with product managers and UX designers to translate requirements and wireframes into working software features through iterative development.",
        "Delivers features through CI/CD pipelines using automated tests, staged rollouts, feature flags, and incremental deployments.",
    ],
    "DevOps Engineer": [
        "Builds and maintains CI/CD pipelines using Jenkins, GitHub Actions, GitLab CI, or CircleCI to automate build, test, security scanning, and deployment workflows.",
        "Manages container orchestration with Kubernetes and Docker, deploying applications as pods, managing namespaces, and configuring auto-scaling across cloud environments.",
        "Provisions and manages cloud infrastructure on AWS, Azure, or GCP using Terraform or CloudFormation to enforce infrastructure-as-code standards.",
        "Manages release management processes including environment promotion gates, deployment approval workflows, change management records, and rollback procedures.",
        "Monitors CI/CD pipeline reliability, identifies bottlenecks in delivery workflows, and improves deployment frequency, lead time, and failure recovery rate.",
        "Sets up and maintains development, staging, and production environments with consistent configuration management using Ansible, Puppet, or Chef.",
        "Responds to deployment failures, infrastructure incidents, and environment misconfiguration issues to restore service availability and prevent recurrence.",
        "Collaborates with development teams to improve build processes, reduce deployment friction, containerize applications, and adopt DevOps best practices.",
        "Writes runbooks, deployment guides, incident playbooks, and infrastructure documentation to support operations and knowledge sharing.",
    ],
    "Android Engineer": [
        "Develops native Android features using Kotlin or Java with Android SDK, Jetpack Compose, and Material Design components following platform conventions.",
        "Implements screen navigation, fragment and activity management, bottom navigation, and deep link handling in Android applications using Navigation Component.",
        "Manages Android application lifecycle, background workers with WorkManager, push notifications via FCM, and device permission request flows.",
        "Integrates Android apps with RESTful backend APIs using Retrofit and OkHttp, handling JSON parsing, authentication headers, and network error recovery.",
        "Implements local data persistence using Room database, DataStore, or SharedPreferences to support offline functionality and data caching.",
        "Profiles and resolves Android app performance issues including memory leaks, overdraw, slow renders, and battery drain using Android Studio profiler.",
        "Implements accessibility features including TalkBack support, content descriptions, focus order, and semantic roles for Android UI components.",
        "Investigates and resolves app crashes, ANR issues, and device-specific compatibility bugs using Crashlytics, Firebase, and Android Logcat.",
        "Prepares Android releases by managing build variants, ProGuard configurations, signing keystores, and Google Play Store submission requirements.",
    ],
    "Ios engineer": [
        "Develops native iOS features using Swift or Objective-C with UIKit or SwiftUI, following Apple Human Interface Guidelines for iPhone and iPad applications.",
        "Implements view controller navigation flows, tab bar controllers, modal presentations, and deep link routing in iOS applications.",
        "Manages reactive state updates, view model bindings, and UI logic using Combine, RxSwift, or Swift Concurrency async/await patterns.",
        "Integrates iOS apps with RESTful backend APIs using URLSession or Alamofire, handling JSON decoding, OAuth tokens, and network error recovery.",
        "Implements CoreData, UserDefaults, or Keychain Services for local data persistence, secure storage, and offline data access in iOS apps.",
        "Manages UIApplicationDelegate lifecycle events, background fetch, silent push notifications, and foreground/background state transitions.",
        "Implements Sign In with Apple, OAuth 2.0 flows, and secure Keychain-based session token storage in iOS applications.",
        "Instruments and optimizes iOS app performance including launch time, memory footprint, main-thread blocking, and frame rate using Instruments.",
        "Diagnoses and fixes iOS-specific defects including memory retain cycles, Auto Layout constraint conflicts, and device-model compatibility issues.",
    ],
    "ML Engineer": [
        "Translates product requirements into machine learning system specifications including feature definitions, model architecture choices, and success metric definitions.",
        "Designs end-to-end ML training pipelines and model inference workflows using TensorFlow, PyTorch, or scikit-learn on cloud ML platforms.",
        "Builds model serving infrastructure to deploy trained models as real-time prediction APIs or batch inference jobs using TorchServe, TensorFlow Serving, or SageMaker.",
        "Prepares, cleans, and transforms training datasets, manages feature stores, and builds feature engineering pipelines for model training.",
        "Evaluates model quality using offline metrics like precision, recall, F1, AUC-ROC, and NDCG, comparing against baselines and business acceptance thresholds.",
        "Monitors production model behavior for data drift, concept drift, and prediction performance degradation using monitoring dashboards and alerting.",
        "Manages model versioning, shadow deployments, A/B test rollouts, and safe rollback procedures using MLflow or SageMaker model registry.",
        "Supports offline experimentation, hyperparameter tuning, and online A/B experiments to improve model quality and investigate production performance issues.",
    ],
    "Data Engineer": [
        "Builds data ingestion pipelines to collect data from transactional databases, third-party APIs, event streams, and file sources into centralized data platforms.",
        "Develops batch and real-time streaming data pipelines using Apache Spark, Apache Kafka, Apache Flink, or Airflow for data movement and processing at scale.",
        "Implements data transformation, cleansing, deduplication, and enrichment logic to convert raw source data into analytics-ready curated datasets.",
        "Designs dimensional models, star schemas, data vault structures, and curated data mart tables to support BI tools and self-service analytics consumption.",
        "Implements data quality validation rules, reconciliation checks, and anomaly detection to ensure data completeness, accuracy, and consistency.",
        "Maintains data catalog entries, column-level data lineage, and technical documentation to support data discoverability and governance across the organization.",
        "Monitors pipeline health, SLA breach alerts, and job failure notifications, and performs root cause analysis for data pipeline incidents.",
        "Optimizes pipeline throughput, partitioning strategies, and query performance across cloud data warehouses like Snowflake, BigQuery, or Redshift.",
        "Works with data analysts, data scientists, and business stakeholders to define data models, ingestion schedules, and data delivery requirements.",
    ],
    "AI Engineer": [
        "Translates product requirements into AI-powered features by integrating large language models like GPT-4, Claude, or Gemini into application workflows via API.",
        "Designs and implements prompt engineering workflows, few-shot examples, chain-of-thought patterns, and structured output parsing for AI feature pipelines.",
        "Integrates AI model API responses with application business logic, database writes, event publishing, and downstream service orchestration.",
        "Defines evaluation frameworks, automated test suites, and human feedback loops to measure AI feature quality, accuracy, and consistency.",
        "Implements safety guardrails, content filtering, output validation, fallback handlers, and retry logic for robust and predictable AI feature behavior.",
        "Monitors AI feature behavior in production including response quality metrics, latency percentiles, token cost per request, and error rates.",
        "Optimizes AI pipeline efficiency by tuning model selection, context window usage, prompt caching, and batching strategies to reduce cost and latency.",
        "Documents AI feature capabilities, known limitations, failure modes, prompt versioning, and operational runbooks for engineering and product teams.",
    ],
    "Cloud Architect": [
        "Defines cloud adoption roadmaps, lift-and-shift vs. refactor migration strategies, and landing zone architectures for workloads moving to AWS, Azure, or GCP.",
        "Designs multi-region and multi-availability-zone cloud infrastructure architectures for high availability, fault tolerance, and horizontal scalability.",
        "Establishes cloud environment standards including VPC topology, workload placement policies, resource tagging taxonomies, and account structure.",
        "Architects blue-green, canary, and immutable infrastructure deployment patterns for zero-downtime releases and fast rollback capabilities.",
        "Designs backup policies, cross-region replication, and disaster recovery runbooks to meet defined RTO and RPO targets for critical workloads.",
        "Designs IAM policies, service control policies, VPC segmentation, private endpoints, and zero-trust network access boundaries for cloud environments.",
        "Establishes cloud governance guardrails including budget alerts, resource quotas, policy-as-code enforcement, and compliance posture management.",
        "Evaluates cloud-native managed services, serverless compute, PaaS databases, and CDN solutions for workload fit and total cost of ownership.",
        "Conducts architecture reviews, approves technical design documents, and guides engineering teams through cloud migration and modernization projects.",
    ],
    "Cloud Security Engineer": [
        "Designs and implements cloud security controls including KMS encryption, secrets management, and data-at-rest protection for AWS, Azure, or GCP workloads.",
        "Reviews cloud infrastructure-as-code configurations for security misconfigurations and implements CIS Benchmark hardening baselines using CSPM tools.",
        "Designs IAM role policies, service account permissions, resource-based policies, and least-privilege access controls for cloud workloads and pipelines.",
        "Implements cloud network security controls including security groups, NACLs, VPC service controls, WAF rules, and private service endpoints.",
        "Configures cloud data protection including column-level encryption, DLP scanning policies, and data classification labels for regulated data.",
        "Defines security requirements for cloud deployments including approved services allowlists, container image scanning, and runtime security policies.",
        "Assesses security risk and compliance posture of new cloud services, third-party SaaS integrations, and infrastructure architecture changes.",
        "Supports cloud security incident response including forensic log analysis, containment procedures, and post-incident remediation for cloud-based breaches.",
        "Documents cloud security standards, approved architecture patterns, security exceptions, and remediation guidance for engineering teams.",
    ],
    "Cybersecurity Engineer": [
        "Designs and implements security controls including SIEM integration, endpoint detection and response, identity management, and firewall rule management.",
        "Conducts security posture assessments, vulnerability scans, and penetration testing to identify weaknesses and evaluate overall system security.",
        "Hardens system and service configurations, applies security baselines, removes unnecessary services, and reduces attack surface across infrastructure.",
        "Performs threat modeling, security architecture reviews, and quantitative risk analysis for new product features and infrastructure changes.",
        "Reviews and enforces access control policies, privilege escalation procedures, role-based access control, and identity governance workflows.",
        "Triages vulnerability scan findings, prioritizes remediation based on CVSS scores and exploitability, and tracks remediation to closure.",
        "Builds SIEM detection rules, correlation queries, and alerts to monitor for threat indicators and suspicious activity across systems.",
        "Leads security incident response investigations including forensic analysis, malware triage, containment actions, and post-incident remediation coordination.",
        "Defines secure engineering standards, secure coding guidelines, threat intelligence feeds, and compliance requirements for the organization.",
    ],
    "ML Ops Engineer": [
        "Manages the end-to-end ML model release lifecycle from training job completion through validation gates to production deployment approval.",
        "Orchestrates model serving deployments to production using Kubernetes, MLflow Model Registry, SageMaker, or Kubeflow Serving infrastructure.",
        "Coordinates model promotion workflows across development, staging, and production environments including integration testing and data contract validation.",
        "Sets up model monitoring dashboards, data drift detection, prediction performance tracking, and alert routing for production ML systems.",
        "Maintains model versioning, experiment lineage, and artifact tracking using MLflow, DVC, or Weights & Biases for reproducibility and auditability.",
        "Defines and executes model rollback procedures including traffic shifting, shadow deployment cutover, and incident-triggered rollback automation.",
        "Supports ML platform incidents by diagnosing model serving failures, feature store pipeline breaks, and training environment configuration issues.",
        "Automates ML platform operations including scheduled retraining triggers, pipeline orchestration, evaluation workflows, and alerting configuration.",
        "Validates model performance benchmarks, data schema contracts, and system integration health before signing off on production release readiness.",
        "Maintains ML platform runbooks, on-call escalation playbooks, and deployment procedure documentation for production operations teams.",
    ],
    "AR/VR Engineer": [
        "Designs and implements immersive user interaction flows and spatial UI elements for augmented reality and virtual reality applications using Unity or Unreal Engine.",
        "Builds real-time 3D scene behaviors, physics-driven object interactions, and dynamic environment responses in XR application experiences.",
        "Implements spatial input handling for VR controllers, hand tracking, eye gaze, and gesture recognition across XR device SDKs.",
        "Develops room-scale locomotion systems, 6DoF interaction mechanics, and comfort-aware movement patterns for immersive VR experiences.",
        "Implements world anchors, horizontal and vertical plane detection, and persistent spatial content placement for AR applications using ARKit and ARCore.",
        "Optimizes rendering pipelines and maintains stable 90fps frame rates through draw call batching, occlusion culling, and level-of-detail management.",
        "Handles device-specific SDK differences and capability gaps across Meta Quest, HoloLens, ARKit, and ARCore runtime environments.",
        "Debugs visual artifacts, tracking jitter, controller input latency, and interaction edge cases in immersive XR application experiences.",
        "Integrates 3D character animations, spatial audio spatialization, and interactive assets into real-time immersive application experiences.",
    ],
    "Hybrid Mobile Developer": [
        "Implements cross-platform mobile features using React Native, Flutter, or Ionic targeting both iOS and Android platforms from a shared codebase.",
        "Builds mobile navigation flows, tab bar routing, modal sheets, and deep link handling using React Navigation or Flutter Navigator.",
        "Manages cross-platform client-side state using Redux, Provider, Riverpod, or MobX for consistent application behavior across iOS and Android.",
        "Integrates cross-platform mobile apps with REST and GraphQL backend APIs, handling JWT authentication, token refresh, and error recovery.",
        "Accesses native device capabilities including camera, location services, Bluetooth, and biometric authentication through React Native or Flutter plugins.",
        "Adapts shared UI codebase to handle platform-specific constraints including different navigation paradigms, gesture behaviors, and OS-level UI patterns.",
        "Debugs cross-platform mobile issues including native module bridge failures, platform-specific rendering differences, and build configuration conflicts.",
        "Optimizes React Native or Flutter app startup time, list rendering performance, image caching, and memory consumption.",
        "Manages build configurations, code signing certificates, and app store release submissions for both Google Play and Apple App Store.",
    ],
    "AI Compliance Officer": [
        "Defines AI governance frameworks including fairness standards, transparency obligations, explainability requirements, and human oversight accountability structures.",
        "Reviews AI use cases and model deployments against applicable regulations, internal ethics policies, and governance guidelines prior to production approval.",
        "Maps AI system behaviors and data processing activities to regulatory requirements including EU AI Act, GDPR, CCPA, and sector-specific compliance frameworks.",
        "Assesses personal data usage, retention schedules, consent mechanisms, and cross-border transfer requirements for AI systems handling sensitive information.",
        "Evaluates AI models for bias in protected attributes, explainability limitations, and transparency requirements in automated decision-making contexts.",
        "Manages AI deployment approval workflows, periodic reassessment calendars, and conditional authorization records for production AI systems.",
        "Monitors deployed AI systems for compliance policy drift, regulatory changes, and emerging requirements affecting existing AI deployments.",
        "Coordinates AI incident response procedures, regulatory breach notification, audit investigation support, and remediation tracking for compliance issues.",
        "Maintains audit trail documentation, evidence records, and compliance certification artifacts for AI system approvals and regulatory review.",
    ],
    "Video Codec Engineer": [
        "Designs and implements video compression algorithms for H.264 AVC, H.265 HEVC, AV1, and VP9 codecs targeting broadcast, streaming, and conferencing use cases.",
        "Implements video decompression and decoder pipelines for real-time playback and file-based transcoding in video processing applications.",
        "Tunes encoder bitrate ladders, CRF quality parameters, and rate control algorithms to achieve target quality-per-bit efficiency across content types.",
        "Optimizes codec performance for CPU, GPU, and hardware encoder targets including NVIDIA NVENC, Intel Quick Sync, and Apple VideoToolbox acceleration.",
        "Implements container format muxing and demuxing for MP4, MKV, and CMAF, and ensures codec stream interoperability across player and browser environments.",
        "Analyzes blocking artifacts, mosquito noise, banding, and ringing defects in encoded video to diagnose encoder tuning and rate control issues.",
        "Packages video streams for adaptive bitrate delivery using HLS and MPEG-DASH manifests with DRM integration for CDN distribution.",
        "Investigates video pipeline stability issues, decoder crashes, bitstream corruption, and frame-dropping under constrained bandwidth conditions.",
        "Validates encoder output quality using objective metrics including PSNR, SSIM, and VMAF scores, and validates against subjective viewing test results.",
    ],
}


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_embedding_sync_client()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


def main():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'), sslmode=os.getenv('DB_SSLMODE', 'disable'),
    )

    total_updated = 0
    try:
        for role_name, kra_texts in NEW_KRAS.items():
            print(f"\n--- {role_name} ({len(kra_texts)} KRAs) ---")

            # Fetch existing rows for this role
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT rk.id, rk.position
                    FROM "skill-library".role_kras rk
                    JOIN "skill-library".roles r ON r.id = rk.role_id
                    WHERE r.display_name = %s
                      AND rk.source_field = %s
                    ORDER BY rk.position
                ''', (role_name, 'primary_responsibility'))
                rows = cur.fetchall()

            if len(rows) != len(kra_texts):
                print(f"  SKIP — DB has {len(rows)} rows, new texts has {len(kra_texts)}. Fix count first.")
                continue

            # Embed all texts for this role in one API call
            print(f"  Embedding {len(kra_texts)} texts...")
            vectors = embed_texts(kra_texts)
            print(f"  Got {len(vectors)} vectors (dim={len(vectors[0])})")

            # Update each row
            with conn.cursor() as cur:
                for (row_id, position), new_text, vec in zip(rows, kra_texts, vectors):
                    vec_str = "[" + ",".join(str(v) for v in vec) + "]"
                    cur.execute('''
                        UPDATE "skill-library".role_kras
                        SET kra_text = %s,
                            kra_embedding = %s::"skill-library".vector
                        WHERE id = %s
                    ''', (new_text, vec_str, row_id))
                    total_updated += 1
            conn.commit()
            print(f"  Updated {len(rows)} rows.")
            time.sleep(0.3)  # gentle rate limiting

    finally:
        conn.close()

    print(f"\nDone. Total rows updated: {total_updated}")


if __name__ == "__main__":
    main()
