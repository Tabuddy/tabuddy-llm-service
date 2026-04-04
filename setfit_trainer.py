"""SetFit Training Script — ported from setfit-model/train_setfit og.ipynb

Usage:
    python setfit_trainer.py --layer tier1
    python setfit_trainer.py --layer tier2
    python setfit_trainer.py --layer tier3
    python setfit_trainer.py --layer all

Models are saved to ./setfit_models/{tier1_router,tier2_digital,tier3_app_eng}/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA
# ══════════════════════════════════════════════════════════════════════════════

# ── Tier 1: The "Gravity" Router  (Digital | Physical | Service) ──────────────
TIER1_DATA = [
    # Architect — Digital
    {"text": "Solutions Architect designing AWS Microservices and API gateways.", "label": "Digital"},
    {"text": "Cloud Architect implementing Azure serverless functions and event grids.", "label": "Digital"},
    {"text": "Software Architect defining micro-frontend patterns and backend interactions.", "label": "Digital"},
    {"text": "System Architect designing fault-tolerant distributed systems in Golang.", "label": "Digital"},
    {"text": "Data Architect modeling snowflake schemas for petabyte-scale warehouses.", "label": "Digital"},
    {"text": "Security Architect defining IAM policies and zero-trust networks.", "label": "Digital"},
    {"text": "Enterprise Architect mapping legacy monoliths to microservices transition plans.", "label": "Digital"},
    {"text": "AI Architect designing neural network pipelines and GPU cluster orchestration.", "label": "Digital"},
    {"text": "Network Architect planning virtual private clouds and software-defined networking.", "label": "Digital"},
    {"text": "Integration Architect building event-driven architectures using Apache Kafka.", "label": "Digital"},
    # Architect — Physical
    {"text": "Naval Architect designing ships and maritime structures.", "label": "Physical"},
    {"text": "Civil Architect creating blueprints for commercial skyscrapers.", "label": "Physical"},
    {"text": "Landscape Architect planning urban parks and green spaces.", "label": "Physical"},
    {"text": "Marine Architect designing hull structures for deep-sea oil rigs.", "label": "Physical"},
    {"text": "Golf Course Architect shaping terrain and drainage for championship links.", "label": "Physical"},
    {"text": "Restoration Architect preserving historical masonry and timber structures.", "label": "Physical"},
    {"text": "Industrial Architect planning layout for an automated Tesla gigafactory.", "label": "Physical"},
    {"text": "Residential Architect drafting blueprints for net-zero energy custom homes.", "label": "Physical"},
    # Architect — Service
    {"text": "Business Architect designing enterprise capabilities and process flows.", "label": "Service"},
    {"text": "Organizational Architect restructuring departments for operational efficiency.", "label": "Service"},
    {"text": "Information Architect structuring taxonomy for a corporate knowledge base.", "label": "Service"},
    {"text": "Solution Architect (Pre-sales) mapping customer pain points to software capabilities.", "label": "Service"},
    {"text": "Brand Architect defining corporate identity and visual language guidelines.", "label": "Service"},
    {"text": "Policy Architect drafting governance frameworks for regulatory compliance.", "label": "Service"},
    # Engineer — Digital
    {"text": "DevOps Engineer managing CI/CD pipelines and scaling systems.", "label": "Digital"},
    {"text": "Software Engineer developing scalable REST APIs in Python.", "label": "Digital"},
    {"text": "Platform Engineer automating infrastructure with Terraform.", "label": "Digital"},
    {"text": "QA Engineer writing automated Selenium tests for web applications.", "label": "Digital"},
    {"text": "Machine Learning Engineer deploying LLMs to production endpoints using Docker.", "label": "Digital"},
    {"text": "Front-End Engineer optimizing critical rendering paths for e-commerce sites.", "label": "Digital"},
    {"text": "Data Engineer building real-time streaming pipelines with Flink.", "label": "Digital"},
    # Engineer — Physical
    {"text": "Mechanical Engineer working on HVAC thermodynamics and piping.", "label": "Physical"},
    {"text": "Chemical Engineer optimizing polymer manufacturing processes.", "label": "Physical"},
    {"text": "Civil Engineer overseeing highway construction projects.", "label": "Physical"},
    {"text": "Petroleum Engineer designing downhole drilling assemblies for oil extraction.", "label": "Physical"},
    {"text": "Biomedical Engineer prototyping prosthetics with biocompatible materials.", "label": "Physical"},
    {"text": "Structural Engineer calculating wind loads for high-rise steel frames.", "label": "Physical"},
    {"text": "Automotive Engineer testing combustion engine efficiency and emissions.", "label": "Physical"},
    # Engineer — Service
    {"text": "Sales Engineer providing pre-sales technical support to clients.", "label": "Service"},
    {"text": "Forward Deployed Engineer managing customer implementations onsite.", "label": "Service"},
    {"text": "Customer Success Engineer troubleshooting integration issues for enterprise clients.", "label": "Service"},
    {"text": "Technical Support Engineer resolving ticket escalations for SaaS platforms.", "label": "Service"},
    {"text": "Process Engineer mapping six sigma workflows for call centers.", "label": "Service"},
    # Designer — Digital
    {"text": "UX Designer creating Figma wireframes for web accessibility.", "label": "Digital"},
    {"text": "Product Designer prototyping mobile app interactions.", "label": "Digital"},
    {"text": "Web Designer building responsive layouts with CSS grid.", "label": "Digital"},
    {"text": "Motion Designer animating micro-interactions for mobile app onboarding.", "label": "Digital"},
    {"text": "Level Designer scripting quests and events for RPG video games.", "label": "Digital"},
    {"text": "Conversational Designer writing scripts and flows for AI chatbots.", "label": "Digital"},
    # Designer — Physical
    {"text": "Industrial Designer using SolidWorks for injection molding.", "label": "Physical"},
    {"text": "Interior Designer planning office layouts and furniture selection.", "label": "Physical"},
    {"text": "Fashion Designer creating patterns for textile manufacturing.", "label": "Physical"},
    {"text": "Packaging Designer engineering cardboard structures to survive shipping.", "label": "Physical"},
    # Designer — Service
    {"text": "Instructional Designer creating marketing collateral and learning modules.", "label": "Service"},
    {"text": "Graphic Designer producing brochures and corporate branding.", "label": "Service"},
    {"text": "Learning Designer creating storyboards for corporate compliance training.", "label": "Service"},
    {"text": "Strategic Designer leading workshops to define business value propositions.", "label": "Service"},
    # Developer — Digital
    {"text": "Full Stack Developer building React interfaces and Node.js APIs.", "label": "Digital"},
    {"text": "Backend Developer optimizing SQL queries and database schemas.", "label": "Digital"},
    {"text": "Mobile Developer building iOS apps with Swift.", "label": "Digital"},
    {"text": "Firmware Developer writing low-level drivers for IoT devices.", "label": "Digital"},
    {"text": "API Developer building GraphQL schemas for data federation.", "label": "Digital"},
    # Developer — Physical
    {"text": "Real Estate Developer managing site acquisition and construction permits.", "label": "Physical"},
    {"text": "Land Developer overseeing zoning and environmental assessments.", "label": "Physical"},
    {"text": "Solar Developer scouting land parcels for photovoltaic farm installations.", "label": "Physical"},
    # Developer — Service
    {"text": "Business Developer focused on lead generation and market expansion.", "label": "Service"},
    {"text": "Community Developer managing partnerships and local relations.", "label": "Service"},
    {"text": "Talent Developer coaching high-potential employees for leadership roles.", "label": "Service"},
    # Analyst — Digital
    {"text": "Cyber Analyst monitoring threat detection logs and dashboards.", "label": "Digital"},
    {"text": "Data Analyst visualizing trends using Tableau and SQL.", "label": "Digital"},
    {"text": "Product Analyst measuring feature adoption using Mixpanel funnels.", "label": "Digital"},
    {"text": "Marketing Analyst tracking ad spend ROI across digital channels.", "label": "Digital"},
    {"text": "Malware Analyst reverse engineering ransomware binaries.", "label": "Digital"},
    # Analyst — Physical
    {"text": "Lab Analyst running chemical titration and material strength tests.", "label": "Physical"},
    {"text": "Environmental Analyst testing groundwater for heavy metal contamination.", "label": "Physical"},
    {"text": "Supply Chain Analyst optimizing warehouse inventory and truck routes.", "label": "Physical"},
    # Analyst — Service
    {"text": "Financial Analyst analyzing P&L and market trends.", "label": "Service"},
    {"text": "Business Analyst gathering requirements for process improvements.", "label": "Service"},
    {"text": "Credit Analyst assessing borrower risk for loan approval.", "label": "Service"},
    {"text": "Risk Analyst modeling portfolio exposure to market volatility.", "label": "Service"},
    # Product — Digital
    {"text": "Product Manager managing SaaS backlog and user stories.", "label": "Digital"},
    {"text": "Technical Product Manager defining API specifications.", "label": "Digital"},
    {"text": "AI Product Manager defining requirements for recommendation engines.", "label": "Digital"},
    # Product — Physical
    {"text": "Hardware Product Manager coordinating PCB fabrication and casing design.", "label": "Physical"},
    {"text": "Pharma Product Manager guiding a new drug through clinical trials.", "label": "Physical"},
    # Product — Service
    {"text": "Financial Product Manager designing new credit card reward tiers.", "label": "Service"},
    {"text": "Insurance Product Manager defining coverage terms for cyber liability.", "label": "Service"},
    # Edge Cases
    {"text": "Network Engineer configuring TCP/IP routers and firewalls.", "label": "Digital"},
    {"text": "Electrical Network Engineer managing high voltage substations.", "label": "Physical"},
    {"text": "Provider Network Manager contracting with physicians for insurance.", "label": "Service"},
    {"text": "Linux Systems Admin managing virtualization and servers.", "label": "Digital"},
    {"text": "Control Systems Engineer programming PLCs and SCADA for conveyors.", "label": "Physical"},
    {"text": "HR Systems Analyst configuring Workday payroll and benefits.", "label": "Service"},
]

# ── Tier 2: Digital Segmenter ─────────────────────────────────────────────────
TIER2_DIGITAL_DATA = [
    # App_Engineering
    {"text": "Senior Java Developer with Spring Boot experience.", "label": "App_Engineering"},
    {"text": "Frontend Developer using React and Redux.", "label": "App_Engineering"},
    {"text": "Mobile App Developer specializing in iOS and Swift.", "label": "App_Engineering"},
    {"text": "Full stack engineer building REST APIs and UI components.", "label": "App_Engineering"},
    {"text": "Software Engineer working on distributed microservices architecture.", "label": "App_Engineering"},
    {"text": "Embedded Systems Engineer writing C++ firmware.", "label": "App_Engineering"},
    {"text": "Android Developer using Kotlin and Jetpack Compose.", "label": "App_Engineering"},
    {"text": "Backend developer optimizing database transactions.", "label": "App_Engineering"},
    {"text": "Python Backend Developer using Django REST framework.", "label": "App_Engineering"},
    {"text": "Node.js Developer building scalable APIs with Express.", "label": "App_Engineering"},
    # App_Engineering with infra-adjacent tooling (they USE Docker/AWS/CI-CD, not MANAGE infra)
    {"text": "Full Stack Developer building React SPAs and Node.js REST APIs, containerized with Docker on AWS.", "label": "App_Engineering"},
    {"text": "Web Application Developer using TypeScript and React, deploying via CI/CD pipelines to cloud.", "label": "App_Engineering"},
    {"text": "Software Engineer developing full-stack features with React frontend and Node.js backend, deployed using Docker.", "label": "App_Engineering"},
    {"text": "Application Developer implementing RESTful APIs and GraphQL services, with CI/CD via GitHub Actions.", "label": "App_Engineering"},
    {"text": "Senior Software Engineer designing scalable web apps with React and Express, containerized deployments on AWS.", "label": "App_Engineering"},
    {"text": "Full Stack Engineer building production web applications with TypeScript, Docker, and AWS.", "label": "App_Engineering"},
    {"text": "Backend Developer building Node.js microservices with Docker containers and automated CI/CD pipelines.", "label": "App_Engineering"},
    {"text": "React Developer building responsive frontends with Next.js, integrating with RESTful APIs on cloud backends.", "label": "App_Engineering"},
    # Data_Intelligence
    {"text": "Data Scientist building predictive models with PyTorch.", "label": "Data_Intelligence"},
    {"text": "Data Engineer designing ETL pipelines in Airflow and Snowflake.", "label": "Data_Intelligence"},
    {"text": "Machine Learning Engineer training NLP models.", "label": "Data_Intelligence"},
    {"text": "BI Developer creating dashboards in Tableau and PowerBI.", "label": "Data_Intelligence"},
    {"text": "Big Data Engineer managing Spark clusters and HDFS.", "label": "Data_Intelligence"},
    {"text": "AI Researcher experimenting with GANs and Transformers.", "label": "Data_Intelligence"},
    {"text": "Data Analyst querying SQL databases for business insights.", "label": "Data_Intelligence"},
    {"text": "Database Administrator tuning Postgres performance.", "label": "Data_Intelligence"},
    # Infra_Cloud
    {"text": "DevOps Engineer managing CI/CD pipelines in Jenkins.", "label": "Infra_Cloud"},
    {"text": "Site Reliability Engineer ensuring 99.9% uptime on AWS.", "label": "Infra_Cloud"},
    {"text": "Cloud Architect designing serverless solutions on Azure.", "label": "Infra_Cloud"},
    {"text": "Platform Engineer managing Kubernetes clusters (EKS/AKS).", "label": "Infra_Cloud"},
    {"text": "SysAdmin maintaining Linux servers and bash scripts.", "label": "Infra_Cloud"},
    {"text": "Network Engineer configuring Cisco switches and firewalls.", "label": "Infra_Cloud"},
    {"text": "Infrastructure Engineer automating provisioning with Terraform.", "label": "Infra_Cloud"},
    {"text": "Release Manager orchestrating deployments to production.", "label": "Infra_Cloud"},
    # Infra_Cloud — primary focus is managing/operating infrastructure at organizational scale
    {"text": "Cloud Infrastructure Engineer managing AWS accounts, VPCs, IAM policies and security groups across the organization.", "label": "Infra_Cloud"},
    {"text": "DevOps Lead building and maintaining shared CI/CD platforms and container orchestration for 20+ engineering teams.", "label": "Infra_Cloud"},
    {"text": "Platform Reliability Engineer managing Kubernetes clusters, service mesh, and observability stack at scale.", "label": "Infra_Cloud"},
    # Product_Design
    {"text": "Product Manager defining user stories for a fintech app.", "label": "Product_Design"},
    {"text": "UI/UX Designer creating wireframes in Figma.", "label": "Product_Design"},
    {"text": "Technical Product Manager maximizing agile team velocity.", "label": "Product_Design"},
    {"text": "Product Owner prioritizing the sprint backlog in Jira.", "label": "Product_Design"},
    {"text": "User Researcher conducting usability testing sessions.", "label": "Product_Design"},
    {"text": "Interaction Designer prototyping user journeys.", "label": "Product_Design"},
    {"text": "Head of Product outlining quarterly roadmap strategy.", "label": "Product_Design"},
    {"text": "Visual Designer establishing design systems and style guides.", "label": "Product_Design"},
    # Cyber_Security
    {"text": "Security Engineer conducting vulnerability assessments.", "label": "Cyber_Security"},
    {"text": "GRC Analyst managing compliance audits (SOC2, ISO).", "label": "Cyber_Security"},
    {"text": "Penetration Tester finding exploits in web applications.", "label": "Cyber_Security"},
    {"text": "SOC Analyst monitoring SIEM logs for intrusions.", "label": "Cyber_Security"},
    {"text": "Application Security Engineer reviewing code for flaws.", "label": "Cyber_Security"},
    {"text": "Identity Access Management specialist.", "label": "Cyber_Security"},
    {"text": "Network Security Engineer hardening firewall rules.", "label": "Cyber_Security"},
    {"text": "CISO defining enterprise security policy.", "label": "Cyber_Security"},
]

# ── Tier 3: App Engineering Stack Classifier ──────────────────────────────────
TIER3_APP_ENG_DATA = [
    # Stack_Java
    {"text": "Java Backend Developer with Spring Boot and Hibernate.", "label": "Stack_Java"},
    {"text": "Senior Java Engineer building microservices with Maven.", "label": "Stack_Java"},
    {"text": "Enterprise Java Developer using Jakarta EE and JBoss.", "label": "Stack_Java"},
    {"text": "Java Architect designing scalable systems with Kafka.", "label": "Stack_Java"},
    {"text": "Backend Engineer proficient in Java 17 and Gradle.", "label": "Stack_Java"},
    {"text": "Spring Developer building REST APIs and SOAP services.", "label": "Stack_Java"},
    {"text": "Java Programmer utilizing JDBC for database connectivity.", "label": "Stack_Java"},
    # Stack_Python
    {"text": "Python Backend Developer using Django and DRF.", "label": "Stack_Python"},
    {"text": "Backend Engineer building APIs with FastAPI and SQLAlchemy.", "label": "Stack_Python"},
    {"text": "Python Developer writing scripts and Flask applications.", "label": "Stack_Python"},
    {"text": "Server-side developer using Python and Celery for async tasks.", "label": "Stack_Python"},
    {"text": "Python Architect designing microservices with gRPC.", "label": "Stack_Python"},
    # Stack_Node
    {"text": "Node.js Developer building scalable APIs with Express.", "label": "Stack_Node"},
    {"text": "Backend Javascript Engineer with NestJS experience.", "label": "Stack_Node"},
    {"text": "Typescript Backend Developer using TypeORM and Apollo.", "label": "Stack_Node"},
    {"text": "Server-side JS developer using Koa and Mongoose.", "label": "Stack_Node"},
    {"text": "Backend Engineer utilizing Fastify for high performance.", "label": "Stack_Node"},
    # Stack_React
    {"text": "Frontend Developer with expert knowledge of React and Hooks.", "label": "Stack_React"},
    {"text": "UI Engineer building components in React.js and Redux.", "label": "Stack_React"},
    {"text": "Frontend Specialist using Next.js for server-side rendering.", "label": "Stack_React"},
    {"text": "React Developer managing state with Context API and Recoil.", "label": "Stack_React"},
    {"text": "UI/UX Developer implementing designs in React and Tailwind.", "label": "Stack_React"},
    # Stack_ReactNative
    {"text": "Mobile Engineer specializing in cross-platform development with React Native.", "label": "Stack_ReactNative"},
    {"text": "React Native Developer proficient in building iOS and Android apps.", "label": "Stack_ReactNative"},
    {"text": "Frontend Developer transitioning to mobile using Expo and React Native.", "label": "Stack_ReactNative"},
    {"text": "Mobile App Developer using React Navigation and Redux for mobile state management.", "label": "Stack_ReactNative"},
    # Stack_Angular
    {"text": "Angular Developer with TypeScript and RxJS expertise.", "label": "Stack_Angular"},
    {"text": "Frontend Engineer building SPAs with Angular 14+.", "label": "Stack_Angular"},
    {"text": "UI Developer using NgRx for state management.", "label": "Stack_Angular"},
    {"text": "Web Developer proficient in Angular Material and CLI.", "label": "Stack_Angular"},
    # Stack_iOS
    {"text": "iOS Developer building native apps with Swift and Xcode.", "label": "Stack_iOS"},
    {"text": "Mobile Engineer developing iOS applications with SwiftUI.", "label": "Stack_iOS"},
    {"text": "iOS Specialist optimizing app performance using Instruments.", "label": "Stack_iOS"},
    {"text": "Objective-C Developer maintaining historical iOS code.", "label": "Stack_iOS"},
    # Stack_Android
    {"text": "Android Developer using Kotlin and Jetpack Compose.", "label": "Stack_Android"},
    {"text": "Android Engineer maintaining Java-based legacy apps.", "label": "Stack_Android"},
    {"text": "Mobile Architect designing Android clean architecture.", "label": "Stack_Android"},
]


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def train_layer(layer_name: str, data: list[dict], output_path: Path) -> Path:
    """Train a single SetFit layer and save the model."""
    try:
        import pandas as pd
        from datasets import Dataset as HFDataset  # type: ignore
        from sentence_transformers.losses import CosineSimilarityLoss  # type: ignore
        from setfit import SetFitModel, Trainer, TrainingArguments  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"Training dependencies missing: {e}. "
            "Run: uv add setfit sentence-transformers datasets pandas"
        ) from e

    print(f"\n🚀 Starting Training for Layer: {layer_name}")
    print(f"📊 Examples: {len(data)}")

    df = pd.DataFrame(data)
    dataset = HFDataset.from_pandas(df)

    model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    args = TrainingArguments(
        batch_size=8,
        num_iterations=40,
        num_epochs=5,
        output_dir=str(output_path),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        metric="accuracy",
    )

    print(f"⏳ Training {layer_name}...")
    trainer.train()

    if output_path.exists():
        shutil.rmtree(output_path)
    model.save_pretrained(str(output_path))
    print(f"✅ Saved {layer_name} to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train SetFit tier classifiers")
    parser.add_argument(
        "--layer",
        choices=["tier1", "tier2", "tier3", "all"],
        default="all",
        help="Which tier to train",
    )
    args = parser.parse_args()

    base = Path(__file__).parent / "setfit_models"
    base.mkdir(exist_ok=True)

    layers = {
        "tier1": (TIER1_DATA, base / "tier1_router"),
        "tier2": (TIER2_DIGITAL_DATA, base / "tier2_digital"),
        "tier3": (TIER3_APP_ENG_DATA, base / "tier3_app_eng"),
    }

    to_train = list(layers.keys()) if args.layer == "all" else [args.layer]

    for layer_key in to_train:
        data, out_path = layers[layer_key]
        train_layer(layer_key.upper(), data, out_path)

    print("\n🎉 Training complete. Models saved to ./setfit_models/")


if __name__ == "__main__":
    main()
