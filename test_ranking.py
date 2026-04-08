"""Test runner: ranks all resumes in resume-test/ against multiple JDs.

Usage:
    Start server first: uv run uvicorn main:app --host 0.0.0.0 --port 8000
    Then: python test_ranking.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
RESUME_DIR = Path(__file__).parent / "resume-test"

# ── 5 Diverse Job Descriptions ───────────────────────────────────────────────

JDS = {
    "Frontend React Engineer (Mid)": """
We are looking for a Mid-Level Frontend React Engineer to join our product team.

You will be responsible for building and maintaining user-facing features using React, TypeScript, and modern frontend tooling.

Responsibilities:
- Build responsive, accessible UI components in React and TypeScript
- Integrate with RESTful APIs and GraphQL endpoints
- Write unit and integration tests using Jest and Cypress
- Collaborate with designers to implement pixel-perfect UIs
- Optimize rendering performance and Core Web Vitals

Requirements:
- 2-4 years of professional experience in frontend development
- Strong proficiency in React, TypeScript, HTML5, CSS3
- Experience with state management (Redux, Zustand, or Context API)
- Familiarity with testing frameworks (Jest, React Testing Library, Cypress)
- Knowledge of CSS frameworks (Tailwind CSS, Styled Components)

Nice to have:
- Next.js experience
- Design system or component library experience
- Animation libraries (Framer Motion, GSAP)
""",

    "Senior Full Stack Engineer (Node + React)": """
We are hiring a Senior Full Stack Engineer to architect and build our core platform.

Responsibilities:
- Design, develop and maintain full-stack features using React on the front end and Node.js on the back end
- Build and document RESTful APIs, integrate third-party services
- Write modular, testable TypeScript code
- Containerize applications with Docker and implement CI/CD pipelines
- Lead technical decisions and mentor junior engineers
- Collaborate with product managers and designers

Requirements:
- 5-8 years of experience building production web applications
- Expert-level: React, Node.js, TypeScript
- Strong: RESTful APIs, Docker, AWS, PostgreSQL / MongoDB
- Experience with CI/CD (GitHub Actions, Jenkins)
- Leadership experience: mentoring, code reviews, architecture decisions

Nice to have:
- GraphQL, Kubernetes, microservices architecture
- Serverless functions (AWS Lambda)
""",

    "Backend Engineer (Python / Django)": """
We are looking for a Backend Engineer to design and build scalable APIs and data pipelines.

Responsibilities:
- Design and build RESTful APIs and GraphQL services in Python / Django
- Implement data pipelines and ETL processes
- Design database schemas (PostgreSQL, Redis)
- Implement caching strategies and microservices
- Write comprehensive tests and documentation

Requirements:
- 3-6 years of backend development experience
- Strong proficiency in Python and Django / FastAPI
- Experience with PostgreSQL, Redis, RabbitMQ / Kafka
- Knowledge of Docker, Kubernetes, AWS / GCP
- Understanding of REST, GraphQL, gRPC

Nice to have:
- Machine learning / data pipeline experience
- Terraform / infrastructure as code
""",

    "DevOps / Platform Engineer": """
We are hiring a DevOps Engineer to manage our cloud infrastructure and CI/CD pipelines.

Responsibilities:
- Design and maintain Kubernetes clusters on AWS
- Build and optimize CI/CD pipelines (GitHub Actions, ArgoCD)
- Implement monitoring and observability (Prometheus, Grafana, ELK)
- Manage infrastructure as code (Terraform, Pulumi)
- Ensure security compliance and cost optimization

Requirements:
- 4-7 years of DevOps / platform engineering experience
- Strong: Kubernetes, Docker, Terraform, AWS
- Experience with CI/CD, GitOps, ArgoCD
- Monitoring: Prometheus, Grafana, Datadog
- Scripting: Python, Bash, Go

Nice to have:
- Service mesh (Istio, Linkerd)
- Cloud security certifications
""",

    "Senior Data Engineer (Scala / Spark)": """
We are looking for a Senior Data Engineer to build our next-generation data platform.

Responsibilities:
- Design and build scalable data pipelines using Apache Spark and Kafka
- Implement real-time streaming data processing systems
- Design data lake and data warehouse architectures on AWS
- Implement data quality and governance frameworks
- Collaborate with data scientists to operationalize ML models

Requirements:
- 5-8 years of data engineering experience
- Strong: Scala or Python, Apache Spark, Kafka
- Experience with data lakes (AWS S3, Iceberg, Delta Lake)
- Knowledge of SQL, PostgreSQL, Redis
- AWS data services (EMR, Redshift, Kinesis)

Nice to have:
- dbt, Airflow, Prefect
- Machine learning pipeline experience
""",
}

# Expected behavior for each JD (for accuracy analysis)
EXPECTED = {
    "Frontend React Engineer (Mid)": {
        "high": ["Sangaraj JS", "Ankit", "Priyanshu"],  # frontend skills
        "mid": ["Dhruv", "GallaBaby"],  # some overlap
        "low": ["Deepti", "R Prathamesh"],  # too senior/wrong domain
    },
    "Senior Full Stack Engineer (Node + React)": {
        "high": ["Ankit", "Ashish", "GallaBaby", "Dhruv"],
        "mid": ["Sangaraj JS", "Priyanshu"],
        "low": ["Deepti", "R Prathamesh"],
    },
    "Backend Engineer (Python / Django)": {
        "high": ["Deepti", "R Prathamesh"],  # senior/experienced
        "mid": ["Dhruv", "Ashish", "GallaBaby"],
        "low": ["Sangaraj JS", "Priyanshu"],  # frontend-heavy
    },
    "DevOps / Platform Engineer": {
        "high": ["Deepti", "Dhruv", "Ashish"],
        "mid": ["GallaBaby", "R Prathamesh"],
        "low": ["Sangaraj JS", "Priyanshu"],
    },
    "Senior Data Engineer (Scala / Spark)": {
        "high": ["Deepti", "R Prathamesh"],
        "mid": ["Dhruv", "Ashish"],
        "low": ["Sangaraj JS", "Priyanshu", "Ankit"],
    },
}


def batch_rank(jd_name: str, jd_text: str, resume_paths: list[Path]) -> list[dict]:
    """Post resumes to /ranking/batch-rank and return results."""
    fd = requests.post(
        f"{BASE}/ranking/batch-rank",
        data={"jd_text": jd_text},
        files=[("files", (p.name, open(p, "rb"), "application/pdf")) for p in resume_paths],
        timeout=600,
    )
    if fd.status_code != 200:
        print(f"  ⚠️  HTTP {fd.status_code}: {jd_name}")
        try:
            print(f"      {fd.json()}")
        except Exception:
            print(f"      {fd.text[:200]}")
        return []
    session = fd.json()
    return session.get("results", [])


def score_label(s: float) -> str:
    return "[HI]" if s >= 75 else "[MD]" if s >= 50 else "[LO]"


def analyze_results(jd_name: str, results: list[dict]) -> None:
    """Print analysis of ranking results."""
    if not results:
        print("  [WARN] No results -- skipped\n")
        return

    print(f"\n{'='*70}")
    print(f"  JD: {jd_name}")
    print(f"{'='*70}\n")
    print(f"{'Rank':<5}{'Score':<7}{'Name':<25}{'Exp Level':<12}{'Tier Match?':<12}{'Domain':<10}")
    print("─" * 70)

    for r in results:
        name = (r.get("candidate_name") or "?")[:24]
        score = r["final_score"]
        level = r.get("experience_level", "?")[:11]
        tier_ok = "✓" if not r.get("tier_mismatch") else "✗ MISMATCH"
        tier_path = r.get("tier_classification", {}).get("hierarchy_path", "?")[:9]

        expected_bucket = "low"
        for bucket, names in EXPECTED.get(jd_name, {}).items():
            if any(n.lower() in name.lower() for n in names):
                expected_bucket = bucket
                break

        actual_bucket = "high" if score >= 70 else "mid" if score >= 50 else "low"
        accuracy = "✅" if expected_bucket == actual_bucket else "❌"

        print(f"{r['rank']:<5}{score:<7.1f}{name:<25}{level:<12}{tier_ok:<12}{accuracy}")

        # Dimension breakdown for top 3
        if r["rank"] <= 3:
            for d in r.get("dimension_scores", []):
                dim = d["dimension"][:22]
                raw = d["raw_score"]
                w = d["weight"]
                print(f"       {' ':<5}{dim:<25}{raw:<7.1f} w={w:.0%}")

    print()


def main():
    # Discover resumes
    resume_paths = sorted(RESUME_DIR.glob("*.pdf"))
    if not resume_paths:
        print(f"No PDFs found in {RESUME_DIR}")
        return 1

    print(f"Found {len(resume_paths)} resumes:\n  ", end="")
    for p in resume_paths:
        print(f"\n  • {p.name}")
    print()

    # Run batch ranking for each JD
    all_results: dict[str, list[dict]] = {}
    accuracy_hits = 0
    accuracy_total = 0

    for jd_name, jd_text in JDS.items():
        print(f"\n⏳ Ranking against: {jd_name}")
        start = time.time()

        results = batch_rank(jd_name, jd_text, resume_paths)
        all_results[jd_name] = results

        elapsed = time.time() - start
        print(f"   ⏱️  Completed in {elapsed:.1f}s ({len(results)} results)")

        if results:
            # Calculate accuracy
            for r in results:
                name = r.get("candidate_name", "").lower()
                score = r["final_score"]
                expected_bucket = "low"
                for bucket, names in EXPECTED.get(jd_name, {}).items():
                    if any(n.lower() in name for n in names):
                        expected_bucket = bucket
                        break
                actual_bucket = "high" if score >= 70 else "mid" if score >= 50 else "low"
                if expected_bucket == actual_bucket:
                    accuracy_hits += 1
                accuracy_total += 1

        analyze_results(jd_name, results)

    # Summary
    print(f"\n{'='*70}")
    print("  ACCURACY SUMMARY")
    print(f"{'='*70}")
    percent = (accuracy_hits / accuracy_total * 100) if accuracy_total else 0
    print(f"\n  Expected vs Actual bucket matches: {accuracy_hits}/{accuracy_total} ({percent:.0f}%)")
    print(f"  (high=75+, mid=50-74, low=50-)\n")

    # Score range across JDs
    for jd_name, results in all_results.items():
        if results:
            scores = [r["final_score"] for r in results]
            top3 = sorted(scores, reverse=True)[:3]
            print(f"  {jd_name:<35} top-3: {' | '.join(f'{s:.1f}' for s in top3)}")

    # Save raw results for further analysis
    out_path = RESUME_DIR / "ranking_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  📄 Full results saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
