"""
Insert comprehensive role aliases for all roles in the skill-library.
Skips any alias_lower that already exists for the role (idempotent).
"""
import os
import psycopg2
from dotenv import load_dotenv
load_dotenv()

# All aliases per canonical role display_name.
# These cover every common JD title variation seen in the wild.
ALIASES: dict[str, list[str]] = {
    "Backend Engineer": [
        # Generic software engineer titles that default to backend
        "Software Engineer", "Software Developer", "SWE", "Software Engineer II",
        "Senior Software Engineer", "Staff Software Engineer",
        # Java
        "Java Developer", "Java Engineer", "Java Backend Developer",
        "Java Software Engineer", "Java Programmer", "Java Spring Developer",
        "Spring Boot Developer", "J2EE Developer", "Java Full Stack Developer",
        # Python
        "Python Developer", "Python Engineer", "Python Backend Developer",
        "Django Developer", "FastAPI Developer", "Flask Developer",
        # Node / JS
        "Node.js Developer", "NodeJS Developer", "Node Developer",
        "Node.js Engineer", "Express Developer",
        # Go / Rust / others
        "Go Developer", "Golang Developer", "Go Engineer",
        "Rust Developer", "Ruby Developer", "Ruby on Rails Developer",
        "PHP Developer", "C# Developer", ".NET Developer",
        # API / Microservices
        "API Developer", "API Engineer", "REST API Developer",
        "Microservices Developer", "Microservices Engineer",
        "Backend API Developer", "Backend API Engineer",
        # General
        "Server Side Developer", "Server-Side Developer",
        "Application Developer", "Application Engineer",
        "Software Application Developer",
    ],
    "Frontend Engineer": [
        # React
        "React Developer", "React Engineer", "React.js Developer",
        "ReactJS Developer", "React Frontend Developer",
        # Angular / Vue
        "Angular Developer", "Angular Engineer", "AngularJS Developer",
        "Vue Developer", "Vue.js Developer", "VueJS Developer",
        # Generic web
        "JavaScript Developer", "JS Developer", "TypeScript Developer",
        "Web Developer", "Web Engineer", "Web Application Developer",
        "Web Frontend Developer", "UI Developer", "UI Engineer",
        "UX Engineer", "Frontend Web Developer", "Client Side Developer",
        # HTML/CSS focused
        "HTML Developer", "CSS Developer",
    ],
    "Full Stack Engineer": [
        "Full Stack Software Engineer", "Full Stack Web Developer",
        "Web Application Engineer", "Web Software Engineer",
        "MEAN Stack Developer", "MERN Stack Developer",
        "LAMP Stack Developer", "Full Stack Java Developer",
        "Full Stack Python Developer", "Full Stack Node Developer",
        "Full Stack React Developer",
    ],
    "DevOps Engineer": [
        # SRE
        "Site Reliability Engineer", "SRE", "Site Reliability Engineer (SRE)",
        # Platform / Infra
        "Platform Engineer", "Infrastructure Engineer",
        "Cloud Infrastructure Engineer", "Cloud Operations Engineer",
        "Infrastructure Software Engineer",
        # CI/CD
        "CI/CD Engineer", "Build Engineer", "Release Manager",
        # Kubernetes / Cloud
        "Kubernetes Engineer", "Kubernetes Administrator",
        "Cloud DevOps Engineer", "DevSecOps Engineer",
        "DevOps Platform Engineer",
    ],
    "Android Engineer": [
        "Native Android Developer", "Android Mobile Developer",
        "Android Application Developer", "Kotlin Developer",
        "Kotlin Engineer", "Android Kotlin Developer",
        "Mobile Developer", "Mobile Application Developer",
        "Mobile Software Engineer", "Mobile Engineer",
    ],
    "Ios engineer": [
        "Swift Developer", "Swift Engineer", "Objective-C Developer",
        "Native iOS Developer", "iOS Application Developer",
        "iOS Mobile Developer", "iOS Software Developer",
        "Mobile iOS Developer", "iPhone Developer",
        "iPhone App Developer", "iPad Developer",
        "iOS Swift Developer", "SwiftUI Developer",
    ],
    "Full Stack Engineer": [
        "Full Stack Software Engineer", "Full Stack Web Developer",
        "Web Application Engineer", "Web Software Engineer",
        "MEAN Stack Developer", "MERN Stack Developer",
        "Full Stack Java Developer", "Full Stack Python Developer",
        "Full Stack Node Developer", "Full Stack React Developer",
        "T-Shaped Developer",
    ],
    "ML Engineer": [
        "Machine Learning Developer", "ML Developer",
        "AI/ML Engineer", "AI ML Engineer",
        "Deep Learning Engineer", "Deep Learning Developer",
        "NLP Engineer", "Natural Language Processing Engineer",
        "Computer Vision Engineer", "CV Engineer",
        "ML Research Engineer", "Research Engineer",
        "ML Software Engineer", "ML Platform Engineer",
        "Recommendation Systems Engineer",
    ],
    "Data Engineer": [
        "Big Data Engineer", "Data Platform Engineer",
        "Analytics Engineer", "Data Infrastructure Engineer",
        "Spark Developer", "Spark Engineer",
        "Kafka Engineer", "Streaming Data Engineer",
        "Hadoop Developer", "Data Warehouse Engineer",
        "ETL Engineer", "ELT Developer",
        "Databricks Engineer", "Snowflake Developer",
        "Data Operations Engineer",
    ],
    "ML Ops Engineer": [
        "ML Platform Engineer", "ML Infrastructure Engineer",
        "AI Operations Engineer", "AI/ML Operations Engineer",
        "Model Operations Engineer", "LLM Ops Engineer",
        "AI Infrastructure Engineer", "ML Systems Engineer",
        "Machine Learning Platform Engineer",
    ],
    "AI Engineer": [
        "LLM Engineer", "Large Language Model Engineer",
        "AI Application Developer", "AI Solutions Engineer",
        "Prompt Engineer", "AI Product Engineer",
        "Conversational AI Engineer", "NLP AI Engineer",
        "AI Integration Engineer", "AI Backend Engineer",
        "AI Software Engineer", "AI Developer II",
        "Foundation Model Engineer", "RAG Engineer",
    ],
    "AI Compliance Officer": [
        "AI Ethics Officer", "AI Policy Officer",
        "Responsible AI Officer", "AI Safety Officer",
        "AI Governance Manager", "AI Risk Officer",
        "AI Regulatory Officer", "AI Ethics Manager",
    ],
    "Cloud Architect": [
        "Cloud Infrastructure Architect", "AWS Architect",
        "Azure Architect", "GCP Architect",
        "Cloud Engineering Architect", "Cloud Design Architect",
        "Enterprise Cloud Architect", "Solutions Architect",
        "Cloud Solutions Engineer", "Principal Cloud Engineer",
    ],
    "Cloud Security Engineer": [
        "AWS Security Engineer", "Azure Security Engineer",
        "GCP Security Engineer", "Cloud SecOps Engineer",
        "Cloud Security Operations Engineer",
        "Cloud Security DevOps Engineer",
        "Cloud Infrastructure Security Engineer",
    ],
    "Cybersecurity Engineer": [
        "Application Security Engineer", "AppSec Engineer",
        "Network Security Engineer", "Information Security Engineer",
        "Security Operations Engineer", "SOC Engineer",
        "Penetration Tester", "Pen Tester",
        "Offensive Security Engineer", "Defensive Security Engineer",
        "Vulnerability Management Engineer",
    ],
    "AR/VR Engineer": [
        "Mixed Reality Engineer", "MR Engineer",
        "Spatial Computing Engineer", "Extended Reality Engineer",
        "XR Developer", "Immersive Technology Engineer",
        "3D Developer", "Unity Developer", "Unity Engineer",
        "Unreal Engine Developer", "Game Developer",
        "HoloLens Developer", "Metaverse Engineer",
    ],
    "Hybrid Mobile Developer": [
        "React Native Developer", "React Native Engineer",
        "Flutter Developer", "Flutter Engineer",
        "Xamarin Developer", "Ionic Developer",
        "Cross Platform Developer", "Mobile App Engineer",
        "PWA Developer", "Progressive Web App Developer",
    ],
    "Video Codec Engineer": [
        "Video Engineer", "Media Engineer",
        "Streaming Engineer", "Video Processing Engineer",
        "Video Technology Engineer", "AV Engineer",
        "Media Software Engineer", "Video Software Engineer",
        "Video Infrastructure Engineer", "Codec Developer",
        "Video Streaming Engineer", "Video Compression Developer",
    ],
}


ACRONYMS = {
    "swe", "sre", "be", "fe", "fse", "mle", "de", "ae",
    "swe ii", "cv", "mr", "xr",
}


def _alias_type(text: str) -> str:
    t = text.strip().lower()
    if t in ACRONYMS or (len(t) <= 4 and t.replace("/", "").replace(".", "").isalpha()):
        return "ACRONYM"
    return "FULL_NAME"


def main():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'), sslmode=os.getenv('DB_SSLMODE', 'disable'),
    )

    total_inserted = 0
    total_skipped = 0

    try:
        for role_name, aliases in ALIASES.items():
            # Get role id
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT id FROM "skill-library".roles WHERE display_name = %s',
                    (role_name,)
                )
                row = cur.fetchone()
            if not row:
                print(f"SKIP (role not found): {role_name}")
                continue
            role_id = row[0]

            # Get existing alias_lowers for this role
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT alias_lower FROM "skill-library".role_aliases WHERE role_id = %s',
                    (role_id,)
                )
                existing = {r[0] for r in cur.fetchall()}

            inserted = 0
            skipped = 0
            for alias_text in aliases:
                alias_lower = alias_text.strip().lower()
                if alias_lower in existing:
                    skipped += 1
                    continue
                with conn.cursor() as cur:
                    cur.execute(
                        'INSERT INTO "skill-library".role_aliases (role_id, alias_text, alias_type) VALUES (%s, %s, %s::"skill-library".alias_type)',
                        (role_id, alias_text.strip(), _alias_type(alias_text))
                    )
                existing.add(alias_lower)
                inserted += 1

            conn.commit()
            total_inserted += inserted
            total_skipped += skipped
            print(f"  {role_name}: +{inserted} inserted, {skipped} skipped")

    finally:
        conn.close()

    print(f"\nDone. Total inserted: {total_inserted}, skipped (already existed): {total_skipped}")


if __name__ == "__main__":
    main()
