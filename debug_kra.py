"""Debug which sentences match which KRAs and why DevOps scores higher."""
import os, math, sys
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, '.')

import psycopg2
from jd_similarity_matcher import split_rr_sentences, embed_sentences_batch, _cosine

# Full JD text from sample_jds.py
jd_text = """Design requirements on small systems or modules of medium systems (large scale) environment and technical documentation.
Apply basic principles of software engineering and can follow instructions. Provide meaningful feedback on the release process, code review, and design review.
Apply modern software development practices (serverless computing, microservices architecture, CI/CD, infrastructure-as-code, etc.). Work across teams to integrate our systems with existing corporate product platforms.
Participate in technology roadmap and architecture discussions to turn business requirements and vision into reality.
Participate in a tight-knit engineering team employing agile software development practices.
6 months+ experience with Java software programming experience.
6 months+ experience with Cloud technology: GCP, AWS, or Azure.
UI development (e.g. HTML, JavaScript, Angular and Bootstrap).
Experience with backend technologies such as JAVA/J2EE, SpringBoot, SOA and Microservices.
Source code control management systems (e.g. SVN/Git, Github) and build tools like Maven & Gradle.
Agile environments (e.g. Scrum, XP). Relational databases (e.g. SQL Server, MySQL).
Automated Testing: JUnit, Selenium, LoadRunner, SoapUI.
Has basic knowledge of devops, SecOps and FinOps practices and levers that help to build effective, efficient and safe working software."""

sentences = split_rr_sentences(jd_text)
print(f"Sentences ({len(sentences)}):")
for i, s in enumerate(sentences):
    print(f"  [{i}] {s}")

print("\nEmbedding sentences...")
vecs = embed_sentences_batch(sentences)

# Fetch all KRAs
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
    dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'), sslmode=os.getenv('DB_SSLMODE','disable'),
)
cur = conn.cursor()
cur.execute("""
    SELECT r.display_name, rk.position, rk.kra_text, rk.kra_embedding::text
    FROM "skill-library".role_kras rk
    JOIN "skill-library".roles r ON r.id = rk.role_id
    WHERE rk.source_field = 'primary_responsibility'
    ORDER BY r.display_name, rk.position
""")
kras = cur.fetchall()
conn.close()

def parse_vec(s):
    return [float(x) for x in s.strip("[]").split(",")]

kra_data = [(role, pos, text, parse_vec(emb)) for role, pos, text, emb in kras]

# For each sentence, find its best KRA match across ALL roles
print("\n=== Per-sentence best KRA match ===")
role_best: dict[str, float] = defaultdict(float)
for i, sent_vec in enumerate(vecs):
    if not sent_vec:
        continue
    best_sim = 0.0
    best_role = ""
    best_kra = ""
    for role, pos, kra_text, kra_vec in kra_data:
        sim = _cosine(sent_vec, kra_vec)
        if sim > best_sim:
            best_sim = sim
            best_role = role
            best_kra = kra_text
        if sim > role_best[role]:
            role_best[role] = sim
    print(f"  [{i}] {sentences[i][:70]}")
    print(f"       -> best: {best_role} (sim={best_sim:.3f}) | KRA: {best_kra[:60]}")
    print()

# Final role ranking
print("=== Final role scores (MAX over all sentences × KRAs) ===")
sorted_roles = sorted(role_best.items(), key=lambda x: x[1], reverse=True)[:8]
for role, score in sorted_roles:
    print(f"  {role:<35} {score:.4f}")
