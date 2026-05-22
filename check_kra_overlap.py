"""Check cross-role KRA semantic overlap after re-embedding."""
import os, math
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()
import psycopg2

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
    dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'), sslmode=os.getenv('DB_SSLMODE', 'disable'),
)
cur = conn.cursor()

cur.execute("""
    SELECT r.display_name, rk.position, rk.kra_text,
           rk.kra_embedding::text
    FROM "skill-library".role_kras rk
    JOIN "skill-library".roles r ON r.id = rk.role_id
    WHERE rk.source_field = 'primary_responsibility'
    ORDER BY r.display_name, rk.position
""")
rows = cur.fetchall()
conn.close()

def parse_vec(s):
    return [float(x) for x in s.strip().lstrip('[').rstrip(']').split(',')]

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb) if na and nb else 0.0

role_kras = defaultdict(list)
for role, pos, text, emb_str in rows:
    role_kras[role].append((pos, text, parse_vec(emb_str)))

roles = sorted(role_kras.keys())
THRESHOLD = 0.82

print(f"Roles: {len(roles)}, Total KRAs: {len(rows)}")
print(f"Checking cross-role KRA overlap (threshold >= {THRESHOLD})...\n")

flagged = []
for i, role_a in enumerate(roles):
    for role_b in roles[i+1:]:
        for pos_a, text_a, vec_a in role_kras[role_a]:
            for pos_b, text_b, vec_b in role_kras[role_b]:
                sim = cosine(vec_a, vec_b)
                if sim >= THRESHOLD:
                    flagged.append((sim, role_a, pos_a, text_a, role_b, pos_b, text_b))

flagged.sort(reverse=True)

if not flagged:
    print("No overlapping KRA pairs above threshold. All clear.")
else:
    print(f"FLAGGED OVERLAPPING PAIRS ({len(flagged)} found):\n")
    for sim, ra, pa, ta, rb, pb, tb in flagged:
        print(f"  sim={sim:.3f}")
        print(f"    [{ra}] p{pa}: {ta[:100]}")
        print(f"    [{rb}] p{pb}: {tb[:100]}")
        print()

# Also show max similarity per role-pair (summarised view)
print("\n--- Max cross-role similarity summary (top 20 role pairs) ---")
pair_max = {}
for i, role_a in enumerate(roles):
    for role_b in roles[i+1:]:
        max_sim = 0.0
        for _, _, vec_a in role_kras[role_a]:
            for _, _, vec_b in role_kras[role_b]:
                sim = cosine(vec_a, vec_b)
                if sim > max_sim:
                    max_sim = sim
        pair_max[(role_a, role_b)] = max_sim

sorted_pairs = sorted(pair_max.items(), key=lambda x: x[1], reverse=True)[:20]
for (ra, rb), max_sim in sorted_pairs:
    marker = " <-- OVERLAP" if max_sim >= THRESHOLD else ""
    print(f"  {max_sim:.3f}  {ra}  vs  {rb}{marker}")
