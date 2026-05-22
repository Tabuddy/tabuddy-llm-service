import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
    dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'), sslmode=os.getenv('DB_SSLMODE','disable'))
cur = conn.cursor()
cur.execute("""
    SELECT r.display_name, array_agg(ra.alias_text ORDER BY ra.alias_text) as aliases
    FROM "skill-library".roles r
    LEFT JOIN "skill-library".role_aliases ra ON ra.role_id = r.id
    GROUP BY r.id, r.display_name
    ORDER BY r.display_name
""")
for row in cur.fetchall():
    print(f"{row[0]}:")
    for a in (row[1] or []):
        print(f"  - {a}")
conn.close()
