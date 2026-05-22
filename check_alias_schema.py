import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
    dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'), sslmode=os.getenv('DB_SSLMODE','disable'))
cur = conn.cursor()
cur.execute("SELECT DISTINCT alias_type FROM \"skill-library\".role_aliases")
print("alias_type values:", [r[0] for r in cur.fetchall()])
cur.execute("""SELECT column_name, column_default, is_nullable
    FROM information_schema.columns
    WHERE table_schema='skill-library' AND table_name='role_aliases'
    ORDER BY ordinal_position""")
for r in cur.fetchall():
    print(r)
conn.close()
