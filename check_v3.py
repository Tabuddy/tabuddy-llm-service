import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
conn = psycopg2.connect(os.environ['SKILL_LIBRARY_PG_DSN'])
schema = os.environ.get('SKILL_LIBRARY_SCHEMA', 'skill-library')

with conn:
    with conn.cursor() as cur:
        cur.execute(f'SET search_path TO "{schema}", public')
        
        # Count v3 enrichment candidates by reason
        cur.execute('''
            WITH kra_counts AS (
                SELECT role_id, COUNT(*) AS n FROM role_kras GROUP BY role_id
            )
            SELECT
                CASE
                    WHEN source = 'AUTOMATED_DISCOVERY' AND COALESCE(k.n,0)=0 THEN 'synth_no_kras'
                    WHEN source = 'AUTOMATED_DISCOVERY' THEN 'synth_with_kras'
                    ELSE 'shell_no_kras'
                END AS needs_v3_reason,
                COUNT(*) AS count
            FROM roles r
            LEFT JOIN kra_counts k ON k.role_id = r.id
            WHERE COALESCE(k.n,0) = 0 OR r.source = 'AUTOMATED_DISCOVERY'
            GROUP BY 1
            ORDER BY 2 DESC
        ''')
        print('v3 Enrichment Candidates by Reason:')
        for reason, count in cur.fetchall():
            print(f'  {reason}: {count}')
