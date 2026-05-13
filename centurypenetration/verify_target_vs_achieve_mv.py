import psycopg2

conn = psycopg2.connect(
    host='localhost',
    port=3307,
    user='postgres',
    password='hello',
    dbname='century_penetration',
)
try:
    with conn.cursor() as cur:
        cur.execute('SELECT COUNT(*) FROM mv_target_vs_achieve_century')
        print('mv rows:', cur.fetchone()[0])

        cur.execute('SELECT COUNT(*) FROM vw_target_vs_achieve_century')
        print('vw rows:', cur.fetchone()[0])

        cur.execute("""
            SELECT indexname
            FROM pg_indexes
            WHERE tablename = 'mv_target_vs_achieve_century'
            ORDER BY indexname
        """)
        print('indexes:', [r[0] for r in cur.fetchall()])
finally:
    conn.close()
