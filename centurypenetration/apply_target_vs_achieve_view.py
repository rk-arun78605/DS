import psycopg2
from pathlib import Path

sql_path = Path(__file__).with_name('create_target_vs_achieve_view.sql')
sql_text = sql_path.read_text(encoding='utf-8')

conn = psycopg2.connect(
    host='localhost',
    port=3307,
    user='postgres',
    password='hello',
    dbname='century_penetration',
)
try:
    with conn:
        with conn.cursor() as cur:
            cur.execute(sql_text)
            cur.execute('SELECT COUNT(*) FROM vw_target_vs_achieve_century')
            count = cur.fetchone()[0]
            cur.execute('SELECT * FROM vw_target_vs_achieve_century ORDER BY item_code LIMIT 5')
            sample = cur.fetchall()
    print(f'View created successfully. Rows: {count}')
    print('Sample rows:')
    for row in sample:
        print(row)
finally:
    conn.close()
