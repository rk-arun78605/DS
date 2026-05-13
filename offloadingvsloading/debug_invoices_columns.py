import psycopg2

conn = psycopg2.connect(host='localhost', port=3307, user='postgres', password='hello', dbname='WH')
with conn.cursor() as cur:
    for table in ['invoices', 'invoices_manager', 'erpdata']:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name=%s ORDER BY ordinal_position", (table,))
        cols = [r[0] for r in cur.fetchall()]
        print(table, cols)

conn.close()
