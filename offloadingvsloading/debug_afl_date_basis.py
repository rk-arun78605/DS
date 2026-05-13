import psycopg2

conn = psycopg2.connect(host='localhost', port=3307, user='postgres', password='hello', dbname='WH')
with conn.cursor() as cur:
    cur.execute("SELECT COUNT(*) FROM erpdata WHERE UPPER(TRIM(store_code))='AFL' AND invdate::date='2026-03-09'")
    erp_inv = cur.fetchone()[0]

    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='invoices'")
    inv_cols = {r[0] for r in cur.fetchall()}
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='invoices_manager'")
    mgr_cols = {r[0] for r in cur.fetchall()}

    has_inv_entry = 'a_entrytime' in inv_cols
    has_mgr_entry = 'a_entrytime' in mgr_cols

    cur.execute("SELECT COUNT(*) FROM invoices WHERE UPPER(TRIM(store_code))='AFL' AND invdate::date='2026-03-09'")
    inv_inv = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM invoices_manager WHERE UPPER(TRIM(store_code))='AFL' AND invdate::date='2026-03-09'")
    mgr_inv = cur.fetchone()[0]

    if has_inv_entry:
        cur.execute("SELECT COUNT(*) FROM invoices WHERE UPPER(TRIM(store_code))='AFL' AND a_entrytime::date='2026-03-09'")
        inv_entry = cur.fetchone()[0]
    else:
        inv_entry = None

    if has_mgr_entry:
        cur.execute("SELECT COUNT(*) FROM invoices_manager WHERE UPPER(TRIM(store_code))='AFL' AND a_entrytime::date='2026-03-09'")
        mgr_entry = cur.fetchone()[0]
    else:
        mgr_entry = None

print({
    'erp_invdate': erp_inv,
    'inv_invdate': inv_inv,
    'mgr_invdate': mgr_inv,
    'has_inv_a_entrytime': has_inv_entry,
    'has_mgr_a_entrytime': has_mgr_entry,
    'inv_a_entrytime': inv_entry,
    'mgr_a_entrytime': mgr_entry,
})

conn.close()
