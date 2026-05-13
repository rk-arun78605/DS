import psycopg2

conn = psycopg2.connect(host='localhost', port=3307, user='postgres', password='hello', dbname='WH')
with conn.cursor() as cur:
    cur.execute("SELECT COUNT(*), COALESCE(SUM(amt),0) FROM invoices WHERE UPPER(TRIM(store_code))='AFL' AND entry_time::date='2026-03-09'")
    inv = cur.fetchone()
    cur.execute("SELECT COUNT(*), COALESCE(SUM(amt),0) FROM invoices_manager WHERE UPPER(TRIM(store_code))='AFL' AND entry_time::date='2026-03-09'")
    mgr = cur.fetchone()
    cur.execute("SELECT COUNT(*), COALESCE(SUM(amt),0) FROM erpdata WHERE UPPER(TRIM(store_code))='AFL' AND invdate::date='2026-03-09'")
    erp = cur.fetchone()

print({
    'erp_invdate_nob': erp[0],
    'erp_invdate_ghs': float(erp[1] or 0),
    'inv_entry_nob': inv[0],
    'inv_entry_ghs': float(inv[1] or 0),
    'mgr_entry_nob': mgr[0],
    'mgr_entry_ghs': float(mgr[1] or 0),
})

conn.close()
