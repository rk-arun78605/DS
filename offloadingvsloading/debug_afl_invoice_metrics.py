import psycopg2

DB = dict(host='localhost', port=3307, user='postgres', password='hello', dbname='WH')
start_date = '2026-03-09'
end_date = '2026-03-09'
shop = 'AFL'

consumable_tills = [
    ('SPN', 34), ('SPN', 32), ('SPN', 33), ('LFS', 23), ('LFS', 26), ('MSS', 13),
    ('MSS', 14), ('MM1', 1), ('MM1', 13), ('MM2', 10), ('MM2', 9), ('WHL', 14)
]

values_sql = ','.join([f"('{s}', {t})" for s, t in consumable_tills])

query = f"""
WITH consumable_tills AS (
  SELECT * FROM (VALUES {values_sql}) AS t(shop_code, till_no)
),
erp_base AS (
  SELECT UPPER(TRIM(store_code)) AS shop_code,
         invno,
         COALESCE(amt, 0)::numeric AS amt,
         CASE WHEN NULLIF(REGEXP_REPLACE(COALESCE(tillno::text,''), '[^0-9]','','g'),'') IS NULL THEN NULL
              ELSE NULLIF(REGEXP_REPLACE(COALESCE(tillno::text,''), '[^0-9]','','g'),'')::int END AS till_no
  FROM erpdata
  WHERE invdate::date BETWEEN %s AND %s
    AND NULLIF(TRIM(COALESCE(invno::text,'')), '') IS NOT NULL
),
invoice_base AS (
  SELECT UPPER(TRIM(store_code)) AS shop_code, invno, COALESCE(amt,0)::numeric AS amt,
         CASE WHEN NULLIF(REGEXP_REPLACE(COALESCE(tillno::text,''), '[^0-9]','','g'),'') IS NULL THEN NULL
              ELSE NULLIF(REGEXP_REPLACE(COALESCE(tillno::text,''), '[^0-9]','','g'),'')::int END AS till_no,
         1 AS src_priority
  FROM invoices
  WHERE invdate::date BETWEEN %s AND %s
    AND NULLIF(TRIM(COALESCE(invno::text,'')), '') IS NOT NULL
  UNION ALL
  SELECT UPPER(TRIM(store_code)) AS shop_code, invno, COALESCE(amt,0)::numeric AS amt,
         CASE WHEN NULLIF(REGEXP_REPLACE(COALESCE(tillno::text,''), '[^0-9]','','g'),'') IS NULL THEN NULL
              ELSE NULLIF(REGEXP_REPLACE(COALESCE(tillno::text,''), '[^0-9]','','g'),'')::int END AS till_no,
         2 AS src_priority
  FROM invoices_manager
  WHERE invdate::date BETWEEN %s AND %s
    AND NULLIF(TRIM(COALESCE(invno::text,'')), '') IS NOT NULL
),
invoice_dedup AS (
  SELECT shop_code, invno, amt, till_no
  FROM (
    SELECT shop_code, invno, amt, till_no,
           ROW_NUMBER() OVER (PARTITION BY shop_code, invno ORDER BY src_priority) AS rn
    FROM invoice_base
  ) d
  WHERE rn = 1
),
invoice_classified AS (
  SELECT i.shop_code, i.invno, i.amt,
         CASE WHEN c.shop_code IS NOT NULL THEN 1 ELSE 0 END AS is_consumable
  FROM invoice_dedup i
  LEFT JOIN consumable_tills c ON c.shop_code = i.shop_code AND c.till_no = i.till_no
)
SELECT
  (SELECT COUNT(*) FROM erp_base WHERE shop_code=%s) AS erp_nob,
  (SELECT COALESCE(SUM(amt),0) FROM erp_base WHERE shop_code=%s) AS erp_ghs,
  (SELECT COUNT(*) FROM invoice_classified WHERE shop_code=%s AND is_consumable=0) AS scan_nob,
  (SELECT COALESCE(SUM(amt),0) FROM invoice_classified WHERE shop_code=%s AND is_consumable=0) AS scan_ghs,
  (SELECT COUNT(*) FROM invoice_classified WHERE shop_code=%s AND is_consumable=1) AS cons_nob,
  (SELECT COALESCE(SUM(amt),0) FROM invoice_classified WHERE shop_code=%s AND is_consumable=1) AS cons_ghs;
"""

with psycopg2.connect(**DB) as conn:
    with conn.cursor() as cur:
        cur.execute(query, (start_date, end_date, start_date, end_date, start_date, end_date, shop, shop, shop, shop, shop, shop))
        row = cur.fetchone()

erp_nob, erp_ghs, scan_nob, scan_ghs, cons_nob, cons_ghs = row
acct_nob = (scan_nob or 0) + (cons_nob or 0)
bill_pct = (acct_nob / erp_nob * 100) if erp_nob else 0
covered_ghs = (scan_ghs or 0) + (cons_ghs or 0)
diff_ghs = (erp_ghs or 0) - covered_ghs
diff_pct = (diff_ghs / erp_ghs * 100) if erp_ghs else 0

print({
    'shop': shop,
    'date': start_date,
    'erp_nob': erp_nob,
    'scan_nob': scan_nob,
    'cons_nob': cons_nob,
    'bill_pct': round(bill_pct, 2),
    'erp_ghs': float(erp_ghs or 0),
    'scan_ghs': float(scan_ghs or 0),
    'cons_ghs': float(cons_ghs or 0),
    'diff_ghs': float(diff_ghs or 0),
    'diff_pct': round(diff_pct, 2)
})
