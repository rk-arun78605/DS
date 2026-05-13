"""
Recreate mv_century_penetration with updated optimum stock formula:
  Optimum Stock = ROS × (30 + lead_days)   [was ROS × 30]
  Min Threshold = Optimum × min_threshold_pct
  Stock Variance = total_stock - Optimum
"""
import psycopg2

conn = psycopg2.connect(host='localhost', port=3307, user='postgres', password='hello', dbname='century_penetration')
conn.autocommit = True
cur = conn.cursor()

# --- check existing column names ---
cur.execute("""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'mv_century_penetration'
            AND column_name IN ('req_21_days','req_30_days','optimum_stock_30d','lead','lead_days','min_threshold_qty','stock_variance')
    ORDER BY ordinal_position
""")
print("Existing relevant columns:", [r[0] for r in cur.fetchall()])

print("Dropping and recreating mv_century_penetration ...")
cur.execute("DROP MATERIALIZED VIEW IF EXISTS mv_century_penetration CASCADE")

# Shop policy inline values (same set as create_century_tables.sql)
create_sql = """
CREATE MATERIALIZED VIEW mv_century_penetration AS
WITH shop_policy AS (
    SELECT shop_code, loading_status, lead_days, min_threshold_pct
    FROM (VALUES
        ('AST','ALTERNATE DAYS',14,0.5),
        ('AWS','ALTERNATE DAYS',14,0.5),
        ('AYA','ALTERNATE DAYS',14,0.5),
        ('BAS','ALTERNATE DAYS',14,0.5),
        ('BKS','ALTERNATE DAYS',14,0.5),
        ('BOL','ALTERNATE DAYS',14,0.5),
        ('CBZ','ALTERNATE DAYS',14,0.5),
        ('DMK','ALTERNATE DAYS',14,0.5),
        ('DWS','ALTERNATE DAYS',14,0.5),
        ('EAS','ALTERNATE DAYS',14,0.5),
        ('FAR','ALTERNATE DAYS',14,0.5),
        ('HAR','ALTERNATE DAYS',14,0.5),
        ('HO2','ALTERNATE DAYS',14,0.5),
        ('HOH','ALTERNATE DAYS',14,0.5),
        ('KAS','ALTERNATE DAYS',14,0.5),
        ('KK1','ALTERNATE DAYS',14,0.5),
        ('KM2','ALTERNATE DAYS',14,0.5),
        ('KMQ','ALTERNATE DAYS',14,0.5),
        ('KS7','ALTERNATE DAYS',14,0.5),
        ('KSI','ALTERNATE DAYS',14,0.5),
        ('KUM','ALTERNATE DAYS',14,0.5),
        ('KWA','ALTERNATE DAYS',14,0.5),
        ('LFS','DAILY',7,0.3),
        ('M03','DAILY',7,0.3),
        ('MBB','ALTERNATE DAYS',14,0.5),
        ('MDN','ALTERNATE DAYS',14,0.5),
        ('MKL','ALTERNATE DAYS',14,0.5),
        ('MM1','DAILY',7,0.3),
        ('MM2','DAILY',7,0.3),
        ('MM3','ALTERNATE DAYS',14,0.5),
        ('MSS','DAILY',7,0.3),
        ('NAN','ALTERNATE DAYS',14,0.5),
        ('NKW','ALTERNATE DAYS',14,0.5),
        ('SD2','ALTERNATE DAYS',14,0.5),
        ('SPN','DAILY',7,0.3),
        ('SPX','DAILY',7,0.3),
        ('SU2','ALTERNATE DAYS',14,0.5),
        ('TKD','ALTERNATE DAYS',14,0.5),
        ('TKW','ALTERNATE DAYS',14,0.5),
        ('TM2','ALTERNATE DAYS',14,0.5),
        ('TML','DAILY',7,0.3),
        ('TMP','DAILY',7,0.3),
        ('WHL','ALTERNATE DAYS',14,0.5),
        ('WNC','ALTERNATE DAYS',14,0.5)
    ) AS t(shop_code, loading_status, lead_days, min_threshold_pct)
)
SELECT
    r.item_code,
    r.item_name,
    r.shop_code,
    r.dept,
    r.brand,

    -- Stock levels
    r.shop_stock                                          AS sih,
    COALESCE(sit.total_sit, 0)                            AS sit,
    r.shop_stock + COALESCE(sit.total_sit, 0)             AS total_stock,

    -- Sales metrics
    COALESCE(sm.sales_30d,  0) AS sales_30d,
    COALESCE(sm.sales_60d,  0) AS sales_60d,
    COALESCE(sm.sales_90d,  0) AS sales_90d,
    COALESCE(sm.sales_365d, 0) AS sales_365d,

    COALESCE(sm.value_30d,  0) AS value_30d,
    COALESCE(sm.value_60d,  0) AS value_60d,
    COALESCE(sm.value_90d,  0) AS value_90d,
    COALESCE(sm.value_365d, 0) AS value_365d,

    -- Rate of Sales (ROS = Sales 90d / 90)
    COALESCE(sm.ros, 0) AS ros,

    -- Optimum Stock = ROS × (30 maintenance days + lead days per shop policy)
    -- DAILY shops (lead=7):  ROS × 37
    -- ALTERNATE DAYS (lead=14): ROS × 44
    ROUND(COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14)), 2) AS req_21_days,
    ROUND(COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14)), 2) AS req_30_days,
    ROUND(COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14)), 2) AS optimum_stock_30d,

    COALESCE(sp.loading_status, 'ALTERNATE DAYS')         AS loading_status,
    COALESCE(sp.lead_days, 14)                            AS lead,
    COALESCE(sp.lead_days, 14)                            AS lead_days,

    -- Min Threshold qty = Optimum × threshold% (30% for DAILY, 50% for ALTERNATE DAYS)
    ROUND((COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14))) * COALESCE(sp.min_threshold_pct, 0.5), 2) AS min_threshold_qty,
    ROUND(COALESCE(sp.min_threshold_pct, 0.5) * 100, 0)  AS min_threshold_pct,

    -- Stock variance vs optimum (positive = surplus, negative = deficit)
    ROUND((r.shop_stock + COALESCE(sit.total_sit, 0)) - (COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14))), 2) AS stock_variance,

    COALESCE(r.pack_size, 0) AS pack_size,

    -- Stock status
    CASE
        WHEN (r.shop_stock + COALESCE(sit.total_sit, 0))
             < ((COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14))) * COALESCE(sp.min_threshold_pct, 0.5))
             THEN 'UnderStock'
        WHEN (r.shop_stock + COALESCE(sit.total_sit, 0))
             > (COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14)))
             THEN 'OverStock'
        ELSE 'Balanced'
    END AS stock_status,

    -- Days of stock remaining
    CASE
        WHEN COALESCE(sm.ros, 0) > 0
        THEN ROUND((r.shop_stock + COALESCE(sit.total_sit, 0)) / COALESCE(sm.ros, 0), 1)
        ELSE NULL
    END AS days_of_stock,

    -- Reorder parameters
    r.min_nu,
    r.max_nu,
    r.reorder_qty,
    r.selling_price,

    -- Last sale info
    sm.last_sale_date,
    sit.latest_transit_date,

    -- Metadata
    CURRENT_TIMESTAMP AS refreshed_at,
    COALESCE(r.loaded_at, CURRENT_TIMESTAMP) AS created_at

FROM reorder_level r
LEFT JOIN shop_policy sp ON UPPER(TRIM(r.shop_code)) = sp.shop_code
LEFT JOIN mv_sales_metrics sm ON r.item_code = sm.item_code AND r.shop_code = sm.shop_code
LEFT JOIN mv_sit_summary sit ON r.item_code = sit.item_code AND r.shop_code = sit.shop_code
WHERE UPPER(r.brand) = 'CENTURY'
"""

cur.execute(create_sql)
print("Materialized view created.")

# Recreate indexes
indexes = [
    "CREATE UNIQUE INDEX idx_century_item_shop ON mv_century_penetration(item_code, shop_code)",
    "CREATE INDEX idx_century_shop ON mv_century_penetration(shop_code)",
    "CREATE INDEX idx_century_dept ON mv_century_penetration(dept)",
    "CREATE INDEX idx_century_status ON mv_century_penetration(stock_status)",
    "CREATE INDEX idx_century_ros ON mv_century_penetration(ros) WHERE ros > 0",
    "CREATE INDEX idx_century_understock ON mv_century_penetration(stock_variance) WHERE stock_status = 'UnderStock'",
    "CREATE INDEX idx_century_overstock ON mv_century_penetration(stock_variance) WHERE stock_status = 'OverStock'",
]
for idx_sql in indexes:
    cur.execute(idx_sql)
print("Indexes created.")

# The Python app uses req_30_days; req_21_days exists in the view — add a view alias just in case
# req_30_days is now a direct column in the MV (same value as req_21_days)
cur.execute("SELECT req_30_days FROM mv_century_penetration LIMIT 1")
print("req_30_days sample:", cur.fetchone())

# Sample check - use req_30_days (= ros * (30 + lead_days))
cur.execute("""
    SELECT shop_code, item_code, ros, lead_days, req_30_days AS optimum, min_threshold_qty, stock_status
    FROM mv_century_penetration
    WHERE ros > 0
    ORDER BY shop_code, ros DESC
    LIMIT 10
""")
print("\nSample rows (optimum = ros × (30+lead)):")
for row in cur.fetchall():
    shop, item, ros, lead, optimum, minthresh, status = row
    expected = round(ros * (30 + lead), 2)
    print(f"  {shop} | {item} | ROS={ros:.2f} | lead={lead} | optimum={optimum} (expect {expected}) | thresh={minthresh} | {status}")

cur.close()
conn.close()
print("\nDone.")
