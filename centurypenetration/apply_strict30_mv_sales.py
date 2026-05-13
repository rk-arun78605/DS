from pathlib import Path
import psycopg2

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'century_penetration',
}

MV_SALES_SQL = """
CREATE MATERIALIZED VIEW mv_sales_metrics AS
WITH date_ranges AS (
    SELECT
        CURRENT_DATE - INTERVAL '1 day' as yesterday,
        CURRENT_DATE - INTERVAL '30 days' as date_30d_start,
        CURRENT_DATE - INTERVAL '61 days' as date_60d_start,
        CURRENT_DATE - INTERVAL '91 days' as date_90d_start,
        CURRENT_DATE - INTERVAL '366 days' as date_365d_start
),
deduplicated_sales AS (
    SELECT DISTINCT ON (s.shop_code, s.item_code, s.date_invoice)
        s.shop_code,
        s.item_code,
        s.item_name,
        s.dept,
        s.date_invoice,
        s.qty,
        s.net_sales
    FROM sales s
    ORDER BY s.shop_code, s.item_code, s.date_invoice, s.loaded_at DESC
)
SELECT
    s.shop_code,
    s.item_code,
    MAX(s.item_name) as item_name,
    MAX(s.dept) as dept,
    SUM(CASE WHEN s.date_invoice >= dr.date_30d_start AND s.date_invoice <= dr.yesterday THEN s.qty ELSE 0 END) as sales_30d,
    SUM(CASE WHEN s.date_invoice >= dr.date_30d_start AND s.date_invoice <= dr.yesterday THEN s.net_sales ELSE 0 END) as value_30d,
    SUM(CASE WHEN s.date_invoice >= dr.date_60d_start AND s.date_invoice <= dr.yesterday THEN s.qty ELSE 0 END) as sales_60d,
    SUM(CASE WHEN s.date_invoice >= dr.date_60d_start AND s.date_invoice <= dr.yesterday THEN s.net_sales ELSE 0 END) as value_60d,
    SUM(CASE WHEN s.date_invoice >= dr.date_90d_start AND s.date_invoice <= dr.yesterday THEN s.qty ELSE 0 END) as sales_90d,
    SUM(CASE WHEN s.date_invoice >= dr.date_90d_start AND s.date_invoice <= dr.yesterday THEN s.net_sales ELSE 0 END) as value_90d,
    SUM(CASE WHEN s.date_invoice >= dr.date_365d_start AND s.date_invoice <= dr.yesterday THEN s.qty ELSE 0 END) as sales_365d,
    SUM(CASE WHEN s.date_invoice >= dr.date_365d_start AND s.date_invoice <= dr.yesterday THEN s.net_sales ELSE 0 END) as value_365d,
    ROUND(SUM(CASE WHEN s.date_invoice >= dr.date_90d_start AND s.date_invoice <= dr.yesterday THEN s.qty ELSE 0 END)::NUMERIC / 90, 2) as ros,
    MAX(s.date_invoice) as last_sale_date,
    CURRENT_TIMESTAMP as refreshed_at
FROM deduplicated_sales s
CROSS JOIN date_ranges dr
GROUP BY s.shop_code, s.item_code
"""

INDEX_SQL = [
    "CREATE UNIQUE INDEX idx_mv_sales_item_shop ON mv_sales_metrics(item_code, shop_code);",
    "CREATE INDEX idx_mv_sales_shop ON mv_sales_metrics(shop_code);",
    "CREATE INDEX idx_mv_sales_item ON mv_sales_metrics(item_code);",
    "CREATE INDEX idx_mv_sales_ros ON mv_sales_metrics(ros) WHERE ros > 0;",
]

recreate_century_sql = Path(__file__).with_name('recreate_century_view.sql').read_text(encoding='utf-8')

with psycopg2.connect(**DB_CONFIG) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP MATERIALIZED VIEW IF EXISTS mv_century_penetration CASCADE;")
        cur.execute("DROP MATERIALIZED VIEW IF EXISTS mv_sales_metrics CASCADE;")
        cur.execute(MV_SALES_SQL)
        for stmt in INDEX_SQL:
            cur.execute(stmt)
        cur.execute(recreate_century_sql)

        cur.execute("SELECT sales_30d FROM mv_century_penetration WHERE item_code=%s AND shop_code=%s", ('CL623','BOL'))
        mv_val = cur.fetchone()[0]
        cur.execute(
            """
            WITH d AS (
                SELECT DISTINCT ON (shop_code, item_code, date_invoice)
                    shop_code, item_code, date_invoice, qty
                FROM sales
                WHERE item_code=%s AND shop_code=%s
                  AND date_invoice BETWEEN CURRENT_DATE - INTERVAL '30 days' AND CURRENT_DATE - INTERVAL '1 day'
                ORDER BY shop_code, item_code, date_invoice, loaded_at DESC
            )
            SELECT COALESCE(SUM(qty),0) FROM d
            """,
            ('CL623','BOL')
        )
        strict30_val = cur.fetchone()[0]

print('Applied strict-30-day MV logic successfully.')
print('mv_century_penetration CL623/BOL sales_30d =', mv_val)
print('dedup strict-30 source sum =', strict30_val)
