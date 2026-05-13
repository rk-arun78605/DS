import psycopg2
from datetime import date, timedelta

conn = psycopg2.connect(host='localhost', port=3307, user='postgres', password='hello', database='century_penetration')
cur = conn.cursor()

today = date.today()
yesterday = today - timedelta(days=1)

print('today:', today)
print('yesterday:', yesterday)

# View logic range
start_view = today - timedelta(days=31)
end_view = yesterday
print('view_range:', start_view, 'to', end_view, '(inclusive)')

# Strict 30-day inclusive range (30 rows of dates)
start_30 = yesterday - timedelta(days=29)
end_30 = yesterday
print('strict_30day_range:', start_30, 'to', end_30, '(inclusive)')

# 31-day inclusive range from yesterday-30 to yesterday
start_31 = yesterday - timedelta(days=30)
end_31 = yesterday
print('31day_range_from_yesterday_minus30:', start_31, 'to', end_31, '(inclusive)')

# Raw sums
for label, start_d, end_d in [
    ('view_range_raw', start_view, end_view),
    ('strict_30_raw', start_30, end_30),
    ('31day_raw', start_31, end_31),
]:
    cur.execute(
        """
        SELECT COALESCE(SUM(qty),0)
        FROM sales
        WHERE item_code=%s AND shop_code=%s
          AND date_invoice BETWEEN %s AND %s
        """,
        ('CL623', 'BOL', start_d, end_d)
    )
    print(label, cur.fetchone()[0])

# Dedup sums (same pattern as mv_sales_metrics: latest loaded_at per day)
for label, start_d, end_d in [
    ('view_range_dedup', start_view, end_view),
    ('strict_30_dedup', start_30, end_30),
    ('31day_dedup', start_31, end_31),
]:
    cur.execute(
        """
        WITH d AS (
          SELECT DISTINCT ON (shop_code, item_code, date_invoice)
                 shop_code, item_code, date_invoice, qty
          FROM sales
          WHERE item_code=%s AND shop_code=%s
            AND date_invoice BETWEEN %s AND %s
          ORDER BY shop_code, item_code, date_invoice, loaded_at DESC
        )
        SELECT COALESCE(SUM(qty),0) FROM d
        """,
        ('CL623', 'BOL', start_d, end_d)
    )
    print(label, cur.fetchone()[0])

# Current values from MVs
cur.execute("SELECT sales_30d FROM mv_sales_metrics WHERE item_code=%s AND shop_code=%s", ('CL623','BOL'))
print('mv_sales_metrics.sales_30d', cur.fetchone()[0])
cur.execute("SELECT sales_30d FROM mv_century_penetration WHERE item_code=%s AND shop_code=%s", ('CL623','BOL'))
print('mv_century_penetration.sales_30d', cur.fetchone()[0])

cur.close()
conn.close()
