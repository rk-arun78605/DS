import psycopg2

conn = psycopg2.connect(host='localhost', port=3307, user='postgres', password='hello', dbname='WH')
with conn.cursor() as cur:
    cur.execute(
        """
        SELECT shop_code, item_code, item_name, vehicle_no, offloading_date,
               cart_qty, offloaded_qty, received_qty, price, diff
        FROM offloading_loading_staging
        WHERE trim(coalesce(offloading_date, '')) !~ '^[0-9]{2}-[A-Za-z]{3}-[0-9]{2}$'
        LIMIT 25
        """
    )
    rows = cur.fetchall()

print(f"Bad rows found: {len(rows)}")
for row in rows:
    print(row)

conn.close()
