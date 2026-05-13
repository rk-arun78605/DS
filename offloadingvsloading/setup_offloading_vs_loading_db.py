import os
import psycopg2

CSV_PATH = r"D:\Arun\Adhoc\0.daily\Mr Ramesh\wms\WMS_shopdiff_combined.csv"

DB_CONFIG = {
    "host": "localhost",
    "port": 3307,
    "user": "postgres",
    "password": "hello",
    "dbname": "WH",
}


def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS offloading_loading_staging;")
            cur.execute(
                """
                CREATE TABLE offloading_loading_staging (
                    shop_code text,
                    shop_name text,
                    item_code text,
                    item_name text,
                    vehicle_no text,
                    offloading_date text,
                    cart_qty numeric,
                    offloaded_qty numeric,
                    received_qty numeric,
                    price numeric,
                    diff numeric
                );
                """
            )

            with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
                cur.copy_expert(
                    """
                    COPY offloading_loading_staging
                    (shop_code, shop_name, item_code, item_name, vehicle_no, offloading_date,
                     cart_qty, offloaded_qty, received_qty, price, diff)
                    FROM STDIN WITH (FORMAT CSV, HEADER TRUE)
                    """,
                    f,
                )

            cur.execute("DROP TABLE IF EXISTS offloading_vs_loading;")
            cur.execute(
                """
                CREATE TABLE offloading_vs_loading AS
                WITH normalized AS (
                    SELECT
                        CASE
                            WHEN trim(COALESCE(offloading_date, '')) ~ '^[0-9]{2}-[A-Za-z]{3}-[0-9]{2}$'
                            THEN to_date(trim(offloading_date), 'DD-MON-YY')::date
                            ELSE NULL
                        END AS parsed_date,
                        upper(trim(shop_code)) AS shop_code,
                        trim(vehicle_no) AS vehicle_no,
                        trim(item_code) AS item_code,
                        trim(item_name) AS item_name,
                        COALESCE(received_qty, 0)::numeric AS qty_loaded,
                        COALESCE(offloaded_qty, 0)::numeric AS qty_offloaded,
                        COALESCE(price, 0)::numeric AS unit_price
                    FROM offloading_loading_staging
                )
                SELECT
                    parsed_date AS date,
                    shop_code,
                    vehicle_no,
                    item_code,
                    item_name,
                    qty_loaded,
                    (qty_loaded * unit_price)::numeric AS value_loaded,
                    qty_offloaded,
                    (qty_offloaded * unit_price)::numeric AS value_offloaded,
                    (qty_offloaded - qty_loaded)::numeric AS diff_qty,
                    ((qty_offloaded - qty_loaded) * unit_price)::numeric AS diff_val
                FROM normalized
                WHERE parsed_date IS NOT NULL;
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_offloading_vs_loading_date ON offloading_vs_loading(date);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_offloading_vs_loading_shop ON offloading_vs_loading(shop_code);")

        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM offloading_vs_loading;")
            count_rows, min_date, max_date = cur.fetchone()
            print(f"Loaded rows: {count_rows}")
            print(f"Date range: {min_date} to {max_date}")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
