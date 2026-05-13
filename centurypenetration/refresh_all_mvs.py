import time
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    "host": "localhost",
    "port": 3307,
    "user": "postgres",
    "password": "hello",
    "dbname": "century_penetration",
}

VIEWS_IN_ORDER = [
    "mv_sales_metrics",
    "mv_sit_summary",
    "mv_century_penetration",
    "mv_target_vs_achieve_century",
]


def refresh_all_materialized_views() -> None:
    print("=" * 72)
    print("CENTURY MATERIALIZED VIEW REFRESH")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 72)

    with psycopg2.connect(**DB_CONFIG) as conn:
        conn.autocommit = True
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for view_name in VIEWS_IN_ORDER:
                start = time.perf_counter()
                print(f"\nRefreshing {view_name} ...")
                cur.execute(f"REFRESH MATERIALIZED VIEW {view_name};")
                elapsed = time.perf_counter() - start

                cur.execute(f"SELECT COUNT(*) AS row_count FROM {view_name};")
                row_count = cur.fetchone()["row_count"]
                print(f"✓ {view_name} refreshed in {elapsed:.2f}s | rows: {row_count:,}")

            cur.execute(
                """
                SELECT
                    matviewname,
                    pg_size_pretty(pg_total_relation_size(schemaname || '.' || matviewname)) AS size
                FROM pg_matviews
                WHERE matviewname = ANY(%s)
                ORDER BY matviewname;
                """,
                (VIEWS_IN_ORDER,),
            )
            print("\nMaterialized view sizes:")
            for row in cur.fetchall():
                print(f"- {row['matviewname']}: {row['size']}")

    print("\n" + "=" * 72)
    print(f"Completed: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 72)


if __name__ == "__main__":
    refresh_all_materialized_views()
