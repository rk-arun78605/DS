import argparse
import csv
import os
from datetime import datetime

import pymysql


DEFAULT_DB = {
    "host": "192.168.0.17",
    "port": 3306,
    "user": "misaccount",
    "password": "Inv@Central@2024",
    "dbname": "invcentral",
}


def _find_databases_with_alerts(conn: pymysql.connections.Connection) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT table_schema
            FROM information_schema.tables
            WHERE table_name = 'alerts'
            ORDER BY table_schema
            """
        )
        return [row[0] for row in cur.fetchall()]


def _resolve_database_name(db_config: dict) -> str:
    if db_config.get("dbname"):
        return db_config["dbname"]

    probe_conn = None
    try:
        probe_conn = pymysql.connect(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            charset="utf8mb4",
            connect_timeout=20,
            read_timeout=120,
            write_timeout=120,
            cursorclass=pymysql.cursors.Cursor,
        )
        matches = _find_databases_with_alerts(probe_conn)
    finally:
        if probe_conn is not None:
            probe_conn.close()

    if not matches:
        raise RuntimeError("No database containing table 'alerts' was found. Provide --dbname.")
    if len(matches) > 1:
        db_list = ", ".join(matches)
        raise RuntimeError(
            f"Multiple databases contain 'alerts': {db_list}. Please rerun with --dbname."
        )
    return matches[0]


def export_alerts_to_csv(output_path: str, db_config: dict) -> None:
    conn = None
    resolved_dbname = _resolve_database_name(db_config)
    try:
        conn = pymysql.connect(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            database=resolved_dbname,
            charset="utf8mb4",
            connect_timeout=20,
            read_timeout=120,
            write_timeout=120,
            cursorclass=pymysql.cursors.DictCursor,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM alerts")
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

            with open(output_path, "w", encoding="utf-8", newline="") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)

        print(f"✅ Export completed from MySQL DB '{resolved_dbname}': {output_path}")
        print(f"📊 Rows exported: {len(rows):,}")
    finally:
        if conn is not None:
            conn.close()


def main() -> None:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"alerts_full_export_{now}.csv"

    parser = argparse.ArgumentParser(
        description="Export complete data from alerts table to CSV"
    )
    parser.add_argument("--host", default=DEFAULT_DB["host"], help="MySQL host")
    parser.add_argument("--port", type=int, default=DEFAULT_DB["port"], help="MySQL port")
    parser.add_argument("--user", default=DEFAULT_DB["user"], help="MySQL user")
    parser.add_argument("--password", default=DEFAULT_DB["password"], help="MySQL password")
    parser.add_argument("--dbname", default=DEFAULT_DB["dbname"], help="MySQL database name (optional; auto-detect if omitted)")
    parser.add_argument(
        "--output",
        default=default_output,
        help="Output CSV path (default: alerts_full_export_YYYYMMDD_HHMMSS.csv)",
    )

    args = parser.parse_args()

    db_config = {
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "password": args.password,
        "dbname": args.dbname,
    }

    output_path = os.path.abspath(args.output)
    export_alerts_to_csv(output_path, db_config)


if __name__ == "__main__":
    main()
