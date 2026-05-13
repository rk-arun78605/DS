import psycopg2
import pandas as pd
import uuid

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'dbname': 'barcode',
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def kill_stale_sessions() -> int:
    """
    Terminate all connections to this database that are idle or belong to
    previous Streamlit sessions. Runs against 'postgres' (admin db) so it
    works even when the target db is congested.
    Returns the number of sessions killed.
    """
    admin_cfg = {**DB_CONFIG, 'dbname': 'postgres'}
    try:
        conn = psycopg2.connect(**admin_cfg)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(pg_terminate_backend(pid))
            FROM pg_stat_activity
            WHERE datname = %s
              AND pid <> pg_backend_pid()
              AND state IN ('idle', 'idle in transaction', 'idle in transaction (aborted)')
            """,
            (DB_CONFIG['dbname'],),
        )
        killed = cur.fetchone()[0]
        cur.close()
        conn.close()
        return killed
    except Exception:
        return 0


def load_itemmaster() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(
            "SELECT item_code, item_name, COALESCE(barcode, '') AS barcode, status "
            "FROM itemmaster ORDER BY item_name",
            conn,
        )


def insert_uploaded_items(df: pd.DataFrame) -> str:
    session_id = str(uuid.uuid4())[:12]
    records = [
        (
            session_id,
            str(row.get('barcode', '') or '').strip(),
            str(row.get('barcode1', '') or '').strip(),
            str(row.get('item_description', '') or '').strip(),
        )
        for _, row in df.iterrows()
    ]
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE uploaded_items")   # keep only latest upload
        cur.executemany(
            "INSERT INTO uploaded_items (session_id, barcode, barcode1, item_description) "
            "VALUES (%s, %s, %s, %s)",
            records,
        )
        conn.commit()
    return session_id


def bulk_insert_itemmaster(df: pd.DataFrame) -> int:
    count = 0
    with get_connection() as conn:
        cur = conn.cursor()
        for _, row in df.iterrows():
            cur.execute(
                """
                INSERT INTO itemmaster (item_code, item_name, barcode, status)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (item_code) DO UPDATE
                  SET item_name = EXCLUDED.item_name,
                      barcode   = EXCLUDED.barcode,
                      status    = EXCLUDED.status
                """,
                (
                    str(row.get('item_code', '')).strip(),
                    str(row.get('item_name', '')).strip(),
                    str(row.get('barcode', '') or '').strip(),
                    str(row.get('status', 'active') or 'active').strip() or 'active',
                ),
            )
            count += 1
        conn.commit()
    return count


def get_itemmaster_count() -> int:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM itemmaster")
        return cur.fetchone()[0]
