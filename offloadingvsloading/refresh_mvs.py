#!/usr/bin/env python3
"""Refresh offloading_vs_loading materialized views."""
import psycopg2
from contextlib import contextmanager

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'dbname': 'WH'
}

MVS = [
    'mv_ovl_detail_all',
    'mv_ovl_detail_short_received',
    'mv_ovl_detail_excess_received'
]

@contextmanager
def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()

def refresh_mvs():
    """Refresh all offloading_vs_loading materialized views."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            for mv in MVS:
                try:
                    print(f"⏳ Refreshing {mv}...")
                    cur.execute(f"REFRESH MATERIALIZED VIEW {mv}")
                    conn.commit()
                    print(f"✅ {mv} refreshed")
                except psycopg2.Error as e:
                    print(f"❌ Error refreshing {mv}: {e}")
                    conn.rollback()

if __name__ == '__main__':
    print("🔄 Starting MV refresh...")
    refresh_mvs()
    print("✅ All MVs refreshed successfully!")
