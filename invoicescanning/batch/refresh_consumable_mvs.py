"""
Refresh all materialized views for the Invoice Scanning (consumable till) dashboard.

Usage:
    python invoicescanning/batch/refresh_consumable_mvs.py

Schedule via Windows Task Scheduler to run after the daily data import.
"""

import psycopg2
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": "localhost",
    "port": 3307,
    "user": "postgres",
    "password": "hello",
    "dbname": "WH",
}

# Order matters: independent MVs first, any that depend on others last.
MATERIALIZED_VIEWS = [
    "mv_wh_erp_daily",
    "mv_wh_invoices_agg_daily",
    "mv_wh_erp_cashier_sessions_daily",
    "mv_wh_erp_test_bills_cashier_daily",
    "mv_wh_manager_handover_daily",
    "mv_wh_alerts_daily",
]


def refresh_views():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    overall_start = datetime.now()
    failed = []

    for mv in MATERIALIZED_VIEWS:
        t0 = datetime.now()
        try:
            logger.info(f"Refreshing {mv} ...")
            cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {mv};")
            elapsed = (datetime.now() - t0).total_seconds()
            logger.info(f"  OK  {mv} refreshed in {elapsed:.1f}s")
        except Exception as exc:
            elapsed = (datetime.now() - t0).total_seconds()
            logger.error(f"  FAIL {mv} after {elapsed:.1f}s: {exc}")
            failed.append(mv)

    cur.close()
    conn.close()

    total = (datetime.now() - overall_start).total_seconds()
    if failed:
        logger.error(f"Refresh completed with errors in {total:.1f}s — failed: {failed}")
    else:
        logger.info(f"All views refreshed successfully in {total:.1f}s")

    return len(failed) == 0


if __name__ == "__main__":
    ok = refresh_views()
    raise SystemExit(0 if ok else 1)
