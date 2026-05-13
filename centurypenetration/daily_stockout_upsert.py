"""
Daily UPSERT script for Century Penetration Stockout Tracking
Runs daily to update current state and append daily snapshot
"""

import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import logging

# Setup logging with UTF-8 encoding to prevent Windows cp1252 errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daily_stockout_upsert.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'century_penetration'
}


def get_connection():
    """Create database connection"""
    return psycopg2.connect(**DB_CONFIG)


def resolve_source_relation(conn):
    """Resolve the best available source relation for stockout processing."""
    cursor = conn.cursor()
    try:
        # Prefer test relation if it exists (can be table or materialized view)
        cursor.execute("""
            SELECT 'table' AS rel_type
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'mv_century_penetration_test'
            UNION ALL
            SELECT 'materialized_view' AS rel_type
            FROM pg_matviews
            WHERE schemaname = 'public' AND matviewname = 'mv_century_penetration_test'
            LIMIT 1
        """)
        row = cursor.fetchone()
        if row:
            return 'mv_century_penetration_test', row[0]

        # Fallback to production materialized view
        cursor.execute("""
            SELECT 'materialized_view' AS rel_type
            FROM pg_matviews
            WHERE schemaname = 'public' AND matviewname = 'mv_century_penetration'
            LIMIT 1
        """)
        row = cursor.fetchone()
        if row:
            return 'mv_century_penetration', row[0]

        raise RuntimeError(
            "Neither mv_century_penetration_test nor mv_century_penetration exists in century_penetration DB."
        )
    finally:
        cursor.close()


def get_relation_columns(conn, relation_name: str):
    """Return lowercase column names for a relation in public schema."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT LOWER(column_name)
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            """,
            (relation_name,)
        )
        return {row[0] for row in cursor.fetchall()}
    finally:
        cursor.close()


def log_missing_ops_manager_rows(conn, source_relation: str, has_ops_col: bool):
    """Log rows/shops where ops manager mapping is missing and how to fix."""
    cursor = conn.cursor()
    try:
        if has_ops_col:
            missing_condition = "cp.ops_manager_name IS NULL OR TRIM(COALESCE(cp.ops_manager_name, '')) = ''"
            logger.warning("[WARNING] ops_manager_name column exists, but some rows are blank/NULL.")
        else:
            missing_condition = "TRUE"
            logger.warning("[WARNING] ops_manager_name column does NOT exist in source relation.")

        cursor.execute(
            f"""
            SELECT
                cp.shop_code,
                COUNT(*) AS item_count
            FROM {source_relation} cp
            WHERE {missing_condition}
            GROUP BY cp.shop_code
            ORDER BY item_count DESC, cp.shop_code
            LIMIT 20
            """
        )
        shop_rows = cursor.fetchall()

        cursor.execute(
            f"""
            SELECT
                cp.shop_code,
                cp.item_code,
                cp.item_name
            FROM {source_relation} cp
            WHERE {missing_condition}
            ORDER BY cp.shop_code, cp.item_code
            LIMIT 30
            """
        )
        sample_rows = cursor.fetchall()

        if shop_rows:
            logger.warning("[ACTION REQUIRED] Shops with missing Ops Manager mapping (top 20):")
            for shop_code, item_count in shop_rows:
                logger.warning(f"  - Shop {shop_code}: {item_count} items")

        if sample_rows:
            logger.warning("[ACTION REQUIRED] Sample rows needing Ops Manager mapping (shop_code, item_code, item_name):")
            for shop_code, item_code, item_name in sample_rows:
                logger.warning(f"  - {shop_code} | {item_code} | {item_name}")

        logger.warning("[HOW TO FIX]")
        logger.warning("  1) Open Century source mapping table/view used to build mv_century_penetration.")
        logger.warning("  2) Add/fix Ops Manager assignment by shop_code for the shops listed above.")
        logger.warning("  3) Refresh mv_century_penetration, then rerun daily_stockout_upsert.py.")
    finally:
        cursor.close()


def upsert_current_state(conn, source_relation: str, source_type: str):
    """
    Refresh source relation if it is a materialized view.
    If source is a table, skip refresh and return row count.
    """
    cursor = conn.cursor()

    if source_type == 'materialized_view':
        logger.info(f"Starting refresh of {source_relation}...")
        cursor.execute(f"REFRESH MATERIALIZED VIEW {source_relation};")
        conn.commit()
    else:
        logger.info(f"Source {source_relation} is a table; skipping refresh")
    
    # Get row count after refresh
    cursor.execute(f"SELECT COUNT(*) FROM {source_relation};")
    row_count = cursor.fetchone()[0]
    cursor.close()
    
    logger.info(f"[OK] Source ready: {source_relation} ({row_count} rows)")
    return row_count


def append_daily_snapshot(conn, source_relation: str):
    """
    APPEND today's snapshot to century_stockout_daily_snapshot
    This enables historical stockout % calculation
    """
    logger.info("Appending daily snapshot...")
    
    # Check if today's snapshot already exists
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) 
        FROM century_stockout_daily_snapshot 
        WHERE snapshot_date = CURRENT_DATE;
    """)
    existing_count = cursor.fetchone()[0]
    
    if existing_count > 0:
        logger.warning(f"[WARNING] Snapshot for today already exists ({existing_count} rows). Skipping...")
        cursor.close()
        return 0
    
    relation_columns = get_relation_columns(conn, source_relation)
    has_ops_col = 'ops_manager_name' in relation_columns

    if has_ops_col:
        ops_expr = "COALESCE(cp.ops_manager_name, 'Unknown')"
    else:
        ops_expr = "'Unknown'"

    # Insert today's snapshot
    insert_query = """
    INSERT INTO century_stockout_daily_snapshot (
        snapshot_date, shop_code, item_code, item_name, dept,
        sih, is_out_of_stock, ops_manager_name, created_at
    )
    SELECT 
        CURRENT_DATE as snapshot_date,
        cp.shop_code,
        cp.item_code,
        cp.item_name,
        cp.dept,
        cp.sih,
        (cp.sih <= 0) as is_out_of_stock,
        {ops_expr} as ops_manager_name,
        CURRENT_TIMESTAMP as created_at
    FROM {source_relation} cp;
    """

    cursor.execute(insert_query.format(source_relation=source_relation, ops_expr=ops_expr))
    rows_inserted = cursor.rowcount
    conn.commit()

    # Show actionable details if ops manager mapping is missing/incomplete
    if has_ops_col:
        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM {source_relation} cp
            WHERE cp.ops_manager_name IS NULL OR TRIM(COALESCE(cp.ops_manager_name, '')) = ''
            """
        )
        missing_count = cursor.fetchone()[0]
        if missing_count > 0:
            logger.warning(f"[WARNING] {missing_count} rows have blank/NULL ops_manager_name in {source_relation}.")
            log_missing_ops_manager_rows(conn, source_relation, has_ops_col=True)
    else:
        logger.warning(f"[WARNING] Source {source_relation} has no ops_manager_name column. Using 'Unknown' for all rows.")
        log_missing_ops_manager_rows(conn, source_relation, has_ops_col=False)

    cursor.close()
    
    logger.info(f"[OK] Daily snapshot appended: {rows_inserted} rows")
    return rows_inserted


def refresh_stockout_analysis(conn):
    """Refresh the stockout analysis materialized view"""
    logger.info("Refreshing stockout analysis view...")
    
    cursor = conn.cursor()
    cursor.execute("REFRESH MATERIALIZED VIEW mv_stockout_analysis;")
    conn.commit()
    cursor.close()
    
    logger.info("[OK] Stockout analysis view refreshed")


def get_summary_stats(conn, source_relation: str):
    """Get summary statistics after update"""
    cursor = conn.cursor()
    
    # Current stockout count (items with SIH <= 0)
    cursor.execute("""
        SELECT COUNT(*) 
        FROM {source_relation}
        WHERE sih <= 0;
    """.format(source_relation=source_relation))
    current_stockouts = cursor.fetchone()[0]
    
    # Total items
    cursor.execute(f"SELECT COUNT(*) FROM {source_relation};")
    total_items = cursor.fetchone()[0]
    
    # Snapshot days count
    cursor.execute("""
        SELECT COUNT(DISTINCT snapshot_date) 
        FROM century_stockout_daily_snapshot;
    """)
    snapshot_days = cursor.fetchone()[0]
    
    # Top ops manager by stockout %
    cursor.execute("""
        SELECT level_value, stockout_pct 
        FROM mv_stockout_analysis 
        WHERE level_type = 'Ops Manager' 
        ORDER BY stockout_pct DESC 
        LIMIT 3;
    """)
    top_ops_managers = cursor.fetchall()
    
    cursor.close()
    
    return {
        'current_stockouts': current_stockouts,
        'total_items': total_items,
        'stockout_rate': round(current_stockouts / total_items * 100, 2) if total_items > 0 else 0,
        'snapshot_days': snapshot_days,
        'top_ops_managers': top_ops_managers
    }


def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("STARTING DAILY STOCKOUT TRACKING UPDATE")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    try:
        conn = get_connection()
        logger.info("[OK] Database connection established")

        source_relation, source_type = resolve_source_relation(conn)
        logger.info(f"Using source relation: {source_relation} ({source_type})")
        
        # Step 1: UPSERT current state
        upsert_count = upsert_current_state(conn, source_relation, source_type)
        
        # Step 2: APPEND daily snapshot
        snapshot_count = append_daily_snapshot(conn, source_relation)
        
        # Step 3: Refresh analysis view
        refresh_stockout_analysis(conn)
        
        # Step 4: Get summary stats
        stats = get_summary_stats(conn, source_relation)
        
        logger.info("=" * 80)
        logger.info("SUMMARY STATISTICS:")
        logger.info(f"  Total Items: {stats['total_items']}")
        logger.info(f"  Current Stockouts: {stats['current_stockouts']} ({stats['stockout_rate']}%)")
        logger.info(f"  Snapshot Days: {stats['snapshot_days']}")
        logger.info(f"  Top Ops Managers by Stockout %:")
        for idx, (manager, pct) in enumerate(stats['top_ops_managers'], 1):
            logger.info(f"    {idx}. {manager}: {pct}%")
        logger.info("=" * 80)
        logger.info("[OK] DAILY UPDATE COMPLETED SUCCESSFULLY")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"[ERROR] ERROR during update: {str(e)}")
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()
