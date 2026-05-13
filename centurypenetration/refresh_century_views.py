"""
Century Stock Penetration - Refresh Materialized Views
Run this script after data updates to refresh all analytics views
"""

import psycopg2
from psycopg2 import Error
import logging
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'century_penetration'
}

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('refresh_views_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# REFRESH VIEWS
# ============================================================

def refresh_materialized_views():
    """Refresh all materialized views in sequence"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        logger.info("=" * 60)
        logger.info("REFRESHING CENTURY PENETRATION MATERIALIZED VIEWS")
        logger.info(f"Started at: {datetime.now()}")
        logger.info("=" * 60)
        
        # Drop indexes first
        logger.info("\nDropping indexes...")
        cursor.execute("DROP INDEX IF EXISTS idx_mv_sales_item_shop;")
        cursor.execute("DROP INDEX IF EXISTS idx_mv_sit_item_shop;")
        cursor.execute("DROP INDEX IF EXISTS idx_mv_century_item_shop;")
        conn.commit()
        logger.info("✓ Indexes dropped")
        
        # Refresh in dependency order (without CONCURRENTLY to avoid duplicate errors)
        views = [
            'mv_sales_metrics',
            'mv_sit_summary',
            'mv_century_penetration',
            'mv_target_vs_achieve_century'
        ]
        
        for view in views:
            logger.info(f"\nRefreshing {view}...")
            start_time = datetime.now()
            
            cursor.execute(f"REFRESH MATERIALIZED VIEW {view};")
            conn.commit()
            
            duration = datetime.now() - start_time
            logger.info(f"✓ {view} refreshed in {duration}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {view};")
            count = cursor.fetchone()[0]
            logger.info(f"  Rows: {count:,}")
        
        # Recreate indexes
        logger.info("\nRecreating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mv_sales_item_shop ON mv_sales_metrics(item_code, shop_code);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mv_sit_item_shop ON mv_sit_summary(item_code, shop_code);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mv_century_item_shop ON mv_century_penetration(item_code, shop_code);")
        conn.commit()
        logger.info("✓ Indexes recreated")
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL VIEWS REFRESHED SUCCESSFULLY")
        logger.info("=" * 60)
        
        cursor.close()
        conn.close()
        
        return True
        
    except Error as e:
        logger.error(f"Error refreshing views: {e}")
        return False


if __name__ == "__main__":
    logger.info("\n" + "=" * 60)
    logger.info("CENTURY STOCK PENETRATION - VIEW REFRESH")
    logger.info("=" * 60)
    
    success = refresh_materialized_views()
    
    if success:
        logger.info("\n✓ View refresh completed successfully!")
        exit(0)
    else:
        logger.error("\n✗ View refresh failed!")
        exit(1)
