"""
Fix duplicate sales data in Century Penetration views
Recreates mv_sales_metrics with deduplication logic
"""

import psycopg2
from psycopg2 import Error
import logging
from datetime import datetime

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'century_penetration'
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fix_duplicates_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def fix_duplicate_views():
    """Recreate views with deduplication logic"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        logger.info("=" * 60)
        logger.info("FIXING DUPLICATE SALES IN MATERIALIZED VIEWS")
        logger.info(f"Started at: {datetime.now()}")
        logger.info("=" * 60)
        
        # Read SQL file
        logger.info("\nReading SQL fix script...")
        with open('fix_duplicate_sales.sql', 'r') as f:
            sql_script = f.read()
        
        # Execute entire script
        logger.info("\nExecuting SQL script...")
        cursor.execute(sql_script)
        conn.commit()
        
        # Get results
        results = cursor.fetchall()
        logger.info("\n✓ Views recreated successfully!")
        
        logger.info("\nRow counts:")
        for row in results[:-1]:  # All except last status message
            logger.info(f"  {row[0]}: {row[1]:,} rows")
        
        logger.info("\n" + "=" * 60)
        logger.info(results[-1][0])  # Status message
        logger.info("=" * 60)
        
        cursor.close()
        conn.close()
        
        return True
        
    except Error as e:
        logger.error(f"\n❌ Error: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    success = fix_duplicate_views()
    exit(0 if success else 1)
