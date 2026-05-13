"""
Century Stock Penetration - Daily Data Update Script
Loads daily sales data and refreshes all views
"""

import pandas as pd
import psycopg2
from psycopg2 import Error
from psycopg2.extras import execute_batch
from datetime import datetime
import logging
from pathlib import Path

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

DATA_DIR = Path(r'D:\Dashboard Code\tbl_data\centurypenetration')
DAILY_SALES_FILE = 'GEN_sales.csv'

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / 'daily_update_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# DAILY UPDATE PROCESS
# ============================================================

def load_daily_sales():
    """Load daily sales data from GEN_sales.csv"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        logger.info("=" * 60)
        logger.info("DAILY SALES UPDATE")
        logger.info(f"Started at: {datetime.now()}")
        logger.info("=" * 60)
        
        # Read CSV
        file_path = DATA_DIR / DAILY_SALES_FILE
        logger.info(f"Reading file: {file_path}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        logger.info(f"Loaded {len(df):,} rows")
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Convert date column
        if 'date_invoice' in df.columns:
            df['date_invoice'] = pd.to_datetime(df['date_invoice'], errors='coerce')
        
        # Filter to CENTURY brand only
        if 'brand' in df.columns:
            df = df[df['brand'].str.upper() == 'CENTURY']
            logger.info(f"Filtered to CENTURY brand: {len(df):,} rows")
        
        if df.empty:
            logger.warning("No CENTURY data to load")
            return False
        
        # Determine which partition to insert into (based on date)
        df['month'] = df['date_invoice'].dt.month
        
        # Group by month and insert
        for month, group_df in df.groupby('month'):
            month_names = {
                1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun',
                7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'
            }
            
            table_name = f"sales_{month_names[month]}25"
            logger.info(f"\nInserting {len(group_df):,} rows into {table_name}...")
            
            # Prepare data for insert
            columns = ['shop_code', 'item_code', 'date_invoice', 'qty', 'net_sales', 
                      'dept', 'groups', 'sub_group', 'brand']
            
            # Ensure all columns exist
            for col in columns:
                if col not in group_df.columns:
                    group_df[col] = None
            
            data = group_df[columns].values.tolist()
            
            # Insert with ON CONFLICT DO NOTHING (avoid duplicates)
            insert_query = f"""
                INSERT INTO {table_name} 
                (shop_code, item_code, date_invoice, qty, net_sales, dept, groups, sub_group, brand)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (shop_code, item_code, date_invoice) 
                DO UPDATE SET
                    qty = EXCLUDED.qty,
                    net_sales = EXCLUDED.net_sales
            """
            
            execute_batch(cursor, insert_query, data, page_size=1000)
            conn.commit()
            logger.info(f"✓ {table_name} updated")
        
        cursor.close()
        conn.close()
        
        logger.info("\n✓ Daily sales data loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading daily sales: {e}")
        return False


def refresh_views():
    """Refresh all materialized views - handling duplicates"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        logger.info("\n" + "=" * 60)
        logger.info("REFRESHING MATERIALIZED VIEWS")
        logger.info("=" * 60)
        
        # Step 1: Drop indexes
        logger.info("\nStep 1: Dropping indexes...")
        cursor.execute("DROP INDEX IF EXISTS idx_mv_sales_item_shop;")
        cursor.execute("DROP INDEX IF EXISTS idx_mv_sit_item_shop;")
        cursor.execute("DROP INDEX IF EXISTS idx_mv_century_item_shop;")
        conn.commit()
        logger.info("✓ Indexes dropped")
        
        # Step 2: Refresh views (normal, not concurrent)
        logger.info("\nStep 2: Refreshing views...")
        
        views = [
            'mv_sales_metrics',
            'mv_sit_summary',
            'mv_century_penetration',
            'mv_target_vs_achieve_century'
        ]
        
        for view in views:
            logger.info(f"  Refreshing {view}...")
            start_time = datetime.now()
            
            cursor.execute(f"REFRESH MATERIALIZED VIEW {view};")
            conn.commit()
            
            duration = datetime.now() - start_time
            logger.info(f"  ✓ {view} refreshed in {duration}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {view};")
            count = cursor.fetchone()[0]
            logger.info(f"    Rows: {count:,}")
        
        # Step 3: Recreate indexes (without duplicates)
        logger.info("\nStep 3: Recreating indexes...")
        
        # For sales metrics - use DISTINCT to avoid duplicates
        logger.info("  Creating index on mv_sales_metrics...")
        cursor.execute("""
            CREATE INDEX idx_mv_sales_item_shop 
            ON mv_sales_metrics(item_code, shop_code);
        """)
        conn.commit()
        logger.info("  ✓ idx_mv_sales_item_shop created")
        
        logger.info("  Creating index on mv_sit_summary...")
        cursor.execute("""
            CREATE INDEX idx_mv_sit_item_shop 
            ON mv_sit_summary(item_code, shop_code);
        """)
        conn.commit()
        logger.info("  ✓ idx_mv_sit_item_shop created")
        
        logger.info("  Creating index on mv_century_penetration...")
        cursor.execute("""
            CREATE INDEX idx_mv_century_item_shop 
            ON mv_century_penetration(item_code, shop_code);
        """)
        conn.commit()
        logger.info("  ✓ idx_mv_century_item_shop created")
        
        cursor.close()
        conn.close()
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ ALL VIEWS AND INDEXES UPDATED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error refreshing views: {e}")
        return False


def main():
    """Main daily update process"""
    logger.info("\n" + "=" * 80)
    logger.info("CENTURY STOCK PENETRATION - DAILY UPDATE")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 80)
    
    overall_start = datetime.now()
    
    # Step 1: Load daily sales
    sales_loaded = load_daily_sales()
    
    if not sales_loaded:
        logger.error("Failed to load daily sales data. Aborting.")
        return False
    
    # Step 2: Refresh views
    views_refreshed = refresh_views()
    
    if not views_refreshed:
        logger.error("Failed to refresh views.")
        return False
    
    # Summary
    total_duration = datetime.now() - overall_start
    logger.info("\n" + "=" * 80)
    logger.info("DAILY UPDATE COMPLETED SUCCESSFULLY")
    logger.info(f"Total Duration: {total_duration}")
    logger.info(f"Completed at: {datetime.now()}")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
