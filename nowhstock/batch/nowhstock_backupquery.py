"""
NO_WH Stock Backup Script - Automated Database Backup
Runs multiple backup queries and logs results with timestamps
Designed for Windows Task Scheduler
Version: 1.0
"""

import psycopg2
import logging
from datetime import datetime
import sys
import os

# ============================================================
# CONFIGURATION
# ============================================================

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'salesdata'
}

# Backup queries to execute (in order)
BACKUP_QUERIES = [
   
    {
        'name': 'NO_WH Stock Table Backup',
        'query': 'INSERT INTO nowhstock_tbl_new_backup SELECT * FROM nowhstock_tbl_new',
        'description': 'Full backup of NO_WH stock table'
    },
    {
        'name': 'Shop GRN Backup',
        'query': 'INSERT INTO sup_shop_grn_backup SELECT * FROM sup_shop_grn',
        'description': 'Full backup of supplier shop GRN data'
    },
    {
        'name': 'Item Details Backup',
        'query': 'INSERT INTO itemdetails_backup SELECT * FROM itemdetails',
        'description': 'Full backup of item details table'
    },
    {
        'name': 'Shop Expiry Backup',
        'query': 'INSERT INTO shopexpiry_backup SELECT * FROM shopexpiry',
        'description': 'Full backup of shop expiry data'
    },
    {
        'name': 'WH GRN Details Backup',
        'query': 'INSERT INTO whgrndetails_backup SELECT * FROM whgrndetails',
        'description': 'Full backup of warehouse GRN details'
    },
    {
        'name': 'SIT Data Backup',
        'query': 'INSERT INTO sit_data_backup SELECT * FROM sit_data',
        'description': 'Full backup of SIT data'
    },
    {
        'name': 'Sales 2025 Incremental Backup',
        'query': '''
            INSERT INTO sales_2025_backup 
            SELECT * FROM sales_2025 
            WHERE "DATE_INVOICE" > (
                SELECT COALESCE(MAX("DATE_INVOICE"), DATE '1900-01-01') 
                FROM sales_2025_backup
            )
        ''',
        'description': 'Backs up new sales records since last backup'
    },
]

# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging():
    """Setup logging with both file and console output"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(os.path.dirname(__file__), 'backup_logs')
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'nowhstock_backup_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# ============================================================
# DATABASE OPERATIONS
# ============================================================

def get_db_connection():
    """Establish database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        raise Exception(f"Failed to connect to database: {e}")

def execute_backup_query(conn, query_info, logger):
    """Execute a single backup query with error handling"""
    query_name = query_info['name']
    query_sql = query_info['query']
    description = query_info['description']
    
    logger.info("="*80)
    logger.info(f"Starting: {query_name}")
    logger.info(f"Description: {description}")
    logger.info("-"*80)
    
    start_time = datetime.now()
    
    try:
        with conn.cursor() as cursor:
            # Execute the backup query
            cursor.execute(query_sql)
            rows_affected = cursor.rowcount
            conn.commit()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"✅ SUCCESS: {query_name}")
            logger.info(f"   Rows affected: {rows_affected:,}")
            logger.info(f"   Duration: {duration:.2f} seconds")
            logger.info(f"   Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            return {
                'name': query_name,
                'status': 'SUCCESS',
                'rows_affected': rows_affected,
                'duration': duration,
                'error': None
            }
            
    except psycopg2.Error as e:
        conn.rollback()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        error_msg = str(e)
        logger.error(f"❌ FAILED: {query_name}")
        logger.error(f"   Error: {error_msg}")
        logger.error(f"   Duration: {duration:.2f} seconds")
        logger.error(f"   Failed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            'name': query_name,
            'status': 'FAILED',
            'rows_affected': 0,
            'duration': duration,
            'error': error_msg
        }
    
    except Exception as e:
        conn.rollback()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        error_msg = str(e)
        logger.error(f"❌ UNEXPECTED ERROR: {query_name}")
        logger.error(f"   Error: {error_msg}")
        logger.error(f"   Duration: {duration:.2f} seconds")
        
        return {
            'name': query_name,
            'status': 'ERROR',
            'rows_affected': 0,
            'duration': duration,
            'error': error_msg
        }

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main backup execution function"""
    logger = setup_logging()
    
    logger.info("="*80)
    logger.info("NO_WH STOCK BACKUP SCRIPT STARTED")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Database: {DB_CONFIG['database']} @ {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    logger.info(f"Total backup operations: {len(BACKUP_QUERIES)}")
    logger.info("="*80)
    
    results = []
    conn = None
    
    try:
        # Establish database connection
        logger.info("Connecting to database...")
        conn = get_db_connection()
        logger.info("✅ Database connection established")
        logger.info("")
        
        # Execute each backup query
        for idx, query_info in enumerate(BACKUP_QUERIES, 1):
            logger.info(f"[{idx}/{len(BACKUP_QUERIES)}] Processing backup operation...")
            result = execute_backup_query(conn, query_info, logger)
            results.append(result)
            logger.info("")
        
        # Generate summary report
        logger.info("="*80)
        logger.info("BACKUP SUMMARY REPORT")
        logger.info("="*80)
        
        total_success = sum(1 for r in results if r['status'] == 'SUCCESS')
        total_failed = sum(1 for r in results if r['status'] in ['FAILED', 'ERROR'])
        total_rows = sum(r['rows_affected'] for r in results)
        total_duration = sum(r['duration'] for r in results)
        
        logger.info(f"Total Operations: {len(results)}")
        logger.info(f"✅ Successful: {total_success}")
        logger.info(f"❌ Failed: {total_failed}")
        logger.info(f"📊 Total Rows Backed Up: {total_rows:,}")
        logger.info(f"⏱️  Total Duration: {total_duration:.2f} seconds")
        logger.info("")
        
        # Detailed results
        logger.info("Detailed Results:")
        logger.info("-"*80)
        for result in results:
            status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
            logger.info(f"{status_icon} {result['name']}")
            logger.info(f"   Status: {result['status']}")
            logger.info(f"   Rows: {result['rows_affected']:,}")
            logger.info(f"   Duration: {result['duration']:.2f}s")
            if result['error']:
                logger.info(f"   Error: {result['error']}")
            logger.info("")
        
        # Final status
        if total_failed == 0:
            logger.info("="*80)
            logger.info("✅✅✅ ALL BACKUP OPERATIONS COMPLETED SUCCESSFULLY ✅✅✅")
            logger.info("="*80)
            return 0
        else:
            logger.warning("="*80)
            logger.warning(f"⚠️  BACKUP COMPLETED WITH {total_failed} FAILURE(S)")
            logger.warning("="*80)
            return 1
            
    except Exception as e:
        logger.error("="*80)
        logger.error("❌❌❌ CRITICAL ERROR - BACKUP PROCESS FAILED ❌❌❌")
        logger.error(f"Error: {str(e)}")
        logger.error("="*80)
        return 2
        
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed")
        
        logger.info("")
        logger.info("="*80)
        logger.info(f"Backup script ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
