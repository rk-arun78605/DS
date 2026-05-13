#!/usr/bin/env python3
"""
Load All Data - Python Wrapper with Progress Tracking
======================================================
This script loads all CSV data into PostgreSQL tables with detailed progress tracking.
Provides better error handling and validation than the pure SQL approach.

Usage:
    python load_all_data.py

Features:
- Pre-validates CSV files exist before starting
- Shows progress for each table load
- Detailed error messages
- Row count validation
- Comprehensive logging
"""

import psycopg2
import sys
import os
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'database': 'salesdata',
    'user': 'postgres',
    'password': 'hello'
}

# Base path for CSV files
CSV_BASE_PATH = Path('D:/Dashboard Code/tbl_data')

# Table loading configuration (order matters!)
TABLES_CONFIG = [
    {
        'name': 'nowhstock_tbl_new',
        'csv_path': CSV_BASE_PATH / 'nowhstock' / 'nowhdata.csv',
        'columns': ['item_code', 'item_name', 'shop_code'],
        'truncate': True,
        'indexes': [],
        'encoding': 'WIN1252'
    },
    {
        'name': 'sup_shop_grn',
        'csv_path': CSV_BASE_PATH / 'shopgrndate' / 'sup_last_grn.csv',
        'columns': ['item_code', 'item_name', 'shop_code', 'shop_grn_date', 'wh_grn_date', 'shop_stock'],
        'truncate': False,
        'indexes': [
            'CREATE INDEX IF NOT EXISTS idx_sup_shop_grn_item_shop ON sup_shop_grn ("item_code", "shop_code")',
            'CREATE INDEX IF NOT EXISTS idx_sup_shop_grn_item ON sup_shop_grn ("item_code")',
            'CREATE INDEX IF NOT EXISTS idx_sup_shop_grn_shop ON sup_shop_grn ("shop_code")'
        ],
        'drop_indexes': [
            'DROP INDEX IF EXISTS idx_sup_shop_grn_item_shop',
            'DROP INDEX IF EXISTS idx_sup_shop_grn_item',
            'DROP INDEX IF EXISTS idx_sup_shop_grn_shop'
        ],
        'encoding': 'WIN1252'
    },
    {
        'name': 'itemdetails',
        'csv_path': CSV_BASE_PATH / 'ITEMdetails' / 'item_details.csv',
        'columns': ['vc_item_code', 'item_name', 'dept', 'groups', 'sub_group', 'type', 'vc_supplier_name', 'nu_qty_received'],
        'truncate': True,
        'indexes': [
            'CREATE INDEX IF NOT EXISTS idx_itemdetails_item_code ON itemdetails (vc_item_code)',
            'CREATE INDEX IF NOT EXISTS idx_itemdetails_groups ON itemdetails (groups, sub_group)',
            'CREATE INDEX IF NOT EXISTS idx_itemdetails_supplier ON itemdetails (vc_supplier_name)'
        ],
        'drop_indexes': [
            'DROP INDEX IF EXISTS idx_itemdetails_pk',
            'DROP INDEX IF EXISTS idx_itemdetails_item_code',
            'DROP INDEX IF EXISTS idx_itemdetails_groups',
            'DROP INDEX IF EXISTS idx_itemdetails_supplier'
        ],
        'encoding': 'WIN1252'
    },
    {
        'name': 'shopexpiry',
        'csv_path': CSV_BASE_PATH / 'shopexpiry' / 'shopexpiry.csv',
        'columns': ['ITEM_CODE', 'SHOP_EXPIRY_DATE', 'SHOP_CODE'],
        'truncate': True,
        'indexes': [],
        'encoding': 'UTF8'
    },
    {
        'name': 'whgrndetails',
        'csv_path': CSV_BASE_PATH / 'whgrndetails' / 'whgrndate.csv',
        'columns': ['ITEM_CODE', 'TYPE', 'SUPPLIER_NAME', 'WH_LAST_GRN_DATE', 'WH_QTY_RECEIVED'],
        'truncate': True,
        'indexes': [],
        'encoding': 'WIN1252'
    },
    {
        'name': 'sit_data',
        'csv_path': CSV_BASE_PATH / 'SIT' / 'sit_sup.csv',
        'columns': ['shop_code', 'item_code', 'dt_trans_date', 'nu_transit_qty'],
        'truncate': False,
        'indexes': [],
        'encoding': 'WIN1252'
    },
    {
        'name': 'sales_2025',
        'csv_path': CSV_BASE_PATH / 'salesdata' / 'dailysales.csv',
        'columns': ['SHOP_CODE', 'ITEM_CODE', 'ITEM_NAME', 'DEPT', 'GROUPS', 'SUB_GROUP', 'QTY', 'NET_SALES', 'DATE_INVOICE'],
        'truncate': False,
        'indexes': [
            'CREATE INDEX IF NOT EXISTS idx_sales_2025_date ON sales_2025 ("DATE_INVOICE")',
            'CREATE INDEX IF NOT EXISTS idx_sales_2025_item_shop ON sales_2025 ("ITEM_CODE", "SHOP_CODE")',
            'CREATE INDEX IF NOT EXISTS idx_sales_2025_covering ON sales_2025 ("DATE_INVOICE", "ITEM_CODE", "SHOP_CODE") INCLUDE ("QTY")'
        ],
        'drop_indexes': [
            'DROP INDEX IF EXISTS idx_sales_2025_date',
            'DROP INDEX IF EXISTS idx_sales_2025_item_shop',
            'DROP INDEX IF EXISTS idx_sales_2025_covering'
        ],
        'disable_autovacuum': True,
        'encoding': 'UTF8'
    }
]

# Script directory
SCRIPT_DIR = Path(__file__).parent
LOG_FILE = SCRIPT_DIR / 'data_load_log.txt'
LOG_RETENTION_DAYS = 30  # Keep only last 30 days of logs

# Date columns for duplicate detection (table_name -> date_column_name)
DATE_COLUMNS = {
    'sup_shop_grn': 'shop_grn_date',  # Primary date column for GRN tracking
    'sales_2025': 'DATE_INVOICE',
    'shopexpiry': 'SHOP_EXPIRY_DATE',
    'whgrndetails': 'WH_LAST_GRN_DATE',
    'sit_data': 'dt_trans_date'
    # nowhstock_tbl_new, itemdetails: No date columns, always load
}

# ============================================================
# LOGGING SETUP
# ============================================================

def clean_old_logs(log_file: Path, retention_days: int = 30):
    """Remove log entries older than retention_days"""
    if not log_file.exists():
        return
    
    try:
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Read existing log file
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Filter lines to keep only recent entries
        filtered_lines = []
        for line in lines:
            # Try to extract timestamp from log line format: "YYYY-MM-DD HH:MM:SS"
            if len(line) > 19:
                try:
                    log_timestamp = datetime.strptime(line[:19], '%Y-%m-%d %H:%M:%S')
                    if log_timestamp >= cutoff_date:
                        filtered_lines.append(line)
                except ValueError:
                    # Not a timestamped line (separator, etc.) - keep if within retention window
                    # Keep if previous line was kept
                    if filtered_lines:
                        filtered_lines.append(line)
            else:
                # Short line (separator, etc.) - keep if within retention window
                if filtered_lines:
                    filtered_lines.append(line)
        
        # Write back filtered logs
        if filtered_lines:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.writelines(filtered_lines)
        else:
            # If all logs are old, clear the file
            log_file.unlink(missing_ok=True)
            
    except Exception as e:
        # If cleanup fails, just continue (don't block the main operation)
        print(f"Warning: Could not clean old logs: {e}")

def setup_logging():
    """Configure logging to both console and file"""
    # Clean old logs before starting new session
    clean_old_logs(LOG_FILE, LOG_RETENTION_DAYS)
    
    logger = logging.getLogger('load_all_data')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(logger, text, char='='):
    """Print a formatted header"""
    separator = char * 80
    logger.info('')
    logger.info(separator)
    logger.info(text)
    logger.info(separator)

def get_max_date_from_db(logger, conn, table_name, date_column):
    """Get maximum date from database table"""
    try:
        cursor = conn.cursor()
        query = f'SELECT MAX("{date_column}") FROM {table_name}'
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        
        if result and result[0]:
            return result[0]
        return None
    except Exception as e:
        logger.warning(f"Could not get max date from {table_name}.{date_column}: {e}")
        return None

def get_max_date_from_csv(logger, csv_path, date_column, encoding='UTF8'):
    """Get maximum date from CSV file by reading the date column"""
    try:
        import csv
        from datetime import datetime
        
        max_date = None
        row_count = 0
        
        with open(csv_path, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f)
            
            # Check if date column exists in CSV
            if date_column not in reader.fieldnames:
                logger.warning(f"Date column '{date_column}' not found in {csv_path.name}")
                return None
            
            for row in reader:
                row_count += 1
                date_str = row.get(date_column, '').strip()
                
                if date_str and date_str.lower() not in ['null', 'none', '']:
                    try:
                        # Try multiple date formats
                        for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d', '%d/%m/%Y']:
                            try:
                                parsed_date = datetime.strptime(date_str, fmt).date()
                                if max_date is None or parsed_date > max_date:
                                    max_date = parsed_date
                                break
                            except ValueError:
                                continue
                    except Exception:
                        pass
        
        logger.info(f"Scanned {row_count:,} rows in {csv_path.name}")
        return max_date
        
    except Exception as e:
        logger.warning(f"Could not parse max date from {csv_path.name}: {e}")
        return None

def validate_data_freshness(logger, conn, table_name, csv_path, encoding='UTF8'):
    """Check if CSV has newer data than database - return True to load, False to skip"""
    
    # Check if table has a date column for validation
    if table_name not in DATE_COLUMNS:
        logger.info(f"📋 {table_name}: No date column - will always load")
        return True
    
    date_column = DATE_COLUMNS[table_name]
    logger.info(f"\n{'='*80}")
    logger.info(f"🔍 DUPLICATE DETECTION: {table_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Table: {table_name}")
    logger.info(f"File: {csv_path.name}")
    logger.info(f"Date Column: {date_column}")
    logger.info(f"-"*80)
    
    # Get max date from database
    db_max_date = get_max_date_from_db(logger, conn, table_name, date_column)
    logger.info(f"DB Max Date: {db_max_date if db_max_date else 'N/A (empty table)'}")
    
    # Get max date from CSV
    csv_max_date = get_max_date_from_csv(logger, csv_path, date_column, encoding)
    logger.info(f"CSV Max Date: {csv_max_date if csv_max_date else 'N/A (no valid dates)'}")
    logger.info(f"-"*80)
    
    # Decision logic
    if db_max_date is None:
        # Table is empty - always load
        logger.info("✅ DECISION: LOAD (table is empty - first load)")
        logger.info(f"{'='*80}\n")
        return True
    
    if csv_max_date is None:
        # Cannot determine CSV date - load anyway (with warning)
        logger.warning("⚠️  DECISION: LOAD (cannot determine CSV max date)")
        logger.info(f"{'='*80}\n")
        return True
    
    # Compare dates
    if csv_max_date <= db_max_date:
        # CSV has same or older data - SKIP
        logger.warning(f"⊘ DECISION: SKIP LOADING")
        logger.warning(f"   Reason: CSV has same or older data")
        logger.warning(f"   CSV max date ({csv_max_date}) <= DB max date ({db_max_date})")
        logger.warning(f"   Loading would create duplicate data!")
        logger.info(f"{'='*80}\n")
        return False
    else:
        # CSV has newer data - LOAD
        days_diff = (csv_max_date - db_max_date).days
        logger.info(f"✅ DECISION: LOAD (CSV has newer data)")
        logger.info(f"   CSV is {days_diff} day(s) newer than DB")
        logger.info(f"{'='*80}\n")
        return True

def validate_csv_files(logger):
    """Check if all CSV files exist before starting"""
    logger.info('Validating CSV files...')
    missing_files = []
    
    for table_config in TABLES_CONFIG:
        csv_path = table_config['csv_path']
        if not csv_path.exists():
            missing_files.append(str(csv_path))
            logger.error(f"✗ Missing: {csv_path}")
        else:
            size_mb = csv_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Found: {csv_path.name} ({size_mb:.2f} MB)")
    
    if missing_files:
        logger.error('')
        logger.error('❌ Cannot proceed - missing CSV files:')
        for file in missing_files:
            logger.error(f"   {file}")
        return False
    
    logger.info('')
    logger.info('✓ All CSV files validated successfully')
    return True

def load_table(logger, conn, table_config):
    """Load data for a single table"""
    table_name = table_config['name']
    csv_path = table_config['csv_path']
    columns = table_config['columns']
    encoding = table_config.get('encoding', 'UTF8')
    
    print_header(logger, f"Loading {table_name}", '-')
    logger.info(f"Source: {csv_path.name}")
    
    cursor = conn.cursor()
    start_time = time.time()
    
    try:
        # Drop indexes if specified
        if 'drop_indexes' in table_config:
            logger.info('Dropping existing indexes...')
            for drop_sql in table_config['drop_indexes']:
                cursor.execute(drop_sql)
            conn.commit()
        
        # Disable autovacuum for large tables
        if table_config.get('disable_autovacuum', False):
            logger.info('Disabling autovacuum for faster loading...')
            cursor.execute(f'ALTER TABLE {table_name} SET (autovacuum_enabled = false)')
            conn.commit()
        
        # Truncate table
        if table_config.get('truncate', False):
            logger.info(f'Truncating {table_name}...')
            cursor.execute(f'TRUNCATE TABLE {table_name}')
            conn.commit()
        
        # Build COPY command
        columns_str = ', '.join([f'"{col}"' for col in columns])
        copy_sql = f"""
            COPY {table_name} ({columns_str})
            FROM '{csv_path.as_posix()}'
            CSV HEADER ENCODING '{encoding}'
        """
        
        logger.info(f'Loading data from CSV...')
        cursor.execute(copy_sql)
        row_count = cursor.rowcount
        conn.commit()
        
        elapsed = time.time() - start_time
        logger.info(f'✓ Loaded {row_count:,} rows in {elapsed:.2f}s')
        
        # Recreate indexes if specified
        if 'indexes' in table_config and table_config['indexes']:
            logger.info('Creating indexes...')
            index_start = time.time()
            for index_sql in table_config['indexes']:
                cursor.execute(index_sql)
            conn.commit()
            index_elapsed = time.time() - index_start
            logger.info(f'✓ Created {len(table_config["indexes"])} indexes in {index_elapsed:.2f}s')
        
        # Re-enable autovacuum
        if table_config.get('disable_autovacuum', False):
            logger.info('Re-enabling autovacuum...')
            cursor.execute(f'ALTER TABLE {table_name} SET (autovacuum_enabled = true)')
            conn.commit()
        
        # Analyze table
        logger.info(f'Analyzing {table_name}...')
        cursor.execute(f'ANALYZE {table_name}')
        conn.commit()
        
        total_elapsed = time.time() - start_time
        logger.info(f'✓ {table_name} completed in {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)')
        
        return True, row_count
        
    except Exception as e:
        logger.error(f'✗ Error loading {table_name}: {e}')
        conn.rollback()
        return False, 0
    finally:
        cursor.close()

def print_summary(logger, conn):
    """Print summary of all loaded tables"""
    print_header(logger, 'DATA LOAD SUMMARY')
    
    cursor = conn.cursor()
    
    logger.info(f"\n{'Table Name':<25} {'Rows':<15} {'Size'}")
    logger.info('-' * 80)
    
    for table_config in TABLES_CONFIG:
        table_name = table_config['name']
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            cursor.execute(f"""
                SELECT pg_size_pretty(pg_total_relation_size('{table_name}'))
            """)
            size = cursor.fetchone()[0]
            
            logger.info(f"{table_name:<25} {row_count:<15,} {size}")
        except Exception as e:
            logger.error(f"{table_name:<25} ERROR: {e}")
    
    cursor.close()

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main data loading function"""
    logger = setup_logging()
    
    # Log separator for new run
    logger.info('\n' + '=' * 80)
    logger.info('=' * 80)
    
    print_header(logger, "LOAD ALL DATA - PostgreSQL Data Import")
    start_time = datetime.now()
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Database: {DB_CONFIG['database']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")
    logger.info(f"Log file: {LOG_FILE}")
    
    # Validate CSV files
    print_header(logger, 'STEP 1: Validating CSV Files')
    if not validate_csv_files(logger):
        return 1
    
    conn = None
    try:
        # Connect to database
        print_header(logger, 'STEP 2: Connecting to Database')
        logger.info(f"Connecting to {DB_CONFIG['database']}...")
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        logger.info('✓ Connected successfully')
        
        # Load each table
        print_header(logger, 'STEP 3: Loading Tables')
        total_rows = 0
        failed_tables = []
        skipped_tables = []
        
        for i, table_config in enumerate(TABLES_CONFIG, 1):
            table_name = table_config['name']
            logger.info(f"\n[{i}/{len(TABLES_CONFIG)}] {table_name}")
            
            # Validate data freshness before loading
            should_load = validate_data_freshness(
                logger, 
                conn, 
                table_name, 
                table_config['csv_path'],
                table_config.get('encoding', 'UTF8')
            )
            
            if not should_load:
                logger.warning(f"⊘ Skipping {table_name} - CSV not updated (duplicate data prevention)")
                skipped_tables.append(table_name)
                continue
            
            # Load the table
            success, row_count = load_table(logger, conn, table_config)
            
            if success:
                total_rows += row_count
            else:
                failed_tables.append(table_name)
        
        # Print summary
        print_header(logger, 'STEP 4: Summary')
        print_summary(logger, conn)
        
        # Close connection
        conn.close()
        
        # Final status
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if failed_tables:
            print_header(logger, '⚠️  COMPLETED WITH ERRORS', '=')
            logger.error(f"Failed tables: {', '.join(failed_tables)}")
        else:
            if skipped_tables:
                print_header(logger, '✅ COMPLETED (Some tables skipped)', '=')
                logger.info(f"Skipped tables (duplicate prevention): {', '.join(skipped_tables)}")
            else:
                print_header(logger, '✅ ALL DATA LOADED SUCCESSFULLY!', '=')
        
        logger.info(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"Total rows loaded: {total_rows:,}")
        logger.info(f"Tables processed: {len(TABLES_CONFIG) - len(skipped_tables) - len(failed_tables)}/{len(TABLES_CONFIG)}")
        if skipped_tables:
            logger.info(f"Tables skipped (no new data): {len(skipped_tables)}")
        if failed_tables:
            logger.info(f"Tables failed: {len(failed_tables)}")
        logger.info('')
        logger.info('💡 Next steps:')
        logger.info('   1. Refresh materialized views: python refresh_materialized_views.py')
        logger.info('   2. Restart Streamlit: streamlit run nowhstock_ds.py')
        logger.info('')
        
        return 1 if failed_tables else 0
        
    except psycopg2.Error as e:
        logger.error('')
        logger.error(f"❌ Database Error: {e}")
        logger.error(f"   Error Code: {e.pgcode}")
        return 1
    except Exception as e:
        logger.error('')
        logger.error(f"❌ Unexpected Error: {e}")
        return 1
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
