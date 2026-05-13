"""
============================================================
DAILY SALES DATA UPLOAD SCRIPT
============================================================
Purpose: Upload daily sales data to sales_2025 table
Process:
  1. Backup yesterday's data to sales_2025_backup
  2. Drop indexes for faster upload
  3. Upload data from CSV
  4. Recreate indexes
  5. Analyze table

Database: salesdata (port 3307)
CSV Source: D:/Dashboard Code/tbl_data/salesdata/dailysales.csv
============================================================
"""

import psycopg2
import os
import sys
from datetime import datetime

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'salesdata'
}

# CSV file path
CSV_FILE = r'D:/Dashboard Code/tbl_data/salesdata/dailysales.csv'

def log(message):
    """Print log message with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def execute_sql(conn, sql, description, fetch=False):
    """Execute SQL statement with error handling"""
    try:
        log(f"⏳ {description}...")
        cursor = conn.cursor()
        cursor.execute(sql)
        
        if fetch:
            result = cursor.fetchone()
            cursor.close()
            return result
        
        conn.commit()
        cursor.close()
        log(f"✅ {description} - SUCCESS")
        return True
    except Exception as e:
        conn.rollback()
        log(f"❌ {description} - FAILED: {str(e)}")
        return False

def main():
    """Main execution function"""
    print("=" * 60)
    log("DAILY SALES DATA UPLOAD STARTING")
    print("=" * 60)
    
    # Step 0: Verify CSV file exists
    if not os.path.exists(CSV_FILE):
        log(f"❌ ERROR: CSV file not found: {CSV_FILE}")
        log("Please ensure the file exists before running this script.")
        input("\nPress Enter to exit...")
        sys.exit(1)
    else:
        log(f"✅ CSV file found: {CSV_FILE}")
    
    # Connect to database
    try:
        log("Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        log("✅ Connected to database successfully")
    except Exception as e:
        log(f"❌ Database connection failed: {str(e)}")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    try:
        # Step 1: Backup yesterday's data
        print("\n" + "=" * 60)
        log("STEP 1: Backing up new data to sales_2025_backup")
        print("=" * 60)
        
        backup_sql = """
        INSERT INTO sales_2025_backup
        SELECT *
        FROM sales_2025
        WHERE "DATE_INVOICE" > (
            SELECT COALESCE(MAX("DATE_INVOICE"), DATE '1900-01-01')
            FROM sales_2025_backup
        );
        """
        
        result = execute_sql(conn, backup_sql, "Backup yesterday's data")
        if result:
            # Get backup count
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sales_2025_backup")
            backup_count = cursor.fetchone()[0]
            cursor.close()
            log(f"📊 Total backup records: {backup_count:,}")
        
        # Step 2: Drop indexes for faster upload
        print("\n" + "=" * 60)
        log("STEP 2: Dropping indexes for faster upload")
        print("=" * 60)
        
        indexes = [
            'idx_sales_2025_date',
            'idx_sales_2025_date_dept',
            'idx_sales_2025_item_date_qty',
            'idx_sales_2025_item_shop',
            'idx_sales_2025_item_shop_date'
        ]
        
        for idx in indexes:
            execute_sql(conn, f"DROP INDEX IF EXISTS {idx};", f"Drop index {idx}")
        
        # Step 3: Disable autovacuum for faster upload
        print("\n" + "=" * 60)
        log("STEP 3: Disabling autovacuum")
        print("=" * 60)
        
        execute_sql(conn, "ALTER TABLE sales_2025 SET (autovacuum_enabled = false);", 
                   "Disable autovacuum")
        
        # Step 4: Upload data from CSV
        print("\n" + "=" * 60)
        log("STEP 4: Uploading data from CSV")
        print("=" * 60)
        
        # Get current row count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sales_2025")
        before_count = cursor.fetchone()[0]
        cursor.close()
        log(f"📊 Records before upload: {before_count:,}")
        
        copy_sql = f"""
        COPY sales_2025 ("SHOP_CODE", "ITEM_CODE", "ITEM_NAME", "DEPT", "GROUPS", "SUB_GROUP", "QTY", "NET_SALES", "DATE_INVOICE") 
        FROM '{CSV_FILE}' CSV HEADER;
        """
        
        result = execute_sql(conn, copy_sql, "Upload CSV data")
        
        if result:
            # Get new row count
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sales_2025")
            after_count = cursor.fetchone()[0]
            cursor.close()
            uploaded = after_count - before_count
            log(f"📊 Records after upload: {after_count:,}")
            log(f"📊 New records uploaded: {uploaded:,}")
        
        # Step 5: Recreate indexes
        print("\n" + "=" * 60)
        log("STEP 5: Recreating indexes (this may take 5-10 minutes)")
        print("=" * 60)
        
        index_definitions = [
            ('idx_sales_2025_date', 
             'CREATE INDEX idx_sales_2025_date ON public.sales_2025 USING btree ("DATE_INVOICE")'),
            
            ('idx_sales_2025_date_dept',
             'CREATE INDEX idx_sales_2025_date_dept ON public.sales_2025 USING btree ("DATE_INVOICE", "DEPT") INCLUDE ("NET_SALES", "QTY")'),
            
            ('idx_sales_2025_item_date_qty',
             'CREATE INDEX idx_sales_2025_item_date_qty ON public.sales_2025 USING btree ("ITEM_CODE", "DATE_INVOICE") INCLUDE ("SHOP_CODE", "QTY")'),
            
            ('idx_sales_2025_item_shop',
             'CREATE INDEX idx_sales_2025_item_shop ON public.sales_2025 USING btree ("ITEM_CODE", "SHOP_CODE")'),
            
            ('idx_sales_2025_item_shop_date',
             'CREATE INDEX idx_sales_2025_item_shop_date ON public.sales_2025 USING btree ("ITEM_CODE", "SHOP_CODE", "DATE_INVOICE")')
        ]
        
        for idx_name, idx_sql in index_definitions:
            execute_sql(conn, idx_sql, f"Create index {idx_name}")
        
        # Step 6: Re-enable autovacuum
        print("\n" + "=" * 60)
        log("STEP 6: Re-enabling autovacuum")
        print("=" * 60)
        
        execute_sql(conn, "ALTER TABLE sales_2025 SET (autovacuum_enabled = true);",
                   "Re-enable autovacuum")
        
        # Step 7: Analyze table
        print("\n" + "=" * 60)
        log("STEP 7: Analyzing table (updating statistics)")
        print("=" * 60)
        
        execute_sql(conn, "ANALYZE sales_2025;", "Analyze sales_2025 table")
        
        # Step 8: Show final statistics
        print("\n" + "=" * 60)
        log("STEP 8: Final Statistics")
        print("=" * 60)
        
        cursor = conn.cursor()
        
        # Get latest date
        cursor.execute('SELECT MAX("DATE_INVOICE")::date FROM sales_2025')
        latest_date = cursor.fetchone()[0]
        log(f"📅 Latest invoice date: {latest_date}")
        
        # Get total records
        cursor.execute('SELECT COUNT(*) FROM sales_2025')
        total_records = cursor.fetchone()[0]
        log(f"📊 Total records: {total_records:,}")
        
        # Get table size
        cursor.execute("""
            SELECT pg_size_pretty(pg_total_relation_size('sales_2025')) as size
        """)
        table_size = cursor.fetchone()[0]
        log(f"💾 Table size: {table_size}")
        
        # Get index sizes
        cursor.execute("""
            SELECT pg_size_pretty(pg_indexes_size('sales_2025')) as size
        """)
        index_size = cursor.fetchone()[0]
        log(f"🔍 Index size: {index_size}")
        
        cursor.close()
        
        # Success message
        print("\n" + "=" * 60)
        log("✅ DAILY SALES DATA UPLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        log("Next steps:")
        log("  1. Run daily_maintenance.sql to refresh materialized views")
        log("  2. Restart Streamlit app if running")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        log(f"❌ CRITICAL ERROR: {str(e)}")
        print("=" * 60)
        log("Upload failed. Database may be in inconsistent state.")
        log("Please check the error message and retry.")
        
    finally:
        # Close database connection
        if conn:
            conn.close()
            log("Database connection closed")
    
    # Pause before exit
    print("\n")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
