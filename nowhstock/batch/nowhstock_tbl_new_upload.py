"""
============================================================
NO_WH STOCK TABLE UPLOAD SCRIPT
============================================================
Purpose: Upload NO_WH stock data to nowhstock_tbl_new table
Process:
  1. Backup existing data to nowhstock_tbl_new_backup
  2. Truncate the main table
  3. Upload data from CSV
  4. Analyze table

Database: salesdata (port 3307)
CSV Source: D:/Dashboard Code/tbl_data/nowhstock/nowhdata.csv
Table: nowhstock_tbl_new
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
CSV_FILE = r'D:/Dashboard Code/tbl_data/nowhstock/nowhdata.csv'

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
    log("NO_WH STOCK TABLE UPLOAD STARTING")
    print("=" * 60)
    
    # Step 0: Verify CSV file exists
    if not os.path.exists(CSV_FILE):
        log(f"❌ ERROR: CSV file not found: {CSV_FILE}")
        log("Please ensure the file exists before running this script.")
        input("\nPress Enter to exit...")
        sys.exit(1)
    else:
        log(f"✅ CSV file found: {CSV_FILE}")
        # Get CSV file size
        file_size = os.path.getsize(CSV_FILE)
        log(f"📁 CSV file size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    
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
        # Step 1: Backup existing data
        print("\n" + "=" * 60)
        log("STEP 1: Backing up existing data to nowhstock_tbl_new_backup")
        print("=" * 60)
        
        # Get current record count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM nowhstock_tbl_new")
        before_count = cursor.fetchone()[0]
        cursor.close()
        log(f"📊 Current records in nowhstock_tbl_new: {before_count:,}")
        
        if before_count > 0:
            backup_sql = """
            INSERT INTO nowhstock_tbl_new_backup
            SELECT * FROM nowhstock_tbl_new;
            """
            
            result = execute_sql(conn, backup_sql, "Backup existing data")
            
            if result:
                # Get backup count
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM nowhstock_tbl_new_backup")
                backup_count = cursor.fetchone()[0]
                cursor.close()
                log(f"📊 Total backup records: {backup_count:,}")
        else:
            log("ℹ️  No data to backup (table is empty)")
        
        # Step 2: Truncate main table
        print("\n" + "=" * 60)
        log("STEP 2: Truncating nowhstock_tbl_new table")
        print("=" * 60)
        
        truncate_sql = "TRUNCATE TABLE nowhstock_tbl_new;"
        result = execute_sql(conn, truncate_sql, "Truncate table")
        
        if result:
            # Verify table is empty
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nowhstock_tbl_new")
            after_truncate = cursor.fetchone()[0]
            cursor.close()
            log(f"📊 Records after truncate: {after_truncate}")
        
        # Step 3: Upload data from CSV
        print("\n" + "=" * 60)
        log("STEP 3: Uploading data from CSV (WIN1252 encoding)")
        print("=" * 60)
        
        copy_sql = f"""
        COPY nowhstock_tbl_new ("item_code", "item_name", "shop_code") 
        FROM '{CSV_FILE}'
        CSV HEADER ENCODING 'WIN1252';
        """
        
        result = execute_sql(conn, copy_sql, "Upload CSV data")
        
        if result:
            # Get new row count
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nowhstock_tbl_new")
            after_count = cursor.fetchone()[0]
            cursor.close()
            log(f"📊 Records after upload: {after_count:,}")
            log(f"📊 New records uploaded: {after_count:,}")
        
        # Step 4: Analyze table
        print("\n" + "=" * 60)
        log("STEP 4: Analyzing table (updating statistics)")
        print("=" * 60)
        
        execute_sql(conn, "ANALYZE nowhstock_tbl_new;", "Analyze nowhstock_tbl_new table")
        
        # Step 5: Show final statistics
        print("\n" + "=" * 60)
        log("STEP 5: Final Statistics")
        print("=" * 60)
        
        cursor = conn.cursor()
        
        # Get total records
        cursor.execute('SELECT COUNT(*) FROM nowhstock_tbl_new')
        total_records = cursor.fetchone()[0]
        log(f"📊 Total records: {total_records:,}")
        
        # Get unique items
        cursor.execute('SELECT COUNT(DISTINCT item_code) FROM nowhstock_tbl_new')
        unique_items = cursor.fetchone()[0]
        log(f"🏷️  Unique items: {unique_items:,}")
        
        # Get unique shops
        cursor.execute('SELECT COUNT(DISTINCT shop_code) FROM nowhstock_tbl_new')
        unique_shops = cursor.fetchone()[0]
        log(f"🏪 Unique shops: {unique_shops}")
        
        # Get table size
        cursor.execute("""
            SELECT pg_size_pretty(pg_total_relation_size('nowhstock_tbl_new')) as size
        """)
        table_size = cursor.fetchone()[0]
        log(f"💾 Table size: {table_size}")
        
        # Get index sizes (if any)
        cursor.execute("""
            SELECT pg_size_pretty(pg_indexes_size('nowhstock_tbl_new')) as size
        """)
        index_size = cursor.fetchone()[0]
        log(f"🔍 Index size: {index_size}")
        
        # Show sample data (first 5 rows)
        cursor.execute("""
            SELECT item_code, item_name, shop_code 
            FROM nowhstock_tbl_new 
            LIMIT 5
        """)
        sample_data = cursor.fetchall()
        
        if sample_data:
            print("\n" + "-" * 60)
            log("Sample data (first 5 rows):")
            print("-" * 60)
            print(f"{'Item Code':<15} {'Item Name':<30} {'Shop Code':<10}")
            print("-" * 60)
            for row in sample_data:
                item_code = row[0] or ''
                item_name = (row[1] or '')[:30]  # Truncate long names
                shop_code = row[2] or ''
                print(f"{item_code:<15} {item_name:<30} {shop_code:<10}")
        
        cursor.close()
        
        # Success message
        print("\n" + "=" * 60)
        log("✅ NO_WH STOCK TABLE UPLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        log("Next steps:")
        log("  1. Verify data in nowhstock_tbl_new table")
        log("  2. Run daily_maintenance.sql if needed")
        log("  3. Restart Streamlit app if running")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        log(f"❌ CRITICAL ERROR: {str(e)}")
        print("=" * 60)
        log("Upload failed. Database may be in inconsistent state.")
        log("Please check the error message and retry.")
        log("")
        log("Recovery options:")
        log("  1. Check if backup table has data: SELECT COUNT(*) FROM nowhstock_tbl_new_backup;")
        log("  2. If needed, restore from backup: INSERT INTO nowhstock_tbl_new SELECT * FROM nowhstock_tbl_new_backup;")
        
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
