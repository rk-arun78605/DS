"""
============================================================
SUP_SHOP_GRN TABLE UPLOAD SCRIPT
============================================================
Purpose: Upload shop GRN data to sup_shop_grn table
Process:
  1. Backup existing data to sup_shop_grn_backup
  2. Drop indexes for faster upload
  3. Upload data from CSV
  4. Recreate indexes
  5. Analyze table
  6. Verify 2025 data

Database: salesdata (port 3307)
CSV Source: D:/Dashboard Code/tbl_data/shopgrndate/sup_last_grn.csv
Table: sup_shop_grn
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
CSV_FILE = r'D:/Dashboard Code/tbl_data/shopgrndate/sup_last_grn.csv'

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
            result = cursor.fetchall()
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
    log("SUP_SHOP_GRN TABLE UPLOAD STARTING")
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
        log("STEP 1: Backing up existing data to sup_shop_grn_backup")
        print("=" * 60)
        
        # Get current record count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sup_shop_grn")
        before_count = cursor.fetchone()[0]
        cursor.close()
        log(f"📊 Current records in sup_shop_grn: {before_count:,}")
        
        if before_count > 0:
            backup_sql = """
            INSERT INTO sup_shop_grn_backup
            SELECT * 
            FROM sup_shop_grn;
            """
            
            result = execute_sql(conn, backup_sql, "Backup existing data")
            
            if result:
                # Get backup count
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sup_shop_grn_backup")
                backup_count = cursor.fetchone()[0]
                cursor.close()
                log(f"📊 Total backup records: {backup_count:,}")
        else:
            log("ℹ️  No data to backup (table is empty)")
        
        # Step 2: Drop indexes for faster upload
        print("\n" + "=" * 60)
        log("STEP 2: Dropping indexes for faster upload")
        print("=" * 60)
        
        indexes = [
            'idx_sup_shop_grn_item_shop',
            'idx_sup_shop_grn_item',
            'idx_sup_shop_grn_shop'
        ]
        
        for idx in indexes:
            execute_sql(conn, f"DROP INDEX IF EXISTS {idx};", f"Drop index {idx}")
        
        # Step 3: Upload data from CSV
        print("\n" + "=" * 60)
        log("STEP 3: Uploading data from CSV (WIN1252 encoding)")
        print("=" * 60)
        
        # Get current row count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sup_shop_grn")
        before_upload = cursor.fetchone()[0]
        cursor.close()
        log(f"📊 Records before upload: {before_upload:,}")
        
        copy_sql = f"""
        COPY sup_shop_grn ("item_code", "item_name", "shop_code", "shop_grn_date", "wh_grn_date", "shop_stock") 
        FROM '{CSV_FILE}' 
        CSV HEADER ENCODING 'WIN1252';
        """
        
        result = execute_sql(conn, copy_sql, "Upload CSV data")
        
        if result:
            # Get new row count
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sup_shop_grn")
            after_count = cursor.fetchone()[0]
            cursor.close()
            uploaded = after_count - before_upload
            log(f"📊 Records after upload: {after_count:,}")
            log(f"📊 New records uploaded: {uploaded:,}")
        
        # Step 4: Recreate indexes
        print("\n" + "=" * 60)
        log("STEP 4: Recreating indexes (this may take 3-5 minutes)")
        print("=" * 60)
        
        index_definitions = [
            ('idx_sup_shop_grn_item_shop', 
             'CREATE INDEX idx_sup_shop_grn_item_shop ON sup_shop_grn ("item_code", "shop_code")'),
            
            ('idx_sup_shop_grn_item',
             'CREATE INDEX idx_sup_shop_grn_item ON sup_shop_grn ("item_code")'),
            
            ('idx_sup_shop_grn_shop',
             'CREATE INDEX idx_sup_shop_grn_shop ON sup_shop_grn ("shop_code")')
        ]
        
        for idx_name, idx_sql in index_definitions:
            execute_sql(conn, idx_sql, f"Create index {idx_name}")
        
        # Step 5: Analyze table
        print("\n" + "=" * 60)
        log("STEP 5: Analyzing table (updating statistics)")
        print("=" * 60)
        
        execute_sql(conn, "ANALYZE sup_shop_grn;", "Analyze sup_shop_grn table")
        
        # Step 6: Verify 2025 data
        print("\n" + "=" * 60)
        log("STEP 6: Verifying 2025 data")
        print("=" * 60)
        
        verify_sql = """
        SELECT "shop_grn_date", SUM("shop_stock") AS total_stock 
        FROM sup_shop_grn 
        WHERE EXTRACT(YEAR FROM shop_grn_date) = 2025 
        GROUP BY "shop_grn_date"
        ORDER BY "shop_grn_date" DESC
        LIMIT 10;
        """
        
        cursor = conn.cursor()
        cursor.execute(verify_sql)
        grn_data = cursor.fetchall()
        
        if grn_data:
            log(f"📅 Found {len(grn_data)} unique GRN dates in 2025 (showing last 10):")
            print("\n" + "-" * 60)
            print(f"{'GRN Date':<20} {'Total Stock':>15}")
            print("-" * 60)
            for row in grn_data:
                grn_date = row[0]
                total_stock = row[1] or 0
                print(f"{str(grn_date):<20} {total_stock:>15,}")
        else:
            log("⚠️  No 2025 data found in sup_shop_grn table")
        
        cursor.close()
        
        # Step 7: Show final statistics
        print("\n" + "=" * 60)
        log("STEP 7: Final Statistics")
        print("=" * 60)
        
        cursor = conn.cursor()
        
        # Get total records
        cursor.execute('SELECT COUNT(*) FROM sup_shop_grn')
        total_records = cursor.fetchone()[0]
        log(f"📊 Total records: {total_records:,}")
        
        # Get unique items
        cursor.execute('SELECT COUNT(DISTINCT item_code) FROM sup_shop_grn')
        unique_items = cursor.fetchone()[0]
        log(f"🏷️  Unique items: {unique_items:,}")
        
        # Get unique shops
        cursor.execute('SELECT COUNT(DISTINCT shop_code) FROM sup_shop_grn')
        unique_shops = cursor.fetchone()[0]
        log(f"🏪 Unique shops: {unique_shops}")
        
        # Get date ranges
        cursor.execute('SELECT MIN(shop_grn_date)::date, MAX(shop_grn_date)::date FROM sup_shop_grn WHERE shop_grn_date IS NOT NULL')
        date_range = cursor.fetchone()
        if date_range and date_range[0]:
            log(f"📅 Shop GRN date range: {date_range[0]} to {date_range[1]}")
        
        cursor.execute('SELECT MIN(wh_grn_date)::date, MAX(wh_grn_date)::date FROM sup_shop_grn WHERE wh_grn_date IS NOT NULL')
        wh_date_range = cursor.fetchone()
        if wh_date_range and wh_date_range[0]:
            log(f"📅 WH GRN date range: {wh_date_range[0]} to {wh_date_range[1]}")
        
        # Get total stock
        cursor.execute('SELECT SUM(shop_stock) FROM sup_shop_grn WHERE shop_stock IS NOT NULL')
        total_stock = cursor.fetchone()[0] or 0
        log(f"📦 Total shop stock: {total_stock:,.0f}")
        
        # Get table size
        cursor.execute("""
            SELECT pg_size_pretty(pg_total_relation_size('sup_shop_grn')) as size
        """)
        table_size = cursor.fetchone()[0]
        log(f"💾 Table size: {table_size}")
        
        # Get index sizes
        cursor.execute("""
            SELECT pg_size_pretty(pg_indexes_size('sup_shop_grn')) as size
        """)
        index_size = cursor.fetchone()[0]
        log(f"🔍 Index size: {index_size}")
        
        # Show sample data (first 5 rows)
        cursor.execute("""
            SELECT item_code, item_name, shop_code, shop_grn_date, wh_grn_date, shop_stock
            FROM sup_shop_grn 
            ORDER BY shop_grn_date DESC NULLS LAST
            LIMIT 5
        """)
        sample_data = cursor.fetchall()
        
        if sample_data:
            print("\n" + "-" * 100)
            log("Sample data (5 most recent by shop_grn_date):")
            print("-" * 100)
            print(f"{'Item Code':<12} {'Item Name':<25} {'Shop':<6} {'Shop GRN':<12} {'WH GRN':<12} {'Stock':>8}")
            print("-" * 100)
            for row in sample_data:
                item_code = (row[0] or '')[:12]
                item_name = (row[1] or '')[:25]
                shop_code = (row[2] or '')[:6]
                shop_grn = str(row[3])[:10] if row[3] else 'NULL'
                wh_grn = str(row[4])[:10] if row[4] else 'NULL'
                stock = row[5] or 0
                print(f"{item_code:<12} {item_name:<25} {shop_code:<6} {shop_grn:<12} {wh_grn:<12} {stock:>8,.0f}")
        
        cursor.close()
        
        # Success message
        print("\n" + "=" * 60)
        log("✅ SUP_SHOP_GRN TABLE UPLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        log("Next steps:")
        log("  1. Run daily_maintenance.sql to refresh materialized views")
        log("  2. Verify GRN dates are current")
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
        log("  1. Check if backup table has data: SELECT COUNT(*) FROM sup_shop_grn_backup;")
        log("  2. If needed, restore from backup: TRUNCATE sup_shop_grn; INSERT INTO sup_shop_grn SELECT * FROM sup_shop_grn_backup;")
        log("  3. Recreate indexes manually if needed")
        
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
