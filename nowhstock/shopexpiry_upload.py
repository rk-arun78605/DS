"""
============================================================
SHOP EXPIRY TABLE UPLOAD SCRIPT
============================================================
Purpose: Upload shop expiry data to shopexpiry table
Process:
  1. Backup existing data to shopexpiry_backup
  2. Truncate the main table
  3. Upload data from CSV
  4. Analyze table

Database: salesdata (port 3307)
CSV Source: D:/Dashboard Code/tbl_data/shopexpiry/shopexpiry.csv
Table: shopexpiry
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
CSV_FILE = r'D:/Dashboard Code/tbl_data/shopexpiry/shopexpiry.csv'

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
    log("SHOP EXPIRY TABLE UPLOAD STARTING")
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
        log("STEP 1: Backing up existing data to shopexpiry_backup")
        print("=" * 60)
        
        # Get current record count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM shopexpiry")
        before_count = cursor.fetchone()[0]
        cursor.close()
        log(f"📊 Current records in shopexpiry: {before_count:,}")
        
        if before_count > 0:
            backup_sql = """
            INSERT INTO shopexpiry_backup 
            SELECT * FROM shopexpiry;
            """
            
            result = execute_sql(conn, backup_sql, "Backup existing data")
            
            if result:
                # Get backup count
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM shopexpiry_backup")
                backup_count = cursor.fetchone()[0]
                cursor.close()
                log(f"📊 Total backup records: {backup_count:,}")
        else:
            log("ℹ️  No data to backup (table is empty)")
        
        # Step 2: Truncate main table
        print("\n" + "=" * 60)
        log("STEP 2: Truncating shopexpiry table")
        print("=" * 60)
        
        truncate_sql = "TRUNCATE TABLE shopexpiry;"
        result = execute_sql(conn, truncate_sql, "Truncate table")
        
        if result:
            # Verify table is empty
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM shopexpiry")
            after_truncate = cursor.fetchone()[0]
            cursor.close()
            log(f"📊 Records after truncate: {after_truncate}")
        
        # Step 3: Upload data from CSV
        print("\n" + "=" * 60)
        log("STEP 3: Uploading data from CSV")
        print("=" * 60)
        
        copy_sql = f"""
        COPY shopexpiry ("ITEM_CODE", "SHOP_EXPIRY_DATE", "SHOP_CODE") 
        FROM '{CSV_FILE}' 
        CSV HEADER;
        """
        
        result = execute_sql(conn, copy_sql, "Upload CSV data")
        
        if result:
            # Get new row count
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM shopexpiry")
            after_count = cursor.fetchone()[0]
            cursor.close()
            log(f"📊 Records after upload: {after_count:,}")
            log(f"📊 New records uploaded: {after_count:,}")
        
        # Step 4: Analyze table
        print("\n" + "=" * 60)
        log("STEP 4: Analyzing table (updating statistics)")
        print("=" * 60)
        
        execute_sql(conn, "ANALYZE shopexpiry;", "Analyze shopexpiry table")
        
        # Step 5: Show final statistics
        print("\n" + "=" * 60)
        log("STEP 5: Final Statistics")
        print("=" * 60)
        
        cursor = conn.cursor()
        
        # Get total records
        cursor.execute('SELECT COUNT(*) FROM shopexpiry')
        total_records = cursor.fetchone()[0]
        log(f"📊 Total records: {total_records:,}")
        
        # Get unique items
        cursor.execute('SELECT COUNT(DISTINCT "ITEM_CODE") FROM shopexpiry')
        unique_items = cursor.fetchone()[0]
        log(f"🏷️  Unique items: {unique_items:,}")
        
        # Get unique shops
        cursor.execute('SELECT COUNT(DISTINCT "SHOP_CODE") FROM shopexpiry')
        unique_shops = cursor.fetchone()[0]
        log(f"🏪 Unique shops: {unique_shops}")
        
        # Get expiry date range
        cursor.execute('SELECT MIN("SHOP_EXPIRY_DATE")::date, MAX("SHOP_EXPIRY_DATE")::date FROM shopexpiry WHERE "SHOP_EXPIRY_DATE" IS NOT NULL')
        date_range = cursor.fetchone()
        if date_range and date_range[0]:
            log(f"📅 Expiry date range: {date_range[0]} to {date_range[1]}")
        
        # Get expiry status breakdown
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN "SHOP_EXPIRY_DATE" < CURRENT_DATE THEN 1 END) as already_expired,
                COUNT(CASE WHEN "SHOP_EXPIRY_DATE" BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '30 days' THEN 1 END) as expiring_within_30d,
                COUNT(CASE WHEN "SHOP_EXPIRY_DATE" > CURRENT_DATE + INTERVAL '30 days' THEN 1 END) as safe_stock,
                COUNT(CASE WHEN "SHOP_EXPIRY_DATE" IS NULL THEN 1 END) as no_expiry_date
            FROM shopexpiry
        """)
        expiry_status = cursor.fetchone()
        
        if expiry_status:
            print("\n" + "-" * 60)
            log("Expiry Status Breakdown:")
            print("-" * 60)
            log(f"  ❌ Already expired: {expiry_status[0]:,} items")
            log(f"  ⚠️  Expiring within 30 days: {expiry_status[1]:,} items")
            log(f"  ✅ Safe stock (>30 days): {expiry_status[2]:,} items")
            log(f"  ℹ️  No expiry date: {expiry_status[3]:,} items")
        
        # Get table size
        cursor.execute("""
            SELECT pg_size_pretty(pg_total_relation_size('shopexpiry')) as size
        """)
        table_size = cursor.fetchone()[0]
        log(f"💾 Table size: {table_size}")
        
        # Get index sizes (if any)
        cursor.execute("""
            SELECT pg_size_pretty(pg_indexes_size('shopexpiry')) as size
        """)
        index_size = cursor.fetchone()[0]
        log(f"🔍 Index size: {index_size}")
        
        # Show items expiring soon (next 30 days) by shop
        cursor.execute("""
            SELECT "SHOP_CODE", COUNT(*) as expiring_items
            FROM shopexpiry
            WHERE "SHOP_EXPIRY_DATE" BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '30 days'
            GROUP BY "SHOP_CODE"
            ORDER BY COUNT(*) DESC
            LIMIT 10
        """)
        expiring_by_shop = cursor.fetchall()
        
        if expiring_by_shop:
            print("\n" + "-" * 60)
            log("Top 10 Shops with Items Expiring Soon (Next 30 Days):")
            print("-" * 60)
            print(f"{'Shop Code':<15} {'Expiring Items':>20}")
            print("-" * 60)
            for row in expiring_by_shop:
                shop = row[0] or 'NULL'
                count = row[1] or 0
                print(f"{shop:<15} {count:>20,}")
        
        # Show sample data (5 items expiring soonest)
        cursor.execute("""
            SELECT "ITEM_CODE", "SHOP_EXPIRY_DATE", "SHOP_CODE",
                   CURRENT_DATE - "SHOP_EXPIRY_DATE"::date as days_until_expiry
            FROM shopexpiry 
            WHERE "SHOP_EXPIRY_DATE" >= CURRENT_DATE
            ORDER BY "SHOP_EXPIRY_DATE" ASC
            LIMIT 5
        """)
        sample_data = cursor.fetchall()
        
        if sample_data:
            print("\n" + "-" * 80)
            log("Sample data (5 items expiring soonest):")
            print("-" * 80)
            print(f"{'Item Code':<15} {'Expiry Date':<15} {'Shop Code':<12} {'Days Until Expiry':>20}")
            print("-" * 80)
            for row in sample_data:
                item_code = (row[0] or '')[:15]
                expiry_date = str(row[1])[:10] if row[1] else 'NULL'
                shop_code = (row[2] or '')[:12]
                days_until = abs(row[3]) if row[3] else 0
                print(f"{item_code:<15} {expiry_date:<15} {shop_code:<12} {days_until:>20}")
        
        # Show already expired items by shop
        cursor.execute("""
            SELECT "SHOP_CODE", COUNT(*) as expired_items
            FROM shopexpiry
            WHERE "SHOP_EXPIRY_DATE" < CURRENT_DATE
            GROUP BY "SHOP_CODE"
            ORDER BY COUNT(*) DESC
            LIMIT 10
        """)
        expired_by_shop = cursor.fetchall()
        
        if expired_by_shop:
            print("\n" + "-" * 60)
            log("⚠️  Top 10 Shops with Already Expired Items:")
            print("-" * 60)
            print(f"{'Shop Code':<15} {'Expired Items':>20}")
            print("-" * 60)
            for row in expired_by_shop:
                shop = row[0] or 'NULL'
                count = row[1] or 0
                print(f"{shop:<15} {count:>20,}")
        
        cursor.close()
        
        # Success message
        print("\n" + "=" * 60)
        log("✅ SHOP EXPIRY TABLE UPLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        log("Next steps:")
        log("  1. Review expiry status breakdown above")
        log("  2. Check shops with expiring items")
        log("  3. Run daily_maintenance.sql to refresh materialized views")
        log("  4. Restart Streamlit app if running")
        print("=" * 60)
        
        # Warnings for expired items
        if expiry_status and expiry_status[0] > 0:
            print("\n" + "⚠️ " * 20)
            log(f"WARNING: {expiry_status[0]:,} items have ALREADY EXPIRED!")
            log("Please review expired items by shop (listed above)")
            print("⚠️ " * 20)
        
    except Exception as e:
        print("\n" + "=" * 60)
        log(f"❌ CRITICAL ERROR: {str(e)}")
        print("=" * 60)
        log("Upload failed. Database may be in inconsistent state.")
        log("Please check the error message and retry.")
        log("")
        log("Recovery options:")
        log("  1. Check if backup table has data: SELECT COUNT(*) FROM shopexpiry_backup;")
        log("  2. If needed, restore from backup: TRUNCATE shopexpiry; INSERT INTO shopexpiry SELECT * FROM shopexpiry_backup;")
        
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
