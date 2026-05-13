"""
============================================================
SIT DATA TABLE UPLOAD SCRIPT
============================================================
Purpose: Upload SIT (Stock In Transit) data to sit_data table
Process:
  1. Backup existing data to sit_data_backup
  2. Upload data from CSV (APPENDS to existing data)
  3. Analyze table

Database: salesdata (port 3307)
CSV Source: D:/Dashboard Code/tbl_data/SIT/sit_sup.csv
Table: sit_data

NOTE: This script APPENDS data (does NOT truncate table)
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
CSV_FILE = r'D:/Dashboard Code/tbl_data/SIT/sit_sup.csv'

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
    log("SIT DATA TABLE UPLOAD STARTING")
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
        log("STEP 1: Backing up existing data to sit_data_backup")
        print("=" * 60)
        
        # Get current record count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sit_data")
        before_count = cursor.fetchone()[0]
        cursor.close()
        log(f"📊 Current records in sit_data: {before_count:,}")
        
        if before_count > 0:
            backup_sql = """
            INSERT INTO sit_data_backup 
            SELECT * FROM sit_data;
            """
            
            result = execute_sql(conn, backup_sql, "Backup existing data")
            
            if result:
                # Get backup count
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sit_data_backup")
                backup_count = cursor.fetchone()[0]
                cursor.close()
                log(f"📊 Total backup records: {backup_count:,}")
        else:
            log("ℹ️  No data to backup (table is empty)")
        
        # Step 2: Upload data from CSV (APPEND mode - no truncate)
        print("\n" + "=" * 60)
        log("STEP 2: Uploading data from CSV (APPEND mode)")
        print("=" * 60)
        log("⚠️  NOTE: This script APPENDS data without truncating table")
        
        # Get current row count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sit_data")
        before_upload = cursor.fetchone()[0]
        cursor.close()
        log(f"📊 Records before upload: {before_upload:,}")
        
        copy_sql = f"""
        COPY sit_data ("shop_code", "item_code", "dt_trans_date", "nu_transit_qty")
        FROM '{CSV_FILE}' 
        CSV HEADER;
        """
        
        result = execute_sql(conn, copy_sql, "Upload CSV data")
        
        if result:
            # Get new row count
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sit_data")
            after_count = cursor.fetchone()[0]
            cursor.close()
            uploaded = after_count - before_upload
            log(f"📊 Records after upload: {after_count:,}")
            log(f"📊 New records uploaded: {uploaded:,}")
        
        # Step 3: Analyze table
        print("\n" + "=" * 60)
        log("STEP 3: Analyzing table (updating statistics)")
        print("=" * 60)
        
        execute_sql(conn, "ANALYZE sit_data;", "Analyze sit_data table")
        
        # Step 4: Show final statistics
        print("\n" + "=" * 60)
        log("STEP 4: Final Statistics")
        print("=" * 60)
        
        cursor = conn.cursor()
        
        # Get total records
        cursor.execute('SELECT COUNT(*) FROM sit_data')
        total_records = cursor.fetchone()[0]
        log(f"📊 Total records: {total_records:,}")
        
        # Get unique items
        cursor.execute('SELECT COUNT(DISTINCT item_code) FROM sit_data')
        unique_items = cursor.fetchone()[0]
        log(f"🏷️  Unique items in transit: {unique_items:,}")
        
        # Get unique shops
        cursor.execute('SELECT COUNT(DISTINCT shop_code) FROM sit_data')
        unique_shops = cursor.fetchone()[0]
        log(f"🏪 Unique shops: {unique_shops}")
        
        # Get transaction date range
        cursor.execute('SELECT MIN(dt_trans_date)::date, MAX(dt_trans_date)::date FROM sit_data WHERE dt_trans_date IS NOT NULL')
        date_range = cursor.fetchone()
        if date_range and date_range[0]:
            log(f"📅 Transaction date range: {date_range[0]} to {date_range[1]}")
            
            # Calculate days since latest transaction
            cursor.execute('SELECT CURRENT_DATE - MAX(dt_trans_date)::date FROM sit_data WHERE dt_trans_date IS NOT NULL')
            days_old = cursor.fetchone()[0]
            if days_old is not None:
                log(f"⏱️  Days since latest transaction: {days_old} days")
        
        # Get total transit quantity
        cursor.execute('SELECT SUM(nu_transit_qty) FROM sit_data WHERE nu_transit_qty IS NOT NULL')
        total_transit = cursor.fetchone()[0] or 0
        log(f"📦 Total transit quantity: {total_transit:,.0f}")
        
        # Get table size
        cursor.execute("""
            SELECT pg_size_pretty(pg_total_relation_size('sit_data')) as size
        """)
        table_size = cursor.fetchone()[0]
        log(f"💾 Table size: {table_size}")
        
        # Get index sizes (if any)
        cursor.execute("""
            SELECT pg_size_pretty(pg_indexes_size('sit_data')) as size
        """)
        index_size = cursor.fetchone()[0]
        log(f"🔍 Index size: {index_size}")
        
        # Show breakdown by shop (top 10 by transit quantity)
        cursor.execute("""
            SELECT shop_code, COUNT(*) as item_count, SUM(nu_transit_qty) as total_transit
            FROM sit_data 
            WHERE shop_code IS NOT NULL
            GROUP BY shop_code
            ORDER BY SUM(nu_transit_qty) DESC
            LIMIT 10
        """)
        shop_data = cursor.fetchall()
        
        if shop_data:
            print("\n" + "-" * 70)
            log("Top 10 Shops by Transit Quantity:")
            print("-" * 70)
            print(f"{'Shop Code':<15} {'Items in Transit':>20} {'Total Qty':>20}")
            print("-" * 70)
            for row in shop_data:
                shop = (row[0] or 'NULL')[:15]
                item_count = row[1] or 0
                total_transit = row[2] or 0
                print(f"{shop:<15} {item_count:>20,} {total_transit:>20,.0f}")
        
        # Show most recent transactions (last 10 dates)
        cursor.execute("""
            SELECT dt_trans_date::date, COUNT(*) as items, SUM(nu_transit_qty) as total_qty
            FROM sit_data 
            WHERE dt_trans_date IS NOT NULL
            GROUP BY dt_trans_date::date
            ORDER BY dt_trans_date::date DESC
            LIMIT 10
        """)
        recent_trans = cursor.fetchall()
        
        if recent_trans:
            print("\n" + "-" * 70)
            log("Most Recent Transactions (Last 10 Dates):")
            print("-" * 70)
            print(f"{'Transaction Date':<20} {'Items':>20} {'Total Qty':>20}")
            print("-" * 70)
            for row in recent_trans:
                trans_date = str(row[0]) if row[0] else 'NULL'
                items = row[1] or 0
                total_qty = row[2] or 0
                print(f"{trans_date:<20} {items:>20,} {total_qty:>20,.0f}")
        
        # Show items with highest transit quantities
        cursor.execute("""
            SELECT item_code, shop_code, dt_trans_date, nu_transit_qty
            FROM sit_data 
            ORDER BY nu_transit_qty DESC NULLS LAST
            LIMIT 10
        """)
        top_transit = cursor.fetchall()
        
        if top_transit:
            print("\n" + "-" * 90)
            log("Top 10 Items by Transit Quantity:")
            print("-" * 90)
            print(f"{'Item Code':<15} {'Shop Code':<12} {'Trans Date':<15} {'Transit Qty':>20}")
            print("-" * 90)
            for row in top_transit:
                item_code = (row[0] or '')[:15]
                shop_code = (row[1] or '')[:12]
                trans_date = str(row[2])[:10] if row[2] else 'NULL'
                qty = row[3] or 0
                print(f"{item_code:<15} {shop_code:<12} {trans_date:<15} {qty:>20,.0f}")
        
        # Show sample data (5 most recent by date)
        cursor.execute("""
            SELECT shop_code, item_code, dt_trans_date, nu_transit_qty
            FROM sit_data 
            ORDER BY dt_trans_date DESC NULLS LAST
            LIMIT 5
        """)
        sample_data = cursor.fetchall()
        
        if sample_data:
            print("\n" + "-" * 90)
            log("Sample data (5 most recent by transaction date):")
            print("-" * 90)
            print(f"{'Shop Code':<12} {'Item Code':<15} {'Transaction Date':<20} {'Transit Qty':>20}")
            print("-" * 90)
            for row in sample_data:
                shop_code = (row[0] or '')[:12]
                item_code = (row[1] or '')[:15]
                trans_date = str(row[2])[:19] if row[2] else 'NULL'
                qty = row[3] or 0
                print(f"{shop_code:<12} {item_code:<15} {trans_date:<20} {qty:>20,.0f}")
        
        # Check for duplicate entries (same shop + item + date)
        cursor.execute("""
            SELECT shop_code, item_code, dt_trans_date::date, COUNT(*) as duplicate_count
            FROM sit_data
            GROUP BY shop_code, item_code, dt_trans_date::date
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC
            LIMIT 5
        """)
        duplicates = cursor.fetchall()
        
        if duplicates:
            print("\n" + "-" * 90)
            log("⚠️  WARNING: Duplicate entries detected (same shop + item + date):")
            print("-" * 90)
            print(f"{'Shop Code':<12} {'Item Code':<15} {'Trans Date':<15} {'Duplicates':>15}")
            print("-" * 90)
            for row in duplicates:
                shop_code = row[0] or 'NULL'
                item_code = row[1] or 'NULL'
                trans_date = str(row[2]) if row[2] else 'NULL'
                dup_count = row[3] or 0
                print(f"{shop_code:<12} {item_code:<15} {trans_date:<15} {dup_count:>15}")
            print("-" * 90)
            log("Consider cleaning up duplicate records to avoid data inconsistency")
        
        cursor.close()
        
        # Success message
        print("\n" + "=" * 60)
        log("✅ SIT DATA TABLE UPLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        log("Next steps:")
        log("  1. Review transaction dates to ensure data is current")
        log("  2. Check for duplicate entries if warned above")
        log("  3. Run daily_maintenance.sql to refresh materialized views")
        log("  4. Restart Streamlit app if running")
        print("=" * 60)
        
        # Warning if data is stale
        if date_range and date_range[1]:
            cursor = conn.cursor()
            cursor.execute('SELECT CURRENT_DATE - MAX(dt_trans_date)::date FROM sit_data WHERE dt_trans_date IS NOT NULL')
            days_old = cursor.fetchone()[0]
            cursor.close()
            
            if days_old and days_old > 3:
                print("\n" + "⚠️ " * 20)
                log(f"WARNING: Latest transaction date is {days_old} days old!")
                log("SIT data should be updated frequently (ideally daily)")
                log("Consider updating sit_sup.csv with more recent data.")
                print("⚠️ " * 20)
        
    except Exception as e:
        print("\n" + "=" * 60)
        log(f"❌ CRITICAL ERROR: {str(e)}")
        print("=" * 60)
        log("Upload failed. Database may be in inconsistent state.")
        log("Please check the error message and retry.")
        log("")
        log("Recovery options:")
        log("  1. Check if backup table has data: SELECT COUNT(*) FROM sit_data_backup;")
        log("  2. If needed, restore from backup: TRUNCATE sit_data; INSERT INTO sit_data SELECT * FROM sit_data_backup;")
        log("  3. If duplicate issue, consider truncating before upload in future")
        
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
