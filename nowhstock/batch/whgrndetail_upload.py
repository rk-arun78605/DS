"""
============================================================
WH GRN DETAILS TABLE UPLOAD SCRIPT
============================================================
Purpose: Upload warehouse GRN details to whgrndetails table
Process:
  1. Backup existing data to whgrndetails_backup
  2. Truncate the main table
  3. Upload data from CSV
  4. Analyze table

Database: salesdata (port 3307)
CSV Source: D:/Dashboard Code/tbl_data/whgrndetails/whgrndate.csv
Table: whgrndetails
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
CSV_FILE = r'D:/Dashboard Code/tbl_data/whgrndetails/whgrndate.csv'

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
    log("WH GRN DETAILS TABLE UPLOAD STARTING")
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
        log("STEP 1: Backing up existing data to whgrndetails_backup")
        print("=" * 60)
        
        # Get current record count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM whgrndetails")
        before_count = cursor.fetchone()[0]
        cursor.close()
        log(f"📊 Current records in whgrndetails: {before_count:,}")
        
        if before_count > 0:
            backup_sql = """
            INSERT INTO whgrndetails_backup 
            SELECT * FROM whgrndetails;
            """
            
            result = execute_sql(conn, backup_sql, "Backup existing data")
            
            if result:
                # Get backup count
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM whgrndetails_backup")
                backup_count = cursor.fetchone()[0]
                cursor.close()
                log(f"📊 Total backup records: {backup_count:,}")
        else:
            log("ℹ️  No data to backup (table is empty)")
        
        # Step 2: Truncate main table
        print("\n" + "=" * 60)
        log("STEP 2: Truncating whgrndetails table")
        print("=" * 60)
        
        truncate_sql = "TRUNCATE TABLE whgrndetails;"
        result = execute_sql(conn, truncate_sql, "Truncate table")
        
        if result:
            # Verify table is empty
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM whgrndetails")
            after_truncate = cursor.fetchone()[0]
            cursor.close()
            log(f"📊 Records after truncate: {after_truncate}")
        
        # Step 3: Upload data from CSV
        print("\n" + "=" * 60)
        log("STEP 3: Uploading data from CSV (WIN1252 encoding)")
        print("=" * 60)
        
        copy_sql = f"""
        COPY whgrndetails ("ITEM_CODE", "TYPE", "SUPPLIER_NAME", "WH_LAST_GRN_DATE", "WH_QTY_RECEIVED")
        FROM '{CSV_FILE}' 
        CSV HEADER
        ENCODING 'WIN1252';
        """
        
        result = execute_sql(conn, copy_sql, "Upload CSV data")
        
        if result:
            # Get new row count
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM whgrndetails")
            after_count = cursor.fetchone()[0]
            cursor.close()
            log(f"📊 Records after upload: {after_count:,}")
            log(f"📊 New records uploaded: {after_count:,}")
        
        # Step 4: Analyze table
        print("\n" + "=" * 60)
        log("STEP 4: Analyzing table (updating statistics)")
        print("=" * 60)
        
        execute_sql(conn, "ANALYZE whgrndetails;", "Analyze whgrndetails table")
        
        # Step 5: Show final statistics
        print("\n" + "=" * 60)
        log("STEP 5: Final Statistics")
        print("=" * 60)
        
        cursor = conn.cursor()
        
        # Get total records
        cursor.execute('SELECT COUNT(*) FROM whgrndetails')
        total_records = cursor.fetchone()[0]
        log(f"📊 Total records: {total_records:,}")
        
        # Get unique items
        cursor.execute('SELECT COUNT(DISTINCT "ITEM_CODE") FROM whgrndetails')
        unique_items = cursor.fetchone()[0]
        log(f"🏷️  Unique items: {unique_items:,}")
        
        # Get unique suppliers
        cursor.execute('SELECT COUNT(DISTINCT "SUPPLIER_NAME") FROM whgrndetails WHERE "SUPPLIER_NAME" IS NOT NULL')
        unique_suppliers = cursor.fetchone()[0]
        log(f"🏭 Unique suppliers: {unique_suppliers}")
        
        # Get unique item types
        cursor.execute('SELECT COUNT(DISTINCT "TYPE") FROM whgrndetails WHERE "TYPE" IS NOT NULL')
        unique_types = cursor.fetchone()[0]
        log(f"📦 Unique item types: {unique_types}")
        
        # Get GRN date range
        cursor.execute('SELECT MIN("WH_LAST_GRN_DATE")::date, MAX("WH_LAST_GRN_DATE")::date FROM whgrndetails WHERE "WH_LAST_GRN_DATE" IS NOT NULL')
        date_range = cursor.fetchone()
        if date_range and date_range[0]:
            log(f"📅 WH GRN date range: {date_range[0]} to {date_range[1]}")
            
            # Calculate days since latest GRN
            cursor.execute('SELECT CURRENT_DATE - MAX("WH_LAST_GRN_DATE")::date FROM whgrndetails WHERE "WH_LAST_GRN_DATE" IS NOT NULL')
            days_old = cursor.fetchone()[0]
            if days_old is not None:
                log(f"⏱️  Days since latest GRN: {days_old} days")
        
        # Get total quantity received
        cursor.execute('SELECT SUM("WH_QTY_RECEIVED") FROM whgrndetails WHERE "WH_QTY_RECEIVED" IS NOT NULL')
        total_qty = cursor.fetchone()[0] or 0
        log(f"📦 Total WH quantity received: {total_qty:,.0f}")
        
        # Get table size
        cursor.execute("""
            SELECT pg_size_pretty(pg_total_relation_size('whgrndetails')) as size
        """)
        table_size = cursor.fetchone()[0]
        log(f"💾 Table size: {table_size}")
        
        # Get index sizes (if any)
        cursor.execute("""
            SELECT pg_size_pretty(pg_indexes_size('whgrndetails')) as size
        """)
        index_size = cursor.fetchone()[0]
        log(f"🔍 Index size: {index_size}")
        
        # Show breakdown by item type
        cursor.execute("""
            SELECT "TYPE", COUNT(*) as item_count, SUM("WH_QTY_RECEIVED") as total_qty
            FROM whgrndetails 
            WHERE "TYPE" IS NOT NULL
            GROUP BY "TYPE"
            ORDER BY COUNT(*) DESC
            LIMIT 10
        """)
        type_data = cursor.fetchall()
        
        if type_data:
            print("\n" + "-" * 60)
            log("Breakdown by Item Type (Top 10):")
            print("-" * 60)
            print(f"{'Item Type':<25} {'Items':>12} {'Qty Received':>20}")
            print("-" * 60)
            for row in type_data:
                item_type = (row[0] or 'NULL')[:25]
                item_count = row[1] or 0
                total_qty = row[2] or 0
                print(f"{item_type:<25} {item_count:>12,} {total_qty:>20,.0f}")
        
        # Show top 10 suppliers by items received
        cursor.execute("""
            SELECT "SUPPLIER_NAME", COUNT(*) as item_count, SUM("WH_QTY_RECEIVED") as total_qty
            FROM whgrndetails 
            WHERE "SUPPLIER_NAME" IS NOT NULL
            GROUP BY "SUPPLIER_NAME"
            ORDER BY COUNT(*) DESC
            LIMIT 10
        """)
        supplier_data = cursor.fetchall()
        
        if supplier_data:
            print("\n" + "-" * 80)
            log("Top 10 Suppliers by Item Count:")
            print("-" * 80)
            print(f"{'Supplier Name':<40} {'Items':>15} {'Qty Received':>20}")
            print("-" * 80)
            for row in supplier_data:
                supplier = (row[0] or 'NULL')[:40]
                item_count = row[1] or 0
                total_qty = row[2] or 0
                print(f"{supplier:<40} {item_count:>15,} {total_qty:>20,.0f}")
        
        # Show most recent GRN dates (last 10)
        cursor.execute("""
            SELECT "WH_LAST_GRN_DATE"::date, COUNT(*) as items_received, SUM("WH_QTY_RECEIVED") as total_qty
            FROM whgrndetails 
            WHERE "WH_LAST_GRN_DATE" IS NOT NULL
            GROUP BY "WH_LAST_GRN_DATE"::date
            ORDER BY "WH_LAST_GRN_DATE"::date DESC
            LIMIT 10
        """)
        recent_grn = cursor.fetchall()
        
        if recent_grn:
            print("\n" + "-" * 70)
            log("Most Recent GRN Dates (Last 10):")
            print("-" * 70)
            print(f"{'GRN Date':<15} {'Items Received':>20} {'Total Qty':>20}")
            print("-" * 70)
            for row in recent_grn:
                grn_date = str(row[0]) if row[0] else 'NULL'
                items = row[1] or 0
                total_qty = row[2] or 0
                print(f"{grn_date:<15} {items:>20,} {total_qty:>20,.0f}")
        
        # Show sample data (5 most recent items)
        cursor.execute("""
            SELECT "ITEM_CODE", "TYPE", "SUPPLIER_NAME", "WH_LAST_GRN_DATE", "WH_QTY_RECEIVED"
            FROM whgrndetails 
            ORDER BY "WH_LAST_GRN_DATE" DESC NULLS LAST
            LIMIT 5
        """)
        sample_data = cursor.fetchall()
        
        if sample_data:
            print("\n" + "-" * 110)
            log("Sample data (5 most recent by WH_LAST_GRN_DATE):")
            print("-" * 110)
            print(f"{'Item Code':<15} {'Type':<20} {'Supplier':<30} {'WH GRN Date':<15} {'Qty':>10}")
            print("-" * 110)
            for row in sample_data:
                item_code = (row[0] or '')[:15]
                item_type = (row[1] or '')[:20]
                supplier = (row[2] or '')[:30]
                grn_date = str(row[3])[:10] if row[3] else 'NULL'
                qty = row[4] or 0
                print(f"{item_code:<15} {item_type:<20} {supplier:<30} {grn_date:<15} {qty:>10,.0f}")
        
        cursor.close()
        
        # Success message
        print("\n" + "=" * 60)
        log("✅ WH GRN DETAILS TABLE UPLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        log("Next steps:")
        log("  1. Verify WH GRN dates are current")
        log("  2. Check supplier data completeness")
        log("  3. Run daily_maintenance.sql to refresh materialized views")
        log("  4. Restart Streamlit app if running")
        print("=" * 60)
        
        # Warning if data is stale
        if date_range and date_range[1]:
            cursor = conn.cursor()
            cursor.execute('SELECT CURRENT_DATE - MAX("WH_LAST_GRN_DATE")::date FROM whgrndetails WHERE "WH_LAST_GRN_DATE" IS NOT NULL')
            days_old = cursor.fetchone()[0]
            cursor.close()
            
            if days_old and days_old > 7:
                print("\n" + "⚠️ " * 20)
                log(f"WARNING: Latest WH GRN date is {days_old} days old!")
                log("Consider updating whgrndate.csv with more recent data.")
                print("⚠️ " * 20)
        
    except Exception as e:
        print("\n" + "=" * 60)
        log(f"❌ CRITICAL ERROR: {str(e)}")
        print("=" * 60)
        log("Upload failed. Database may be in inconsistent state.")
        log("Please check the error message and retry.")
        log("")
        log("Recovery options:")
        log("  1. Check if backup table has data: SELECT COUNT(*) FROM whgrndetails_backup;")
        log("  2. If needed, restore from backup: TRUNCATE whgrndetails; INSERT INTO whgrndetails SELECT * FROM whgrndetails_backup;")
        
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
