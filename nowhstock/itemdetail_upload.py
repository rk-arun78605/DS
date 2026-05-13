"""
============================================================
ITEM DETAILS TABLE UPLOAD SCRIPT
============================================================
Purpose: Upload item details to itemdetails table
Process:
  1. Backup existing data to itemdetails_backup
  2. Drop indexes for faster upload
  3. Truncate the main table
  4. Upload data from CSV
  5. Recreate indexes
  6. Analyze table

Database: salesdata (port 3307)
CSV Source: D:/Dashboard Code/tbl_data/ITEMdetails/item_details.csv
Table: itemdetails
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
CSV_FILE = r'D:/Dashboard Code/tbl_data/ITEMdetails/item_details.csv'

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
    log("ITEM DETAILS TABLE UPLOAD STARTING")
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
        log("STEP 1: Backing up existing data to itemdetails_backup")
        print("=" * 60)
        
        # Get current record count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM itemdetails")
        before_count = cursor.fetchone()[0]
        cursor.close()
        log(f"📊 Current records in itemdetails: {before_count:,}")
        
        if before_count > 0:
            backup_sql = """
            INSERT INTO itemdetails_backup
            SELECT * 
            FROM itemdetails;
            """
            
            result = execute_sql(conn, backup_sql, "Backup existing data")
            
            if result:
                # Get backup count
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM itemdetails_backup")
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
            'idx_itemdetails_pk',
            'idx_itemdetails_item_code',
            'idx_itemdetails_groups',
            'idx_itemdetails_supplier'
        ]
        
        for idx in indexes:
            execute_sql(conn, f"DROP INDEX IF EXISTS {idx};", f"Drop index {idx}")
        
        # Step 3: Truncate main table
        print("\n" + "=" * 60)
        log("STEP 3: Truncating itemdetails table")
        print("=" * 60)
        
        truncate_sql = "TRUNCATE TABLE itemdetails;"
        result = execute_sql(conn, truncate_sql, "Truncate table")
        
        if result:
            # Verify table is empty
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM itemdetails")
            after_truncate = cursor.fetchone()[0]
            cursor.close()
            log(f"📊 Records after truncate: {after_truncate}")
        
        # Step 4: Upload data from CSV
        print("\n" + "=" * 60)
        log("STEP 4: Uploading data from CSV (WIN1252 encoding)")
        print("=" * 60)
        
        copy_sql = f"""
        COPY itemdetails (vc_item_code, item_name, dept, groups, sub_group, type, vc_supplier_name, nu_qty_received) 
        FROM '{CSV_FILE}' 
        CSV HEADER ENCODING 'WIN1252';
        """
        
        result = execute_sql(conn, copy_sql, "Upload CSV data")
        
        if result:
            # Get new row count
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM itemdetails")
            after_count = cursor.fetchone()[0]
            cursor.close()
            log(f"📊 Records after upload: {after_count:,}")
            log(f"📊 New records uploaded: {after_count:,}")
        
        # Step 5: Recreate indexes
        print("\n" + "=" * 60)
        log("STEP 5: Recreating indexes (this may take 2-5 minutes)")
        print("=" * 60)
        
        index_definitions = [
            ('idx_itemdetails_item_code', 
             'CREATE INDEX idx_itemdetails_item_code ON itemdetails (vc_item_code)'),
            
            ('idx_itemdetails_groups',
             'CREATE INDEX idx_itemdetails_groups ON itemdetails (groups, sub_group)'),
            
            ('idx_itemdetails_supplier',
             'CREATE INDEX idx_itemdetails_supplier ON itemdetails (vc_supplier_name)')
        ]
        
        for idx_name, idx_sql in index_definitions:
            execute_sql(conn, idx_sql, f"Create index {idx_name}")
        
        # Step 6: Analyze table
        print("\n" + "=" * 60)
        log("STEP 6: Analyzing table (updating statistics)")
        print("=" * 60)
        
        execute_sql(conn, "ANALYZE itemdetails;", "Analyze itemdetails table")
        
        # Step 7: Show final statistics
        print("\n" + "=" * 60)
        log("STEP 7: Final Statistics")
        print("=" * 60)
        
        cursor = conn.cursor()
        
        # Get total records
        cursor.execute('SELECT COUNT(*) FROM itemdetails')
        total_records = cursor.fetchone()[0]
        log(f"📊 Total records: {total_records:,}")
        
        # Get unique items
        cursor.execute('SELECT COUNT(DISTINCT vc_item_code) FROM itemdetails')
        unique_items = cursor.fetchone()[0]
        log(f"🏷️  Unique items: {unique_items:,}")
        
        # Get unique departments
        cursor.execute('SELECT COUNT(DISTINCT dept) FROM itemdetails WHERE dept IS NOT NULL')
        unique_depts = cursor.fetchone()[0]
        log(f"🏢 Unique departments: {unique_depts}")
        
        # Get unique groups
        cursor.execute('SELECT COUNT(DISTINCT groups) FROM itemdetails WHERE groups IS NOT NULL')
        unique_groups = cursor.fetchone()[0]
        log(f"📁 Unique groups: {unique_groups}")
        
        # Get unique sub-groups
        cursor.execute('SELECT COUNT(DISTINCT sub_group) FROM itemdetails WHERE sub_group IS NOT NULL')
        unique_subgroups = cursor.fetchone()[0]
        log(f"📑 Unique sub-groups: {unique_subgroups}")
        
        # Get unique suppliers
        cursor.execute('SELECT COUNT(DISTINCT vc_supplier_name) FROM itemdetails WHERE vc_supplier_name IS NOT NULL')
        unique_suppliers = cursor.fetchone()[0]
        log(f"🏭 Unique suppliers: {unique_suppliers}")
        
        # Get total quantity received
        cursor.execute('SELECT SUM(nu_qty_received) FROM itemdetails WHERE nu_qty_received IS NOT NULL')
        total_qty = cursor.fetchone()[0] or 0
        log(f"📦 Total quantity received: {total_qty:,.0f}")
        
        # Get table size
        cursor.execute("""
            SELECT pg_size_pretty(pg_total_relation_size('itemdetails')) as size
        """)
        table_size = cursor.fetchone()[0]
        log(f"💾 Table size: {table_size}")
        
        # Get index sizes
        cursor.execute("""
            SELECT pg_size_pretty(pg_indexes_size('itemdetails')) as size
        """)
        index_size = cursor.fetchone()[0]
        log(f"🔍 Index size: {index_size}")
        
        # Show breakdown by department
        cursor.execute("""
            SELECT dept, COUNT(*) as item_count, SUM(nu_qty_received) as total_qty
            FROM itemdetails 
            WHERE dept IS NOT NULL
            GROUP BY dept
            ORDER BY COUNT(*) DESC
            LIMIT 5
        """)
        dept_data = cursor.fetchall()
        
        if dept_data:
            print("\n" + "-" * 60)
            log("Top 5 Departments by Item Count:")
            print("-" * 60)
            print(f"{'Department':<30} {'Items':>12} {'Qty Received':>15}")
            print("-" * 60)
            for row in dept_data:
                dept = (row[0] or 'NULL')[:30]
                item_count = row[1] or 0
                total_qty = row[2] or 0
                print(f"{dept:<30} {item_count:>12,} {total_qty:>15,.0f}")
        
        # Show breakdown by supplier (top 10)
        cursor.execute("""
            SELECT vc_supplier_name, COUNT(*) as item_count, SUM(nu_qty_received) as total_qty
            FROM itemdetails 
            WHERE vc_supplier_name IS NOT NULL
            GROUP BY vc_supplier_name
            ORDER BY COUNT(*) DESC
            LIMIT 10
        """)
        supplier_data = cursor.fetchall()
        
        if supplier_data:
            print("\n" + "-" * 60)
            log("Top 10 Suppliers by Item Count:")
            print("-" * 60)
            print(f"{'Supplier Name':<35} {'Items':>10} {'Qty Received':>12}")
            print("-" * 60)
            for row in supplier_data:
                supplier = (row[0] or 'NULL')[:35]
                item_count = row[1] or 0
                total_qty = row[2] or 0
                print(f"{supplier:<35} {item_count:>10,} {total_qty:>12,.0f}")
        
        # Show sample data (first 5 rows)
        cursor.execute("""
            SELECT vc_item_code, item_name, dept, groups, sub_group, type, vc_supplier_name, nu_qty_received
            FROM itemdetails 
            LIMIT 5
        """)
        sample_data = cursor.fetchall()
        
        if sample_data:
            print("\n" + "-" * 120)
            log("Sample data (first 5 rows):")
            print("-" * 120)
            print(f"{'Item Code':<12} {'Item Name':<25} {'Dept':<15} {'Group':<15} {'SubGroup':<15} {'Supplier':<20}")
            print("-" * 120)
            for row in sample_data:
                item_code = (row[0] or '')[:12]
                item_name = (row[1] or '')[:25]
                dept = (row[2] or '')[:15]
                groups = (row[3] or '')[:15]
                sub_group = (row[4] or '')[:15]
                supplier = (row[6] or '')[:20]
                print(f"{item_code:<12} {item_name:<25} {dept:<15} {groups:<15} {sub_group:<15} {supplier:<20}")
        
        cursor.close()
        
        # Success message
        print("\n" + "=" * 60)
        log("✅ ITEM DETAILS TABLE UPLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        log("Next steps:")
        log("  1. Verify item details data")
        log("  2. Run daily_maintenance.sql to refresh materialized views")
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
        log("  1. Check if backup table has data: SELECT COUNT(*) FROM itemdetails_backup;")
        log("  2. If needed, restore from backup: TRUNCATE itemdetails; INSERT INTO itemdetails SELECT * FROM itemdetails_backup;")
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
