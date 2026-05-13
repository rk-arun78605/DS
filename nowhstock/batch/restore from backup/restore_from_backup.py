#!/usr/bin/env python3
"""
Restore Tables from Backup - Temporary Utility Script
=====================================================
This script restores PostgreSQL tables from their backup copies.

BACKUP → PRODUCTION RESTORE:
- sales_2025_backup → sales_2025
- nowhstock_tbl_new_backup → nowhstock_tbl_new
- sup_shop_grn_backup → sup_shop_grn
- itemdetails_backup → itemdetails
- shopexpiry_backup → shopexpiry
- whgrndetails_backup → whgrndetails
- sit_data_backup → sit_data

Usage:
    python restore_from_backup.py

WARNING: This will TRUNCATE production tables and restore from backups!
"""

import psycopg2
import sys
from datetime import datetime
import time

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'database': 'salesdata',
    'user': 'postgres',
    'password': 'hello'
}

# Tables to restore (backup_table → production_table)
RESTORE_CONFIG = [

    {
        'backup_table': 'nowhstock_tbl_new_backup',
        'production_table': 'nowhstock_tbl_new',
        'description': 'NO_WH stock data'
    },
    {
        'backup_table': 'sup_shop_grn_backup',
        'production_table': 'sup_shop_grn',
        'description': 'Supplier/Shop GRN dates'
    },
    {
        'backup_table': 'itemdetails_backup',
        'production_table': 'itemdetails',
        'description': 'Item details and metadata'
    },
    {
        'backup_table': 'shopexpiry_backup',
        'production_table': 'shopexpiry',
        'description': 'Shop expiry dates'
    },
    {
        'backup_table': 'whgrndetails_backup',
        'production_table': 'whgrndetails',
        'description': 'Warehouse GRN details'
    },
    {
        'backup_table': 'sit_data_backup',
        'production_table': 'sit_data',
        'description': 'Stock in transit data'
    },
    {
        'backup_table': 'sales_2025_backup',
        'production_table': 'sales_2025',
        'description': 'Sales transactions 2025'
    }
]

def print_header(text, char='='):
    """Print formatted header"""
    separator = char * 80
    print('')
    print(separator)
    print(text)
    print(separator)

def check_backup_exists(cursor, backup_table):
    """Check if backup table exists and has data"""
    try:
        # Check if table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            )
        """, (backup_table,))
        
        exists = cursor.fetchone()[0]
        if not exists:
            return False, 0
        
        # Check row count
        cursor.execute(f"SELECT COUNT(*) FROM {backup_table}")
        row_count = cursor.fetchone()[0]
        
        return True, row_count
    except Exception as e:
        print(f"✗ Error checking backup table {backup_table}: {e}")
        return False, 0

def restore_table(conn, backup_table, production_table, description):
    """Restore a single table from backup"""
    print(f"\n{'='*80}")
    print(f"Restoring: {production_table}")
    print(f"Description: {description}")
    print(f"Source: {backup_table}")
    print(f"{'='*80}")
    
    cursor = conn.cursor()
    start_time = time.time()
    
    try:
        # Check if backup exists
        backup_exists, backup_rows = check_backup_exists(cursor, backup_table)
        
        if not backup_exists:
            print(f"✗ SKIP: Backup table '{backup_table}' does not exist!")
            return False, 0
        
        if backup_rows == 0:
            print(f"⚠ WARNING: Backup table '{backup_table}' is empty (0 rows)")
            response = input("  Continue anyway? (y/N): ").strip().lower()
            if response != 'y':
                print(f"✗ SKIPPED by user")
                return False, 0
        else:
            print(f"✓ Backup table found: {backup_rows:,} rows")
        
        # Get production table row count before restore
        cursor.execute(f"SELECT COUNT(*) FROM {production_table}")
        prod_rows_before = cursor.fetchone()[0]
        print(f"  Production table before: {prod_rows_before:,} rows")
        
        # Clear production table (DELETE then TRUNCATE for safety)
        # For tables with complex constraints, DELETE first then TRUNCATE
        print(f"  Clearing {production_table}...")
        
        # Step 1: DELETE all rows first
        print(f"    Deleting all rows...")
        cursor.execute(f"DELETE FROM {production_table}")
        conn.commit()
        
        # Step 2: Verify table is empty
        cursor.execute(f"SELECT COUNT(*) FROM {production_table}")
        remaining_rows = cursor.fetchone()[0]
        if remaining_rows > 0:
            print(f"    ⚠ WARNING: {remaining_rows} rows still remain after DELETE!")
            raise Exception(f"Table {production_table} not empty after DELETE - cannot proceed")
        
        print(f"  ✓ Cleared ({prod_rows_before:,} rows deleted)")
        
        # Get list of columns excluding generated columns
        print(f"  Getting column list (excluding generated columns)...")
        cursor.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
              AND table_name = '{production_table}'
              AND is_generated = 'NEVER'
            ORDER BY ordinal_position
        """)
        columns = [row[0] for row in cursor.fetchall()]
        columns_str = ', '.join([f'"{col}"' for col in columns])
        print(f"  ✓ Found {len(columns)} columns to copy")
        
        # Copy data from backup to production (only non-generated columns)
        print(f"  Copying data from {backup_table} to {production_table}...")
        cursor.execute(f"""
            INSERT INTO {production_table} ({columns_str})
            SELECT {columns_str} FROM {backup_table}
        """)
        rows_restored = cursor.rowcount
        conn.commit()
        
        elapsed = time.time() - start_time
        print(f"✓ Restored {rows_restored:,} rows in {elapsed:.2f}s")
        
        # Verify row count
        cursor.execute(f"SELECT COUNT(*) FROM {production_table}")
        prod_rows_after = cursor.fetchone()[0]
        
        if prod_rows_after != backup_rows:
            print(f"⚠ WARNING: Row count mismatch!")
            print(f"  Expected: {backup_rows:,}")
            print(f"  Got: {prod_rows_after:,}")
        else:
            print(f"✓ Verification passed: {prod_rows_after:,} rows")
        
        # Analyze table
        print(f"  Analyzing {production_table}...")
        cursor.execute(f"ANALYZE {production_table}")
        conn.commit()
        print(f"  ✓ Analyzed")
        
        return True, rows_restored
        
    except Exception as e:
        print(f"✗ Error restoring {production_table}: {e}")
        conn.rollback()
        return False, 0
    finally:
        cursor.close()

def main():
    """Main restore function"""
    print_header("RESTORE TABLES FROM BACKUP - PostgreSQL Data Restore Utility")
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {DB_CONFIG['database']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")
    
    print("\n⚠️  WARNING: This will TRUNCATE production tables and restore from backups!")
    print(f"Tables to restore: {len(RESTORE_CONFIG)}")
    for config in RESTORE_CONFIG:
        print(f"  - {config['production_table']} ← {config['backup_table']}")
    
    print("\n")
    response = input("Are you sure you want to continue? (yes/NO): ").strip().lower()
    if response != 'yes':
        print("\n✗ Restore cancelled by user")
        return 1
    
    conn = None
    try:
        # Connect to database
        print_header("STEP 1: Connecting to Database")
        print(f"Connecting to {DB_CONFIG['database']}...")
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        print('✓ Connected successfully')
        
        # Restore each table
        print_header("STEP 2: Restoring Tables")
        total_rows = 0
        success_count = 0
        failed_tables = []
        skipped_tables = []
        
        for i, config in enumerate(RESTORE_CONFIG, 1):
            print(f"\n[{i}/{len(RESTORE_CONFIG)}] {config['production_table']}")
            success, row_count = restore_table(
                conn,
                config['backup_table'],
                config['production_table'],
                config['description']
            )
            
            if success:
                total_rows += row_count
                success_count += 1
            else:
                if row_count == 0:
                    skipped_tables.append(config['production_table'])
                else:
                    failed_tables.append(config['production_table'])
        
        # Print summary
        print_header("STEP 3: Summary")
        print(f"\n{'Table Name':<30} {'Status':<15} {'Rows'}")
        print('-' * 80)
        
        cursor = conn.cursor()
        for config in RESTORE_CONFIG:
            table_name = config['production_table']
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                if table_name in failed_tables:
                    status = "❌ FAILED"
                elif table_name in skipped_tables:
                    status = "⊘ SKIPPED"
                else:
                    status = "✓ SUCCESS"
                
                print(f"{table_name:<30} {status:<15} {row_count:,}")
            except Exception as e:
                print(f"{table_name:<30} {'❌ ERROR':<15} {str(e)[:30]}")
        
        cursor.close()
        conn.close()
        
        # Final status
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if failed_tables:
            print_header('⚠️  COMPLETED WITH ERRORS', '=')
            print(f"Failed tables: {', '.join(failed_tables)}")
        else:
            if skipped_tables:
                print_header('✅ COMPLETED (Some tables skipped)', '=')
                print(f"Skipped tables: {', '.join(skipped_tables)}")
            else:
                print_header('✅ ALL TABLES RESTORED SUCCESSFULLY!', '=')
        
        print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"Total rows restored: {total_rows:,}")
        print(f"Tables restored: {success_count}/{len(RESTORE_CONFIG)}")
        if skipped_tables:
            print(f"Tables skipped: {len(skipped_tables)}")
        if failed_tables:
            print(f"Tables failed: {len(failed_tables)}")
        print('')
        print('💡 Next steps:')
        print('   1. Verify data integrity in restored tables')
        print('   2. Refresh materialized views: python refresh_all_views.py')
        print('   3. Restart applications if needed')
        print('')
        
        return 1 if failed_tables else 0
        
    except psycopg2.Error as e:
        print('')
        print(f"❌ Database Error: {e}")
        print(f"   Error Code: {e.pgcode}")
        return 1
    except Exception as e:
        print('')
        print(f"❌ Unexpected Error: {e}")
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
