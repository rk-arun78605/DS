#!/usr/bin/env python3
"""
Refresh All Materialized Views and Analyze Tables
==================================================
This script refreshes all materialized views and analyzes tables used by the application.
Run this after data changes to update all statistics and views.

Usage:
    python refresh_all_views.py
"""

import psycopg2
import sys
from datetime import datetime, timedelta
import time

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'database': 'salesdata',
    'user': 'postgres',
    'password': 'hello'
}

# Tables and views to refresh
BASE_TABLES = [
    'inventory_master',
    'sales_2024',
    'sales_2025',
    'sup_shop_grn',
    'shopexpiry',
    'itemdetails'
]

MATERIALIZED_VIEWS = [
    'mv_last_30d_sales',  # Optional
    'mv_recommendations_complete'  # Main view
]

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(text)
    print("="*60)

def execute_query(cursor, query, description=""):
    """Execute a query and print the result"""
    try:
        start = time.time()
        cursor.execute(query)
        elapsed = time.time() - start
        if description:
            print(f"✓ {description} ({elapsed:.2f}s)")
        return True
    except Exception as e:
        print(f"✗ {description}: {e}")
        return False

def main():
    """Main refresh function"""
    print_header("REFRESH ALL MATERIALIZED VIEWS AND TABLES")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Connect to database
        print("\n🔌 Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cursor = conn.cursor()
        print("✓ Connected successfully")
        
        # Step 1: Analyze base tables
        print_header("STEP 1: Analyzing Base Tables")
        for table in BASE_TABLES:
            execute_query(cursor, f"ANALYZE {table};", f"Analyzed {table}")
        
        # Step 2: Refresh materialized views
        print_header("STEP 2: Refreshing Materialized Views")
        for view in MATERIALIZED_VIEWS:
            # Check if view exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_matviews 
                    WHERE matviewname = %s
                );
            """, (view,))
            exists = cursor.fetchone()[0]
            
            if exists:
                print(f"\n🔄 Refreshing {view}...")
                success = execute_query(
                    cursor, 
                    f"REFRESH MATERIALIZED VIEW {view};",
                    f"Refreshed {view}"
                )
                if not success and view == 'mv_recommendations_complete':
                    print("⚠️  WARNING: Main view refresh failed!")
                    print("    Run: psql -U postgres -d salesdata -p 3307 -f create_mv_recommendations_complete.sql")
            else:
                print(f"⊘ {view} does not exist (skipping)")
        
        # Step 3: Analyze materialized views
        print_header("STEP 3: Analyzing Materialized Views")
        for view in MATERIALIZED_VIEWS:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_matviews 
                    WHERE matviewname = %s
                );
            """, (view,))
            exists = cursor.fetchone()[0]
            
            if exists:
                execute_query(cursor, f"ANALYZE {view};", f"Analyzed {view}")
        
        # Step 4: Display statistics
        print_header("STEP 4: View Statistics")
        cursor.execute("""
            SELECT 
                matviewname AS view_name,
                pg_size_pretty(pg_total_relation_size('public.'||matviewname)) AS size,
                last_refresh
            FROM pg_matviews 
            WHERE matviewname IN ('mv_recommendations_complete', 'mv_last_30d_sales')
            ORDER BY matviewname;
        """)
        
        print(f"\n{'View Name':<35} {'Size':<15} {'Last Refresh'}")
        print("-" * 60)
        for row in cursor.fetchall():
            view_name, size, last_refresh = row
            refresh_str = last_refresh.strftime('%Y-%m-%d %H:%M:%S') if last_refresh else 'Never'
            print(f"{view_name:<35} {size:<15} {refresh_str}")
        
        # Step 5: Display table statistics
        print_header("STEP 5: Table Statistics")
        cursor.execute("""
            SELECT 
                tablename AS table_name,
                pg_size_pretty(pg_total_relation_size('public.'||tablename)) AS total_size,
                n_live_tup AS row_count,
                last_analyze
            FROM pg_stat_user_tables 
            WHERE tablename IN ('inventory_master', 'sales_2024', 'sales_2025', 
                                'sup_shop_grn', 'shopexpiry', 'itemdetails')
            ORDER BY tablename;
        """)
        
        print(f"\n{'Table Name':<25} {'Size':<15} {'Rows':<12} {'Last Analyze'}")
        print("-" * 80)
        for row in cursor.fetchall():
            table_name, size, row_count, last_analyze = row
            analyze_str = last_analyze.strftime('%Y-%m-%d %H:%M:%S') if last_analyze else 'Never'
            print(f"{table_name:<25} {size:<15} {row_count:<12,} {analyze_str}")
        
        # Close connection
        cursor.close()
        conn.close()
        
        print_header("✅ REFRESH COMPLETE!")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n💡 You can now restart the Streamlit application:")
        print("   cd d:\\Dashboard Code\\NO_WH\\DS")
        print("   streamlit run nowhstock_ds.py")
        
        return 0
        
    except psycopg2.Error as e:
        print(f"\n❌ Database Error: {e}")
        print(f"   Error Code: {e.pgcode}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
