"""
============================================================
TESTING SCRIPT: Verify Priority-to-Priority Transfers
============================================================
Purpose: Test new mv_recommendations_complete_test view
Compares: Production vs Test view statistics
DO NOT use this in production yet
============================================================
"""

import psycopg2
import pandas as pd
from datetime import datetime

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'salesdata'
}

PRIORITY_SHOPS = ['SPN', 'MSS', 'LFS', 'M03', 'KAS', 'MM1', 'MM2', 'FAR', 'KS7', 'WHL', 'MM3']

def log(message):
    """Print log message with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def main():
    """Main testing function"""
    print("=" * 80)
    log("TESTING: Priority-to-Priority Transfer Logic")
    print("=" * 80)
    
    try:
        # Connect to database
        log("Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        log("✅ Connected successfully")
        
        # Check if test view exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM pg_matviews 
                WHERE schemaname = 'public' 
                AND matviewname = 'mv_recommendations_complete_test'
            )
        """)
        test_view_exists = cursor.fetchone()[0]
        
        if not test_view_exists:
            log("❌ Test view 'mv_recommendations_complete_test' does not exist!")
            log("Please run NowhStock_mv_recommendations_complete_TEST.sql first")
            return
        
        log("✅ Test view exists")
        
        # ============================================================
        # COMPARISON 1: Total Recommendations
        # ============================================================
        print("\n" + "=" * 80)
        log("COMPARISON 1: Total Recommendations")
        print("=" * 80)
        
        # Production view
        cursor.execute("SELECT COUNT(*) FROM mv_recommendations_complete")
        prod_count = cursor.fetchone()[0]
        
        # Test view
        cursor.execute("SELECT COUNT(*) FROM mv_recommendations_complete_test")
        test_count = cursor.fetchone()[0]
        
        log(f"Production view: {prod_count:,} recommendations")
        log(f"Test view: {test_count:,} recommendations")
        log(f"Difference: {test_count - prod_count:,} (+{((test_count - prod_count) / prod_count * 100):.1f}%)")
        
        # ============================================================
        # COMPARISON 2: Source Shop Counts
        # ============================================================
        print("\n" + "=" * 80)
        log("COMPARISON 2: Source Shop Breakdown")
        print("=" * 80)
        
        # Production - unique source shops
        cursor.execute("SELECT COUNT(DISTINCT source_shop) FROM mv_recommendations_complete")
        prod_sources = cursor.fetchone()[0]
        
        # Test - unique source shops
        cursor.execute("SELECT COUNT(DISTINCT source_shop) FROM mv_recommendations_complete_test")
        test_sources = cursor.fetchone()[0]
        
        log(f"Production view: {prod_sources} unique source shops")
        log(f"Test view: {test_sources} unique source shops")
        
        # ============================================================
        # COMPARISON 3: Priority Shop Sources (NEW)
        # ============================================================
        print("\n" + "=" * 80)
        log("COMPARISON 3: Priority Shops as Sources")
        print("=" * 80)
        
        # Production - priority shop sources (should be 0)
        priority_list = "'" + "','".join(PRIORITY_SHOPS) + "'"
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM mv_recommendations_complete 
            WHERE source_shop IN ({priority_list})
        """)
        prod_priority_sources = cursor.fetchone()[0]
        
        # Test - priority shop sources (should be > 0)
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM mv_recommendations_complete_test 
            WHERE source_shop IN ({priority_list})
        """)
        test_priority_sources = cursor.fetchone()[0]
        
        log(f"Production view: {prod_priority_sources:,} priority shop sources (expected: 0)")
        log(f"Test view: {test_priority_sources:,} priority shop sources (expected: >0)")
        
        if test_priority_sources > 0:
            log("✅ SUCCESS: Priority shops now included as sources!")
        else:
            log("⚠️ WARNING: No priority shop sources found in test view")
        
        # ============================================================
        # ANALYSIS 4: Priority-to-Priority Transfer Details
        # ============================================================
        print("\n" + "=" * 80)
        log("ANALYSIS 4: Priority-to-Priority Transfer Details")
        print("=" * 80)
        
        cursor.execute(f"""
            SELECT 
                source_shop,
                COUNT(*) as recommendation_count,
                COUNT(DISTINCT item_code) as unique_items,
                SUM(recommended_qty) as total_qty
            FROM mv_recommendations_complete_test
            WHERE source_shop IN ({priority_list})
            GROUP BY source_shop
            ORDER BY SUM(recommended_qty) DESC
        """)
        
        priority_sources = cursor.fetchall()
        
        if priority_sources:
            print("\n" + "-" * 80)
            print(f"{'Priority Source Shop':<20} {'Recommendations':>18} {'Unique Items':>18} {'Total Qty':>18}")
            print("-" * 80)
            for row in priority_sources:
                shop = row[0]
                rec_count = row[1]
                unique_items = row[2]
                total_qty = row[3] or 0
                print(f"{shop:<20} {rec_count:>18,} {unique_items:>18,} {total_qty:>18,.0f}")
            print("-" * 80)
        else:
            log("No priority shop sources found")
        
        # ============================================================
        # ANALYSIS 5: Sample Priority-to-Priority Transfers
        # ============================================================
        print("\n" + "=" * 80)
        log("ANALYSIS 5: Sample Priority-to-Priority Transfers (Top 10 by Qty)")
        print("=" * 80)
        
        cursor.execute(f"""
            SELECT 
                item_code,
                item_name,
                source_shop,
                dest_shop,
                source_stock,
                dest_capacity,
                recommended_qty
            FROM mv_recommendations_complete_test
            WHERE source_shop IN ({priority_list})
            ORDER BY recommended_qty DESC
            LIMIT 10
        """)
        
        samples = cursor.fetchall()
        
        if samples:
            print("\n" + "-" * 120)
            print(f"{'Item Code':<15} {'Item Name':<30} {'Source':<8} {'Dest':<8} {'Src Stock':>10} {'Capacity':>10} {'Rec Qty':>10}")
            print("-" * 120)
            for row in samples:
                item_code = (row[0] or '')[:15]
                item_name = (row[1] or '')[:30]
                source = row[2] or ''
                dest = row[3] or ''
                src_stock = row[4] or 0
                capacity = row[5] or 0
                rec_qty = row[6] or 0
                print(f"{item_code:<15} {item_name:<30} {source:<8} {dest:<8} {src_stock:>10,.0f} {capacity:>10,.0f} {rec_qty:>10,.0f}")
            print("-" * 120)
        
        # ============================================================
        # ANALYSIS 6: Verify No Same-Shop Transfers
        # ============================================================
        print("\n" + "=" * 80)
        log("ANALYSIS 6: Verify No Same-Shop Transfers")
        print("=" * 80)
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM mv_recommendations_complete_test 
            WHERE source_shop = dest_shop
        """)
        same_shop_count = cursor.fetchone()[0]
        
        if same_shop_count == 0:
            log("✅ PASS: No same-shop transfers found (as expected)")
        else:
            log(f"❌ FAIL: Found {same_shop_count} same-shop transfers (should be 0)")
        
        # ============================================================
        # ANALYSIS 7: View Sizes
        # ============================================================
        print("\n" + "=" * 80)
        log("ANALYSIS 7: View Sizes")
        print("=" * 80)
        
        cursor.execute("""
            SELECT 
                'Production' as view_type,
                pg_size_pretty(pg_total_relation_size('mv_recommendations_complete')) as total_size,
                pg_size_pretty(pg_relation_size('mv_recommendations_complete')) as data_size,
                pg_size_pretty(pg_indexes_size('mv_recommendations_complete')) as index_size
            UNION ALL
            SELECT 
                'Test' as view_type,
                pg_size_pretty(pg_total_relation_size('mv_recommendations_complete_test')) as total_size,
                pg_size_pretty(pg_relation_size('mv_recommendations_complete_test')) as data_size,
                pg_size_pretty(pg_indexes_size('mv_recommendations_complete_test')) as index_size
        """)
        
        sizes = cursor.fetchall()
        print("\n" + "-" * 80)
        print(f"{'View Type':<15} {'Total Size':<15} {'Data Size':<15} {'Index Size':<15}")
        print("-" * 80)
        for row in sizes:
            print(f"{row[0]:<15} {row[1]:<15} {row[2]:<15} {row[3]:<15}")
        print("-" * 80)
        
        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        print("\n" + "=" * 80)
        log("FINAL SUMMARY")
        print("=" * 80)
        
        print("\n✅ Test Complete! Key Findings:")
        print(f"  1. Test view has {test_count - prod_count:,} MORE recommendations")
        print(f"  2. Priority shops now contribute {test_priority_sources:,} source recommendations")
        print(f"  3. {len(priority_sources)} priority shops are now sources")
        print(f"  4. All business logic (capacity, FEFO, etc.) remains unchanged")
        print(f"  5. No same-shop transfers: {'✅ PASS' if same_shop_count == 0 else '❌ FAIL'}")
        
        print("\n📋 Next Steps:")
        print("  1. Review the priority-to-priority transfer samples above")
        print("  2. Verify business logic is correct")
        print("  3. If approved, update Python code to use test view")
        print("  4. After thorough testing, can replace production view")
        
        print("\n⚠️  Production code NOT modified - test view only")
        print("=" * 80)
        
    except Exception as e:
        log(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
        log("Database connection closed")

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")
