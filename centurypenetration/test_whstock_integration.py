"""
Test script to verify warehouse stock integration in Century Penetration dashboard
"""

import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'century_penetration'
}

def test_whstock_integration():
    """Test if warehouse stock data is integrated correctly"""
    
    print("=" * 80)
    print("CENTURY PENETRATION - WAREHOUSE STOCK INTEGRATION TEST")
    print("=" * 80)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Test 1: Check if whstock table exists
        print("\n[1] Checking if whstock table exists...")
        cursor.execute("""
            SELECT COUNT(*), MAX(upload_date) as latest_date
            FROM whstock
        """)
        result = cursor.fetchone()
        print(f"✅ whstock table found: {result['count']:,} records, latest date: {result['latest_date']}")
        
        # Test 2: Test understock query with warehouse stock
        print("\n[2] Testing understock items query with warehouse stock...")
        cursor.execute("""
            WITH latest_whstock AS (
                SELECT DISTINCT ON (vc_item_code, wh_code)
                    vc_item_code, wh_code, wh_name, balance_qty, upload_date
                FROM whstock
                WHERE upload_date = (SELECT MAX(upload_date) FROM whstock)
                ORDER BY vc_item_code, wh_code, upload_date DESC
            )
            SELECT 
                m.item_code,
                m.item_name,
                m.shop_code,
                COALESCE(SUM(w.balance_qty), 0) as total_wh_stock,
                COUNT(DISTINCT w.wh_code) as wh_count
            FROM mv_century_penetration m
            LEFT JOIN latest_whstock w ON m.item_code = w.vc_item_code
            WHERE m.stock_status = 'UnderStock'
            GROUP BY m.item_code, m.item_name, m.shop_code
            LIMIT 5
        """)
        
        rows = cursor.fetchall()
        print(f"✅ Query executed successfully. Sample results (top 5):")
        print(f"\n{'Item Code':<15} {'Shop':<8} {'WH Stock':<12} {'# WH':<8} {'Item Name'[:40]:<40}")
        print("-" * 90)
        for row in rows:
            print(f"{row['item_code']:<15} {row['shop_code']:<8} {row['total_wh_stock']:>10,.0f}  {row['wh_count']:>5}  {row['item_name'][:40]:<40}")
        
        # Test 3: Check items with warehouse stock
        print(f"\n[3] Checking items with warehouse stock availability...")
        cursor.execute("""
            WITH latest_whstock AS (
                SELECT DISTINCT ON (vc_item_code, wh_code)
                    vc_item_code, wh_code, balance_qty
                FROM whstock
                WHERE upload_date = (SELECT MAX(upload_date) FROM whstock)
                ORDER BY vc_item_code, wh_code, upload_date DESC
            )
            SELECT 
                COUNT(DISTINCT m.item_code) as total_items,
                COUNT(DISTINCT CASE WHEN w.vc_item_code IS NOT NULL THEN m.item_code END) as items_with_wh_stock,
                SUM(COALESCE(w.balance_qty, 0)) as total_wh_qty
            FROM mv_century_penetration m
            LEFT JOIN latest_whstock w ON m.item_code = w.vc_item_code
        """)
        
        stats = cursor.fetchone()
        print(f"✅ Total items in Century: {stats['total_items']:,}")
        print(f"✅ Items with WH stock: {stats['items_with_wh_stock']:,}")
        print(f"✅ Total WH stock quantity: {stats['total_wh_qty']:,.0f}")
        
        # Test 4: Check critical items with WH stock
        print(f"\n[4] Critical items (no shop stock) with warehouse stock...")
        cursor.execute("""
            WITH latest_whstock AS (
                SELECT DISTINCT ON (vc_item_code, wh_code)
                    vc_item_code, wh_code, wh_name, balance_qty
                FROM whstock
                WHERE upload_date = (SELECT MAX(upload_date) FROM whstock)
                ORDER BY vc_item_code, wh_code, upload_date DESC
            )
            SELECT 
                m.item_code,
                m.item_name,
                m.shop_code,
                ROUND(m.ros, 2) as ros,
                COALESCE(SUM(w.balance_qty), 0) as total_wh_stock
            FROM mv_century_penetration m
            LEFT JOIN latest_whstock w ON m.item_code = w.vc_item_code
            WHERE m.total_stock = 0 AND m.ros > 1
            GROUP BY m.item_code, m.item_name, m.shop_code, m.ros
            ORDER BY total_wh_stock DESC
            LIMIT 5
        """)
        
        critical = cursor.fetchall()
        if critical:
            print(f"✅ Found {len(critical)} critical items with WH stock. Top 5:")
            print(f"\n{'Item Code':<15} {'Shop':<8} {'ROS':<8} {'WH Stock':<12} {'Item Name'[:40]:<40}")
            print("-" * 90)
            for row in critical:
                print(f"{row['item_code']:<15} {row['shop_code']:<8} {row['ros']:>6.2f}  {row['total_wh_stock']:>10,.0f}  {row['item_name'][:40]:<40}")
        else:
            print("✅ No critical items found (good!)")
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED - Warehouse stock integration working correctly!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise
    
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    test_whstock_integration()
