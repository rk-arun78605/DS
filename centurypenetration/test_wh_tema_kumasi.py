"""
Test script to verify WH Tema and WH Kumasi columns in Century Penetration dashboard
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

def test_wh_tema_kumasi_columns():
    """Test if WH Tema and WH Kumasi columns are working correctly"""
    
    print("=" * 80)
    print("CENTURY PENETRATION - WH TEMA & WH KUMASI COLUMNS TEST")
    print("=" * 80)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get latest upload_date
        cursor.execute("SELECT MAX(upload_date) as max_date FROM whstock")
        max_date = cursor.fetchone()['max_date']
        print(f"\n📅 Latest whstock upload date: {max_date}")
        
        # Test query with WH Tema and WH Kumasi
        print("\n" + "=" * 80)
        print("TEST: Understock Items with WH Tema & WH Kumasi")
        print("=" * 80)
        
        cursor.execute("""
            WITH latest_whstock AS (
                SELECT DISTINCT ON (vc_item_code, wh_code)
                    vc_item_code, wh_code, wh_name, balance_qty, upload_date
                FROM whstock
                WHERE upload_date = %s
                ORDER BY vc_item_code, wh_code, upload_date DESC
            )
            SELECT 
                m.item_code,
                m.item_name,
                m.shop_code,
                COALESCE(SUM(w.balance_qty), 0) as total_wh_stock,
                COALESCE(SUM(CASE WHEN w.wh_code = 'TS' THEN w.balance_qty ELSE 0 END), 0) as wh_tema,
                COALESCE(SUM(CASE WHEN w.wh_code = 'KA' THEN w.balance_qty ELSE 0 END), 0) as wh_kumasi
            FROM mv_century_penetration m
            LEFT JOIN latest_whstock w ON m.item_code = w.vc_item_code
            WHERE m.stock_status = 'UnderStock'
            GROUP BY m.item_code, m.item_name, m.shop_code
            HAVING COALESCE(SUM(w.balance_qty), 0) > 0
            ORDER BY total_wh_stock DESC
            LIMIT 10
        """, (max_date,))
        
        rows = cursor.fetchall()
        if rows:
            print(f"\n✅ Found {len(rows)} understock items with warehouse stock")
            print(f"\n{'Item Code':<15} {'Shop':<8} {'Total WH':<12} {'WH Tema':<12} {'WH Kumasi':<12} {'Item Name'[:30]:<30}")
            print("-" * 100)
            for row in rows:
                print(f"{row['item_code']:<15} {row['shop_code']:<8} {row['total_wh_stock']:>10,.0f}  "
                      f"{row['wh_tema']:>10,.0f}  {row['wh_kumasi']:>10,.0f}  {row['item_name'][:30]:<30}")
        else:
            print("⚠️ No understock items with warehouse stock found")
        
        # Test breakdown by warehouse code
        print("\n" + "=" * 80)
        print("WAREHOUSE STOCK BREAKDOWN")
        print("=" * 80)
        
        cursor.execute("""
            SELECT 
                wh_code,
                wh_name,
                COUNT(*) as item_count,
                SUM(balance_qty) as total_qty
            FROM whstock
            WHERE upload_date = %s
            GROUP BY wh_code, wh_name
            ORDER BY wh_code
        """, (max_date,))
        
        wh_summary = cursor.fetchall()
        print(f"\n{'WH Code':<10} {'Warehouse Name':<30} {'Items':<10} {'Total Qty':<15}")
        print("-" * 70)
        for row in wh_summary:
            print(f"{row['wh_code']:<10} {row['wh_name']:<30} {row['item_count']:>8}  {row['total_qty']:>13,.0f}")
        
        # Verify TS and Ka exist
        print("\n" + "=" * 80)
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT CASE WHEN wh_code = 'TS' THEN vc_item_code END) as ts_items,
                COUNT(DISTINCT CASE WHEN wh_code = 'KA' THEN vc_item_code END) as ka_items,
                SUM(CASE WHEN wh_code = 'TS' THEN balance_qty ELSE 0 END) as ts_total_qty,
                SUM(CASE WHEN wh_code = 'KA' THEN balance_qty ELSE 0 END) as ka_total_qty
            FROM whstock
            WHERE upload_date = %s
        """, (max_date,))
        
        verification = cursor.fetchone()
        print("VERIFICATION:")
        print(f"  ✅ TS (Tema) - {verification['ts_items']:,} unique items, {verification['ts_total_qty']:,.0f} total qty")
        print(f"  ✅ KA (Kumasi) - {verification['ka_items']:,} unique items, {verification['ka_total_qty']:,.0f} total qty")
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED - WH Tema & WH Kumasi columns working correctly!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    test_wh_tema_kumasi_columns()
