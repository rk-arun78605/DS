#!/usr/bin/env python3
"""Find rows with the values user reported seeing (-100, -200, 59742)."""
import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'dbname': 'WH'
}

def find_reported_values():
    """Search for rows with values -100, -200 in diff_val, value_loaded, value_offloaded."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        print("🔍 Searching for rows with -100 diff_val OR -100 value_loaded OR -200 value_offloaded:\n")
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Search in main table
            cur.execute("""
                SELECT 
                    date, shop_code, item_code, item_name,
                    qty_loaded, qty_offloaded, price,
                    value_loaded, value_offloaded, diff_val
                FROM offloading_vs_loading
                WHERE value_loaded = -100 OR value_offloaded = -200 OR diff_val = -100 OR diff_val = -200
                LIMIT 5
            """)
            rows = cur.fetchall()
            
            if rows:
                print(f"  Found {len(rows)} rows in base table:\n")
                for row in rows:
                    print(f"    Item: {row['item_code']}, Date: {row['date']}")
                    print(f"    Qty: {row['qty_loaded']} → {row['qty_offloaded']}, Price: {row['price']}")
                    print(f"    Values: {row['value_loaded']} → {row['value_offloaded']}, Diff: {row['diff_val']}")
                    print()
            else:
                print("  ❌ No rows with those values found in base table\n")
        
        print("🔍 Searching for rows with LFS shop_code and large value (59742):\n")
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Search for LFS rows with sum around 59742
            cur.execute("""
                SELECT 
                    shop_code,
                    SUM(value_offloaded) as total_offloaded,
                    SUM(CASE WHEN (qty_offloaded - qty_loaded) < 0 THEN (qty_offloaded * price) ELSE 0 END) as short_val
                FROM offloading_vs_loading
                WHERE shop_code = 'LFS'
                GROUP BY shop_code
            """)
            rows = cur.fetchall()
            for row in rows:
                print(f"    LFS Total Offloaded Value: {row['total_offloaded']}")
                print(f"    LFS Short Received Value: {row['short_val']}")
                print()
        
        conn.close()
        
    except psycopg2.Error as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    find_reported_values()
