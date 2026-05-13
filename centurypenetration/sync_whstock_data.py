"""
Sync WHStock data from salesdata database to century_penetration database
Run this script whenever whstock data is updated in salesdata
"""

import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime

DB_CONFIG_SOURCE = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'salesdata'
}

DB_CONFIG_TARGET = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'century_penetration'
}

def sync_whstock():
    """Copy whstock data from salesdata to century_penetration"""
    
    print(f"[{datetime.now()}] Starting WHStock sync...")
    
    # Connect to source database (salesdata)
    print("Connecting to salesdata database...")
    conn_source = psycopg2.connect(**DB_CONFIG_SOURCE)
    cursor_source = conn_source.cursor()
    
    # Connect to target database (century_penetration)
    print("Connecting to century_penetration database...")
    conn_target = psycopg2.connect(**DB_CONFIG_TARGET)
    cursor_target = conn_target.cursor()
    
    try:
        # Get latest upload_date from source
        cursor_source.execute("SELECT MAX(upload_date) FROM whstock")
        latest_date = cursor_source.fetchone()[0]
        print(f"Latest upload date in salesdata.whstock: {latest_date}")
        
        # Check if this date already exists in target
        cursor_target.execute("SELECT MAX(upload_date) FROM whstock")
        target_latest = cursor_target.fetchone()[0]
        print(f"Latest upload date in century_penetration.whstock: {target_latest}")
        
        if target_latest and target_latest >= latest_date:
            print(f"✅ century_penetration.whstock is already up to date ({target_latest})")
            return
        
        # Fetch all data from source for latest upload_date
        print(f"Fetching whstock data for {latest_date}...")
        cursor_source.execute("""
            SELECT vc_item_code, wh_code, wh_name, balance_qty, upload_date
            FROM whstock
            WHERE upload_date = %s
        """, (latest_date,))
        
        rows = cursor_source.fetchall()
        print(f"Found {len(rows):,} records to sync")
        
        if not rows:
            print("⚠️ No data to sync")
            return
        
        # Insert into target database
        print("Inserting data into century_penetration.whstock...")
        insert_query = """
            INSERT INTO whstock (vc_item_code, wh_code, wh_name, balance_qty, upload_date)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """
        
        execute_batch(cursor_target, insert_query, rows, page_size=1000)
        conn_target.commit()
        
        print(f"✅ Successfully synced {len(rows):,} records")
        print(f"✅ century_penetration.whstock now has data up to {latest_date}")
        
    except Exception as e:
        print(f"❌ Error during sync: {e}")
        conn_target.rollback()
        raise
    
    finally:
        cursor_source.close()
        conn_source.close()
        cursor_target.close()
        conn_target.close()
        print(f"[{datetime.now()}] Sync completed")


if __name__ == "__main__":
    sync_whstock()
