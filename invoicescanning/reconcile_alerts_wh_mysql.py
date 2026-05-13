"""
Reconcile PostgreSQL WH alerts table with MySQL source by removing stale records.

This script:
1. Fetches all valid a_id values from MySQL source (invcentral.alerts)
2. Deletes PostgreSQL WH rows where a_id no longer exists in MySQL
3. Reports deleted count and final reconciliation status
"""

import pymysql
import psycopg2


def reconcile_alerts():
    """Remove stale alerts from PostgreSQL WH that were deleted from MySQL source."""
    
    # MySQL source configuration
    mysql_cfg = {
        'host': '192.168.0.17',
        'port': 3306,
        'user': 'misaccount',
        'password': 'Inv@Central@2024',
        'database': 'invcentral',
        'charset': 'utf8mb4'
    }
    
    # PostgreSQL WH configuration
    pg_cfg = {
        'host': 'localhost',
        'port': 3307,
        'user': 'postgres',
        'password': 'hello',
        'dbname': 'WH'
    }
    
    print("=" * 60)
    print("ALERTS TABLE RECONCILIATION: MySQL Source → PostgreSQL WH")
    print("=" * 60)
    
    # Step 1: Fetch all valid a_id from MySQL source
    print("\n[1/4] Connecting to MySQL source (192.168.0.17:3306/invcentral)...")
    mysql_conn = pymysql.connect(**mysql_cfg)
    try:
        with mysql_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM alerts")
            mysql_total = cur.fetchone()[0]
            print(f"      MySQL source has {mysql_total:,} total alerts")
            
            print("\n[2/4] Fetching all valid a_id from MySQL...")
            cur.execute("SELECT a_id FROM alerts WHERE a_id IS NOT NULL")
            mysql_ids = {row[0] for row in cur.fetchall()}
            print(f"      Fetched {len(mysql_ids):,} valid IDs from MySQL")
    finally:
        mysql_conn.close()
    
    # Step 2: Find and delete stale records in PostgreSQL
    print("\n[3/4] Connecting to PostgreSQL WH (localhost:3307/WH)...")
    pg_conn = psycopg2.connect(**pg_cfg)
    try:
        with pg_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM public.alerts")
            pg_total_before = cur.fetchone()[0]
            print(f"      PostgreSQL WH has {pg_total_before:,} total alerts (before cleanup)")
            
            # Get PostgreSQL IDs that don't exist in MySQL
            cur.execute("SELECT a_id FROM public.alerts WHERE a_id IS NOT NULL")
            pg_ids = {row[0] for row in cur.fetchall()}
            stale_ids = pg_ids - mysql_ids
            
            if not stale_ids:
                print("\n✅ No stale records found. PostgreSQL WH is already synchronized!")
            else:
                print(f"\n      Found {len(stale_ids):,} stale records in PostgreSQL WH")
                print(f"      Sample stale IDs: {sorted(list(stale_ids))[:20]}")
                
                print("\n[4/4] Deleting stale records from PostgreSQL WH...")
                # Use ANY for efficient bulk delete
                cur.execute(
                    "DELETE FROM public.alerts WHERE a_id = ANY(%s)",
                    (list(stale_ids),)
                )
                deleted_count = cur.rowcount
                pg_conn.commit()
                
                cur.execute("SELECT COUNT(*) FROM public.alerts")
                pg_total_after = cur.fetchone()[0]
                
                print(f"      ✅ Deleted {deleted_count:,} stale records")
                print(f"      PostgreSQL WH now has {pg_total_after:,} alerts")
    finally:
        pg_conn.close()
    
    # Step 3: Verification
    print("\n" + "=" * 60)
    print("RECONCILIATION COMPLETE")
    print("=" * 60)
    print(f"MySQL source:       {mysql_total:,} alerts")
    print(f"PostgreSQL WH:      {pg_total_after:,} alerts")
    
    if pg_total_after > mysql_total:
        diff = pg_total_after - mysql_total
        print(f"\n⚠️  PostgreSQL still has {diff:,} more records than MySQL")
        print("    This is expected if MySQL has new records above the sync cursor")
        print("    that haven't synced yet. Run the normal sync to pull them.")
    elif pg_total_after < mysql_total:
        diff = mysql_total - pg_total_after
        print(f"\n⚠️  PostgreSQL has {diff:,} fewer records than MySQL")
        print("    Run the MySQL → PostgreSQL sync to pull missing records.")
    else:
        print("\n✅ Perfect synchronization achieved!")
    
    # ACH bill date mismatch verification
    print("\n" + "-" * 60)
    print("ACH BILL DATE MISMATCH VERIFICATION")
    print("-" * 60)
    
    mysql_conn = pymysql.connect(**mysql_cfg)
    pg_conn = psycopg2.connect(**pg_cfg)
    try:
        with mysql_conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM alerts
                WHERE UPPER(TRIM(a_store_code)) = 'ACH'
                  AND LOWER(TRIM(a_type)) = 'bill date mismatch'
            """)
            mysql_ach_count = cur.fetchone()[0]
        
        with pg_conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM public.alerts
                WHERE UPPER(TRIM(a_store_code)) = 'ACH'
                  AND LOWER(TRIM(a_type)) = 'bill date mismatch'
            """)
            pg_ach_count = cur.fetchone()[0]
        
        print(f"MySQL source ACH bill date mismatch:      {mysql_ach_count:,}")
        print(f"PostgreSQL WH ACH bill date mismatch:     {pg_ach_count:,}")
        
        if pg_ach_count == mysql_ach_count:
            print("✅ ACH bill date mismatch counts match!")
        else:
            diff = pg_ach_count - mysql_ach_count
            print(f"⚠️  Difference: {diff:+,} records")
            if diff > 0:
                print("   PostgreSQL has more (likely stale/deleted in MySQL)")
            else:
                print("   MySQL has more (need to sync new records)")
    finally:
        mysql_conn.close()
        pg_conn.close()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        reconcile_alerts()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
