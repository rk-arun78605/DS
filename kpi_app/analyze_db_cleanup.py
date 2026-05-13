"""
Analyze database tables and views to find unused ones
"""
import psycopg2
from contextlib import contextmanager

@contextmanager
def get_db_connection(dbname):
    conn = psycopg2.connect(
        host='localhost',
        port=3307,
        user='postgres',
        password='hello',
        database=dbname
    )
    try:
        yield conn
    finally:
        conn.close()

def analyze_salesdata_db():
    """Analyze salesdata database for unused tables/views"""
    print("="*80)
    print("SALESDATA DATABASE ANALYSIS")
    print("="*80)
    
    with get_db_connection('salesdata') as conn:
        with conn.cursor() as cur:
            # Get all tables
            cur.execute("""
                SELECT schemaname, tablename, 
                       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            """)
            tables = cur.fetchall()
            
            print(f"\n{'TABLE NAME':<40} {'SIZE':<15}")
            print("-"*80)
            total_size = 0
            for schema, table, size in tables:
                print(f"{table:<40} {size:<15}")
            
            # Get all materialized views
            print("\n" + "="*80)
            print("MATERIALIZED VIEWS")
            print("="*80)
            cur.execute("""
                SELECT schemaname, matviewname,
                       pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) as size
                FROM pg_matviews
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||matviewname) DESC
            """)
            mvs = cur.fetchall()
            
            print(f"\n{'MATERIALIZED VIEW':<50} {'SIZE':<15}")
            print("-"*80)
            for schema, mv, size in mvs:
                print(f"{mv:<50} {size:<15}")
            
            # Get all regular views
            print("\n" + "="*80)
            print("REGULAR VIEWS")
            print("="*80)
            cur.execute("""
                SELECT schemaname, viewname
                FROM pg_views
                WHERE schemaname = 'public'
                ORDER BY viewname
            """)
            views = cur.fetchall()
            
            for schema, view in views:
                print(f"  - {view}")
            
            # Get all indexes
            print("\n" + "="*80)
            print("INDEXES (Top 20 by size)")
            print("="*80)
            cur.execute("""
                SELECT
                    schemaname,
                    tablename,
                    indexname,
                    pg_size_pretty(pg_relation_size(indexrelid)) as size,
                    idx_scan as scans
                FROM pg_stat_user_indexes
                WHERE schemaname = 'public'
                ORDER BY pg_relation_size(indexrelid) DESC
                LIMIT 20
            """)
            indexes = cur.fetchall()
            
            print(f"\n{'INDEX NAME':<50} {'SIZE':<12} {'SCANS':<15}")
            print("-"*80)
            for schema, table, index, size, scans in indexes:
                status = "✅ USED" if scans and scans > 0 else "❌ UNUSED"
                print(f"{index:<50} {size:<12} {scans or 0:<10} {status}")

def suggest_cleanup():
    """Suggest tables/views to remove"""
    print("\n" + "="*80)
    print("CLEANUP RECOMMENDATIONS")
    print("="*80)
    
    # Tables that are typically temporary or not needed
    potential_unused = [
        'pg_stat_statements',
        'test_',  # Any table starting with test_
        'temp_',  # Any table starting with temp_
        'backup_',  # Any table starting with backup_
        '_old',  # Any table ending with _old
    ]
    
    with get_db_connection('salesdata') as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename
            """)
            all_tables = [row[0] for row in cur.fetchall()]
    
    print("\n📋 TABLES TO CHECK:")
    for table in all_tables:
        for pattern in potential_unused:
            if pattern in table.lower():
                print(f"  ❓ {table} - Check if needed")
                break
    
    print("\n💡 TO DELETE UNUSED TABLES:")
    print("   DROP TABLE IF EXISTS table_name CASCADE;")
    print("\n💡 TO DELETE UNUSED VIEWS:")
    print("   DROP VIEW IF EXISTS view_name CASCADE;")
    print("\n💡 TO DELETE UNUSED MATERIALIZED VIEWS:")
    print("   DROP MATERIALIZED VIEW IF EXISTS mv_name CASCADE;")
    print("\n💡 TO DELETE UNUSED INDEXES:")
    print("   DROP INDEX IF EXISTS index_name;")
    
    print("\n⚠️  RECOMMENDED ACTIONS:")
    print("1. Review unused indexes (scans = 0)")
    print("2. Check for old backup tables")
    print("3. Remove test/temp tables")
    print("4. Vacuum full after deletions: VACUUM FULL ANALYZE;")

if __name__ == "__main__":
    try:
        analyze_salesdata_db()
        suggest_cleanup()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
