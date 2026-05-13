"""
Setup whstock table in century_penetration database
"""

import psycopg2

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'century_penetration'
}

def create_whstock_table():
    """Create whstock table with indexes"""
    
    print("Connecting to century_penetration database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        print("Creating whstock table...")
        
        # Read SQL file
        with open('create_whstock_table.sql', 'r') as f:
            sql_script = f.read()
        
        # Execute SQL
        cursor.execute(sql_script)
        conn.commit()
        
        print("✅ Successfully created whstock table with indexes")
        
        # Verify table exists
        cursor.execute("""
            SELECT table_name, column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'whstock'
            ORDER BY ordinal_position
        """)
        
        print("\nTable structure:")
        for row in cursor.fetchall():
            print(f"  {row[1]}: {row[2]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
        raise
    
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    create_whstock_table()
