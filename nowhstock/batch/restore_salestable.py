import psycopg2
import sys
import time

# Database connection parameters
DB_NAME = "salesdata"
DB_USER = "postgres"
DB_PASSWORD = "hello"
DB_HOST = "localhost"
DB_PORT = "3307"

SOURCE_TABLE = "sales_2025_backup"  # <-- REPLACE WITH YOUR SOURCE TABLE NAME
DESTINATION_TABLE = "sales_2025" # <-- REPLACE WITH YOUR DESTINATION TABLE NAME

def copy_table_data_with_feedback(source, destination):
    conn = None
    cursor = None
    try:
        print(f"Connecting to database '{DB_NAME}' on port {DB_PORT}...")
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
        print("✅ Connection successful.")

        copy_query = f"""
        INSERT INTO {destination}
        SELECT * FROM {source};
        """
        
        print("-" * 40)
        print(f"🔄 Starting large data copy operation...")
        print(f"FROM: '{source}'")
        print(f"TO:   '{destination}'")
        print("⚠️  This window may remain blank for a while until complete.")
        print("-" * 40)

        start_time = time.time()
        # --- THE EXECUTION STARTS HERE (The blocking operation) ---
        cursor.execute(copy_query)
        # --- THE EXECUTION ENDS HERE (Execution resumes below) ---
        end_time = time.time()

        conn.commit()

        duration = end_time - start_time
        print("-" * 40)
        print(f"✅ Process Completed Successfully in {duration:.2f} seconds.")
        print(f"Total rows copied: {cursor.rowcount}")
        print("-" * 40)

    except psycopg2.OperationalError as e:
        print(f"❌ Error connecting to the database. Check port {DB_PORT}, user/password, and if the DB is running.")
        print(f"Details: {e}")
        sys.exit(1)
    except psycopg2.Error as e:
        print(f"❌ A database error occurred during the copy operation: {e}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    copy_table_data_with_feedback(SOURCE_TABLE, DESTINATION_TABLE)
