"""
Century Penetration Database Setup Script
Creates database, tables, and materialized views
"""
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello'
}

def create_database():
    """Create century_penetration database if it doesn't exist"""
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(**DB_CONFIG, database='postgres')
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = 'century_penetration'"
        )
        exists = cursor.fetchone()
        
        if exists:
            logger.info("✅ Database 'century_penetration' already exists")
        else:
            # Create database
            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier('century_penetration')
                )
            )
            logger.info("✅ Created database 'century_penetration'")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating database: {e}")
        return False

def run_sql_file(sql_file_path):
    """Execute SQL file to create tables and views"""
    try:
        # Read SQL file
        sql_path = Path(sql_file_path)
        if not sql_path.exists():
            logger.error(f"❌ SQL file not found: {sql_file_path}")
            return False
        
        with open(sql_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        # Connect to century_penetration database
        conn = psycopg2.connect(**DB_CONFIG, database='century_penetration')
        cursor = conn.cursor()
        
        logger.info(f"📋 Executing SQL script: {sql_file_path}")
        
        # Execute the entire script
        cursor.execute(sql_script)
        conn.commit()
        
        logger.info("✅ Successfully created tables and views")
        
        # Verify created objects
        cursor.execute("""
            SELECT table_name, table_type 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_type, table_name
        """)
        
        objects = cursor.fetchall()
        logger.info(f"\n📊 Created database objects:")
        for obj_name, obj_type in objects:
            logger.info(f"  - {obj_type}: {obj_name}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error executing SQL file: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

def verify_setup():
    """Verify database setup is complete"""
    try:
        conn = psycopg2.connect(**DB_CONFIG, database='century_penetration')
        cursor = conn.cursor()
        
        # Check for required tables
        required_tables = ['reorder_level', 'sales', 'sit']
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            AND table_name = ANY(%s)
        """, (required_tables,))
        
        found_tables = [row[0] for row in cursor.fetchall()]
        
        # Check for required views
        required_views = [
            'mv_sales_metrics', 
            'mv_sit_summary', 
            'mv_century_penetration'
        ]
        cursor.execute("""
            SELECT matviewname 
            FROM pg_matviews 
            WHERE schemaname = 'public'
            AND matviewname = ANY(%s)
        """, (required_views,))
        
        found_views = [row[0] for row in cursor.fetchall()]
        
        # Check for refresh function
        cursor.execute("""
            SELECT routine_name 
            FROM information_schema.routines 
            WHERE routine_schema = 'public'
            AND routine_name = 'refresh_century_views'
        """)
        
        has_function = cursor.fetchone() is not None
        
        cursor.close()
        conn.close()
        
        # Report results
        logger.info("\n🔍 Verification Results:")
        logger.info(f"  Tables: {len(found_tables)}/{len(required_tables)}")
        for table in required_tables:
            status = "✅" if table in found_tables else "❌"
            logger.info(f"    {status} {table}")
        
        logger.info(f"  Materialized Views: {len(found_views)}/{len(required_views)}")
        for view in required_views:
            status = "✅" if view in found_views else "❌"
            logger.info(f"    {status} {view}")
        
        logger.info(f"  Functions:")
        logger.info(f"    {'✅' if has_function else '❌'} refresh_century_views()")
        
        all_good = (
            len(found_tables) == len(required_tables) and
            len(found_views) == len(required_views) and
            has_function
        )
        
        if all_good:
            logger.info("\n✅ Database setup is complete and verified!")
        else:
            logger.warning("\n⚠️ Database setup is incomplete")
        
        return all_good
        
    except Exception as e:
        logger.error(f"❌ Error verifying setup: {e}")
        return False

def main():
    """Main setup workflow"""
    logger.info("=" * 60)
    logger.info("Century Penetration Database Setup")
    logger.info("=" * 60)
    
    # Step 1: Create database
    logger.info("\n📌 Step 1: Creating database...")
    if not create_database():
        logger.error("Failed to create database. Exiting.")
        return
    
    # Step 2: Create tables and views
    logger.info("\n📌 Step 2: Creating tables and views...")
    sql_file = Path(__file__).parent / 'create_century_tables.sql'
    if not run_sql_file(sql_file):
        logger.error("Failed to create tables and views. Exiting.")
        return
    
    # Step 3: Verify setup
    logger.info("\n📌 Step 3: Verifying setup...")
    if verify_setup():
        logger.info("\n🎉 Setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("  1. Run: python load_century_data.py")
        logger.info("  2. This will load reorder_level and SIT data")
        logger.info("  3. Add sales CSV files (jan25.csv - dec25.csv) when ready")
        logger.info("  4. Run load_century_data.py again to load sales")
    else:
        logger.warning("\n⚠️ Setup completed with warnings. Check logs above.")
    
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
