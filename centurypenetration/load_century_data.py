"""
Century Stock Penetration - Data Loading Script
Loads CSV files into PostgreSQL database with validation and logging
"""

import pandas as pd
import psycopg2
from psycopg2 import Error
from psycopg2.extras import execute_batch
import os
from datetime import datetime
import logging
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'century_penetration'
}

DATA_DIR = Path(r'D:\Dashboard Code\NO_WH\DS\centurypenetration\data')

# File mappings
FILES = {
    'reorder_level': 'GEN_reorder_till10dec25.csv',
    'sit': 'GEN_SIT_till10dec25.csv',
    'sales': {
        'jan25': 'jan25.csv',
        'feb25': 'feb25.csv',
        'mar25': 'mar25.csv',
        'apr25': 'apr25.csv',
        'may25': 'may25.csv',
        'jun25': 'jun25.csv',
        'jul25': 'jul25.csv',
        'aug25': 'aug25.csv',
        'sep25': 'sep25.csv',
        'oct25': 'oct25.csv',
        'nov25': 'nov25.csv',
        'dec25': 'dec25.csv'
    }
}

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / 'data_load_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():
    """Create database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Error as e:
        logger.error(f"Error connecting to database: {e}")
        raise


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataframe(df, required_columns, table_name):
    """Validate dataframe has required columns"""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"{table_name}: Missing columns: {missing_cols}")
        return False
    
    logger.info(f"{table_name}: All required columns present. Shape: {df.shape}")
    return True


def clean_column_names(df):
    """Clean column names - lowercase, remove spaces"""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df


# ============================================================
# LOAD REORDER LEVEL DATA
# ============================================================

def load_reorder_level():
    """Load reorder level master data"""
    logger.info("=" * 60)
    logger.info("LOADING REORDER LEVEL DATA")
    logger.info("=" * 60)
    
    file_path = DATA_DIR / FILES['reorder_level']
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    try:
        # Read CSV
        logger.info(f"Reading file: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df):,} rows")
        
        # Clean column names
        df = clean_column_names(df)
        logger.info(f"Columns: {list(df.columns)}")
        
        # Filter for CENTURY brand only
        if 'brand' in df.columns:
            df_century = df[df['brand'].str.upper() == 'CENTURY'].copy()
            logger.info(f"Filtered to CENTURY brand: {len(df_century):,} rows")
            df = df_century
        
        # Required columns
        required = ['item_code', 'shop_code']
        if not validate_dataframe(df, required, 'reorder_level'):
            return False
        
        # Map columns to database schema
        column_mapping = {
            'item_code': 'item_code',
            'item_name': 'item_name',
            'shop_code': 'shop_code',
            'item_code_shop': 'item_code_shop',
            'dept': 'dept',
            'brand': 'brand',
            'shop_stock': 'shop_stock',
            'wh_grn_shop_stock': 'wh_grn_shop_stock',
            'stc_nu': 'stc_nu',
            'min_nu': 'min_nu',
            'max_nu': 'max_nu',
            'reorder_qty': 'reorder_qty',
            'selling_price': 'selling_price'
        }
        
        # Select and rename columns
        available_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df_insert = df[list(available_cols.keys())].copy()
        df_insert.columns = list(available_cols.values())
        
        # Fill NaN values
        numeric_cols = ['shop_stock', 'wh_grn_shop_stock', 'stc_nu', 'min_nu', 'max_nu', 'reorder_qty', 'selling_price']
        for col in numeric_cols:
            if col in df_insert.columns:
                df_insert[col] = df_insert[col].fillna(0)
        
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Truncate table
        logger.info("Truncating reorder_level table...")
        cursor.execute("TRUNCATE TABLE reorder_level;")
        
        # Prepare insert query
        columns = list(df_insert.columns)
        placeholders = ', '.join(['%s'] * len(columns))
        insert_query = f"""
            INSERT INTO reorder_level ({', '.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT (item_code, shop_code) DO UPDATE SET
                {', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col not in ['item_code', 'shop_code']])}
        """
        
        # Batch insert
        logger.info("Inserting data...")
        data_tuples = [tuple(row) for row in df_insert.values]
        execute_batch(cursor, insert_query, data_tuples, page_size=1000)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Successfully loaded {len(df_insert):,} records into reorder_level")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading reorder_level: {e}")
        return False


# ============================================================
# LOAD SIT (STOCK IN TRANSIT) DATA
# ============================================================

def load_sit():
    """Load Stock In Transit data"""
    logger.info("=" * 60)
    logger.info("LOADING SIT DATA")
    logger.info("=" * 60)
    
    file_path = DATA_DIR / FILES['sit']
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False
    
    try:
        # Read CSV
        logger.info(f"Reading file: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df):,} rows")
        
        # Clean column names
        df = clean_column_names(df)
        logger.info(f"Columns: {list(df.columns)}")
        
        # Required columns
        required = ['shop_code', 'item_code']
        if not validate_dataframe(df, required, 'sit'):
            return False
        
        # Map columns
        column_mapping = {
            'shop_code': 'shop_code',
            'item_code': 'item_code',
            'dt_trans_date': 'dt_trans_date',
            'nu_transit_qty': 'nu_transit_qty'
        }
        
        # Select and rename columns
        available_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df_insert = df[list(available_cols.keys())].copy()
        df_insert.columns = list(available_cols.values())
        
        # Convert date column
        if 'dt_trans_date' in df_insert.columns:
            df_insert['dt_trans_date'] = pd.to_datetime(df_insert['dt_trans_date'], errors='coerce')
        
        # Fill NaN values
        if 'nu_transit_qty' in df_insert.columns:
            df_insert['nu_transit_qty'] = df_insert['nu_transit_qty'].fillna(0)
        
        # Remove rows with invalid dates
        df_insert = df_insert.dropna(subset=['dt_trans_date'])
        
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Truncate table
        logger.info("Truncating sit table...")
        cursor.execute("TRUNCATE TABLE sit;")
        
        # Prepare insert query
        columns = list(df_insert.columns)
        placeholders = ', '.join(['%s'] * len(columns))
        insert_query = f"""
            INSERT INTO sit ({', '.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT (shop_code, item_code, dt_trans_date) DO UPDATE SET
                nu_transit_qty = EXCLUDED.nu_transit_qty
        """
        
        # Batch insert
        logger.info("Inserting data...")
        data_tuples = [tuple(row) for row in df_insert.values]
        execute_batch(cursor, insert_query, data_tuples, page_size=1000)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Successfully loaded {len(df_insert):,} records into sit")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading sit: {e}")
        return False


# ============================================================
# LOAD SALES DATA (MULTIPLE MONTHLY FILES)
# ============================================================

def load_sales():
    """Load sales data from multiple monthly CSV files"""
    logger.info("=" * 60)
    logger.info("LOADING SALES DATA")
    logger.info("=" * 60)
    
    total_loaded = 0
    
    for month, filename in FILES['sales'].items():
        file_path = DATA_DIR / filename
        
        if not file_path.exists():
            logger.warning(f"File not found (skipping): {file_path}")
            continue
        
        try:
            # Read CSV
            logger.info(f"Reading {month}: {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df):,} rows for {month}")
            
            # Clean column names
            df = clean_column_names(df)
            
            # Required columns
            required = ['shop_code', 'item_code', 'date_invoice']
            if not validate_dataframe(df, required, f'sales_{month}'):
                continue
            
            # Map columns
            column_mapping = {
                'shop_code': 'shop_code',
                'item_code': 'item_code',
                'item_name': 'item_name',
                'dept': 'dept',
                'groups': 'groups',
                'sub_group': 'sub_group',
                'date_invoice': 'date_invoice',
                'qty': 'qty',
                'net_sales': 'net_sales'
            }
            
            # Select and rename columns
            available_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
            df_insert = df[list(available_cols.keys())].copy()
            df_insert.columns = list(available_cols.values())
            
            # Convert date column
            df_insert['date_invoice'] = pd.to_datetime(df_insert['date_invoice'], errors='coerce')
            
            # Fill NaN values
            numeric_cols = ['qty', 'net_sales']
            for col in numeric_cols:
                if col in df_insert.columns:
                    df_insert[col] = df_insert[col].fillna(0)
            
            # Remove rows with invalid dates
            df_insert = df_insert.dropna(subset=['date_invoice'])
            
            # Connect to database
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Prepare insert query
            columns = list(df_insert.columns)
            placeholders = ', '.join(['%s'] * len(columns))
            insert_query = f"""
                INSERT INTO sales ({', '.join(columns)})
                VALUES ({placeholders})
                ON CONFLICT (shop_code, item_code, date_invoice) DO UPDATE SET
                    qty = EXCLUDED.qty,
                    net_sales = EXCLUDED.net_sales
            """
            
            # Batch insert
            logger.info(f"Inserting {month} data...")
            data_tuples = [tuple(row) for row in df_insert.values]
            execute_batch(cursor, insert_query, data_tuples, page_size=1000)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            total_loaded += len(df_insert)
            logger.info(f"✅ Loaded {len(df_insert):,} records for {month}")
            
        except Exception as e:
            logger.error(f"❌ Error loading {month}: {e}")
            continue
    
    logger.info(f"✅ Total sales records loaded: {total_loaded:,}")
    return total_loaded > 0


# ============================================================
# REFRESH MATERIALIZED VIEWS
# ============================================================

def refresh_views():
    """Refresh all materialized views"""
    logger.info("=" * 60)
    logger.info("REFRESHING MATERIALIZED VIEWS")
    logger.info("=" * 60)
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Call refresh function
        cursor.execute("SELECT refresh_century_views();")
        result = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ {result}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error refreshing views: {e}")
        return False


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main execution function"""
    start_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info("CENTURY STOCK PENETRATION - DATA LOADING")
    logger.info(f"Started at: {start_time}")
    logger.info("=" * 60)
    
    # Load data in sequence
    success_reorder = load_reorder_level()
    success_sit = load_sit()
    success_sales = load_sales()
    
    # Refresh views if data loaded successfully
    if success_reorder or success_sales:
        refresh_views()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("=" * 60)
    logger.info("DATA LOADING COMPLETED")
    logger.info(f"Duration: {duration}")
    logger.info(f"Reorder Level: {'✅ Success' if success_reorder else '❌ Failed'}")
    logger.info(f"SIT: {'✅ Success' if success_sit else '❌ Failed'}")
    logger.info(f"Sales: {'✅ Success' if success_sales else '❌ Failed'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
