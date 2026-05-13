'''
Daily Barcode Master Data Loader
Loads barcodedata file and updates only NEW entries
Source: D:/Dashboard Code/NO_WH/DS/barcode_matcher/sample_data/barcodedata
'''
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import os
from datetime import datetime
import glob

# ============================================================
# CONFIGURATION
# ============================================================

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'database': 'salesdata',
    'user': 'postgres',
    'password': 'hello'
}

DATA_FOLDER = "D:/Dashboard Code/NO_WH/DS/barcode_matcher/sample_data"
FILE_PATTERN = "barcodedata*"  # Matches barcodedata.xlsx, barcodedata.csv, etc.

LOG_FILE = os.path.join(DATA_FOLDER, "barcode_load_log.txt")

# ============================================================
# LOGGING
# ============================================================

def log(message):
    """Log message to console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_message + "\n")

# ============================================================
# CORE FUNCTIONS
# ============================================================

def find_barcode_file():
    """Find barcodedata file (xlsx or csv)"""
    search_pattern = os.path.join(DATA_FOLDER, FILE_PATTERN)
    files = glob.glob(search_pattern)
    
    if not files:
        return None
    
    # Return most recent file if multiple exist
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def read_barcode_file(filepath):
    """Read barcodedata file (supports CSV and Excel)"""
    try:
        if filepath.endswith('.csv'):
            # Try different encodings
            for encoding in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    log(f"✅ Read CSV with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                log("❌ Could not read CSV with any encoding")
                return None
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath)
            log("✅ Read Excel file")
        else:
            log(f"❌ Unsupported file format: {filepath}")
            return None
        
        # Validate required columns
        required_cols = ['VC_ITEM_BARCODE', 'VC_ITEM_CODE']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            log(f"❌ Missing columns: {missing_cols}")
            log(f"   Available columns: {df.columns.tolist()}")
            return None
        
        # Clean data
        df = df[['VC_ITEM_BARCODE', 'VC_ITEM_CODE']].copy()
        df['VC_ITEM_BARCODE'] = df['VC_ITEM_BARCODE'].astype(str).str.strip()
        df['VC_ITEM_CODE'] = df['VC_ITEM_CODE'].astype(str).str.strip()
        
        # Remove empty rows
        df = df[
            (df['VC_ITEM_BARCODE'] != '') & 
            (df['VC_ITEM_BARCODE'] != 'nan') &
            (df['VC_ITEM_CODE'] != '') & 
            (df['VC_ITEM_CODE'] != 'nan')
        ]
        
        # Remove duplicates (keep first occurrence)
        df = df.drop_duplicates(subset=['VC_ITEM_BARCODE'], keep='first')
        
        log(f"✅ Read {len(df)} valid rows from file")
        return df
    
    except Exception as e:
        log(f"❌ Error reading file: {str(e)}")
        return None

def get_existing_barcodes(conn):
    """Get set of existing barcodes from database"""
    cursor = conn.cursor()
    cursor.execute("SELECT vc_item_barcode FROM barcode_item_master WHERE is_active = TRUE")
    existing = {row[0] for row in cursor.fetchall()}
    cursor.close()
    return existing

def load_new_entries(df, conn):
    """Load only NEW barcode entries to database"""
    cursor = conn.cursor()
    
    # Get existing barcodes
    log("🔍 Checking for existing barcodes...")
    existing_barcodes = get_existing_barcodes(conn)
    log(f"   Found {len(existing_barcodes)} existing barcodes in database")
    
    # Filter to only new entries
    df_new = df[~df['VC_ITEM_BARCODE'].isin(existing_barcodes)].copy()
    
    if len(df_new) == 0:
        log("✅ No new entries to load. Database is up to date.")
        cursor.close()
        return 0, 0
    
    log(f"📦 Found {len(df_new)} NEW entries to load")
    
    # Prepare data for insert
    records = [
        (row['VC_ITEM_BARCODE'], row['VC_ITEM_CODE'])
        for _, row in df_new.iterrows()
    ]
    
    # Bulk insert using execute_values
    try:
        execute_values(
            cursor,
            """
            INSERT INTO barcode_item_master (vc_item_barcode, vc_item_code)
            VALUES %s
            ON CONFLICT (vc_item_barcode) DO NOTHING
            """,
            records
        )
        
        conn.commit()
        inserted = cursor.rowcount
        log(f"✅ Successfully inserted {inserted} new entries")
        
        cursor.close()
        return len(df_new), inserted
    
    except Exception as e:
        conn.rollback()
        log(f"❌ Error inserting data: {str(e)}")
        cursor.close()
        return len(df_new), 0

def update_statistics(conn):
    """Update and display database statistics"""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total_barcodes,
            COUNT(DISTINCT vc_item_code) as unique_items,
            MIN(created_at) as oldest_entry,
            MAX(created_at) as newest_entry
        FROM barcode_item_master
        WHERE is_active = TRUE
    """)
    
    stats = cursor.fetchone()
    
    log(f"\n📊 DATABASE STATISTICS:")
    log(f"   Total Barcodes: {stats[0]:,}")
    log(f"   Unique Item Codes: {stats[1]:,}")
    log(f"   Oldest Entry: {stats[2]}")
    log(f"   Newest Entry: {stats[3]}")
    
    cursor.close()

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main execution flow"""
    log("=" * 80)
    log("BARCODE MASTER DATA LOADER - DAILY UPDATE")
    log("=" * 80)
    
    # Step 1: Find barcode file
    log("\n[1/4] Locating barcodedata file...")
    barcode_file = find_barcode_file()
    
    if not barcode_file:
        log(f"❌ No barcodedata file found in: {DATA_FOLDER}")
        log(f"   Looking for pattern: {FILE_PATTERN}")
        return False
    
    log(f"✅ Found file: {barcode_file}")
    file_size = os.path.getsize(barcode_file) / 1024  # KB
    log(f"   File size: {file_size:.2f} KB")
    
    # Step 2: Read barcode file
    log("\n[2/4] Reading barcode file...")
    df = read_barcode_file(barcode_file)
    
    if df is None or len(df) == 0:
        log("❌ No valid data to load")
        return False
    
    log(f"✅ Loaded {len(df)} rows")
    log(f"   Sample barcodes: {df['VC_ITEM_BARCODE'].head(3).tolist()}")
    
    # Step 3: Connect to database
    log("\n[3/4] Connecting to database...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        log(f"✅ Connected to {DB_CONFIG['database']} on port {DB_CONFIG['port']}")
    except Exception as e:
        log(f"❌ Database connection failed: {str(e)}")
        return False
    
    # Step 4: Load new entries
    log("\n[4/4] Loading new entries to database...")
    new_count, inserted_count = load_new_entries(df, conn)
    
    if inserted_count > 0:
        log(f"✅ Load complete: {inserted_count} new barcodes added")
    elif new_count == 0:
        log("✅ Database already up to date")
    else:
        log("⚠️  Some entries may have failed to insert")
    
    # Display statistics
    update_statistics(conn)
    
    conn.close()
    
    log("\n" + "=" * 80)
    log("LOAD COMPLETED SUCCESSFULLY")
    log("=" * 80)
    
    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
