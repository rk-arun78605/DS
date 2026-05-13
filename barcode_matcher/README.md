# Barcode-to-Item Code Matcher System

## Overview
Daily barcode matching system where users upload CSV/Excel with **ONLY barcode column** and get item codes matched from master database.

## System Architecture

### 📁 File Structure
```
barcode_matcher/
├── barcode_matcher_app.py           # Streamlit app
├── README.md                         # This file
├── sql/
│   └── create_barcode_tables.sql    # Database schema
├── batch/
│   ├── setup_barcode_system_updated.bat  # One-time setup
│   ├── load_barcode_master.bat      # Daily master data update
│   ├── load_barcode_master.py       # Data loader script
│   └── run_barcode_matcher.bat      # Run application
└── sample_data/
    ├── barcodedata.csv              # Master data (daily updated)
    └── user_upload_sample.csv       # User upload example
```

## Database Design

### Master Table: `barcode_item_master`
```sql
Columns:
- vc_item_barcode VARCHAR(50)  -- Primary key
- vc_item_code VARCHAR(50)     -- Item code
- created_at TIMESTAMP
- updated_at TIMESTAMP
- is_active BOOLEAN
```

**Data Source:** `D:\Dashboard Code\NO_WH\DS\barcode_matcher\sample_data\barcodedata.*`

## Workflow

### 1️⃣ Daily Master Data Update (Automated)

**Source File:** `barcodedata.csv` or `barcodedata.xlsx`  
**Location:** `D:\Dashboard Code\NO_WH\DS\barcode_matcher\sample_data\`  
**Columns Required:** `VC_ITEM_BARCODE`, `VC_ITEM_CODE`

**Example barcodedata.csv:**
```csv
VC_ITEM_BARCODE,VC_ITEM_CODE
038841004018,D448
607481821827,1683
8886008101534,157596
```

**Run Daily Load:**
```bash
cd barcode_matcher\batch
load_barcode_master.bat
```

**What it does:**
- ✅ Reads barcodedata file (CSV or Excel)
- ✅ Compares with existing database
- ✅ Inserts ONLY NEW entries (no duplicates)
- ✅ Logs all operations to `barcode_load_log.txt`

**Smart Features:**
- Skips existing barcodes
- Handles both CSV and Excel
- Auto-detects file (barcodedata.csv or barcodedata.xlsx)
- Updates only new entries (fast)

### 2️⃣ User Upload & Matching

**User File:** Single column CSV/Excel  
**Column Required:** `barcode`

**Example user_upload.csv:**
```csv
barcode
038841004018
607481821827
INVALID_BARCODE_123
```

**Upload via App:**
1. Open dashboard: http://localhost:8503
2. Upload CSV/Excel with single `barcode` column
3. Get instant results with item codes

**Output:**
```csv
barcode,item_code,match_status
038841004018,D448,Matched
607481821827,1683,Matched
INVALID_BARCODE_123,,Not Found
```

**Key Points:**
- ✅ **ALL rows preserved** (matched + unmatched)
- ✅ **Blank item_code** for unmatched barcodes
- ✅ **match_status** shows Matched/Not Found
- ✅ Download as CSV or Excel

## Installation & Setup

### Prerequisites
- PostgreSQL 16+ (port 3307)
- Python 3.8+
- Packages: `streamlit pandas psycopg2 openpyxl`

### One-Time Setup

**Step 1: Install Python packages**
```bash
pip install streamlit pandas psycopg2 openpyxl
```

**Step 2: Create database & load initial data**
```bash
cd barcode_matcher\batch
setup_barcode_system_updated.bat
```

This will:
1. Create database tables
2. Load initial data from `barcodedata.*`
3. Verify setup

**Step 3: Run application**
```bash
run_barcode_matcher.bat
```

Dashboard opens at: **http://localhost:8503**

## Daily Operations

### Morning Routine (Automated)
```bash
# Update master data with new barcodes
cd barcode_matcher\batch
load_barcode_master.bat
```

**Recommended:** Schedule this as Windows Task (runs at 6 AM daily)

### User Operations (All Day)
1. Users upload barcode files via dashboard
2. Download results with item codes
3. All matched and unmatched barcodes preserved

## Application Features

### Tab 1: Upload & Match
- Upload CSV/Excel with single `barcode` column
- Instant matching (< 1 second for 1000 barcodes)
- Color-coded results (green=matched, red=not found)
- Download as CSV or Excel
- Save to database (audit trail)

### Tab 2: History
- View all past uploads
- Statistics: total rows, matched, unmatched
- Match rate percentage
- Date/time tracking

### Tab 3: Refresh Master Data
- Run daily load manually
- View load log
- Database statistics

## Usage Examples

### Example 1: Daily Master Update

**barcodedata.csv (updated daily by supplier):**
```csv
VC_ITEM_BARCODE,VC_ITEM_CODE
038841004018,D448
607481821827,1683
NEW_BARCODE_001,NEW_ITEM_001
```

**Run:**
```bash
load_barcode_master.bat
```

**Output:**
```
Found 3 rows in file
Existing: 2 barcodes
NEW: 1 entry to load
✅ Inserted 1 new barcode
Total database: 3 barcodes
```

### Example 2: User Upload

**User file (single column):**
```csv
barcode
038841004018
607481821827
INVALID123
NEW_BARCODE_001
```

**Upload via dashboard → Download result:**
```csv
barcode,item_code,match_status
038841004018,D448,Matched
607481821827,1683,Matched
INVALID123,,Not Found
NEW_BARCODE_001,NEW_ITEM_001,Matched
```

## Database Queries

### Check master table
```sql
SELECT COUNT(*) FROM barcode_item_master WHERE is_active = TRUE;
SELECT * FROM barcode_item_master LIMIT 10;
```

### View upload history
```sql
SELECT * FROM barcode_upload_history ORDER BY uploaded_at DESC LIMIT 10;
```

### Find unmatched barcodes
```sql
SELECT barcode, COUNT(*) as times_seen
FROM barcode_upload_details
WHERE match_status = 'Not Found'
GROUP BY barcode
ORDER BY times_seen DESC;
```

### Match rate by date
```sql
SELECT 
    DATE(uploaded_at) as date,
    SUM(total_rows) as total,
    SUM(matched_rows) as matched,
    ROUND(100.0 * SUM(matched_rows) / SUM(total_rows), 2) as match_rate_pct
FROM barcode_upload_history
GROUP BY DATE(uploaded_at)
ORDER BY date DESC;
```

## Performance

- **Master data load:** ~1000 entries/second
- **User matching:** <100ms for 1000 barcodes
- **Cache:** 1-hour TTL (auto-refresh)
- **Database:** Indexed lookups (O(1) complexity)

## Maintenance

### Add new barcodes manually
```sql
INSERT INTO barcode_item_master (vc_item_barcode, vc_item_code)
VALUES ('NEW_BARCODE', 'NEW_CODE')
ON CONFLICT (vc_item_barcode) DO UPDATE
SET vc_item_code = EXCLUDED.vc_item_code;
```

### Update existing mapping
```sql
UPDATE barcode_item_master
SET vc_item_code = 'NEW_CODE'
WHERE vc_item_barcode = 'EXISTING_BARCODE';
```

### Clean old upload history (90+ days)
```sql
DELETE FROM barcode_upload_history 
WHERE uploaded_at < CURRENT_DATE - INTERVAL '90 days';
```

## Troubleshooting

### Issue: No barcodedata file found
**Solution:** 
- Check file exists: `D:\Dashboard Code\NO_WH\DS\barcode_matcher\sample_data\barcodedata.*`
- File must be named `barcodedata.csv` or `barcodedata.xlsx`
- Check columns: `VC_ITEM_BARCODE`, `VC_ITEM_CODE`

### Issue: No matches found
**Solution:**
- Verify master data loaded: `SELECT COUNT(*) FROM barcode_item_master;`
- Check barcode format (case-sensitive, spaces)
- Run daily load: `load_barcode_master.bat`

### Issue: Slow performance
**Solution:**
- Clear cache in dashboard (Tab 3)
- Rebuild indexes: `REINDEX TABLE barcode_item_master;`
- Check database connection

### Issue: Database connection failed
**Solution:**
- Verify PostgreSQL running: `psql -U postgres -p 3307 -d salesdata`
- Check credentials in `barcode_matcher_app.py` (line 20-25)
- Ensure port 3307 is open

## Automation (Windows Task Scheduler)

### Schedule Daily Load at 6:00 AM

1. Open Task Scheduler
2. Create Basic Task
3. Name: "Barcode Master Daily Load"
4. Trigger: Daily at 6:00 AM
5. Action: Start Program
6. Program: `D:\Dashboard Code\NO_WH\DS\barcode_matcher\batch\load_barcode_master.bat`
7. Start in: `D:\Dashboard Code\NO_WH\DS\barcode_matcher\batch`

**Result:** Master data auto-updates every morning before users arrive

## File Format Requirements

### Master Data (barcodedata.csv)
```csv
VC_ITEM_BARCODE,VC_ITEM_CODE
<barcode1>,<item_code1>
<barcode2>,<item_code2>
```

**Requirements:**
- Must have EXACTLY 2 columns: `VC_ITEM_BARCODE`, `VC_ITEM_CODE`
- CSV or Excel format
- No empty rows
- Unique barcodes (duplicates auto-handled)

### User Upload File
```csv
barcode
<barcode1>
<barcode2>
<barcode3>
```

**Requirements:**
- Single column: `barcode`
- CSV or Excel format
- Max 10MB file size
- Any number of rows

## Support & Logs

### Check Logs
- **Load log:** `sample_data\barcode_load_log.txt`
- **App log:** Terminal output when running `run_barcode_matcher.bat`

### Statistics
```sql
-- Total barcodes in database
SELECT COUNT(*) FROM barcode_item_master WHERE is_active = TRUE;

-- Latest load time
SELECT MAX(created_at) FROM barcode_item_master;

-- Total uploads today
SELECT COUNT(*) FROM barcode_upload_history 
WHERE DATE(uploaded_at) = CURRENT_DATE;
```

---

**Version:** 2.0 (Daily Auto-Update)  
**Last Updated:** December 13, 2025  
**Database:** PostgreSQL 16 (port 3307)  
**Application Port:** 8503

## Features

### ✅ Core Functionality
- **Upload Excel/CSV** with barcode column
- **Instant matching** against master barcode database
- **Keep all rows** - matched and unmatched
- **Blank item codes** for non-matches
- **Download results** in CSV or Excel format

### 📊 Advanced Features
- **Upload history** - Track all uploads with statistics
- **Unmatched queue** - Track frequently unmatched barcodes
- **Manual mapping** - Add/update barcode mappings
- **Bulk mapping upload** - Upload multiple mappings at once
- **Match rate analytics** - See match percentages

### 🗄️ Database Architecture

#### Tables:
1. **barcode_item_master** - Master mapping table (barcode → item_code)
2. **barcode_upload_history** - Audit trail of all uploads
3. **barcode_upload_details** - Detailed results of each upload
4. **barcode_unmatched_queue** - Tracks unmatched barcodes for resolution

#### Views:
- **v_active_barcodes** - Active barcode mappings
- **v_recent_uploads** - Recent upload summary
- **v_unmatched_barcodes_pending** - Unmatched barcodes needing attention

#### Functions:
- **process_barcode_upload()** - Main matching function
- **add_barcode_mapping()** - Add new mapping
- **track_unmatched_barcode()** - Track unmatched occurrences

## Installation

### Prerequisites
- PostgreSQL 16+ (port 3307)
- Python 3.8+
- Streamlit

### Setup Steps

1. **Install Python packages:**
   ```bash
   pip install streamlit pandas psycopg2 openpyxl
   ```

2. **Setup database:**
   ```bash
   cd barcode_matcher\batch
   setup_barcode_system.bat
   ```
   
   This will:
   - Create all tables and functions
   - Load existing barcodes from `vc_item_barcode` table
   - Create indexes for performance

3. **Run the application:**
   ```bash
   run_barcode_matcher.bat
   ```
   
   Dashboard opens at: http://localhost:8503

## Usage Guide

### 1. Upload Barcode File

**File Requirements:**
- Must have column named `barcode` (or specify custom column name)
- Supported formats: CSV, XLSX, XLS
- Max file size: 10MB

**Example CSV format:**
```csv
barcode,quantity,location
038841004018,10,Warehouse A
607481821827,5,Store B
INVALID123,3,Store C
```

**Steps:**
1. Go to **Upload** tab
2. Enter barcode column name (default: "barcode")
3. Upload your file
4. View results instantly

### 2. Download Results

**Output Format:**
```csv
barcode_original,item_code,item_name,match_status,quantity,location
038841004018,D448,Product Name A,Matched,10,Warehouse A
607481821827,1683,Product Name B,Matched,5,Store B
INVALID123,,,Not Found,3,Store C
```

**Key Points:**
- ✅ **Matched rows** have item_code filled
- ❌ **Unmatched rows** have blank item_code
- **All original columns preserved**
- **match_status** shows Matched/Not Found

**Download Options:**
- CSV format (lightweight)
- Excel format (formatted)
- Save to database (for audit trail)

### 3. View History

Track all past uploads:
- Upload date/time
- File name
- Total rows, matched, unmatched
- Match rate percentage

### 4. Unmatched Barcodes

See barcodes that failed to match:
- Barcode value
- Times seen (occurrence count)
- First/last seen dates
- Days pending resolution

**Action Items:**
- Investigate with supplier
- Add missing mappings
- Fix data entry errors

### 5. Add Mappings

**Manual Entry:**
- Enter barcode
- Enter item code
- Optional: item name
- Click "Add Mapping"

**Bulk Upload:**
- Prepare CSV with columns: `barcode, item_code, item_name`
- Upload file
- All mappings added at once

## Database Queries

### Check master table:
```sql
SELECT COUNT(*) FROM barcode_item_master WHERE is_active = TRUE;
SELECT * FROM v_active_barcodes LIMIT 10;
```

### View recent uploads:
```sql
SELECT * FROM v_recent_uploads;
```

### Find unmatched barcodes:
```sql
SELECT * FROM v_unmatched_barcodes_pending;
```

### Get upload details:
```sql
SELECT * FROM barcode_upload_details WHERE upload_id = 1;
```

### Match rate by date:
```sql
SELECT 
    DATE(uploaded_at) AS upload_date,
    SUM(total_rows) AS total,
    SUM(matched_rows) AS matched,
    ROUND(100.0 * SUM(matched_rows) / SUM(total_rows), 2) AS match_rate_pct
FROM barcode_upload_history
GROUP BY DATE(uploaded_at)
ORDER BY upload_date DESC;
```

## Architecture Highlights

### Performance Optimizations:
- **Indexed lookups** - O(1) barcode matching
- **Cached master table** - 1-hour TTL in Streamlit
- **Bulk inserts** - execute_values() for upload details
- **Connection pooling** - Context managers for DB connections

### Data Integrity:
- **Unique constraint** on barcode (no duplicates)
- **Foreign keys** for referential integrity
- **Audit trail** - All uploads tracked
- **JSONB storage** - Original data preserved

### Scalability:
- Handles 100K+ barcodes in master table
- Upload files up to 10MB
- Concurrent user support
- Fast lookups (<100ms)

## Maintenance

### Add new barcodes from source:
```sql
INSERT INTO barcode_item_master (barcode, item_code, item_name)
SELECT DISTINCT
    TRIM(vc_item_barcode),
    TRIM(vc_item_code),
    NULL
FROM vc_item_barcode
WHERE vc_item_barcode IS NOT NULL
ON CONFLICT (barcode) DO NOTHING;
```

### Clean old upload history:
```sql
DELETE FROM barcode_upload_history 
WHERE uploaded_at < CURRENT_DATE - INTERVAL '90 days';
```

### Mark unmatched as resolved:
```sql
UPDATE barcode_unmatched_queue
SET resolved = TRUE,
    resolved_item_code = 'D448',
    resolved_at = CURRENT_TIMESTAMP,
    resolved_by = 'admin'
WHERE barcode = '038841004018';
```

## Troubleshooting

### No matches found:
- Check column name is correct
- Verify barcodes are in master table
- Check for leading/trailing spaces
- Ensure case-insensitive matching (auto-uppercased)

### Slow performance:
- Clear Streamlit cache: `st.cache_data.clear()`
- Rebuild indexes: `REINDEX TABLE barcode_item_master;`
- Run VACUUM ANALYZE

### Database connection issues:
- Verify PostgreSQL running on port 3307
- Check credentials in Config class
- Test connection: `psql -U postgres -p 3307 -d salesdata`

## Support

For issues or questions:
1. Check upload history for error messages
2. Review unmatched barcodes queue
3. Check PostgreSQL logs
4. Contact system administrator

---

**Version:** 1.0  
**Last Updated:** December 13, 2025  
**Database:** PostgreSQL 16 (port 3307)  
**Application Port:** 8503
