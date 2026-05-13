# Century Penetration - Warehouse Stock Integration

## Overview
The Century Penetration dashboard now includes warehouse stock data from the `whstock` table. This data is synced from the `salesdata.whstock` table to `century_penetration.whstock`.

## Setup (One-Time)

### 1. Create whstock table in century_penetration database
```powershell
cd "d:\Dashboard Code\NO_WH\DS\centurypenetration"
python setup_whstock_table.py
```

This creates:
- `whstock` table with columns: vc_item_code, wh_code, wh_name, balance_qty, upload_date
- 5 indexes for optimal query performance

### 2. Initial data sync
```powershell
python sync_whstock_data.py
```

This copies the latest warehouse stock data from `salesdata.whstock` to `century_penetration.whstock`.

## Daily Maintenance

### Sync warehouse stock data (run after new data is uploaded to salesdata.whstock)
```powershell
cd "d:\Dashboard Code\NO_WH\DS\centurypenetration"
python sync_whstock_data.py
```

**Recommended schedule:**
- Run sync_whstock_data.py after uploading new WHStock CSV via home dashboard
- The script only syncs new data (checks upload_date to avoid duplicates)
- Takes ~1-2 seconds for typical dataset (400-500 records)

## Dashboard Features

### Warehouse Stock Columns (All 4 Tabs)
Each tab now displays 3 additional warehouse-related columns:

1. **🏭 WH Stock** - Total warehouse stock quantity across all warehouses
2. **# WH** - Number of warehouses that have stock for this item
3. **🏭 Warehouse Details** - Detailed breakdown by warehouse (WH Code: Name (Quantity))

### Tabs with Warehouse Integration:
- ⚠️ **UnderStock Items** - Items with shop stock < 70% of monthly sales + warehouse stock availability
- 📦 **OverStock Items** - Items with shop stock > 2x monthly sales + warehouse stock levels
- 🚨 **Critical Items** - Zero stock items with high demand + available warehouse stock
- 🐌 **Slow Moving Items** - Items with ROS < 0.5 + warehouse stock to monitor

## Technical Details

### Database Schema
```sql
-- century_penetration.whstock table
CREATE TABLE whstock (
    id SERIAL PRIMARY KEY,
    vc_item_code VARCHAR(50) NOT NULL,
    wh_code VARCHAR(20) NOT NULL,
    wh_name VARCHAR(100),
    balance_qty NUMERIC(15, 2) DEFAULT 0,
    upload_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Query Pattern
All queries use this CTE pattern to fetch latest warehouse stock:
```sql
WITH latest_whstock AS (
    SELECT DISTINCT ON (vc_item_code, wh_code)
        vc_item_code, wh_code, wh_name, balance_qty, upload_date
    FROM whstock
    WHERE upload_date = (SELECT MAX(upload_date) FROM whstock)
    ORDER BY vc_item_code, wh_code, upload_date DESC
)
SELECT 
    m.*,
    COALESCE(SUM(w.balance_qty), 0) as total_wh_stock,
    COUNT(DISTINCT w.wh_code) as wh_count,
    ARRAY_AGG(...) as wh_details
FROM mv_century_penetration m
LEFT JOIN latest_whstock w ON m.item_code = w.vc_item_code
GROUP BY ...
```

### Files
- `create_whstock_table.sql` - SQL schema for whstock table
- `setup_whstock_table.py` - Python script to create table
- `sync_whstock_data.py` - Python script to sync data from salesdata
- `centuryPenetration.py` - Main dashboard (updated with warehouse columns)

## Troubleshooting

### Data not showing in dashboard
1. Check if whstock table has data:
```sql
SELECT COUNT(*), MAX(upload_date) FROM century_penetration.whstock;
```

2. Verify latest upload_date matches salesdata:
```sql
-- In salesdata database
SELECT MAX(upload_date) FROM whstock;

-- In century_penetration database  
SELECT MAX(upload_date) FROM whstock;
```

3. If out of sync, run:
```powershell
python sync_whstock_data.py
```

### Sync script errors
- Ensure both databases are accessible on port 3307
- Check PostgreSQL connection credentials in sync_whstock_data.py
- Verify salesdata.whstock has data

### Dashboard cache issues
If warehouse columns don't show after sync:
1. Clear Streamlit cache: Click "C" in dashboard or restart
2. Verify query returns warehouse columns:
```python
# Run in Python console
from centuryPenetration import get_understock_items
df = get_understock_items()
print(df.columns)  # Should include total_wh_stock, wh_count, wh_details
```

## Performance Notes
- Warehouse stock queries use indexes on (vc_item_code, upload_date)
- DISTINCT ON ensures one row per item-warehouse combination
- LEFT JOIN preserves all Century items even without warehouse stock
- Typical query time: < 1 second for 50K records
