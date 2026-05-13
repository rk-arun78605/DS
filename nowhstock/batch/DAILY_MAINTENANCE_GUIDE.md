# ============================================================
# DAILY MAINTENANCE GUIDE - NO_WH INVENTORY SYSTEM
# ============================================================

## Overview
Daily maintenance script that refreshes all materialized views, updates table statistics, and verifies data freshness. Run every morning before business hours (recommended: 6:00 AM).

## Quick Start

### Manual Run (Recommended for first time)
```powershell
cd "d:\Dashboard Code\NO_WH\DS\nowhstock\batch"
$env:PGPASSWORD='hello'
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -f daily_maintenance.sql
```

### Batch File Run
```powershell
cd "d:\Dashboard Code\NO_WH\DS\nowhstock\batch"
.\run_daily_maintenance.bat
```

## What It Does

### Step 1: Analyze Base Tables (2-3 minutes)
Updates PostgreSQL statistics for query optimization:
- `inventory_master` - Current stock levels
- `sales_2024` / `sales_2025` - Sales transactions
- `sup_shop_grn` - GRN dates (warehouse + shop)
- `shopexpiry` - Product expiry tracking
- `itemdetails` - Item metadata

### Step 2: Refresh Materialized Views (5-10 minutes)
Refreshes all 8 materialized views:
1. **mv_recommendations_complete** (182 MB) - Main view with 489-line business logic
2. **mv_item_meta** (16 KB) - Item metadata lookup
3. **mv_last_grn_dates** (92 MB) - Latest GRN dates per item
4. **mv_last_shop_grn** (79 MB) - Latest shop GRN per item
5. **mv_last_wh_grn** (2.6 MB) - Latest warehouse GRN per item
6. **mv_latest_shop_stock** (79 MB) - Current shop stock levels
7. **mv_slow_fast_moving_summary** (206 MB) - Item velocity classification
8. **mv_transit** (2 MB) - Items in transit between shops

### Step 3: Analyze Materialized Views (1-2 minutes)
Updates statistics for each refreshed view

### Step 4: Reindex (Optional - COMMENTED OUT)
Rebuilds indexes to prevent bloat (recommended weekly, not daily)

### Step 5: Vacuum Analyze (Optional - COMMENTED OUT)
Deep table maintenance (recommended weekly during off-hours)

### Step 6: Display Statistics
Shows:
- View sizes (data + indexes)
- Table sizes and row counts
- Index usage statistics
- Database total size

### Step 7: Data Freshness Check
Verifies latest dates in key tables:
- ✅ Up-to-date: Data ≤1 day old
- ⚠️ Slightly stale: Data 2-3 days old
- ❌ Very stale: Data >3 days old (UPDATE REQUIRED)

## Execution Time
- **Normal (daily)**: 5-15 minutes
- **With reindex**: 20-30 minutes
- **With vacuum**: 30-60 minutes

## Schedule with Windows Task Scheduler

### Create Scheduled Task
1. Open Task Scheduler (`taskschd.msc`)
2. **Action** → **Create Task**
3. **General** tab:
   - Name: `NO_WH Daily Maintenance`
   - Description: `Refresh materialized views and analyze tables`
   - Run whether user is logged on or not: ✓
   - Run with highest privileges: ✓

4. **Triggers** tab → **New**:
   - Begin the task: `On a schedule`
   - Settings: `Daily`
   - Start: `6:00:00 AM`
   - Recur every: `1 days`
   - Enabled: ✓

5. **Actions** tab → **New**:
   - Action: `Start a program`
   - Program/script: `d:\Dashboard Code\NO_WH\DS\nowhstock\batch\run_daily_maintenance.bat`
   - Start in: `d:\Dashboard Code\NO_WH\DS\nowhstock\batch`

6. **Conditions** tab:
   - Uncheck: `Start the task only if the computer is on AC power`
   - Check: `Wake the computer to run this task`

7. **Settings** tab:
   - Allow task to be run on demand: ✓
   - If the task fails, restart every: `10 minutes`
   - Attempt to restart up to: `3 times`

### Test Scheduled Task
```powershell
# Run manually from Task Scheduler
schtasks /Run /TN "NO_WH Daily Maintenance"
```

## Monitoring

### Check Last Run Status
```powershell
cd "d:\Dashboard Code\NO_WH\DS\nowhstock\batch"
Get-Content daily_maintenance_log.txt -Tail 50
```

### Check Last Refresh Time
```sql
SELECT 
    matviewname,
    pg_size_pretty(pg_total_relation_size('public.'||matviewname)) as size
FROM pg_matviews 
WHERE schemaname = 'public'
ORDER BY matviewname;
```

### Verify Data Freshness
```sql
SELECT 
    'sales_2025' as table_name,
    MAX("DATE_INVOICE")::date as latest_date,
    CURRENT_DATE - MAX("DATE_INVOICE")::date as days_old
FROM sales_2025;
```

## Troubleshooting

### Issue: Script takes too long (>20 minutes)
**Cause**: Large data volume or slow disk I/O
**Solution**:
1. Run during off-peak hours (2:00 AM - 5:00 AM)
2. Consider upgrading to SSD storage
3. Use `REFRESH MATERIALIZED VIEW CONCURRENTLY` (requires unique index)

### Issue: "mv_recommendations_complete does not exist"
**Cause**: Main view not created yet
**Solution**:
```powershell
psql -U postgres -p 3307 -d salesdata -f NowhStock_mv_recommendations_complete.sql
```

### Issue: Out of memory errors
**Cause**: Insufficient `work_mem` or `maintenance_work_mem`
**Solution**:
```sql
-- Increase temporarily for maintenance
SET maintenance_work_mem = '2GB';
SET work_mem = '512MB';
```

### Issue: Lock timeout errors
**Cause**: Streamlit app still reading from views
**Solution**:
1. Stop Streamlit app: `Ctrl+C` in terminal
2. Run maintenance script
3. Restart Streamlit: `streamlit run nowhstock_ds.py`

### Issue: Data freshness shows "Very stale"
**Cause**: Data not loaded from source system
**Solution**:
```powershell
cd "d:\Dashboard Code\NO_WH\DS\nowhstock\batch"
python load_all_data.py  # Load latest CSV files
.\run_daily_maintenance.bat  # Then refresh views
```

## Weekly Deep Maintenance (Recommended)

Run once per week (Sunday 2:00 AM) with full vacuum and reindex:

### Edit daily_maintenance.sql
Uncomment these sections:
```sql
-- STEP 4: REINDEX CRITICAL INDEXES
REINDEX INDEX CONCURRENTLY idx_mv_recs_complete_dest_shop;
REINDEX INDEX CONCURRENTLY idx_mv_recs_complete_item_code;
-- ... (other indexes)

-- STEP 5: VACUUM ANALYZE
VACUUM ANALYZE inventory_master;
VACUUM ANALYZE sales_2025;
-- ... (other tables)
```

### Create separate weekly script
```powershell
# Copy daily_maintenance.sql to weekly_maintenance.sql
cp daily_maintenance.sql weekly_maintenance.sql

# Uncomment STEP 4 and STEP 5 in weekly_maintenance.sql
# Schedule as separate task: "NO_WH Weekly Maintenance" at 2:00 AM Sundays
```

## Integration with Data Load Process

Recommended workflow:
1. **5:30 AM**: Load data from source (`load_all_data.py`)
2. **6:00 AM**: Run daily maintenance (`daily_maintenance.sql`)
3. **6:20 AM**: Restart Streamlit app (if running as service)
4. **6:30 AM**: System ready for business

### Combined batch file
```batch
@echo off
REM Load data then refresh views
cd /d "d:\Dashboard Code\NO_WH\DS\nowhstock\batch"

echo Loading data...
python load_all_data.py

echo Refreshing views...
run_daily_maintenance.bat

echo Done! System ready.
pause
```

## Performance Tips

1. **Concurrent refresh**: Use `CONCURRENTLY` keyword (requires unique index)
   ```sql
   REFRESH MATERIALIZED VIEW CONCURRENTLY mv_recommendations_complete;
   ```

2. **Partial refresh**: For large views, consider incremental updates
   ```sql
   -- Delete old data, insert new data
   DELETE FROM mv_recommendations_complete WHERE source_date < CURRENT_DATE - 90;
   INSERT INTO mv_recommendations_complete SELECT * FROM ...;
   ```

3. **Parallel execution**: Increase `max_parallel_workers` in postgresql.conf
   ```conf
   max_parallel_workers = 8
   max_parallel_maintenance_workers = 4
   ```

4. **Monitor progress**: Check `pg_stat_progress_create_index` view
   ```sql
   SELECT * FROM pg_stat_progress_create_index;
   ```

## Log Files

- `daily_maintenance_log.txt` - Last run output
- `refresh_views_log.txt` - Historical log (append mode)
- PostgreSQL logs: `C:\Program Files\PostgreSQL\18\data\log\`

## Support

For issues:
1. Check log files for error messages
2. Verify PostgreSQL service is running: `Get-Service postgresql*`
3. Test database connection: `psql -U postgres -p 3307 -d salesdata -c "SELECT version();"`
4. Check disk space: `Get-PSDrive C | Select-Object Used,Free`

---

**Last Updated**: December 12, 2025  
**Database**: salesdata (PostgreSQL 16, port 3307)  
**Application**: NO_WH Inventory Pulse Dashboard
