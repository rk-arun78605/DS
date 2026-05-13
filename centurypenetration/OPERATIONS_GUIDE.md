# Century Stock Penetration Dashboard - Operations Guide

## 📋 Table of Contents
1. [Initial Setup](#initial-setup)
2. [Daily Operations](#daily-operations)
3. [Data Upload Guide](#data-upload-guide)
4. [View Refresh Guide](#view-refresh-guide)
5. [Dashboard Launch](#dashboard-launch)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Initial Setup

### One-Time Database Setup
Run once when setting up the system for the first time:

```powershell
cd "d:\Dashboard Code\NO_WH\DS\centurypenetration"
python setup_database.py
```

This creates:
- Database: `century_penetration`
- Tables: `reorder_level`, `sit`, `sales` (with monthly partitions)
- Materialized Views: `mv_sales_metrics`, `mv_sit_summary`, `mv_century_penetration`
- Indexes for fast querying

### Initial Data Load
Load historical data (run once after database setup):

```powershell
cd "d:\Dashboard Code\NO_WH\DS\centurypenetration"
python load_century_data.py
```

**What it loads:**
- ✅ Reorder Level data from: `data/GEN_reorder_till10dec25.csv`
  - 6,653 CENTURY brand items
  - Min/Max/Reorder quantities
  - Selling prices
  
- ✅ SIT (Stock In Transit) from: `data/GEN_SIT_till10dec25.csv`
  - Transit records across all shops
  
- ✅ Sales History from: `data/*.csv` (jan25.csv to dec25.csv)
  - Monthly sales files for 2025
  - ~7 million transactions
  - Takes ~15 minutes to load

**Expected Duration:** 15-20 minutes

---

## 📅 Daily Operations

### 1. Upload Yesterday's Sales Data

**IMPORTANT:** Only upload **yesterday's sales**, not today's sales (today's data is incomplete until EOD).

**File Required:** `data/gendailysale.csv`

**What to include in gendailysale.csv:**
- Date: Yesterday's date only (e.g., if today is Dec 12, include only Dec 11 sales)
- Format: Same as monthly sales files
- Columns required:
  ```
  shop_code, item_code, date_invoice, quantity, price, amount, dept, brand, item_name
  ```

**Upload Command:**
```powershell
cd "d:\Dashboard Code\NO_WH\DS\centurypenetration"
python daily_update.py
```

**What daily_update.py does:**
1. ✅ Reads `data/gendailysale.csv`
2. ✅ Validates date columns
3. ✅ Routes data to correct monthly partition (e.g., Dec sales → sales_dec2025)
4. ✅ Uses **UPSERT** (ON CONFLICT DO UPDATE) - safe to run multiple times
5. ✅ Refreshes all materialized views automatically
6. ✅ Drops and recreates indexes to handle duplicates

**Expected Duration:** 2-5 minutes (depending on sales volume)

**Output:**
```
✅ Successfully loaded X records into sales partition
✅ Refreshing materialized views...
✅ All views refreshed successfully
```

---

## 📊 Data Upload Guide

### Sales Data Rules

#### ✅ DO:
- Upload **yesterday's sales only** in `gendailysale.csv`
- Run `daily_update.py` once per day (preferably in the morning)
- Keep file format consistent with column names
- Include all required columns

#### ❌ DON'T:
- Don't include today's sales (incomplete data)
- Don't upload future dates
- Don't run multiple times with same date (though upsert handles this)
- Don't modify column names in CSV

### Reorder Level Updates

**When to update:** When min/max/reorder quantities change

**File:** `data/GEN_reorder_till[date].csv`

**Command:**
```powershell
python load_century_data.py
```

**Note:** This does **TRUNCATE and reload** - replaces all existing data

### SIT (Stock In Transit) Updates

**When to update:** When transit data changes (weekly or as needed)

**File:** `data/GEN_SIT_till[date].csv`

**Command:**
```powershell
python load_century_data.py
```

---

## 🔄 View Refresh Guide

### When to Refresh Views

Refresh views when:
1. ✅ After uploading sales data (daily_update.py does this automatically)
2. ✅ After updating reorder levels manually
3. ✅ After updating SIT data manually
4. ✅ If dashboard shows stale data

### Manual View Refresh

**Method 1: Python Script (Recommended)**
```powershell
cd "d:\Dashboard Code\NO_WH\DS\centurypenetration"
python refresh_century_views.py
```

**What it does:**
1. Drops all indexes on materialized views
2. Refreshes `mv_sales_metrics` (sales calculations with yesterday-based date logic)
3. Refreshes `mv_sit_summary` (aggregated transit data)
4. Refreshes `mv_century_penetration` (main analytical view)
5. Recreates indexes (non-unique to handle duplicates)
6. Shows row counts for verification

**Expected Duration:** 3-5 minutes

**Method 2: SQL Script**
```powershell
cd "d:\Dashboard Code\NO_WH\DS\centurypenetration"
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d century_penetration -f refresh_views_complete.sql
```

### What Each View Contains

**mv_sales_metrics:**
- Sales calculations: Last 30d, 60d, 90d, 365d (yesterday-based)
- ROS (Rate of Sales) = Sales (Last 90d) ÷ 90
- Last sale date per item-shop combination
- **Date Logic:** Uses (Yesterday-30) to (Yesterday), NOT including today

**mv_sit_summary:**
- Total SIT per item-shop
- Latest transit date
- Transit count (number of transit records)

**mv_century_penetration:**
- Complete analytical view combining:
  - Reorder level data (min/max/reorder qty)
  - Sales metrics from mv_sales_metrics
  - SIT data from mv_sit_summary
- Calculated fields:
  - **Req 21 Days** = ROS × 21
  - **Stock Variance** = (SIH + SIT) - Req 21 Days
  - **Days of Stock** = (SIH + SIT) ÷ ROS
  - **Stock Status** = UnderStock/OverStock/Balanced

---

## 🖥️ Dashboard Launch

### Start the Dashboard

**Command:**
```powershell
cd "d:\Dashboard Code\NO_WH\DS\centurypenetration"
streamlit run centuryPenetration.py
```

**Or use the batch file:**
```powershell
.\run_dashboard.bat
```

**Default URL:** http://localhost:8501

### Dashboard Features

**6 Main Tabs:**

1. **📊 Shop Analysis**
   - Stock status distribution by shop
   - UnderStock/OverStock/Balanced counts
   - Stacked bar chart visualization

2. **⚠️ UnderStock Items**
   - Items below 21-day requirement
   - Sorted by largest shortage
   - Download CSV option

3. **📦 OverStock Items**
   - Items above 21-day requirement
   - Sorted by largest excess
   - Download CSV option

4. **🚨 Critical Items**
   - Zero stock + high demand
   - Requires immediate attention
   - Download CSV option

5. **🐌 Slow Moving Items**
   - Zero sales in last 90 days
   - High stock on hand
   - Download CSV option

6. **📁 Department Analysis**
   - Stock issues grouped by department
   - Grouped bar chart visualization

**Column Tooltips:**
- Hover over any column header to see formula
- Examples:
  - ROS: "Rate of Sales = Sales (Last 90 days) ÷ 90"
  - Req 21 Days: "ROS × 21 - Projected sales for next 3 weeks"
  - Stock Variance: "(SIH + SIT) - Req 21 Days"

---

## 🔍 Troubleshooting

### Issue: Dashboard shows zero rows

**Solution:**
```powershell
# Check if views have data
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d century_penetration -c "SELECT COUNT(*) FROM mv_century_penetration;"

# If zero, refresh views
python refresh_century_views.py
```

### Issue: Sales data not updating

**Check:**
1. Is `gendailysale.csv` in the `data/` folder?
2. Does it contain yesterday's date?
3. Run daily_update.py again:
   ```powershell
   python daily_update.py
   ```

### Issue: "Duplicate key" error during refresh

**Solution:** Already handled! The refresh scripts:
- Drop indexes before refresh
- Use regular REFRESH (not CONCURRENTLY)
- Recreate indexes as non-unique

If still failing:
```powershell
# Manual fix
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d century_penetration -f update_sales_date_logic.sql
```

### Issue: Python script fails with encoding error

**Known Issue:** Checkmark emojis (✅) in logs
**Impact:** Cosmetic only - data loads successfully
**Ignore:** Script continues and completes successfully

### Issue: Views showing wrong date range

**Verify date logic:**
```sql
-- Should show yesterday as end date
SELECT 
    CURRENT_DATE - INTERVAL '1 day' as yesterday,
    CURRENT_DATE - INTERVAL '31 days' as date_30d_start
FROM mv_sales_metrics LIMIT 1;
```

If wrong, run:
```powershell
python refresh_century_views.py
```

---

## 📝 Daily Checklist

**Every Morning:**
1. ✅ Prepare `gendailysale.csv` with yesterday's sales
2. ✅ Place file in `centurypenetration/data/` folder
3. ✅ Run: `python daily_update.py`
4. ✅ Wait 2-5 minutes for completion
5. ✅ Dashboard automatically shows updated data

**Weekly (Optional):**
1. Update SIT data if available
2. Update reorder levels if changed
3. Run full data load: `python load_century_data.py`

---

## 🎯 Quick Reference Commands

```powershell
# Navigate to project folder
cd "d:\Dashboard Code\NO_WH\DS\centurypenetration"

# Daily sales upload
python daily_update.py

# Manual view refresh
python refresh_century_views.py

# Full data reload (use sparingly)
python load_century_data.py

# Launch dashboard
streamlit run centuryPenetration.py

# Check view status
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d century_penetration -c "SELECT matviewname, pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) FROM pg_matviews WHERE schemaname='public';"
```

---

## 📞 Support

For issues not covered here:
1. Check log files in `data/data_load_log.txt`
2. Review error messages in terminal
3. Verify PostgreSQL is running on port 3307
4. Ensure Python packages installed: `pip install streamlit pandas psycopg2 plotly`

---

**Last Updated:** December 12, 2025
**Version:** 1.0
