# NO_WH Inventory Pulse - Operations Guide

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Initial Setup](#initial-setup)
3. [Daily Operations](#daily-operations)
4. [Data Upload Guide](#data-upload-guide)
5. [View Refresh Guide](#view-refresh-guide)
6. [Dashboard Usage](#dashboard-usage)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 System Overview

**NO_WH Inventory Pulse** is a stock transfer recommendation system that identifies:
- ✅ Which items to transfer between shops
- ✅ How much quantity to transfer
- ✅ Where to transfer from (source shops)
- ✅ Where to transfer to (priority destination shops)

**Core Business Logic:**
- **11 Priority Shops** receive transfers: SPN, MSS, LFS, M03, KAS, MM1, MM2, FAR, KS7, WHL, MM3
- **Source Shops:** All other shops with excess stock (stock > 30-day sales)
- **Allocation:** Oldest stock first (FEFO - First Expiry First Out)
- **Safety Buffer:** Items expiring within 30 days are blocked from transfer

**Key Principle:** ALL business logic lives in the materialized view SQL. Python dashboard just queries and displays results.

---

## 🚀 Initial Setup

### One-Time Database Setup

**Database:** salesdata (shared with other dashboards)
**Port:** 3307

**Required Tables:**
1. `inventory_master` - Main inventory with stock levels
2. `sales_2024` / `sales_2025` - Sales transactions by year
3. `sup_shop_grn` - GRN dates (warehouse and shop level)
4. `shopexpiry` - Expiry dates per item+shop
5. `itemdetails` - Item master data

**Create Materialized View:**
```powershell
cd "d:\Dashboard Code\NO_WH\DS"
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -f NowhStock_mv_recommendations_complete.sql
```

This creates `mv_recommendations_complete` - the heart of the system (489-line SQL with complete recommendation logic).

### Initial Data Load

Load all CSV files into PostgreSQL:

```powershell
cd "d:\Dashboard Code\NO_WH\DS\nowhstock\batch"
python load_all_data.py
```

**Files loaded from project root:**
- `inventory_master.csv` - Current stock levels (all shops)
- `sales_2024.csv` - 2024 sales history
- `sales_2025.csv` - 2025 sales history
- `sup_shop_grn.csv` - GRN dates (WH GRN and Shop GRN)
- `shopexpiry.csv` - Expiry dates
- `itemdetails.csv` - Item master

**Expected Duration:** 5-10 minutes (depends on file sizes)

---

## 📅 Daily Operations

### Recommended Daily Workflow

**Every Morning (After EOD Batch):**

1. **Update inventory_master.csv** with yesterday's closing stock
2. **Update sales files** with yesterday's transactions
3. **Run data load** to refresh tables
4. **Refresh materialized view** to regenerate recommendations
5. **Dashboard automatically shows updated recommendations**

### Quick Daily Update Command

```powershell
cd "d:\Dashboard Code\NO_WH\DS\nowhstock\batch"

# Step 1: Load updated data
python load_all_data.py

# Step 2: Refresh materialized view
python refresh_all_views.py
```

**Total Time:** 8-12 minutes

---

## 📊 Data Upload Guide

### Files to Update Daily

#### 1. inventory_master.csv (REQUIRED DAILY)
**Location:** `d:\Dashboard Code\NO_WH\DS\inventory_master.csv`

**What to update:** Yesterday's closing stock for all items/shops

**Columns:**
```
itemcode, itemname, shopcode, shopstock, sales_30d_wh, shopgrn_dt, 
groupp, subgroup, importexport, suppliername
```

**Critical columns:**
- `shopstock` - Current stock level (must be up-to-date)
- `sales_30d_wh` - Sales from WH GRN date + 30 days (calculated)
- `shopgrn_dt` - Latest shop GRN date

#### 2. Sales Files (UPDATE AS NEEDED)

**sales_2025.csv** - Current year transactions
**sales_2024.csv** - Previous year (for year-over-year comparisons)

**Columns:**
```
shop, item_code, date_invoice, quantity, price, amount
```

**Rules:**
- Add new transactions daily or weekly
- Keep complete transaction history
- Used for 30-day sales calculations in view

#### 3. GRN Data (UPDATE WHEN NEW RECEIPTS)

**sup_shop_grn.csv**

**Columns:**
```
item_code, shop_code, wh_grn_date, shop_grn_date
```

**Two types of dates:**
- `wh_grn_date` - When warehouse received from supplier (item-level, same for all shops)
- `shop_grn_date` - When individual shop received from warehouse (shop-specific)

#### 4. Expiry Data (UPDATE MONTHLY)

**shopexpiry.csv**

**Columns:**
```
ITEM_CODE, SHOP_CODE, SHOP_EXPIRY_DATE
```

**Used for:**
- Blocking transfers of items expiring within 30 days
- FEFO allocation (oldest expiry first)

#### 5. Item Details (UPDATE RARELY)

**itemdetails.csv**

**Columns:**
```
item_code, item_name, group, subgroup, supplier, import_export
```

**Update when:** New items added or item attributes change

---

## 🔄 View Refresh Guide

### When to Refresh the Materialized View

Refresh `mv_recommendations_complete` after:
1. ✅ Daily data load (inventory/sales updates)
2. ✅ GRN data changes
3. ✅ Expiry data updates
4. ✅ If dashboard shows stale recommendations

### Refresh Methods

**Method 1: Python Script (Recommended)**
```powershell
cd "d:\Dashboard Code\NO_WH\DS\nowhstock\batch"
python refresh_all_views.py
```

**What it does:**
- Refreshes `mv_recommendations_complete`
- Logs refresh time and status
- Shows row count for verification

**Duration:** 2-4 minutes

**Method 2: SQL Script**
```powershell
cd "d:\Dashboard Code\NO_WH\DS\nowhstock\batch"
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -f refresh_all_views.sql
```

**Method 3: Windows Task Scheduler (Automated)**

Already configured in `refresh_views_scheduled.bat`

To set up:
1. Open Task Scheduler
2. Create new task: "Refresh NO_WH Views"
3. Trigger: Daily at 7:00 AM (after EOD batch completes)
4. Action: Run `d:\Dashboard Code\NO_WH\DS\nowhstock\batch\refresh_views_scheduled.bat`

---

## 🖥️ Dashboard Usage

### Launch the Dashboard

```powershell
cd "d:\Dashboard Code\NO_WH\DS"
streamlit run nowhstock_ds.py
```

**Or use batch file:**
```powershell
.\run_nowhstock_backup.bat
```

**Default URL:** http://localhost:8501

### Dashboard Features

#### **1. Login System**
- Username: Employee ID
- Password: Encrypted in `users` database
- Access control: Table-based permissions

#### **2. Main Filters (Left Sidebar)**

**Shop Selection:**
- Source Shop: Where to transfer FROM
- Destination Shop: Where to transfer TO (11 priority shops only)

**Item Filters:**
- Group: Product group (e.g., ELECTRICAL, GROCERIES)
- Sub Group: Product sub-category
- Item Code: Direct item search
- Item Name: Search by name
- Supplier: Filter by supplier

**Sales Threshold:**
- Fast-moving items: Sales ≥ threshold in 30 days
- Default: 30 units
- Adjust to filter by sales velocity

**Other Filters:**
- Import/Export: Item origin type
- Expiry Status: Safe, Expiring, No Expiry, All

#### **3. Recommendations Table**

**Main columns:**
- **Item Code/Name**: What to transfer
- **Source Shop**: Where from
- **Source Stock**: Available at source
- **Destination Shop**: Where to (with priority rank)
- **Allocated Qty**: How much to transfer
- **Dest Cap**: Destination capacity (max to receive)
- **Remaining Cap**: How much more destination can receive

**Sales Metrics:**
- **Dest Sales 30d**: Regular 30-day sales at destination
- **WH GRN +30d Sales**: Sales from WH GRN date to +30 days (demand spike after warehouse receipt)
- Destination cap = MAX of these two (uses higher demand signal)

**Stock Status:**
- **Source Expiry**: Days until expiry at source shop
- **GRN Age**: Days since shop received item
- **Block Reason**: Why transfer is blocked (if applicable)

#### **4. Summary Metrics (Top Cards)**

- **Total Items**: Unique items with recommendations
- **Total Quantity**: Sum of allocated quantities
- **Avg Allocated Qty**: Average per recommendation
- **Destinations Covered**: How many priority shops receive stock

#### **5. Visualizations**

**Top 10 Items by Quantity:**
- Bar chart of highest allocation items
- Shows which items have highest transfer volumes

**Allocation by Destination:**
- Shows how much each priority shop receives
- Helps balance distribution

**Allocation by Group:**
- Product group breakdown
- Identifies categories with most transfers

#### **6. Export Options**

**Download Filtered Recommendations:**
- CSV export of current filtered view
- Includes all columns
- Filename: `nowhstock_recommendations_YYYYMMDD_HHMMSS.csv`

---

## 🔍 Understanding Business Logic

### Priority Shop Allocation

**11 Priority Shops (in order):**
1. SPN (Spintex)
2. MSS (Mallam)
3. LFS (La Wireless)
4. M03 (Melcom 3)
5. KAS (Kasoa)
6. MM1 (Melcom 1)
7. MM2 (Melcom 2)
8. FAR (Farisco)
9. KS7 (Kumasi 7)
10. WHL (Wholesale)
11. MM3 (Melcom 3)

**Allocation Priority:**
- SPN gets first pick from available sources
- MM3 gets remaining capacity after others allocated
- Each destination processed independently (no competition between destinations)

### Source Shop Eligibility

**To be a SOURCE shop:**
✅ Must NOT be a priority shop (priority shops only receive)
✅ Must have stock > 30-day sales (excess stock criterion)
✅ Item must not be expiring within 30 days
✅ Item must have demand at destination (dest sales > 0)

**Example:**
- Shop KSI has 100 units of item ABC
- Sales in last 30 days: 20 units
- Excess stock: 100 - 20 = 80 units available for transfer

### Destination Capacity Calculation

**Formula:**
```
Destination Cap = MAX(dest_sales_30d, dest_wh_grn_30d_sales)
```

**Why use MAX?**
- Regular sales (dest_sales_30d) shows ongoing demand
- WH GRN +30d sales captures demand spike after warehouse receipt
- Take whichever is higher to avoid understocking during high demand

**Example:**
- Dest Sales 30d: 50 units (regular demand)
- WH GRN +30d Sales: 80 units (post-receipt spike)
- Cap = 80 units (use higher demand signal)

### Allocation Algorithm

**Step 1: Calculate Excess per Source**
```
Source Excess = Source Stock - Source Sales 30d
```

**Step 2: Rank Sources by FEFO**
```
ORDER BY:
  1. GRN Age DESC (oldest stock first)
  2. Expiry Days ASC (earliest expiry first)
  3. Source Priority Rank
```

**Step 3: Cumulative Allocation**
```sql
SUM(allocated_qty) OVER (
  PARTITION BY item_code, dest_shop
  ORDER BY grn_age DESC, expiry_days ASC
  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
) <= destination_cap
```

**Step 4: Cap Each Source Allocation**
```
Allocated Qty = LEAST(
  source_excess,
  destination_cap - cumulative_allocated
)
```

### Blocking Rules

**Transfers are BLOCKED when:**

1. **Same Shop Transfer**
   - Block Reason: "Same shop transfer"
   - Source = Destination not allowed

2. **Priority Shop as Source**
   - Block Reason: "Priority shop cannot be source"
   - Priority shops only receive, never give

3. **Expiry Check**
   - Block Reason: "Source expiring within 30 days"
   - Safety buffer to avoid transferring soon-to-expire stock

4. **No Demand**
   - Block Reason: "No demand at destination"
   - All three sales metrics = 0

5. **Cap Exceeded**
   - Block Reason: "Cap exceeded"
   - Cumulative allocation already reached destination capacity

### Expiry Handling

**Expiry Status:**
- **Safe**: > 30 days to expiry
- **Expiring**: < 30 days to expiry (blocks transfer)
- **No Expiry**: NULL expiry date (allowed to transfer)

**FEFO Sorting:**
```sql
ORDER BY:
  source_expiry_days ASC NULLS LAST
```
- Prioritizes oldest stock (shortest time to expiry)
- NULL expiry dates sorted last

---

## 🛠️ Troubleshooting

### Issue: Dashboard shows zero recommendations

**Check 1: View exists and has data**
```sql
SELECT COUNT(*) FROM mv_recommendations_complete;
```

If zero, refresh view:
```powershell
cd "d:\Dashboard Code\NO_WH\DS\nowhstock\batch"
python refresh_all_views.py
```

**Check 2: Filters too restrictive**
- Reset all filters to "All"
- Check sales threshold (try setting to 1)

**Check 3: Data loaded**
```sql
SELECT COUNT(*) FROM inventory_master;
SELECT COUNT(*) FROM sales_2025;
```

If zero, run data load:
```powershell
python load_all_data.py
```

### Issue: Recommendations seem incorrect

**Verify business logic in view:**
```sql
-- Check source eligibility
SELECT 
    item_code, 
    shop_code as source_shop,
    stock,
    sales_30d,
    stock - sales_30d as excess
FROM inventory_master
WHERE stock > sales_30d
  AND shop_code NOT IN ('SPN','MSS','LFS','M03','KAS','MM1','MM2','FAR','KS7','WHL','MM3')
LIMIT 10;
```

**Check destination caps:**
```sql
SELECT 
    item_code,
    dest_shop,
    dest_sales_30d,
    dest_wh_grn_30d_sales,
    GREATEST(dest_sales_30d, dest_wh_grn_30d_sales) as calculated_cap
FROM mv_recommendations_complete
WHERE allocated_qty > 0
LIMIT 10;
```

### Issue: View refresh takes too long (> 5 minutes)

**Solution 1: Check table sizes**
```sql
SELECT 
    relname as table_name,
    pg_size_pretty(pg_total_relation_size(relid)) as size
FROM pg_stat_user_tables
WHERE schemaname = 'public'
  AND relname IN ('inventory_master', 'sales_2024', 'sales_2025', 'sup_shop_grn', 'shopexpiry')
ORDER BY pg_total_relation_size(relid) DESC;
```

**Solution 2: Analyze tables**
```sql
ANALYZE inventory_master;
ANALYZE sales_2024;
ANALYZE sales_2025;
ANALYZE sup_shop_grn;
ANALYZE shopexpiry;
```

**Solution 3: Recreate view**
```powershell
cd "d:\Dashboard Code\NO_WH\DS"
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -f NowhStock_mv_recommendations_complete.sql
```

### Issue: Login fails

**Check users table:**
```sql
-- In users database
SELECT employee_id, is_active, table_access 
FROM "user" 
WHERE employee_id = 'YOUR_ID';
```

**Required permissions:**
- `is_active` = 'true' (lowercase string)
- `table_access` must include 'nowhstock' or 'all'

### Issue: Data load fails

**Check CSV file paths:**
Files must be in project root:
- `d:\Dashboard Code\NO_WH\DS\inventory_master.csv`
- `d:\Dashboard Code\NO_WH\DS\sales_2024.csv`
- etc.

**Check CSV format:**
- Proper headers matching table columns
- No extra quotes or special characters
- Dates in YYYY-MM-DD format

**Manual load per table:**
```powershell
# Edit load_all_data.py to comment out other tables
# Uncomment only the table you want to load
python load_all_data.py
```

### Issue: Expiry dates blocking all transfers

**Check expiry data quality:**
```sql
SELECT 
    "ITEM_CODE",
    "SHOP_CODE",
    "SHOP_EXPIRY_DATE",
    EXTRACT(YEAR FROM "SHOP_EXPIRY_DATE"::date) as expiry_year
FROM shopexpiry
WHERE EXTRACT(YEAR FROM "SHOP_EXPIRY_DATE"::date) > EXTRACT(YEAR FROM CURRENT_DATE) + 2
LIMIT 10;
```

If future dates are wrong, update `shopexpiry.csv` and reload.

---

## 📝 Daily Checklist

**Every Morning (After EOD Batch):**

1. ✅ Update `inventory_master.csv` with yesterday's closing stock
2. ✅ Update `sales_2025.csv` with yesterday's transactions (optional: weekly)
3. ✅ Place updated CSV files in: `d:\Dashboard Code\NO_WH\DS\`
4. ✅ Run: `cd nowhstock\batch` then `python load_all_data.py`
5. ✅ Wait 5-10 minutes for data load
6. ✅ Run: `python refresh_all_views.py`
7. ✅ Wait 2-4 minutes for view refresh
8. ✅ Dashboard automatically shows updated recommendations

**Weekly (Optional):**
1. Update `shopexpiry.csv` if new expiry data available
2. Update `sup_shop_grn.csv` if new GRN records
3. Run full data load

---

## 🎯 Quick Reference Commands

```powershell
# Navigate to batch folder
cd "d:\Dashboard Code\NO_WH\DS\nowhstock\batch"

# Load all data from CSV files
python load_all_data.py

# Refresh materialized view
python refresh_all_views.py

# Launch dashboard
cd "d:\Dashboard Code\NO_WH\DS"
streamlit run nowhstock_ds.py

# Check view status
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "SELECT COUNT(*) as total_recommendations FROM mv_recommendations_complete;"

# Check view refresh time
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -p 3307 -d salesdata -c "SELECT pg_size_pretty(pg_total_relation_size('mv_recommendations_complete')) as view_size;"
```

---

## 📊 Key Performance Indicators

**View Performance:**
- Refresh time: 2-4 minutes (normal)
- Row count: 5,000 - 50,000 recommendations (varies by stock levels)
- Query time: < 100ms (with proper filters)

**Data Freshness:**
- Inventory: Updated daily
- Sales: Updated daily or weekly
- GRN: Updated when new receipts
- Expiry: Updated monthly

---

## 📞 Support

**For issues not covered here:**
1. Check log files in `nowhstock\batch\data_load_log.txt`
2. Review error messages in terminal
3. Verify PostgreSQL is running on port 3307
4. Ensure all CSV files exist in correct locations
5. Check materialized view exists: `SELECT * FROM mv_recommendations_complete LIMIT 1;`

---

**Last Updated:** December 12, 2025  
**Version:** 2.3  
**Database:** salesdata (port 3307)
