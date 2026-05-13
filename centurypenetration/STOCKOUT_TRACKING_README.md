# 📊 CENTURY PENETRATION - STOCKOUT TRACKING SYSTEM

## ✅ IMPLEMENTATION COMPLETED

### **System Overview**

A comprehensive stockout tracking system that records daily stock status and calculates stockout % at multiple dimensions.

---

## 📁 **Files Created**

1. **`create_stockout_tracking.sql`** - Database schema (tables + views)
2. **`daily_stockout_upsert.py`** - Daily UPSERT script
3. **`run_daily_stockout_update.bat`** - Windows batch runner
4. **Updated:** `test_centurypenetration_openai.py` - Dashboard with new Stockout % Analysis tab

---

## 🗄️ **Database Structure**

### **1. `mv_century_penetration_test` (Current State - UPSERT Daily)**

Replaces `mv_century_penetration` with additional tracking columns:

| Column | Type | Description |
|--------|------|-------------|
| `stock_out_date` | DATE | First date when SIH became ≤ 0 |
| `days_out_of_stock` | INT | Consecutive days item has been out |
| `ops_manager_name` | VARCHAR(100) | From `opsmgr` table (matched by shop_code) |
| `last_updated` | TIMESTAMP | Last update timestamp |

**UPSERT Logic:**
- If `sih <= 0` AND `stock_out_date IS NULL` → Set `stock_out_date = TODAY`
- If `sih <= 0` AND `stock_out_date EXISTS` → Keep existing date (first occurrence)
- If `sih > 0` → Reset `stock_out_date = NULL`

### **2. `century_stockout_daily_snapshot` (Historical - APPEND Daily)**

Daily snapshot for calculating stockout %:

| Column | Type | Description |
|--------|------|-------------|
| `snapshot_date` | DATE | Date of snapshot |
| `shop_code` | VARCHAR(50) | Shop code |
| `item_code` | VARCHAR(50) | Item code |
| `sih` | DECIMAL | Stock in hand |
| `is_out_of_stock` | BOOLEAN | TRUE if SIH ≤ 0 |
| `ops_manager_name` | VARCHAR(100) | Ops manager for shop |

### **3. `mv_stockout_analysis` (Materialized View - Refresh After Snapshot)**

Multi-dimensional stockout % analysis:

**Stockout % Formula:**
```
(Count of days item was out of stock / Total days tracked) × 100
```

**Dimensions:**
- ✅ Overall
- 👤 Ops Manager
- 🏪 Shop
- 📂 Department
- 📦 Item
- 📅 Month
- 📅 Year
- ⏱️ Last 7 Days
- ⏱️ Last 30 Days
- 📅 Weekly

### **4. `v_ops_manager_stockout_detail` (View - Real-time)**

Detailed ops manager stockout metrics per shop.

---

## 🔄 **Daily Update Process**

### **Automated Script** (`daily_stockout_upsert.py`)

```python
# 1. UPSERT current state from mv_century_penetration
# 2. APPEND today's snapshot to history table
# 3. REFRESH mv_stockout_analysis view
# 4. Log summary statistics
```

### **Run Manually:**
```bash
cd "d:\Dashboard Code\NO_WH\DS\centurypenetration"
python daily_stockout_upsert.py
```

### **Run via Batch:**
```bash
run_daily_stockout_update.bat
```

### **Schedule in Windows Task Scheduler:**
- **Trigger:** Daily at 1:00 AM
- **Action:** `d:\Dashboard Code\NO_WH\DS\centurypenetration\run_daily_stockout_update.bat`
- **Condition:** Run whether user is logged in or not

---

## 📊 **Dashboard Features**

### **New Tab: 📊 Stockout % Analysis**

**Sub-tabs:**

1. **👤 Ops Manager**
   - Stockout % by operations manager
   - Detailed table with shop breakdown
   - Tracks: Days tracked, total items, stockout events, items with stockout

2. **🏪 Shop**
   - Top 20 shops by stockout %
   - Full shop list with metrics

3. **📂 Department**
   - Stockout % by department
   - Identifies problem departments

4. **📦 Item**
   - Top 50 items by stockout %
   - Shows worst-performing items

5. **📅 Time-based**
   - Last 7 days vs Last 30 days vs All time
   - Monthly trend chart
   - Weekly trend (last 12 weeks)

---

## 📈 **Current Status**

✅ **Initial Load Completed:**
- Total Items: 7,210
- Current Stockouts: 373 (5.17%)
- Snapshot Days: 1 (will grow daily)

---

## 🔧 **Maintenance**

### **View Refresh**
```sql
REFRESH MATERIALIZED VIEW mv_stockout_analysis;
```

### **Check Snapshot Status**
```sql
SELECT 
    MIN(snapshot_date) as first_snapshot,
    MAX(snapshot_date) as last_snapshot,
    COUNT(DISTINCT snapshot_date) as total_days,
    COUNT(*) as total_records
FROM century_stockout_daily_snapshot;
```

### **Query Stockout % for Specific Manager**
```sql
SELECT * FROM v_ops_manager_stockout_detail
WHERE ops_manager_name = 'Frank'
ORDER BY stockout_pct DESC;
```

### **Query Monthly Trend**
```sql
SELECT level_value, stockout_pct
FROM mv_stockout_analysis
WHERE level_type = 'Month'
ORDER BY level_value;
```

---

## 🎯 **Key Business Rules**

1. **Stockout Definition:** Item is out of stock when `sih <= 0`
2. **Ops Manager Mapping:** Matched by `shop_code` → `opsmgr.shopcode`
3. **Date Range:** Uses yesterday as latest date (today's data incomplete)
4. **Historical Tracking:** Keeps daily snapshots indefinitely (can be purged after analysis)
5. **Multiple Levels:** Stockout % can be viewed at ANY aggregation level

---

## ⚠️ **Important Notes**

- **First Run:** Only 1 day of data, stockout % will stabilize after ~30 days
- **Run Daily:** Schedule the Python script to run daily at 1 AM
- **Table Reference:** Dashboard currently uses `mv_century_penetration` - will migrate to `mv_century_penetration_test` after testing
- **Performance:** Materialized view refresh takes ~1 second, queries are instant

---

## 🚀 **Next Steps**

1. ✅ Run daily for 7 days to accumulate data
2. ✅ Verify stockout % calculations are accurate
3. ✅ Monitor ops manager rankings
4. ✅ After validation, rename `_test` table to production
5. ✅ Update all SQL queries to use new table

---

## 📞 **Support**

- **Logs:** Check `daily_stockout_upsert.log` for update status
- **Database:** PostgreSQL on localhost:3307
- **Schema:** All objects in `century_penetration` database
