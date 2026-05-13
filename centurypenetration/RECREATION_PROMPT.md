# Century Stock Penetration Dashboard - Recreation Prompt

## 🎯 Purpose
This file contains the complete prompt to recreate the Century Stock Penetration Dashboard system from scratch using AI assistance.

---

## 📝 Complete Prompt for AI Agent

```
I need you to create a complete Century Stock Penetration Dashboard system with the following specifications:

### SYSTEM OVERVIEW
Build a Streamlit dashboard to analyze CENTURY brand stock positioning across multiple shops, identify understock/overstock situations, and calculate reorder requirements.

### DATABASE ARCHITECTURE

**PostgreSQL 16 Setup:**
- Port: 3307
- Database name: century_penetration
- User: postgres
- Password: hello

**Tables Required:**

1. **reorder_level** (Main inventory table)
   - item_code VARCHAR(50)
   - item_name VARCHAR(200)
   - shop_code VARCHAR(50)
   - dept VARCHAR(100)
   - brand VARCHAR(100)
   - shop_grn_date DATE
   - wh_grn_date DATE
   - shop_stock DECIMAL(15,2) -- Stock In Hand (SIH)
   - min_nu DECIMAL(15,2) -- Minimum stock level
   - max_nu DECIMAL(15,2) -- Maximum stock level
   - reorder_qty DECIMAL(15,2) -- Reorder quantity
   - selling_price DECIMAL(15,2)

2. **sit** (Stock In Transit)
   - shop_code VARCHAR(50)
   - item_code VARCHAR(50)
   - transit_date DATE
   - transit_qty DECIMAL(15,2)

3. **sales** (Partitioned by month)
   - Parent table with monthly partitions: sales_jan2025, sales_feb2025, etc.
   - shop_code VARCHAR(50)
   - item_code VARCHAR(50)
   - date_invoice DATE
   - quantity DECIMAL(15,2)
   - price DECIMAL(15,2)
   - amount DECIMAL(15,2)
   - dept VARCHAR(100)
   - brand VARCHAR(100)
   - item_name VARCHAR(200)
   - Partition key: date_invoice
   - Partitions: Create 12 monthly partitions for 2025 (jan2025 to dec2025)

**CRITICAL DATE CALCULATION LOGIC:**
All sales calculations MUST use yesterday-based date ranges:
- Last 30d = Sales from (CURRENT_DATE - 31 days) to (CURRENT_DATE - 1 day)
- Last 60d = Sales from (CURRENT_DATE - 61 days) to (CURRENT_DATE - 1 day)
- Last 90d = Sales from (CURRENT_DATE - 91 days) to (CURRENT_DATE - 1 day)
- Last 365d = Sales from (CURRENT_DATE - 366 days) to (CURRENT_DATE - 1 day)

**Rationale:** Today's sales data is incomplete until end-of-day batch runs, so always exclude today.

**Materialized Views Required:**

1. **mv_sales_metrics** (Sales analysis per item-shop)
   ```sql
   Columns:
   - item_code, shop_code
   - sales_30d (quantity sum for last 30 days from yesterday)
   - sales_60d (quantity sum for last 60 days from yesterday)
   - sales_90d (quantity sum for last 90 days from yesterday)
   - sales_365d (quantity sum for last 365 days from yesterday)
   - value_30d (amount sum for last 30 days)
   - value_60d (amount sum for last 60 days)
   - value_90d (amount sum for last 90 days)
   - value_365d (amount sum for last 365 days)
   - ros (Rate of Sales = sales_90d / 90)
   - last_sale_date (MAX date_invoice)
   - refreshed_at TIMESTAMP
   
   Join sales table with date ranges using BETWEEN for yesterday-based calculations.
   ```

2. **mv_sit_summary** (Transit aggregation per item-shop)
   ```sql
   Columns:
   - shop_code, item_code
   - total_sit (SUM of transit_qty)
   - latest_transit_date (MAX transit_date)
   - transit_count (COUNT of records)
   - refreshed_at TIMESTAMP
   ```

3. **mv_century_penetration** (Main analytical view - CENTURY brand only)
   ```sql
   Combines:
   - reorder_level (base table)
   - mv_sales_metrics (LEFT JOIN)
   - mv_sit_summary (LEFT JOIN)
   
   Additional calculated columns:
   - sih (shop_stock from reorder_level)
   - sit (total_sit from mv_sit_summary, default 0)
   - total_stock (sih + sit)
   - sales_30d, sales_60d, sales_90d, sales_365d (from mv_sales_metrics)
   - value_30d, value_60d, value_90d, value_365d
   - ros (from mv_sales_metrics)
   - req_21_days = ROUND(ros * 21, 2)
   - stock_variance = ROUND((sih + sit) - (ros * 21), 2)
   - stock_status = CASE 
       WHEN stock_variance > 0 THEN 'OverStock'
       WHEN stock_variance < 0 THEN 'UnderStock'
       ELSE 'Balanced'
     END
   - days_of_stock = ROUND((sih + sit) / ros, 1) when ros > 0, else NULL
   - last_sale_date, latest_transit_date
   - refreshed_at TIMESTAMP
   
   Filter: WHERE UPPER(brand) = 'CENTURY'
   ```

**Indexes:**
- Create indexes on all materialized views for: item_code, shop_code, dept, stock_status
- Use NON-UNIQUE indexes (source data may have duplicates)
- Indexes on mv_century_penetration:
  - idx_century_item_shop (item_code, shop_code)
  - idx_century_shop (shop_code)
  - idx_century_dept (dept)
  - idx_century_status (stock_status)
  - idx_century_ros (ros) WHERE ros > 0
  - idx_century_understock (stock_variance) WHERE stock_status = 'UnderStock'
  - idx_century_overstock (stock_variance) WHERE stock_status = 'OverStock'

### PYTHON SCRIPTS REQUIRED

**1. setup_database.py**
- Creates database century_penetration
- Creates all tables with proper data types
- Creates monthly partitions for sales table
- Creates all three materialized views
- Creates indexes
- Includes error handling and logging

**2. load_century_data.py**
- Loads reorder level from: data/GEN_reorder_till10dec25.csv
  - Filter to CENTURY brand only
  - TRUNCATE and reload strategy
- Loads SIT from: data/GEN_SIT_till10dec25.csv
  - TRUNCATE and reload strategy
- Loads monthly sales from: data/jan25.csv, data/feb25.csv, etc.
  - Route to correct partition based on date_invoice
  - Use execute_batch for performance
  - TRUNCATE each partition before loading
- Refresh all materialized views after loading
- Progress logging and error handling
- Create data_load_log.txt

**3. daily_update.py**
- Reads: data/gendailysale.csv (yesterday's sales only)
- Determines target partition from date_invoice
- Uses UPSERT: ON CONFLICT (shop_code, item_code, date_invoice) DO UPDATE
- Automatically refreshes all materialized views after insert
- Drops indexes before refresh, recreates after (to handle duplicates)
- Safe to run multiple times

**4. refresh_century_views.py**
- Step 1: Drop all indexes on materialized views
- Step 2: Refresh mv_sales_metrics (without CONCURRENTLY)
- Step 3: Refresh mv_sit_summary
- Step 4: Refresh mv_century_penetration
- Step 5: Recreate all indexes as NON-UNIQUE
- Step 6: Display row counts for verification
- Logging and error handling

**5. centuryPenetration.py** (Streamlit Dashboard)

**Configuration:**
```python
class Config:
    DB_CONFIG = {
        'host': 'localhost',
        'port': 3307,
        'user': 'postgres',
        'password': 'hello',
        'database': 'century_penetration'
    }
    PAGE_TITLE = "Century Stock Penetration"
    PAGE_ICON = "📊"
    COLORS = {
        'UnderStock': '#f5576c',
        'OverStock': '#667eea',
        'Balanced': '#00d2ff'
    }
```

**Connection Pattern:**
```python
@st.cache_resource
def get_connection_pool():
    return psycopg2.pool.SimpleConnectionPool(minconn=1, maxconn=5, **Config.DB_CONFIG)

@contextmanager
def get_db_connection():
    pool = get_connection_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)
```

**Data Functions (all with @st.cache_data(ttl=600)):**

1. get_overview_metrics()
   - Query mv_century_penetration
   - Return: total_items, total_shops, understock_items, overstock_items, balanced_items, total_sih, total_sit, total_requirement, avg_ros

2. get_shop_wise_summary()
   - Query mv_century_penetration grouped by shop_code
   - Count items by stock_status
   - Return: shop_code, total_items, understock, overstock, balanced, total_sih, total_sit

3. get_understock_items(limit=100)
   - Query mv_century_penetration WHERE stock_status = 'UnderStock'
   - ORDER BY stock_variance ASC (most negative first)
   - Return: item_code, item_name, shop_code, dept, sih, sit, sales_30d, sales_60d, sales_90d, ros, req_21_days, stock_variance, days_of_stock

4. get_overstock_items(limit=100)
   - Query mv_century_penetration WHERE stock_status = 'OverStock'
   - ORDER BY stock_variance DESC (most positive first)
   - Return: same columns as understock

5. get_critical_items(limit=50)
   - Query mv_century_penetration WHERE sih = 0 AND sit = 0 AND ros > 0
   - ORDER BY ros DESC
   - Return: item_code, item_name, shop_code, dept, sih, sit, sales_30d, sales_60d, sales_90d, ros, req_21_days

6. get_slow_moving_items(limit=100)
   - Query mv_century_penetration WHERE sales_90d = 0 AND (sih > 0 OR sit > 0)
   - ORDER BY (sih + sit) DESC
   - Return: item_code, item_name, shop_code, dept, sih, sit, days_of_stock, last_sale_date

7. search_items(search_term)
   - Query mv_century_penetration WHERE item_code ILIKE '%term%' OR item_name ILIKE '%term%'
   - Return all columns

8. get_department_analysis()
   - Query mv_century_penetration grouped by dept
   - Count by stock_status
   - Return: dept, total_items, understock, overstock, balanced, avg_ros, total_requirement

**Column Configuration Function:**
Create get_column_config() that returns st.column_config with tooltips:
- sih: "📦 Stock In Hand - Current shop inventory"
- sit: "🚚 Stock In Transit - Items en route to shop"
- total_stock: "📊 Total Stock = SIH + SIT"
- sales_30d: "📈 Sales (Yesterday-30 to Yesterday) - Last 30 days sales excluding today"
- sales_60d: "📈 Sales (Yesterday-60 to Yesterday) - Last 60 days sales excluding today"
- sales_90d: "📈 Sales (Yesterday-90 to Yesterday) - Last 90 days sales excluding today"
- ros: "⚡ Rate of Sales = Sales (Last 90 days) ÷ 90 - Average daily sales rate"
- req_21_days: "🎯 Requirement for 21 Days = ROS × 21 - Projected sales for next 3 weeks"
- stock_variance: "📊 Stock Variance = (SIH + SIT) - Req 21 Days - Positive = OverStock, Negative = UnderStock"
- days_of_stock: "📅 Days of Stock = (SIH + SIT) ÷ ROS - How many days current stock will last"
- stock_status: "⚠️ Stock Status: UnderStock (below 21d req), OverStock (above 21d req), Balanced (at target)"

**Dashboard Layout:**

Header:
- Title: "Century Stock Penetration Dashboard"
- Logo: Melcom logo
- Search bar at top (filter by item_code or item_name)

Overview Section:
- 5 metric cards in columns:
  - Total Items, Total Shops, UnderStock (red), OverStock (blue), Balanced (cyan)
  - Show percentages as delta

Main Tabs (if no search):
1. **📊 Shop Analysis**
   - Call get_shop_wise_summary()
   - Display stacked bar chart (plotly): understock/overstock/balanced per shop
   - Data table below chart
   - Download CSV button

2. **⚠️ UnderStock Items**
   - Call get_understock_items()
   - Display dataframe with column_config
   - Download CSV button

3. **📦 OverStock Items**
   - Call get_overstock_items()
   - Display dataframe with column_config
   - Download CSV button

4. **🚨 Critical Items**
   - Call get_critical_items()
   - Warning message: "X critical items require immediate attention!"
   - Display dataframe with column_config
   - Download CSV button

5. **🐌 Slow Moving Items**
   - Call get_slow_moving_items()
   - Info message about zero sales in 90 days
   - Display dataframe with column_config
   - Download CSV button

6. **📁 Department Analysis**
   - Call get_department_analysis()
   - Display grouped bar chart (plotly): understock/overstock per department
   - Data table below chart
   - Download CSV button

If search term provided:
- Display search results in single dataframe
- Show all columns from mv_century_penetration
- Download CSV button

All dataframes must use: `st.dataframe(df, use_container_width=True, height=500, column_config=get_column_config())`

**Additional Files:**

6. **refresh_views_complete.sql**
   - SQL script for manual refresh via psql
   - Drop indexes, refresh views, recreate indexes, show counts

7. **update_sales_date_logic.sql**
   - Script to update mv_sales_metrics definition if date logic needs fixing
   - Drop CASCADE, recreate with yesterday-based logic, recreate indexes

8. **run_dashboard.bat**
   ```batch
   @echo off
   cd "d:\Dashboard Code\NO_WH\DS\centurypenetration"
   streamlit run centuryPenetration.py
   pause
   ```

### DATA FILES STRUCTURE

Expected in `data/` folder:
- GEN_reorder_till10dec25.csv (reorder level - all brands)
- GEN_SIT_till10dec25.csv (stock in transit)
- jan25.csv, feb25.csv, mar25.csv, ... dec25.csv (monthly sales)
- gendailysale.csv (daily sales for yesterday only)

All CSV files must have proper headers matching table columns.

### KEY BUSINESS RULES

1. **Date Logic:** Always use yesterday as end date, never include today's sales
2. **ROS Calculation:** Based on 90-day rolling window (yesterday-90 to yesterday)
3. **21-Day Requirement:** Target stock level = ROS × 21 days
4. **Stock Status:**
   - UnderStock: (SIH + SIT) < Req 21 Days
   - OverStock: (SIH + SIT) > Req 21 Days
   - Balanced: (SIH + SIT) = Req 21 Days
5. **CENTURY Filter:** Only show items where UPPER(brand) = 'CENTURY'
6. **Upsert Strategy:** Daily updates use ON CONFLICT to prevent duplicates
7. **Index Strategy:** Non-unique indexes to handle source data duplicates

### PERFORMANCE OPTIMIZATIONS

1. Use connection pooling (SimpleConnectionPool)
2. Cache data functions with 10-minute TTL
3. Use execute_batch for bulk inserts
4. Drop indexes before view refresh, recreate after
5. Materialized views for pre-aggregation
6. Partitioned sales table by month

### ERROR HANDLING

1. All scripts must have try-except blocks
2. Log errors to data_load_log.txt
3. Provide helpful error messages
4. Handle missing files gracefully
5. Handle duplicate key scenarios

### DELIVERABLES

Please create all files with:
- Complete working code
- Proper error handling
- Logging
- Comments explaining business logic
- SQL scripts fully functional
- Batch files for easy execution
```

---

## 🔧 Additional Context for AI

**Database Type:** PostgreSQL 16  
**Python Version:** 3.11+  
**Key Libraries:** streamlit, pandas, psycopg2, plotly  

**Critical Implementation Details:**
1. Sales date ranges MUST exclude CURRENT_DATE (today)
2. Use `BETWEEN (CURRENT_DATE - INTERVAL '31 days') AND (CURRENT_DATE - INTERVAL '1 day')` for 30-day window
3. Materialized view refresh strategy handles duplicates by dropping/recreating indexes
4. Dashboard uses column_config parameter for hover tooltips with formulas
5. All dataframes must show formulas in tooltips when hovering column headers

**Folder Structure:**
```
centurypenetration/
├── centuryPenetration.py          # Main dashboard
├── setup_database.py              # One-time setup
├── load_century_data.py           # Initial data load
├── daily_update.py                # Daily sales upload
├── refresh_century_views.py       # View refresh script
├── create_century_tables.sql      # Complete SQL definitions
├── refresh_views_complete.sql     # Manual refresh SQL
├── update_sales_date_logic.sql    # Date logic fix SQL
├── run_dashboard.bat              # Quick launcher
├── OPERATIONS_GUIDE.md            # This guide
└── data/
    ├── GEN_reorder_till10dec25.csv
    ├── GEN_SIT_till10dec25.csv
    ├── jan25.csv to dec25.csv
    ├── gendailysale.csv
    └── data_load_log.txt
```

---

## 📋 Validation Checklist

After recreation, verify:

✅ Database created with correct name and port  
✅ All 3 tables created with proper schemas  
✅ Monthly partitions created for sales table  
✅ All 3 materialized views created  
✅ Indexes created (non-unique)  
✅ Data loads successfully from CSV files  
✅ Views refresh without errors  
✅ Dashboard launches and displays data  
✅ Column tooltips show formulas on hover  
✅ Search functionality works  
✅ Download CSV buttons work  
✅ Charts display correctly  
✅ Date logic uses yesterday as end date  

**Test Query to Verify Date Logic:**
```sql
SELECT 
    CURRENT_DATE as today,
    CURRENT_DATE - INTERVAL '1 day' as yesterday,
    CURRENT_DATE - INTERVAL '31 days' as start_30d,
    COUNT(*) as sales_count
FROM mv_sales_metrics 
WHERE sales_30d > 0;
```

Expected: yesterday should be max date in calculations, not today.

---

**Version:** 1.0  
**Last Updated:** December 12, 2025  
**Compatible With:** PostgreSQL 16, Python 3.11+, Streamlit 1.28+
