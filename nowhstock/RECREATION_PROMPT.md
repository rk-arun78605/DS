# NO_WH Inventory Pulse - Recreation Prompt

## 🎯 Purpose
This file contains the complete prompt to recreate the NO_WH Inventory Pulse (Stock Transfer Recommendation System) from scratch using AI assistance.

---

## 📝 Complete Prompt for AI Agent

```
I need you to create a complete NO_WH Inventory Pulse system - a stock transfer recommendation dashboard that suggests which items to transfer between shops based on stock levels, sales velocity, and demand patterns.

### SYSTEM OVERVIEW
Build a Streamlit dashboard that:
- Identifies items with excess stock at non-priority shops
- Recommends transfers to 11 priority shops based on demand
- Uses FEFO (First Expiry First Out) for allocation
- Prevents transfers of soon-to-expire items
- ALL business logic in SQL materialized view (Python just queries and displays)

### DATABASE ARCHITECTURE

**PostgreSQL 16 Setup:**
- Port: 3307
- Database: salesdata (shared with other dashboards)
- User: postgres
- Password: hello

**Tables Required:**

1. **inventory_master** (Current stock levels)
   - itemcode VARCHAR(50) - Item identifier
   - itemname VARCHAR(200) - Item description
   - shopcode VARCHAR(50) - Shop identifier
   - shopstock DECIMAL(15,2) - Current stock in hand
   - sales_30d_wh DECIMAL(15,2) - Sales from WH GRN date + 30 days
   - shopgrn_dt DATE - Latest shop GRN date
   - groupp VARCHAR(100) - Product group
   - subgroup VARCHAR(100) - Product sub-group
   - importexport VARCHAR(50) - Import/Export/Local
   - suppliername VARCHAR(200) - Supplier name

2. **sales_2024** / **sales_2025** (Sales transactions by year)
   - shop VARCHAR(50) - Shop code
   - item_code VARCHAR(50) - Item code
   - date_invoice DATE - Transaction date
   - quantity DECIMAL(15,2) - Quantity sold
   - price DECIMAL(15,2) - Unit price
   - amount DECIMAL(15,2) - Total amount

3. **sup_shop_grn** (GRN dates)
   - item_code VARCHAR(50)
   - shop_code VARCHAR(50)
   - wh_grn_date DATE - Warehouse GRN (item-level, same for all shops)
   - shop_grn_date DATE - Shop GRN (shop-specific)

4. **shopexpiry** (Expiry dates)
   - "ITEM_CODE" VARCHAR(50) - Note: uppercase column names with quotes
   - "SHOP_CODE" VARCHAR(50)
   - "SHOP_EXPIRY_DATE" DATE - Latest expiry date

5. **itemdetails** (Item master)
   - item_code VARCHAR(50)
   - item_name VARCHAR(200)
   - "group" VARCHAR(100) - Note: quoted because it's a reserved word
   - subgroup VARCHAR(100)
   - supplier VARCHAR(200)
   - import_export VARCHAR(50)

**CRITICAL: Materialized View (Heart of the System)**

**mv_recommendations_complete** - Complete 489-line SQL with ALL business logic

**Business Rules in SQL:**

1. **Priority Shops (Destinations Only - Ranked 1-11):**
   ```sql
   priority_shops AS (
       SELECT shop_code, priority_rank
       FROM (VALUES 
           ('SPN', 1), ('MSS', 2), ('LFS', 3), ('M03', 4), ('KAS', 5),
           ('MM1', 6), ('MM2', 7), ('FAR', 8), ('KS7', 9), ('WHL', 10), ('MM3', 11)
       ) AS t(shop_code, priority_rank)
   )
   ```

2. **Source Shop Eligibility:**
   - Must NOT be a priority shop
   - Must have stock > sales_30d (excess stock criterion)
   - Filter: `stock > COALESCE(sales_30d, 0)`

3. **WH GRN Date Logic:**
   ```sql
   item_wh_grn AS (
       SELECT 
           TRIM(UPPER(item_code)) AS item_code,
           MAX(wh_grn_date) AS wh_grn_date,
           MAX(wh_grn_date) + INTERVAL '30 days' AS wh_grn_plus_30
       FROM sup_shop_grn
       WHERE wh_grn_date IS NOT NULL
       GROUP BY TRIM(UPPER(item_code))
   )
   ```

4. **Expiry Handling:**
   ```sql
   shop_expiry_latest AS (
       SELECT 
           TRIM(UPPER("ITEM_CODE")) AS item_code,
           TRIM(UPPER("SHOP_CODE")) AS shop_code,
           MAX("SHOP_EXPIRY_DATE") AS latest_expiry_date
       FROM shopexpiry
       GROUP BY TRIM(UPPER("ITEM_CODE")), TRIM(UPPER("SHOP_CODE"))
   )
   ```
   - Calculate days to expiry: `(expiry_date::date - CURRENT_DATE)::integer`
   - Block transfers if < 30 days

5. **Destination Sales Calculations (3 metrics):**
   
   a) **dest_sales_30d** - Regular 30-day sales
   ```sql
   SUM(CASE 
       WHEN s.date_invoice >= CURRENT_DATE - INTERVAL '30 days'
       THEN s.quantity 
       ELSE 0 
   END) as dest_sales_30d
   ```
   
   b) **dest_wh_grn_30d_sales** - Sales from WH GRN date to +30 days
   ```sql
   SUM(CASE 
       WHEN wh.wh_grn_date IS NOT NULL
        AND s.date_invoice >= wh.wh_grn_date
        AND s.date_invoice <= wh.wh_grn_plus_30
       THEN s.quantity 
       ELSE 0 
   END) as dest_wh_grn_30d_sales
   ```
   
   c) **dest_wh_grn_date_to_now** - Sales from WH GRN to today
   ```sql
   SUM(CASE 
       WHEN wh.wh_grn_date IS NOT NULL
        AND s.date_invoice >= wh.wh_grn_date
       THEN s.quantity 
       ELSE 0 
   END) as dest_wh_grn_date_to_now
   ```

6. **Destination Capacity Formula:**
   ```sql
   GREATEST(dest_sales_30d, dest_wh_grn_30d_sales) as destination_cap
   ```
   - Use whichever is higher (captures demand spikes)

7. **FEFO Allocation Logic:**
   ```sql
   ORDER BY:
       1. dest_priority_rank ASC (process destinations in order)
       2. grn_age DESC (oldest stock first)
       3. source_expiry_days ASC NULLS LAST (earliest expiry first)
   ```

8. **Cumulative Allocation (Window Function):**
   ```sql
   SUM(available_to_allocate) OVER (
       PARTITION BY item_code, dest_shop
       ORDER BY grn_age DESC, source_expiry_days ASC NULLS LAST
       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
   ) as cumulative_allocated
   ```
   
   **CRITICAL:** Each destination processes independently (no competition between destinations)

9. **Allocated Quantity Calculation:**
   ```sql
   LEAST(
       source_excess,  -- How much source can give
       destination_cap - (cumulative_allocated - available_to_allocate)  -- Remaining cap
   ) as allocated_qty
   ```

10. **Blocking Rules:**
    ```sql
    CASE
        WHEN source_shop = dest_shop THEN 'Same shop transfer'
        WHEN source_shop IN (SELECT shop_code FROM priority_shops) THEN 'Priority shop cannot be source'
        WHEN source_expiry_days < 30 THEN 'Source expiring within 30 days'
        WHEN dest_sales_30d = 0 AND dest_wh_grn_30d_sales = 0 AND dest_wh_grn_date_to_now = 0 THEN 'No demand at destination'
        WHEN cumulative_allocated > destination_cap THEN 'Cap exceeded'
        ELSE NULL
    END as block_reason
    ```

11. **Final Filtering:**
    ```sql
    WHERE allocated_qty > 0
      AND block_reason IS NULL
    ```

**Complete View Columns (Must Include):**
- item_code, item_name, groups, sub_group, item_type, supplier_name
- source_shop, source_stock, source_sales_30d, source_grn_age
- source_expiry_date, source_expiry_days, expiry_check
- dest_shop, dest_priority_rank
- dest_sales_30d, dest_wh_grn_30d_sales, dest_wh_grn_date_to_now
- destination_cap, allocated_qty, remaining_cap
- wh_grn_date
- block_reason (NULL for valid recommendations)
- refreshed_at TIMESTAMP

**Indexes:**
```sql
CREATE INDEX idx_rec_item ON mv_recommendations_complete(item_code);
CREATE INDEX idx_rec_source ON mv_recommendations_complete(source_shop);
CREATE INDEX idx_rec_dest ON mv_recommendations_complete(dest_shop);
CREATE INDEX idx_rec_allocated ON mv_recommendations_complete(allocated_qty) WHERE allocated_qty > 0;
```

### PYTHON SCRIPTS REQUIRED

**1. load_all_data.py** (Data Loading Script)

Located in: `nowhstock/batch/`

**Functionality:**
- Loads CSV files from project root into PostgreSQL tables
- Uses TRUNCATE and reload strategy
- Progress logging to `data_load_log.txt`
- execute_batch for performance

**Files to load:**
```python
FILES_TO_LOAD = {
    'inventory_master': 'd:/Dashboard Code/NO_WH/DS/inventory_master.csv',
    'sales_2024': 'd:/Dashboard Code/NO_WH/DS/sales_2024.csv',
    'sales_2025': 'd:/Dashboard Code/NO_WH/DS/sales_2025.csv',
    'sup_shop_grn': 'd:/Dashboard Code/NO_WH/DS/sup_shop_grn.csv',
    'shopexpiry': 'd:/Dashboard Code/NO_WH/DS/shopexpiry.csv',
    'itemdetails': 'd:/Dashboard Code/NO_WH/DS/itemdetails.csv'
}
```

**Connection:**
```python
conn = psycopg2.connect(
    host='localhost',
    port=3307,
    user='postgres',
    password='hello',
    dbname='salesdata'
)
```

**Load Strategy:**
```python
# For each table:
1. TRUNCATE TABLE {table_name}
2. Read CSV with pandas
3. Use execute_batch with page_size=1000
4. INSERT INTO {table_name} VALUES (...)
5. Log success/failure
```

**2. refresh_all_views.py** (View Refresh Script)

Located in: `nowhstock/batch/`

**Functionality:**
```python
def refresh_views():
    conn = psycopg2.connect(...)
    cursor = conn.cursor()
    
    # Refresh materialized view
    cursor.execute("REFRESH MATERIALIZED VIEW mv_recommendations_complete")
    conn.commit()
    
    # Get row count
    cursor.execute("SELECT COUNT(*) FROM mv_recommendations_complete")
    count = cursor.fetchone()[0]
    
    print(f"✅ View refreshed: {count} recommendations")
    
    cursor.close()
    conn.close()
```

**3. nowhstock_ds.py** (Streamlit Dashboard)

**Configuration Class:**
```python
class Config:
    DB_CONFIG = {
        'host': 'localhost',
        'user': 'postgres',
        'password': 'hello',
        'port': 3307
    }
    
    PRIORITY_SHOPS = ['SPN', 'MSS', 'LFS', 'M03', 'KAS', 'MM1', 'MM2', 'FAR', 'KS7', 'WHL', 'MM3']
    ALL_SHOPS = ['SPN', 'MSS', ...] # All 80+ shops
    DEFAULT_THRESHOLD = 30
    BUFFER_DAYS = 30
    
    PAGE_TITLE = "Inventory Pulse NO_WH"
    LOGO_URL = "https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg"
```

**Connection Pattern (CRITICAL):**
```python
@st.cache_resource
def get_connection_pool(dbname: str):
    return psycopg2.pool.SimpleConnectionPool(
        minconn=1, maxconn=5,
        dbname=dbname, **Config.DB_CONFIG
    )

@contextmanager
def get_db_connection(dbname: str):
    pool = get_connection_pool(dbname)
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)
```

**Authentication (Uses shared 'users' database):**
```python
def authenticate_user(employee_id: str, password: str):
    with get_db_connection('users') as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            "SELECT * FROM \"user\" WHERE employee_id = %s",
            (employee_id,)
        )
        user = cursor.fetchone()
        
        if user and user['password'] == hashlib.sha256(password.encode()).hexdigest():
            if user['is_active'] == 'true':  # String comparison
                return user
    return None

def check_table_access(user: dict, required_table: str):
    tables = [t.strip().lower() for t in user.get('table_access', '').split(',')]
    return required_table.lower() in tables or 'all' in tables
```

**Main Query Function:**
```python
@st.cache_data(ttl=300)
def load_recommendations(filters: dict):
    """
    Query mv_recommendations_complete with filters
    """
    query = """
        SELECT * FROM mv_recommendations_complete
        WHERE 1=1
    """
    params = []
    
    # Add filter conditions
    if filters.get('source_shop'):
        query += " AND source_shop = %s"
        params.append(filters['source_shop'])
    
    if filters.get('dest_shop'):
        query += " AND dest_shop = %s"
        params.append(filters['dest_shop'])
    
    if filters.get('item_code'):
        query += " AND item_code ILIKE %s"
        params.append(f"%{filters['item_code']}%")
    
    # ... more filters
    
    query += " ORDER BY dest_priority_rank, item_code, source_shop"
    
    with get_db_connection('salesdata') as conn:
        df = pd.read_sql(query, conn, params=params)
    
    return df
```

**Dashboard Layout:**

**Header:**
- Logo (Melcom favicon)
- Title: "Inventory Pulse NO_WH"
- Subtitle: "Stock Transfer Recommendations"

**Sidebar Filters:**
1. Source Shop (dropdown, all non-priority shops)
2. Destination Shop (dropdown, 11 priority shops only)
3. Group (multi-select)
4. Sub Group (multi-select)
5. Item Code (text input)
6. Item Name (text input)
7. Supplier (multi-select)
8. Sales Threshold (number input, default 30)
9. Import/Export (multi-select)
10. Expiry Status (Safe, Expiring, No Expiry, All)

**Main Content:**

**Metrics Row (4 cards):**
```python
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Items", f"{df['item_code'].nunique():,}")
col2.metric("Total Quantity", f"{df['allocated_qty'].sum():,.0f}")
col3.metric("Avg Allocated Qty", f"{df['allocated_qty'].mean():.1f}")
col4.metric("Destinations Covered", f"{df['dest_shop'].nunique()}")
```

**Recommendations Table:**
```python
st.dataframe(
    df,
    use_container_width=True,
    height=600,
    column_config={
        'allocated_qty': st.column_config.NumberColumn('Allocated Qty', format='%d'),
        'source_stock': st.column_config.NumberColumn('Source Stock', format='%d'),
        'destination_cap': st.column_config.NumberColumn('Dest Cap', format='%d'),
        'dest_priority_rank': st.column_config.NumberColumn('Priority', format='%d'),
        'source_expiry_days': st.column_config.NumberColumn('Expiry Days', format='%d')
    }
)
```

**Visualizations (3 tabs):**

Tab 1: Top 10 Items by Quantity
```python
top_items = df.groupby('item_name')['allocated_qty'].sum().nlargest(10)
fig = px.bar(top_items, x=top_items.values, y=top_items.index, orientation='h')
st.plotly_chart(fig)
```

Tab 2: Allocation by Destination
```python
dest_summary = df.groupby('dest_shop')['allocated_qty'].sum().sort_values(ascending=False)
fig = px.bar(dest_summary, x=dest_summary.index, y=dest_summary.values)
st.plotly_chart(fig)
```

Tab 3: Allocation by Group
```python
group_summary = df.groupby('groups')['allocated_qty'].sum().nlargest(10)
fig = px.pie(values=group_summary.values, names=group_summary.index)
st.plotly_chart(fig)
```

**Download Button:**
```python
csv = df.to_csv(index=False)
st.download_button(
    label="📥 Download Recommendations",
    data=csv,
    file_name=f"nowhstock_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)
```

**Additional Files:**

4. **refresh_all_views.sql**
```sql
-- Manual refresh via psql
REFRESH MATERIALIZED VIEW mv_recommendations_complete;

SELECT 
    'mv_recommendations_complete' as view_name,
    COUNT(*) as row_count,
    pg_size_pretty(pg_total_relation_size('mv_recommendations_complete')) as size
FROM mv_recommendations_complete;
```

5. **refresh_views_scheduled.bat**
```batch
@echo off
cd "d:\Dashboard Code\NO_WH\DS\nowhstock\batch"
python refresh_all_views.py >> refresh_views_log.txt 2>&1
```

6. **load_all_data.bat**
```batch
@echo off
cd "d:\Dashboard Code\NO_WH\DS\nowhstock\batch"
python load_all_data.py
pause
```

7. **run_nowhstock_backup.bat**
```batch
@echo off
cd "d:\Dashboard Code\NO_WH\DS"
streamlit run nowhstock_ds.py
pause
```

### KEY BUSINESS RULES (CRITICAL)

1. **Priority Shops ONLY Receive, NEVER Give**
   - 11 shops: SPN, MSS, LFS, M03, KAS, MM1, MM2, FAR, KS7, WHL, MM3
   - Filter sources: `WHERE source_shop NOT IN (priority_shops)`

2. **Source Eligibility: Stock > Sales 30d**
   - NOT stock > 30 units
   - Criterion: Has excess stock based on sales velocity

3. **Destination Cap = MAX of Two Metrics**
   - dest_sales_30d (regular demand)
   - dest_wh_grn_30d_sales (post-receipt demand spike)
   - Take whichever is HIGHER

4. **FEFO Allocation Order**
   - First: Oldest GRN (highest grn_age)
   - Second: Earliest expiry (lowest expiry_days)
   - Ensures oldest/expiring stock transferred first

5. **30-Day Expiry Buffer**
   - Block transfers if source_expiry_days < 30
   - Safety margin to avoid transferring about-to-expire items

6. **Independent Destination Processing**
   - Each destination's allocation calculated separately
   - No competition between destinations
   - PARTITION BY (item_code, dest_shop)

7. **Cumulative Cap Enforcement**
   - Running sum of allocations per destination
   - Stop when cumulative_allocated >= destination_cap

### PERFORMANCE OPTIMIZATIONS

1. **Materialized View Strategy**
   - ALL heavy computation in SQL
   - Python just filters and displays
   - Refresh time: 2-4 minutes (acceptable)

2. **Connection Pooling**
   - SimpleConnectionPool (1-5 connections)
   - Context managers for safe connection handling
   - Never hold connections

3. **Caching**
   - @st.cache_data(ttl=300) for queries
   - @st.cache_resource for connection pools

4. **Indexes on View**
   - item_code, source_shop, dest_shop
   - allocated_qty (filtered index WHERE > 0)

### ERROR HANDLING

1. All database operations in try-except blocks
2. Log errors to data_load_log.txt
3. User-friendly error messages in Streamlit
4. Handle missing CSV files gracefully
5. Validate user access before showing data

### DELIVERABLES

1. NowhStock_mv_recommendations_complete.sql (489-line complete view definition)
2. nowhstock_ds.py (Streamlit dashboard with filters, visualizations, download)
3. load_all_data.py (Data loading script)
4. refresh_all_views.py (View refresh script)
5. refresh_all_views.sql (SQL script for manual refresh)
6. Batch files for easy execution
7. Documentation (operations guide)

### VALIDATION CHECKLIST

After creation, verify:

✅ Materialized view exists and has data (> 0 rows)
✅ View contains all required columns
✅ Priority shops only appear as destinations, not sources
✅ Allocated qty never exceeds destination cap per item-destination
✅ FEFO ordering works (oldest GRN first)
✅ Expiry blocking works (< 30 days blocked)
✅ Dashboard launches and shows recommendations
✅ Filters work correctly
✅ Charts display properly
✅ Download CSV works
✅ Login system works with users database

**Test Query:**
```sql
-- Should show only non-priority shops as sources
SELECT DISTINCT source_shop 
FROM mv_recommendations_complete
WHERE source_shop IN ('SPN','MSS','LFS','M03','KAS','MM1','MM2','FAR','KS7','WHL','MM3');
-- Should return 0 rows

-- Check cap enforcement
SELECT 
    item_code, dest_shop,
    SUM(allocated_qty) as total_allocated,
    MAX(destination_cap) as cap,
    CASE WHEN SUM(allocated_qty) > MAX(destination_cap) THEN 'CAP EXCEEDED!' ELSE 'OK' END as status
FROM mv_recommendations_complete
GROUP BY item_code, dest_shop
HAVING SUM(allocated_qty) > MAX(destination_cap);
-- Should return 0 rows (no cap violations)
```
```

---

## 🔧 Additional Context for AI

**Database Type:** PostgreSQL 16  
**Python Version:** 3.11+  
**Key Libraries:** streamlit, pandas, psycopg2, plotly, numpy  

**Critical Implementation Details:**

1. **Materialized view is the engine** - 489 lines of SQL with complete business logic
2. **Python is just UI layer** - Queries view with WHERE clauses, displays results
3. **Priority shop concept** - 11 specific shops that only receive, never give
4. **Dual sales metrics** - Regular 30d and WH GRN +30d to capture demand spikes
5. **FEFO allocation** - Oldest stock (by GRN age and expiry) allocated first
6. **Window functions** - Cumulative SUM per destination for cap enforcement
7. **Safety buffers** - 30-day expiry threshold prevents risky transfers

**Folder Structure:**
```
d:\Dashboard Code\NO_WH\DS\
├── nowhstock_ds.py                     # Main dashboard
├── NowhStock_mv_recommendations_complete.sql  # View definition
├── inventory_master.csv                # Daily stock data
├── sales_2024.csv / sales_2025.csv    # Sales history
├── sup_shop_grn.csv                   # GRN dates
├── shopexpiry.csv                     # Expiry dates
├── itemdetails.csv                    # Item master
└── nowhstock/
    ├── OPERATIONS_GUIDE.md            # This guide
    └── batch/
        ├── load_all_data.py           # Data loader
        ├── refresh_all_views.py       # View refresh
        ├── refresh_all_views.sql      # SQL refresh
        ├── refresh_views_scheduled.bat
        ├── load_all_data.bat
        └── data_load_log.txt
```

---

**Version:** 2.3  
**Last Updated:** December 12, 2025  
**Database:** salesdata (port 3307)  
**Compatible With:** PostgreSQL 16, Python 3.11+, Streamlit 1.28+
