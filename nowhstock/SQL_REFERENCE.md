# NO_WH Inventory Pulse - SQL Reference

## Database: salesdata (Port 3307)

---

## Table Definitions

### 1. inventory_master (Current Stock Levels)
```sql
CREATE TABLE inventory_master (
    itemcode VARCHAR(50),
    itemname VARCHAR(200),
    shopcode VARCHAR(50),
    shopstock DECIMAL(15,2),
    sales_30d_wh DECIMAL(15,2),
    shopgrn_dt DATE,
    groupp VARCHAR(100),
    subgroup VARCHAR(100),
    importexport VARCHAR(50),
    suppliername VARCHAR(200)
);

COMMENT ON TABLE inventory_master IS 'Current inventory levels across all shops';
COMMENT ON COLUMN inventory_master.shopstock IS 'Stock In Hand - current inventory at shop';
COMMENT ON COLUMN inventory_master.sales_30d_wh IS 'Sales from WH GRN date to +30 days';
COMMENT ON COLUMN inventory_master.shopgrn_dt IS 'Latest shop GRN date';
```

### 2. sales_2024 / sales_2025 (Sales Transactions)
```sql
CREATE TABLE sales_2024 (
    shop VARCHAR(50),
    item_code VARCHAR(50),
    date_invoice DATE,
    quantity DECIMAL(15,2),
    price DECIMAL(15,2),
    amount DECIMAL(15,2)
);

CREATE TABLE sales_2025 (
    shop VARCHAR(50),
    item_code VARCHAR(50),
    date_invoice DATE,
    quantity DECIMAL(15,2),
    price DECIMAL(15,2),
    amount DECIMAL(15,2)
);

COMMENT ON TABLE sales_2024 IS 'Sales transactions for 2024';
COMMENT ON TABLE sales_2025 IS 'Sales transactions for 2025';

-- Indexes for performance
CREATE INDEX idx_sales_2024_date ON sales_2024(date_invoice);
CREATE INDEX idx_sales_2024_item ON sales_2024(item_code);
CREATE INDEX idx_sales_2024_shop ON sales_2024(shop);

CREATE INDEX idx_sales_2025_date ON sales_2025(date_invoice);
CREATE INDEX idx_sales_2025_item ON sales_2025(item_code);
CREATE INDEX idx_sales_2025_shop ON sales_2025(shop);
```

### 3. sup_shop_grn (GRN Dates)
```sql
CREATE TABLE sup_shop_grn (
    item_code VARCHAR(50),
    shop_code VARCHAR(50),
    wh_grn_date DATE,
    shop_grn_date DATE
);

COMMENT ON TABLE sup_shop_grn IS 'Goods Received Note dates - warehouse and shop level';
COMMENT ON COLUMN sup_shop_grn.wh_grn_date IS 'Warehouse GRN date (item-level, same for all shops)';
COMMENT ON COLUMN sup_shop_grn.shop_grn_date IS 'Shop GRN date (shop-specific)';

-- Indexes
CREATE INDEX idx_grn_item ON sup_shop_grn(item_code);
CREATE INDEX idx_grn_shop ON sup_shop_grn(shop_code);
CREATE INDEX idx_grn_wh_date ON sup_shop_grn(wh_grn_date);
```

### 4. shopexpiry (Expiry Dates)
```sql
CREATE TABLE shopexpiry (
    "ITEM_CODE" VARCHAR(50),
    "SHOP_CODE" VARCHAR(50),
    "SHOP_EXPIRY_DATE" DATE
);

COMMENT ON TABLE shopexpiry IS 'Item expiry dates per shop';

-- Note: Column names are uppercase with quotes
-- Indexes
CREATE INDEX idx_expiry_item ON shopexpiry("ITEM_CODE");
CREATE INDEX idx_expiry_shop ON shopexpiry("SHOP_CODE");
```

### 5. itemdetails (Item Master)
```sql
CREATE TABLE itemdetails (
    item_code VARCHAR(50),
    item_name VARCHAR(200),
    "group" VARCHAR(100),  -- Reserved word, must be quoted
    subgroup VARCHAR(100),
    supplier VARCHAR(200),
    import_export VARCHAR(50)
);

COMMENT ON TABLE itemdetails IS 'Item master data';

-- Index
CREATE INDEX idx_itemdetails_code ON itemdetails(item_code);
```

---

## Materialized View: mv_recommendations_complete

**Purpose:** Complete stock transfer recommendation engine with ALL business logic in SQL

**Total Lines:** 489 lines
**Refresh Time:** 2-4 minutes
**Query Time:** < 100ms (with filters)

### View Structure (CTEs)

#### 1. priority_shops (Destination Shops Only)
```sql
priority_shops AS (
    SELECT shop_code, priority_rank
    FROM (VALUES 
        ('SPN', 1),  -- Spintex (highest priority)
        ('MSS', 2),  -- Mallam
        ('LFS', 3),  -- La Wireless
        ('M03', 4),  -- Melcom 3
        ('KAS', 5),  -- Kasoa
        ('MM1', 6),  -- Melcom 1
        ('MM2', 7),  -- Melcom 2
        ('FAR', 8),  -- Farisco
        ('KS7', 9),  -- Kumasi 7
        ('WHL', 10), -- Wholesale
        ('MM3', 11)  -- Melcom 3
    ) AS t(shop_code, priority_rank)
)
```

#### 2. item_wh_grn (WH GRN Dates)
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
**Business Logic:**
- WH GRN date is item-level (same for all shops)
- Used to calculate WH GRN +30d sales (demand spike after warehouse receipt)

#### 3. shop_expiry_latest (Latest Expiry per Item+Shop)
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

#### 4. sources (Source Shop Candidates)
```sql
sources AS (
    SELECT 
        im.itemcode AS item_code,
        im.itemname AS item_name,
        im.shopcode AS shop_code,
        im.shopstock AS stock,
        COALESCE(im.sales_30d_wh, 0) AS sales_30d,
        im.shopgrn_dt AS last_grn_date,
        -- GRN age calculation
        CASE 
            WHEN im.shopgrn_dt IS NOT NULL 
            THEN (CURRENT_DATE - im.shopgrn_dt::date)::integer
            ELSE 0
        END AS grn_age,
        wh.wh_grn_date,
        im.groupp AS groups,
        im.subgroup AS sub_group,
        -- Expiry handling
        se.latest_expiry_date AS source_expiry_date,
        CASE 
            WHEN se.latest_expiry_date IS NOT NULL 
            THEN (se.latest_expiry_date::date - CURRENT_DATE)::integer
            ELSE NULL
        END AS source_expiry_days,
        im.importexport AS item_type,
        im.suppliername AS supplier_name
    FROM inventory_master im
    LEFT JOIN item_wh_grn wh ON TRIM(UPPER(im.itemcode)) = wh.item_code
    LEFT JOIN shop_expiry_latest se 
        ON TRIM(UPPER(im.itemcode)) = se.item_code 
        AND TRIM(UPPER(im.shopcode)) = se.shop_code
    WHERE im.shopstock > COALESCE(im.sales_30d_wh, 0)  -- CRITICAL: Excess stock criterion
      AND im.shopcode NOT IN (SELECT shop_code FROM priority_shops)  -- Not a priority shop
)
```

**Key Filters:**
- `stock > sales_30d` - Must have excess stock (NOT stock > 30)
- `NOT IN priority_shops` - Priority shops only receive, never give

#### 5. destinations (Priority Shops with Demand Metrics)
```sql
destinations AS (
    SELECT 
        ps.shop_code AS dest_shop,
        ps.priority_rank AS dest_priority_rank,
        TRIM(UPPER(s.item_code)) AS item_code,
        wh.wh_grn_date,
        wh.wh_grn_plus_30,
        
        -- Metric 1: Regular 30-day sales
        SUM(CASE 
            WHEN s.date_invoice >= CURRENT_DATE - INTERVAL '30 days'
            THEN s.quantity 
            ELSE 0 
        END) as dest_sales_30d,
        
        -- Metric 2: WH GRN +30d sales (demand spike after receipt)
        SUM(CASE 
            WHEN wh.wh_grn_date IS NOT NULL
             AND s.date_invoice >= wh.wh_grn_date
             AND s.date_invoice <= wh.wh_grn_plus_30
            THEN s.quantity 
            ELSE 0 
        END) as dest_wh_grn_30d_sales,
        
        -- Metric 3: WH GRN to now (total demand since receipt)
        SUM(CASE 
            WHEN wh.wh_grn_date IS NOT NULL
             AND s.date_invoice >= wh.wh_grn_date
            THEN s.quantity 
            ELSE 0 
        END) as dest_wh_grn_date_to_now
        
    FROM priority_shops ps
    CROSS JOIN (SELECT DISTINCT item_code FROM inventory_master) items
    LEFT JOIN item_wh_grn wh ON TRIM(UPPER(items.itemcode)) = wh.item_code
    LEFT JOIN sales_2025 s 
        ON ps.shop_code = s.shop 
        AND TRIM(UPPER(items.itemcode)) = TRIM(UPPER(s.item_code))
    GROUP BY ps.shop_code, ps.priority_rank, items.item_code, wh.wh_grn_date, wh.wh_grn_plus_30
)
```

**Three Sales Metrics Explained:**

1. **dest_sales_30d** - Regular rolling 30-day sales
   - Standard demand pattern
   - Last 30 days from today

2. **dest_wh_grn_30d_sales** - Sales from WH GRN to +30 days
   - Captures demand spike after warehouse receives new stock
   - Many items see higher sales immediately after arrival
   - Example: WH received item on Dec 1, count sales Dec 1 to Dec 31

3. **dest_wh_grn_date_to_now** - Total sales since WH GRN
   - Cumulative demand since item arrived at warehouse
   - Not used for cap, but useful for analysis

#### 6. source_dest_pairs (Join Sources with Destinations)
```sql
source_dest_pairs AS (
    SELECT 
        s.item_code,
        s.item_name,
        s.groups,
        s.sub_group,
        s.item_type,
        s.supplier_name,
        s.shop_code AS source_shop,
        s.stock AS source_stock,
        s.sales_30d AS source_sales_30d,
        s.grn_age AS source_grn_age,
        s.source_expiry_date,
        s.source_expiry_days,
        d.dest_shop,
        d.dest_priority_rank,
        d.dest_sales_30d,
        d.dest_wh_grn_30d_sales,
        d.dest_wh_grn_date_to_now,
        s.wh_grn_date,
        
        -- Destination capacity = MAX of two metrics
        GREATEST(d.dest_sales_30d, d.dest_wh_grn_30d_sales) as destination_cap,
        
        -- Source excess = what source can give
        (s.stock - COALESCE(s.sales_30d, 0)) as source_excess,
        
        -- Available to allocate = source excess (before cumulative cap check)
        (s.stock - COALESCE(s.sales_30d, 0)) as available_to_allocate
        
    FROM sources s
    INNER JOIN destinations d ON s.item_code = d.item_code
    WHERE s.stock > COALESCE(s.sales_30d, 0)  -- Reconfirm excess stock
)
```

**Destination Cap Formula:**
```
destination_cap = GREATEST(dest_sales_30d, dest_wh_grn_30d_sales)
```
- Uses whichever metric is HIGHER
- Captures both regular demand and post-receipt spikes
- Prevents understocking during high-demand periods

#### 7. allocated_recommendations (FEFO Allocation with Window Functions)
```sql
allocated_recommendations AS (
    SELECT 
        *,
        -- Cumulative allocation per destination
        SUM(available_to_allocate) OVER (
            PARTITION BY item_code, dest_shop
            ORDER BY 
                dest_priority_rank ASC,         -- Process by priority
                source_grn_age DESC,            -- Oldest stock first
                source_expiry_days ASC NULLS LAST  -- Earliest expiry first
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as cumulative_allocated,
        
        -- Actual allocated quantity (capped by destination capacity)
        LEAST(
            available_to_allocate,
            destination_cap - (
                SUM(available_to_allocate) OVER (
                    PARTITION BY item_code, dest_shop
                    ORDER BY 
                        dest_priority_rank ASC,
                        source_grn_age DESC,
                        source_expiry_days ASC NULLS LAST
                    ROWS BETWEEN UNBOUNDED PRECEDING AND PRECEDING 1 ROW
                ) 
            )
        ) as allocated_qty,
        
        -- Remaining capacity after this allocation
        destination_cap - SUM(available_to_allocate) OVER (
            PARTITION BY item_code, dest_shop
            ORDER BY 
                dest_priority_rank ASC,
                source_grn_age DESC,
                source_expiry_days ASC NULLS LAST
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as remaining_cap
        
    FROM source_dest_pairs
)
```

**CRITICAL Window Function Logic:**

**PARTITION BY (item_code, dest_shop):**
- Each destination processes independently
- No competition between destinations for same source
- Example: Item ABC to SPN independent of Item ABC to MSS

**ORDER BY:**
1. `dest_priority_rank ASC` - Process SPN (rank 1) before MM3 (rank 11)
2. `source_grn_age DESC` - Oldest stock first (FEFO)
3. `source_expiry_days ASC NULLS LAST` - Earliest expiry first

**Cumulative Allocation:**
```
Running SUM of available_to_allocate from first row to current row
```
- Ensures total allocation never exceeds destination_cap
- Each source contributes until cap reached

**Allocated Quantity:**
```
LEAST(
    what source can give,
    remaining capacity at destination
)
```

#### 8. Final Recommendations (With Blocking Rules)
```sql
SELECT 
    item_code,
    item_name,
    groups,
    sub_group,
    item_type,
    supplier_name,
    source_shop,
    source_stock,
    source_sales_30d,
    source_grn_age,
    source_expiry_date,
    source_expiry_days,
    dest_shop,
    dest_priority_rank,
    dest_sales_30d,
    dest_wh_grn_30d_sales,
    dest_wh_grn_date_to_now,
    destination_cap,
    allocated_qty,
    remaining_cap,
    wh_grn_date,
    
    -- Blocking logic
    CASE
        WHEN source_shop = dest_shop THEN 'Same shop transfer'
        WHEN source_shop IN (SELECT shop_code FROM priority_shops) THEN 'Priority shop cannot be source'
        WHEN source_expiry_days < 30 THEN 'Source expiring within 30 days'
        WHEN dest_sales_30d = 0 AND dest_wh_grn_30d_sales = 0 AND dest_wh_grn_date_to_now = 0 THEN 'No demand at destination'
        WHEN cumulative_allocated > destination_cap THEN 'Cap exceeded'
        ELSE NULL
    END as block_reason,
    
    CURRENT_TIMESTAMP as refreshed_at
    
FROM allocated_recommendations
WHERE allocated_qty > 0
  AND block_reason IS NULL  -- Only valid recommendations
ORDER BY dest_priority_rank, item_code, source_shop;
```

**Blocking Rules:**

1. **Same Shop Transfer**
   - Source = Destination not allowed
   - Business rule: No self-transfers

2. **Priority Shop as Source**
   - SPN, MSS, LFS, etc. cannot be sources
   - Priority shops only RECEIVE

3. **Expiry Check**
   - Block if < 30 days to expiry
   - Safety buffer (not < 0)

4. **No Demand**
   - All three sales metrics = 0
   - No point transferring if no demand

5. **Cap Exceeded**
   - Cumulative allocation > destination capacity
   - Should not happen with window function, but safety check

---

## Indexes on Materialized View

```sql
CREATE INDEX idx_rec_item ON mv_recommendations_complete(item_code);
CREATE INDEX idx_rec_source ON mv_recommendations_complete(source_shop);
CREATE INDEX idx_rec_dest ON mv_recommendations_complete(dest_shop);
CREATE INDEX idx_rec_groups ON mv_recommendations_complete(groups);
CREATE INDEX idx_rec_subgroup ON mv_recommendations_complete(sub_group);
CREATE INDEX idx_rec_allocated ON mv_recommendations_complete(allocated_qty) WHERE allocated_qty > 0;
CREATE INDEX idx_rec_priority ON mv_recommendations_complete(dest_priority_rank);
```

---

## Refresh Commands

### Refresh Materialized View
```sql
REFRESH MATERIALIZED VIEW mv_recommendations_complete;
```

### Check Refresh Status
```sql
SELECT 
    'mv_recommendations_complete' as view_name,
    COUNT(*) as row_count,
    COUNT(DISTINCT item_code) as unique_items,
    COUNT(DISTINCT source_shop) as source_shops,
    COUNT(DISTINCT dest_shop) as dest_shops,
    SUM(allocated_qty) as total_allocated_qty,
    pg_size_pretty(pg_total_relation_size('mv_recommendations_complete')) as view_size,
    (SELECT MAX(refreshed_at) FROM mv_recommendations_complete) as last_refresh
FROM mv_recommendations_complete;
```

---

## Verification Queries

### 1. Check Priority Shops Not in Sources
```sql
-- Should return 0 rows
SELECT DISTINCT source_shop 
FROM mv_recommendations_complete
WHERE source_shop IN ('SPN','MSS','LFS','M03','KAS','MM1','MM2','FAR','KS7','WHL','MM3');
```

### 2. Verify Cap Enforcement
```sql
-- Check if any destination exceeded capacity
SELECT 
    item_code, 
    dest_shop,
    SUM(allocated_qty) as total_allocated,
    MAX(destination_cap) as cap,
    SUM(allocated_qty) - MAX(destination_cap) as excess,
    CASE 
        WHEN SUM(allocated_qty) > MAX(destination_cap) THEN '❌ CAP EXCEEDED!'
        ELSE '✅ OK'
    END as status
FROM mv_recommendations_complete
GROUP BY item_code, dest_shop
HAVING SUM(allocated_qty) > MAX(destination_cap);

-- Should return 0 rows (no violations)
```

### 3. Check FEFO Ordering
```sql
-- Verify oldest stock allocated first
SELECT 
    item_code,
    dest_shop,
    source_shop,
    source_grn_age,
    source_expiry_days,
    allocated_qty,
    ROW_NUMBER() OVER (PARTITION BY item_code, dest_shop ORDER BY source_grn_age DESC) as age_rank
FROM mv_recommendations_complete
WHERE allocated_qty > 0
ORDER BY item_code, dest_shop, age_rank
LIMIT 20;
```

### 4. Top Items by Allocated Quantity
```sql
SELECT 
    item_code,
    item_name,
    COUNT(DISTINCT source_shop) as num_sources,
    COUNT(DISTINCT dest_shop) as num_destinations,
    SUM(allocated_qty) as total_allocated,
    ROUND(AVG(allocated_qty), 2) as avg_per_transfer
FROM mv_recommendations_complete
GROUP BY item_code, item_name
ORDER BY total_allocated DESC
LIMIT 20;
```

### 5. Allocation by Destination
```sql
SELECT 
    dest_shop,
    dest_priority_rank,
    COUNT(DISTINCT item_code) as items_receiving,
    SUM(allocated_qty) as total_allocated,
    ROUND(AVG(allocated_qty), 2) as avg_per_item
FROM mv_recommendations_complete
GROUP BY dest_shop, dest_priority_rank
ORDER BY dest_priority_rank;
```

### 6. Source Shop Contribution
```sql
SELECT 
    source_shop,
    COUNT(DISTINCT item_code) as items_giving,
    COUNT(DISTINCT dest_shop) as destinations_served,
    SUM(allocated_qty) as total_given,
    ROUND(AVG(source_stock), 0) as avg_stock_level
FROM mv_recommendations_complete
GROUP BY source_shop
ORDER BY total_given DESC
LIMIT 20;
```

### 7. Items Expiring Soon (< 60 days)
```sql
SELECT 
    item_code,
    item_name,
    source_shop,
    source_expiry_days,
    source_stock,
    allocated_qty,
    CASE 
        WHEN source_expiry_days < 30 THEN '🔴 Blocked'
        WHEN source_expiry_days < 60 THEN '🟡 Warning'
        ELSE '🟢 Safe'
    END as expiry_status
FROM mv_recommendations_complete
WHERE source_expiry_days IS NOT NULL
  AND source_expiry_days < 60
ORDER BY source_expiry_days ASC;
```

---

## Maintenance Commands

### Analyze Tables (Before Refresh)
```sql
ANALYZE inventory_master;
ANALYZE sales_2024;
ANALYZE sales_2025;
ANALYZE sup_shop_grn;
ANALYZE shopexpiry;
ANALYZE itemdetails;
```

### Vacuum (After Large Data Changes)
```sql
VACUUM ANALYZE inventory_master;
VACUUM ANALYZE sales_2025;
```

### Check Table Sizes
```sql
SELECT 
    schemaname,
    relname as table_name,
    n_live_tup as row_count,
    pg_size_pretty(pg_total_relation_size(relid)) as total_size,
    pg_size_pretty(pg_relation_size(relid)) as table_size,
    pg_size_pretty(pg_total_relation_size(relid) - pg_relation_size(relid)) as index_size
FROM pg_stat_user_tables
WHERE schemaname = 'public'
  AND relname IN ('inventory_master', 'sales_2024', 'sales_2025', 'sup_shop_grn', 'shopexpiry', 'mv_recommendations_complete')
ORDER BY pg_total_relation_size(relid) DESC;
```

---

**Last Updated:** December 12, 2025  
**Database:** salesdata (port 3307)  
**PostgreSQL Version:** 16+
