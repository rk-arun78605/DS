# Century Stock Penetration - Complete SQL Reference

## Database Creation
```sql
CREATE DATABASE century_penetration
    WITH 
    OWNER = postgres
    ENCODING = 'UTF8'
    CONNECTION LIMIT = -1;
```

## Table Definitions

### 1. reorder_level (Main Inventory Table)
```sql
CREATE TABLE IF NOT EXISTS reorder_level (
    item_code VARCHAR(50),
    item_name VARCHAR(200),
    shop_code VARCHAR(50),
    dept VARCHAR(100),
    brand VARCHAR(100),
    shop_grn_date DATE,
    wh_grn_date DATE,
    shop_stock DECIMAL(15,2),  -- Stock In Hand (SIH)
    min_nu DECIMAL(15,2),      -- Minimum stock level
    max_nu DECIMAL(15,2),      -- Maximum stock level
    reorder_qty DECIMAL(15,2), -- Reorder quantity
    selling_price DECIMAL(15,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE reorder_level IS 'Master inventory table with stock levels and reorder parameters';
COMMENT ON COLUMN reorder_level.shop_stock IS 'Stock In Hand (SIH) - current inventory at shop';
COMMENT ON COLUMN reorder_level.min_nu IS 'Minimum stock level - reorder point';
COMMENT ON COLUMN reorder_level.max_nu IS 'Maximum stock level - upper limit';
COMMENT ON COLUMN reorder_level.reorder_qty IS 'Quantity to order when stock falls below minimum';
```

### 2. sit (Stock In Transit)
```sql
CREATE TABLE IF NOT EXISTS sit (
    shop_code VARCHAR(50),
    item_code VARCHAR(50),
    transit_date DATE,
    transit_qty DECIMAL(15,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE sit IS 'Stock In Transit - items en route to shops';
```

### 3. sales (Partitioned Parent Table)
```sql
CREATE TABLE IF NOT EXISTS sales (
    shop_code VARCHAR(50),
    item_code VARCHAR(50),
    date_invoice DATE NOT NULL,
    quantity DECIMAL(15,2),
    price DECIMAL(15,2),
    amount DECIMAL(15,2),
    dept VARCHAR(100),
    brand VARCHAR(100),
    item_name VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) PARTITION BY RANGE (date_invoice);

COMMENT ON TABLE sales IS 'Sales transactions partitioned by month';
```

### 4. Sales Partitions (Monthly for 2025)
```sql
-- January 2025
CREATE TABLE IF NOT EXISTS sales_jan2025 PARTITION OF sales
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- February 2025
CREATE TABLE IF NOT EXISTS sales_feb2025 PARTITION OF sales
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

-- March 2025
CREATE TABLE IF NOT EXISTS sales_mar2025 PARTITION OF sales
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

-- April 2025
CREATE TABLE IF NOT EXISTS sales_apr2025 PARTITION OF sales
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');

-- May 2025
CREATE TABLE IF NOT EXISTS sales_may2025 PARTITION OF sales
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');

-- June 2025
CREATE TABLE IF NOT EXISTS sales_jun2025 PARTITION OF sales
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');

-- July 2025
CREATE TABLE IF NOT EXISTS sales_jul2025 PARTITION OF sales
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');

-- August 2025
CREATE TABLE IF NOT EXISTS sales_aug2025 PARTITION OF sales
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');

-- September 2025
CREATE TABLE IF NOT EXISTS sales_sep2025 PARTITION OF sales
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');

-- October 2025
CREATE TABLE IF NOT EXISTS sales_oct2025 PARTITION OF sales
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');

-- November 2025
CREATE TABLE IF NOT EXISTS sales_nov2025 PARTITION OF sales
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

-- December 2025
CREATE TABLE IF NOT EXISTS sales_dec2025 PARTITION OF sales
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Create indexes on each partition for faster queries
CREATE INDEX IF NOT EXISTS idx_sales_jan2025_date ON sales_jan2025(date_invoice);
CREATE INDEX IF NOT EXISTS idx_sales_feb2025_date ON sales_feb2025(date_invoice);
CREATE INDEX IF NOT EXISTS idx_sales_mar2025_date ON sales_mar2025(date_invoice);
CREATE INDEX IF NOT EXISTS idx_sales_apr2025_date ON sales_apr2025(date_invoice);
CREATE INDEX IF NOT EXISTS idx_sales_may2025_date ON sales_may2025(date_invoice);
CREATE INDEX IF NOT EXISTS idx_sales_jun2025_date ON sales_jun2025(date_invoice);
CREATE INDEX IF NOT EXISTS idx_sales_jul2025_date ON sales_jul2025(date_invoice);
CREATE INDEX IF NOT EXISTS idx_sales_aug2025_date ON sales_aug2025(date_invoice);
CREATE INDEX IF NOT EXISTS idx_sales_sep2025_date ON sales_sep2025(date_invoice);
CREATE INDEX IF NOT EXISTS idx_sales_oct2025_date ON sales_oct2025(date_invoice);
CREATE INDEX IF NOT EXISTS idx_sales_nov2025_date ON sales_nov2025(date_invoice);
CREATE INDEX IF NOT EXISTS idx_sales_dec2025_date ON sales_dec2025(date_invoice);
```

## Materialized Views

### 1. mv_sales_metrics (Sales Analysis with Yesterday-Based Date Logic)
```sql
CREATE MATERIALIZED VIEW mv_sales_metrics AS
WITH date_ranges AS (
    SELECT 
        CURRENT_DATE - INTERVAL '1 day' as yesterday,
        CURRENT_DATE - INTERVAL '31 days' as date_30d_start,
        CURRENT_DATE - INTERVAL '61 days' as date_60d_start,
        CURRENT_DATE - INTERVAL '91 days' as date_90d_start,
        CURRENT_DATE - INTERVAL '366 days' as date_365d_start
),
sales_agg AS (
    SELECT 
        s.item_code,
        s.shop_code,
        -- Sales quantities (yesterday-based date ranges)
        SUM(CASE 
            WHEN s.date_invoice >= dr.date_30d_start 
             AND s.date_invoice <= dr.yesterday 
            THEN s.quantity 
            ELSE 0 
        END) as sales_30d,
        
        SUM(CASE 
            WHEN s.date_invoice >= dr.date_60d_start 
             AND s.date_invoice <= dr.yesterday 
            THEN s.quantity 
            ELSE 0 
        END) as sales_60d,
        
        SUM(CASE 
            WHEN s.date_invoice >= dr.date_90d_start 
             AND s.date_invoice <= dr.yesterday 
            THEN s.quantity 
            ELSE 0 
        END) as sales_90d,
        
        SUM(CASE 
            WHEN s.date_invoice >= dr.date_365d_start 
             AND s.date_invoice <= dr.yesterday 
            THEN s.quantity 
            ELSE 0 
        END) as sales_365d,
        
        -- Sales values (yesterday-based date ranges)
        SUM(CASE 
            WHEN s.date_invoice >= dr.date_30d_start 
             AND s.date_invoice <= dr.yesterday 
            THEN s.amount 
            ELSE 0 
        END) as value_30d,
        
        SUM(CASE 
            WHEN s.date_invoice >= dr.date_60d_start 
             AND s.date_invoice <= dr.yesterday 
            THEN s.amount 
            ELSE 0 
        END) as value_60d,
        
        SUM(CASE 
            WHEN s.date_invoice >= dr.date_90d_start 
             AND s.date_invoice <= dr.yesterday 
            THEN s.amount 
            ELSE 0 
        END) as value_90d,
        
        SUM(CASE 
            WHEN s.date_invoice >= dr.date_365d_start 
             AND s.date_invoice <= dr.yesterday 
            THEN s.amount 
            ELSE 0 
        END) as value_365d,
        
        MAX(s.date_invoice) as last_sale_date
    FROM sales s
    CROSS JOIN date_ranges dr
    GROUP BY s.item_code, s.shop_code
)
SELECT 
    item_code,
    shop_code,
    sales_30d,
    sales_60d,
    sales_90d,
    sales_365d,
    value_30d,
    value_60d,
    value_90d,
    value_365d,
    -- Rate of Sales: Average daily sales based on 90-day window
    ROUND(sales_90d / 90.0, 2) as ros,
    last_sale_date,
    CURRENT_TIMESTAMP as refreshed_at
FROM sales_agg;

-- Indexes for mv_sales_metrics
CREATE INDEX idx_mv_sales_item_shop ON mv_sales_metrics(item_code, shop_code);
CREATE INDEX idx_mv_sales_shop ON mv_sales_metrics(shop_code);
CREATE INDEX idx_mv_sales_item ON mv_sales_metrics(item_code);

COMMENT ON MATERIALIZED VIEW mv_sales_metrics IS 'Sales metrics with yesterday-based date ranges (excludes today)';
COMMENT ON COLUMN mv_sales_metrics.ros IS 'Rate of Sales = sales_90d / 90 - average daily sales';
COMMENT ON COLUMN mv_sales_metrics.sales_30d IS 'Sales from (yesterday-30) to yesterday, excluding today';
```

### 2. mv_sit_summary (Stock In Transit Aggregation)
```sql
CREATE MATERIALIZED VIEW mv_sit_summary AS
SELECT 
    shop_code,
    item_code,
    SUM(transit_qty) as total_sit,
    MAX(transit_date) as latest_transit_date,
    COUNT(*) as transit_count,
    CURRENT_TIMESTAMP as refreshed_at
FROM sit
GROUP BY shop_code, item_code;

-- Indexes for mv_sit_summary
CREATE INDEX idx_mv_sit_item_shop ON mv_sit_summary(item_code, shop_code);
CREATE INDEX idx_mv_sit_shop ON mv_sit_summary(shop_code);

COMMENT ON MATERIALIZED VIEW mv_sit_summary IS 'Aggregated Stock In Transit per item-shop combination';
```

### 3. mv_century_penetration (Main Analytical View)
```sql
CREATE MATERIALIZED VIEW mv_century_penetration AS
SELECT 
    -- Identification
    r.item_code,
    r.item_name,
    r.shop_code,
    r.dept,
    r.brand,
    
    -- Stock levels
    r.shop_stock as sih, -- Stock In Hand
    COALESCE(sit.total_sit, 0) as sit, -- Stock In Transit
    r.shop_stock + COALESCE(sit.total_sit, 0) as total_stock, -- SIH + SIT
    
    -- Sales metrics
    COALESCE(sm.sales_30d, 0) as sales_30d,
    COALESCE(sm.sales_60d, 0) as sales_60d,
    COALESCE(sm.sales_90d, 0) as sales_90d,
    COALESCE(sm.sales_365d, 0) as sales_365d,
    
    COALESCE(sm.value_30d, 0) as value_30d,
    COALESCE(sm.value_60d, 0) as value_60d,
    COALESCE(sm.value_90d, 0) as value_90d,
    COALESCE(sm.value_365d, 0) as value_365d,
    
    -- Rate of Sales (ROS)
    COALESCE(sm.ros, 0) as ros,
    
    -- Requirement for 21 days
    ROUND(COALESCE(sm.ros, 0) * 21, 2) as req_21_days,
    
    -- Stock position (SIH + SIT - Req 21 days)
    ROUND((r.shop_stock + COALESCE(sit.total_sit, 0)) - (COALESCE(sm.ros, 0) * 21), 2) as stock_variance,
    
    -- Stock status
    CASE 
        WHEN (r.shop_stock + COALESCE(sit.total_sit, 0)) - (COALESCE(sm.ros, 0) * 21) > 0 THEN 'OverStock'
        WHEN (r.shop_stock + COALESCE(sit.total_sit, 0)) - (COALESCE(sm.ros, 0) * 21) < 0 THEN 'UnderStock'
        ELSE 'Balanced'
    END as stock_status,
    
    -- Days of stock remaining
    CASE 
        WHEN COALESCE(sm.ros, 0) > 0 THEN 
            ROUND((r.shop_stock + COALESCE(sit.total_sit, 0)) / COALESCE(sm.ros, 0), 1)
        ELSE NULL
    END as days_of_stock,
    
    -- Reorder parameters
    r.min_nu,
    r.max_nu,
    r.reorder_qty,
    r.selling_price,
    
    -- Last sale info
    sm.last_sale_date,
    sit.latest_transit_date,
    
    -- Metadata
    CURRENT_TIMESTAMP as refreshed_at
    
FROM reorder_level r
LEFT JOIN mv_sales_metrics sm ON r.item_code = sm.item_code AND r.shop_code = sm.shop_code
LEFT JOIN mv_sit_summary sit ON r.item_code = sit.item_code AND r.shop_code = sit.shop_code
WHERE UPPER(r.brand) = 'CENTURY'; -- Filter for CENTURY brand only

-- Indexes for fast querying
CREATE INDEX idx_century_item_shop ON mv_century_penetration(item_code, shop_code);
CREATE INDEX idx_century_shop ON mv_century_penetration(shop_code);
CREATE INDEX idx_century_dept ON mv_century_penetration(dept);
CREATE INDEX idx_century_status ON mv_century_penetration(stock_status);
CREATE INDEX idx_century_ros ON mv_century_penetration(ros) WHERE ros > 0;
CREATE INDEX idx_century_understock ON mv_century_penetration(stock_variance) WHERE stock_status = 'UnderStock';
CREATE INDEX idx_century_overstock ON mv_century_penetration(stock_variance) WHERE stock_status = 'OverStock';

COMMENT ON MATERIALIZED VIEW mv_century_penetration IS 'Main analytical view for Century brand stock penetration dashboard';
COMMENT ON COLUMN mv_century_penetration.sih IS 'Stock In Hand - current shop inventory';
COMMENT ON COLUMN mv_century_penetration.sit IS 'Stock In Transit - items en route';
COMMENT ON COLUMN mv_century_penetration.total_stock IS 'Total available stock = SIH + SIT';
COMMENT ON COLUMN mv_century_penetration.ros IS 'Rate of Sales = average daily sales (90-day basis)';
COMMENT ON COLUMN mv_century_penetration.req_21_days IS 'Required stock for 21 days = ROS × 21';
COMMENT ON COLUMN mv_century_penetration.stock_variance IS 'Stock variance = Total Stock - Req 21 Days (positive = overstock, negative = understock)';
COMMENT ON COLUMN mv_century_penetration.days_of_stock IS 'Days of stock remaining = Total Stock / ROS';
```

## Refresh Function
```sql
CREATE OR REPLACE FUNCTION refresh_century_views()
RETURNS TEXT AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    result_text TEXT;
BEGIN
    start_time := CURRENT_TIMESTAMP;
    
    -- Drop indexes
    DROP INDEX IF EXISTS idx_mv_sales_item_shop;
    DROP INDEX IF EXISTS idx_mv_sit_item_shop;
    DROP INDEX IF EXISTS idx_century_item_shop;
    
    -- Refresh views (without CONCURRENTLY to handle duplicates)
    REFRESH MATERIALIZED VIEW mv_sales_metrics;
    REFRESH MATERIALIZED VIEW mv_sit_summary;
    REFRESH MATERIALIZED VIEW mv_century_penetration;
    
    -- Recreate indexes (non-unique)
    CREATE INDEX idx_mv_sales_item_shop ON mv_sales_metrics(item_code, shop_code);
    CREATE INDEX idx_mv_sit_item_shop ON mv_sit_summary(item_code, shop_code);
    CREATE INDEX idx_century_item_shop ON mv_century_penetration(item_code, shop_code);
    
    end_time := CURRENT_TIMESTAMP;
    
    result_text := 'All materialized views refreshed successfully at: ' || end_time || 
                   ' (Duration: ' || (EXTRACT(EPOCH FROM (end_time - start_time)))::TEXT || ' seconds)';
    
    RETURN result_text;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION refresh_century_views() IS 'Refreshes all materialized views with index drop/recreate to handle duplicates';
```

## Verification Queries

### Check View Freshness
```sql
SELECT 
    matviewname,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) as size,
    (SELECT refreshed_at FROM mv_sales_metrics LIMIT 1) as last_refresh
FROM pg_matviews 
WHERE schemaname = 'public' 
  AND matviewname LIKE 'mv_%'
ORDER BY matviewname;
```

### Check Row Counts
```sql
SELECT 
    'reorder_level' as table_name,
    COUNT(*) as total_rows,
    COUNT(*) FILTER (WHERE UPPER(brand) = 'CENTURY') as century_items
FROM reorder_level
UNION ALL
SELECT 
    'sit',
    COUNT(*),
    NULL
FROM sit
UNION ALL
SELECT 
    'sales',
    COUNT(*),
    NULL
FROM sales
UNION ALL
SELECT 
    'mv_sales_metrics',
    COUNT(*),
    NULL
FROM mv_sales_metrics
UNION ALL
SELECT 
    'mv_sit_summary',
    COUNT(*),
    NULL
FROM mv_sit_summary
UNION ALL
SELECT 
    'mv_century_penetration',
    COUNT(*),
    NULL
FROM mv_century_penetration;
```

### Verify Date Logic
```sql
SELECT 
    CURRENT_DATE as today,
    CURRENT_DATE - INTERVAL '1 day' as yesterday,
    CURRENT_DATE - INTERVAL '31 days' as start_30d,
    CURRENT_DATE - INTERVAL '91 days' as start_90d,
    COUNT(*) as items_with_sales,
    SUM(sales_30d) as total_sales_30d,
    SUM(sales_90d) as total_sales_90d,
    ROUND(AVG(ros), 2) as avg_ros
FROM mv_sales_metrics 
WHERE sales_30d > 0 OR sales_90d > 0;
```

### Check Stock Status Distribution
```sql
SELECT 
    stock_status,
    COUNT(*) as item_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage,
    ROUND(AVG(stock_variance), 2) as avg_variance,
    ROUND(AVG(ros), 2) as avg_ros
FROM mv_century_penetration
GROUP BY stock_status
ORDER BY 
    CASE stock_status 
        WHEN 'UnderStock' THEN 1
        WHEN 'Balanced' THEN 2
        WHEN 'OverStock' THEN 3
    END;
```

### Top UnderStock Items
```sql
SELECT 
    item_code,
    item_name,
    shop_code,
    sih,
    sit,
    total_stock,
    sales_30d,
    ros,
    req_21_days,
    stock_variance,
    days_of_stock
FROM mv_century_penetration
WHERE stock_status = 'UnderStock'
ORDER BY stock_variance ASC
LIMIT 20;
```

### Critical Items (No Stock + High Demand)
```sql
SELECT 
    item_code,
    item_name,
    shop_code,
    dept,
    sih,
    sit,
    sales_30d,
    sales_60d,
    sales_90d,
    ros,
    req_21_days
FROM mv_century_penetration
WHERE sih = 0 
  AND sit = 0 
  AND ros > 0
ORDER BY ros DESC
LIMIT 20;
```

## Maintenance Commands

### Refresh All Views Manually
```sql
-- Drop indexes
DROP INDEX IF EXISTS idx_mv_sales_item_shop;
DROP INDEX IF EXISTS idx_mv_sit_item_shop;
DROP INDEX IF EXISTS idx_century_item_shop;

-- Refresh views
REFRESH MATERIALIZED VIEW mv_sales_metrics;
REFRESH MATERIALIZED VIEW mv_sit_summary;
REFRESH MATERIALIZED VIEW mv_century_penetration;

-- Recreate indexes
CREATE INDEX idx_mv_sales_item_shop ON mv_sales_metrics(item_code, shop_code);
CREATE INDEX idx_mv_sit_item_shop ON mv_sit_summary(item_code, shop_code);
CREATE INDEX idx_century_item_shop ON mv_century_penetration(item_code, shop_code);

-- Show counts
SELECT 'mv_sales_metrics' as view_name, COUNT(*) FROM mv_sales_metrics
UNION ALL
SELECT 'mv_sit_summary', COUNT(*) FROM mv_sit_summary
UNION ALL
SELECT 'mv_century_penetration', COUNT(*) FROM mv_century_penetration;
```

### Vacuum and Analyze
```sql
VACUUM ANALYZE reorder_level;
VACUUM ANALYZE sit;
VACUUM ANALYZE sales;
VACUUM ANALYZE mv_sales_metrics;
VACUUM ANALYZE mv_sit_summary;
VACUUM ANALYZE mv_century_penetration;
```

### Check Partition Sizes
```sql
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public' 
  AND tablename LIKE 'sales_%2025'
ORDER BY tablename;
```

---

**Last Updated:** December 12, 2025  
**Database:** century_penetration  
**PostgreSQL Version:** 16+
