-- Fix for duplicate sales data in mv_sales_metrics
-- Recreates view with GROUP BY to handle duplicates properly

-- ============================================================
-- STEP 1: DROP EXISTING VIEW AND INDEXES
-- ============================================================
DROP MATERIALIZED VIEW IF EXISTS mv_sales_metrics CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_century_penetration CASCADE;

-- ============================================================
-- STEP 2: RECREATE mv_sales_metrics WITH DEDUPLICATION
-- ============================================================
CREATE MATERIALIZED VIEW mv_sales_metrics AS
WITH date_ranges AS (
    SELECT 
        CURRENT_DATE - INTERVAL '1 day' as yesterday,
        CURRENT_DATE - INTERVAL '30 days' as date_30d_start,
        CURRENT_DATE - INTERVAL '61 days' as date_60d_start,
        CURRENT_DATE - INTERVAL '91 days' as date_90d_start,
        CURRENT_DATE - INTERVAL '366 days' as date_365d_start
),
-- Deduplicate sales data first
deduplicated_sales AS (
    SELECT DISTINCT ON (shop_code, item_code, date_invoice)
        shop_code,
        item_code,
        item_name,
        dept,
        date_invoice,
        qty,
        net_sales
    FROM sales
    ORDER BY shop_code, item_code, date_invoice, loaded_at DESC
)
SELECT 
    s.shop_code,
    s.item_code,
    MAX(s.item_name) as item_name, -- Use MAX to get single value after dedup
    MAX(s.dept) as dept,
    
    -- 30 days sales (yesterday-30 to yesterday)
    SUM(CASE WHEN s.date_invoice >= dr.date_30d_start AND s.date_invoice <= dr.yesterday THEN s.qty ELSE 0 END) as sales_30d,
    SUM(CASE WHEN s.date_invoice >= dr.date_30d_start AND s.date_invoice <= dr.yesterday THEN s.net_sales ELSE 0 END) as value_30d,
    
    -- 60 days sales (yesterday-60 to yesterday)
    SUM(CASE WHEN s.date_invoice >= dr.date_60d_start AND s.date_invoice <= dr.yesterday THEN s.qty ELSE 0 END) as sales_60d,
    SUM(CASE WHEN s.date_invoice >= dr.date_60d_start AND s.date_invoice <= dr.yesterday THEN s.net_sales ELSE 0 END) as value_60d,
    
    -- 90 days sales (yesterday-90 to yesterday)
    SUM(CASE WHEN s.date_invoice >= dr.date_90d_start AND s.date_invoice <= dr.yesterday THEN s.qty ELSE 0 END) as sales_90d,
    SUM(CASE WHEN s.date_invoice >= dr.date_90d_start AND s.date_invoice <= dr.yesterday THEN s.net_sales ELSE 0 END) as value_90d,
    
    -- 365 days sales (yesterday-365 to yesterday)
    SUM(CASE WHEN s.date_invoice >= dr.date_365d_start AND s.date_invoice <= dr.yesterday THEN s.qty ELSE 0 END) as sales_365d,
    SUM(CASE WHEN s.date_invoice >= dr.date_365d_start AND s.date_invoice <= dr.yesterday THEN s.net_sales ELSE 0 END) as value_365d,
    
    -- Rate of Sales (ROS) = 90 days sales / 90
    ROUND(SUM(CASE WHEN s.date_invoice >= dr.date_90d_start AND s.date_invoice <= dr.yesterday THEN s.qty ELSE 0 END)::NUMERIC / 90, 2) as ros,
    
    -- Last sale date
    MAX(s.date_invoice) as last_sale_date,
    
    -- Metadata
    CURRENT_TIMESTAMP as refreshed_at
FROM deduplicated_sales s
CROSS JOIN date_ranges dr
GROUP BY s.shop_code, s.item_code;

-- Create NON-UNIQUE indexes (changed from UNIQUE to handle any remaining duplicates)
CREATE INDEX idx_mv_sales_item_shop ON mv_sales_metrics(item_code, shop_code);
CREATE INDEX idx_mv_sales_shop ON mv_sales_metrics(shop_code);
CREATE INDEX idx_mv_sales_item ON mv_sales_metrics(item_code);
CREATE INDEX idx_mv_sales_ros ON mv_sales_metrics(ros) WHERE ros > 0;

COMMENT ON MATERIALIZED VIEW mv_sales_metrics IS 'Pre-aggregated sales metrics with deduplication - refresh daily';

-- ============================================================
-- STEP 3: RECREATE mv_century_penetration
-- ============================================================
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
    
    -- Pack size
    COALESCE(r.pack_size, 0) as pack_size,
    
    -- Stock status (with pack_size tolerance)
    CASE 
        WHEN (r.shop_stock + COALESCE(sit.total_sit, 0)) - (COALESCE(sm.ros, 0) * 21) > COALESCE(r.pack_size, 0) THEN 'OverStock'
        WHEN (r.shop_stock + COALESCE(sit.total_sit, 0)) - (COALESCE(sm.ros, 0) * 21) < -COALESCE(r.pack_size, 0) THEN 'UnderStock'
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
    CURRENT_TIMESTAMP as refreshed_at,
    COALESCE(r.loaded_at, CURRENT_TIMESTAMP) as created_at
    
FROM reorder_level r
LEFT JOIN mv_sales_metrics sm ON r.item_code = sm.item_code AND r.shop_code = sm.shop_code
LEFT JOIN mv_sit_summary sit ON r.item_code = sit.item_code AND r.shop_code = sit.shop_code
WHERE UPPER(r.brand) = 'CENTURY'; -- Filter for CENTURY brand only

-- Create NON-UNIQUE indexes
CREATE INDEX idx_century_item_shop ON mv_century_penetration(item_code, shop_code);
CREATE INDEX idx_century_shop ON mv_century_penetration(shop_code);
CREATE INDEX idx_century_dept ON mv_century_penetration(dept);
CREATE INDEX idx_century_status ON mv_century_penetration(stock_status);
CREATE INDEX idx_century_ros ON mv_century_penetration(ros) WHERE ros > 0;
CREATE INDEX idx_century_understock ON mv_century_penetration(stock_variance) WHERE stock_status = 'UnderStock';
CREATE INDEX idx_century_overstock ON mv_century_penetration(stock_variance) WHERE stock_status = 'OverStock';

-- ============================================================
-- VERIFICATION
-- ============================================================
SELECT 'mv_sales_metrics' as view_name, COUNT(*) as row_count FROM mv_sales_metrics
UNION ALL
SELECT 'mv_century_penetration', COUNT(*) FROM mv_century_penetration;

SELECT 'VIEWS RECREATED SUCCESSFULLY WITH DEDUPLICATION!' as status;
