-- ============================================================
-- Update mv_sales_metrics to use Yesterday-based date ranges
-- This will recreate the view with updated logic
-- ============================================================

-- Drop the existing view
DROP MATERIALIZED VIEW IF EXISTS mv_sales_metrics CASCADE;

-- Recreate with updated date logic (yesterday-30 to yesterday)
CREATE MATERIALIZED VIEW mv_sales_metrics AS
WITH date_ranges AS (
    SELECT 
        CURRENT_DATE - INTERVAL '1 day' as yesterday,
        CURRENT_DATE - INTERVAL '30 days' as date_30d_start,
        CURRENT_DATE - INTERVAL '61 days' as date_60d_start,
        CURRENT_DATE - INTERVAL '91 days' as date_90d_start,
        CURRENT_DATE - INTERVAL '366 days' as date_365d_start
)
SELECT 
    s.shop_code,
    s.item_code,
    s.item_name,
    s.dept,
    
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
FROM sales s
CROSS JOIN date_ranges dr
GROUP BY s.shop_code, s.item_code, s.item_name, s.dept;

-- Recreate indexes
CREATE INDEX idx_mv_sales_item_shop ON mv_sales_metrics(item_code, shop_code);
CREATE INDEX idx_mv_sales_shop ON mv_sales_metrics(shop_code);
CREATE INDEX idx_mv_sales_item ON mv_sales_metrics(item_code);
CREATE INDEX idx_mv_sales_ros ON mv_sales_metrics(ros) WHERE ros > 0;

-- Now refresh the dependent views
REFRESH MATERIALIZED VIEW mv_sit_summary;
REFRESH MATERIALIZED VIEW mv_century_penetration;

-- Verify the changes
SELECT 
    'View updated successfully!' as status,
    COUNT(*) as total_records,
    MIN(refreshed_at) as refreshed_at
FROM mv_sales_metrics;
