-- ============================================================================
-- CENTURY PENETRATION - STOCKOUT TRACKING SYSTEM
-- ============================================================================

-- 1. Create current state table (UPSERT daily)
CREATE TABLE IF NOT EXISTS mv_century_penetration_test (
    shop_code VARCHAR(50),
    item_code VARCHAR(50),
    item_name VARCHAR(200),
    dept VARCHAR(100),
    brand VARCHAR(100),
    sih DECIMAL(15,2),
    sit DECIMAL(15,2),
    total_stock DECIMAL(15,2),
    sales_30d DECIMAL(15,2),
    sales_60d DECIMAL(15,2),
    sales_90d DECIMAL(15,2),
    ros DECIMAL(15,2),
    req_21_days DECIMAL(15,2),
    stock_variance DECIMAL(15,2),
    pack_size DECIMAL(15,2),
    days_of_stock DECIMAL(15,2),
    stock_status VARCHAR(20),
    selling_price DECIMAL(15,2),
    value_30d DECIMAL(15,2),
    last_sale_date DATE,
    
    -- NEW COLUMNS FOR STOCKOUT TRACKING
    stock_out_date DATE,              -- First date when SIH became <= 0
    days_out_of_stock INT DEFAULT 0,  -- Running count of consecutive days out
    ops_manager_name VARCHAR(100),    -- From opsmgr table
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (shop_code, item_code)
);

COMMENT ON TABLE mv_century_penetration_test IS 'Current state of Century items with stockout tracking (UPSERT daily)';
COMMENT ON COLUMN mv_century_penetration_test.stock_out_date IS 'Date when item first went out of stock (SIH <= 0)';
COMMENT ON COLUMN mv_century_penetration_test.days_out_of_stock IS 'Consecutive days item has been out of stock';
COMMENT ON COLUMN mv_century_penetration_test.ops_manager_name IS 'Operations Manager responsible for this shop';

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_test_stock_status ON mv_century_penetration_test(stock_status);
CREATE INDEX IF NOT EXISTS idx_test_stock_out_date ON mv_century_penetration_test(stock_out_date);
CREATE INDEX IF NOT EXISTS idx_test_ops_manager ON mv_century_penetration_test(ops_manager_name);
CREATE INDEX IF NOT EXISTS idx_test_dept ON mv_century_penetration_test(dept);
CREATE INDEX IF NOT EXISTS idx_test_shop ON mv_century_penetration_test(shop_code);


-- 2. Create daily stockout snapshot table (APPEND daily - for historical tracking)
CREATE TABLE IF NOT EXISTS century_stockout_daily_snapshot (
    snapshot_date DATE NOT NULL,
    shop_code VARCHAR(50) NOT NULL,
    item_code VARCHAR(50) NOT NULL,
    item_name VARCHAR(200),
    dept VARCHAR(100),
    sih DECIMAL(15,2),
    is_out_of_stock BOOLEAN,          -- TRUE if SIH <= 0
    ops_manager_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (snapshot_date, shop_code, item_code)
);

COMMENT ON TABLE century_stockout_daily_snapshot IS 'Daily snapshot of stock status for stockout % calculation (APPEND daily)';
COMMENT ON COLUMN century_stockout_daily_snapshot.is_out_of_stock IS 'TRUE when SIH <= 0';

-- Create indexes for fast aggregation
CREATE INDEX IF NOT EXISTS idx_snapshot_date ON century_stockout_daily_snapshot(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_snapshot_shop ON century_stockout_daily_snapshot(shop_code);
CREATE INDEX IF NOT EXISTS idx_snapshot_item ON century_stockout_daily_snapshot(item_code);
CREATE INDEX IF NOT EXISTS idx_snapshot_ops ON century_stockout_daily_snapshot(ops_manager_name);
CREATE INDEX IF NOT EXISTS idx_snapshot_dept ON century_stockout_daily_snapshot(dept);
CREATE INDEX IF NOT EXISTS idx_snapshot_stockout ON century_stockout_daily_snapshot(is_out_of_stock);


-- 3. Create materialized view for multi-dimensional stockout % analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_stockout_analysis AS
WITH date_params AS (
    SELECT 
        CURRENT_DATE - INTERVAL '1 day' as yesterday,
        CURRENT_DATE - INTERVAL '8 days' as last_7days,
        CURRENT_DATE - INTERVAL '31 days' as last_30days,
        DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 day') as current_month_start,
        DATE_TRUNC('year', CURRENT_DATE - INTERVAL '1 day') as current_year_start,
        (SELECT MIN(snapshot_date) FROM century_stockout_daily_snapshot) as first_snapshot_date
),
stockout_base AS (
    SELECT 
        s.shop_code,
        s.item_code,
        s.item_name,
        s.dept,
        s.ops_manager_name,
        s.snapshot_date,
        s.is_out_of_stock,
        EXTRACT(YEAR FROM s.snapshot_date) as year,
        EXTRACT(MONTH FROM s.snapshot_date) as month,
        DATE_TRUNC('week', s.snapshot_date) as week_start
    FROM century_stockout_daily_snapshot s
),
-- Overall stockout % (since first snapshot to yesterday)
overall_stats AS (
    SELECT 
        'Overall' as level_type,
        'All' as level_value,
        COUNT(DISTINCT snapshot_date) as total_days,
        COUNT(*) FILTER (WHERE is_out_of_stock = TRUE) as stockout_days,
        ROUND(
            (COUNT(*) FILTER (WHERE is_out_of_stock = TRUE)::DECIMAL / 
             NULLIF(COUNT(*), 0) * 100), 2
        ) as stockout_pct
    FROM stockout_base, date_params
    WHERE snapshot_date <= yesterday
),
-- Ops Manager wise
ops_manager_stats AS (
    SELECT 
        'Ops Manager' as level_type,
        ops_manager_name as level_value,
        COUNT(DISTINCT snapshot_date) as total_days,
        COUNT(*) FILTER (WHERE is_out_of_stock = TRUE) as stockout_days,
        ROUND(
            (COUNT(*) FILTER (WHERE is_out_of_stock = TRUE)::DECIMAL / 
             NULLIF(COUNT(*), 0) * 100), 2
        ) as stockout_pct
    FROM stockout_base, date_params
    WHERE snapshot_date <= yesterday
    GROUP BY ops_manager_name
),
-- Shop wise
shop_stats AS (
    SELECT 
        'Shop' as level_type,
        shop_code as level_value,
        COUNT(DISTINCT snapshot_date) as total_days,
        COUNT(*) FILTER (WHERE is_out_of_stock = TRUE) as stockout_days,
        ROUND(
            (COUNT(*) FILTER (WHERE is_out_of_stock = TRUE)::DECIMAL / 
             NULLIF(COUNT(*), 0) * 100), 2
        ) as stockout_pct
    FROM stockout_base, date_params
    WHERE snapshot_date <= yesterday
    GROUP BY shop_code
),
-- Department wise
dept_stats AS (
    SELECT 
        'Department' as level_type,
        dept as level_value,
        COUNT(DISTINCT snapshot_date) as total_days,
        COUNT(*) FILTER (WHERE is_out_of_stock = TRUE) as stockout_days,
        ROUND(
            (COUNT(*) FILTER (WHERE is_out_of_stock = TRUE)::DECIMAL / 
             NULLIF(COUNT(*), 0) * 100), 2
        ) as stockout_pct
    FROM stockout_base, date_params
    WHERE snapshot_date <= yesterday
    GROUP BY dept
),
-- Item wise
item_stats AS (
    SELECT 
        'Item' as level_type,
        item_code || ' - ' || item_name as level_value,
        COUNT(DISTINCT snapshot_date) as total_days,
        COUNT(*) FILTER (WHERE is_out_of_stock = TRUE) as stockout_days,
        ROUND(
            (COUNT(*) FILTER (WHERE is_out_of_stock = TRUE)::DECIMAL / 
             NULLIF(COUNT(*), 0) * 100), 2
        ) as stockout_pct
    FROM stockout_base, date_params
    WHERE snapshot_date <= yesterday
    GROUP BY item_code, item_name
),
-- Month wise
month_stats AS (
    SELECT 
        'Month' as level_type,
        TO_CHAR(DATE_TRUNC('month', snapshot_date), 'YYYY-MM') as level_value,
        COUNT(DISTINCT snapshot_date) as total_days,
        COUNT(*) FILTER (WHERE is_out_of_stock = TRUE) as stockout_days,
        ROUND(
            (COUNT(*) FILTER (WHERE is_out_of_stock = TRUE)::DECIMAL / 
             NULLIF(COUNT(*), 0) * 100), 2
        ) as stockout_pct
    FROM stockout_base, date_params
    WHERE snapshot_date <= yesterday
    GROUP BY DATE_TRUNC('month', snapshot_date)
),
-- Year wise
year_stats AS (
    SELECT 
        'Year' as level_type,
        year::TEXT as level_value,
        COUNT(DISTINCT snapshot_date) as total_days,
        COUNT(*) FILTER (WHERE is_out_of_stock = TRUE) as stockout_days,
        ROUND(
            (COUNT(*) FILTER (WHERE is_out_of_stock = TRUE)::DECIMAL / 
             NULLIF(COUNT(*), 0) * 100), 2
        ) as stockout_pct
    FROM stockout_base, date_params
    WHERE snapshot_date <= yesterday
    GROUP BY year
),
-- Last 7 days
last_7days_stats AS (
    SELECT 
        'Last 7 Days' as level_type,
        'All' as level_value,
        COUNT(DISTINCT snapshot_date) as total_days,
        COUNT(*) FILTER (WHERE is_out_of_stock = TRUE) as stockout_days,
        ROUND(
            (COUNT(*) FILTER (WHERE is_out_of_stock = TRUE)::DECIMAL / 
             NULLIF(COUNT(*), 0) * 100), 2
        ) as stockout_pct
    FROM stockout_base, date_params
    WHERE snapshot_date >= last_7days AND snapshot_date <= yesterday
),
-- Last 30 days
last_30days_stats AS (
    SELECT 
        'Last 30 Days' as level_type,
        'All' as level_value,
        COUNT(DISTINCT snapshot_date) as total_days,
        COUNT(*) FILTER (WHERE is_out_of_stock = TRUE) as stockout_days,
        ROUND(
            (COUNT(*) FILTER (WHERE is_out_of_stock = TRUE)::DECIMAL / 
             NULLIF(COUNT(*), 0) * 100), 2
        ) as stockout_pct
    FROM stockout_base, date_params
    WHERE snapshot_date >= last_30days AND snapshot_date <= yesterday
),
-- Weekly (by week start date)
weekly_stats AS (
    SELECT 
        'Weekly' as level_type,
        TO_CHAR(week_start, 'YYYY-"W"IW') as level_value,
        COUNT(DISTINCT snapshot_date) as total_days,
        COUNT(*) FILTER (WHERE is_out_of_stock = TRUE) as stockout_days,
        ROUND(
            (COUNT(*) FILTER (WHERE is_out_of_stock = TRUE)::DECIMAL / 
             NULLIF(COUNT(*), 0) * 100), 2
        ) as stockout_pct
    FROM stockout_base, date_params
    WHERE snapshot_date <= yesterday
    GROUP BY week_start
)
SELECT * FROM overall_stats
UNION ALL SELECT * FROM ops_manager_stats
UNION ALL SELECT * FROM shop_stats
UNION ALL SELECT * FROM dept_stats
UNION ALL SELECT * FROM item_stats
UNION ALL SELECT * FROM month_stats
UNION ALL SELECT * FROM year_stats
UNION ALL SELECT * FROM last_7days_stats
UNION ALL SELECT * FROM last_30days_stats
UNION ALL SELECT * FROM weekly_stats;

COMMENT ON MATERIALIZED VIEW mv_stockout_analysis IS 'Multi-dimensional stockout % analysis - refresh daily after snapshot';

-- Create index on the view
CREATE INDEX IF NOT EXISTS idx_stockout_analysis_level ON mv_stockout_analysis(level_type);
CREATE INDEX IF NOT EXISTS idx_stockout_analysis_value ON mv_stockout_analysis(level_value);


-- 4. Create detailed ops manager stockout view
CREATE OR REPLACE VIEW v_ops_manager_stockout_detail AS
SELECT 
    s.ops_manager_name,
    s.shop_code,
    COUNT(DISTINCT s.snapshot_date) as total_days_tracked,
    COUNT(DISTINCT s.item_code) as total_items,
    COUNT(*) FILTER (WHERE s.is_out_of_stock = TRUE) as total_stockout_events,
    COUNT(DISTINCT s.item_code) FILTER (WHERE s.is_out_of_stock = TRUE) as items_with_stockout,
    ROUND(
        (COUNT(*) FILTER (WHERE s.is_out_of_stock = TRUE)::DECIMAL / 
         NULLIF(COUNT(*), 0) * 100), 2
    ) as stockout_pct,
    MIN(s.snapshot_date) as tracking_start_date,
    MAX(s.snapshot_date) as tracking_end_date
FROM century_stockout_daily_snapshot s
WHERE s.snapshot_date <= CURRENT_DATE - INTERVAL '1 day'
GROUP BY s.ops_manager_name, s.shop_code
ORDER BY stockout_pct DESC NULLS LAST;

COMMENT ON VIEW v_ops_manager_stockout_detail IS 'Detailed stockout % by Ops Manager and Shop';


-- 5. Grant permissions
GRANT SELECT ON mv_century_penetration_test TO PUBLIC;
GRANT SELECT ON century_stockout_daily_snapshot TO PUBLIC;
GRANT SELECT ON mv_stockout_analysis TO PUBLIC;
GRANT SELECT ON v_ops_manager_stockout_detail TO PUBLIC;

-- Done
SELECT 'Stockout tracking system created successfully!' as status;
