-- ============================================================
-- CENTURY STOCK PENETRATION DATABASE SCHEMA
-- Optimized for fast analytics and reporting
-- ============================================================

-- NOTE: Database should be created before running this script
-- Run: python setup_database.py

-- ============================================================
-- TABLE 1: REORDER LEVEL (Master Stock Data)
-- Stores current stock position, reorder levels, and item master
-- ============================================================
DROP TABLE IF EXISTS reorder_level CASCADE;

CREATE TABLE reorder_level (
    -- Primary identification
    item_code VARCHAR(50) NOT NULL,
    item_name VARCHAR(255),
    shop_code VARCHAR(50) NOT NULL,
    item_code_shop VARCHAR(100), -- Composite key item_code + shop_code
    
    -- Product classification
    dept VARCHAR(100),
    brand VARCHAR(100),
    
    -- Stock levels
    shop_stock INTEGER DEFAULT 0, -- Stock in Hand (SIH)
    wh_grn_shop_stock INTEGER DEFAULT 0,
    stc_nu INTEGER DEFAULT 0,
    
    -- Reorder parameters
    min_nu INTEGER DEFAULT 0,
    max_nu INTEGER DEFAULT 0,
    reorder_qty INTEGER DEFAULT 0,
    
    -- Pricing
    selling_price NUMERIC(12, 2),
    
    -- Metadata
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    PRIMARY KEY (item_code, shop_code)
);

-- Indexes for fast filtering (CENTURY brand specific)
CREATE INDEX idx_reorder_brand ON reorder_level(brand) WHERE UPPER(brand) = 'CENTURY';
CREATE INDEX idx_reorder_shop ON reorder_level(shop_code);
CREATE INDEX idx_reorder_dept ON reorder_level(dept);
CREATE INDEX idx_reorder_item ON reorder_level(item_code);
CREATE INDEX idx_reorder_stock ON reorder_level(shop_stock) WHERE shop_stock > 0;

-- Composite index for common query patterns
CREATE INDEX idx_reorder_item_shop ON reorder_level(item_code, shop_code);
CREATE INDEX idx_reorder_brand_shop ON reorder_level(brand, shop_code) WHERE UPPER(brand) = 'CENTURY';

COMMENT ON TABLE reorder_level IS 'Master stock and reorder level data for all items';
COMMENT ON COLUMN reorder_level.shop_stock IS 'Stock In Hand (SIH) - Current inventory at shop';
COMMENT ON COLUMN reorder_level.brand IS 'Brand name - Filter for CENTURY';


-- ============================================================
-- TABLE 2: SALES DATA (Transaction History)
-- Stores daily sales transactions for trend analysis
-- Partitioned by date for performance
-- ============================================================
DROP TABLE IF EXISTS sales CASCADE;

CREATE TABLE sales (
    -- Identification
    shop_code VARCHAR(50) NOT NULL,
    item_code VARCHAR(50) NOT NULL,
    item_name VARCHAR(255),
    
    -- Classification
    dept VARCHAR(100),
    groups VARCHAR(100),
    sub_group VARCHAR(100),
    
    -- Transaction details
    date_invoice DATE NOT NULL,
    qty INTEGER DEFAULT 0,
    net_sales NUMERIC(12, 2) DEFAULT 0,
    
    -- Metadata
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    PRIMARY KEY (shop_code, item_code, date_invoice)
) PARTITION BY RANGE (date_invoice);

-- Create partitions for each month (2025)
CREATE TABLE sales_jan2025 PARTITION OF sales
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE sales_feb2025 PARTITION OF sales
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

CREATE TABLE sales_mar2025 PARTITION OF sales
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

CREATE TABLE sales_apr2025 PARTITION OF sales
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');

CREATE TABLE sales_may2025 PARTITION OF sales
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');

CREATE TABLE sales_jun2025 PARTITION OF sales
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');

CREATE TABLE sales_jul2025 PARTITION OF sales
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');

CREATE TABLE sales_aug2025 PARTITION OF sales
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');

CREATE TABLE sales_sep2025 PARTITION OF sales
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');

CREATE TABLE sales_oct2025 PARTITION OF sales
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');

CREATE TABLE sales_nov2025 PARTITION OF sales
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE sales_dec2025 PARTITION OF sales
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Indexes on partitioned table
CREATE INDEX idx_sales_date ON sales(date_invoice DESC);
CREATE INDEX idx_sales_item ON sales(item_code);
CREATE INDEX idx_sales_shop ON sales(shop_code);
CREATE INDEX idx_sales_item_shop ON sales(item_code, shop_code);
CREATE INDEX idx_sales_date_item_shop ON sales(date_invoice, item_code, shop_code);

COMMENT ON TABLE sales IS 'Daily sales transactions partitioned by month for performance';


-- ============================================================
-- TABLE 3: SIT (Stock In Transit)
-- Items that are in transit from warehouse to shops
-- ============================================================
DROP TABLE IF EXISTS sit CASCADE;

CREATE TABLE sit (
    -- Identification
    shop_code VARCHAR(50) NOT NULL,
    item_code VARCHAR(50) NOT NULL,
    
    -- Transit details
    dt_trans_date DATE NOT NULL,
    nu_transit_qty INTEGER DEFAULT 0,
    
    -- Metadata
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    PRIMARY KEY (shop_code, item_code, dt_trans_date)
);

-- Indexes for fast aggregation
CREATE INDEX idx_sit_shop ON sit(shop_code);
CREATE INDEX idx_sit_item ON sit(item_code);
CREATE INDEX idx_sit_item_shop ON sit(item_code, shop_code);
CREATE INDEX idx_sit_date ON sit(dt_trans_date DESC);

COMMENT ON TABLE sit IS 'Stock In Transit (SIT) - Items moving from warehouse to shops';
COMMENT ON COLUMN sit.nu_transit_qty IS 'Quantity in transit';


-- ============================================================
-- MATERIALIZED VIEW: AGGREGATED SALES METRICS
-- Pre-calculated sales for 30, 60, 90, 365 days
-- Refresh daily for performance
-- ============================================================
DROP MATERIALIZED VIEW IF EXISTS mv_sales_metrics CASCADE;

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

-- Indexes on materialized view
CREATE UNIQUE INDEX idx_mv_sales_item_shop ON mv_sales_metrics(item_code, shop_code);
CREATE INDEX idx_mv_sales_shop ON mv_sales_metrics(shop_code);
CREATE INDEX idx_mv_sales_item ON mv_sales_metrics(item_code);
CREATE INDEX idx_mv_sales_ros ON mv_sales_metrics(ros) WHERE ros > 0;

COMMENT ON MATERIALIZED VIEW mv_sales_metrics IS 'Pre-aggregated sales metrics for 30/60/90/365 days - refresh daily';


-- ============================================================
-- MATERIALIZED VIEW: SIT AGGREGATED
-- Total stock in transit per item per shop
-- ============================================================
DROP MATERIALIZED VIEW IF EXISTS mv_sit_summary CASCADE;

CREATE MATERIALIZED VIEW mv_sit_summary AS
SELECT 
    shop_code,
    item_code,
    SUM(nu_transit_qty) as total_sit,
    MAX(dt_trans_date) as latest_transit_date,
    COUNT(*) as transit_count,
    CURRENT_TIMESTAMP as refreshed_at
FROM sit
WHERE dt_trans_date >= CURRENT_DATE - INTERVAL '90 days' -- Only recent transits
GROUP BY shop_code, item_code;

-- Indexes
CREATE UNIQUE INDEX idx_mv_sit_item_shop ON mv_sit_summary(item_code, shop_code);
CREATE INDEX idx_mv_sit_shop ON mv_sit_summary(shop_code);

COMMENT ON MATERIALIZED VIEW mv_sit_summary IS 'Aggregated SIT per item per shop - refresh daily';


-- ============================================================
-- MAIN ANALYTICAL VIEW: CENTURY STOCK PENETRATION REPORT
-- Combines all metrics for dashboard
-- ============================================================
DROP MATERIALIZED VIEW IF EXISTS mv_century_penetration CASCADE;

CREATE MATERIALIZED VIEW mv_century_penetration AS
WITH shop_policy AS (
    SELECT *
    FROM (VALUES
        ('ACH','DAILY',7,0.3),
        ('AF2','ALTERNATE DAYS',14,0.5),
        ('AFL','ALTERNATE DAYS',14,0.5),
        ('AMA','ALTERNATE DAYS',14,0.5),
        ('ASF','ALTERNATE DAYS',14,0.5),
        ('ASH','ALTERNATE DAYS',14,0.5),
        ('BIB','ALTERNATE DAYS',14,0.5),
        ('BOL','ALTERNATE DAYS',14,0.5),
        ('BRE','ALTERNATE DAYS',14,0.5),
        ('CAP','ALTERNATE DAYS',14,0.5),
        ('CLC','ALTERNATE DAYS',14,0.5),
        ('DNS','ALTERNATE DAYS',14,0.5),
        ('ELS','DAILY',7,0.3),
        ('FAR','ALTERNATE DAYS',14,0.5),
        ('GBA','ALTERNATE DAYS',14,0.5),
        ('HAA','ALTERNATE DAYS',14,0.5),
        ('HOE','ALTERNATE DAYS',14,0.5),
        ('HOV','ALTERNATE DAYS',14,0.5),
        ('KA2','ALTERNATE DAYS',14,0.5),
        ('KAS','DAILY',7,0.3),
        ('KFH','ALTERNATE DAYS',14,0.5),
        ('KS2','ALTERNATE DAYS',14,0.5),
        ('KS3','ALTERNATE DAYS',14,0.5),
        ('KS5','ALTERNATE DAYS',14,0.5),
        ('KSI','DAILY',7,0.3),
        ('KSS','ALTERNATE DAYS',14,0.5),
        ('LCC','ALTERNATE DAYS',14,0.5),
        ('LFS','DAILY',7,0.3),
        ('M06','DAILY',7,0.3),
        ('MAS','DAILY',7,0.3),
        ('MDN','ALTERNATE DAYS',14,0.5),
        ('MKL','ALTERNATE DAYS',14,0.5),
        ('MM1','DAILY',7,0.3),
        ('MM2','DAILY',7,0.3),
        ('MM3','ALTERNATE DAYS',14,0.5),
        ('MSS','DAILY',7,0.3),
        ('NAN','ALTERNATE DAYS',14,0.5),
        ('NKW','ALTERNATE DAYS',14,0.5),
        ('SD2','ALTERNATE DAYS',14,0.5),
        ('SPX','DAILY',7,0.3),
        ('SU2','ALTERNATE DAYS',14,0.5),
        ('TKD','ALTERNATE DAYS',14,0.5),
        ('TKW','ALTERNATE DAYS',14,0.5),
        ('TM2','ALTERNATE DAYS',14,0.5),
        ('TML','DAILY',7,0.3),
        ('TMP','DAILY',7,0.3),
        ('WHL','ALTERNATE DAYS',14,0.5),
        ('WNC','ALTERNATE DAYS',14,0.5)
    ) AS t(shop_code, loading_status, lead_days, min_threshold_pct)
)
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
    
    -- Optimum Stock = ROS × (30 maintenance days + lead days per shop policy)
    -- e.g. DAILY shop (lead=7): ROS×37  |  ALTERNATE DAYS shop (lead=14): ROS×44
    ROUND(COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14)), 2) as req_21_days,
    ROUND(COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14)), 2) as req_30_days,
    ROUND(COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14)), 2) as optimum_stock_30d,
    COALESCE(sp.loading_status, 'ALTERNATE DAYS') as loading_status,
    COALESCE(sp.lead_days, 14) as lead,
    COALESCE(sp.lead_days, 14) as lead_days,
    ROUND((COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14))) * COALESCE(sp.min_threshold_pct, 0.5), 2) as min_threshold_qty,
    ROUND(COALESCE(sp.min_threshold_pct, 0.5) * 100, 0) as min_threshold_pct,
    
    -- Stock variance vs optimum (30 days + lead days)
    ROUND((r.shop_stock + COALESCE(sit.total_sit, 0)) - (COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14))), 2) as stock_variance,

    -- Pack size retained for backward compatibility in UI tables
    COALESCE(r.pack_size, 0) as pack_size,
    
    -- Stock status
    CASE 
        WHEN (r.shop_stock + COALESCE(sit.total_sit, 0)) < ((COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14))) * COALESCE(sp.min_threshold_pct, 0.5)) THEN 'UnderStock'
        WHEN (r.shop_stock + COALESCE(sit.total_sit, 0)) > (COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14))) THEN 'OverStock'
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
LEFT JOIN shop_policy sp ON UPPER(TRIM(r.shop_code)) = sp.shop_code
LEFT JOIN mv_sales_metrics sm ON r.item_code = sm.item_code AND r.shop_code = sm.shop_code
LEFT JOIN mv_sit_summary sit ON r.item_code = sit.item_code AND r.shop_code = sit.shop_code
WHERE UPPER(r.brand) = 'CENTURY'; -- Filter for CENTURY brand only

-- Indexes for fast querying
CREATE UNIQUE INDEX idx_century_item_shop ON mv_century_penetration(item_code, shop_code);
CREATE INDEX idx_century_shop ON mv_century_penetration(shop_code);
CREATE INDEX idx_century_dept ON mv_century_penetration(dept);
CREATE INDEX idx_century_status ON mv_century_penetration(stock_status);
CREATE INDEX idx_century_ros ON mv_century_penetration(ros) WHERE ros > 0;
CREATE INDEX idx_century_understock ON mv_century_penetration(stock_variance) WHERE stock_status = 'UnderStock';
CREATE INDEX idx_century_overstock ON mv_century_penetration(stock_variance) WHERE stock_status = 'OverStock';

COMMENT ON MATERIALIZED VIEW mv_century_penetration IS 'Main analytical view for Century brand stock penetration dashboard';


-- ============================================================
-- REFRESH FUNCTION FOR ALL MATERIALIZED VIEWS
-- Call this daily or on-demand
-- ============================================================
CREATE OR REPLACE FUNCTION refresh_century_views()
RETURNS TEXT AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_sales_metrics;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_sit_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_century_penetration;
    
    RETURN 'All materialized views refreshed at: ' || CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION refresh_century_views IS 'Refresh all materialized views for Century dashboard';


-- ============================================================
-- PERMISSIONS
-- PostgreSQL by default grants all permissions to postgres user
-- ============================================================


-- ============================================================
-- SAMPLE QUERIES FOR DASHBOARD
-- ============================================================

-- Query 1: Overview metrics
COMMENT ON TABLE reorder_level IS '
-- Total items tracked
SELECT COUNT(DISTINCT item_code) as total_items FROM mv_century_penetration;

-- Total shops
SELECT COUNT(DISTINCT shop_code) as total_shops FROM mv_century_penetration;

-- Understock items
SELECT COUNT(*) as understock_items FROM mv_century_penetration WHERE stock_status = ''UnderStock'';

-- Overstock items
SELECT COUNT(*) as overstock_items FROM mv_century_penetration WHERE stock_status = ''OverStock'';
';

-- Query 2: Top understock items
COMMENT ON TABLE sales IS '
SELECT item_code, item_name, shop_code, sih, sit, req_21_days, stock_variance
FROM mv_century_penetration
WHERE stock_status = ''UnderStock''
ORDER BY stock_variance ASC
LIMIT 20;
';

-- Query 3: Shop-wise stock status
COMMENT ON TABLE sit IS '
SELECT 
    shop_code,
    COUNT(*) as total_items,
    SUM(CASE WHEN stock_status = ''UnderStock'' THEN 1 ELSE 0 END) as understock,
    SUM(CASE WHEN stock_status = ''OverStock'' THEN 1 ELSE 0 END) as overstock,
    SUM(CASE WHEN stock_status = ''Balanced'' THEN 1 ELSE 0 END) as balanced
FROM mv_century_penetration
GROUP BY shop_code
ORDER BY understock DESC;
';

-- ============================================================
-- VACUUM AND ANALYZE FOR OPTIMAL PERFORMANCE
-- Note: Run these manually after data load, not during schema creation
-- ============================================================
-- VACUUM ANALYZE reorder_level;
-- VACUUM ANALYZE sales;
-- VACUUM ANALYZE sit;
