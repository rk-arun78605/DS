-- Recreate mv_century_penetration after CASCADE drop
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
    
    -- Stock policy components
    ROUND(COALESCE(sm.ros, 0) * 30, 2) as maintenance_stock_30d,
    ROUND(COALESCE(sm.ros, 0) * COALESCE(sp.lead_days, 14), 2) as lead_time_safety_stock,
    -- Requirement includes maintenance stock + lead-time safety stock
    ROUND(COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14)), 2) as req_30_days,
    ROUND(COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14)), 2) as req_21_days,
    ROUND(COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14)), 2) as optimum_stock_30d,
    COALESCE(sp.loading_status, 'ALTERNATE DAYS') as loading_status,
    COALESCE(sp.lead_days, 14) as lead,
    COALESCE(sp.lead_days, 14) as lead_days,
    ROUND(
        (COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14)))
        * CASE WHEN COALESCE(sp.lead_days, 14) <= 7 THEN 0.3 ELSE 0.5 END,
        2
    ) as min_threshold_qty,
    ROUND((CASE WHEN COALESCE(sp.lead_days, 14) <= 7 THEN 0.3 ELSE 0.5 END) * 100, 0) as min_threshold_pct,
    
    -- Stock position against optimum including lead-time safety stock
    ROUND((r.shop_stock + COALESCE(sit.total_sit, 0)) - (COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14))), 2) as stock_variance,
    
    -- Pack size
    COALESCE(r.pack_size, 0) as pack_size,
    
    -- Stock status based on lead-time-aware policy
    CASE 
        WHEN (r.shop_stock + COALESCE(sit.total_sit, 0)) < (
            (COALESCE(sm.ros, 0) * (30 + COALESCE(sp.lead_days, 14)))
            * CASE WHEN COALESCE(sp.lead_days, 14) <= 7 THEN 0.3 ELSE 0.5 END
        ) THEN 'UnderStock'
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
CREATE INDEX idx_century_item_shop ON mv_century_penetration(item_code, shop_code);
CREATE INDEX idx_century_shop ON mv_century_penetration(shop_code);
CREATE INDEX idx_century_dept ON mv_century_penetration(dept);
CREATE INDEX idx_century_status ON mv_century_penetration(stock_status);
CREATE INDEX idx_century_ros ON mv_century_penetration(ros) WHERE ros > 0;
CREATE INDEX idx_century_understock ON mv_century_penetration(stock_variance) WHERE stock_status = 'UnderStock';
CREATE INDEX idx_century_overstock ON mv_century_penetration(stock_variance) WHERE stock_status = 'OverStock';

-- Verify
SELECT COUNT(*) as total_rows FROM mv_century_penetration;
