-- ============================================================
-- STAGING ENVIRONMENT - TEST MATERIALIZED VIEWS
-- Location: D:\Dashboard Code\NO_WH\DS\nowhstock\staging\sql\
-- Purpose: Complete test environment for priority-to-priority transfers
-- ============================================================
-- 
-- CHANGES from production:
-- 1. All views renamed with _staging suffix
-- 2. Priority shops CAN be sources (transfer to OTHER priority shops)
-- 3. Enforces source_shop != dest_shop (no same-shop transfers)
-- 4. All indexes renamed with _staging suffix
--
-- Business Logic (SAME as production):
-- 1. Each source keeps minimum 30 units as safety stock
-- 2. Only excess above 30 units available for transfer  
-- 3. Destination cap = MAX(dest_sales_30d, dest_wh_grn_30d_sales)
-- 4. Window functions partition by (item_code, dest_shop)
-- 5. Expiry threshold < 30 days - safety buffer
-- 6. FEFO ordering (First Expiry First Out - oldest stock first)
-- 7. Cumulative allocation per destination <= cap
-- ============================================================

-- Drop existing staging views
DROP MATERIALIZED VIEW IF EXISTS mv_recommendations_complete_staging CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_slow_fast_moving_summary_staging CASCADE;

-- ============================================================
-- 1. SLOW/FAST MOVING SUMMARY (STAGING)
-- ============================================================

CREATE MATERIALIZED VIEW mv_slow_fast_moving_summary_staging AS
SELECT 
    TRIM(UPPER(item_code)) AS item_code,
    TRIM(UPPER(shop_code)) AS shop_code,
    sales_30d,
    sales_60d,
    sales_90d,
    CASE 
        WHEN sales_30d >= 30 THEN 'Fast Moving'
        WHEN sales_30d BETWEEN 10 AND 29 THEN 'Medium Moving'
        WHEN sales_30d BETWEEN 1 AND 9 THEN 'Slow Moving'
        ELSE 'No Sales'
    END AS movement_category,
    CASE 
        WHEN sales_30d >= 30 THEN 1
        WHEN sales_30d BETWEEN 10 AND 29 THEN 2
        WHEN sales_30d BETWEEN 1 AND 9 THEN 3
        ELSE 4
    END AS movement_rank
FROM (
    SELECT 
        item_code,
        shop_code,
        COALESCE(sales_30d, 0) AS sales_30d,
        COALESCE(sales_60d, 0) AS sales_60d,
        COALESCE(sales_90d, 0) AS sales_90d
    FROM inventory_master
) sub;

-- Indexes for staging slow/fast moving
CREATE INDEX idx_mv_slow_fast_staging_item_shop ON mv_slow_fast_moving_summary_staging(item_code, shop_code);
CREATE INDEX idx_mv_slow_fast_staging_category ON mv_slow_fast_moving_summary_staging(movement_category);

ANALYZE mv_slow_fast_moving_summary_staging;

-- ============================================================
-- 2. COMPLETE RECOMMENDATIONS (STAGING - WITH PRIORITY-TO-PRIORITY)
-- ============================================================

CREATE MATERIALIZED VIEW mv_recommendations_complete_staging AS
WITH 
-- Priority destination shops (ranked 1-11)
priority_shops AS (
    SELECT shop_code, priority_rank
    FROM (VALUES 
        ('SPN', 1), ('MSS', 2), ('LFS', 3), ('M03', 4), ('KAS', 5),
        ('MM1', 6), ('MM2', 7), ('FAR', 8), ('KS7', 9), ('WHL', 10), ('MM3', 11)
    ) AS t(shop_code, priority_rank)
),

-- Item WH GRN dates (item-level, warehouse received date)
item_wh_grn AS (
    SELECT 
        TRIM(UPPER(item_code)) AS item_code,
        MAX(wh_grn_date) AS wh_grn_date,
        MAX(wh_grn_date) + INTERVAL '30 days' AS wh_grn_plus_30
    FROM sup_shop_grn
    WHERE wh_grn_date IS NOT NULL
    GROUP BY TRIM(UPPER(item_code))
),

-- Get latest expiry date per item+shop
shop_expiry_latest AS (
    SELECT 
        TRIM(UPPER("ITEM_CODE")) AS item_code,
        TRIM(UPPER("SHOP_CODE")) AS shop_code,
        MAX("SHOP_EXPIRY_DATE") AS latest_expiry_date
    FROM shopexpiry
    GROUP BY TRIM(UPPER("ITEM_CODE")), TRIM(UPPER("SHOP_CODE"))
),

-- Source shops: ALL shops with transferable stock (INCLUDING priority shops)
-- STAGING CHANGE: Removed priority shop exclusion
sources AS (
    SELECT 
        im.itemcode AS item_code,
        im.itemname AS item_name,
        im.shopcode AS shop_code,
        im.shopstock AS stock,
        COALESCE(im.sales_30d_wh, 0) AS sales_30d,
        im.shopgrn_dt AS last_grn_date,
        CASE 
            WHEN im.shopgrn_dt IS NOT NULL 
            THEN (CURRENT_DATE - im.shopgrn_dt::date)::integer
            ELSE 0
        END AS grn_age,
        wh.wh_grn_date,
        im.groupp AS groups,
        im.subgroup AS sub_group,
        se.latest_expiry_date AS source_expiry_date,
        CASE 
            WHEN se.latest_expiry_date IS NOT NULL 
            THEN (se.latest_expiry_date::date - CURRENT_DATE)::integer
            ELSE NULL
        END AS source_expiry_days,
        CASE 
            WHEN se.latest_expiry_date IS NULL THEN NULL
            WHEN EXTRACT(YEAR FROM se.latest_expiry_date::date) > EXTRACT(YEAR FROM CURRENT_DATE) + 2 
            THEN 'Check expiry date'
            ELSE NULL
        END AS expiry_check,
        im.importexport AS item_type,
        im.suppliername AS supplier_name
    FROM inventory_master im
    LEFT JOIN item_wh_grn wh ON TRIM(UPPER(im.itemcode)) = wh.item_code
    LEFT JOIN shop_expiry_latest se ON TRIM(UPPER(im.itemcode)) = se.item_code 
        AND TRIM(UPPER(im.shopcode)) = se.shop_code
    WHERE im.shopstock > COALESCE(im.sales_30d_wh, 0)  -- Stock > sales (ALL shops, including priority)
      AND im.shopstock > 0
),

-- Destinations: Priority shops only
destinations AS (
    SELECT 
        im.itemcode AS item_code,
        im.shopcode AS shop_code,
        im.shopstock AS stock,
        COALESCE(im.sales_30d_wh, 0) AS sales_30d,
        im.shopgrn_dt AS last_grn_date,
        CASE 
            WHEN im.shopgrn_dt IS NOT NULL 
            THEN (CURRENT_DATE - im.shopgrn_dt::date)::integer
            ELSE 0
        END AS grn_age,
        wh.wh_grn_date,
        wh.wh_grn_plus_30,
        im.wh_grn_30d_sales AS wh_grn_30d_sales
    FROM inventory_master im
    LEFT JOIN item_wh_grn wh ON TRIM(UPPER(im.itemcode)) = wh.item_code
    WHERE im.shopcode IN (SELECT shop_code FROM priority_shops)
),

-- Pair sources with destinations (same item, different shops)
-- STAGING CHANGE: Added WHERE clause to enforce source != destination
source_dest_pairs AS (
    SELECT 
        src.*,
        dst.shop_code AS dest_shop,
        dst.sales_30d AS dest_sales_30d,
        dst.stock AS dest_stock,
        dst.last_grn_date AS dest_last_grn,
        dst.grn_age AS dest_grn_age,
        dst.wh_grn_date AS dest_wh_grn_date,
        dst.wh_grn_plus_30 AS dest_wh_grn_plus_30,
        COALESCE(dst.wh_grn_30d_sales, 0) AS dest_wh_grn_30d_sales,
        ps.priority_rank AS dest_priority_rank,
        GREATEST(COALESCE(dst.wh_grn_30d_sales, 0), COALESCE(dst.sales_30d, 0)) AS dest_capacity,
        LEAST(src.stock, src.stock) AS source_available_for_transfer,
        LEAST(
            src.stock,
            GREATEST(COALESCE(dst.wh_grn_30d_sales, 0), COALESCE(dst.sales_30d, 0))
        ) AS uncapped_qty
    FROM sources src
    INNER JOIN destinations dst ON TRIM(UPPER(src.item_code)) = TRIM(UPPER(dst.item_code))
    INNER JOIN priority_shops ps ON TRIM(UPPER(dst.shop_code)) = ps.shop_code
    WHERE TRIM(UPPER(src.shop_code)) != TRIM(UPPER(dst.shop_code))  -- STAGING: No same-shop transfers
),

-- Allocate stock with cumulative tracking per destination
allocated_recommendations AS (
    SELECT 
        item_code,
        item_name,
        groups,
        sub_group,
        item_type,
        supplier_name,
        shop_code AS source_shop,
        stock AS source_stock,
        sales_30d AS source_sales,
        last_grn_date AS source_last_grn,
        grn_age AS source_grn_age,
        wh_grn_date AS source_wh_grn_date,
        source_expiry_date,
        source_expiry_days,
        expiry_check,
        dest_shop,
        dest_stock,
        dest_sales_30d AS dest_sales,
        dest_last_grn,
        dest_grn_age,
        dest_wh_grn_date,
        dest_wh_grn_plus_30,
        dest_wh_grn_30d_sales,
        dest_capacity AS dest_cap,
        dest_priority_rank AS priority_rank,
        uncapped_qty,
        source_available_for_transfer AS source_available,
        
        -- Expiry status
        CASE 
            WHEN source_expiry_date IS NULL THEN 'OK'
            WHEN source_expiry_days < 0 THEN 'Expired'
            WHEN source_expiry_days BETWEEN 1 AND 29 THEN 'Expiring'
            WHEN source_expiry_days >= 30 THEN 'OK'
            ELSE 'OK'
        END AS expiry_status,
        
        -- Cumulative sum BEFORE this row
        SUM(
            CASE 
                WHEN source_expiry_date IS NULL THEN uncapped_qty
                WHEN source_expiry_days < 30 THEN 0
                WHEN source_expiry_days >= 30 THEN uncapped_qty
                ELSE 0
            END
        ) OVER (
            PARTITION BY item_code, dest_shop
            ORDER BY dest_priority_rank, grn_age DESC, COALESCE(source_expiry_days, 999999), stock DESC, shop_code
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS cumulative_before_this_row,
        
        -- Recommended qty (respect expiry + cap)
        CASE 
            WHEN source_expiry_date IS NOT NULL AND source_expiry_days < 30 THEN 0
            ELSE LEAST(
                uncapped_qty,
                GREATEST(
                    dest_capacity - COALESCE(
                        SUM(
                            CASE 
                                WHEN source_expiry_date IS NULL THEN uncapped_qty
                                WHEN source_expiry_days < 30 THEN 0
                                WHEN source_expiry_days >= 30 THEN uncapped_qty
                                ELSE 0
                            END
                        ) OVER (
                            PARTITION BY item_code, dest_shop
                            ORDER BY dest_priority_rank, grn_age DESC, COALESCE(source_expiry_days, 999999), stock DESC, shop_code
                            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                        ), 0
                    ), 0
                )
            )
        END AS allocated_qty
    FROM source_dest_pairs
),

-- Final recommendations with cumulative tracking
final_recommendations AS (
    SELECT 
        item_code,
        item_name,
        groups,
        sub_group,
        item_type,
        supplier_name,
        source_shop,
        source_stock,
        source_sales,
        source_last_grn,
        source_grn_age,
        source_wh_grn_date,
        source_expiry_date,
        source_expiry_days,
        expiry_check,
        expiry_status,
        dest_shop,
        dest_stock,
        dest_sales,
        dest_last_grn,
        dest_grn_age,
        dest_wh_grn_date,
        dest_wh_grn_plus_30,
        dest_wh_grn_30d_sales,
        dest_cap AS dest_sales_used,
        priority_rank,
        allocated_qty AS recommended_qty,
        SUM(allocated_qty) OVER (
            PARTITION BY item_code, dest_shop
            ORDER BY priority_rank, source_grn_age DESC, COALESCE(source_expiry_days, 999999), source_stock DESC, source_shop
        ) AS cumulative_qty,
        dest_cap - COALESCE(cumulative_before_this_row, 0) AS dest_remaining_cap_before,
        dest_stock + allocated_qty AS dest_updated_stock,
        CASE 
            WHEN dest_sales > 0 
            THEN ROUND((dest_stock + allocated_qty)::numeric / NULLIF(dest_sales, 0), 1)
            ELSE 999.9
        END AS dest_final_stock_days,
        CASE 
            WHEN allocated_qty = 0 AND source_expiry_days < 30 THEN 'Blocked: Expiring < 30 days'
            WHEN allocated_qty = 0 THEN 'Blocked: Capacity reached'
            WHEN allocated_qty > 0 THEN 'Transfer recommended'
            ELSE 'Review'
        END AS remark
    FROM allocated_recommendations
    WHERE allocated_qty > 0
)

-- Final output
SELECT * FROM final_recommendations
ORDER BY item_code, priority_rank, source_grn_age DESC;

-- ============================================================
-- 3. CREATE STAGING INDEXES
-- ============================================================

CREATE INDEX idx_mv_recs_staging_item_dest ON mv_recommendations_complete_staging(item_code, dest_shop);
CREATE INDEX idx_mv_recs_staging_source ON mv_recommendations_complete_staging(source_shop);
CREATE INDEX idx_mv_recs_staging_dest ON mv_recommendations_complete_staging(dest_shop);
CREATE INDEX idx_mv_recs_staging_item ON mv_recommendations_complete_staging(item_code);
CREATE INDEX idx_mv_recs_staging_qty ON mv_recommendations_complete_staging(recommended_qty);
CREATE INDEX idx_mv_recs_staging_groups ON mv_recommendations_complete_staging(groups);
CREATE INDEX idx_mv_recs_staging_subgroup ON mv_recommendations_complete_staging(sub_group);

-- Analyze staging views
ANALYZE mv_slow_fast_moving_summary_staging;
ANALYZE mv_recommendations_complete_staging;

-- ============================================================
-- 4. VERIFICATION QUERIES
-- ============================================================

-- Total recommendations
SELECT 
    'Total Recommendations' as metric,
    COUNT(*) as count
FROM mv_recommendations_complete_staging;

-- Priority shop sources (should be > 0 in staging)
SELECT 
    'Priority Shop Sources' as metric,
    COUNT(DISTINCT source_shop) as count
FROM mv_recommendations_complete_staging
WHERE source_shop IN ('SPN', 'MSS', 'LFS', 'M03', 'KAS', 'MM1', 'MM2', 'FAR', 'KS7', 'WHL', 'MM3');

-- Same-shop transfers (should be 0)
SELECT 
    'Same-Shop Transfers (should be 0)' as metric,
    COUNT(*) as count
FROM mv_recommendations_complete_staging
WHERE source_shop = dest_shop;

-- Top 10 priority shop sources
SELECT 
    source_shop,
    COUNT(*) as transfer_count,
    COUNT(DISTINCT item_code) as unique_items,
    SUM(recommended_qty) as total_qty
FROM mv_recommendations_complete_staging
WHERE source_shop IN ('SPN', 'MSS', 'LFS', 'M03', 'KAS', 'MM1', 'MM2', 'FAR', 'KS7', 'WHL', 'MM3')
GROUP BY source_shop
ORDER BY SUM(recommended_qty) DESC;

-- View sizes
SELECT 
    schemaname,
    matviewname,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||matviewname)) as data_size,
    pg_size_pretty(pg_indexes_size(schemaname||'.'||matviewname)) as index_size
FROM pg_matviews
WHERE matviewname LIKE '%_staging'
ORDER BY matviewname;

-- ============================================================
-- STAGING SETUP COMPLETE
-- ============================================================
-- Next steps:
-- 1. Run this SQL to create staging views
-- 2. Update nowhstock_ds_STAGING.py to use mv_recommendations_complete_staging
-- 3. Test the staging dashboard
-- 4. Compare results with production
-- 5. If approved, promote staging to production
-- ============================================================
