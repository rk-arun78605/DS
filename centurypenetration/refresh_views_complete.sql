-- Century Stock Penetration - Complete Daily Refresh
-- Handles duplicates by dropping indexes before refresh

-- ============================================================
-- STEP 1: DROP ALL INDEXES
-- ============================================================
DROP INDEX IF EXISTS idx_mv_sales_item_shop;
DROP INDEX IF EXISTS idx_mv_sit_item_shop;
DROP INDEX IF EXISTS idx_mv_century_item_shop;

-- ============================================================
-- STEP 2: REFRESH ALL MATERIALIZED VIEWS (WITHOUT CONCURRENTLY)
-- ============================================================

-- Refresh sales metrics
REFRESH MATERIALIZED VIEW mv_sales_metrics;

-- Refresh SIT summary
REFRESH MATERIALIZED VIEW mv_sit_summary;

-- Refresh main penetration view
REFRESH MATERIALIZED VIEW mv_century_penetration;

-- ============================================================
-- STEP 3: RECREATE INDEXES (NON-UNIQUE - Views already handle deduplication)
-- ============================================================

-- Index on sales metrics (non-unique, deduplication handled in view)
CREATE INDEX IF NOT EXISTS idx_mv_sales_item_shop 
ON mv_sales_metrics(item_code, shop_code);

-- Index on SIT summary
CREATE INDEX IF NOT EXISTS idx_mv_sit_item_shop 
ON mv_sit_summary(item_code, shop_code);

-- Index on main penetration view
CREATE INDEX IF NOT EXISTS idx_mv_century_item_shop 
ON mv_century_penetration(item_code, shop_code);

-- ============================================================
-- VERIFICATION: Check row counts
-- ============================================================
SELECT 
    'mv_sales_metrics' as view_name, 
    COUNT(*) as row_count 
FROM mv_sales_metrics
UNION ALL
SELECT 
    'mv_sit_summary', 
    COUNT(*) 
FROM mv_sit_summary
UNION ALL
SELECT 
    'mv_century_penetration', 
    COUNT(*) 
FROM mv_century_penetration;

-- Show completion message
SELECT 'ALL VIEWS REFRESHED SUCCESSFULLY!' as status;
