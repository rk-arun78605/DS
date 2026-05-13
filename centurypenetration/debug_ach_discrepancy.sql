-- Debug ACH discrepancy between dashboard and Excel
-- Run this to see the raw data used in calculations

-- Step 1: Check date ranges being used
SELECT 
    'Date Ranges' as check_type,
    CURRENT_DATE as today,
    CURRENT_DATE - INTERVAL '1 day' as yesterday,
    CURRENT_DATE - INTERVAL '91 days' as date_90d_start,
    CURRENT_DATE - INTERVAL '31 days' as date_30d_start;

-- Step 2: Check raw sales data for ACH shop
SELECT 
    'Raw Sales Count' as check_type,
    shop_code,
    COUNT(*) as total_records,
    COUNT(DISTINCT (shop_code, item_code, date_invoice)) as unique_records,
    MIN(date_invoice) as earliest_date,
    MAX(date_invoice) as latest_date
FROM sales
WHERE shop_code = 'ACH'
GROUP BY shop_code;

-- Step 3: Check for duplicates in sales table
SELECT 
    'Duplicate Check' as check_type,
    shop_code,
    item_code,
    date_invoice,
    COUNT(*) as duplicate_count,
    STRING_AGG(DISTINCT loaded_at::text, ', ') as load_timestamps
FROM sales
WHERE shop_code = 'ACH'
GROUP BY shop_code, item_code, date_invoice
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC
LIMIT 10;

-- Step 4: Check 90-day sales calculation from mv_sales_metrics
SELECT 
    'MV Sales Metrics - ACH' as check_type,
    shop_code,
    item_code,
    item_name,
    sales_90d,
    ros,
    last_sale_date
FROM mv_sales_metrics
WHERE shop_code = 'ACH'
  AND sales_90d > 0
ORDER BY sales_90d DESC
LIMIT 20;

-- Step 5: Check reorder_level data for ACH
SELECT 
    'Reorder Level - ACH' as check_type,
    shop_code,
    item_code,
    item_name,
    shop_stock as sih,
    pack_size,
    brand
FROM reorder_level
WHERE shop_code = 'ACH'
  AND UPPER(brand) = 'CENTURY'
ORDER BY item_code
LIMIT 20;

-- Step 6: Check SIT data for ACH
SELECT 
    'SIT Summary - ACH' as check_type,
    shop_code,
    item_code,
    total_sit,
    latest_transit_date
FROM mv_sit_summary
WHERE shop_code = 'ACH'
ORDER BY total_sit DESC
LIMIT 20;

-- Step 7: Check final century_penetration calculation for ACH
SELECT 
    'Century Penetration - ACH' as check_type,
    shop_code,
    item_code,
    item_name,
    sih,
    sit,
    total_stock,
    sales_90d,
    ros,
    req_21_days,
    stock_variance,
    pack_size,
    stock_status,
    -- Show the calculation breakdown
    CONCAT(
        'SIH:', sih, 
        ' + SIT:', sit, 
        ' = ', total_stock,
        ' | ROS:', ros,
        ' × 21 = ', req_21_days,
        ' | Variance: ', stock_variance,
        ' | Pack:', pack_size
    ) as calculation_details
FROM mv_century_penetration
WHERE shop_code = 'ACH'
ORDER BY 
    CASE 
        WHEN stock_status = 'UnderStock' THEN 1
        WHEN stock_status = 'OverStock' THEN 2
        ELSE 3
    END,
    stock_variance ASC
LIMIT 30;

-- Step 8: Count by status for ACH
SELECT 
    'Status Summary - ACH' as check_type,
    stock_status,
    COUNT(*) as item_count,
    ROUND(AVG(stock_variance), 2) as avg_variance,
    ROUND(AVG(pack_size), 2) as avg_pack_size
FROM mv_century_penetration
WHERE shop_code = 'ACH'
GROUP BY stock_status
ORDER BY stock_status;

-- Step 9: Compare ROS calculation manually for ACH
WITH manual_calculation AS (
    SELECT 
        s.shop_code,
        s.item_code,
        MAX(s.item_name) as item_name,
        -- Count records in 90-day window
        COUNT(CASE WHEN s.date_invoice >= CURRENT_DATE - INTERVAL '91 days' 
                    AND s.date_invoice <= CURRENT_DATE - INTERVAL '1 day' 
                    THEN 1 END) as record_count_90d,
        -- Sum sales in 90-day window
        SUM(CASE WHEN s.date_invoice >= CURRENT_DATE - INTERVAL '91 days' 
                  AND s.date_invoice <= CURRENT_DATE - INTERVAL '1 day' 
                  THEN s.qty ELSE 0 END) as manual_sales_90d,
        -- Manual ROS
        ROUND(SUM(CASE WHEN s.date_invoice >= CURRENT_DATE - INTERVAL '91 days' 
                       AND s.date_invoice <= CURRENT_DATE - INTERVAL '1 day' 
                       THEN s.qty ELSE 0 END)::NUMERIC / 90, 2) as manual_ros
    FROM (
        SELECT DISTINCT ON (shop_code, item_code, date_invoice)
            shop_code, item_code, item_name, date_invoice, qty
        FROM sales
        WHERE shop_code = 'ACH'
        ORDER BY shop_code, item_code, date_invoice, loaded_at DESC
    ) s
    GROUP BY s.shop_code, s.item_code
)
SELECT 
    'Manual vs MV Comparison' as check_type,
    mc.shop_code,
    mc.item_code,
    mc.item_name,
    mc.manual_sales_90d,
    COALESCE(mv.sales_90d, 0) as mv_sales_90d,
    mc.manual_ros,
    COALESCE(mv.ros, 0) as mv_ros,
    CASE 
        WHEN mc.manual_sales_90d != COALESCE(mv.sales_90d, 0) THEN '❌ MISMATCH'
        ELSE '✅ MATCH'
    END as status
FROM manual_calculation mc
LEFT JOIN mv_sales_metrics mv ON mc.shop_code = mv.shop_code AND mc.item_code = mv.item_code
WHERE mc.manual_sales_90d > 0
ORDER BY ABS(mc.manual_sales_90d - COALESCE(mv.sales_90d, 0)) DESC
LIMIT 20;
