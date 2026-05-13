# Index Analysis & Optimization Report

## Current Situation

### ✅ Indexes Being Used (Keep These)
- `idx_sales_2025_date_invoice` - **73 scans** - ACTIVE, used for date filtering
- `idx_sales_2024_date_invoice_date` - **5 scans, 59M rows** - ACTIVE, date range queries
- `idx_sales_2025_date_invoice_date` - **1 scan, 1.5M rows** - ACTIVE, date range queries

### ❌ Unused Indexes (0 scans - Consider Dropping)
All other indexes (33 indexes) have **0 scans** = NOT BEING USED

## Problem: Redundant & Unused Indexes

### Why Indexes Are Not Used:
1. **Duplicate indexes**: Multiple indexes on same columns
2. **Query mismatch**: Queries don't match index column order
3. **Better alternatives**: PostgreSQL prefers simpler indexes
4. **Wrong column order**: Composite indexes need correct order (most selective first)

## Storage Cost
- **Total index size**: ~6.5 GB
- **Wasted space**: ~5.5 GB (unused indexes)
- **Maintenance overhead**: Slows down INSERT/UPDATE operations

## Recommended Actions

### 1. Drop Redundant/Unused Indexes

```sql
-- Drop unused indexes (safe - they're not being used anyway)

-- Duplicates of date_invoice
DROP INDEX IF EXISTS idx_sales_2024_date_invoice; -- Keep date_invoice_date instead
DROP INDEX IF EXISTS idx_sales_2025_date; -- Duplicate
DROP INDEX IF EXISTS idx_sales2024_date; -- Duplicate

-- Unused composite indexes
DROP INDEX IF EXISTS idx_sales_2024_dept_date;
DROP INDEX IF EXISTS idx_sales_2024_groups_date;
DROP INDEX IF EXISTS idx_sales_2024_group_date;
DROP INDEX IF EXISTS idx_sales_2024_subgroup_date;
DROP INDEX IF EXISTS idx_sales_2024_shop_date;
DROP INDEX IF EXISTS idx_sales_2024_dept_groups_date;
DROP INDEX IF EXISTS idx_sales_2024_dept_groups_subgroup_date;
DROP INDEX IF EXISTS idx_sales_2024_shop_groups_date;
DROP INDEX IF EXISTS idx_sales_2024_dept_group_date;

DROP INDEX IF EXISTS idx_sales_2025_dept_date;
DROP INDEX IF EXISTS idx_sales_2025_groups_date;
DROP INDEX IF EXISTS idx_sales_2025_group_date;
DROP INDEX IF EXISTS idx_sales_2025_subgroup_date;
DROP INDEX IF EXISTS idx_sales_2025_shop_date;
DROP INDEX IF EXISTS idx_sales_2025_dept_groups_date;
DROP INDEX IF EXISTS idx_sales_2025_dept_groups_subgroup_date;
DROP INDEX IF EXISTS idx_sales_2025_shop_groups_date;
DROP INDEX IF EXISTS idx_sales_2025_dept_group_date;

-- Unused aggregation indexes
DROP INDEX IF EXISTS idx_sales_2024_net_sales;
DROP INDEX IF EXISTS idx_sales_2024_qty;
DROP INDEX IF EXISTS idx_sales_2025_net_sales;

-- Unused covering/composite indexes
DROP INDEX IF EXISTS idx_sales_2024_date_sales_qty;
DROP INDEX IF EXISTS idx_sales_2025_date_sales_qty;
DROP INDEX IF EXISTS idx_sales_2024_date_month_year;
DROP INDEX IF EXISTS idx_sales_2025_date_month_year;
DROP INDEX IF EXISTS idx_sales_2025_covering;
DROP INDEX IF EXISTS idx_sales_2025_item_shop;
DROP INDEX IF EXISTS idx_sales2024_shop;
DROP INDEX IF EXISTS idx_sales2024_item;
DROP INDEX IF EXISTS idx_sales_2025_recent_date;

-- Reclaim disk space
VACUUM FULL sales_2024;
VACUUM FULL sales_2025;
```

### 2. Keep Only These Essential Indexes

```sql
-- Essential indexes that are actually being used or will be used:

-- 1. Date filtering (CRITICAL - actively used)
-- idx_sales_2024_date_invoice_date - KEEP (5 scans, 59M rows)
-- idx_sales_2025_date_invoice_date - KEEP (1 scan, 1.5M rows)
-- idx_sales_2025_date_invoice - KEEP (73 scans)

-- 2. Create optimized composite indexes for dashboard queries
CREATE INDEX IF NOT EXISTS idx_sales_2024_date_dept_sales 
ON sales_2024 (("DATE_INVOICE"::date), "DEPT", "NET_SALES");

CREATE INDEX IF NOT EXISTS idx_sales_2025_date_dept_sales 
ON sales_2025 (("DATE_INVOICE"::date), "DEPT", "NET_SALES");

CREATE INDEX IF NOT EXISTS idx_sales_2024_date_shop_sales 
ON sales_2024 (("DATE_INVOICE"::date), "SHOP_CODE", "NET_SALES");

CREATE INDEX IF NOT EXISTS idx_sales_2025_date_shop_sales 
ON sales_2025 (("DATE_INVOICE"::date), "SHOP_CODE", "NET_SALES");

-- 3. For group/subgroup analysis (covering index)
CREATE INDEX IF NOT EXISTS idx_sales_2024_date_group_subgroup_sales 
ON sales_2024 (("DATE_INVOICE"::date), "group_name", "subgroup_name", "NET_SALES", "QTY")
WHERE "group_name" IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sales_2025_date_group_subgroup_sales 
ON sales_2025 (("DATE_INVOICE"::date), "group_name", "subgroup_name", "NET_SALES", "QTY")
WHERE "group_name" IS NOT NULL;

ANALYZE sales_2024;
ANALYZE sales_2025;
```

## Expected Results After Cleanup

### Before:
- **37 indexes** on sales tables
- **6.5 GB** index storage
- **Slow writes** (every INSERT updates 37 indexes)
- **Confused query planner** (too many options)

### After:
- **8 indexes** on sales tables (11 removed)
- **~1 GB** index storage (85% reduction)
- **Faster writes** (fewer indexes to update)
- **Better query plans** (clear best choices)
- **Same or better query speed** (right indexes for actual queries)

## Why This Works Better

1. **Date-first indexes**: All queries filter by date first, so date should be first column
2. **Covering indexes**: Include all columns needed by query (no table lookup needed)
3. **Partial indexes**: WHERE clause reduces index size and speeds up matching queries
4. **No redundancy**: One index per query pattern, not multiple overlapping ones

## Monitoring After Changes

```sql
-- Check index usage after 24 hours
SELECT 
    schemaname,
    relname as tablename,
    indexrelname as indexname,
    idx_scan as scans,
    pg_size_pretty(pg_relation_size(indexrelid)) as size
FROM pg_stat_user_indexes
WHERE schemaname = 'public' 
  AND relname IN ('sales_2024', 'sales_2025')
ORDER BY idx_scan DESC;

-- All remaining indexes should show > 0 scans
```

## Dashboard Query Optimization

Your dashboard queries should use these patterns to utilize indexes:

```sql
-- GOOD: Date first, then filters
SELECT ... 
FROM sales_2025 
WHERE "DATE_INVOICE"::date BETWEEN '2025-01-01' AND '2025-11-22'
  AND "DEPT" = 'ELECTRONICS';

-- GOOD: Matches date_group_subgroup_sales index
SELECT 
    "group_name",
    "subgroup_name",
    SUM("NET_SALES"),
    SUM("QTY")
FROM sales_2025
WHERE "DATE_INVOICE"::date BETWEEN '2025-01-01' AND '2025-11-22'
  AND "group_name" IS NOT NULL
GROUP BY "group_name", "subgroup_name";
```

## Action Plan

1. **Backup first**: `pg_dump salesdata > backup.sql`
2. **Run DROP statements**: Remove unused indexes
3. **Create new optimized indexes**: Better composite indexes
4. **VACUUM FULL**: Reclaim disk space
5. **Monitor for 24 hours**: Check idx_scan values
6. **Verify performance**: Dashboard should be faster or same speed

## Risk Assessment

- **Risk**: LOW - We're only dropping indexes with 0 scans (not used)
- **Backup**: Always recommended before major changes
- **Rollback**: Can recreate indexes if needed (takes time)
- **Impact**: Positive - Faster writes, same/better read speed, less storage

Execute the SQL statements in order for best results.
