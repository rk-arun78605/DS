-- ============================================================
-- Serial Tracker: Dept Compliance Materialized View
-- Source: item_dept_map (WH) joined with serialno_check_yes_no
-- Refresh: run setup_dept_compliance_mv.py whenever item master changes
-- ============================================================

-- Step 1: Item dept mapping table (populated from century_penetration.sales via Python)
CREATE TABLE IF NOT EXISTS item_dept_map (
    item_code   TEXT PRIMARY KEY,
    dept        TEXT,
    grp         TEXT,
    sub_group   TEXT
);

CREATE INDEX IF NOT EXISTS idx_item_dept_map_dept     ON item_dept_map(dept);
CREATE INDEX IF NOT EXISTS idx_item_dept_map_grp      ON item_dept_map(grp);
CREATE INDEX IF NOT EXISTS idx_item_dept_map_subgroup ON item_dept_map(sub_group);

-- Step 2: Materialized view — compliance by dept/grp/sub_group/date/shop
DROP MATERIALIZED VIEW IF EXISTS mv_dept_compliance CASCADE;

CREATE MATERIALIZED VIEW mv_dept_compliance AS
SELECT
    DATE(s.bill_date)                                               AS activity_date,
    s.shop_code,
    COALESCE(m.dept,     'Unclassified')                           AS dept,
    COALESCE(m.grp,      'Unclassified')                           AS grp,
    COALESCE(m.sub_group,'Unclassified')                           AS sub_group,
    COUNT(*)                                                        AS total_serials,
    SUM(CASE WHEN UPPER(TRIM(s.serial_check)) = 'Y' THEN 1 ELSE 0 END) AS total_yes,
    ROUND(
        SUM(CASE WHEN UPPER(TRIM(s.serial_check)) = 'Y' THEN 1 ELSE 0 END)::NUMERIC
        / NULLIF(COUNT(*), 0) * 100, 1
    )                                                               AS compliance_pct
FROM serialno_check_yes_no s
LEFT JOIN item_dept_map m ON TRIM(s.item_code) = TRIM(m.item_code)
WHERE s.bill_date IS NOT NULL
  AND s.item_code IS NOT NULL
  AND TRIM(s.item_code) <> ''
GROUP BY DATE(s.bill_date), s.shop_code, m.dept, m.grp, m.sub_group
WITH DATA;

-- Step 3: Indexes for fast dashboard queries
CREATE INDEX IF NOT EXISTS idx_mv_dept_compliance_date     ON mv_dept_compliance(activity_date);
CREATE INDEX IF NOT EXISTS idx_mv_dept_compliance_dept     ON mv_dept_compliance(dept);
CREATE INDEX IF NOT EXISTS idx_mv_dept_compliance_grp      ON mv_dept_compliance(grp);
CREATE INDEX IF NOT EXISTS idx_mv_dept_compliance_subgroup ON mv_dept_compliance(sub_group);
CREATE INDEX IF NOT EXISTS idx_mv_dept_compliance_shop     ON mv_dept_compliance(shop_code);
CREATE INDEX IF NOT EXISTS idx_mv_dept_compliance_date_dept ON mv_dept_compliance(activity_date, dept);
