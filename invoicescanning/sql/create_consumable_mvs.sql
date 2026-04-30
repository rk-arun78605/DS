-- ============================================================
-- Materialized Views for Invoice Scanning Dashboard (WH database)
-- Run ONCE to create all views.
-- Refresh daily with:  python invoicescanning\batch\refresh_consumable_mvs.py
-- Or directly:         psql -U postgres -d WH -p 3307 -f refresh_consumable_mvs.sql
-- ============================================================

-- Drop in reverse dependency order
DROP MATERIALIZED VIEW IF EXISTS mv_wh_alerts_daily CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_wh_manager_handover_daily CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_wh_erp_test_bills_cashier_daily CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_wh_erp_cashier_sessions_daily CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_wh_invoices_agg_daily CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_wh_erp_daily CASCADE;

-- ============================================================
-- MV 1: ERP daily aggregates per shop
--   Used by: load_owner_view, load_daily_diff_series
-- ============================================================
CREATE MATERIALIZED VIEW mv_wh_erp_daily AS
SELECT
    e.invdate::date                                                AS bill_date,
    UPPER(TRIM(e.store_code))                                      AS shop_code,
    COUNT(e.invno)                                                 AS erp_nob,
    SUM(COALESCE(e.amt, 0)::numeric)                               AS erp_nob_ghs,
    COUNT(*) FILTER (WHERE COALESCE(e.amt, 0) = 0.01)             AS test_bills
FROM erpdata e
WHERE NULLIF(TRIM(COALESCE(e.store_code, '')), '') IS NOT NULL
  AND NULLIF(TRIM(COALESCE(e.invno::text, '')), '') IS NOT NULL
GROUP BY e.invdate::date, UPPER(TRIM(e.store_code))
WITH DATA;

CREATE UNIQUE INDEX uix_mv_wh_erp_daily ON mv_wh_erp_daily (bill_date, shop_code);


-- ============================================================
-- MV 2: Invoices (invoices + invoices_manager, deduped) daily aggregates per shop
--   Used by: load_owner_view, load_daily_diff_series
-- ============================================================
CREATE MATERIALIZED VIEW mv_wh_invoices_agg_daily AS
WITH consumable_tills AS (
    SELECT * FROM (VALUES
        ('SPN', 34), ('SPN', 32), ('SPN', 33),
        ('LFS', 23), ('LFS', 26),
        ('MSS', 13), ('MSS', 14),
        ('MM1',  1), ('MM1', 13),
        ('MM2', 10), ('MM2',  9),
        ('WHL', 14)
    ) AS t(shop_code, till_no)
),
invoice_union AS (
    SELECT
        i.invdate::date                                                              AS bill_date,
        UPPER(TRIM(i.store_code))                                                   AS shop_code,
        TRIM(COALESCE(i.invno::text, ''))                                           AS invno,
        COALESCE(i.amt, 0)::numeric                                                 AS amt,
        CASE
            WHEN NULLIF(REGEXP_REPLACE(COALESCE(i.tillno::text, ''), '[^0-9]', '', 'g'), '') IS NULL THEN NULL
            ELSE NULLIF(REGEXP_REPLACE(COALESCE(i.tillno::text, ''), '[^0-9]', '', 'g'), '')::int
        END                                                                         AS till_no,
        1                                                                           AS src_priority
    FROM invoices i
    WHERE NULLIF(TRIM(COALESCE(i.invno::text, '')), '') IS NOT NULL

    UNION ALL

    SELECT
        m.invdate::date,
        UPPER(TRIM(m.store_code)),
        TRIM(COALESCE(m.invno::text, '')),
        COALESCE(m.amt, 0)::numeric,
        CASE
            WHEN NULLIF(REGEXP_REPLACE(COALESCE(m.tillno::text, ''), '[^0-9]', '', 'g'), '') IS NULL THEN NULL
            ELSE NULLIF(REGEXP_REPLACE(COALESCE(m.tillno::text, ''), '[^0-9]', '', 'g'), '')::int
        END,
        2
    FROM invoices_manager m
    WHERE NULLIF(TRIM(COALESCE(m.invno::text, '')), '') IS NOT NULL
),
deduped AS (
    SELECT bill_date, shop_code, invno, amt, till_no
    FROM (
        SELECT
            bill_date, shop_code, invno, amt, till_no,
            ROW_NUMBER() OVER (PARTITION BY bill_date, shop_code, invno ORDER BY src_priority) AS rn
        FROM invoice_union
    ) d
    WHERE rn = 1
),
classified AS (
    SELECT
        d.bill_date,
        d.shop_code,
        d.amt,
        CASE WHEN c.shop_code IS NOT NULL THEN 1 ELSE 0 END AS is_consumable
    FROM deduped d
    LEFT JOIN consumable_tills c ON c.shop_code = d.shop_code AND c.till_no = d.till_no
)
SELECT
    bill_date,
    shop_code,
    COUNT(*) FILTER (WHERE is_consumable = 0)   AS scan_nob,
    SUM(amt)  FILTER (WHERE is_consumable = 0)   AS scanned_nob_ghs,
    COUNT(*) FILTER (WHERE is_consumable = 1)   AS consumable_till_nob,
    SUM(amt)  FILTER (WHERE is_consumable = 1)   AS consumable_nob_ghs
FROM classified
GROUP BY bill_date, shop_code
WITH DATA;

CREATE UNIQUE INDEX uix_mv_wh_invoices_agg_daily ON mv_wh_invoices_agg_daily (bill_date, shop_code);


-- ============================================================
-- MV 3: ERP cashier sessions — per day, shop, cashier, till
--   Used by: load_owner_view (test_bill_not_generated),
--            load_shopwise_test_bill_analysis (cashier_login)
-- ============================================================
CREATE MATERIALIZED VIEW mv_wh_erp_cashier_sessions_daily AS
WITH base AS (
    SELECT
        e.invdate::date                                                              AS bill_date,
        UPPER(TRIM(e.store_code))                                                   AS shop_code,
        UPPER(TRIM(COALESCE(e.cashier, '')))                                        AS cashier_name,
        CASE
            WHEN NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '') IS NULL THEN NULL
            ELSE NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '')::int
        END                                                                         AS till_no,
        COALESCE(e.amt, 0)::numeric                                                 AS amt
    FROM erpdata e
    WHERE NULLIF(TRIM(COALESCE(e.cashier, '')), '') IS NOT NULL
      AND NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '') IS NOT NULL
      AND NULLIF(TRIM(COALESCE(e.invno::text, '')), '') IS NOT NULL
)
SELECT
    bill_date,
    shop_code,
    cashier_name,
    till_no,
    MAX(CASE WHEN amt = 0.01 THEN 1 ELSE 0 END) AS has_test_bill
FROM base
GROUP BY bill_date, shop_code, cashier_name, till_no
WITH DATA;

CREATE UNIQUE INDEX uix_mv_wh_erp_cashier_sessions ON mv_wh_erp_cashier_sessions_daily (bill_date, shop_code, cashier_name, till_no);
CREATE INDEX ix_mv_wh_erp_cashier_sessions_shop ON mv_wh_erp_cashier_sessions_daily (bill_date, shop_code);


-- ============================================================
-- MV 4: ERP test-bill rows per day, shop, cashier
--   Used by: load_shopwise_test_bill_analysis (test_bills_generated, unique_test_bills_generated)
-- ============================================================
CREATE MATERIALIZED VIEW mv_wh_erp_test_bills_cashier_daily AS
SELECT
    e.invdate::date                                     AS bill_date,
    UPPER(TRIM(e.store_code))                           AS shop_code,
    UPPER(TRIM(COALESCE(e.cashier, '')))                AS cashier_name,
    COUNT(*)                                            AS test_bill_count
FROM erpdata e
WHERE COALESCE(e.amt, 0) = 0.01
  AND NULLIF(TRIM(COALESCE(e.invno::text, '')), '') IS NOT NULL
  AND NULLIF(TRIM(COALESCE(e.cashier, '')), '') IS NOT NULL
GROUP BY e.invdate::date, UPPER(TRIM(e.store_code)), UPPER(TRIM(COALESCE(e.cashier, '')))
WITH DATA;

CREATE UNIQUE INDEX uix_mv_wh_erp_test_bills_cashier ON mv_wh_erp_test_bills_cashier_daily (bill_date, shop_code, cashier_name);
CREATE INDEX ix_mv_wh_erp_test_bills_shop ON mv_wh_erp_test_bills_cashier_daily (bill_date, shop_code);


-- ============================================================
-- MV 5: Manager handover — distinct cashier per day per shop
--   Used by: load_shopwise_test_bill_analysis (handover_test_bill_to_manager)
-- ============================================================
CREATE MATERIALIZED VIEW mv_wh_manager_handover_daily AS
SELECT
    m.invdate::date                                     AS bill_date,
    UPPER(TRIM(m.store_code))                           AS shop_code,
    UPPER(TRIM(COALESCE(m.cashier, '')))                AS cashier_name
FROM invoices_manager m
WHERE NULLIF(TRIM(COALESCE(m.invno::text, '')), '') IS NOT NULL
  AND NULLIF(TRIM(COALESCE(m.cashier, '')), '') IS NOT NULL
GROUP BY m.invdate::date, UPPER(TRIM(m.store_code)), UPPER(TRIM(COALESCE(m.cashier, '')))
WITH DATA;

CREATE UNIQUE INDEX uix_mv_wh_manager_handover ON mv_wh_manager_handover_daily (bill_date, shop_code, cashier_name);
CREATE INDEX ix_mv_wh_manager_handover_shop ON mv_wh_manager_handover_daily (bill_date, shop_code);


-- ============================================================
-- MV 6: Alerts — pre-normalised, per day, shop, alert type
--   Used by: load_alert_type_comparison
-- ============================================================
CREATE MATERIALIZED VIEW mv_wh_alerts_daily AS
WITH norm AS (
    SELECT
        UPPER(TRIM(a.a_store_code))                 AS shop_code,
        CASE
            WHEN LOWER(TRIM(a.a_type)) IN ('bill date mismatched', 'bill date mismatch') THEN 'Bill Date Mismatch'
            WHEN LOWER(TRIM(a.a_type)) = 'test bill'                                     THEN 'Test Bill'
            WHEN LOWER(TRIM(a.a_type)) IN ('duplicate', 'duplicate bill')                THEN 'Duplicate'
            WHEN LOWER(TRIM(a.a_type)) IN ('wrong shop', 'wrong sho')                    THEN 'Wrong Shop'
            WHEN LOWER(TRIM(a.a_type)) = 'high bill amount'                              THEN 'High Bill Amount'
            ELSE INITCAP(TRIM(COALESCE(a.a_type, 'Unknown')))
        END                                         AS a_type_normalized,
        -- Use scanned_date when available, otherwise fall back to entry time.
        COALESCE(a.scanned_date::date, a.a_entrytime::date)            AS alert_date
    FROM alerts a
    WHERE NULLIF(TRIM(COALESCE(a.a_store_code, '')), '') IS NOT NULL
)
SELECT
    shop_code,
    a_type_normalized,
    alert_date,
    COUNT(*) AS alert_count
FROM norm
GROUP BY shop_code, a_type_normalized, alert_date
WITH DATA;

CREATE UNIQUE INDEX uix_mv_wh_alerts_daily ON mv_wh_alerts_daily (alert_date, shop_code, a_type_normalized);
CREATE INDEX ix_mv_wh_alerts_daily_date ON mv_wh_alerts_daily (alert_date, shop_code);


-- ============================================================
-- Verify
-- ============================================================
SELECT
    matviewname,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || matviewname)) AS size
FROM pg_matviews
WHERE matviewname LIKE 'mv_wh_%'
ORDER BY matviewname;
