-- Target vs Achieve view for Century penetration
-- Logic:
-- 1) Target qty/month = (last 6-month qty total / 182.5) * days in current report month
-- 2) Target value/month = (last 6-month value total / 182.5) * days in current report month
-- 3) Avg retail price = AVG(reorder_level.selling_price) by item_code
-- 4) Actual qty/value = current month-to-yesterday sales
-- 5) Shortfall/Excess = target - actual (negative means excess)
-- 6) WH stock split by Tema/Kumasi from latest whstock snapshot

DROP VIEW IF EXISTS vw_target_vs_achieve_century;
DROP MATERIALIZED VIEW IF EXISTS mv_target_vs_achieve_century;

CREATE MATERIALIZED VIEW mv_target_vs_achieve_century AS
WITH params AS (
    SELECT
        (CURRENT_DATE - INTERVAL '1 day')::date AS as_of_date,
        date_trunc('month', CURRENT_DATE - INTERVAL '1 day')::date AS report_month_start,
    (date_trunc('month', CURRENT_DATE - INTERVAL '1 day')::date - INTERVAL '6 months')::date AS trailing_six_month_start,
    (date_trunc('month', CURRENT_DATE - INTERVAL '1 day')::date - INTERVAL '1 day')::date AS trailing_six_month_end,
        EXTRACT(day FROM (date_trunc('month', CURRENT_DATE - INTERVAL '1 day') + INTERVAL '1 month - 1 day'))::int AS days_in_report_month
),
items AS (
    SELECT
        rl.item_code,
        MAX(rl.item_name) AS item_name,
        AVG(NULLIF(rl.selling_price, 0))::numeric(14, 4) AS avg_rl_price
    FROM reorder_level rl
    WHERE UPPER(COALESCE(rl.brand, '')) = 'CENTURY'
    GROUP BY rl.item_code
),
trailing_six_month_sales AS (
    SELECT
        s.item_code,
        SUM(COALESCE(s.qty, 0))::numeric(14, 2) AS trailing_qty_6m,
        SUM(COALESCE(s.net_sales, 0))::numeric(14, 2) AS trailing_value_6m
    FROM sales s
    CROSS JOIN params p
    WHERE s.date_invoice >= p.trailing_six_month_start
      AND s.date_invoice <= p.trailing_six_month_end
    GROUP BY s.item_code
),
target_calc AS (
    SELECT
        tys.item_code,
        (COALESCE(tys.trailing_qty_6m, 0) / 182.5 * p.days_in_report_month)::numeric(14, 2) AS target_sales_month_pcs,
        (COALESCE(tys.trailing_value_6m, 0) / 182.5 * p.days_in_report_month)::numeric(14, 2) AS target_value_month_ghc
    FROM trailing_six_month_sales tys
    CROSS JOIN params p
),
actual_month AS (
    SELECT
        s.item_code,
        SUM(COALESCE(s.qty, 0))::numeric(14, 2) AS actual_sales_month_qty,
        SUM(COALESCE(s.net_sales, 0))::numeric(14, 2) AS actual_sales_month_value
    FROM sales s
    CROSS JOIN params p
    WHERE s.date_invoice >= p.report_month_start
      AND s.date_invoice <= p.as_of_date
    GROUP BY s.item_code
),
latest_wh AS (
    SELECT MAX(upload_date) AS max_upload_date FROM whstock
),
wh_stock AS (
    SELECT
        w.vc_item_code AS item_code,
        SUM(
            CASE
                WHEN UPPER(COALESCE(w.wh_code, '')) IN ('TS', 'TEMA')
                  OR UPPER(COALESCE(w.wh_name, '')) LIKE '%TEMA%'
                    THEN COALESCE(w.balance_qty, 0)
                ELSE 0
            END
        )::numeric(14, 2) AS wh_stock_tema,
        SUM(
            CASE
                WHEN UPPER(COALESCE(w.wh_code, '')) IN ('KA', 'KUMASI')
                  OR UPPER(COALESCE(w.wh_name, '')) LIKE '%KUMASI%'
                    THEN COALESCE(w.balance_qty, 0)
                ELSE 0
            END
        )::numeric(14, 2) AS wh_stock_kumasi
    FROM whstock w
    JOIN latest_wh lw ON w.upload_date = lw.max_upload_date
    GROUP BY w.vc_item_code
)
SELECT
    i.item_code,
    COALESCE(i.item_name, i.item_code) AS item_name,
    TO_CHAR(p.report_month_start, 'Mon''YY') AS report_month,
    COALESCE(i.avg_rl_price, 0)::numeric(14, 4) AS item_retail_price_avg,
    COALESCE(tc.target_sales_month_pcs, 0)::numeric(14, 2) AS target_sales_month_pcs,
    COALESCE(tc.target_value_month_ghc, 0)::numeric(14, 2) AS target_value_month_ghc,
    COALESCE(am.actual_sales_month_qty, 0)::numeric(14, 2) AS actual_sales_month_qty,
    (
        COALESCE(tc.target_sales_month_pcs, 0)
        - COALESCE(am.actual_sales_month_qty, 0)
    )::numeric(14, 2) AS shortfall_excess_qty,
    (
        COALESCE(tc.target_value_month_ghc, 0)
        - COALESCE(am.actual_sales_month_value, 0)
    )::numeric(14, 2) AS shortfall_excess_value,
    COALESCE(am.actual_sales_month_value, 0)::numeric(14, 2) AS actual_sales_month_value_net,
    COALESCE(ws.wh_stock_tema, 0)::numeric(14, 2) AS wh_stock_tema,
    COALESCE(ws.wh_stock_kumasi, 0)::numeric(14, 2) AS wh_stock_kumasi,
    CASE
        WHEN COALESCE(tc.target_sales_month_pcs, 0) = 0 THEN 'No Target'
        WHEN COALESCE(am.actual_sales_month_qty, 0) < COALESCE(tc.target_sales_month_pcs, 0) * 0.90 THEN 'Behind Target'
        WHEN COALESCE(am.actual_sales_month_qty, 0) <= COALESCE(tc.target_sales_month_pcs, 0) * 1.05 THEN 'On Target'
        ELSE 'Target Achieved'
    END AS remark,
    p.report_month_start AS report_month_start,
    p.as_of_date AS report_as_of_date
FROM items i
LEFT JOIN target_calc tc ON tc.item_code = i.item_code
LEFT JOIN actual_month am ON am.item_code = i.item_code
LEFT JOIN wh_stock ws ON ws.item_code = i.item_code
CROSS JOIN params p;

CREATE INDEX IF NOT EXISTS idx_mv_tva_item_code
    ON mv_target_vs_achieve_century(item_code);
CREATE INDEX IF NOT EXISTS idx_mv_tva_status
    ON mv_target_vs_achieve_century(remark);
CREATE INDEX IF NOT EXISTS idx_mv_tva_shortfall_value
    ON mv_target_vs_achieve_century(shortfall_excess_value DESC);

CREATE OR REPLACE VIEW vw_target_vs_achieve_century AS
SELECT * FROM mv_target_vs_achieve_century;

CREATE INDEX IF NOT EXISTS idx_whstock_vc_item_code ON whstock(vc_item_code);
CREATE INDEX IF NOT EXISTS idx_whstock_upload_date ON whstock(upload_date);
