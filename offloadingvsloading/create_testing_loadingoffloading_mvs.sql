-- Materialized views for fast detail rendering in testing_loadingoffloading.py
-- DB: WH

DROP MATERIALIZED VIEW IF EXISTS mv_ovl_detail_short_received;
DROP MATERIALIZED VIEW IF EXISTS mv_ovl_detail_excess_received;
DROP MATERIALIZED VIEW IF EXISTS mv_ovl_detail_all;

CREATE MATERIALIZED VIEW mv_ovl_detail_all AS
SELECT
    t.date,
    t.shop_code,
    t.vehicle_no,
    t.item_code,
    t.item_name,
    COALESCE(t.qty_loaded, 0)::numeric AS qty_loaded,
    (COALESCE(t.qty_loaded, 0) * COALESCE(t.price, 0))::numeric AS value_loaded,
    COALESCE(t.qty_offloaded, 0)::numeric AS qty_offloaded,
    (COALESCE(t.qty_offloaded, 0) * COALESCE(t.price, 0))::numeric AS value_offloaded,
    (COALESCE(t.qty_offloaded, 0) - COALESCE(t.qty_loaded, 0))::numeric AS diff_qty,
    ((COALESCE(t.qty_offloaded, 0) - COALESCE(t.qty_loaded, 0)) * COALESCE(t.price, 0))::numeric AS diff_val
FROM offloading_vs_loading t
WHERE t.date IS NOT NULL;

CREATE MATERIALIZED VIEW mv_ovl_detail_short_received AS
SELECT *
FROM mv_ovl_detail_all
WHERE diff_val < 0;

CREATE MATERIALIZED VIEW mv_ovl_detail_excess_received AS
SELECT *
FROM mv_ovl_detail_all
WHERE diff_val > 0;

CREATE INDEX idx_mv_ovl_all_date_shop ON mv_ovl_detail_all(date, shop_code);
CREATE INDEX idx_mv_ovl_all_sort ON mv_ovl_detail_all(date DESC, shop_code, vehicle_no, item_code);
CREATE INDEX idx_mv_ovl_short_date_shop ON mv_ovl_detail_short_received(date, shop_code);
CREATE INDEX idx_mv_ovl_excess_date_shop ON mv_ovl_detail_excess_received(date, shop_code);
