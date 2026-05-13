set work_mem='1GB';


refresh materialized view mv_erpdata_test_bills_daily;
refresh materialized view mv_ovl_detail_all;
refresh materialized view mv_ovl_detail_excess_received;
refresh materialized view mv_ovl_detail_short_received;
refresh materialized view mv_wh_alerts_daily;
refresh materialized view mv_wh_erp_cashier_sessions_daily;
refresh materialized view mv_wh_erp_daily;
refresh materialized view mv_wh_erp_test_bills_cashier_daily;
refresh materialized view mv_wh_invoices_agg_daily;
refresh materialized view mv_wh_manager_handover_daily;


set work_mem='128MB';