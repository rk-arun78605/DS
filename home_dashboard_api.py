"""
Melcom Analytics Hub - Modern Web Dashboard API
Flask-based backend API for the modern web dashboard
Replaces original Streamlit home_dashboard.py with REST API architecture

Server: localhost:8501 (or configurable)
Frontend: home_dashboard_web.html
WebSocket: home_dashboard_ws.py (real-time updates)

Features:
- REST API for dashboard status checking
- Real-time server IP detection
- Database statistics retrieval
- WebSocket integration for live updates
- CORS support for multi-port access
- Caching for performance
"""

import os
import sys
import json
import socket
import logging
from datetime import datetime, timedelta
from functools import wraps
import time
import io

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from flask_cors import CORS
from flask_caching import Cache
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import pandas as pd

# ===================== CONFIGURATION =====================
APP_HOST = os.getenv('MELCOM_API_HOST', '0.0.0.0')
APP_PORT = int(os.getenv('MELCOM_API_PORT', 8501))
WORKER_CLASS = 'sync'
WORKERS = 4

# Database Config
DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
}

CHECK_SERIAL_MV_NAME = "mv_shop_compliance_check_serial_no"

# Dashboard Config
DASHBOARDS = [
    {'name': 'MELCOM STAR', 'port': 8502, 'file': 'serial_funnel_dashboard.py'},
    {'name': 'KPI Dashboard', 'port': 8503, 'file': 'kpi_dashboard.py'},
    {'name': 'STST', 'port': 8504, 'file': 'nowhstock_ds_final.py'},
    {'name': 'Barcode Matcher', 'port': 8505, 'file': 'barcode_matcher_app.py'},
    {'name': 'Century Penetration', 'port': 8506, 'file': 'centuryPenetration.py'},
]

# Table Categories for Upload Filtering
# Categories allow users to filter tables by business domain
# - "All Tables": Shows all available tables
# - "Century Penetration": Century brand analytics tables
# - "Serial No Tracking": Serial number lifecycle tracking tables  
# - "Others": Remaining tables (sales, inventory, etc.)
TABLE_CATEGORIES = {
    "Century Penetration": ["GEN_sales", "GEN_SIT", "GEN_reorder_level", "GEN_whstock"],
    "Serial No Tracking": ["serial_no_dailydata", "whreceived_serialno", "serialno_check_yes_no"],
    "Others": []  # Will be populated dynamically with remaining tables
}

# Table Configurations with Upload Modes
# Each table has:
# - columns: Expected CSV columns
# - upload_mode: "incremental" (backup new + append), "truncate" (backup all + replace), "append" (no backup + add)
# - database: Target database (salesdata, century_penetration, WH)
# - target_table: ACTUAL table where data is uploaded (may differ from config key)
# - backup_table: Backup table name (None for append mode)
# - date_column: Date column for incremental backups
# - category: Business category for filtering
#
# KEY MAPPING (User Selection → Actual Destination):
# ┌─────────────────────────┬──────────────────────────┬──────────────────────┐
# │ User Selects            │ Actual Table             │ Database             │
# ├─────────────────────────┼──────────────────────────┼──────────────────────┤
# │ GEN_sales               │ sales                    │ century_penetration  │
# │ GEN_SIT                 │ sit                      │ century_penetration  │
# │ GEN_reorder_level       │ reorder_level            │ century_penetration  │
# │ GEN_whstock             │ whstock                  │ century_penetration  │
# │ serial_no_dailydata     │ serial_no_dailydata_stag │ WH                   │
# │ whreceived_serialno     │ whreceived_serialno      │ WH                   │
# │ serialno_check_yes_no   │ serialno_check_yes_no    │ WH                   │
# │ sales_2025/2026         │ sales_2025/2026          │ salesdata            │
# │ whgrndetails            │ whgrndetails             │ salesdata            │
# │ (etc...)                │ (same as selection)      │ salesdata            │
# └─────────────────────────┴──────────────────────────┴──────────────────────┘
TABLE_CONFIGS = {
    "sales_2025": {
        "columns": ["SHOP_CODE", "ITEM_CODE", "ITEM_NAME", "DEPT", "GROUPS", "SUB_GROUP", "QTY", "NET_SALES", "DATE_INVOICE"],
        "backup_table": "sales_2025_backup",
        "date_column": "DATE_INVOICE",
        "upload_mode": "incremental",
        "database": "salesdata",
        "indexes": [
            {"name": "idx_sales_2025_date", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_2025_date ON public.sales_2025 USING btree (date_invoice)"},
            {"name": "idx_sales_2025_date_dept", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_2025_date_dept ON public.sales_2025 USING btree (date_invoice, dept) INCLUDE (net_sales, qty)"},
            {"name": "idx_sales_2025_item_date_qty", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_2025_item_date_qty ON public.sales_2025 USING btree (item_code, date_invoice) INCLUDE (shop_code, qty)"},
            {"name": "idx_sales_2025_item_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_2025_item_shop ON public.sales_2025 USING btree (item_code, shop_code)"},
            {"name": "idx_sales_2025_item_shop_date", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_2025_item_shop_date ON public.sales_2025 USING btree (item_code, shop_code, date_invoice)"}
        ],
        "category": "Others"
    },
    "sales_2026": {
        "columns": ["SHOP_CODE", "ITEM_CODE", "ITEM_NAME", "DEPT", "GROUPS", "SUB_GROUP", "QTY", "NET_SALES", "DATE_INVOICE"],
        "backup_table": "sales_2026_backup",
        "date_column": "DATE_INVOICE",
        "upload_mode": "incremental",
        "database": "salesdata",
        "indexes": [
            {"name": "idx_sales_2026_date", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_2026_date ON public.sales_2026 USING btree (date_invoice)"},
            {"name": "idx_sales_2026_date_dept", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_2026_date_dept ON public.sales_2026 USING btree (date_invoice, dept) INCLUDE (net_sales, qty)"},
            {"name": "idx_sales_2026_item_date_qty", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_2026_item_date_qty ON public.sales_2026 USING btree (item_code, date_invoice) INCLUDE (shop_code, qty)"},
            {"name": "idx_sales_2026_item_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_2026_item_shop ON public.sales_2026 USING btree (item_code, shop_code)"},
            {"name": "idx_sales_2026_item_shop_date", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_2026_item_shop_date ON public.sales_2026 USING btree (item_code, shop_code, date_invoice)"}
        ],
        "category": "Others"
    },
    "whgrndetails": {
        "columns": ["ITEM_CODE", "TYPE", "SUPPLIER_NAME", "WH_LAST_GRN_DATE", "WH_QTY_RECEIVED"],
        "backup_table": "whgrndetails_backup",
        "date_column": "WH_LAST_GRN_DATE",
        "upload_mode": "truncate",
        "encoding": "WIN1252",
        "database": "salesdata",
        "category": "Others"
    },
    "sit_data": {
        "columns": ["shop_code", "item_code", "dt_trans_date", "nu_transit_qty"],
        "backup_table": "sit_data_backup",
        "date_column": "dt_trans_date",
        "upload_mode": "truncate",
        "encoding": "UTF-8",
        "database": "salesdata",
        "category": "Others"
    },
    "shopexpiry": {
        "columns": ["ITEM_CODE", "SHOP_EXPIRY_DATE", "SHOP_CODE"],
        "backup_table": "shopexpiry_backup",
        "date_column": "SHOP_EXPIRY_DATE",
        "upload_mode": "truncate",
        "encoding": "UTF-8",
        "database": "salesdata",
        "category": "Others"
    },
    "itemdetails": {
        "columns": ["vc_item_code", "item_name", "dept", "groups", "sub_group", "type", "vc_supplier_name", "nu_qty_received"],
        "backup_table": "itemdetails_backup",
        "date_column": None,
        "upload_mode": "truncate",
        "encoding": "WIN1252",
        "database": "salesdata",
        "category": "Others"
    },
    "sup_shop_grn": {
        "columns": ["item_code", "item_name", "shop_code", "shop_grn_date", "wh_grn_date", "shop_stock"],
        "backup_table": "sup_shop_grn_backup",
        "date_column": "shop_grn_date",
        "upload_mode": "incremental",
        "encoding": "WIN1252",
        "database": "salesdata",
        "indexes": [
            {"name": "idx_grn_norm", "sql": "CREATE INDEX IF NOT EXISTS idx_grn_norm ON public.sup_shop_grn USING btree (normalized_itemcode, shop_code)"},
            {"name": "idx_sup_clean", "sql": "CREATE INDEX IF NOT EXISTS idx_sup_clean ON public.sup_shop_grn USING btree (clean_itemcode, shop_code)"},
            {"name": "idx_sup_shop_grn_item", "sql": "CREATE INDEX IF NOT EXISTS idx_sup_shop_grn_item ON public.sup_shop_grn USING btree (item_code)"},
            {"name": "idx_sup_shop_grn_item_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_sup_shop_grn_item_shop ON public.sup_shop_grn USING btree (item_code, shop_code)"},
            {"name": "idx_sup_shop_grn_item_shop_dates", "sql": "CREATE INDEX IF NOT EXISTS idx_sup_shop_grn_item_shop_dates ON public.sup_shop_grn USING btree (item_code, shop_code, shop_grn_date, wh_grn_date)"},
            {"name": "idx_sup_shop_grn_item_wh", "sql": "CREATE INDEX IF NOT EXISTS idx_sup_shop_grn_item_wh ON public.sup_shop_grn USING btree (TRIM(BOTH FROM upper(item_code)), wh_grn_date) WHERE (wh_grn_date IS NOT NULL)"},
            {"name": "idx_sup_shop_grn_item_wh_grn", "sql": "CREATE INDEX IF NOT EXISTS idx_sup_shop_grn_item_wh_grn ON public.sup_shop_grn USING btree (item_code, wh_grn_date) WHERE (wh_grn_date IS NOT NULL)"},
            {"name": "idx_sup_shop_grn_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_sup_shop_grn_shop ON public.sup_shop_grn USING btree (shop_code)"}
        ],
        "refresh_views": ["mv_recommendations_complete", "mv_last_30d_sales"],
        "post_upload_sql": "ANALYZE sup_shop_grn",
        "category": "Others"
    },
    "GEN_reorder_level": {
        "columns": ["item_code", "item_name", "shop_code", "dept", "brand", "shop_stock", "shop_grn_date", "wh_grn_date", "nu_min_qty", "nu_max_qty", "nu_reord_qty", "selling_price", "pack_size"],
        "backup_table": "reorder_level_backup",
        "date_column": None,
        "upload_mode": "truncate",
        "encoding": "UTF-8",
        "database": "century_penetration",
        "target_table": "reorder_level",
        "category": "Century Penetration"
    },
    "GEN_sales": {
        "columns": ["shop_code", "item_code", "item_name", "dept", "groups", "sub_group", "date_invoice", "qty", "net_sales"],
        "backup_table": "sales_backup",
        "date_column": "date_invoice",
        "upload_mode": "append",
        "encoding": "UTF-8",
        "database": "century_penetration",
        "target_table": "sales",
        "is_partitioned": True,
        "indexes": [
            {"name": "idx_sales_date", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_date ON public.sales USING btree (date_invoice DESC)"},
            {"name": "idx_sales_date_item_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_date_item_shop ON public.sales USING btree (date_invoice, item_code, shop_code)"},
            {"name": "idx_sales_item", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_item ON public.sales USING btree (item_code)"},
            {"name": "idx_sales_item_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_item_shop ON public.sales USING btree (item_code, shop_code)"},
            {"name": "idx_sales_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_shop ON public.sales USING btree (shop_code)"}
        ],
        "category": "Century Penetration"
    },
    "GEN_SIT": {
        "columns": ["shop_code", "item_code", "dt_trans_date", "nu_transit_qty"],
        "backup_table": "sit_backup",
        "date_column": "dt_trans_date",
        "upload_mode": "truncate",
        "encoding": "UTF-8",
        "database": "century_penetration",
        "target_table": "sit",
        "indexes": [
            {"name": "idx_sit_date", "sql": "CREATE INDEX IF NOT EXISTS idx_sit_date ON public.sit USING btree (dt_trans_date DESC)"},
            {"name": "idx_sit_item", "sql": "CREATE INDEX IF NOT EXISTS idx_sit_item ON public.sit USING btree (item_code)"},
            {"name": "idx_sit_item_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_sit_item_shop ON public.sit USING btree (item_code, shop_code)"},
            {"name": "idx_sit_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_sit_shop ON public.sit USING btree (shop_code)"}
        ],
        "refresh_views": ["mv_sit_summary", "mv_sales_metrics", "mv_century_penetration"],
        "category": "Century Penetration"
    },
    "GEN_whstock": {
        "columns": ["vc_item_code", "wh_code", "wh_name", "balance_qty"],
        "backup_table": None,
        "date_column": None,
        "upload_mode": "truncate",
        "encoding": "WIN1252",
        "database": "century_penetration",
        "target_table": "whstock",
        "add_upload_date": True,
        "category": "Century Penetration"
    },
    "serial_no_dailydata": {
        "columns": ["VC_WAREHOUSE_DESC", "VC_WH_CODE", "NU_DOC_ID", "DT_DOC_DATE", "LOADED_DATETIME", 
                    "VC_SHOP_CODE", "SHOP_NAME", "VC_ITEM_CODE", "VC_ITEM_DESC", "NU_SELLING_PRICE", 
                    "SERIAL_NO", "WH_LOAD_USER", "LOADINGNO", "LOADINGDATE", "VC_LOAD_NO", 
                    "DT_LOAD_DATE", "VC_VEHICLE_NO", "VC_SERAIL_NO", "VC_VEHICLE_NO.1", "DT_MOD_DATE", 
                    "SHOP_SOLD", "VC_INVOICE_NO", "DT_INVOICE_DATE", "SHOP_SERAIL_NO"],
        "backup_table": None,
        "date_column": "LOADED_DATETIME",
        "upload_mode": "append",
        "encoding": "UTF-8",
        "database": "WH",
        "target_table": "serial_no_dailydata_staging",
        "add_upload_date": True,
        "category": "Serial No Tracking"
    },
    "whreceived_serialno": {
        "columns": ["INBOUND_TYPE", "WAREHOUSE_NAME", "SUPP_NAME", "SERIAL_NO", "GRN_DATE", 
                    "ITEM_CODE", "ITEM_DESC", "VC_INBOND_TYPE", "SERIAL_QTY"],
        "backup_table": None,
        "date_column": "GRN_DATE",
        "upload_mode": "append",
        "encoding": "UTF-8",
        "database": "WH",
        "target_table": "whreceived_serialno",
        "category": "Serial No Tracking"
    },
    "serialno_check_yes_no": {
        "columns": ["ITEM_CODE", "ITEM_NAME", "SERIAL_NUMBER", "SHOP_CODE", "BILL_NO", 
                    "BILL_DATE", "TILL_NUMBER", "CASHIER_NAME", "SERIAL_CHECK"],
        "backup_table": None,
        "date_column": "BILL_DATE",
        "upload_mode": "append",
        "encoding": "UTF-8",
        "database": "WH",
        "target_table": "serialno_check_yes_no",
        "indexes": [
            {"name": "idx_serialcheck_serial", "sql": "CREATE INDEX IF NOT EXISTS idx_serialcheck_serial ON public.serialno_check_yes_no (serial_number)"},
            {"name": "idx_serialcheck_item", "sql": "CREATE INDEX IF NOT EXISTS idx_serialcheck_item ON public.serialno_check_yes_no (item_code)"},
            {"name": "idx_serialcheck_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_serialcheck_shop ON public.serialno_check_yes_no (shop_code)"},
            {"name": "idx_serialcheck_date", "sql": "CREATE INDEX IF NOT EXISTS idx_serialcheck_date ON public.serialno_check_yes_no (bill_date DESC)"},
            {"name": "idx_serialcheck_upload_date", "sql": "CREATE INDEX IF NOT EXISTS idx_serialcheck_upload_date ON public.serialno_check_yes_no (uploaded_data_date DESC)"}
        ],
        "category": "Serial No Tracking"
    }
}

# ===================== SETUP =====================
app = Flask(__name__, 
    static_folder='static',
    template_folder='templates'
)
app.config['JSON_SORT_KEYS'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Enable CORS for all routes (allow cross-port access)
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Cache Configuration
cache_config = {
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300
}
cache = Cache(app, config=cache_config)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== HELPER FUNCTIONS =====================

def get_ipv4_address():
    """Get machine's IPv4 address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.warning(f"Could not detect IPv4: {e}")
        return "localhost"

def check_dashboard_online(port, timeout=2):
    """Check if dashboard is running on port"""
    try:
        response = requests.get(f'http://localhost:{port}', timeout=timeout)
        return response.status_code == 200
    except:
        return False

def get_db_connection(database='salesdata'):
    """Get PostgreSQL connection"""
    try:
        config = DB_CONFIG.copy()
        config['database'] = database
        conn = psycopg2.connect(**config)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def get_target_table_indexes(table_config):
    """Get index definitions from table config"""
    return table_config.get('indexes', []) or []

def drop_table_indexes(conn, table_config):
    """Drop configured indexes before heavy upload"""
    dropped = []
    indexes = get_target_table_indexes(table_config)
    if not indexes:
        return dropped

    cursor = conn.cursor()
    try:
        target_table = table_config.get('target_table')
        for index_def in indexes:
            index_name = index_def.get('name')
            if not index_name:
                continue
            try:
                if 'pkey' in index_name.lower() and target_table:
                    cursor.execute(f"ALTER TABLE {target_table} DROP CONSTRAINT IF EXISTS {index_name}")
                else:
                    cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
                dropped.append(index_name)
            except Exception as exc:
                logger.warning(f"Could not drop index {index_name}: {exc}")
                conn.rollback()
                cursor = conn.cursor()
        conn.commit()
    finally:
        cursor.close()

    return dropped

def recreate_table_indexes(conn, table_config):
    """Recreate configured indexes after upload"""
    created = []
    indexes = get_target_table_indexes(table_config)
    if not indexes:
        return created

    cursor = conn.cursor()
    try:
        for index_def in indexes:
            index_name = index_def.get('name')
            index_sql = index_def.get('sql')
            if not index_name or not index_sql:
                continue
            try:
                cursor.execute(index_sql)
                created.append(index_name)
                conn.commit()
            except Exception as exc:
                logger.warning(f"Could not create index {index_name}: {exc}")
                conn.rollback()
        return created
    finally:
        cursor.close()

def execute_post_upload_sql(conn, sql_text):
    """Run optional post-upload SQL statements from config"""
    if not sql_text:
        return []

    executed = []
    cursor = conn.cursor()
    try:
        statements = [stmt.strip() for stmt in sql_text.split(';') if stmt.strip()]
        for stmt in statements:
            try:
                cursor.execute(stmt)
                conn.commit()
                executed.append(stmt)
            except Exception as exc:
                logger.warning(f"Post-upload SQL failed: {exc}")
                conn.rollback()
        return executed
    finally:
        cursor.close()

def refresh_materialized_views(conn, views):
    """Refresh configured materialized views"""
    refreshed = []
    if not views:
        return refreshed

    cursor = conn.cursor()
    try:
        for view_name in views:
            try:
                cursor.execute(f"REFRESH MATERIALIZED VIEW {view_name}")
                conn.commit()
                refreshed.append(view_name)
            except Exception as exc:
                logger.warning(f"Could not refresh materialized view {view_name}: {exc}")
                conn.rollback()
        return refreshed
    finally:
        cursor.close()


def rebuild_check_serial_mv_query_path(conn):
    """Drop/create/refresh indexed materialized view for Check Serial query path."""
    rebuilt = []
    cursor = conn.cursor()
    try:
        cursor.execute("SET statement_timeout = 0")
        cursor.execute(f"DROP MATERIALIZED VIEW IF EXISTS {CHECK_SERIAL_MV_NAME}")
        cursor.execute(f"""
            CREATE MATERIALIZED VIEW {CHECK_SERIAL_MV_NAME} AS
            SELECT
                TRIM(serial_check) AS "serial check",
                TRIM(serial_number) AS "serial number(where N)",
                TRIM(item_code) AS "item code",
                COALESCE(NULLIF(TRIM(item_name), ''), 'Unknown') AS "item desc",
                DATE(bill_date) AS "billdate",
                COALESCE(NULLIF(TRIM(shop_code), ''), 'Unknown') AS "shop code",
                COALESCE(NULLIF(TRIM(cashier_name), ''), 'Unknown') AS "cashier name",
                NULL::TEXT AS "Correct serial number by other shop",
                NULL::TEXT AS "corrected done by cashier",
                NULL::TEXT AS "corrected done at shop",
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM serialno_check_yes_no ysrc
                        WHERE TRIM(COALESCE(ysrc.serial_check, '')) = 'Y'
                          AND REGEXP_REPLACE(TRIM(COALESCE(ysrc.serial_number, '')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$', '', 'g') =
                              REGEXP_REPLACE(TRIM(COALESCE(serialno_check_yes_no.serial_number, '')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$', '', 'g')
                    ) THEN 'Y'
                    ELSE TRIM(serial_check)
                END AS "correct_source",
                'Incorrect SerialNo Sold'::TEXT AS "remark"
            FROM serialno_check_yes_no
            WHERE TRIM(serial_check) = 'N'
              AND serial_number IS NOT NULL
              AND TRIM(COALESCE(serial_number, '')) != ''
              AND COALESCE(item_code, '') != 'sales data not available'
        """)
        cursor.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{CHECK_SERIAL_MV_NAME}_billdate ON {CHECK_SERIAL_MV_NAME} ("billdate")'
        )
        cursor.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{CHECK_SERIAL_MV_NAME}_shop_billdate ON {CHECK_SERIAL_MV_NAME} ("shop code", "billdate")'
        )
        cursor.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{CHECK_SERIAL_MV_NAME}_serial ON {CHECK_SERIAL_MV_NAME} ("serial number(where N)")'
        )
        cursor.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{CHECK_SERIAL_MV_NAME}_correct_source ON {CHECK_SERIAL_MV_NAME} ("correct_source")'
        )
        cursor.execute(f"REFRESH MATERIALIZED VIEW {CHECK_SERIAL_MV_NAME}")
        conn.commit()
        rebuilt.append(CHECK_SERIAL_MV_NAME)
        return rebuilt
    except Exception as exc:
        logger.warning(f"Could not rebuild check serial MV query path: {exc}")
        conn.rollback()
        return rebuilt
    finally:
        cursor.close()

# ===================== API ROUTES =====================

@app.route('/')
def index():
    """Serve main dashboard HTML"""
    try:
        # Serve the premium version (newest)
        html_path = os.path.join(os.path.dirname(__file__), 'home_dashboard_premium.html')
        logger.info(f"Loading HTML from: {html_path}")
        logger.info(f"File exists: {os.path.exists(html_path)}")
        
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            logger.error(f"Premium HTML not found at {html_path}")
            raise FileNotFoundError(f"home_dashboard_premium.html not found at {html_path}")
    except Exception as e:
        logger.error(f"Error loading premium dashboard: {e}", exc_info=True)
        try:
            # Fallback to v2
            html_path = os.path.join(os.path.dirname(__file__), 'home_dashboard_web_v2.html')
            logger.info(f"Trying fallback v2: {html_path}")
            if os.path.exists(html_path):
                with open(html_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.error(f"Fallback v2 not found at {html_path}")
        except Exception as e2:
            logger.error(f"Error loading fallback v2: {e2}")
        
        return jsonify({'error': str(e), 'status': 500}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0'
    })

@app.route('/api/server-ip')
@cache.cached(timeout=300)
def get_server_ip():
    """Get server's IPv4 address"""
    ip = get_ipv4_address()
    return jsonify({
        'ip': ip,
        'host': socket.gethostname(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/dashboards/status')
def get_dashboards_status():
    """Get status of all dashboards"""
    status = {}
    online_count = 0
    
    for dashboard in DASHBOARDS:
        is_online = check_dashboard_online(dashboard['port'])
        status[dashboard['name']] = {
            'port': dashboard['port'],
            'online': is_online,
            'url': f'http://localhost:{dashboard["port"]}'
        }
        if is_online:
            online_count += 1
    
    return jsonify({
        'dashboards': status,
        'online_count': online_count,
        'total_count': len(DASHBOARDS),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats/barcodes')
@cache.cached(timeout=600)
def get_barcode_stats():
    """Get barcode count from database"""
    try:
        conn = get_db_connection('salesdata')
        if not conn:
            return jsonify({'count': 0, 'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM barcode_item_master 
            WHERE is_active = TRUE
        """)
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return jsonify({
            'count': result['count'] if result else 0,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching barcode stats: {e}")
        return jsonify({
            'count': 0,
            'error': str(e)
        }), 500

@app.route('/api/stats/sales')
@cache.cached(timeout=600)
def get_sales_stats():
    """Get sales data statistics"""
    try:
        conn = get_db_connection('salesdata')
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get sales 2024/2025/2026 row counts
        cursor.execute("""
            SELECT 
                'sales_2024' as table_name,
                COUNT(*) as row_count
            FROM sales_2024
            UNION ALL
            SELECT 'sales_2025', COUNT(*) FROM sales_2025
            UNION ALL
            SELECT 'sales_2026', COUNT(*) FROM sales_2026
        """)
        
        sales_stats = {row['table_name']: row['row_count'] for row in cursor.fetchall()}
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'sales_stats': sales_stats,
            'total_rows': sum(sales_stats.values()),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching sales stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats/database')
@cache.cached(timeout=600)
def get_database_stats():
    """Get database size and stats"""
    try:
        conn = get_db_connection('salesdata')
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get database size
        cursor.execute("""
            SELECT 
                pg_database.datname,
                pg_size_pretty(pg_database_size(pg_database.datname)) AS size
            FROM pg_database
            WHERE datname = 'salesdata'
        """)
        db_size = cursor.fetchone()
        
        # Get table count
        cursor.execute("""
            SELECT COUNT(*) as table_count
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        table_count = cursor.fetchone()['table_count']
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'database': 'salesdata',
            'size': db_size['size'] if db_size else 'N/A',
            'table_count': table_count,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching database stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/info')
def get_system_info():
    """Get system information"""
    try:
        return jsonify({
            'hostname': socket.gethostname(),
            'ip_address': get_ipv4_address(),
            'python_version': sys.version.split()[0],
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': time.time()  # Relative uptime
        })
    except Exception as e:
        logger.error(f"Error fetching system info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs/recent')
def get_recent_logs():
    """Get recent application logs"""
    # This is a placeholder - implement based on your logging strategy
    return jsonify({
        'logs': [
            {
                'timestamp': datetime.now().isoformat(),
                'level': 'INFO',
                'message': 'Dashboard API initialized'
            }
        ]
    })

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear application cache"""
    try:
        cache.clear()
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ===================== UPLOAD API ENDPOINTS =====================

@app.route('/api/upload/categories')
def get_upload_categories():
    """Get all upload table categories"""
    try:
        # Build others category dynamically
        century_tables = set(TABLE_CATEGORIES["Century Penetration"])
        serial_tables = set(TABLE_CATEGORIES["Serial No Tracking"])
        others = [t for t in TABLE_CONFIGS.keys() if t not in century_tables and t not in serial_tables]
        
        return jsonify({
            'categories': {
                "All Tables": list(TABLE_CONFIGS.keys()),
                "Century Penetration": TABLE_CATEGORIES["Century Penetration"],
                "Serial No Tracking": TABLE_CATEGORIES["Serial No Tracking"],
                "Others": others
            },
            'count': len(TABLE_CONFIGS),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/tables/<category>')
def get_tables_by_category(category):
    """Get tables filtered by category"""
    try:
        if category == "All Tables":
            tables = list(TABLE_CONFIGS.keys())
        elif category == "Century Penetration":
            tables = TABLE_CATEGORIES["Century Penetration"]
        elif category == "Serial No Tracking":
            tables = TABLE_CATEGORIES["Serial No Tracking"]
        elif category == "Others":
            century_tables = set(TABLE_CATEGORIES["Century Penetration"])
            serial_tables = set(TABLE_CATEGORIES["Serial No Tracking"])
            tables = [t for t in TABLE_CONFIGS.keys() if t not in century_tables and t not in serial_tables]
        else:
            return jsonify({'error': 'Invalid category'}), 400
        
        # Add config details for each table
        table_details = {}
        for table in tables:
            config = TABLE_CONFIGS[table]
            target_table = config.get('target_table', table)
            table_details[table] = {
                'upload_mode': config.get('upload_mode', 'incremental'),
                'database': config.get('database', 'salesdata'),
                'target_table': target_table,
                'destination': f"{config.get('database', 'salesdata')}.{target_table}",
                'date_column': config.get('date_column'),
                'columns_count': len(config['columns']),
                'category': config.get('category', 'Others')
            }
        
        return jsonify({
            'category': category,
            'tables': tables,
            'details': table_details,
            'count': len(tables),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting tables by category: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/table/<table_name>/config')
def get_table_config(table_name):
    """Get configuration for specific table"""
    try:
        if table_name not in TABLE_CONFIGS:
            return jsonify({'error': 'Table not found'}), 404
        
        config = TABLE_CONFIGS[table_name].copy()
        
        # Get current table stats
        database = config.get('database', 'salesdata')
        target_table = config.get('target_table', table_name)
        
        conn = get_db_connection(database)
        if conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) as count FROM {target_table}")
            row_count = cursor.fetchone()['count']
            
            # Get latest date if date column exists
            latest_date = None
            if config.get('date_column'):
                date_col = config['date_column'].lower()
                try:
                    cursor.execute(f"SELECT MAX({date_col}) as latest_date FROM {target_table}")
                    result = cursor.fetchone()
                    if result and result['latest_date']:
                        latest_date = result['latest_date'].isoformat() if hasattr(result['latest_date'], 'isoformat') else str(result['latest_date'])
                except Exception as date_error:
                    logger.warning(f"Could not get latest date: {date_error}")
            
            cursor.close()
            conn.close()
            
            config['current_stats'] = {
                'row_count': row_count,
                'latest_date': latest_date
            }
        
        # Add destination info (CRITICAL: Shows actual table where data will be uploaded)
        config['destination_info'] = {
            'selected_table': table_name,
            'actual_table': target_table,
            'database': database,
            'full_path': f"{database}.{target_table}",
            'note': f"CSV will be uploaded to '{target_table}' table in '{database}' database"
        }
        
        # Add upload mode badge and description
        upload_mode = config.get('upload_mode', 'incremental')
        if upload_mode == "incremental":
            config['mode_badge'] = "🔄 INCREMENTAL"
            config['mode_description'] = "Backup NEW → Append (preserves existing data, adds only new records)"
        elif upload_mode == "truncate":
            config['mode_badge'] = "🗑️ TRUNCATE"
            config['mode_description'] = "Backup ALL → Replace (empties table, uploads fresh data)"
        elif upload_mode == "append":
            config['mode_badge'] = "➕ APPEND"
            config['mode_description'] = "No Backup → Add New (directly inserts new records, no backup)"
        
        return jsonify({
            'table': table_name,
            'config': config,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting table config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/modes')
def get_upload_modes():
    """Get information about upload modes"""
    return jsonify({
        'modes': {
            'incremental': {
                'name': 'Incremental',
                'icon': '🔄',
                'description': 'Backup only new data (dates after last backup), then append to existing table',
                'use_case': 'Daily sales uploads, transaction logs',
                'backup': True,
                'preserves_data': True
            },
            'truncate': {
                'name': 'Truncate',
                'icon': '🗑️',
                'description': 'Backup ALL data, empty table completely, then upload fresh data',
                'use_case': 'Master data files, full refreshes',
                'backup': True,
                'preserves_data': False
            },
            'append': {
                'name': 'Append',
                'icon': '➕',
                'description': 'No backup - directly insert/upsert new records into table',
                'use_case': 'Serial tracking, staging tables with UPSERT logic',
                'backup': False,
                'preserves_data': True
            }
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/upload/mappings')
def get_upload_mappings():
    """Get table name to destination mappings"""
    try:
        mappings = []
        for table_name, config in TABLE_CONFIGS.items():
            target_table = config.get('target_table', table_name)
            database = config.get('database', 'salesdata')
            mappings.append({
                'user_selection': table_name,
                'actual_table': target_table,
                'database': database,
                'destination': f"{database}.{target_table}",
                'upload_mode': config.get('upload_mode', 'incremental'),
                'category': config.get('category', 'Others'),
                'same_name': table_name == target_table
            })
        
        # Sort by category then by user selection
        mappings.sort(key=lambda x: (x['category'], x['user_selection']))
        
        return jsonify({
            'mappings': mappings,
            'count': len(mappings),
            'note': 'This shows where each selected table uploads to in the database',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting mappings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/detect', methods=['POST'])
def detect_table_from_csv():
    """Auto-detect target table based on CSV column headers"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        
        # Read CSV headers
        try:
            # Try to read with different encodings
            csv_content = file.read()
            file.seek(0)  # Reset file pointer
            
            # Try UTF-8 first, then other encodings
            for encoding in ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(io.BytesIO(csv_content), encoding=encoding, nrows=0)
                    csv_columns = [col.strip().upper() for col in df.columns]
                    break
                except:
                    continue
            else:
                return jsonify({'error': 'Could not read CSV file'}), 400
            
        except Exception as e:
            return jsonify({'error': f'Failed to read CSV: {str(e)}'}), 400
        
        # Find matching tables based on column headers
        matches = []
        for table_name, config in TABLE_CONFIGS.items():
            expected_columns = [col.strip().upper() for col in config['columns']]
            
            # Check if all expected columns are in CSV
            missing_columns = set(expected_columns) - set(csv_columns)
            extra_columns = set(csv_columns) - set(expected_columns)
            
            # Calculate match score (percentage of matching columns)
            if len(expected_columns) > 0:
                match_score = len(set(expected_columns) & set(csv_columns)) / len(expected_columns) * 100
            else:
                match_score = 0
            
            if match_score >= 80:  # At least 80% match
                matches.append({
                    'table_name': table_name,
                    'match_score': round(match_score, 2),
                    'database': config.get('database', 'salesdata'),
                    'target_table': config.get('target_table', table_name),
                    'destination': f"{config.get('database', 'salesdata')}.{config.get('target_table', table_name)}",
                    'missing_columns': list(missing_columns),
                    'extra_columns': list(extra_columns),
                    'upload_mode': config.get('upload_mode', 'incremental'),
                    'category': config.get('category', 'Others'),
                    'is_perfect_match': match_score == 100 and len(extra_columns) == 0
                })
        
        # Sort by match score (highest first)
        matches.sort(key=lambda x: (x['is_perfect_match'], x['match_score']), reverse=True)
        
        if len(matches) == 0:
            return jsonify({
                'status': 'no_match',
                'message': 'No matching table found for this CSV',
                'csv_columns': csv_columns,
                'available_tables': list(TABLE_CONFIGS.keys())
            })
        
        # Get best match
        best_match = matches[0]
        
        return jsonify({
            'status': 'success',
            'best_match': best_match,
            'all_matches': matches,
            'csv_columns': csv_columns,
            'csv_column_count': len(csv_columns),
            'recommendation': f"Upload to '{best_match['table_name']}' (Match: {best_match['match_score']}%)",
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error detecting table: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/validate', methods=['POST'])
def validate_upload():
    """Validate uploaded CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        table = request.form.get('table')
        
        if not file or not table:
            return jsonify({'error': 'Missing file or table parameter'}), 400
        
        # Read first few lines for preview
        try:
            lines = file.read(2048).decode('utf-8').split('\n')[:5]
            return jsonify({
                'status': 'valid',
                'table': table,
                'preview': lines,
                'file_size': len(file.read()),
                'message': 'File is valid CSV'
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'File validation failed: {str(e)}'
            }), 400
    
    except Exception as e:
        logger.error(f"Upload validation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/process', methods=['POST'])
def process_upload():
    """Process CSV file upload to database with proper upload mode handling"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        table_name = request.form.get('table')
        
        if not file or not table_name:
            return jsonify({'error': 'Missing parameters'}), 400
        
        if table_name not in TABLE_CONFIGS:
            return jsonify({'error': f'Unknown table: {table_name}'}), 400
        
        config = TABLE_CONFIGS[table_name]
        upload_mode = config.get('upload_mode', 'incremental')
        database = config.get('database', 'salesdata')
        target_table = config.get('target_table', table_name)
        backup_table = config.get('backup_table')
        date_column = config.get('date_column')
        
        # Read CSV file
        encoding = config.get('encoding', 'utf-8')
        if encoding.upper() == 'WIN1252':
            encoding = 'cp1252'
        
        try:
            df = pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            # Try alternative encodings
            file.seek(0)
            for enc in ['cp1252', 'latin1', 'iso-8859-1', 'utf-8']:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=enc)
                    logger.info(f"Auto-detected encoding: {enc}")
                    break
                except:
                    continue
            else:
                return jsonify({'error': 'Could not detect file encoding'}), 400

        if target_table == 'serialno_check_yes_no':
            serial_col = None
            for col in df.columns:
                if str(col).strip().lower() == 'serial_number':
                    serial_col = col
                    break
            if serial_col is not None:
                df[serial_col] = (
                    df[serial_col]
                    .astype(str)
                    .str.replace(r'\s+$', '', regex=True)
                    .replace({'nan': ''})
                )
                logger.info("Trimmed trailing spaces from serial_number during upload")
        
        # Connect to database
        conn = get_db_connection(database)
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        rows_affected = 0
        rows_backed_up = 0
        dropped_indexes = []
        recreated_indexes = []
        refreshed_views = []
        post_sql_executed_count = 0
        
        try:
            if target_table == 'serialno_check_yes_no':
                logger.info("Rebuilding serialno_check_yes_no table: drop then create")
                cursor.execute("DROP TABLE IF EXISTS serialno_check_yes_no CASCADE")
                cursor.execute("""
                    CREATE TABLE serialno_check_yes_no (
                        item_code TEXT,
                        item_name TEXT,
                        serial_number TEXT,
                        shop_code TEXT,
                        bill_no TEXT,
                        bill_date DATE,
                        till_number TEXT,
                        cashier_name TEXT,
                        serial_check TEXT,
                        uploaded_data_date DATE DEFAULT CURRENT_DATE
                    )
                """)
                conn.commit()

            # Step 1: Drop indexes for faster upload (when configured)
            dropped_indexes = drop_table_indexes(conn, {
                **config,
                'target_table': target_table
            })

            # Step 2: Handle backup based on upload mode
            if upload_mode == "truncate" and backup_table:
                # TRUNCATE MODE: Backup ALL data, then empty table
                logger.info(f"TRUNCATE mode: Backing up all data from {target_table} to {backup_table}")
                
                # Drop and recreate backup table
                cursor.execute(f"DROP TABLE IF EXISTS {backup_table}")
                cursor.execute(f"CREATE TABLE {backup_table} (LIKE {target_table} INCLUDING INDEXES EXCLUDING GENERATED)")
                
                # Backup all data
                cursor.execute(f"INSERT INTO {backup_table} SELECT * FROM {target_table}")
                rows_backed_up = cursor.rowcount
                
                # Truncate main table
                cursor.execute(f"TRUNCATE TABLE {target_table}")
                conn.commit()
                
                logger.info(f"✅ Backed up {rows_backed_up} rows, truncated {target_table}")
                
            elif upload_mode == "incremental" and backup_table and date_column:
                # INCREMENTAL MODE: Backup only NEW data (dates after last backup)
                logger.info(f"INCREMENTAL mode: Backing up new data from {target_table} to {backup_table}")
                
                # Ensure backup table exists
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {backup_table} 
                    (LIKE {target_table} INCLUDING INDEXES EXCLUDING GENERATED)
                """)
                
                # Backup only new data
                date_col_lower = date_column.lower()
                cursor.execute(f"""
                    INSERT INTO {backup_table}
                    SELECT * FROM {target_table}
                    WHERE {date_col_lower} > (
                        SELECT COALESCE(MAX({date_col_lower}), DATE '1900-01-01')
                        FROM {backup_table}
                    )
                """)
                rows_backed_up = cursor.rowcount
                conn.commit()
                
                logger.info(f"✅ Backed up {rows_backed_up} new rows")
                
            elif upload_mode == "append":
                # APPEND MODE: No backup, just upload
                logger.info(f"APPEND mode: No backup needed for {target_table}")
                rows_backed_up = 0
            else:
                rows_backed_up = 0
            
            # Step 3: Upload data using COPY for performance
            upload_col = None
            if config.get('add_upload_date'):
                upload_col = 'upload_date'
                if target_table in ('whreceived_serialno', 'serialno_check_yes_no'):
                    upload_col = 'uploaded_data_date'

                # Force upload date value for every row (prevents NOT NULL failures)
                df[upload_col] = datetime.now().strftime('%Y-%m-%d')

                # Ensure deterministic column order for COPY
                base_cols = [col.strip() for col in config['columns']]
                ordered_cols = [col for col in base_cols if col in df.columns]
                if upload_col in df.columns and upload_col not in ordered_cols:
                    ordered_cols.append(upload_col)
                if ordered_cols:
                    df = df[ordered_cols]

            buffer = io.StringIO()
            df.to_csv(buffer, index=False, header=False)
            buffer.seek(0)
            
            # Map column names (handle case sensitivity)
            columns = config['columns']
            col_mapping = {col.upper(): col.lower() for col in columns}
            if upload_col:
                col_mapping[upload_col.upper()] = upload_col
            csv_cols = [col.strip() for col in df.columns.tolist()]
            target_cols = [col_mapping.get(col.upper(), col.lower()) for col in csv_cols]
            
            copy_sql = f"""
                COPY {target_table} ({', '.join(target_cols)})
                FROM STDIN WITH CSV
            """
            
            logger.info(f"Uploading {len(df)} rows to {target_table}...")
            cursor.copy_expert(copy_sql, buffer)
            conn.commit()
            
            rows_affected = len(df)

            # Step 4: Recreate dropped indexes
            recreated_indexes = recreate_table_indexes(conn, {
                **config,
                'target_table': target_table
            })

            # Step 5: Run post-upload SQL (if configured)
            post_sql_executed = execute_post_upload_sql(conn, config.get('post_upload_sql'))
            post_sql_executed_count = len(post_sql_executed)

            # Step 6: Refresh MVs (if configured)
            refreshed_views = refresh_materialized_views(conn, config.get('refresh_views', []))

            if target_table == 'serialno_check_yes_no':
                refreshed_views.extend(rebuild_check_serial_mv_query_path(conn))

            # Step 7: Analyze table
            try:
                cursor.execute(f"ANALYZE {target_table}")
                conn.commit()
            except Exception as analyze_error:
                logger.warning(f"ANALYZE skipped for {target_table}: {analyze_error}")
                conn.rollback()
            
            cursor.close()
            conn.close()
            
            return jsonify({
                'status': 'success',
                'selected_table': table_name,
                'actual_table': target_table,
                'database': database,
                'destination': f"{database}.{target_table}",
                'upload_mode': upload_mode,
                'rows_uploaded': rows_affected,
                'rows_backed_up': rows_backed_up if upload_mode != 'append' else None,
                'indexes_dropped': len(dropped_indexes),
                'indexes_recreated': len(recreated_indexes),
                'views_refreshed': refreshed_views,
                'post_sql_executed': post_sql_executed_count,
                'message': f'✅ Successfully uploaded {rows_affected:,} rows to {database}.{target_table}',
                'detail': f'Selected: {table_name} → Uploaded to: {target_table} ({upload_mode} mode)',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as upload_error:
            conn.rollback()
            cursor.close()
            conn.close()
            logger.error(f"Upload error: {upload_error}")
            return jsonify({
                'status': 'error',
                'error': str(upload_error),
                'table': table_name
            }), 500
    
    except Exception as e:
        logger.error(f"Upload processing error: {e}")
        return jsonify({'error': str(e)}), 500

# ===================== INDEX MANAGEMENT ENDPOINTS =====================

@app.route('/api/indexes/list/<table_name>')
@cache.cached(timeout=300)
def list_indexes(table_name):
    """List all indexes for a table"""
    try:
        conn = get_db_connection('salesdata')
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get indexes for the table
        cursor.execute("""
            SELECT 
                indexname,
                indexdef,
                pg_size_pretty(pg_relation_size(indexrelname::regclass)) as size
            FROM pg_indexes
            WHERE tablename = %s
            ORDER BY indexname
        """, (table_name,))
        
        indexes = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({
            'table': table_name,
            'count': len(indexes),
            'indexes': indexes,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error listing indexes: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/indexes/analyze/<table_name>')
def analyze_indexes(table_name):
    """Analyze table for index recommendations"""
    try:
        conn = get_db_connection('salesdata')
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get table statistics
        cursor.execute("""
            SELECT 
                n_live_tup as row_count,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
            FROM pg_stat_user_tables
            WHERE tablename = %s
        """, (table_name,))
        
        stats = cursor.fetchone()
        
        # Get column information
        cursor.execute("""
            SELECT 
                column_name,
                data_type,
                is_nullable
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """, (table_name,))
        
        columns = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({
            'table': table_name,
            'statistics': stats,
            'columns': columns,
            'recommendations': [
                'Add index on date columns for faster queries',
                'Consider indexes on frequently filtered columns',
                'Review join columns for potential indexes'
            ],
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error analyzing indexes: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/indexes/create', methods=['POST'])
def create_index():
    """Create a new index on a table"""
    try:
        data = request.get_json()
        table = data.get('table')
        index_name = data.get('index_name')
        columns = data.get('columns', [])
        unique = data.get('unique', False)
        
        if not all([table, index_name, columns]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        conn = get_db_connection('salesdata')
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        
        # Build CREATE INDEX statement
        unique_str = 'UNIQUE' if unique else ''
        columns_str = ', '.join(columns)
        
        sql = f"""
            CREATE {unique_str} INDEX {index_name}
            ON {table} ({columns_str})
        """
        
        logger.info(f"Creating index: {sql}")
        cursor.execute(sql)
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'message': f'Index {index_name} created successfully',
            'table': table,
            'index_name': index_name,
            'columns': columns,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        return jsonify({'error': str(e)}), 500

# ===================== ERROR HANDLERS =====================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not found',
        'status': 404,
        'message': 'The requested resource was not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'status': 500,
        'message': 'An unexpected error occurred'
    }), 500

# ===================== REQUEST HANDLERS =====================

@app.before_request
def log_request():
    """Log incoming requests"""
    if not request.path.startswith('/static'):
        logger.debug(f"{request.method} {request.path}")

@app.after_request
def add_security_headers(response):
    """Add security headers"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# ===================== MAIN =====================

if __name__ == '__main__':
    logger.info(f"Starting Melcom Analytics Hub API on {APP_HOST}:{APP_PORT}")
    logger.info(f"Frontend URL: http://{get_ipv4_address()}:{APP_PORT}")
    logger.info(f"Dashboard Monitor: {APP_PORT}/api/dashboards/status")
    
    # Run development server (use gunicorn in production)
    app.run(
        host=APP_HOST,
        port=APP_PORT,
        debug=False,
        threaded=True
    )
