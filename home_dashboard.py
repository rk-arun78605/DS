"""
Melcom Analytics Hub - Main Dashboard Landing Page with Data Upload
Centralized access to all analytics dashboards + Data Upload functionality
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import socket
from datetime import datetime
from datetime import date
from typing import Optional
import io
import time
import threading
import hashlib
import re

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="Melcom Analytics Hub",
    page_icon="https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM CSS
# ===========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background: #0d1117; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .stMarkdown p, .stMarkdown li { color: #c9d1d9; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #e6edf3 !important; }

    /* Inputs */
    .stTextInput > div > div > input {
        background: #161b22 !important; border: 1px solid #30363d !important;
        border-radius: 10px !important; color: #e6edf3 !important;
        padding: 12px 16px !important; font-size: 15px !important;
    }
    .stTextInput > div > div > input:focus { border-color: #E31837 !important; box-shadow: 0 0 0 3px rgba(227,24,55,0.15) !important; }
    .stTextInput label, .stSelectbox label, .stMultiSelect label {
        color: #8b949e !important; font-size: 12px !important;
        font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.5px !important;
    }
    .stSelectbox > div > div { background: #161b22 !important; border-color: #30363d !important; border-radius: 10px !important; color: #e6edf3 !important; }
    .stMultiSelect > div > div { background: #161b22 !important; border-color: #30363d !important; border-radius: 10px !important; }

    /* Buttons */
    .stButton > button {
        background: #21262d !important; color: #e6edf3 !important;
        border: 1px solid #30363d !important; border-radius: 10px !important;
        font-weight: 600 !important; transition: all 0.2s ease !important;
    }
    .stButton > button:hover { background: #30363d !important; border-color: #6e7681 !important; transform: translateY(-1px) !important; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #E31837 0%, #b01028 100%) !important;
        border-color: transparent !important; color: white !important;
        box-shadow: 0 4px 15px rgba(227,24,55,0.3) !important;
    }
    .stButton > button[kind="primary"]:hover { box-shadow: 0 8px 25px rgba(227,24,55,0.5) !important; transform: translateY(-2px) !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent !important; border-bottom: 1px solid #30363d; }
    .stTabs [data-baseweb="tab-list"] button { background: transparent !important; color: #8b949e !important; border: none !important; border-radius: 0 !important; padding: 10px 20px !important; font-weight: 500 !important; border-bottom: 2px solid transparent !important; }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { color: #E31837 !important; border-bottom-color: #E31837 !important; font-weight: 700 !important; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 20px; }

    /* Metrics */
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; }
    [data-testid="stMetricValue"] { color: #e6edf3 !important; font-weight: 800 !important; }
    [data-testid="stMetricDelta"] { font-size: 12px !important; }

    /* Alerts */
    .stAlert { border-radius: 10px !important; }
    [data-testid="stNotification"] { background: #161b22 !important; border-radius: 10px !important; }

    /* Checkbox */
    .stCheckbox label { color: #e6edf3 !important; }
    .stCheckbox label span { color: #e6edf3 !important; }

    /* Caption */
    .stCaptionContainer p { color: #6e7681 !important; }

    /* Divider */
    hr { border-color: #30363d !important; margin: 16px 0 !important; }

    /* Dataframe */
    .stDataFrame { border-radius: 12px !important; }
    [data-testid="stDataFrame"] { background: #161b22 !important; }

    /* Text area */
    .stTextArea textarea { background: #161b22 !important; border-color: #30363d !important; color: #e6edf3 !important; border-radius: 10px !important; }

    /* Select slider */
    .stSelectSlider label { color: #8b949e !important; }

    /* Progress */
    .stProgress > div > div { background-color: #E31837 !important; }

    /* Form */
    [data-testid="stForm"] { background: #161b22; border: 1px solid #30363d; border-radius: 16px; padding: 16px; }

    /* Expander */
    .streamlit-expanderHeader { background: #161b22 !important; border-radius: 10px !important; color: #e6edf3 !important; }
    .streamlit-expanderContent { background: #0d1117 !important; border-color: #30363d !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #6e7681; }
</style>
""", unsafe_allow_html=True)

# ===========================
# FUNCTIONS
# ===========================

def get_ipv4_address():
    """Get the machine's IPv4 address"""
    try:
        # Create a socket connection to get the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except Exception:
        return "localhost"

def check_dashboard_status(port, host='localhost'):
    """Check if a dashboard is running on the given host and port."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

@st.cache_data(ttl=300)
def get_database_stats():
    """Get quick stats from database"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=3307,
            user='postgres',
            password='hello',
            database='salesdata'
        )
        cursor = conn.cursor()
        
        # Get barcode count
        cursor.execute("SELECT COUNT(*) FROM barcode_item_master WHERE is_active = TRUE")
        barcode_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return barcode_count
    except:
        return None

# ===========================
# DATA UPLOAD CONFIGURATIONS
# ===========================

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'salesdata'
}

# Session state for upload control
if 'upload_cancelled' not in st.session_state:
    st.session_state.upload_cancelled = False
if 'upload_in_progress' not in st.session_state:
    st.session_state.upload_in_progress = False
if 'current_connection' not in st.session_state:
    st.session_state.current_connection = None

TABLE_CONFIGS = {
    "sales_2025": {
        "columns": ["SHOP_CODE", "ITEM_CODE", "ITEM_NAME", "DEPT", "GROUPS", "SUB_GROUP", "QTY", "NET_SALES", "DATE_INVOICE"],
        "backup_table": "sales_2025_backup",
        "date_column": "DATE_INVOICE",
        "upload_mode": "incremental",  # Incremental backup
        "indexes": [
            {"name": "idx_sales_2025_date", "sql": 'CREATE INDEX idx_sales_2025_date ON public.sales_2025 USING btree ("DATE_INVOICE")'},
            {"name": "idx_sales_2025_date_dept", "sql": 'CREATE INDEX idx_sales_2025_date_dept ON public.sales_2025 USING btree ("DATE_INVOICE", "DEPT") INCLUDE ("NET_SALES", "QTY")'},
            {"name": "idx_sales_2025_item_date_qty", "sql": 'CREATE INDEX idx_sales_2025_item_date_qty ON public.sales_2025 USING btree ("ITEM_CODE", "DATE_INVOICE") INCLUDE ("SHOP_CODE", "QTY")'},
            {"name": "idx_sales_2025_item_shop", "sql": 'CREATE INDEX idx_sales_2025_item_shop ON public.sales_2025 USING btree ("ITEM_CODE", "SHOP_CODE")'},
            {"name": "idx_sales_2025_item_shop_date", "sql": 'CREATE INDEX idx_sales_2025_item_shop_date ON public.sales_2025 USING btree ("ITEM_CODE", "SHOP_CODE", "DATE_INVOICE")'}
        ]
    },
    "sales_2026": {
        "columns": ["SHOP_CODE", "ITEM_CODE", "ITEM_NAME", "DEPT", "GROUPS", "SUB_GROUP", "QTY", "NET_SALES", "DATE_INVOICE"],
        "backup_table": "sales_2026_backup",
        "date_column": "DATE_INVOICE",
        "upload_mode": "incremental",  # Incremental backup
        "indexes": [
            {"name": "idx_sales_2026_date", "sql": 'CREATE INDEX idx_sales_2026_date ON public.sales_2026 USING btree (date_invoice)'},
            {"name": "idx_sales_2026_date_dept", "sql": 'CREATE INDEX idx_sales_2026_date_dept ON public.sales_2026 USING btree (date_invoice, dept) INCLUDE (net_sales, qty)'},
            {"name": "idx_sales_2026_item_date_qty", "sql": 'CREATE INDEX idx_sales_2026_item_date_qty ON public.sales_2026 USING btree (item_code, date_invoice) INCLUDE (shop_code, qty)'},
            {"name": "idx_sales_2026_item_shop", "sql": 'CREATE INDEX idx_sales_2026_item_shop ON public.sales_2026 USING btree (item_code, shop_code)'},
            {"name": "idx_sales_2026_item_shop_date", "sql": 'CREATE INDEX idx_sales_2026_item_shop_date ON public.sales_2026 USING btree (item_code, shop_code, date_invoice)'}
        ]
    },
    "whgrndetails": {
        "columns": ["ITEM_CODE", "TYPE", "SUPPLIER_NAME", "WH_LAST_GRN_DATE", "WH_QTY_RECEIVED"],
        # Note: GRN_PLUS_30_DATE is a generated column (ALWAYS) - do NOT include in upload
        "backup_table": "whgrndetails_backup",
        "date_column": "WH_LAST_GRN_DATE",
        "upload_mode": "truncate",  # Full replace with TRUNCATE
        "encoding": "WIN1252",
        "indexes": []  # No indexes for this table
    },
    "sit_data": {
        "columns": ["shop_code", "item_code", "dt_trans_date", "nu_transit_qty"],
        "backup_table": "sit_data_backup",
        "date_column": "dt_trans_date",
        "upload_mode": "truncate",  # Full replace with TRUNCATE
        "encoding": "UTF-8",
        "indexes": []  # No indexes for this table
    },
    "shopexpiry": {
        "columns": ["ITEM_CODE", "SHOP_EXPIRY_DATE", "SHOP_CODE"],
        "backup_table": "shopexpiry_backup",
        "date_column": "SHOP_EXPIRY_DATE",
        "upload_mode": "truncate",  # Full replace with TRUNCATE
        "encoding": "UTF-8",
        "indexes": []  # No indexes for this table
    },
    "itemdetails": {
        "columns": ["vc_item_code", "item_name", "dept", "groups", "sub_group", "type", "vc_supplier_name", "nu_qty_received"],
        "backup_table": "itemdetails_backup",
        "date_column": None,  # No date column
        "upload_mode": "truncate",  # Full replace with TRUNCATE
        "encoding": "WIN1252",
        "indexes": [
            {"name": "idx_itemdetails_item_code", "sql": "CREATE INDEX idx_itemdetails_item_code ON itemdetails (vc_item_code)"},
            {"name": "idx_itemdetails_groups", "sql": "CREATE INDEX idx_itemdetails_groups ON itemdetails (groups, sub_group)"},
            {"name": "idx_itemdetails_supplier", "sql": "CREATE INDEX idx_itemdetails_supplier ON itemdetails (vc_supplier_name)"}
        ]
    },
    "sup_shop_grn": {
        "columns": ["item_code", "item_name", "shop_code", "shop_grn_date", "wh_grn_date", "shop_stock"],
        "backup_table": "sup_shop_grn_backup",
        "date_column": "shop_grn_date",
        "upload_mode": "incremental",  # Incremental backup (no truncate)
        "encoding": "WIN1252",
        "indexes": [
            {"name": "idx_grn_norm", "sql": "CREATE INDEX idx_grn_norm ON public.sup_shop_grn USING btree (normalized_itemcode, shop_code)"},
            {"name": "idx_sup_clean", "sql": "CREATE INDEX idx_sup_clean ON public.sup_shop_grn USING btree (clean_itemcode, shop_code)"},
            {"name": "idx_sup_shop_grn_item", "sql": "CREATE INDEX idx_sup_shop_grn_item ON public.sup_shop_grn USING btree (item_code)"},
            {"name": "idx_sup_shop_grn_item_shop", "sql": "CREATE INDEX idx_sup_shop_grn_item_shop ON public.sup_shop_grn USING btree (item_code, shop_code)"},
            {"name": "idx_sup_shop_grn_item_shop_dates", "sql": "CREATE INDEX idx_sup_shop_grn_item_shop_dates ON public.sup_shop_grn USING btree (item_code, shop_code, shop_grn_date, wh_grn_date)"},
            {"name": "idx_sup_shop_grn_item_wh", "sql": "CREATE INDEX idx_sup_shop_grn_item_wh ON public.sup_shop_grn USING btree (TRIM(BOTH FROM upper(item_code)), wh_grn_date) WHERE (wh_grn_date IS NOT NULL)"},
            {"name": "idx_sup_shop_grn_item_wh_grn", "sql": "CREATE INDEX idx_sup_shop_grn_item_wh_grn ON public.sup_shop_grn USING btree (item_code, wh_grn_date) WHERE (wh_grn_date IS NOT NULL)"},
            {"name": "idx_sup_shop_grn_shop", "sql": "CREATE INDEX idx_sup_shop_grn_shop ON public.sup_shop_grn USING btree (shop_code)"},
            {"name": "sup_shop_grn_pkey", "sql": "CREATE UNIQUE INDEX sup_shop_grn_pkey ON public.sup_shop_grn USING btree (item_code, shop_code, shop_grn_date)"}
        ],
        "post_upload_sql": "ANALYZE sup_shop_grn; SELECT shop_grn_date, SUM(shop_stock) AS total_stock FROM sup_shop_grn WHERE EXTRACT(YEAR FROM shop_grn_date) = '2025' GROUP BY shop_grn_date;"
    },
    "GEN_reorder_level": {
        "columns": ["item_code", "item_name", "shop_code", "dept", "brand", "shop_stock", "shop_grn_date", "wh_grn_date", "nu_min_qty", "nu_max_qty", "nu_reord_qty", "selling_price", "pack_size"],
        "backup_table": "reorder_level_backup",
        "date_column": None,
        "upload_mode": "truncate",  # Truncate mode - backup all, empty table, upload new
        "encoding": "UTF-8",
        "database": "century_penetration",
        "target_table": "reorder_level",
        "column_mapping": {
            "nu_min_qty": "min_nu",
            "nu_max_qty": "max_nu", 
            "nu_reord_qty": "reorder_qty"
        },
        "type_conversions": {
            "min_nu": "int",
            "max_nu": "int",
            "reorder_qty": "int",
            "shop_stock": "int",
            "pack_size": "int"
        },
        "indexes": [
            {"name": "idx_reorder_brand", "sql": "CREATE INDEX idx_reorder_brand ON public.reorder_level USING btree (brand) WHERE (upper((brand)::text) = 'CENTURY'::text)"},
            {"name": "idx_reorder_brand_shop", "sql": "CREATE INDEX idx_reorder_brand_shop ON public.reorder_level USING btree (brand, shop_code) WHERE (upper((brand)::text) = 'CENTURY'::text)"},
            {"name": "idx_reorder_dept", "sql": "CREATE INDEX idx_reorder_dept ON public.reorder_level USING btree (dept)"},
            {"name": "idx_reorder_item", "sql": "CREATE INDEX idx_reorder_item ON public.reorder_level USING btree (item_code)"},
            {"name": "idx_reorder_item_shop", "sql": "CREATE INDEX idx_reorder_item_shop ON public.reorder_level USING btree (item_code, shop_code)"},
            {"name": "idx_reorder_shop", "sql": "CREATE INDEX idx_reorder_shop ON public.reorder_level USING btree (shop_code)"},
            {"name": "idx_reorder_stock", "sql": "CREATE INDEX idx_reorder_stock ON public.reorder_level USING btree (shop_stock) WHERE (shop_stock > 0)"},
            {"name": "reorder_level_pkey", "sql": "CREATE UNIQUE INDEX reorder_level_pkey ON public.reorder_level USING btree (item_code, shop_code)"}
        ]
    },
    "GEN_sales": {
        "columns": ["shop_code", "item_code", "item_name", "dept", "groups", "sub_group", "date_invoice", "qty", "net_sales"],
        "backup_table": "sales_backup",
        "date_column": "date_invoice",
        "upload_mode": "append",  # Append mode - data added to existing table
        "encoding": "UTF-8",
        "database": "century_penetration",
        "target_table": "sales",  # Sales table in century_penetration DB (PARTITIONED BY MONTH)
        "is_partitioned": True,  # Month-partitioned: sales_jan2025, sales_feb2025, ..., sales_dec2026, etc.
        "parse_dates": ["date_invoice"],  # Parse date with dayfirst=True
        "partition_info": "Month-partitioned sales table (sales_jan2025...sales_dec2026)",
        "indexes": [
            {"name": "idx_sales_date", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_date ON public.sales USING btree (date_invoice DESC)"},
            {"name": "idx_sales_date_item_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_date_item_shop ON public.sales USING btree (date_invoice, item_code, shop_code)"},
            {"name": "idx_sales_item", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_item ON public.sales USING btree (item_code)"},
            {"name": "idx_sales_item_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_item_shop ON public.sales USING btree (item_code, shop_code)"},
            {"name": "idx_sales_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_sales_shop ON public.sales USING btree (shop_code)"},
            {"name": "sales_pkey", "sql": "CREATE UNIQUE INDEX IF NOT EXISTS sales_pkey ON public.sales USING btree (shop_code, item_code, date_invoice)"}
        ],
        "refresh_views": ["mv_sales_metrics", "mv_century_penetration", "mv_target_vs_achieve_century"]
    },
    "GEN_SIT": {
        "columns": ["shop_code", "item_code", "dt_trans_date", "nu_transit_qty"],
        "backup_table": "sit_backup",
        "date_column": "dt_trans_date",
        "upload_mode": "truncate",  # TRUNCATE MODE - Table will be emptied
        "encoding": "UTF-8",
        "database": "century_penetration",
        "target_table": "sit",
        "parse_dates": ["dt_trans_date"],  # Parse date with dayfirst=True
        "type_conversions": {
            "nu_transit_qty": "int"
        },
        "indexes": [
            {"name": "idx_sit_date", "sql": "CREATE INDEX idx_sit_date ON public.sit USING btree (dt_trans_date DESC)"},
            {"name": "idx_sit_item", "sql": "CREATE INDEX idx_sit_item ON public.sit USING btree (item_code)"},
            {"name": "idx_sit_item_shop", "sql": "CREATE INDEX idx_sit_item_shop ON public.sit USING btree (item_code, shop_code)"},
            {"name": "idx_sit_shop", "sql": "CREATE INDEX idx_sit_shop ON public.sit USING btree (shop_code)"},
            {"name": "sit_pkey", "sql": "CREATE UNIQUE INDEX sit_pkey ON public.sit USING btree (shop_code, item_code, dt_trans_date)"}
        ],
        "refresh_views": ["mv_sit_summary", "mv_sales_metrics", "mv_century_penetration", "mv_target_vs_achieve_century"]
    },
    "GEN_whstock": {
        "columns": ["vc_item_code", "wh_code", "wh_name", "balance_qty"],
        "backup_table": None,  # No backup - append mode
        "date_column": None,  # No date column in CSV
        "upload_mode": "truncate",  # APPEND MODE - Data will be added to existing table
        "encoding": "WIN1252",  # Windows encoding for special characters
        "database": "century_penetration",
        "target_table": "whstock",
        "add_upload_date": True,  # Add upload_date column automatically
        "type_conversions": {
            "balance_qty": "float"
        },
        "indexes": [
            {"name": "idx_whstock_item_code", "sql": "CREATE INDEX IF NOT EXISTS idx_whstock_item_code ON whstock(vc_item_code)"},
            {"name": "idx_whstock_wh_code", "sql": "CREATE INDEX IF NOT EXISTS idx_whstock_wh_code ON whstock(wh_code)"},
            {"name": "idx_whstock_upload_date", "sql": "CREATE INDEX IF NOT EXISTS idx_whstock_upload_date ON whstock(upload_date DESC)"},
            {"name": "idx_whstock_item_wh_date", "sql": "CREATE INDEX IF NOT EXISTS idx_whstock_item_wh_date ON whstock(vc_item_code, wh_code, upload_date DESC)"}
        ]
    }
    ,
    "serial_no_dailydata": {
        "columns": ["VC_WAREHOUSE_DESC", "VC_WH_CODE", "NU_DOC_ID", "DT_DOC_DATE", "LOADED_DATETIME", 
                    "VC_SHOP_CODE", "SHOP_NAME", "VC_ITEM_CODE", "VC_ITEM_DESC", "NU_SELLING_PRICE", 
                    "SERIAL_NO", "WH_LOAD_USER", "LOADINGNO", "LOADINGDATE", "VC_LOAD_NO", 
                    "DT_LOAD_DATE", "VC_VEHICLE_NO", "VC_SERAIL_NO", "VC_VEHICLE_NO_1", "DT_MOD_DATE", 
                    "SHOP_SOLD", "VC_INVOICE_NO", "DT_INVOICE_DATE", "SHOP_SERAIL_NO"],
        "backup_table": None,  # No backup for staging table (append mode)
        "date_column": "LOADED_DATETIME",
        "upload_mode": "append",
        "encoding": "UTF-8",
        "database": "WH",
        "target_table": "serial_no_dailydata_staging",
        "add_upload_date": True,  # Add upload_date column for tracking
        "parse_dates": ["DT_DOC_DATE", "LOADED_DATETIME", "LOADINGDATE", "DT_LOAD_DATE", "DT_MOD_DATE", "DT_INVOICE_DATE"],
        "prefer_month_first_dates": ["DT_DOC_DATE", "DT_MOD_DATE", "DT_INVOICE_DATE"],
        "prefer_day_first_dates": ["LOADINGDATE", "DT_LOAD_DATE"],
        "type_conversions": {
            "NU_DOC_ID": "int",
            "NU_SELLING_PRICE": "float",
            "LOADINGNO": "int"
        },
        "indexes": [
            {"name": "idx_staging_item", "sql": "CREATE INDEX IF NOT EXISTS idx_staging_item ON public.serial_no_dailydata_staging (vc_item_code)"},
            {"name": "idx_staging_loaded_date", "sql": "CREATE INDEX IF NOT EXISTS idx_staging_loaded_date ON public.serial_no_dailydata_staging (loaded_datetime)"},
            {"name": "idx_staging_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_staging_shop ON public.serial_no_dailydata_staging (vc_shop_code)"}
        ],
        "post_upload_sql": """
            CREATE TABLE IF NOT EXISTS serial_no_dailydata (
                vc_warehouse_desc VARCHAR(200),
                vc_wh_code VARCHAR(10),
                nu_doc_id BIGINT,
                dt_doc_date DATE,
                loaded_datetime TIMESTAMP,
                vc_shop_code VARCHAR(10),
                shop_name VARCHAR(200),
                vc_item_code VARCHAR(50),
                vc_item_desc TEXT,
                nu_selling_price NUMERIC(15,2),
                serial_no VARCHAR(100),
                wh_load_user VARCHAR(100),
                loadingno INTEGER,
                loadingdate DATE,
                vc_load_no VARCHAR(50),
                dt_load_date DATE,
                vc_vehicle_no VARCHAR(50),
                vc_serail_no VARCHAR(100),
                "VC_VEHICLE_NO_1" VARCHAR(50),
                dt_mod_date DATE,
                shop_sold VARCHAR(10),
                vc_invoice_no VARCHAR(50),
                dt_invoice_date DATE,
                shop_serail_no VARCHAR(100),
                upload_date DATE DEFAULT CURRENT_DATE,
                remarks2 VARCHAR(100)
            );
            DELETE FROM serial_no_dailydata
            WHERE loadingdate >= (SELECT MIN(loadingdate) FROM serial_no_dailydata_staging WHERE loadingdate IS NOT NULL)
              AND loadingdate <= (SELECT MAX(loadingdate) FROM serial_no_dailydata_staging WHERE loadingdate IS NOT NULL)
              AND (SELECT COUNT(*) FROM serial_no_dailydata_staging WHERE loadingdate IS NOT NULL) > 0;
            INSERT INTO serial_no_dailydata SELECT * FROM serial_no_dailydata_staging;
            SELECT (SELECT COUNT(*) FROM serial_no_dailydata_staging) as staging_count, (SELECT COUNT(*) FROM serial_no_dailydata) as main_count, (SELECT MIN(loadingdate) FROM serial_no_dailydata) as main_min_date, (SELECT MAX(loadingdate) FROM serial_no_dailydata) as main_max_date
        """
    },
    "whreceived_serialno": {
        "columns": ["INBOUND_TYPE", "WAREHOUSE_NAME", "SUPP_NAME", "SERIAL_NO", "GRN_DATE", 
                    "ITEM_CODE", "ITEM_DESC", "VC_INBOND_TYPE", "SERIAL_QTY"],
        "backup_table": None,  # No backup for append mode
        "date_column": "GRN_DATE",
        "upload_mode": "append",
        "encoding": "UTF-8",
        "database": "WH",
        "target_table": "whreceived_serialno",
        "add_upload_date": False,  # Table already has uploaded_data_date column with DEFAULT
        "parse_dates": ["GRN_DATE"],
        "prefer_month_first_dates": ["GRN_DATE"],
        "date_format": "%Y-%m-%d",
        "date_parse_dayfirst": False,
        "type_conversions": {
            "SERIAL_QTY": "int"
        },
        "indexes": [
            {"name": "idx_whreceived_serial", "sql": "CREATE INDEX IF NOT EXISTS idx_whreceived_serial ON public.whreceived_serialno (serial_no)"},
            {"name": "idx_whreceived_item", "sql": "CREATE INDEX IF NOT EXISTS idx_whreceived_item ON public.whreceived_serialno (item_code)"},
            {"name": "idx_whreceived_grn_date", "sql": "CREATE INDEX IF NOT EXISTS idx_whreceived_grn_date ON public.whreceived_serialno (grn_date DESC)"},
            {"name": "idx_whreceived_upload_date", "sql": "CREATE INDEX IF NOT EXISTS idx_whreceived_upload_date ON public.whreceived_serialno (uploaded_data_date DESC)"}
        ]
    },
    "serialno_check_yes_no": {
        "columns": ["ITEM_CODE", "ITEM_NAME", "SERIAL_NUMBER", "SHOP_CODE", "BILL_NO", 
                    "BILL_DATE", "TILL_NUMBER", "CASHIER_NAME", "SERIAL_CHECK"],
        "backup_table": None,  # No backup for append mode
        "date_column": "BILL_DATE",
        "skiprows": 4,  # Ignore first 4 non-data rows in uploaded CSV
        "upload_mode": "append",
        "encoding": "UTF-8",
        "database": "WH",
        "target_table": "serialno_check_yes_no",
        "add_upload_date": False,  # Table already has uploaded_data_date column with DEFAULT
        "parse_dates": ["BILL_DATE"],
        "indexes": [
            {"name": "idx_serialcheck_serial", "sql": "CREATE INDEX IF NOT EXISTS idx_serialcheck_serial ON public.serialno_check_yes_no (serial_number)"},
            {"name": "idx_serialcheck_item", "sql": "CREATE INDEX IF NOT EXISTS idx_serialcheck_item ON public.serialno_check_yes_no (item_code)"},
            {"name": "idx_serialcheck_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_serialcheck_shop ON public.serialno_check_yes_no (shop_code)"},
            {"name": "idx_serialcheck_date", "sql": "CREATE INDEX IF NOT EXISTS idx_serialcheck_date ON public.serialno_check_yes_no (bill_date DESC)"},
            {"name": "idx_serialcheck_upload_date", "sql": "CREATE INDEX IF NOT EXISTS idx_serialcheck_upload_date ON public.serialno_check_yes_no (uploaded_data_date DESC)"}
        ]
    },
    "LVO_offloading_vs_loading": {
        "columns": ["date", "shop_code", "vehicle_no", "item_code", "item_name", "qty_loaded", "value_loaded", "qty_offloaded", "value_offloaded", "diff_qty", "diff_val", "price", "diff"],
        "backup_table": "offloading_vs_loading_backup",
        "date_column": "date",
        "upload_mode": "append",
        "encoding": "UTF-8",
        "database": "WH",
        "target_table": "offloading_vs_loading",
        "history_mode": "append_with_history",
        "history_table": "offloading_vs_loading_history",
        "indexes": [
            {"name": "idx_ovl_date_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_ovl_date_shop ON public.offloading_vs_loading (date, shop_code)"},
            {"name": "idx_ovl_item_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_ovl_item_shop ON public.offloading_vs_loading (item_code, shop_code)"}
        ]
    },
    "LVO_offloading_loading_staging": {
        "columns": ["shop_code", "shop_name"],
        "backup_table": "offloading_loading_staging_backup",
        "date_column": None,
        "upload_mode": "append",
        "encoding": "UTF-8",
        "database": "WH",
        "target_table": "offloading_loading_staging",
        "history_mode": "append_with_history",
        "history_table": "offloading_loading_staging_history",
        "indexes": [
            {"name": "idx_ovl_staging_shop", "sql": "CREATE INDEX IF NOT EXISTS idx_ovl_staging_shop ON public.offloading_loading_staging (shop_code)"}
        ]
    },
    "LVO_shopmgrname": {
        "columns": ["shop_code", "shop_description", "shop_manager_name"],
        "backup_table": "shopmgrname_backup",
        "date_column": None,
        "upload_mode": "append",
        "encoding": "UTF-8",
        "database": "WH",
        "target_table": "shopmgrname",
        "history_mode": "shopmgr_scd2",
        "indexes": [
            {"name": "idx_shopmgrname_current_code", "sql": "CREATE INDEX IF NOT EXISTS idx_shopmgrname_current_code ON public.shopmgrname (shop_code) WHERE is_current = TRUE"}
        ]
    }
}

# ===========================
# DATA UPLOAD HELPER FUNCTIONS
# ===========================

def cancel_upload():
    """Cancel ongoing upload"""
    st.session_state.upload_cancelled = True
    if st.session_state.current_connection:
        try:
            # Cancel backend queries
            cur = st.session_state.current_connection.cursor()
            cur.execute("SELECT pg_cancel_backend(pid) FROM pg_stat_activity WHERE pid != pg_backend_pid() AND usename = 'postgres' AND state = 'active'")
            st.session_state.current_connection.close()
        except:
            pass
    st.session_state.upload_in_progress = False

def get_table_count(table_name, database='salesdata'):
    """Get total row count from a table (handles partitioned tables)"""
    try:
        db_config = DB_CONFIG.copy()
        db_config['database'] = database
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        # For partitioned sales table in salesdata, sum counts from all year partitions
        if table_name.lower() == 'sales' and database.lower() == 'salesdata':
            cur.execute("""
                SELECT COALESCE(SUM(row_count), 0) FROM (
                    SELECT COUNT(*) as row_count FROM sales_2024
                    UNION ALL
                    SELECT COUNT(*) FROM sales_2025
                    UNION ALL
                    SELECT COUNT(*) FROM sales_2026
                ) counts
            """)
        # For month-partitioned sales table in century_penetration
        elif table_name.lower() == 'sales' and database.lower() == 'century_penetration':
            cur.execute("""
                SELECT COALESCE(SUM(row_count), 0) FROM (
                    SELECT COUNT(*) as row_count FROM sales_jan2025
                    UNION ALL SELECT COUNT(*) FROM sales_feb2025
                    UNION ALL SELECT COUNT(*) FROM sales_mar2025
                    UNION ALL SELECT COUNT(*) FROM sales_apr2025
                    UNION ALL SELECT COUNT(*) FROM sales_may2025
                    UNION ALL SELECT COUNT(*) FROM sales_jun2025
                    UNION ALL SELECT COUNT(*) FROM sales_jul2025
                    UNION ALL SELECT COUNT(*) FROM sales_aug2025
                    UNION ALL SELECT COUNT(*) FROM sales_sep2025
                    UNION ALL SELECT COUNT(*) FROM sales_oct2025
                    UNION ALL SELECT COUNT(*) FROM sales_nov2025
                    UNION ALL SELECT COUNT(*) FROM sales_dec2025
                    UNION ALL SELECT COUNT(*) FROM sales_jan2026
                    UNION ALL SELECT COUNT(*) FROM sales_feb2026
                    UNION ALL SELECT COUNT(*) FROM sales_mar2026
                    UNION ALL SELECT COUNT(*) FROM sales_apr2026
                    UNION ALL SELECT COUNT(*) FROM sales_may2026
                    UNION ALL SELECT COUNT(*) FROM sales_jun2026
                    UNION ALL SELECT COUNT(*) FROM sales_jul2026
                    UNION ALL SELECT COUNT(*) FROM sales_aug2026
                    UNION ALL SELECT COUNT(*) FROM sales_sep2026
                    UNION ALL SELECT COUNT(*) FROM sales_oct2026
                    UNION ALL SELECT COUNT(*) FROM sales_nov2026
                    UNION ALL SELECT COUNT(*) FROM sales_dec2026
                ) counts
            """)
        else:
            # Regular table count
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        
        result = cur.fetchone()[0]
        count = result if result is not None else 0
        cur.close()
        conn.close()
        return count
    except Exception as e:
        return None

def get_latest_date(table_name, date_column, database='salesdata'):
    """Get latest date from a table (handles partitioned tables)"""
    try:
        db_config = DB_CONFIG.copy()
        db_config['database'] = database
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        # Quote the date_column for case-insensitive handling
        quoted_date_col = f'"{date_column}"'
        
        # For partitioned sales table in salesdata, get max from all year partitions
        if table_name.lower() == 'sales' and database.lower() == 'salesdata':
            cur.execute(f"""
                SELECT MAX(max_date) FROM (
                    SELECT MAX({quoted_date_col}) as max_date FROM sales_2024
                    UNION ALL
                    SELECT MAX({quoted_date_col}) FROM sales_2025
                    UNION ALL
                    SELECT MAX({quoted_date_col}) FROM sales_2026
                ) dates
            """)
            latest_date = cur.fetchone()[0]
        # For month-partitioned sales table in century_penetration
        elif table_name.lower() == 'sales' and database.lower() == 'century_penetration':
            cur.execute(f"""
                SELECT MAX(max_date) FROM (
                    SELECT MAX({quoted_date_col}) as max_date FROM sales_jan2025
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_feb2025
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_mar2025
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_apr2025
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_may2025
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_jun2025
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_jul2025
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_aug2025
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_sep2025
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_oct2025
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_nov2025
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_dec2025
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_jan2026
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_feb2026
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_mar2026
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_apr2026
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_may2026
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_jun2026
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_jul2026
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_aug2026
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_sep2026
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_oct2026
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_nov2026
                    UNION ALL SELECT MAX({quoted_date_col}) FROM sales_dec2026
                ) dates
            """)
            latest_date = cur.fetchone()[0]
        elif table_name.lower() in ['sales_2024', 'sales_2025']:
            # Specific year partition 2024/2025 - use quoted uppercase column names
            cur.execute(f'SELECT MAX({quoted_date_col}) FROM {table_name}')
            latest_date = cur.fetchone()[0]
        elif table_name.lower() == 'sales_2026':
            # sales_2026 uses lowercase column names (unquoted)
            cur.execute(f'SELECT MAX(date_invoice) FROM {table_name}')
            latest_date = cur.fetchone()[0]
        else:
            # Regular table query - try both uppercase (quoted) and lowercase column names
            # because old tables have uppercase columns, new tables have lowercase
            latest_date = None
            try:
                # First try with quoted uppercase (for old tables)
                cur.execute(f'SELECT MAX({quoted_date_col}) FROM {table_name}')
                result = cur.fetchone()
                if result and result[0] is not None:
                    latest_date = result[0]
            except:
                pass
            
            # Fall back to lowercase if uppercase failed or returned NULL
            if latest_date is None:
                try:
                    lowercase_col = date_column.lower()
                    cur.execute(f'SELECT MAX({lowercase_col}) FROM {table_name}')
                    result = cur.fetchone()
                    if result and result[0] is not None:
                        latest_date = result[0]
                except:
                    pass
        
        cur.close()
        conn.close()
        return latest_date
    except Exception as e:
        return None

def drop_indexes(conn, table_config, status_container):
    """Drop indexes for fast processing"""
    if st.session_state.upload_cancelled:
        return []
    
    dropped = []
    cur = conn.cursor()
    
    # Extract table name from config
    table_name = None
    for idx in table_config['indexes']:
        if 'sql' in idx:
            # Extract table name from CREATE INDEX statement
            sql_lower = idx['sql'].lower()
            if ' on ' in sql_lower:
                parts = sql_lower.split(' on ')
                if len(parts) > 1:
                    table_part = parts[1].split()[0].replace('public.', '')
                    table_name = table_part
                    break
    
    for idx in table_config['indexes']:
        if st.session_state.upload_cancelled:
            break
        try:
            status_container.text(f"🔽 Dropping index: {idx['name']}...")
            
            # Check if this is a primary key constraint
            if 'pkey' in idx['name'].lower() and table_name:
                # Drop constraint instead of index
                cur.execute(f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {idx['name']}")
            else:
                # Drop regular index
                cur.execute(f"DROP INDEX IF EXISTS {idx['name']}")
            
            conn.commit()
            dropped.append(idx['name'])
        except Exception as e:
            # Rollback transaction on error
            conn.rollback()
            status_container.warning(f"Could not drop index {idx['name']}: {e}")
    cur.close()
    return dropped

def create_indexes(conn, table_config, status_container):
    """Recreate indexes after upload"""
    if st.session_state.upload_cancelled:
        return []
    
    created = []
    cur = conn.cursor()
    
    # Extract table name from config
    table_name = table_config.get('target_table', None)
    if not table_name and table_config.get('indexes'):
        for idx in table_config['indexes']:
            if 'sql' in idx:
                # Extract table name from CREATE INDEX statement
                sql_lower = idx['sql'].lower()
                if ' on ' in sql_lower:
                    parts = sql_lower.split(' on ')
                    if len(parts) > 1:
                        table_part = parts[1].split()[0].replace('public.', '')
                        table_name = table_part
                        break
    
    for idx in table_config['indexes']:
        if st.session_state.upload_cancelled:
            break
        try:
            status_container.text(f"🔼 Creating index: {idx['name']}...")
            
            # Check if this is a primary key constraint
            if 'pkey' in idx['name'].lower() and 'UNIQUE INDEX' in idx['sql'].upper() and table_name:
                # Extract columns from CREATE UNIQUE INDEX statement
                # e.g., CREATE UNIQUE INDEX reorder_level_pkey ON public.reorder_level USING btree (item_code, shop_code)
                try:
                    # Try to create as constraint instead
                    sql_parts = idx['sql'].split('(', 1)
                    if len(sql_parts) > 1:
                        columns = sql_parts[1].split(')')[0]
                        # First try to drop if exists
                        cur.execute(f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {idx['name']}")
                        # Create as primary key constraint
                        cur.execute(f"ALTER TABLE {table_name} ADD CONSTRAINT {idx['name']} PRIMARY KEY ({columns})")
                except:
                    # Fallback to creating as unique index
                    cur.execute(idx['sql'])
            else:
                # Create regular index
                cur.execute(idx['sql'])
            
            conn.commit()
            created.append(idx['name'])
        except Exception as e:
            # Rollback transaction on error and reset to clean state
            conn.rollback()
            status_container.error(f"Error creating index {idx['name']}: {e}")
            # Close and recreate cursor to reset transaction state
            try:
                cur.close()
                cur = conn.cursor()
            except:
                pass
    cur.close()
    return created

def backup_existing_data(conn, table_name, backup_table, upload_mode, status_container, date_column=None):
    """Backup existing data to backup table - supports incremental, truncate, and append modes"""
    if st.session_state.upload_cancelled:
        return 0
    
    cur = conn.cursor()
    try:
        # First, check if main table exists; if not, skip backup
        cur.execute(f"""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = '{table_name}'
            )
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            status_container.text(f"ℹ️ Main table {table_name} doesn't exist yet, skipping backup...")
            cur.close()
            return 0

        # For sales_2026 and similar partitioned tables, just truncate instead of complex backup
        if table_name.lower().startswith('sales_'):
            status_container.text(f"� Dropping and recreating {table_name} for clean upload...")
            # Don't drop sales tables - just append data instead
            status_container.text(f"📝 Table {table_name} exists. Data will be appended...")
            cur.close()
            return 0
        
        # Drop backup table if exists to ensure structure matches
        status_container.text(f"🔧 Ensuring backup table structure matches...")
        cur.execute(f"DROP TABLE IF EXISTS {backup_table}")
        conn.commit()
        
        status_container.text(f"💾 Creating backup table...")
        # For tables with generated columns (like whgrndetails), exclude them from the backup table
        # Use INCLUDING INDEXES but EXCLUDING GENERATED to avoid copying auto-computed columns
        try:
            cur.execute(f"""
                CREATE TABLE {backup_table} 
                (LIKE {table_name} INCLUDING INDEXES EXCLUDING GENERATED)
            """)
        except Exception as exclude_gen_error:
            # If EXCLUDING GENERATED is not supported (older PostgreSQL), fall back to INCLUDING ALL
            # but this may fail for tables with generated columns - warn user
            if "generated" in str(exclude_gen_error).lower() or "syntax" in str(exclude_gen_error).lower():
                status_container.warning(f"⚠️ Using legacy backup method (may fail for tables with generated columns)...")
                cur.execute(f"""
                    CREATE TABLE {backup_table} 
                    (LIKE {table_name} INCLUDING ALL)
                """)
            else:
                raise exclude_gen_error
        conn.commit()
        
        if upload_mode == "append":
            # Append mode: No backup needed, just upload
            status_container.text(f"📝 Append mode: No backup needed...")
            rows_backed_up = 0
        elif upload_mode == "truncate":
            # Full backup before truncate
            status_container.text(f"💾 Full backup to {backup_table} (TRUNCATE mode)...")
            cur.execute(f"INSERT INTO {backup_table} SELECT * FROM {table_name}")
            conn.commit()
            rows_backed_up = cur.rowcount
            
            # Truncate main table
            status_container.text(f"🗑️ Truncating {table_name}...")
            cur.execute(f"TRUNCATE TABLE {table_name}")
            conn.commit()
        else:
            # Incremental backup (only new dates) - use lowercase column names
            status_container.text(f"💾 Incremental backup to {backup_table}...")
            if date_column:
                # Use specified date column for incremental backup
                # Convert to lowercase for compatibility with existing tables
                date_col_lower = date_column.lower()
                try:
                    cur.execute(f"""
                        INSERT INTO {backup_table}
                        SELECT *
                        FROM {table_name}
                        WHERE {date_col_lower} > (
                            SELECT COALESCE(MAX({date_col_lower}), DATE '1900-01-01')
                            FROM {backup_table}
                        )
                    """)
                except Exception as backup_error:
                    # If backup fails, roll back and try without the date filter
                    conn.rollback()
                    status_container.warning(f"⚠️ Incremental backup failed ({str(backup_error)[:50]}), attempting full backup...")
                    cur.execute(f"INSERT INTO {backup_table} SELECT * FROM {table_name}")
            else:
                # No date column, backup all (shouldn't happen for incremental mode)
                cur.execute(f"INSERT INTO {backup_table} SELECT * FROM {table_name}")
            conn.commit()
            rows_backed_up = cur.rowcount
        
        cur.close()
        return rows_backed_up
    except Exception as e:
        conn.rollback()
        cur.close()
        raise e

def parse_mixed_date_series(series, prefer_month_first=False):
    """Parse mixed date formats safely.

    When prefer_month_first=True, ambiguous values (e.g. 02/06/2026) are treated as MM/DD/YYYY first,
    with automatic fallback to DD/MM/YYYY for values that fail primary parsing (e.g. 21/02/2026).
    """
    clean_series = series.replace(r'^\s*$', pd.NA, regex=True)
    primary = pd.to_datetime(clean_series, dayfirst=not prefer_month_first, errors='coerce')
    fallback = pd.to_datetime(clean_series, dayfirst=prefer_month_first, errors='coerce')
    return primary.fillna(fallback)


def _normalize_header_key(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(name).strip().lower())


def _rename_by_expected_columns(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    rename_map = {}
    normalized_actual = {_normalize_header_key(c): c for c in df.columns}
    for expected in expected_columns:
        key = _normalize_header_key(expected)
        if key in normalized_actual:
            actual = normalized_actual[key]
            if actual != expected:
                rename_map[actual] = expected
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _extract_date_from_filename(source_file_name: str) -> Optional[date]:
    if not source_file_name:
        return None

    name = source_file_name.lower()

    patterns = [
        r"(\d{4})[-_](\d{1,2})[-_](\d{1,2})",             # YYYY-MM-DD
        r"(\d{1,2})[-_](\d{1,2})[-_](\d{4})",             # DD-MM-YYYY
        r"(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*(\d{2,4})",  # 7mar26 / 7 mar 2026
    ]

    for pat in patterns:
        m = re.search(pat, name)
        if not m:
            continue
        try:
            if pat.startswith("(\\d{4})"):
                y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return datetime(y, mm, dd).date()
            if "jan|feb|mar" in pat:
                d = int(m.group(1))
                mon_txt = m.group(2)
                y = int(m.group(3))
                if y < 100:
                    y += 2000
                mon_map = {
                    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
                }
                return datetime(y, mon_map[mon_txt], d).date()
            d, mm, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return datetime(y, mm, d).date()
        except Exception:
            continue

    return None


def _prepare_lvo_offloading_df(df: pd.DataFrame, source_file_name: str = "") -> pd.DataFrame:
    """Normalize LVO upload columns to dashboard canonical schema.

    Canonical target columns:
    date, shop_code, vehicle_no, item_code, item_name,
    qty_loaded, value_loaded, qty_offloaded, value_offloaded,
    diff_qty, diff_val
    """
    out = df.copy()
    original_cols = list(out.columns)

    def _find_original_col(*keys: str):
        normalized_to_actual = { _normalize_header_key(col): col for col in original_cols }
        for key in keys:
            actual = normalized_to_actual.get(key)
            if actual is not None:
                return actual
        return None

    alias_map = {
        'date': 'date',
        'dateinvoice': 'date',
        'date_invoice': 'date',
        'dtdate': 'date',
        'offloa': 'date',
        'offloadi': 'date',
        'offloadingdate': 'date',
        'offload_date': 'date',
        'offloaddate': 'date',
        'offload': 'date',
        'shopcode': 'shop_code',
        'shop_code': 'shop_code',
        'shop': 'shop_code',
        'vehicleno': 'vehicle_no',
        'vehicle_no': 'vehicle_no',
        'vehiclenumber': 'vehicle_no',
        'itemcode': 'item_code',
        'item_code': 'item_code',
        'itemname': 'item_name',
        'item_name': 'item_name',
        'qtyloaded': 'qty_loaded',
        'qty_loaded': 'qty_loaded',
        'whloadedqty': 'qty_loaded',
        'loadedqty': 'qty_loaded',
        'offloadedqty': 'qty_loaded',
        'offloadqty': 'qty_loaded',
        'offloadingqty': 'qty_loaded',
        'valueloaded': 'value_loaded',
        'value_loaded': 'value_loaded',
        'whloadedvalue': 'value_loaded',
        'loadedvalue': 'value_loaded',
        'offloadedvalue': 'value_loaded',
        'offloadvalue': 'value_loaded',
        'offloadingvalue': 'value_loaded',
        'qtyoffloaded': 'qty_offloaded',
        'qty_offloaded': 'qty_offloaded',
        'shopreceivingqty': 'qty_offloaded',
        'receive': 'qty_offloaded',
        'received': 'qty_offloaded',
        'receivedqty': 'qty_offloaded',
        'receivingqty': 'qty_offloaded',
        'valueoffloaded': 'value_offloaded',
        'value_offloaded': 'value_offloaded',
        'shopreceivingvalue': 'value_offloaded',
        'cartq': 'qty_loaded',
        'cart_qty': 'qty_loaded',
        'cartonqty': 'qty_loaded',
        'diffqty': 'diff_qty',
        'diff_qty': 'diff_qty',
        'diff': 'diff',
        'diffval': 'diff_val',
        'diff_val': 'diff_val',
        'shopdif': 'diff_val',
        'shopdiff': 'diff_val',
        'shopdifference': 'diff_val',
        'shopvaluediff': 'diff_val',
        'valuediff': 'diff_val',
        'price': 'price',
    }

    rename_map = {}
    for c in original_cols:
        key = _normalize_header_key(c)
        target = alias_map.get(key)
        if target and c != target and target not in out.columns:
            rename_map[c] = target
    if rename_map:
        out = out.rename(columns=rename_map)

    # Explicit CSV business rules from user:
    # - RECEIVED_QTY is warehouse loaded qty
    # - OFFLOADED_QTY is shop receiving qty
    # - Use the actual CSV price column for unit price when present
    # - Use the actual CSV diff column when present
    # - Only fall back to positional J/K when those named columns are absent
    offloaded_qty_col = _find_original_col('offloadedqty')
    received_qty_col = _find_original_col('receivedqty')
    explicit_price_col = _find_original_col('price')
    explicit_diff_col = _find_original_col('diff')
    if received_qty_col is not None:
        out['qty_loaded'] = pd.to_numeric(df[received_qty_col], errors='coerce').fillna(0)
    if offloaded_qty_col is not None:
        out['qty_offloaded'] = pd.to_numeric(df[offloaded_qty_col], errors='coerce').fillna(0)

    if explicit_price_col is not None:
        out['price'] = pd.to_numeric(df[explicit_price_col], errors='coerce').fillna(0)
    elif len(original_cols) >= 10:
        out['price'] = pd.to_numeric(df[original_cols[9]], errors='coerce').fillna(0)

    if explicit_diff_col is not None:
        out['diff'] = pd.to_numeric(df[explicit_diff_col], errors='coerce').fillna(0)
    elif len(original_cols) >= 11:
        out['diff'] = pd.to_numeric(df[original_cols[10]], errors='coerce').fillna(0)

    # Keep canonical quantity diff mapped from raw DIFF while preserving the raw DIFF column.
    if 'diff' in out.columns and 'diff_qty' not in out.columns:
        out['diff_qty'] = pd.to_numeric(out['diff'], errors='coerce')

    # Ensure canonical difference columns exist.
    if 'diff_qty' not in out.columns:
        out['diff_qty'] = 0
    if 'diff_val' not in out.columns:
        out['diff_val'] = 0

    diff_qty_num = pd.to_numeric(out['diff_qty'], errors='coerce').fillna(0)
    diff_val_num = pd.to_numeric(out['diff_val'], errors='coerce').fillna(0)

    # Derive missing loaded/offloaded qty/value in both directions.
    if 'qty_loaded' not in out.columns and 'qty_offloaded' in out.columns:
        out['qty_loaded'] = pd.to_numeric(out['qty_offloaded'], errors='coerce').fillna(0) - diff_qty_num
    if 'qty_offloaded' not in out.columns and 'qty_loaded' in out.columns:
        out['qty_offloaded'] = pd.to_numeric(out['qty_loaded'], errors='coerce').fillna(0) + diff_qty_num

    if 'value_loaded' not in out.columns and 'value_offloaded' in out.columns:
        out['value_loaded'] = pd.to_numeric(out['value_offloaded'], errors='coerce').fillna(0) - diff_val_num
    if 'value_offloaded' not in out.columns and 'value_loaded' in out.columns:
        out['value_offloaded'] = pd.to_numeric(out['value_loaded'], errors='coerce').fillna(0) + diff_val_num

    # Final fallbacks for required numeric columns.
    for req_num in ['qty_loaded', 'value_loaded', 'qty_offloaded', 'value_offloaded']:
        if req_num not in out.columns:
            out[req_num] = 0

    qty_loaded_num = pd.to_numeric(out.get('qty_loaded', 0), errors='coerce').fillna(0)
    qty_offloaded_num = pd.to_numeric(out.get('qty_offloaded', 0), errors='coerce').fillna(0)

    # Canonical business meaning for dashboard:
    # - qty_loaded: warehouse loaded quantity (from RECEIVED_QTY)
    # - qty_offloaded: shop receiving quantity (from OFFLOADED_QTY)
    # - diff_qty: OFFLOADED_QTY - RECEIVED_QTY
    if 'qty_loaded' in out.columns and 'qty_offloaded' in out.columns:
        out['diff_qty'] = qty_offloaded_num - qty_loaded_num
        diff_qty_num = pd.to_numeric(out['diff_qty'], errors='coerce').fillna(0)

    # Rebuild value columns using unit price when value columns are missing/blank/zero.
    # User rule: value columns are always qty * price.
    if 'price' in out.columns:
        unit_price = pd.to_numeric(out.get('price', 0), errors='coerce').fillna(0)
        out['price'] = unit_price
    else:
        unit_price = pd.Series(0.0, index=out.index)
        out['price'] = unit_price

    derived_loaded_val = qty_loaded_num * unit_price
    derived_offloaded_val = qty_offloaded_num * unit_price
    out['value_loaded'] = derived_loaded_val
    out['value_offloaded'] = derived_offloaded_val

    value_loaded_num = pd.to_numeric(out.get('value_loaded', 0), errors='coerce').fillna(0)
    value_offloaded_num = pd.to_numeric(out.get('value_offloaded', 0), errors='coerce').fillna(0)
    out['diff_val'] = value_offloaded_num - value_loaded_num

    # Date fallback for files missing explicit date column.
    if 'date' not in out.columns:
        best_parsed = None
        best_count = 0
        for col in out.columns:
            key = _normalize_header_key(col)
            if not any(tok in key for tok in ['date', 'offload', 'offloading']):
                continue
            try:
                parsed = pd.to_datetime(out[col], errors='coerce', dayfirst=True, format='mixed')
            except Exception:
                try:
                    parsed = pd.to_datetime(out[col], errors='coerce', dayfirst=True)
                except Exception:
                    continue

            parsed_count = int(parsed.notna().sum())
            if parsed_count > best_count:
                best_count = parsed_count
                best_parsed = parsed

        if best_parsed is not None and best_count > 0:
            out['date'] = best_parsed.dt.date

    if 'date' not in out.columns:
        fallback_date = _extract_date_from_filename(source_file_name)
        if fallback_date is None:
            raise ValueError(
                "Date column not found in upload file. Add a date/date_invoice column or include a date in filename (e.g., 2026-03-07 or 7Mar26)."
            )
        out['date'] = fallback_date

    # Keep compatibility columns in sync when present.
    if 'price' in out.columns and 'diff_val' in out.columns:
        out['price'] = pd.to_numeric(out['price'], errors='coerce').fillna(0)
    if 'diff' in out.columns and 'diff_qty' in out.columns:
        out['diff'] = pd.to_numeric(out['diff'], errors='coerce').fillna(pd.to_numeric(out['diff_qty'], errors='coerce').fillna(0))

    return out


def _ensure_loadingvsoffloading_tables(conn, target_table: str, history_mode: str, history_table: str | None = None):
    cur = conn.cursor()
    if target_table == 'offloading_vs_loading':
        cur.execute("""
            CREATE TABLE IF NOT EXISTS public.offloading_vs_loading (
                date DATE,
                shop_code TEXT,
                vehicle_no TEXT,
                item_code TEXT,
                item_name TEXT,
                qty_loaded NUMERIC,
                value_loaded NUMERIC,
                qty_offloaded NUMERIC,
                value_offloaded NUMERIC,
                diff_qty NUMERIC,
                diff_val NUMERIC,
                price NUMERIC,
                diff NUMERIC
            )
        """)
        cur.execute("ALTER TABLE public.offloading_vs_loading ADD COLUMN IF NOT EXISTS price NUMERIC")
        cur.execute("ALTER TABLE public.offloading_vs_loading ADD COLUMN IF NOT EXISTS diff NUMERIC")
    elif target_table == 'offloading_loading_staging':
        cur.execute("""
            CREATE TABLE IF NOT EXISTS public.offloading_loading_staging (
                shop_code TEXT,
                shop_name TEXT
            )
        """)
    elif target_table == 'shopmgrname':
        cur.execute("""
            CREATE TABLE IF NOT EXISTS public.shopmgrname (
                id BIGSERIAL PRIMARY KEY,
                shop_code TEXT NOT NULL,
                shop_description TEXT,
                shop_manager_name TEXT NOT NULL,
                valid_from TIMESTAMP NOT NULL DEFAULT NOW(),
                valid_to TIMESTAMP NULL,
                is_current BOOLEAN NOT NULL DEFAULT TRUE,
                source_file TEXT,
                load_batch_ts TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_shopmgrname_current_triplet
            ON public.shopmgrname (shop_code, COALESCE(shop_description, ''), shop_manager_name)
            WHERE is_current = TRUE
        """)

    if history_mode in ('snapshot_replace', 'append_with_history') and history_table:
        if target_table == 'offloading_vs_loading':
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.offloading_vs_loading_history (
                    id BIGSERIAL PRIMARY KEY,
                    date DATE,
                    shop_code TEXT,
                    vehicle_no TEXT,
                    item_code TEXT,
                    item_name TEXT,
                    qty_loaded NUMERIC,
                    value_loaded NUMERIC,
                    qty_offloaded NUMERIC,
                    value_offloaded NUMERIC,
                    diff_qty NUMERIC,
                    diff_val NUMERIC,
                    price NUMERIC,
                    diff NUMERIC,
                    history_action TEXT NOT NULL,
                    history_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    source_file TEXT
                )
            """)
        elif target_table == 'offloading_loading_staging':
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.offloading_loading_staging_history (
                    id BIGSERIAL PRIMARY KEY,
                    shop_code TEXT,
                    shop_name TEXT,
                    history_action TEXT NOT NULL,
                    history_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    source_file TEXT
                )
            """)

    conn.commit()
    cur.close()


def _upload_snapshot_replace_with_history(conn, table_name: str, history_table: str, df_upload: pd.DataFrame, columns: list[str], source_file: str, status_container):
    cur = conn.cursor()
    col_sql = ', '.join([_quote_ident(c) for c in columns])

    status_container.info(f"🕘 Saving previous snapshot to history: {history_table}")
    cur.execute(
        f"INSERT INTO {_quote_ident(history_table)} ({col_sql}, history_action, history_at, source_file) "
        f"SELECT {col_sql}, 'before_replace', NOW(), %s FROM {_quote_ident(table_name)}",
        (source_file,),
    )

    status_container.info(f"🧹 Replacing current data in {table_name}")
    cur.execute(f"TRUNCATE TABLE {_quote_ident(table_name)}")

    _null = lambda v: None if pd.isnull(v) else v
    values = [tuple(_null(row.get(c)) for c in columns) for _, row in df_upload.iterrows()]
    if values:
        execute_values(
            cur,
            f"INSERT INTO {_quote_ident(table_name)} ({col_sql}) VALUES %s",
            values,
            page_size=5000,
        )

    if table_name == 'offloading_vs_loading':
        cur.execute(
            """
            UPDATE public.offloading_vs_loading
            SET diff = diff_qty
            WHERE diff IS DISTINCT FROM diff_qty
            """
        )

    status_container.info(f"📝 Saving uploaded snapshot to history: {history_table}")
    cur.execute(
        f"INSERT INTO {_quote_ident(history_table)} ({col_sql}, history_action, history_at, source_file) "
        f"SELECT {col_sql}, 'uploaded_snapshot', NOW(), %s FROM {_quote_ident(table_name)}",
        (source_file,),
    )

    if table_name == 'offloading_vs_loading':
        cur.execute(
            """
            UPDATE public.offloading_vs_loading_history
            SET diff = diff_qty
            WHERE history_action = 'uploaded_snapshot'
              AND history_at >= NOW() - INTERVAL '5 minutes'
            """
        )

    conn.commit()
    cur.close()
    return len(values)


def _upload_append_with_history(conn, table_name: str, history_table: str, df_upload: pd.DataFrame, columns: list[str], source_file: str, status_container):
    cur = conn.cursor()
    col_sql = ', '.join([_quote_ident(c) for c in columns])
    _null = lambda v: None if pd.isnull(v) else v
    values = [tuple(_null(row.get(c)) for c in columns) for _, row in df_upload.iterrows()]

    if values:
        status_container.info(f"📥 Appending rows into {table_name}")
        execute_values(
            cur,
            f"INSERT INTO {_quote_ident(table_name)} ({col_sql}) VALUES %s",
            values,
            page_size=5000,
        )

        if table_name == 'offloading_vs_loading':
            cur.execute(
                """
                UPDATE public.offloading_vs_loading
                SET diff = diff_qty
                WHERE diff IS DISTINCT FROM diff_qty
                """
            )

        status_container.info(f"📝 Writing append history in {history_table}")
        now_ts = datetime.now()
        history_values = [tuple(_null(row.get(c)) for c in columns) + ('append_upload', now_ts, source_file) for _, row in df_upload.iterrows()]
        history_col_sql = ', '.join([_quote_ident(c) for c in columns + ['history_action', 'history_at', 'source_file']])
        execute_values(
            cur,
            f"INSERT INTO {_quote_ident(history_table)} ({history_col_sql}) VALUES %s",
            history_values,
            page_size=5000,
        )

        if table_name == 'offloading_vs_loading':
            cur.execute(
                """
                UPDATE public.offloading_vs_loading_history
                SET diff = diff_qty
                WHERE history_action = 'append_upload'
                  AND history_at >= NOW() - INTERVAL '10 minutes'
                """
            )

    conn.commit()
    cur.close()
    return len(values)


def _sync_shopmgrname_scd2(conn, df_upload: pd.DataFrame, source_file: str, status_container):
    cur = conn.cursor()
    ts = datetime.now()

    clean = df_upload.copy()
    clean['shop_code'] = clean['shop_code'].astype(str).str.strip().str.upper()
    clean['shop_description'] = clean['shop_description'].astype(str).str.strip()
    clean['shop_manager_name'] = clean['shop_manager_name'].astype(str).str.strip()
    clean['shop_manager_name'] = clean['shop_manager_name'].str.replace(r'\s*\(SHOP MANAGER\)\s*', ' ', regex=True, flags=re.IGNORECASE)
    clean['shop_manager_name'] = clean['shop_manager_name'].str.replace(r'\s{2,}', ' ', regex=True).str.strip()
    clean = clean[(clean['shop_code'] != '') & (clean['shop_manager_name'] != '')]
    clean = clean.drop_duplicates(subset=['shop_code', 'shop_description', 'shop_manager_name'])

    cur.execute("DROP TABLE IF EXISTS tmp_shopmgr_upload")
    cur.execute("""
        CREATE TEMP TABLE tmp_shopmgr_upload (
            shop_code TEXT NOT NULL,
            shop_description TEXT,
            shop_manager_name TEXT NOT NULL
        ) ON COMMIT DROP
    """)

    values = [tuple(x) for x in clean[['shop_code', 'shop_description', 'shop_manager_name']].to_records(index=False)]
    if values:
        execute_values(
            cur,
            "INSERT INTO tmp_shopmgr_upload (shop_code, shop_description, shop_manager_name) VALUES %s",
            values,
            page_size=1000,
        )

    status_container.info("🕘 Closing outdated current manager rows")
    cur.execute(
        """
        UPDATE public.shopmgrname t
        SET is_current = FALSE,
            valid_to = %(ts)s,
            updated_at = %(ts)s
        WHERE t.is_current = TRUE
          AND NOT EXISTS (
                SELECT 1
                FROM tmp_shopmgr_upload u
                WHERE u.shop_code = t.shop_code
                  AND COALESCE(u.shop_description, '') = COALESCE(t.shop_description, '')
                  AND u.shop_manager_name = t.shop_manager_name
          )
        """,
        {"ts": ts},
    )

    status_container.info("➕ Inserting new current manager rows")
    cur.execute(
        """
        INSERT INTO public.shopmgrname (
            shop_code,
            shop_description,
            shop_manager_name,
            valid_from,
            valid_to,
            is_current,
            source_file,
            load_batch_ts,
            updated_at
        )
        SELECT
            u.shop_code,
            u.shop_description,
            u.shop_manager_name,
            %(ts)s,
            NULL,
            TRUE,
            %(src)s,
            %(ts)s,
            %(ts)s
        FROM tmp_shopmgr_upload u
        WHERE NOT EXISTS (
            SELECT 1
            FROM public.shopmgrname t
            WHERE t.is_current = TRUE
              AND t.shop_code = u.shop_code
              AND COALESCE(t.shop_description, '') = COALESCE(u.shop_description, '')
              AND t.shop_manager_name = u.shop_manager_name
        )
        """,
        {"ts": ts, "src": source_file},
    )

    conn.commit()
    cur.close()
    return len(values)

def upload_data_to_table(conn, table_name, df, columns, status_container, is_partitioned=False, database=None, add_upload_date=False, table_config=None, source_file_name=None):
    """Upload dataframe to table using COPY
    
    Args:
        add_upload_date: If True, adds upload_date column with current timestamp
    """
    if st.session_state.upload_cancelled:
        return 0
    
    # Add upload_date column if requested (for append mode tracking)
    if add_upload_date:
        df['upload_date'] = datetime.now().strftime('%Y-%m-%d')
        if 'upload_date' not in columns:
            columns = list(columns) + ['upload_date']
        status_container.info(f"📅 Added upload_date column: {df['upload_date'].iloc[0]}")

    if table_config and table_config.get('history_mode') in ('snapshot_replace', 'append_with_history', 'shopmgr_scd2'):
        history_mode = table_config.get('history_mode')
        history_table = table_config.get('history_table')
        _ensure_loadingvsoffloading_tables(conn, table_name, history_mode, history_table)
        if table_name == 'offloading_vs_loading':
            df = _prepare_lvo_offloading_df(df, source_file_name or '')
        df = _rename_by_expected_columns(df, columns)
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for {table_name}: {', '.join(missing)}")
        df_upload = df[columns].copy()

        if 'date' in df_upload.columns:
            df_upload['date'] = pd.to_datetime(df_upload['date'], errors='coerce').dt.date
        for num_col in ['qty_loaded', 'value_loaded', 'qty_offloaded', 'value_offloaded', 'diff_qty', 'diff_val']:
            if num_col in df_upload.columns:
                df_upload[num_col] = pd.to_numeric(df_upload[num_col], errors='coerce').fillna(0)

        source_name = source_file_name or ''
        if history_mode == 'snapshot_replace':
            return _upload_snapshot_replace_with_history(conn, table_name, history_table, df_upload, columns, source_name, status_container)
        if history_mode == 'append_with_history':
            return _upload_append_with_history(conn, table_name, history_table, df_upload, columns, source_name, status_container)
        return _sync_shopmgrname_scd2(conn, df_upload, source_name, status_container)
    
    cur = conn.cursor()
    total_rows_uploaded = 0
    
    try:
        # For partitioned sales table, split by appropriate period and upload to partitions
        if is_partitioned and table_name.lower() == 'sales' and 'date_invoice' in columns:
            # First, parse the date column with correct format (DD/MM/YYYY from Excel)
            date_col = 'date_invoice' if 'date_invoice' in df.columns else 'DATE_INVOICE'
            try:
                # Parse with dayfirst=True (DD/MM/YYYY format from Excel)
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, format='mixed')
                status_container.info(f"✅ Parsed date column with DD/MM/YYYY format")
            except:
                try:
                    # Fallback: Try ISO format (YYYY-MM-DD)
                    df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d')
                    status_container.info(f"✅ Parsed date column with YYYY-MM-DD format")
                except:
                    # Final fallback
                    df[date_col] = pd.to_datetime(df[date_col])
                    status_container.warning(f"⚠️ Auto-detected date format. Verify dates are correct!")
            
            if database == 'salesdata':
                # Year-partitioned (salesdata database: sales_2024, sales_2025, sales_2026)
                status_container.text(f"📤 Uploading {len(df):,} rows to year-partitioned tables...")
                df['_partition_key'] = df[date_col].dt.year
                partition_label = 'year'
            elif database == 'century_penetration':
                # Month-partitioned (century_penetration: sales_jan2025, sales_feb2025, ..., sales_dec2026)
                status_container.text(f"📤 Uploading {len(df):,} rows to month-partitioned tables...")
                df['_partition_key'] = df[date_col].dt.strftime('%b%Y').str.lower()
                partition_label = 'month'
            else:
                raise ValueError(f"Unknown database: {database}")
            
            partitions = sorted(df['_partition_key'].unique())
            
            for partition_val in partitions:
                if st.session_state.upload_cancelled:
                    return total_rows_uploaded
                
                if database == 'salesdata':
                    partition_name = f"sales_{int(partition_val)}"
                else:  # century_penetration
                    # Convert from 'jan2026' to 'sales_jan2026'
                    partition_name = f"sales_{partition_val}"
                
                # Ensure partition table exists with correct structure
                status_container.text(f"🔍 Checking/creating partition table: {partition_name}...")
                try:
                    # First, check if table exists and has the right structure
                    cur.execute(f"""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_name = '{partition_name}'
                        )
                    """)
                    table_exists = cur.fetchone()[0]
                    
                    if table_exists:
                        # Table exists, check if it has the required columns (case-insensitive)
                        required_cols = {"shop_code", "item_code", "item_name", "dept", "groups", 
                                       "sub_group", "qty", "net_sales", "date_invoice"}
                        cur.execute(f"""
                            SELECT LOWER(column_name) FROM information_schema.columns 
                            WHERE table_name = '{partition_name}'
                        """)
                        existing_cols = {row[0] for row in cur.fetchall()}
                        
                        if not required_cols.issubset(existing_cols):
                            # Missing columns, drop and recreate
                            status_container.text(f"🔧 Recreating {partition_name} with correct schema...")
                            cur.execute(f"DROP TABLE IF EXISTS {partition_name}")
                            conn.commit()
                            table_exists = False
                        else:
                            # Table exists with all required columns
                            status_container.text(f"✅ Table {partition_name} exists with correct schema")
                    
                    if not table_exists:
                        # Create table with correct schema (lowercase for compatibility)
                        cur.execute(f"""
                            CREATE TABLE {partition_name} (
                                shop_code VARCHAR,
                                item_code VARCHAR,
                                item_name VARCHAR,
                                dept VARCHAR,
                                groups VARCHAR,
                                sub_group VARCHAR,
                                qty NUMERIC,
                                net_sales NUMERIC,
                                date_invoice DATE
                            )
                        """)
                        conn.commit()
                        status_container.text(f"✅ Created table {partition_name}")
                except Exception as create_error:
                    conn.rollback()
                    status_container.error(f"❌ Error with table {partition_name}: {str(create_error)[:100]}")
                    continue
                
                df_partition = df[df['_partition_key'] == partition_val].drop('_partition_key', axis=1)
                
                status_container.text(f"📤 Uploading {len(df_partition):,} rows to partition: {partition_name}")
                
                # FIX: Convert qty to integer (database column is INTEGER, pandas reads CSV as float)
                # and fill NaN values to prevent COPY errors
                if 'qty' in df_partition.columns:
                    df_partition['qty'] = df_partition['qty'].fillna(0).round(0).astype('int64')
                if 'net_sales' in df_partition.columns:
                    df_partition['net_sales'] = df_partition['net_sales'].fillna(0)
                
                buffer = io.StringIO()
                df_partition.to_csv(buffer, index=False, header=False)
                buffer.seek(0)
                
                # Map uppercase column names to lowercase for COPY command
                col_map = {
                    'SHOP_CODE': 'shop_code',
                    'ITEM_CODE': 'item_code',
                    'ITEM_NAME': 'item_name',
                    'DEPT': 'dept',
                    'GROUPS': 'groups',
                    'SUB_GROUP': 'sub_group',
                    'QTY': 'qty',
                    'NET_SALES': 'net_sales',
                    'DATE_INVOICE': 'date_invoice'
                }
                
                copy_cols = [col_map.get(col, col.lower()) for col in columns if col != '_partition_key']
                copy_sql = f"""
                    COPY {partition_name} ({', '.join(copy_cols)})
                    FROM STDIN WITH CSV
                """
                
                try:
                    cur.copy_expert(copy_sql, buffer)
                    conn.commit()
                except Exception as copy_error:
                    conn.rollback()
                    status_container.error(f"❌ Error uploading to {partition_name}: {str(copy_error)[:100]}")
                    continue
                
                rows_uploaded = len(df_partition)
                total_rows_uploaded += rows_uploaded
                status_container.success(f"✅ {partition_name}: {rows_uploaded:,} rows uploaded")
            
            # Remove temporary partition key column
            if '_partition_key' in df.columns:
                df = df.drop('_partition_key', axis=1)
        
        else:
            # Non-partitioned table upload (century_penetration.sales or other tables)
            # Try to disable autovacuum (may fail for partitioned tables, that's OK)
            if not is_partitioned:
                # For non-partitioned tables, ensure table exists first
                status_container.text(f"🔍 Checking/creating table {table_name}...")
                try:
                    # Check if table exists
                    cur.execute(f"""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_name = '{table_name}'
                        )
                    """)
                    table_exists = cur.fetchone()[0]
                    
                    # For serial_no_dailydata_staging, always drop and recreate to ensure correct schema
                    if table_name == 'serial_no_dailydata_staging' and table_exists:
                        status_container.text(f"🔄 Dropping existing table {table_name} to recreate with correct schema...")
                        cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
                        conn.commit()
                        table_exists = False
                        status_container.success(f"✅ Dropped old table {table_name}")
                    
                    if not table_exists:
                        # Create table with lowercase columns for compatibility
                        status_container.text(f"🔨 Creating table {table_name}...")
                        
                        # Dynamically build CREATE TABLE statement based on columns
                        # For whstock and similar tables, include upload_date if configured
                        if table_name == 'whstock':
                            cur.execute(f"""
                                CREATE TABLE {table_name} (
                                    vc_item_code VARCHAR(50) NOT NULL,
                                    wh_code VARCHAR(10) NOT NULL,
                                    wh_name VARCHAR(100),
                                    balance_qty NUMERIC(15, 2) DEFAULT 0,
                                    upload_date DATE NOT NULL,
                                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                )
                            """)
                        elif table_name == 'serial_no_dailydata_staging':
                            # Create staging table with all 24 columns from CSV (lowercase)
                            # No PRIMARY KEY on staging - allows duplicates, uniqueness enforced on main table
                            cur.execute(f"""
                                CREATE TABLE {table_name} (
                                    vc_warehouse_desc VARCHAR(200),
                                    vc_wh_code VARCHAR(10),
                                    nu_doc_id BIGINT,
                                    dt_doc_date DATE,
                                    loaded_datetime TIMESTAMP,
                                    vc_shop_code VARCHAR(10),
                                    shop_name VARCHAR(200),
                                    vc_item_code VARCHAR(50),
                                    vc_item_desc TEXT,
                                    nu_selling_price NUMERIC(15,2),
                                    serial_no VARCHAR(100),
                                    wh_load_user VARCHAR(100),
                                    loadingno INTEGER,
                                    loadingdate DATE,
                                    vc_load_no VARCHAR(50),
                                    dt_load_date DATE,
                                    vc_vehicle_no VARCHAR(50),
                                    vc_serail_no VARCHAR(100),
                                    "vc_vehicle_no_1" VARCHAR(50),
                                    dt_mod_date DATE,
                                    shop_sold VARCHAR(10),
                                    vc_invoice_no VARCHAR(50),
                                    dt_invoice_date DATE,
                                    shop_serail_no VARCHAR(100),
                                    upload_date DATE DEFAULT CURRENT_DATE
                                )
                            """)
                        elif table_name == 'whreceived_serialno':
                            # Create table with correct structure matching CSV (no need to drop as add_upload_date handles schema)
                            cur.execute(f"""
                                CREATE TABLE {table_name} (
                                    inbound_type TEXT,
                                    warehouse_name TEXT,
                                    supp_name TEXT,
                                    serial_no TEXT,
                                    grn_date DATE,
                                    item_code TEXT,
                                    item_desc TEXT,
                                    vc_inbond_type TEXT,
                                    serial_qty INTEGER,
                                    uploaded_data_date DATE DEFAULT CURRENT_DATE
                                )
                            """)
                        elif table_name == 'serialno_check_yes_no':
                            # Create table for serial number check report (Y/N)
                            cur.execute(f"""
                                CREATE TABLE {table_name} (
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
                        else:
                            # Standard sales table structure
                            cur.execute(f"""
                                CREATE TABLE {table_name} (
                                    shop_code VARCHAR,
                                    item_code VARCHAR,
                                    item_name VARCHAR,
                                    dept VARCHAR,
                                    groups VARCHAR,
                                    sub_group VARCHAR,
                                    qty NUMERIC,
                                    net_sales NUMERIC,
                                    date_invoice DATE
                                )
                            """)
                        conn.commit()
                        status_container.success(f"✅ Created table {table_name}")
                    else:
                        # Table exists - check if upload_date column exists for whstock
                        if table_name == 'whstock' or table_name == 'serial_no_dailydata':
                            try:
                                cur.execute(f"""
                                    SELECT column_name 
                                    FROM information_schema.columns 
                                    WHERE table_name = '{table_name}' AND column_name = 'upload_date'
                                """)
                                has_upload_date = cur.fetchone() is not None
                                
                                if not has_upload_date:
                                    status_container.info(f"ℹ️ Adding upload_date column to {table_name}...")
                                    cur.execute(f"ALTER TABLE {table_name} ADD COLUMN upload_date DATE")
                                    conn.commit()
                                    status_container.success(f"✅ Added upload_date column")
                            except Exception as col_error:
                                conn.rollback()
                                status_container.warning(f"⚠️ Could not add upload_date column: {str(col_error)[:100]}")
                        
                        status_container.text(f"✅ Table {table_name} exists, proceeding with upload")
                except Exception as create_err:
                    conn.rollback()
                    status_container.error(f"❌ Error creating table {table_name}: {str(create_err)[:100]}")
                    raise
                
                # Now disable autovacuum
                try:
                    status_container.text(f"⚙️ Disabling autovacuum...")
                    cur.execute(f"ALTER TABLE {table_name} SET (autovacuum_enabled = false)")
                    conn.commit()
                except Exception as e:
                    # Skip autovacuum if it fails
                    conn.rollback()
                    status_container.info(f"ℹ️ Skipping autovacuum settings")
            
            status_container.text(f"📤 Uploading {len(df):,} rows...")
            
            # Convert date columns to DATE format (without time) for COPY compatibility
            # Expected format: DD/MM/YYYY or YYYY-MM-DD (dayfirst=True for DD/MM/YYYY Excel format)
            date_cols = ['date_invoice', 'DATE_INVOICE', 'Date_Invoice']
            for col in date_cols:
                if col in df.columns:
                    try:
                        # First attempt: Parse with dayfirst=True (DD/MM/YYYY format from Excel)
                        df[col] = pd.to_datetime(df[col], dayfirst=True, format='mixed').dt.strftime('%Y-%m-%d')
                        status_container.info(f"✅ Parsed date column '{col}' with DD/MM/YYYY format")
                    except Exception as date_err:
                        # Second attempt: Try ISO format (YYYY-MM-DD)
                        try:
                            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
                            status_container.info(f"✅ Parsed date column '{col}' with YYYY-MM-DD format")
                        except:
                            # Final fallback: Let pandas infer but show warning
                            try:
                                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
                                status_container.warning(f"⚠️ Auto-detected date format for '{col}'. Verify dates are correct!")
                            except:
                                status_container.error(f"❌ Could not parse date column '{col}'. Please use DD/MM/YYYY or YYYY-MM-DD format")
                                raise ValueError(f"Invalid date format in column '{col}'. Expected: DD/MM/YYYY (e.g., 29/01/2026) or YYYY-MM-DD (e.g., 2026-01-29)")
                    break
            
            buffer = io.StringIO()
            df.to_csv(buffer, index=False, header=False)
            buffer.seek(0)
            
            # Map uppercase column names based on table type
            # Sales tables use lowercase mapping, other tables preserve uppercase (quoted)
            if table_name.lower().startswith('sales_') or table_name.lower() == 'sales':
                col_map = {
                    'SHOP_CODE': 'shop_code',
                    'ITEM_CODE': 'item_code',
                    'ITEM_NAME': 'item_name',
                    'DEPT': 'dept',
                    'GROUPS': 'groups',
                    'SUB_GROUP': 'sub_group',
                    'QTY': 'qty',
                    'NET_SALES': 'net_sales',
                    'DATE_INVOICE': 'date_invoice'
                }
                copy_cols = [col_map.get(col, col.lower()) for col in columns]
            elif table_name.lower() in ('whstock', 'serial_no_dailydata_staging', 'whreceived_serialno', 'serialno_check_yes_no'):
                # WHStock and serial tables - use lowercase column names
                # Quote column names that contain dots (from pandas duplicate column renaming)
                copy_cols = []
                for col in columns:
                    col_lower = col.lower()
                    if '.' in col_lower:
                        # Column contains dot (e.g., VC_VEHICLE_NO_1) - quote it
                        copy_cols.append(f'"{col_lower}"')
                    else:
                        # Normal column - no quotes needed
                        copy_cols.append(col_lower)
            else:
                # Other tables (shopexpiry, etc.) - quote uppercase column names to preserve case
                copy_cols = [f'"{col}"' for col in columns]
            copy_sql = f"""
                COPY {table_name} ({', '.join(copy_cols)})
                FROM STDIN WITH CSV
            """
            
            # Special upsert flow for serial_no_dailydata in 'wh' DB: load into temp table then upsert
            if table_name.lower() == 'serial_no_dailydata' and (database or '').lower() == 'wh':
                try:
                    tmp_table = f"tmp_{table_name}_{os.getpid()}"
                    status_container.text(f"🔁 Creating temp table for upsert: {tmp_table}...")
                    # Create a temp table like the target to preserve structure
                    cur.execute(f"CREATE TEMP TABLE {tmp_table} (LIKE {table_name} INCLUDING ALL) ON COMMIT DROP")
                    conn.commit()

                    # Copy into temp table
                    copy_sql_tmp = f"COPY {tmp_table} ({', '.join(copy_cols)}) FROM STDIN WITH CSV"
                    cur.copy_expert(copy_sql_tmp, buffer)
                    conn.commit()

                    # Determine intersection of columns between temp and target (lowercase)
                    cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'")
                    target_cols = [r[0].lower() for r in cur.fetchall()]
                    insert_cols = [c for c in copy_cols if c.lower() in target_cols]

                    if 'serial_no' not in [c.lower() for c in insert_cols]:
                        raise ValueError('serial_no column is required for upsert')

                    # Build upsert SQL
                    insert_cols_sql = ', '.join(insert_cols)
                    select_cols_sql = ', '.join(insert_cols)

                    # Build update assignments excluding serial_no and timestamps
                    update_cols = [c for c in insert_cols if c.lower() not in ('serial_no', 'created_at', 'updated_at')]
                    set_sql_parts = []
                    for c in update_cols:
                        if c.lower() == 'uploaded_data_date':
                            # set uploaded_data_date to current date on update
                            set_sql_parts.append(f"{c} = CURRENT_DATE")
                        else:
                            set_sql_parts.append(f"{c} = EXCLUDED.{c}")
                    set_sql = ', '.join(set_sql_parts) if set_sql_parts else 'NOTHING'

                    upsert_sql = f"""
                        INSERT INTO {table_name} ({insert_cols_sql})
                        SELECT {select_cols_sql} FROM {tmp_table}
                        ON CONFLICT (serial_no) DO UPDATE SET {set_sql}
                    """

                    status_container.text("🔄 Performing upsert into target table...")
                    cur.execute(upsert_sql)
                    conn.commit()
                    rows_affected = cur.rowcount if cur.rowcount is not None else 0
                    status_container.success(f"✅ Upsert complete: {rows_affected:,} rows affected")
                except Exception as up_err:
                    conn.rollback()
                    status_container.error(f"❌ Upsert failed, falling back to direct COPY: {str(up_err)[:150]}")
                    # Fallback to direct copy to target table
                    buffer.seek(0)
                    cur.copy_expert(copy_sql, buffer)
                    conn.commit()
            else:
                cur.copy_expert(copy_sql, buffer)
                conn.commit()
            
            # Try to re-enable autovacuum (may fail, that's OK)
            if not is_partitioned:
                try:
                    status_container.text(f"⚙️ Re-enabling autovacuum...")
                    cur.execute(f"ALTER TABLE {table_name} SET (autovacuum_enabled = true)")
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    status_container.info(f"ℹ️ Skipping autovacuum re-enable")
            
            total_rows_uploaded = len(df)
        
        # ANALYZE table
        try:
            status_container.text(f"📊 Analyzing table...")
            cur.execute(f"ANALYZE {table_name}")
            conn.commit()
        except Exception as e:
            # ANALYZE may fail, skip it
            conn.rollback()
            status_container.info(f"ℹ️ Skipping ANALYZE (may not be needed)")
        
        cur.close()
        return total_rows_uploaded
    except Exception as e:
        conn.rollback()
        cur.close()
        raise e

def execute_index_query(conn, query, status_container):
    """Execute a single index creation query"""
    if st.session_state.upload_cancelled:
        return False
    
    try:
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        conn.rollback()
        status_container.error(f"❌ Error: {e}")
        return False

def _mysql_to_postgres_type(mysql_data_type: str, mysql_column_type: str) -> str:
    """Map MySQL data types to PostgreSQL types."""
    dt = (mysql_data_type or '').lower()
    ct = (mysql_column_type or '').lower()

    if dt in ('tinyint', 'smallint'):
        if dt == 'tinyint' and ct.startswith('tinyint(1)'):
            return 'BOOLEAN'
        return 'SMALLINT'
    if dt in ('int', 'integer', 'mediumint'):
        return 'INTEGER'
    if dt == 'bigint':
        return 'BIGINT'
    if dt in ('decimal', 'numeric'):
        m = re.search(r'\((\d+),(\d+)\)', ct)
        if m:
            return f"NUMERIC({m.group(1)},{m.group(2)})"
        return 'NUMERIC'
    if dt == 'float':
        return 'REAL'
    if dt in ('double', 'double precision'):
        return 'DOUBLE PRECISION'
    if dt == 'date':
        return 'DATE'
    if dt in ('datetime', 'timestamp'):
        return 'TIMESTAMP'
    if dt == 'time':
        return 'TIME'
    if dt in ('char', 'varchar'):
        m = re.search(r'\((\d+)\)', ct)
        if m:
            return f"VARCHAR({m.group(1)})"
        return 'VARCHAR'
    if 'text' in dt:
        return 'TEXT'
    if dt in ('json',):
        return 'JSONB'
    if dt in ('blob', 'longblob', 'mediumblob', 'tinyblob', 'binary', 'varbinary'):
        return 'BYTEA'
    return 'TEXT'

def _ensure_sync_metadata_tables(pg_conn):
    """Create metadata tables for synchronization state and logs."""
    cur = pg_conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS external_sync_state (
            source_system TEXT NOT NULL,
            source_table TEXT NOT NULL,
            strategy TEXT NOT NULL,
            cursor_column TEXT,
            cursor_value TEXT,
            last_synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source_system, source_table)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS external_sync_log (
            id BIGSERIAL PRIMARY KEY,
            source_system TEXT NOT NULL,
            source_table TEXT NOT NULL,
            target_table TEXT NOT NULL,
            strategy TEXT NOT NULL,
            fetched_rows BIGINT NOT NULL DEFAULT 0,
            inserted_rows BIGINT NOT NULL DEFAULT 0,
            duplicate_rows BIGINT NOT NULL DEFAULT 0,
            deleted_rows BIGINT NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            error_message TEXT,
            started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            finished_at TIMESTAMP
        )
    """)
    # Add deleted_rows column if it doesn't exist (for existing installations)
    try:
        cur.execute("""
            ALTER TABLE external_sync_log 
            ADD COLUMN IF NOT EXISTS deleted_rows BIGINT NOT NULL DEFAULT 0
        """)
    except Exception:
        pass  # Column already exists or error, safe to ignore
    pg_conn.commit()
    cur.close()

def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'

def _serialize_row_for_hash(row_dict: dict, ordered_columns: list) -> str:
    parts = []
    for col in ordered_columns:
        value = row_dict.get(col)
        parts.append('' if value is None else str(value))
    return '|'.join(parts)

def _check_port_accessible(host: str, port: int = 3306, timeout: int = 5) -> tuple[bool, str]:
    """Quick TCP port check before attempting MySQL connection."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            return True, ""
        else:
            return False, f"Port {port} unreachable on {host} (connection refused)."
    except socket.timeout:
        return False, f"Port {port} timeout on {host}. Firewall/network blocking."
    except socket.gaierror:
        return False, f"Cannot resolve {host}. Check DNS."
    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        sock.close()


def _validate_mysql_connection(mysql_config: dict) -> tuple[bool, str]:
    """Test MySQL connection with TCP port check first to fail fast."""
    # Step 1: Quick TCP check (prevents repeated auth fails that trigger block)
    port_ok, port_msg = _check_port_accessible(mysql_config['host'], 3306, timeout=5)
    if not port_ok:
        return False, f"❌ Port unreachable\n\nReason: {port_msg}\n\nCheck: ping {mysql_config['host']}"
    
    # Step 2: Test auth once
    try:
        import pymysql
    except ImportError:
        return False, "pymysql not installed. pip install pymysql"
    
    try:
        conn = pymysql.connect(
            host=mysql_config['host'],
            user=mysql_config['user'],
            password=mysql_config['password'],
            database=mysql_config['database'],
            charset='utf8mb4',
            connect_timeout=8, read_timeout=30
        )
        conn.close()
        return True, "✅ Connection OK"
    except pymysql.err.OperationalError as e:
        error_code = e.args[0] if e.args else 0
        error_msg = str(e)
        if error_code == 1129:
            return False, f"🔒 Host blocked (error 1129)\n\nMySQL admin recovery:\nmysqladmin -u root -p flush-hosts\n\nOr restart MySQL. Wait 30s, retry.\n\nError: {error_msg}"
        elif error_code == 1045:
            return False, f"❌ Auth failed ({error_code}). Check user/password. Error: {error_msg}"
        else:
            return False, f"❌ Connection error ({error_code}): {error_msg}"
    except Exception as e:
        return False, f"❌ Unexpected error: {str(e)}"


def _reconcile_snapshot_table(
    mysql_cur,
    pg_cur,
    pg_conn,
    source_table: str,
    target_table: str,
    pk_cols: list,
    columns: list,
    status_container=None,
    batch_size: int = 1000,
) -> dict:
    """
    Reconcile eligible snapshot-style tables by deleting stale PostgreSQL rows and
    backfilling missing primary keys from MySQL before the normal incremental sync runs.

    Returns a dict with deleted/fetched/inserted counts.
    """
    reconcile_tables = {'ALERTS', 'INVOICES_MANAGER'}
    if source_table.upper() not in reconcile_tables or len(pk_cols) != 1:
        return {'deleted_rows': 0, 'fetched_rows': 0, 'inserted_rows': 0}

    try:
        pk_col = pk_cols[0]
        pg_col_lower = pk_col.lower()

        if status_container:
            status_container.info(f"🔄 Reconciling `{target_table}` snapshot state...")

        mysql_cur.execute(f"SELECT `{pk_col}` FROM `{source_table}` WHERE `{pk_col}` IS NOT NULL")
        mysql_ids = {row[pk_col] for row in mysql_cur.fetchall()}

        if not mysql_ids:
            return {'deleted_rows': 0, 'fetched_rows': 0, 'inserted_rows': 0}

        pg_cur.execute(
            f'SELECT {_quote_ident(pg_col_lower)} FROM {_quote_ident(target_table)} '
            f'WHERE {_quote_ident(pg_col_lower)} IS NOT NULL'
        )
        pg_ids = {row[0] for row in pg_cur.fetchall()}

        stale_ids = pg_ids - mysql_ids
        missing_ids = mysql_ids - pg_ids

        deleted_count = 0
        fetched_missing = 0
        inserted_missing = 0

        if stale_ids:
            pg_cur.execute(
                f'DELETE FROM {_quote_ident(target_table)} WHERE {_quote_ident(pg_col_lower)} = ANY(%s)',
                (list(stale_ids),)
            )
            deleted_count = pg_cur.rowcount if pg_cur.rowcount is not None else 0
            pg_conn.commit()

        if missing_ids:
            pg_cols = [c.lower() for c in columns]
            insert_col_sql = ', '.join([_quote_ident(c) for c in pg_cols])
            pk_conflict = ', '.join([_quote_ident(c.lower()) for c in pk_cols])
            insert_sql = (
                f"INSERT INTO {_quote_ident(target_table)} ({insert_col_sql}) VALUES %s "
                f"ON CONFLICT ({pk_conflict}) DO NOTHING"
            )

            sorted_missing_ids = sorted(missing_ids)
            for start_index in range(0, len(sorted_missing_ids), batch_size):
                chunk_ids = sorted_missing_ids[start_index:start_index + batch_size]
                placeholders = ', '.join(['%s'] * len(chunk_ids))
                select_cols_mysql = ', '.join([f"`{c}`" for c in columns])
                mysql_cur.execute(
                    f"SELECT {select_cols_mysql} FROM `{source_table}` WHERE `{pk_col}` IN ({placeholders})",
                    chunk_ids
                )
                rows = mysql_cur.fetchall()
                if not rows:
                    continue

                value_rows = []
                for row in rows:
                    normalized = {k.lower(): v for k, v in row.items()}
                    value_rows.append(tuple([normalized.get(c) for c in pg_cols]))

                execute_values(pg_cur, insert_sql, value_rows, page_size=min(batch_size, 5000))
                inserted = pg_cur.rowcount if pg_cur.rowcount is not None else 0
                pg_conn.commit()

                fetched_missing += len(rows)
                inserted_missing += inserted

        if status_container:
            if not stale_ids and not missing_ids:
                status_container.success(f"✅ `{target_table}` already matches MySQL source")
            else:
                status_container.success(
                    f"✅ Reconciled `{target_table}`: deleted {deleted_count:,} stale rows, "
                    f"restored {inserted_missing:,} missing rows"
                )

        return {
            'deleted_rows': deleted_count,
            'fetched_rows': fetched_missing,
            'inserted_rows': inserted_missing,
        }

    except Exception as e:
        pg_conn.rollback()
        if status_container:
            status_container.warning(f"⚠️ Reconciliation failed for `{target_table}`: {str(e)}")
        return {'deleted_rows': 0, 'fetched_rows': 0, 'inserted_rows': 0}


def delete_and_refetch_tables_from_date(
        mysql_config: dict,
        pg_config: dict,
        from_date_str: str,
        tables: list,
        batch_size: int = 20000,
        status_container=None) -> list:
    """
    For each table: delete rows >= from_date_str from PostgreSQL WH, then re-fetch from MySQL and re-insert.
    - ALERTS  : uses a_ENTRYTIME (datetime) for MySQL filter; a_entrytime in PG
    - ERPDATA : invdate is VARCHAR YYYYMMDD in MySQL/PG; cast to date when filtering
    - INVOICES / invoices_manager : invdate is VARCHAR YYYYMMDD in MySQL; cast ::date in PG
    """
    import pymysql

    TABLE_DATE_CFG = {
        'alerts':           {'pg_col': 'a_entrytime', 'mysql_col': 'a_ENTRYTIME', 'is_varchar': False},
        'erpdata':          {'pg_col': 'invdate',     'mysql_col': 'invdate',     'is_varchar': True},
        'invoices':         {'pg_col': 'invdate',     'mysql_col': 'invdate',     'is_varchar': True},
        'invoices_manager': {'pg_col': 'invdate',     'mysql_col': 'invdate',     'is_varchar': True},
    }

    from_dt_iso     = from_date_str                  # '2026-04-01'
    from_dt_compact = from_date_str.replace('-', '')  # '20260401'

    results = []
    mysql_conn = None
    pg_conn = None
    try:
        mysql_conn = pymysql.connect(
            host=mysql_config['host'], user=mysql_config['user'],
            password=mysql_config['password'], database=mysql_config['database'],
            charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=20, read_timeout=300,
        )
        pg_conn = psycopg2.connect(**pg_config)

        for tbl in tables:
            tbl_lower = tbl.lower()
            cfg = TABLE_DATE_CFG.get(tbl_lower)
            if not cfg:
                results.append({'table': tbl_lower, 'deleted': 0, 'fetched': 0,
                                 'inserted': 0, 'status': 'skipped', 'error': 'Unknown table'})
                continue

            pg_col    = cfg['pg_col']
            mysql_col = cfg['mysql_col']

            # ── Step 1: Delete from PostgreSQL ──
            try:
                with pg_conn.cursor() as cur:
                    cur.execute(
                        f'DELETE FROM {_quote_ident(tbl_lower)} WHERE {_quote_ident(pg_col)}::date >= %s',
                        (from_dt_iso,)
                    )
                    deleted = cur.rowcount or 0
                    pg_conn.commit()
            except Exception as de:
                pg_conn.rollback()
                results.append({'table': tbl_lower, 'deleted': 0, 'fetched': 0,
                                 'inserted': 0, 'status': 'error', 'error': f'Delete failed: {de}'})
                if status_container:
                    status_container.error(f"❌ `{tbl_lower}`: delete error — {de}")
                continue

            if status_container:
                status_container.info(f"🗑 `{tbl_lower}`: deleted {deleted:,} rows from {from_dt_iso} onwards.")

            # ── Step 2: Read column list from MySQL ──
            mysql_cur = mysql_conn.cursor()
            mysql_cur.execute(
                "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s ORDER BY ORDINAL_POSITION",
                (mysql_config['database'], tbl)
            )
            col_meta = mysql_cur.fetchall()
            if not col_meta:
                results.append({'table': tbl_lower, 'deleted': deleted, 'fetched': 0,
                                 'inserted': 0, 'status': 'error',
                                 'error': f'MySQL table {tbl} not found'})
                if status_container:
                    status_container.error(
                        f"❌ MySQL table `{tbl}` not found in `{mysql_config['database']}`")
                continue

            columns = [r['COLUMN_NAME'] for r in col_meta]
            pg_cols = [c.lower() for c in columns]

            # ── Step 3: Detect PG primary key for conflict handling ──
            try:
                with pg_conn.cursor() as cur:
                    cur.execute("""
                        SELECT a.attname
                        FROM pg_index i
                        JOIN pg_attribute a ON a.attrelid = i.indrelid
                                           AND a.attnum = ANY(i.indkey)
                        WHERE i.indrelid = %s::regclass AND i.indisprimary
                    """, (tbl_lower,))
                    pk_cols = [r[0] for r in cur.fetchall()]
            except Exception:
                pk_cols = []

            conflict_sql = (
                "ON CONFLICT ({}) DO NOTHING".format(
                    ', '.join([_quote_ident(c) for c in pk_cols])
                ) if pk_cols else ""
            )

            invoices_triggers_disabled = False
            if tbl_lower == 'invoices':
                try:
                    with pg_conn.cursor() as cur:
                        cur.execute("ALTER TABLE public.invoices DISABLE TRIGGER USER")
                    pg_conn.commit()
                    invoices_triggers_disabled = True
                    if status_container:
                        status_container.info("⏸️ `invoices`: user triggers disabled for refetch load")
                except Exception as trigger_err:
                    pg_conn.rollback()
                    if status_container:
                        status_container.warning(
                            f"⚠️ `invoices`: could not disable user triggers ({trigger_err}). Continuing load."
                        )

            # ── Step 4: Fetch from MySQL + insert into PostgreSQL ──
            select_cols = ', '.join([f'`{c}`' for c in columns])
            if cfg['is_varchar']:
                where_sql   = f"WHERE `{mysql_col}` >= %s"
                mysql_param = from_dt_compact
            else:
                where_sql   = f"WHERE DATE(`{mysql_col}`) >= %s"
                mysql_param = from_dt_iso

            insert_col_sql = ', '.join([_quote_ident(c) for c in pg_cols])
            total_fetched = total_inserted = 0
            batch_counter = 0

            # ── Adaptive batch sizing: estimate rows to refetch ──
            try:
                mysql_cur.execute(
                    "SELECT TABLE_ROWS FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s",
                    (mysql_config['database'], tbl)
                )
                src_est = mysql_cur.fetchone()
                src_row_est = src_est['TABLE_ROWS'] if src_est else 0
            except Exception:
                src_row_est = 0
            effective_batch = _calculate_adaptive_batch_size(tbl, src_row_est, batch_size)
            if status_container and effective_batch != batch_size:
                status_container.info(
                    f"📦 `{tbl_lower}`: adapted batch size {batch_size:,} → {effective_batch:,} "
                    f"(est. {src_row_est:,} rows in table)"
                )

            ins_sql = (
                f"INSERT INTO {_quote_ident(tbl_lower)} ({insert_col_sql}) "
                f"VALUES %s {conflict_sql}"
            )

            # For invoices-like varchar date tables, use ID keyset pagination (same fast pattern as temp backfill).
            id_col = next((c for c in columns if c.lower() == 'id'), None)
            if cfg['is_varchar'] and id_col:
                if status_container:
                    status_container.info(f"⚡ `{tbl_lower}`: using keyset pagination on `{id_col}`")

                last_id = 0
                while True:
                    keyset_sql = (
                        f"SELECT {select_cols} FROM `{tbl}` "
                        f"WHERE `{mysql_col}` >= %s AND `{id_col}` > %s "
                        f"ORDER BY `{id_col}` ASC LIMIT {effective_batch}"
                    )
                    try:
                        mysql_cur.execute(keyset_sql, [mysql_param, last_id])
                        rows = mysql_cur.fetchall() or []
                    except Exception as fe:
                        results.append({'table': tbl_lower, 'deleted': deleted,
                                        'fetched': total_fetched, 'inserted': total_inserted,
                                        'status': 'error', 'error': f'Fetch failed: {fe}'})
                        if status_container:
                            status_container.error(f"❌ `{tbl_lower}` fetch error: {fe}")
                        rows = []
                        break

                    if not rows:
                        break

                    value_rows = [
                        tuple([{k.lower(): v for k, v in r.items()}.get(c) for c in pg_cols])
                        for r in rows
                    ]
                    try:
                        with pg_conn.cursor() as cur:
                            execute_values(cur, ins_sql, value_rows, page_size=min(effective_batch, 5000))
                            pg_conn.commit()
                            total_inserted += max(0, cur.rowcount or 0)
                    except Exception as ie:
                        pg_conn.rollback()
                        if status_container:
                            status_container.warning(
                                f"⚠️ `{tbl_lower}` insert error: {ie}")
                        break

                    total_fetched += len(rows)
                    batch_counter += 1
                    try:
                        last_id = int(rows[-1].get(id_col) or last_id)
                    except Exception:
                        pass

                    if batch_counter % 10 == 0:
                        try:
                            mysql_conn.ping(reconnect=True)
                        except Exception:
                            pass

                    if status_container and batch_counter % 5 == 0:
                        status_container.info(
                            f"⏳ `{tbl_lower}`: {total_fetched:,} fetched / {total_inserted:,} inserted…")

            else:
                sql = (
                    f"SELECT {select_cols} FROM `{tbl}` {where_sql} "
                    f"ORDER BY `{mysql_col}` ASC"
                )
                data_cur = mysql_conn.cursor(pymysql.cursors.SSDictCursor)
                try:
                    data_cur.execute(sql, [mysql_param])
                except Exception as fe:
                    try:
                        data_cur.close()
                    except Exception:
                        pass
                    results.append({'table': tbl_lower, 'deleted': deleted,
                                    'fetched': total_fetched, 'inserted': total_inserted,
                                    'status': 'error', 'error': f'Fetch failed: {fe}'})
                    if status_container:
                        status_container.error(f"❌ `{tbl_lower}` fetch error: {fe}")
                    if invoices_triggers_disabled:
                        try:
                            with pg_conn.cursor() as cur:
                                cur.execute("ALTER TABLE public.invoices ENABLE " \
                                "")
                            pg_conn.commit()
                            invoices_triggers_disabled = False
                            if status_container:
                                status_container.info("▶️ `invoices`: user triggers re-enabled after refetch load")
                        except Exception as trigger_err:
                            pg_conn.rollback()
                            if status_container:
                                status_container.error(
                                    f"❌ Failed to re-enable invoices triggers: {trigger_err}"
                                )
                    continue

                while True:
                    rows = data_cur.fetchmany(effective_batch)

                    if not rows:
                        break

                    value_rows = [
                        tuple([{k.lower(): v for k, v in r.items()}.get(c) for c in pg_cols])
                        for r in rows
                    ]
                    try:
                        with pg_conn.cursor() as cur:
                            execute_values(cur, ins_sql, value_rows, page_size=min(effective_batch, 5000))
                            pg_conn.commit()
                            total_inserted += max(0, cur.rowcount or 0)
                    except Exception as ie:
                        pg_conn.rollback()
                        if status_container:
                            status_container.warning(
                                f"⚠️ `{tbl_lower}` insert error: {ie}")
                        break

                    total_fetched += len(rows)
                    batch_counter += 1

                    if batch_counter % 10 == 0:
                        try:
                            mysql_conn.ping(reconnect=True)
                        except Exception:
                            pass

                    if status_container and batch_counter % 5 == 0:
                        status_container.info(
                            f"⏳ `{tbl_lower}`: {total_fetched:,} fetched / {total_inserted:,} inserted…")

                try:
                    data_cur.close()
                except Exception:
                    pass

            if invoices_triggers_disabled:
                try:
                    with pg_conn.cursor() as cur:
                        cur.execute("ALTER TABLE public.invoices ENABLE TRIGGER USER")
                    pg_conn.commit()
                    if status_container:
                        status_container.info("▶️ `invoices`: user triggers re-enabled after refetch load")
                except Exception as trigger_err:
                    pg_conn.rollback()
                    results.append({'table': tbl_lower, 'deleted': deleted,
                                    'fetched': total_fetched, 'inserted': total_inserted,
                                    'status': 'error', 'error': f'Failed to re-enable triggers: {trigger_err}'})
                    if status_container:
                        status_container.error(f"❌ Failed to re-enable invoices triggers: {trigger_err}")
                    continue

            results.append({
                'table': tbl_lower, 'deleted': deleted,
                'fetched': total_fetched, 'inserted': total_inserted,
                'status': 'success', 'error': None,
            })
            if status_container:
                status_container.success(
                    f"✅ `{tbl_lower}`: deleted {deleted:,} · fetched {total_fetched:,} · "
                    f"inserted {total_inserted:,}")

    finally:
        if mysql_conn:
            try:
                mysql_conn.close()
            except Exception:
                pass
        if pg_conn:
            try:
                pg_conn.close()
            except Exception:
                pass

    return results


def normalize_invoices_duplicate_flags(pg_config: dict, status_container=None) -> int:
        """
        Recompute invoices.duplicate flags from fullqrcode rules:
        - first row per fullqrcode by min(id) => duplicate = 1
        - remaining rows in same fullqrcode => duplicate = 0
        - NULL/empty fullqrcode => duplicate = 1
        Returns number of rows updated.
        """
        sql = """
        WITH ranked AS (
            SELECT
                id,
                CASE
                    WHEN fullqrcode IS NULL OR TRIM(fullqrcode) = '' THEN 1
                    WHEN ROW_NUMBER() OVER (PARTITION BY fullqrcode ORDER BY id) = 1 THEN 1
                    ELSE 0
                END AS correct_duplicate
            FROM public.invoices
        )
        UPDATE public.invoices i
        SET duplicate = r.correct_duplicate
        FROM ranked r
        WHERE i.id = r.id
            AND i.duplicate IS DISTINCT FROM r.correct_duplicate;
        """

        with psycopg2.connect(**pg_config) as conn:
                with conn.cursor() as cur:
                        cur.execute(sql)
                        updated_rows = cur.rowcount if cur.rowcount is not None else 0
                conn.commit()

        if status_container:
                status_container.info(f"🧮 Recomputed invoices duplicate flags. Rows updated: {updated_rows:,}")
        return updated_rows


def _calculate_adaptive_batch_size(table_name: str, source_row_estimate: int, base_batch_size: int = 20000) -> int:
    """
    Calculate adaptive batch size to avoid MySQL timeout.
    Larger tables get smaller batches to prevent operations from exceeding net_read_timeout (30s).
    
    - < 50K rows: use base_batch_size (20K)
    - 50K-200K rows: 10K batches
    - 200K-500K rows: 5K batches
    - > 500K rows: 2K batches
    """
    if source_row_estimate < 50000:
        return base_batch_size
    elif source_row_estimate < 200000:
        return max(5000, base_batch_size // 2)
    elif source_row_estimate < 500000:
        return 5000
    else:
        return 2000


def _execute_mysql_query_with_retry(mysql_cur, sql: str, params: list = None, table_name: str = '', max_retries: int = 3):
    """
    Execute MySQL query with automatic retry on timeout errors.
    Implements exponential backoff: 2s, 4s, 8s between retries.
    """
    import time
    if params is None:
        params = []
    
    last_error = None
    for attempt in range(max_retries):
        try:
            mysql_cur.execute(sql, params)
            return mysql_cur.fetchall()
        except Exception as e:
            error_str = str(e).lower()
            # Retry on timeout / lost connection errors
            if any(x in error_str for x in ['timeout', 'lost connection', 'timed out', '2013', '2006']):
                last_error = e
                if attempt < max_retries - 1:
                    wait_secs = 2 ** (attempt + 1)
                    print(f"⚠️ MySQL {table_name} query timeout on attempt {attempt+1}. Retrying in {wait_secs}s...")
                    time.sleep(wait_secs)
                    continue
            # Not a timeout/connection error, raise immediately
            raise e
    
    # All retries exhausted
    raise RuntimeError(f"MySQL query failed for {table_name} after {max_retries} attempts: {last_error}")


def _sync_erpdata_from_csv(pg_config: dict, status_container=None) -> dict:
    """
    Sync ERPDATA from daily shopbillcount CSV files at \\\\10.10.0.30\\mis\\shopbillcount_YYYYMMDD.csv.

    Logic:
    1. Check MAX(invdate) in PostgreSQL erpdata (format YYYYMMDD varchar).
    2. Build a list of dates from (max_date + 1 day) up to and including yesterday.
    3. For each date try to open \\\\10.10.0.30\\mis\\shopbillcount_YYYYMMDD.csv.
    4. Normalize column names to lowercase, compute a row hash for dedupe, and
       INSERT … ON CONFLICT DO NOTHING into public.erpdata.
    5. Returns a dict compatible with the per-table summary expected by the caller.
    """
    import os as _os
    from datetime import date as _date, timedelta as _td

    CSV_BASE_PATH = r"\\10.10.0.30\mis"
    yesterday = _date.today() - _td(days=1)

    start_time = datetime.now()
    total_fetched = 0
    total_inserted = 0
    files_loaded: list = []
    files_missing: list = []

    pg_conn = psycopg2.connect(**pg_config)
    try:
        with pg_conn.cursor() as cur:
            # ── Ensure erpdata table exists (minimal schema) ──────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.erpdata (
                    invdate         TEXT,
                    invno           TEXT,
                    store_code      TEXT,
                    cashier         TEXT,
                    tillno          TEXT,
                    amt             NUMERIC(18,4),
                    _source_row_hash TEXT,
                    _synced_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_erpdata_source_hash
                ON public.erpdata (_source_row_hash)
            """)
            pg_conn.commit()

            # ── Drop NOT NULL constraints on all data columns so CSV nulls don't fail ──
            # (The table may have been created with stricter constraints previously)
            # Use SAVEPOINT per-column so a failure on one doesn't abort the whole transaction
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name   = 'erpdata'
                  AND is_nullable  = 'NO'
                  AND column_name NOT IN ('_source_row_hash', '_synced_at')
            """)
            not_null_cols = [r[0] for r in cur.fetchall()]
            for nn_col in not_null_cols:
                try:
                    cur.execute("SAVEPOINT drop_nn")
                    cur.execute(
                        f"ALTER TABLE public.erpdata ALTER COLUMN {_quote_ident(nn_col)} DROP NOT NULL"
                    )
                    cur.execute("RELEASE SAVEPOINT drop_nn")
                except Exception:
                    cur.execute("ROLLBACK TO SAVEPOINT drop_nn")
            if not_null_cols:
                pg_conn.commit()

            # ── Get max date already in PostgreSQL ─────────────────────────────
            cur.execute("""
                SELECT MAX(invdate)
                FROM public.erpdata
                WHERE invdate ~ '^[0-9]{8}$'
            """)
            pg_max = cur.fetchone()[0]  # e.g. '20260416' or None

        if pg_max:
            max_dt = datetime.strptime(str(pg_max)[:8], '%Y%m%d').date()
            start_date = max_dt + _td(days=1)
        else:
            start_date = yesterday  # full load if table is empty

        if start_date > yesterday:
            if status_container:
                status_container.info(
                    f"✅ ERPDATA already up to date (PostgreSQL max invdate: {pg_max})"
                )
            return {
                'source_table': 'ERPDATA', 'target_table': 'erpdata',
                'strategy': 'csv_file_append',
                'fetched_rows': 0, 'inserted_rows': 0, 'duplicate_rows': 0,
                'deleted_rows': 0, 'source_total_rows': 0, 'target_total_rows': 0,
                'sync_note': f'Already up to date (max date: {pg_max})',
                'status': 'success', 'error': None,
                'started_at': start_time, 'finished_at': datetime.now(),
            }

        dates_to_load = []
        d = start_date
        while d <= yesterday:
            dates_to_load.append(d)
            d += _td(days=1)

        if status_container:
            status_container.info(
                f"📅 ERPDATA: will attempt {len(dates_to_load)} CSV file(s) "
                f"({start_date} → {yesterday})"
            )

        # ── Detect actual PG columns (beyond the minimal schema) ─────────────
        with pg_conn.cursor() as cur:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'erpdata'
                  AND column_name NOT IN ('_source_row_hash', '_synced_at')
                ORDER BY ordinal_position
            """)
            pg_data_cols = [r[0] for r in cur.fetchall()]

        for target_date in dates_to_load:
            date_str = target_date.strftime('%Y%m%d')
            csv_path = _os.path.join(CSV_BASE_PATH, f"shopbillcount_{date_str}.csv")

            if not _os.path.exists(csv_path):
                files_missing.append(f"shopbillcount_{date_str}.csv")
                if status_container:
                    status_container.warning(
                        f"⚠️ shopbillcount_{date_str}.csv not found at {CSV_BASE_PATH} — skipping"
                    )
                continue

            try:
                try:
                    df = pd.read_csv(csv_path, dtype=str, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(csv_path, dtype=str, encoding='latin1')

                # Normalize CSV headers to lowercase, strip whitespace
                df.columns = [c.strip().lower() for c in df.columns]

                # ── Ensure erpdata table has all CSV columns ──────────────────
                with pg_conn.cursor() as cur:
                    for col in df.columns:
                        if col not in pg_data_cols and col not in ('_source_row_hash', '_synced_at'):
                            try:
                                cur.execute(
                                    f"ALTER TABLE public.erpdata "
                                    f"ADD COLUMN IF NOT EXISTS {_quote_ident(col)} TEXT"
                                )
                                pg_conn.commit()
                                pg_data_cols.append(col)
                            except Exception:
                                pg_conn.rollback()

                # Columns present in both CSV and PG (excluding meta cols)
                meta_cols = {'_source_row_hash', '_synced_at'}
                use_cols = [c for c in df.columns if c in pg_data_cols and c not in meta_cols]

                if not use_cols:
                    if status_container:
                        status_container.warning(
                            f"⚠️ shopbillcount_{date_str}.csv: no matching columns found — skipping"
                        )
                    continue

                df_insert = df[use_cols].copy()
                total_fetched += len(df_insert)

                insert_col_sql = ', '.join([_quote_ident(c) for c in use_cols] + ['_source_row_hash'])
                insert_sql = (
                    f"INSERT INTO public.erpdata ({insert_col_sql}) VALUES %s "
                    f"ON CONFLICT (_source_row_hash) DO NOTHING"
                )

                value_rows = []
                for _, row in df_insert.iterrows():
                    normalized = {c: (None if pd.isnull(row[c]) else str(row[c]).strip() or None) for c in use_cols}
                    # Skip rows where every data column is null/empty (blank CSV rows)
                    if all(v is None for v in normalized.values()):
                        continue
                    row_hash = hashlib.md5(
                        _serialize_row_for_hash(normalized, use_cols).encode('utf-8', errors='ignore')
                    ).hexdigest()
                    value_rows.append(tuple([normalized.get(c) for c in use_cols] + [row_hash]))

                with pg_conn.cursor() as cur:
                    execute_values(cur, insert_sql, value_rows, page_size=5000)
                    inserted = cur.rowcount if cur.rowcount is not None else 0
                    total_inserted += inserted

                pg_conn.commit()
                files_loaded.append(f"shopbillcount_{date_str}.csv ({len(df_insert):,} rows, {inserted:,} inserted)")

                if status_container:
                    status_container.info(
                        f"✅ shopbillcount_{date_str}.csv — "
                        f"fetched {len(df_insert):,} / inserted {inserted:,}"
                    )

            except Exception as file_err:
                pg_conn.rollback()
                files_missing.append(f"shopbillcount_{date_str}.csv (error: {file_err})")
                if status_container:
                    status_container.error(
                        f"❌ Error loading shopbillcount_{date_str}.csv: {file_err}"
                    )

        # ── Final row count ───────────────────────────────────────────────────
        with pg_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM public.erpdata")
            target_total = cur.fetchone()[0]

        sync_note = (
            f"Loaded {len(files_loaded)} file(s) from {CSV_BASE_PATH}. "
            f"Skipped/missing: {len(files_missing)}. "
            f"Target now {target_total:,} rows."
        )
        if status_container:
            status_container.success(
                f"✅ ERPDATA CSV sync done — "
                f"fetched {total_fetched:,} / inserted {total_inserted:,} / "
                f"skipped {len(files_missing)} file(s)"
            )

        return {
            'source_table': 'ERPDATA', 'target_table': 'erpdata',
            'strategy': 'csv_file_append',
            'fetched_rows': total_fetched, 'inserted_rows': total_inserted,
            'duplicate_rows': max(total_fetched - total_inserted, 0),
            'deleted_rows': 0, 'source_total_rows': total_fetched,
            'target_total_rows': target_total,
            'sync_note': sync_note,
            'status': 'success', 'error': None,
            'started_at': start_time, 'finished_at': datetime.now(),
        }

    except Exception as e:
        pg_conn.rollback()
        if status_container:
            status_container.error(f"❌ ERPDATA CSV sync failed: {e}")
        return {
            'source_table': 'ERPDATA', 'target_table': 'erpdata',
            'strategy': 'csv_file_append',
            'fetched_rows': total_fetched, 'inserted_rows': total_inserted,
            'duplicate_rows': 0, 'deleted_rows': 0,
            'source_total_rows': 0, 'target_total_rows': 0,
            'sync_note': str(e), 'status': 'failed', 'error': str(e),
            'started_at': start_time, 'finished_at': datetime.now(),
        }
    finally:
        pg_conn.close()


def sync_mysql_tables_to_postgres(mysql_config: dict, pg_config: dict, source_tables: list, status_container=None, batch_size: int = 20000):
    """
    Sync selected MySQL tables into PostgreSQL with:
    - auto-create destination table if missing
    - incremental strategy using numeric PK or datetime cursor when possible
    - hash-based dedupe fallback when no suitable cursor
    - persisted sync state + audit logs
    - ERPDATA is sourced from \\\\10.10.0.30\\mis\\shopbillcount_YYYYMMDD.csv (not MySQL)
    """
    try:
        try:
            import pymysql
        except ImportError:
            # Attempt automatic installation
            import subprocess
            import sys
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pymysql", "--quiet"])
                import pymysql
            except Exception as install_error:
                raise ImportError(
                    f"pymysql installation failed: {install_error}\n"
                    "Manual install: pip install pymysql\n"
                    "Or run: python -m pip install pymysql --upgrade"
                )
        
        # Fast-fail network check (no MySQL auth attempt)
        port_ok, port_msg = _check_port_accessible(mysql_config['host'], 3306, timeout=5)
        if not port_ok:
            raise RuntimeError(
                f"Cannot reach MySQL host '{mysql_config['host']}' on port 3306. {port_msg}"
            )
        
        mysql_conn = pymysql.connect(
            host=mysql_config['host'],
            user=mysql_config['user'],
            password=mysql_config['password'],
            database=mysql_config['database'],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=20,
            read_timeout=120
        )
    except Exception as e:
        error_str = str(e)
        if "Host" in error_str and "blocked" in error_str:
            raise RuntimeError(
                f"🔒 MySQL host is blocked. Recovery: \n"
                f"  1. MySQL admin runs: mysqladmin -u root -p flush-hosts\n"
                f"  2. Or restart MySQL server\n"
                f"  3. Wait 30s, then retry sync\n\n"
                f"Original error: {e}"
            )
        raise RuntimeError(f"MySQL connection failed: {e}")

    pg_conn = psycopg2.connect(**pg_config)
    _ensure_sync_metadata_tables(pg_conn)

    mysql_cur = mysql_conn.cursor()
    pg_cur = pg_conn.cursor()

    # HARD SAFETY: source MySQL/phpMyAdmin connection is read-only
    # We only fetch metadata/data from source and never write to it.
    try:
        mysql_cur.execute("SET SESSION TRANSACTION READ ONLY")
    except Exception:
        # Some MySQL setups may not allow this statement; logic below still uses SELECT-only queries.
        pass

    summary = []
    source_system = f"mysql://{mysql_config['host']}/{mysql_config['database']}"

    try:
        for source_table in source_tables:
            start_time = datetime.now()
            invoices_triggers_disabled = False
            table_status = {
                'source_table': source_table,
                'target_table': source_table.lower(),
                'strategy': None,
                'fetched_rows': 0,
                'inserted_rows': 0,
                'duplicate_rows': 0,
                'deleted_rows': 0,
                'source_total_rows': 0,
                'target_total_rows': 0,
                'sync_note': None,
                'status': 'success',
                'error': None,
                'started_at': start_time,
                'finished_at': None
            }

            try:
                # ── ERPDATA: read from daily CSV file, not MySQL ──────────────
                if source_table.upper() == 'ERPDATA':
                    csv_result = _sync_erpdata_from_csv(
                        pg_config=pg_config,
                        status_container=status_container,
                    )
                    # Merge CSV result into table_status and append to summary.
                    table_status.update(csv_result)
                    table_status['finished_at'] = datetime.now()
                    summary.append(table_status)
                    continue  # skip MySQL path entirely for ERPDATA

                target_table = source_table.lower()
                if status_container:
                    status_container.info(f"🔎 Inspecting source table `{source_table}`...")

                mysql_cur.execute("""
                    SELECT COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                    ORDER BY ORDINAL_POSITION
                """, (mysql_config['database'], source_table))
                col_meta = mysql_cur.fetchall()

                if not col_meta:
                    raise RuntimeError(f"Source table `{source_table}` not found in MySQL database `{mysql_config['database']}`")

                columns = [c['COLUMN_NAME'] for c in col_meta]
                pk_cols = [c['COLUMN_NAME'] for c in col_meta if c['COLUMN_KEY'] == 'PRI']

                pg_col_defs = []
                for c in col_meta:
                    col_name = c['COLUMN_NAME'].lower()
                    pg_type = _mysql_to_postgres_type(c['DATA_TYPE'], c['COLUMN_TYPE'])
                    nullable = '' if c['IS_NULLABLE'] == 'YES' else ' NOT NULL'
                    pg_col_defs.append(f"{_quote_ident(col_name)} {pg_type}{nullable}")

                pg_col_defs.append("_source_row_hash TEXT")
                pg_col_defs.append("_synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP")

                create_sql = f"CREATE TABLE IF NOT EXISTS {_quote_ident(target_table)} ({', '.join(pg_col_defs)})"
                pg_cur.execute(create_sql)

                if pk_cols:
                    uq_name = f"uq_{target_table}_{'_'.join([c.lower() for c in pk_cols])}"
                    key_list = ', '.join([_quote_ident(c.lower()) for c in pk_cols])
                    pg_cur.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {_quote_ident(uq_name)} ON {_quote_ident(target_table)} ({key_list})")
                else:
                    # Hash index is only needed when table has no PK conflict target.
                    pg_cur.execute(
                        f"CREATE UNIQUE INDEX IF NOT EXISTS {_quote_ident(f'uq_{target_table}_source_hash')} "
                        f"ON {_quote_ident(target_table)} (_source_row_hash)"
                    )
                pg_conn.commit()

                table_status['deleted_rows'] = 0

                datetime_candidates = [
                    'invdate', 'alert_date', 'entry_time', 'dt_mod_date',
                    'a_entrytime', 'scanned_date',
                    'created_at', 'updated_at', 'modified_at', 'date_invoice',
                    'bill_date', 'dt_invoice_date', 'sync_time'
                ]
                lower_map = {c['COLUMN_NAME'].lower(): c for c in col_meta}
                date_column = None
                date_column_is_varchar = False
                for candidate in datetime_candidates:
                    if candidate not in lower_map:
                        continue
                    data_type = lower_map[candidate]['DATA_TYPE'].lower()
                    if data_type in ('datetime', 'timestamp', 'date'):
                        date_column = lower_map[candidate]['COLUMN_NAME']
                        break
                    # INVOICES/invoices_manager use YYYYMMDD varchar in source.
                    # Treat it as a lexical date cursor to keep incremental sync fast and correct.
                    if candidate == 'invdate' and data_type in ('varchar', 'char', 'text'):
                        date_column = lower_map[candidate]['COLUMN_NAME']
                        date_column_is_varchar = True
                        break

                if not date_column:
                    raise RuntimeError(
                        f"No date/datetime cursor column found for `{source_table}`. "
                        "Expected one of invdate/alert_date/entry_time/created_at/updated_at."
                    )

                strategy = 'date_append'
                table_status['strategy'] = strategy

                if date_column_is_varchar:
                    # Fast MAX on compact YYYYMMDD values avoids expensive DATE() casts.
                    pg_cur.execute(
                        f"SELECT MAX({_quote_ident(date_column.lower())}) "
                        f"FROM {_quote_ident(target_table)} "
                        f"WHERE {_quote_ident(date_column.lower())} ~ '^[0-9]{{8}}$'"
                    )
                else:
                    pg_cur.execute(
                        f"SELECT MAX(DATE({_quote_ident(date_column.lower())})) FROM {_quote_ident(target_table)}"
                    )
                pg_max_date = pg_cur.fetchone()[0]

                if status_container:
                    if pg_max_date:
                        status_container.info(
                            f"📥 Syncing `{source_table}` → `{target_table}` using `{strategy}` "
                            f"using PostgreSQL target max date {pg_max_date}; fetching MySQL rows after this date"
                        )
                    else:
                        status_container.info(
                            f"📥 Syncing `{source_table}` → `{target_table}` using `{strategy}` (initial full load)"
                        )

                total_inserted = 0
                total_fetched = 0
                total_dupes = 0
                next_cursor_value = str(pg_max_date) if pg_max_date is not None else None
                base_cursor_value = next_cursor_value
                batch_counter = 0

                # ── Adaptive batch sizing for timeout resilience ──
                # Estimate source row count for this table
                mysql_cur.execute(
                    "SELECT TABLE_ROWS FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s",
                    (mysql_config['database'], source_table)
                )
                src_row = mysql_cur.fetchone()
                src_row_count = src_row['TABLE_ROWS'] if src_row else 0
                effective_batch_size = _calculate_adaptive_batch_size(source_table, src_row_count, batch_size)
                if status_container and effective_batch_size != batch_size:
                    status_container.info(f"📦 Adapted batch size: {batch_size:,} → {effective_batch_size:,} for {source_table} ({src_row_count:,} rows)")

                select_cols_mysql = ', '.join([f"`{c}`" for c in columns])
                params = []
                where_clause = ''
                if base_cursor_value is not None:
                    if date_column_is_varchar:
                        where_clause = f"WHERE `{date_column}` > %s"
                        params.append(str(base_cursor_value).replace('-', '')[:8])
                    else:
                        where_clause = f"WHERE DATE(`{date_column}`) > %s"
                        params.append(base_cursor_value)

                sql = (
                    f"SELECT {select_cols_mysql} "
                    f"FROM `{source_table}` {where_clause} "
                    f"ORDER BY `{date_column}` ASC"
                )

                pg_cols = [c.lower() for c in columns]

                # Speed optimization:
                # - If PK exists, rely on ON CONFLICT(pk) and skip expensive row-hash computation.
                # - If PK does not exist, keep hash-based dedupe.
                use_hash_dedupe = not bool(pk_cols)
                insert_cols = pg_cols + (['_source_row_hash'] if use_hash_dedupe else [])
                insert_col_sql = ', '.join([_quote_ident(c) for c in insert_cols])

                conflict_target = ''
                if pk_cols:
                    pk_conflict = ', '.join([_quote_ident(c.lower()) for c in pk_cols])
                    conflict_target = f"ON CONFLICT ({pk_conflict}) DO NOTHING"
                else:
                    conflict_target = "ON CONFLICT (_source_row_hash) DO NOTHING"

                insert_sql = f"INSERT INTO {_quote_ident(target_table)} ({insert_col_sql}) VALUES %s {conflict_target}"

                # Use same incremental batched process for all tables (including invoices/invoices_manager).
                if status_container:
                    status_container.info(f"⚡ `{source_table}`: using incremental batch sync")

                if target_table == 'invoices':
                    try:
                        pg_cur.execute("ALTER TABLE public.invoices ENABLE TRIGGER USER")
                        pg_conn.commit()
                        invoices_triggers_disabled = True
                        if status_container:
                            status_container.info("▶️ `invoices`: user triggers ENABLED before bulk sync")
                    except Exception as trigger_err:
                        pg_conn.rollback()
                        if status_container:
                            status_container.warning(
                                f"⚠️ `invoices`: could not enable user triggers ({trigger_err}). Continuing sync."
                            )

                # INVOICES/invoices_manager can stall before first fetch when MySQL scans by invdate.
                # Use ID keyset pagination to force fast, incremental reads like the backfill utility.
                id_col = next((c for c in columns if c.lower() == 'id'), None)
                use_keyset = bool(date_column_is_varchar and id_col)
                first_batch_started_at = datetime.now()

                if use_keyset:
                    # Seed last_id from PostgreSQL MAX(id) so MySQL starts scanning from where
                    # we left off (near the end of the table) instead of id=0 (full table scan).
                    pg_cur.execute(
                        f"SELECT COALESCE(MAX({_quote_ident(id_col.lower())}), 0) "
                        f"FROM {_quote_ident(target_table)}"
                    )
                    last_id = int(pg_cur.fetchone()[0])

                    if status_container:
                        status_container.info(
                            f"🚀 `{source_table}`: keyset pagination on `{id_col}` — "
                            f"starting from id={last_id:,} (PostgreSQL MAX)"
                        )

                    while True:
                        if base_cursor_value is not None:
                            keyset_sql = (
                                f"SELECT {select_cols_mysql} "
                                f"FROM `{source_table}` "
                                f"WHERE `{date_column}` > %s AND `{id_col}` > %s "
                                f"ORDER BY `{id_col}` ASC LIMIT {effective_batch_size}"
                            )
                            keyset_params = [str(base_cursor_value).replace('-', '')[:8], last_id]
                        else:
                            keyset_sql = (
                                f"SELECT {select_cols_mysql} "
                                f"FROM `{source_table}` "
                                f"WHERE `{id_col}` > %s "
                                f"ORDER BY `{id_col}` ASC LIMIT {effective_batch_size}"
                            )
                            keyset_params = [last_id]

                        try:
                            mysql_cur.execute(keyset_sql, keyset_params)
                            rows = mysql_cur.fetchall() or []
                        except Exception as fetch_err:
                            raise RuntimeError(f"Failed keyset fetch for `{source_table}`: {fetch_err}")

                        if not rows:
                            break

                        if batch_counter == 0 and status_container:
                            elapsed = (datetime.now() - first_batch_started_at).total_seconds()
                            status_container.info(f"✅ `{source_table}` first batch arrived in {elapsed:.1f}s")

                        fetched = len(rows)
                        total_fetched += fetched

                        value_rows = []
                        for row in rows:
                            normalized = {k.lower(): v for k, v in row.items()}
                            if use_hash_dedupe:
                                row_hash = hashlib.md5(
                                    _serialize_row_for_hash(normalized, pg_cols).encode('utf-8', errors='ignore')
                                ).hexdigest()
                                value_rows.append(tuple([normalized.get(c) for c in pg_cols] + [row_hash]))
                            else:
                                value_rows.append(tuple([normalized.get(c) for c in pg_cols]))

                        execute_values(pg_cur, insert_sql, value_rows, page_size=min(effective_batch_size, 10000))
                        inserted = pg_cur.rowcount if pg_cur.rowcount is not None else 0

                        total_inserted += inserted
                        total_dupes += max(fetched - inserted, 0)
                        batch_counter += 1

                        try:
                            last_id = int(rows[-1].get(id_col) or last_id)
                        except Exception:
                            pass

                        batch_last = rows[-1].get(date_column)
                        if batch_last is not None:
                            next_cursor_value = str(batch_last).replace('-', '')[:8]

                        if batch_counter % 5 == 0:
                            pg_conn.commit()
                            if status_container:
                                status_container.info(
                                    f"⏳ `{source_table}`: {total_fetched:,} fetched / {total_inserted:,} inserted..."
                                )

                        if batch_counter % 10 == 0:
                            try:
                                mysql_conn.ping(reconnect=True)
                            except Exception:
                                pass
                else:
                    data_cur = mysql_conn.cursor(pymysql.cursors.SSDictCursor)
                    try:
                        data_cur.execute(sql, params)
                    except Exception as fetch_err:
                        try:
                            data_cur.close()
                        except Exception:
                            pass
                        raise RuntimeError(f"Failed to start fetch for `{source_table}`: {fetch_err}")

                    try:
                        while True:
                            rows = data_cur.fetchmany(effective_batch_size)
                            if not rows:
                                break

                            if batch_counter == 0 and status_container:
                                elapsed = (datetime.now() - first_batch_started_at).total_seconds()
                                status_container.info(f"✅ `{source_table}` first batch arrived in {elapsed:.1f}s")

                            fetched = len(rows)
                            total_fetched += fetched

                            value_rows = []
                            for row in rows:
                                normalized = {k.lower(): v for k, v in row.items()}
                                if use_hash_dedupe:
                                    row_hash = hashlib.md5(
                                        _serialize_row_for_hash(normalized, pg_cols).encode('utf-8', errors='ignore')
                                    ).hexdigest()
                                    value_rows.append(tuple([normalized.get(c) for c in pg_cols] + [row_hash]))
                                else:
                                    value_rows.append(tuple([normalized.get(c) for c in pg_cols]))

                            execute_values(pg_cur, insert_sql, value_rows, page_size=min(effective_batch_size, 10000))
                            inserted = pg_cur.rowcount if pg_cur.rowcount is not None else 0

                            total_inserted += inserted
                            total_dupes += max(fetched - inserted, 0)
                            batch_counter += 1

                            if batch_counter % 5 == 0:
                                pg_conn.commit()
                                if status_container:
                                    status_container.info(
                                        f"⏳ `{source_table}`: {total_fetched:,} fetched / {total_inserted:,} inserted..."
                                    )

                            if batch_counter % 10 == 0:
                                try:
                                    mysql_conn.ping(reconnect=True)
                                except Exception:
                                    pass

                            batch_last = rows[-1].get(date_column)
                            if batch_last is not None:
                                if date_column_is_varchar:
                                    next_cursor_value = str(batch_last).replace('-', '')[:8]
                                else:
                                    next_cursor_value = str(batch_last)[:10]
                    finally:
                        try:
                            data_cur.close()
                        except Exception:
                            pass

                # Commit remaining batched inserts.
                pg_conn.commit()

                # Keep invoices.invdate_ap in sync with requested business rule.
                if target_table == 'invoices':
                    pg_cur.execute("ALTER TABLE public.invoices ADD COLUMN IF NOT EXISTS invdate_ap date")
                    pg_cur.execute("""
                        UPDATE public.invoices
                        SET invdate_ap = CASE
                            WHEN NULLIF(TRIM(COALESCE(invdate::text, '')), '') IS NULL THEN NULL
                            ELSE CASE
                                WHEN to_date(REGEXP_REPLACE(invdate::text, '[^0-9]', '', 'g'), 'YYYYMMDD') > DATE '2026-04-15'
                                    THEN to_date(REGEXP_REPLACE(invdate::text, '[^0-9]', '', 'g'), 'YYYYMMDD') - 1
                                ELSE to_date(REGEXP_REPLACE(invdate::text, '[^0-9]', '', 'g'), 'YYYYMMDD')
                            END
                        END
                        WHERE _synced_at >= %s
                          AND invdate_ap IS DISTINCT FROM CASE
                            WHEN NULLIF(TRIM(COALESCE(invdate::text, '')), '') IS NULL THEN NULL
                            ELSE CASE
                                WHEN to_date(REGEXP_REPLACE(invdate::text, '[^0-9]', '', 'g'), 'YYYYMMDD') > DATE '2026-04-15'
                                    THEN to_date(REGEXP_REPLACE(invdate::text, '[^0-9]', '', 'g'), 'YYYYMMDD') - 1
                                ELSE to_date(REGEXP_REPLACE(invdate::text, '[^0-9]', '', 'g'), 'YYYYMMDD')
                            END
                        END
                    """, (start_time,))
                    pg_conn.commit()

                pg_cur.execute("""
                    INSERT INTO external_sync_state (source_system, source_table, strategy, cursor_column, cursor_value, last_synced_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (source_system, source_table)
                    DO UPDATE SET
                        strategy = EXCLUDED.strategy,
                        cursor_column = EXCLUDED.cursor_column,
                        cursor_value = EXCLUDED.cursor_value,
                        last_synced_at = CURRENT_TIMESTAMP
                """, (source_system, source_table, strategy, date_column, str(next_cursor_value) if next_cursor_value is not None else None))
                pg_conn.commit()

                mysql_cur.execute(
                    """
                    SELECT TABLE_ROWS
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                    """,
                    (mysql_config['database'], source_table)
                )
                source_total_row = mysql_cur.fetchone()
                source_total_rows = source_total_row['TABLE_ROWS'] if source_total_row else 0
                pg_cur.execute(f"SELECT COUNT(*) FROM {_quote_ident(target_table)}")
                target_total_rows = pg_cur.fetchone()[0]

                table_status['fetched_rows'] = total_fetched
                table_status['inserted_rows'] = total_inserted
                table_status['duplicate_rows'] = total_dupes
                table_status['source_total_rows'] = int(source_total_rows or 0)
                table_status['target_total_rows'] = int(target_total_rows or 0)

                if total_fetched == 0 and total_inserted == 0 and table_status['deleted_rows'] == 0:
                    if table_status['source_total_rows'] == table_status['target_total_rows']:
                        table_status['sync_note'] = (
                            f"No new rows. Source and target already match at "
                            f"{table_status['target_total_rows']:,} rows."
                        )
                    else:
                        table_status['sync_note'] = (
                            f"No new rows after latest synced date. Source has {table_status['source_total_rows']:,} rows; "
                            f"target has {table_status['target_total_rows']:,} rows."
                        )
                else:
                    table_status['sync_note'] = (
                        f"Source {table_status['source_total_rows']:,} rows, "
                        f"target now {table_status['target_total_rows']:,} rows."
                    )

                pg_cur.execute("""
                    INSERT INTO external_sync_log
                    (source_system, source_table, target_table, strategy, fetched_rows, inserted_rows, duplicate_rows, deleted_rows, status, started_at, finished_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (
                    source_system,
                    source_table,
                    target_table,
                    strategy,
                    total_fetched,
                    total_inserted,
                    total_dupes,
                    table_status.get('deleted_rows', 0),
                    'success',
                    start_time
                ))
                pg_conn.commit()

            except Exception as table_error:
                pg_conn.rollback()
                table_status['status'] = 'failed'
                table_status['error'] = str(table_error)
                try:
                    pg_cur.execute("""
                        INSERT INTO external_sync_log
                        (source_system, source_table, target_table, strategy, fetched_rows, inserted_rows, duplicate_rows, deleted_rows, status, error_message, started_at, finished_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """, (
                        source_system,
                        source_table,
                        table_status.get('target_table', source_table.lower()),
                        table_status.get('strategy') or 'unknown',
                        table_status.get('fetched_rows', 0),
                        table_status.get('inserted_rows', 0),
                        table_status.get('duplicate_rows', 0),
                        table_status.get('deleted_rows', 0),
                        'failed',
                        str(table_error),
                        start_time
                    ))
                    pg_conn.commit()
                except Exception:
                    pg_conn.rollback()

            finally:
                if invoices_triggers_disabled:
                    try:
                        pg_cur.execute("ALTER TABLE public.invoices DISABLE TRIGGER USER")
                        pg_conn.commit()
                        if status_container:
                            status_container.info("⏸️ `invoices`: user triggers DISABLED after sync")
                    except Exception as trigger_err:
                        pg_conn.rollback()
                        table_status['status'] = 'failed'
                        existing_error = table_status.get('error')
                        trigger_error_text = f"Failed to disable invoices triggers after sync: {trigger_err}"
                        table_status['error'] = (
                            f"{existing_error}; {trigger_error_text}" if existing_error else trigger_error_text
                        )
                        if status_container:
                            status_container.error(f"❌ {trigger_error_text}")

            table_status['finished_at'] = datetime.now()
            summary.append(table_status)

    finally:
        try:
            mysql_cur.close()
            mysql_conn.close()
        except Exception:
            pass
        try:
            pg_cur.close()
            pg_conn.close()
        except Exception:
            pass

    return summary


def verify_mysql_postgres_sync_by_day_shop(mysql_config: dict, pg_config: dict, source_tables: list, lookback_days: int = 7):
    """Post-sync verification: compare MySQL vs PostgreSQL row counts grouped by date and shop."""
    try:
        import pymysql
    except ImportError:
        raise RuntimeError("pymysql not installed. Install with: pip install pymysql")

    mysql_conn = pymysql.connect(
        host=mysql_config['host'],
        user=mysql_config['user'],
        password=mysql_config['password'],
        database=mysql_config['database'],
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=20,
        read_timeout=120
    )
    pg_conn = psycopg2.connect(**pg_config)

    mysql_cur = mysql_conn.cursor()
    pg_cur = pg_conn.cursor()

    date_candidates = [
        'invdate', 'alert_date', 'entry_time', 'dt_mod_date',
        'a_entrytime', 'scanned_date',
        'created_at', 'updated_at', 'modified_at', 'date_invoice',
        'bill_date', 'dt_invoice_date', 'sync_time'
    ]
    shop_candidates = ['store_code', 'a_store_code', 'shop_code', 'shop', 'shopcode']

    summary_rows = []
    detail_rows = []

    try:
        for source_table in source_tables:
            target_table = source_table.lower()

            mysql_cur.execute(
                """
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                ORDER BY ORDINAL_POSITION
                """,
                (mysql_config['database'], source_table)
            )
            mysql_cols = mysql_cur.fetchall() or []
            mysql_lower_map = {str(c['COLUMN_NAME']).lower(): c for c in mysql_cols}

            date_column = None
            for candidate in date_candidates:
                if candidate in mysql_lower_map and str(mysql_lower_map[candidate]['DATA_TYPE']).lower() in ('datetime', 'timestamp', 'date'):
                    date_column = mysql_lower_map[candidate]['COLUMN_NAME']
                    break

            shop_column = None
            for candidate in shop_candidates:
                if candidate in mysql_lower_map:
                    shop_column = mysql_lower_map[candidate]['COLUMN_NAME']
                    break

            if not date_column:
                summary_rows.append({
                    'source_table': source_table,
                    'target_table': target_table,
                    'date_column': None,
                    'shop_column': shop_column,
                    'compared_groups': 0,
                    'mismatch_groups': 0,
                    'status': 'skipped',
                    'note': 'No date/datetime column found'
                })
                continue

            pg_cur.execute("SELECT to_regclass(%s)", (f'public.{target_table}',))
            if pg_cur.fetchone()[0] is None:
                summary_rows.append({
                    'source_table': source_table,
                    'target_table': target_table,
                    'date_column': date_column,
                    'shop_column': shop_column,
                    'compared_groups': 0,
                    'mismatch_groups': 0,
                    'status': 'skipped',
                    'note': 'Target table missing in PostgreSQL'
                })
                continue

            if shop_column:
                mysql_shop_expr = f"UPPER(TRIM(COALESCE(`{shop_column}`, '')))"
                pg_shop_expr = f"UPPER(TRIM(COALESCE({_quote_ident(shop_column.lower())}::text, '')))"
            else:
                mysql_shop_expr = "'ALL'"
                pg_shop_expr = "'ALL'"

            mysql_query = (
                f"SELECT DATE(`{date_column}`) AS d, {mysql_shop_expr} AS shop_code, COUNT(*) AS cnt "
                f"FROM `{source_table}` "
                f"WHERE DATE(`{date_column}`) >= DATE_SUB(CURDATE(), INTERVAL %s DAY) "
                f"GROUP BY DATE(`{date_column}`), {mysql_shop_expr}"
            )
            mysql_cur.execute(mysql_query, (lookback_days,))
            mysql_rows = mysql_cur.fetchall() or []

            pg_query = (
                f"SELECT DATE({_quote_ident(date_column.lower())}) AS d, {pg_shop_expr} AS shop_code, COUNT(*) AS cnt "
                f"FROM {_quote_ident(target_table)} "
                f"WHERE DATE({_quote_ident(date_column.lower())}) >= CURRENT_DATE - (%s::int) "
                f"GROUP BY DATE({_quote_ident(date_column.lower())}), {pg_shop_expr}"
            )
            pg_cur.execute(pg_query, (int(lookback_days),))
            pg_rows = pg_cur.fetchall() or []

            mysql_map = {}
            for r in mysql_rows:
                if isinstance(r, dict):
                    d_val = r.get('d')
                    shop_val = r.get('shop_code')
                    cnt_val = r.get('cnt')
                else:
                    d_val = r[0] if len(r) > 0 else None
                    shop_val = r[1] if len(r) > 1 else None
                    cnt_val = r[2] if len(r) > 2 else 0
                key = (str(d_val), str(shop_val or '').strip().upper())
                if key[1]:
                    mysql_map[key] = int(cnt_val or 0)

            pg_map = {}
            for r in pg_rows:
                key = (str(r[0]), str(r[1] or '').strip().upper())
                if key[1]:
                    pg_map[key] = int(r[2] or 0)

            all_keys = sorted(set(mysql_map.keys()) | set(pg_map.keys()))
            mismatches = 0
            for d, shop_code in all_keys:
                mysql_cnt = mysql_map.get((d, shop_code), 0)
                pg_cnt = pg_map.get((d, shop_code), 0)
                if mysql_cnt != pg_cnt:
                    mismatches += 1
                    detail_rows.append({
                        'source_table': source_table,
                        'target_table': target_table,
                        'date': d,
                        'shop_code': shop_code,
                        'mysql_count': mysql_cnt,
                        'postgres_count': pg_cnt,
                        'delta': mysql_cnt - pg_cnt
                    })

            summary_rows.append({
                'source_table': source_table,
                'target_table': target_table,
                'date_column': date_column,
                'shop_column': shop_column,
                'compared_groups': len(all_keys),
                'mismatch_groups': mismatches,
                'status': 'ok' if mismatches == 0 else 'mismatch',
                'note': f'Compared last {int(lookback_days)} days'
            })

    finally:
        try:
            mysql_cur.close()
            mysql_conn.close()
        except Exception:
            pass
        try:
            pg_cur.close()
            pg_conn.close()
        except Exception:
            pass

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(by=['status', 'mismatch_groups', 'source_table'], ascending=[True, False, True])
    if not detail_df.empty:
        detail_df = detail_df.sort_values(by=['delta', 'source_table', 'date', 'shop_code'], ascending=[False, True, True, True])

    return summary_df, detail_df


def _ensure_external_sync_delta_log_table(pg_conn):
    cur = pg_conn.cursor()
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS external_sync_delta_log (
                id BIGSERIAL PRIMARY KEY,
                logged_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                source_table TEXT NOT NULL,
                target_table TEXT NOT NULL,
                group_date DATE,
                shop_code TEXT,
                mysql_count BIGINT NOT NULL,
                postgres_count BIGINT NOT NULL,
                delta BIGINT NOT NULL
            )
            """
        )
        pg_conn.commit()
    finally:
        cur.close()


def build_offloading_positive_delta_preview(pg_config: dict, verify_detail_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare preview rows for Offloading vs Loading deltas > 0 and mark NEW vs ALREADY_EXISTS."""
    if verify_detail_df is None or verify_detail_df.empty:
        return pd.DataFrame()

    required_cols = {
        'source_table', 'target_table', 'date', 'shop_code',
        'mysql_count', 'postgres_count', 'delta'
    }
    if not required_cols.issubset(set(verify_detail_df.columns)):
        return pd.DataFrame()

    rows_df = verify_detail_df.copy()
    rows_df['target_table'] = rows_df['target_table'].astype(str)
    rows_df['source_table'] = rows_df['source_table'].astype(str)

    mask_offloading = (
        rows_df['target_table'].str.lower().eq('offloading_vs_loading') |
        rows_df['source_table'].str.lower().str.contains('offloading_vs_loading', na=False)
    )
    rows_df = rows_df[mask_offloading & (pd.to_numeric(rows_df['delta'], errors='coerce').fillna(0) > 0)].copy()
    if rows_df.empty:
        return pd.DataFrame()

    rows_df['group_date'] = pd.to_datetime(rows_df['date'], errors='coerce').dt.date
    rows_df = rows_df.dropna(subset=['group_date'])
    if rows_df.empty:
        return pd.DataFrame()

    rows_df['shop_code'] = rows_df['shop_code'].fillna('').astype(str).str.strip().str.upper()
    rows_df['mysql_count'] = pd.to_numeric(rows_df['mysql_count'], errors='coerce').fillna(0).astype(int)
    rows_df['postgres_count'] = pd.to_numeric(rows_df['postgres_count'], errors='coerce').fillna(0).astype(int)
    rows_df['delta'] = pd.to_numeric(rows_df['delta'], errors='coerce').fillna(0).astype(int)

    conn = psycopg2.connect(**pg_config)
    cur = conn.cursor()
    try:
        _ensure_external_sync_delta_log_table(conn)

        min_d = rows_df['group_date'].min()
        max_d = rows_df['group_date'].max()
        cur.execute(
            """
            SELECT source_table, target_table, group_date, shop_code, mysql_count, postgres_count, delta
            FROM external_sync_delta_log
            WHERE group_date BETWEEN %s AND %s
            """,
            (min_d, max_d)
        )
        existing = {
            (
                str(r[0] or '').strip().lower(),
                str(r[1] or '').strip().lower(),
                r[2],
                str(r[3] or '').strip().upper(),
                int(r[4] or 0),
                int(r[5] or 0),
                int(r[6] or 0),
            )
            for r in (cur.fetchall() or [])
        }
    finally:
        cur.close()
        conn.close()

    def _key(row):
        return (
            str(row['source_table']).strip().lower(),
            str(row['target_table']).strip().lower(),
            row['group_date'],
            str(row['shop_code']).strip().upper(),
            int(row['mysql_count']),
            int(row['postgres_count']),
            int(row['delta']),
        )

    rows_df['dedup_status'] = rows_df.apply(
        lambda r: 'ALREADY_EXISTS' if _key(r) in existing else 'NEW',
        axis=1,
    )

    preview_df = rows_df[[
        'source_table', 'target_table', 'group_date', 'shop_code',
        'mysql_count', 'postgres_count', 'delta', 'dedup_status'
    ]].copy()
    preview_df = preview_df.sort_values(
        by=['dedup_status', 'delta', 'group_date', 'shop_code'],
        ascending=[True, False, True, True],
    )
    return preview_df


def append_confirmed_offloading_deltas(pg_config: dict, preview_df: pd.DataFrame) -> int:
    """Insert only NEW preview rows into external_sync_delta_log after user confirmation."""
    if preview_df is None or preview_df.empty:
        return 0

    rows_df = preview_df.copy()
    if 'dedup_status' in rows_df.columns:
        rows_df = rows_df[rows_df['dedup_status'] == 'NEW']
    if rows_df.empty:
        return 0

    to_insert = []
    for _, r in rows_df.iterrows():
        to_insert.append((
            str(r.get('source_table') or ''),
            str(r.get('target_table') or ''),
            str(r.get('group_date') or ''),
            str(r.get('shop_code') or ''),
            int(pd.to_numeric(r.get('mysql_count'), errors='coerce') or 0),
            int(pd.to_numeric(r.get('postgres_count'), errors='coerce') or 0),
            int(pd.to_numeric(r.get('delta'), errors='coerce') or 0),
        ))

    conn = psycopg2.connect(**pg_config)
    cur = conn.cursor()
    inserted = 0
    try:
        _ensure_external_sync_delta_log_table(conn)
        cur.executemany(
            """
            INSERT INTO external_sync_delta_log
            (source_table, target_table, group_date, shop_code, mysql_count, postgres_count, delta)
            VALUES (%s, %s, %s::date, %s, %s, %s, %s)
            """,
            to_insert,
        )
        inserted = len(to_insert)
        conn.commit()
    except Exception:
        conn.rollback()
        inserted = 0
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass

    return inserted


def _ensure_external_sync_repair_log_table(pg_conn):
    cur = pg_conn.cursor()
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS external_sync_repair_log (
                id BIGSERIAL PRIMARY KEY,
                repaired_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                source_table TEXT NOT NULL,
                target_table TEXT NOT NULL,
                group_date DATE NOT NULL,
                shop_code TEXT NOT NULL,
                delta_at_verify BIGINT NOT NULL,
                rows_fetched BIGINT NOT NULL DEFAULT 0,
                rows_inserted BIGINT NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                message TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_external_sync_repair_log_group
            ON external_sync_repair_log (source_table, group_date, shop_code)
            """
        )
        pg_conn.commit()
    finally:
        cur.close()


def build_invoices_positive_delta_group_preview(pg_config: dict, verify_detail_df: pd.DataFrame) -> pd.DataFrame:
    """Preview invoices* groups where delta > 0 and mark NEW_GROUP vs ALREADY_REPAIRED."""
    if verify_detail_df is None or verify_detail_df.empty:
        return pd.DataFrame()

    required_cols = {
        'source_table', 'target_table', 'date', 'shop_code',
        'mysql_count', 'postgres_count', 'delta'
    }
    if not required_cols.issubset(set(verify_detail_df.columns)):
        return pd.DataFrame()

    rows_df = verify_detail_df.copy()
    rows_df['source_table'] = rows_df['source_table'].astype(str)
    rows_df['target_table'] = rows_df['target_table'].astype(str)
    rows_df = rows_df[
        rows_df['source_table'].str.lower().str.startswith('invoices') &
        (pd.to_numeric(rows_df['delta'], errors='coerce').fillna(0) > 0)
    ].copy()
    if rows_df.empty:
        return pd.DataFrame()

    rows_df['group_date'] = pd.to_datetime(rows_df['date'], errors='coerce').dt.date
    rows_df = rows_df.dropna(subset=['group_date'])
    if rows_df.empty:
        return pd.DataFrame()

    rows_df['shop_code'] = rows_df['shop_code'].fillna('ALL').astype(str).str.strip().str.upper()
    rows_df['mysql_count'] = pd.to_numeric(rows_df['mysql_count'], errors='coerce').fillna(0).astype(int)
    rows_df['postgres_count'] = pd.to_numeric(rows_df['postgres_count'], errors='coerce').fillna(0).astype(int)
    rows_df['delta'] = pd.to_numeric(rows_df['delta'], errors='coerce').fillna(0).astype(int)

    conn = psycopg2.connect(**pg_config)
    cur = conn.cursor()
    try:
        _ensure_external_sync_repair_log_table(conn)
        min_d = rows_df['group_date'].min()
        max_d = rows_df['group_date'].max()
        cur.execute(
            """
            SELECT source_table, group_date, shop_code
            FROM external_sync_repair_log
            WHERE group_date BETWEEN %s AND %s
              AND status = 'success'
            """,
            (min_d, max_d)
        )
        repaired_keys = {
            (str(r[0] or '').strip().lower(), r[1], str(r[2] or '').strip().upper())
            for r in (cur.fetchall() or [])
        }
    finally:
        cur.close()
        conn.close()

    rows_df['dedup_status'] = rows_df.apply(
        lambda r: 'ALREADY_REPAIRED' if (
            str(r['source_table']).strip().lower(),
            r['group_date'],
            str(r['shop_code']).strip().upper(),
        ) in repaired_keys else 'NEW_GROUP',
        axis=1,
    )

    preview_df = rows_df[[
        'source_table', 'target_table', 'group_date', 'shop_code',
        'mysql_count', 'postgres_count', 'delta', 'dedup_status'
    ]].copy()
    preview_df = preview_df.sort_values(
        by=['dedup_status', 'delta', 'source_table', 'group_date', 'shop_code'],
        ascending=[True, False, True, True, True],
    )
    return preview_df


def repair_invoices_positive_delta_groups(mysql_config: dict, pg_config: dict, preview_df: pd.DataFrame, batch_size: int = 5000) -> dict:
    """Fetch MySQL rows for NEW_GROUP invoices* positive-delta date/shop groups and insert into PostgreSQL target tables."""
    result = {
        'groups_total': 0,
        'groups_processed': 0,
        'rows_fetched': 0,
        'rows_inserted': 0,
        'groups_failed': 0,
    }

    if preview_df is None or preview_df.empty:
        return result

    work_df = preview_df.copy()
    if 'dedup_status' in work_df.columns:
        work_df = work_df[work_df['dedup_status'] == 'NEW_GROUP']
    if work_df.empty:
        return result

    work_df['source_table'] = work_df['source_table'].astype(str)
    work_df['group_date'] = pd.to_datetime(work_df['group_date'], errors='coerce').dt.date
    work_df['shop_code'] = work_df['shop_code'].fillna('ALL').astype(str).str.strip().str.upper()
    work_df['delta'] = pd.to_numeric(work_df['delta'], errors='coerce').fillna(0).astype(int)
    work_df = work_df.dropna(subset=['group_date'])

    result['groups_total'] = len(work_df)
    if work_df.empty:
        return result

    try:
        import pymysql
    except ImportError:
        raise RuntimeError("pymysql not installed. Install with: pip install pymysql")

    mysql_conn = pymysql.connect(
        host=mysql_config['host'],
        user=mysql_config['user'],
        password=mysql_config['password'],
        database=mysql_config['database'],
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=20,
        read_timeout=120,
    )
    pg_conn = psycopg2.connect(**pg_config)
    mysql_cur = mysql_conn.cursor()
    pg_cur = pg_conn.cursor()

    date_candidates = [
        'invdate', 'alert_date', 'entry_time', 'dt_mod_date',
        'a_entrytime', 'scanned_date',
        'created_at', 'updated_at', 'modified_at', 'date_invoice',
        'bill_date', 'dt_invoice_date', 'sync_time'
    ]
    shop_candidates = ['store_code', 'a_store_code', 'shop_code', 'shop', 'shopcode']

    try:
        _ensure_external_sync_repair_log_table(pg_conn)

        table_groups = {}
        for _, r in work_df.iterrows():
            table_groups.setdefault(str(r['source_table']), []).append(r)

        for source_table, groups in table_groups.items():
            target_table = source_table.lower()

            mysql_cur.execute(
                """
                SELECT COLUMN_NAME, DATA_TYPE, COLUMN_KEY
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                ORDER BY ORDINAL_POSITION
                """,
                (mysql_config['database'], source_table),
            )
            col_meta = mysql_cur.fetchall() or []
            if not col_meta:
                result['groups_failed'] += len(groups)
                continue

            columns = [c['COLUMN_NAME'] for c in col_meta]
            lower_map = {c['COLUMN_NAME'].lower(): c for c in col_meta}
            pk_cols = [c['COLUMN_NAME'] for c in col_meta if c.get('COLUMN_KEY') == 'PRI']

            date_column = None
            for candidate in date_candidates:
                if candidate in lower_map and str(lower_map[candidate]['DATA_TYPE']).lower() in ('datetime', 'timestamp', 'date'):
                    date_column = lower_map[candidate]['COLUMN_NAME']
                    break
            if not date_column:
                result['groups_failed'] += len(groups)
                continue

            shop_column = None
            for candidate in shop_candidates:
                if candidate in lower_map:
                    shop_column = lower_map[candidate]['COLUMN_NAME']
                    break

            pg_cur.execute("SELECT to_regclass(%s)", (f'public.{target_table}',))
            if pg_cur.fetchone()[0] is None:
                result['groups_failed'] += len(groups)
                continue

            pg_cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                """,
                (target_table,)
            )
            pg_target_cols = {str(r[0]).lower() for r in (pg_cur.fetchall() or [])}
            base_pg_cols = [c.lower() for c in columns if c.lower() in pg_target_cols]
            if not base_pg_cols:
                result['groups_failed'] += len(groups)
                continue

            include_hash = '_source_row_hash' in pg_target_cols
            insert_cols = list(base_pg_cols) + (['_source_row_hash'] if include_hash else [])
            insert_col_sql = ', '.join([_quote_ident(c) for c in insert_cols])

            for grp in groups:
                grp_date = grp['group_date']
                grp_shop = str(grp['shop_code']).strip().upper()
                grp_delta = int(grp['delta'])

                try:
                    where_sql = f"WHERE DATE(`{date_column}`) = %s"
                    params = [grp_date]
                    if shop_column and grp_shop and grp_shop != 'ALL':
                        where_sql += f" AND UPPER(TRIM(COALESCE(`{shop_column}`, ''))) = %s"
                        params.append(grp_shop)

                    mysql_sql = (
                        f"SELECT {', '.join([f'`{c}`' for c in columns])} "
                        f"FROM `{source_table}` {where_sql}"
                    )
                    mysql_cur.execute(mysql_sql, params)
                    rows = mysql_cur.fetchall() or []
                    fetched = len(rows)

                    inserted = 0
                    if fetched > 0:
                        value_rows = []
                        for row in rows:
                            normalized = {k.lower(): v for k, v in row.items()}
                            out_vals = [normalized.get(c) for c in base_pg_cols]
                            if include_hash:
                                row_hash = hashlib.md5(
                                    _serialize_row_for_hash(normalized, [c.lower() for c in columns]).encode('utf-8', errors='ignore')
                                ).hexdigest()
                                out_vals.append(row_hash)
                            value_rows.append(tuple(out_vals))

                        insert_sql = (
                            f"INSERT INTO {_quote_ident(target_table)} ({insert_col_sql}) VALUES %s ON CONFLICT DO NOTHING"
                        )
                        execute_values(pg_cur, insert_sql, value_rows, page_size=min(int(batch_size), 5000))
                        inserted = pg_cur.rowcount if pg_cur.rowcount is not None else 0
                        pg_conn.commit()

                    result['groups_processed'] += 1
                    result['rows_fetched'] += fetched
                    result['rows_inserted'] += max(inserted, 0)

                    pg_cur.execute(
                        """
                        INSERT INTO external_sync_repair_log
                        (source_table, target_table, group_date, shop_code, delta_at_verify, rows_fetched, rows_inserted, status, message)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, 'success', %s)
                        ON CONFLICT (source_table, group_date, shop_code)
                        DO UPDATE SET
                            repaired_at = CURRENT_TIMESTAMP,
                            target_table = EXCLUDED.target_table,
                            delta_at_verify = EXCLUDED.delta_at_verify,
                            rows_fetched = EXCLUDED.rows_fetched,
                            rows_inserted = EXCLUDED.rows_inserted,
                            status = EXCLUDED.status,
                            message = EXCLUDED.message
                        """,
                        (
                            source_table,
                            target_table,
                            grp_date,
                            grp_shop,
                            grp_delta,
                            fetched,
                            max(inserted, 0),
                            'delta>0 group backfill',
                        )
                    )
                    pg_conn.commit()
                except Exception as grp_err:
                    result['groups_failed'] += 1
                    pg_conn.rollback()
                    try:
                        pg_cur.execute(
                            """
                            INSERT INTO external_sync_repair_log
                            (source_table, target_table, group_date, shop_code, delta_at_verify, rows_fetched, rows_inserted, status, message)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, 'failed', %s)
                            ON CONFLICT (source_table, group_date, shop_code)
                            DO UPDATE SET
                                repaired_at = CURRENT_TIMESTAMP,
                                status = EXCLUDED.status,
                                message = EXCLUDED.message
                            """,
                            (
                                source_table,
                                target_table,
                                grp_date,
                                grp_shop,
                                grp_delta,
                                0,
                                0,
                                str(grp_err),
                            )
                        )
                        pg_conn.commit()
                    except Exception:
                        pg_conn.rollback()
    finally:
        try:
            mysql_cur.close()
            mysql_conn.close()
        except Exception:
            pass
        try:
            pg_cur.close()
            pg_conn.close()
        except Exception:
            pass

    return result

# ===========================
# AUTH SYSTEM
# ===========================

ADMIN_USERNAME = "admin"
_ADMIN_PW_HASH = hashlib.sha256("Melcom@Admin2024".encode()).hexdigest()

_AUTH_DB = {
    'host': 'localhost', 'port': 3307, 'user': 'postgres',
    'password': 'hello', 'database': 'salesdata'
}

def _auth_conn():
    return psycopg2.connect(**_AUTH_DB)

def _setup_auth_tables():
    try:
        conn = _auth_conn()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS portal_users (
                username    VARCHAR(50) PRIMARY KEY,
                display_name VARCHAR(100) NOT NULL,
                password_hash VARCHAR(64) NOT NULL,
                is_admin    BOOLEAN DEFAULT FALSE,
                created_at  TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS portal_access (
                username      VARCHAR(50) REFERENCES portal_users(username) ON DELETE CASCADE,
                dashboard_id  VARCHAR(50) NOT NULL,
                granted_at    TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (username, dashboard_id)
            )
        """)
        conn.commit()
        cur.close(); conn.close()
    except Exception:
        pass

def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def verify_login(username: str, password: str):
    """Returns (display_name, is_admin) tuple or None if invalid."""
    if username.strip().lower() == ADMIN_USERNAME and _hash_pw(password) == _ADMIN_PW_HASH:
        return ("Administrator", True)
    try:
        conn = _auth_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT display_name, is_admin FROM portal_users WHERE username=%s AND password_hash=%s",
            (username.strip().lower(), _hash_pw(password))
        )
        row = cur.fetchone()
        cur.close(); conn.close()
        if row:
            return (row[0], row[1])
    except Exception:
        pass
    return None

def get_all_users():
    try:
        conn = _auth_conn()
        cur = conn.cursor()
        cur.execute("SELECT username, display_name, is_admin, created_at FROM portal_users ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close(); conn.close()
        return rows
    except Exception:
        return []

def add_user(username: str, display_name: str, password: str, is_admin: bool = False):
    conn = _auth_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO portal_users (username, display_name, password_hash, is_admin) VALUES (%s,%s,%s,%s) ON CONFLICT(username) DO UPDATE SET display_name=EXCLUDED.display_name, password_hash=EXCLUDED.password_hash, is_admin=EXCLUDED.is_admin",
        (username.strip().lower(), display_name.strip(), _hash_pw(password), is_admin)
    )
    conn.commit(); cur.close(); conn.close()

def delete_user(username: str):
    conn = _auth_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM portal_users WHERE username=%s", (username,))
    conn.commit(); cur.close(); conn.close()

def change_password(username: str, old_password: str, new_password: str) -> tuple[bool, str]:
    """Returns (success, message)."""
    if username.strip().lower() == ADMIN_USERNAME:
        if _hash_pw(old_password) != _ADMIN_PW_HASH:
            return False, "Current password is incorrect."
        return False, "Admin password can only be changed in the source code."
    try:
        conn = _auth_conn()
        cur  = conn.cursor()
        cur.execute(
            "SELECT 1 FROM portal_users WHERE username=%s AND password_hash=%s",
            (username.strip().lower(), _hash_pw(old_password))
        )
        if not cur.fetchone():
            cur.close(); conn.close()
            return False, "Current password is incorrect."
        cur.execute(
            "UPDATE portal_users SET password_hash=%s WHERE username=%s",
            (_hash_pw(new_password), username.strip().lower())
        )
        conn.commit(); cur.close(); conn.close()
        return True, "Password changed successfully."
    except Exception as e:
        return False, f"Database error: {e}"

def get_user_access(username: str) -> set:
    try:
        conn = _auth_conn()
        cur = conn.cursor()
        cur.execute("SELECT dashboard_id FROM portal_access WHERE username=%s", (username,))
        ids = {r[0] for r in cur.fetchall()}
        cur.close(); conn.close()
        return ids
    except Exception:
        return set()

def set_user_access(username: str, dashboard_ids: list):
    conn = _auth_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM portal_access WHERE username=%s", (username,))
    if dashboard_ids:
        execute_values(cur, "INSERT INTO portal_access (username, dashboard_id) VALUES %s",
                       [(username, did) for did in dashboard_ids])
    conn.commit(); cur.close(); conn.close()

def get_accessible_dashboards(username: str, is_admin: bool):
    if is_admin:
        return DASHBOARDS
    accessible = get_user_access(username)
    return [d for d in DASHBOARDS if d.get("id") in accessible]

# ===========================
# DASHBOARD CONFIGURATION
# ===========================

DASHBOARDS = [
    {
        "id": "melcom_star",
        "name": "MELCOM STAR\n(Serial Tracking & Reporting)",
        "icon": "📦",
        "description": "Supply chain funnel for serialized items. Track loaded, offloaded, and sold items with vehicle analysis.",
        "file": "serial_funnel_dashboard.py",
        "port": 8502,
        "color": "#3498db",
        "features": ["Vehicle Tracking", "Serial Funnel", "Status Analysis"]
    },
    {
        "id": "kpi_dashboard",
        "name": "KPI Dashboard",
        "icon": "📊",
        "description": "Multi-page sales analytics with year-over-year comparisons, department analysis, and shop rankings.",
        "file": "kpi_dashboard.py",
        "port": 8503,
        "color": "#e74c3c",
        "features": ["Sales Analytics", "YoY Comparison", "Shop Rankings"]
    },
    {
        "id": "stst",
        "name": "STST",
        "icon": "📋",
        "description": "Stock transfer recommendations via materialized views. Priority shop allocation with expiry checks.",
        "file": "nowhstock_ds_final.py",
        "port": 8504,
        "color": "#f39c12",
        "features": ["Stock Transfer", "FEFO Logic", "Priority Allocation"]
    },
    {
        "id": "barcode_matcher",
        "name": "Barcode Matcher",
        "icon": "🔍",
        "description": "Upload barcodes to match with item codes. Search 229K+ barcodes, add new mappings, track history.",
        "file": "barcode_matcher_app.py",
        "port": 8505,
        "color": "#9b59b6",
        "features": ["Barcode Search", "Bulk Matching", "Master Update"]
    },
    {
        "id": "century_penetration",
        "name": "Century Penetration",
        "icon": "🎯",
        "description": "Century brand penetration analysis across shops and regions with performance tracking.",
        "file": "centuryPenetration.py",
        "port": 8506,
        "color": "#1abc9c",
        "features": ["Brand Analysis", "Penetration Rate", "Regional Comparison"]
    },
    {
        "id": "loading_offloading",
        "name": "Loading vs Offloading",
        "icon": "🚚",
        "description": "Warehouse loading versus shop offloading reconciliation with quantity and value difference analysis.",
        "file": "offloading_vs_loading_dashboard.py",
        "port": 8522,
        "url": "http://10.10.1.79:8522",
        "status_host": "10.10.1.79",
        "status_port": 8522,
        "color": "#16a085",
        "features": ["Loading Reconciliation", "Offloading Variance", "Shop Difference Analysis"]
    },
    {
        "id": "invoice_scanning",
        "name": "Invoice Scanning",
        "icon": "🧾",
        "description": "Track bill scanning compliance across all shops. Monitor duplicate bills, alert errors, and daily scan performance.",
        "file": "invoicescanning/consumable_till_dashboard.py",
        "port": 8508,
        "color": "#2980b9",
        "features": ["Bill Scan Tracking", "Duplicate Detection", "Alert Error Analysis"]
    },
    {
        "id": "cost_control",
        "name": "Cost Control Portal",
        "icon": "💰",
        "description": "Monitor wastage, cost entries, and departmental cost control across all Melcom branches.",
        "file": "cost_control_portal/manage.py",
        "port": 8507,
        "color": "#27ae60",
        "features": ["Wastage Tracking", "Cost Entry", "Branch Cost Control"]
    },
    {
        "id": "pi_dashboard",
        "name": "PI Dashboard",
        "icon": "📈",
        "description": "Physical inventory performance dashboard. Track inventory counts, variances, and shop-wise PI progress.",
        "file": "PI_Dashboard/pidashboard.py",
        "port": 8509,
        "color": "#8e44ad",
        "features": ["PI Progress", "Variance Analysis", "Shop Performance"]
    },
    {
        "id": "serial_tracker",
        "name": "Serial Number Tracking",
        "icon": "🔢",
        "description": "Track serialized items from warehouse loading through shop receipt to customer sale. Full serial lifecycle visibility.",
        "file": "SerialNoReport/serial_tracker_dashboard_new.py",
        "port": 8514,
        "color": "#d35400",
        "features": ["Serial Lifecycle", "Shop Receipt", "Sale Tracking"]
    }
]

# ===========================
# MAIN PAGE
# ===========================

# ── Session state bootstrap ──────────────────────────────────────────────────
for _k, _v in {
    "authenticated": False, "username": None,
    "is_admin": False, "display_name": None,
    "portal_view": "home",  # home | admin | upload | indexmgr
    "admin_subtab": "users",
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── DB setup ─────────────────────────────────────────────────────────────────
_setup_auth_tables()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _navbar():
    cols = st.columns([1, 3, 1])
    with cols[0]:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:10px;padding:8px 0;">'
            '<img src="https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg" width="32" style="border-radius:6px;">'
            '<span style="font-size:18px;font-weight:800;color:#e6edf3;letter-spacing:-0.3px;">Melcom Analytics Hub</span>'
            '</div>',
            unsafe_allow_html=True
        )
    with cols[2]:
        badge = (
            '<span style="background:rgba(255,215,0,0.15);border:1px solid rgba(255,215,0,0.4);'
            'color:#ffd700;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;margin-right:8px;">ADMIN</span>'
            if st.session_state.is_admin else ""
        )
        st.markdown(
            f'<div style="text-align:right;padding:8px 0;">'
            f'{badge}'
            f'<span style="color:#8b949e;font-size:13px;">👤 {st.session_state.display_name}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        if st.button("Sign Out", key="signout_btn"):
            for k in ["authenticated","username","is_admin","display_name","portal_view"]:
                st.session_state[k] = False if k == "authenticated" else None
            st.session_state.portal_view = "home"
            st.rerun()
    st.markdown('<hr style="margin:0 0 20px 0;">', unsafe_allow_html=True)

def _sidebar_nav():
    with st.sidebar:
        st.markdown(
            '<p style="color:#8b949e;font-size:11px;font-weight:700;text-transform:uppercase;'
            'letter-spacing:1px;margin-bottom:12px;">Navigation</p>',
            unsafe_allow_html=True
        )
        nav_items = [("🏠", "Home", "home"), ("📤", "Upload Data", "upload"), ("🔧", "Index Manager", "indexmgr"), ("🔑", "Change Password", "change_password")]
        if st.session_state.is_admin:
            nav_items.insert(1, ("⚙️", "Admin Panel", "admin"))
        for icon, label, view in nav_items:
            is_active = st.session_state.portal_view == view
            btn_style = (
                "background:linear-gradient(135deg,#E31837,#b01028);color:white;border:none;"
                if is_active else
                "background:#21262d;color:#c9d1d9;border:1px solid #30363d;"
            )
            if st.button(f"{icon}  {label}", key=f"nav_{view}", use_container_width=True):
                st.session_state.portal_view = view
                st.rerun()


# ── Login Page ────────────────────────────────────────────────────────────────
def _render_login():
    if 'login_mode' not in st.session_state:
        st.session_state.login_mode = 'signin'   # 'signin' | 'change_pw'

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:center;margin-bottom:8px;">'
            '<img src="https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg" width="64" style="border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,0.5);">'
            '</div>',
            unsafe_allow_html=True
        )

        # ── SIGN IN mode ──────────────────────────────────────────────────────
        if st.session_state.login_mode == 'signin':
            st.markdown(
                '<h1 style="text-align:center;color:#e6edf3;font-size:28px;font-weight:800;margin:16px 0 4px 0;">Welcome Back</h1>'
                '<p style="text-align:center;color:#8b949e;font-size:14px;margin-bottom:32px;">Sign in to Melcom Analytics Hub</p>',
                unsafe_allow_html=True
            )
            with st.container():
                username_in = st.text_input("Username", placeholder="Enter your username", key="login_user")
                password_in = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                login_btn = st.button("Sign In →", type="primary", use_container_width=True, key="login_btn")
                if login_btn:
                    if not username_in or not password_in:
                        st.error("Please enter both username and password.")
                    else:
                        result = verify_login(username_in, password_in)
                        if result:
                            display_name, is_admin = result
                            st.session_state.authenticated = True
                            st.session_state.username = username_in.strip().lower()
                            st.session_state.is_admin = is_admin
                            st.session_state.display_name = display_name
                            st.session_state.portal_view = "home"
                            st.rerun()
                        else:
                            st.error("Invalid username or password. Please try again.")

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown(
                '<p style="text-align:center;color:#6e7681;font-size:12px;">'
                'Forgot or want to change your password?</p>',
                unsafe_allow_html=True
            )
            if st.button("🔑  Change Password", use_container_width=True, key="goto_change_pw"):
                st.session_state.login_mode = 'change_pw'
                st.rerun()

        # ── CHANGE PASSWORD mode ──────────────────────────────────────────────
        else:
            st.markdown(
                '<h1 style="text-align:center;color:#e6edf3;font-size:24px;font-weight:800;margin:16px 0 4px 0;">Change Password</h1>'
                '<p style="text-align:center;color:#8b949e;font-size:14px;margin-bottom:32px;">Enter your current password to set a new one</p>',
                unsafe_allow_html=True
            )
            with st.form("login_change_pw_form", clear_on_submit=True):
                cp_user   = st.text_input("Username",         placeholder="Your username",        key="cp_user")
                cp_old    = st.text_input("Current Password", type="password",
                                          placeholder="Your current password", key="cp_old")
                cp_new    = st.text_input("New Password",     type="password",
                                          placeholder="New password (min 6 chars)", key="cp_new")
                cp_conf   = st.text_input("Confirm Password", type="password",
                                          placeholder="Re-enter new password",     key="cp_conf")
                cp_submit = st.form_submit_button("Update Password", type="primary", use_container_width=True)

            if cp_submit:
                if not all([cp_user, cp_old, cp_new, cp_conf]):
                    st.error("All fields are required.")
                elif cp_new != cp_conf:
                    st.error("New passwords do not match.")
                elif len(cp_new) < 6:
                    st.error("New password must be at least 6 characters.")
                else:
                    ok, msg = change_password(cp_user.strip().lower(), cp_old, cp_new)
                    if ok:
                        st.success(f"✓ {msg}  Please sign in with your new password.")
                        st.session_state.login_mode = 'signin'
                        st.rerun()
                    else:
                        st.error(msg)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            if st.button("← Back to Sign In", use_container_width=True, key="back_to_login"):
                st.session_state.login_mode = 'signin'
                st.rerun()

        st.markdown(
            '<p style="text-align:center;color:#6e7681;font-size:12px;margin-top:24px;">'
            '🔒 Secure Access · Melcom Group Limited</p>',
            unsafe_allow_html=True
        )


# ── Dashboard Portal ─────────────────────────────────────────────────────────
def _render_portal():
    ipv4 = get_ipv4_address()
    my_dashboards = get_accessible_dashboards(st.session_state.username, st.session_state.is_admin)

    if not my_dashboards:
        st.markdown(
            '<div style="text-align:center;padding:80px 20px;">'
            '<div style="font-size:64px;margin-bottom:16px;">🔒</div>'
            '<h2 style="color:#e6edf3;">No Dashboards Assigned</h2>'
            '<p style="color:#8b949e;max-width:400px;margin:0 auto;">You don\'t have access to any dashboards yet. Please contact your administrator.</p>'
            '</div>',
            unsafe_allow_html=True
        )
        return

    # Status check
    status_map = {
        d["name"]: check_dashboard_status(d.get("status_port", d["port"]), d.get("status_host", "localhost"))
        for d in my_dashboards
    }
    online_count = sum(status_map.values())
    total = len(my_dashboards)

    # Welcome banner
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")
    st.markdown(
        f'<div style="background:linear-gradient(135deg,rgba(227,24,55,0.12) 0%,rgba(118,75,162,0.12) 100%);'
        f'border:1px solid rgba(227,24,55,0.2);border-radius:16px;padding:24px 32px;margin-bottom:24px;">'
        f'<div style="font-size:22px;font-weight:700;color:#e6edf3;">{greeting}, {st.session_state.display_name} 👋</div>'
        f'<div style="font-size:14px;color:#8b949e;margin-top:4px;">Here are your analytics dashboards</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Stats row
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;text-align:center;">'
            f'<div style="font-size:32px;font-weight:800;color:#e6edf3;">{total}</div>'
            f'<div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:0.8px;margin-top:4px;">Your Dashboards</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with s2:
        st.markdown(
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;text-align:center;">'
            f'<div style="font-size:32px;font-weight:800;color:#3fb950;">{online_count}</div>'
            f'<div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:0.8px;margin-top:4px;">Currently Online</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with s3:
        offline = total - online_count
        st.markdown(
            f'<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;text-align:center;">'
            f'<div style="font-size:32px;font-weight:800;color:#{"f85149" if offline else "8b949e"}">{offline}</div>'
            f'<div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:0.8px;margin-top:4px;">Offline</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Dashboard cards (rendered as HTML component for rich hover effects)
    cards_html = ""
    for d in my_dashboards:
        is_online = status_map.get(d["name"], False)
        url = d.get("url") or f"http://{ipv4}:{d['port']}"
        features_html = "".join(f'<span class="feature-chip">{f}</span>' for f in d.get("features", []))
        status_color = "#3fb950" if is_online else "#f85149"
        status_dot = "🟢" if is_online else "🔴"
        status_label = "Online" if is_online else "Offline"
        btn_class = "launch-btn-active" if is_online else "launch-btn-inactive"
        btn_label = "Open Dashboard →" if is_online else "Dashboard Offline"
        onclick_val = f"window.open('{url}', '_blank')" if is_online else ""
        a_onclick = 'onclick="event.stopPropagation()"' if is_online else 'onclick="return false;"'
        card_name_html = d['name'].replace('\n', '<br>')
        cards_html += f"""
        <div class="dash-card" onclick="{onclick_val}">
            <div class="card-header">
                <span class="card-icon">{d['icon']}</span>
                <span class="status-dot" style="color:{status_color};">{status_dot} {status_label}</span>
            </div>
            <h3 class="card-name">{card_name_html}</h3>
            <p class="card-desc">{d['description']}</p>
            <div class="feature-row">{features_html}</div>
            <a href="{url}" target="_blank" class="{btn_class}" {a_onclick}>
                {btn_label}
            </a>
        </div>"""

    components.html(f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; font-family:'Inter',-apple-system,sans-serif; }}
  body {{ background:#0d1117; padding:4px 0 20px 0; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:20px; }}
  .dash-card {{
    background:#161b22; border:1px solid #30363d; border-radius:16px;
    padding:24px; cursor:pointer; transition:all 0.25s ease;
    display:flex; flex-direction:column; gap:12px;
    position:relative; overflow:hidden;
  }}
  .dash-card::before {{
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,#E31837,#764ba2);
    opacity:0; transition:opacity 0.25s ease;
  }}
  .dash-card:hover {{ border-color:#6e7681; transform:translateY(-4px); box-shadow:0 12px 40px rgba(0,0,0,0.5); }}
  .dash-card:hover::before {{ opacity:1; }}
  .card-header {{ display:flex; align-items:center; justify-content:space-between; }}
  .card-icon {{ font-size:36px; line-height:1; }}
  .status-dot {{ font-size:12px; font-weight:600; }}
  .card-name {{ font-size:18px; font-weight:700; color:#e6edf3; line-height:1.3; }}
  .card-desc {{ font-size:13px; color:#8b949e; line-height:1.6; flex-grow:1; }}
  .feature-row {{ display:flex; flex-wrap:wrap; gap:6px; }}
  .feature-chip {{
    background:rgba(255,255,255,0.06); border:1px solid #30363d;
    color:#8b949e; font-size:11px; padding:3px 10px; border-radius:20px; font-weight:500;
  }}
  .launch-btn-active {{
    display:block; text-align:center; background:linear-gradient(135deg,#E31837,#b01028);
    color:white; text-decoration:none; padding:12px 20px; border-radius:10px;
    font-weight:700; font-size:14px; margin-top:4px;
    transition:all 0.2s ease; box-shadow:0 4px 15px rgba(227,24,55,0.25);
  }}
  .launch-btn-active:hover {{ box-shadow:0 8px 25px rgba(227,24,55,0.5); transform:translateY(-1px); }}
  .launch-btn-inactive {{
    display:block; text-align:center; background:#21262d;
    color:#6e7681; text-decoration:none; padding:12px 20px; border-radius:10px;
    font-weight:600; font-size:14px; margin-top:4px; cursor:not-allowed;
    border:1px solid #30363d;
  }}
</style>
</head>
<body>
  <div class="grid">{cards_html}</div>
</body>
</html>
""", height=max(400, len(my_dashboards) * 90))


# ── Admin Panel ────────────────────────────────────────────────────────────────
def _render_admin():
    st.markdown('<h2 style="color:#e6edf3;font-weight:800;margin-bottom:4px;">Admin Panel</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;font-size:14px;margin-bottom:24px;">Manage users and control dashboard access</p>', unsafe_allow_html=True)

    tab_users, tab_access = st.tabs(["👥  User Management", "🔑  Dashboard Access"])

    # ── Tab: Users ────────────────────────────────────────────────────────────
    with tab_users:
        st.markdown("#### Add New User")
        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                new_username = st.text_input("Username", key="nu_user", placeholder="e.g. john.doe")
                new_display  = st.text_input("Full Name", key="nu_display", placeholder="e.g. John Doe")
            with c2:
                new_password = st.text_input("Password", type="password", key="nu_pass", placeholder="Min. 6 characters")
                new_is_admin = st.checkbox("Grant Admin Rights", key="nu_admin")
            if st.button("➕ Create User", type="primary", key="create_user_btn"):
                if not new_username or not new_password or not new_display:
                    st.error("Please fill in all fields.")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_username.strip().lower() == ADMIN_USERNAME:
                    st.error("Cannot create a user with the reserved admin username.")
                else:
                    try:
                        add_user(new_username, new_display, new_password, new_is_admin)
                        st.success(f"✅ User '{new_display}' created successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.markdown("---")
        st.markdown("#### Existing Users")
        users = get_all_users()
        if not users:
            st.info("No users created yet. Add your first user above.")
        else:
            for uname, dname, is_adm, created in users:
                with st.container():
                    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
                    with c1:
                        role_badge = (
                            '<span style="background:rgba(255,215,0,0.15);border:1px solid rgba(255,215,0,0.4);'
                            'color:#ffd700;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:700;margin-left:8px;">ADMIN</span>'
                            if is_adm else
                            '<span style="background:#21262d;border:1px solid #30363d;'
                            'color:#8b949e;padding:2px 8px;border-radius:12px;font-size:11px;">USER</span>'
                        )
                        st.markdown(
                            f'<div style="padding:10px 0;">'
                            f'<span style="color:#e6edf3;font-weight:600;">{dname}</span>'
                            f'{role_badge}</div>',
                            unsafe_allow_html=True
                        )
                    with c2:
                        st.markdown(
                            f'<div style="color:#8b949e;font-size:13px;padding:14px 0;">@{uname}</div>',
                            unsafe_allow_html=True
                        )
                    with c3:
                        access_count = len(get_user_access(uname))
                        st.markdown(
                            f'<div style="color:#8b949e;font-size:13px;padding:14px 0;">'
                            f'{"All" if is_adm else access_count} dashboard{"s" if (is_adm or access_count!=1) else ""}</div>',
                            unsafe_allow_html=True
                        )
                    with c4:
                        if st.button("🗑 Remove", key=f"del_{uname}"):
                            delete_user(uname)
                            st.success(f"User '{dname}' removed.")
                            st.rerun()
                    st.markdown('<hr style="margin:4px 0;">', unsafe_allow_html=True)

    # ── Tab: Access ───────────────────────────────────────────────────────────
    with tab_access:
        users = get_all_users()
        non_admin_users = [(uname, dname) for uname, dname, is_adm, _ in users if not is_adm]
        if not non_admin_users:
            st.info("No regular users to configure. Create users in the User Management tab first.")
            return

        user_options = {f"{dname} (@{uname})": uname for uname, dname in non_admin_users}
        selected_label = st.selectbox(
            "Select User to Configure",
            options=list(user_options.keys()),
            key="access_user_select"
        )
        if not selected_label:
            return
        target_username = user_options[selected_label]

        st.markdown(f"<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(f"#### Dashboard Access for **{selected_label}**")
        st.markdown(
            '<p style="color:#8b949e;font-size:13px;margin-bottom:16px;">'
            'Tick the dashboards this user is allowed to open.</p>',
            unsafe_allow_html=True
        )

        current_access = get_user_access(target_username)
        new_access = []
        for d in DASHBOARDS:
            did = d.get("id", d["name"])
            is_granted = did in current_access
            label_html = f"{d['icon']} {d['name'].replace(chr(10), ' — ')}"
            checked = st.checkbox(label_html, value=is_granted, key=f"acc_{target_username}_{did}")
            if checked:
                new_access.append(did)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("💾 Save Access Settings", type="primary", key="save_access_btn"):
            set_user_access(target_username, new_access)
            st.success(f"✅ Access updated for {selected_label}")
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
def _render_change_password():
    st.markdown('<h2 style="color:#e6edf3;font-weight:800;margin-bottom:4px;">Change Password</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;font-size:14px;margin-bottom:24px;">Update your account password</p>', unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:16px;
             padding:32px;position:relative;overflow:hidden;">
          <div style="position:absolute;top:0;left:0;right:0;height:3px;
               background:linear-gradient(90deg,#E31837,#764ba2);"></div>
          <p style="color:#8b949e;font-size:12px;font-weight:700;text-transform:uppercase;
               letter-spacing:.8px;margin-bottom:20px;">Account Security</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("change_pw_form", clear_on_submit=True):
            current_pw  = st.text_input("Current Password",  type="password", placeholder="Enter current password")
            new_pw      = st.text_input("New Password",      type="password", placeholder="Enter new password")
            confirm_pw  = st.text_input("Confirm Password",  type="password", placeholder="Re-enter new password")
            submitted   = st.form_submit_button("Update Password", type="primary", use_container_width=True)

        if submitted:
            if not current_pw or not new_pw or not confirm_pw:
                st.error("All fields are required.")
            elif new_pw != confirm_pw:
                st.error("New passwords do not match.")
            elif len(new_pw) < 6:
                st.error("New password must be at least 6 characters.")
            else:
                ok, msg = change_password(st.session_state.username, current_pw, new_pw)
                if ok:
                    st.success(f"✓ {msg}  Please log in again.")
                    for k in ["authenticated", "username", "is_admin", "display_name", "portal_view"]:
                        st.session_state[k] = False if k == "authenticated" else None
                    st.rerun()
                else:
                    st.error(msg)


# ROUTING
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.authenticated:
    _render_login()
else:
    _navbar()
    _sidebar_nav()

    view = st.session_state.portal_view

    if view == "home":
        _render_portal()

    elif view == "admin" and st.session_state.is_admin:
        _render_admin()

    elif view == "change_password":
        _render_change_password()

    elif view == "upload":
        if not st.session_state.is_admin:
            st.warning("🔒 Data upload is restricted to administrators.")
        else:
            st.markdown('<h2 style="color:#e6edf3;font-weight:800;margin-bottom:4px;">Data Upload</h2>', unsafe_allow_html=True)
            st.markdown('<p style="color:#8b949e;font-size:14px;margin-bottom:24px;">Upload and manage data files for dashboards</p>', unsafe_allow_html=True)
            # ---- Original upload tab content ----
            st.markdown("### 📤 Data Upload Manager")
            st.info("Upload CSV files to database tables with automatic backup, indexing, and validation")

            # Report filter
            col_filter1, col_filter2 = st.columns([1, 2])
            with col_filter1:
                report_type = st.selectbox(
                    "Report Category",
                    options=["All Tables", "Century Penetration", "Serial No Tracking", "Loading vs Offloading", "Others"],
                    help="Filter tables by category"
                )

            with col_filter2:
                # Filter table list based on report type
                if report_type == "Century Penetration":
                    available_tables = ["GEN_sales", "GEN_SIT", "GEN_reorder_level", "GEN_whstock"]
                elif report_type == "Serial No Tracking":
                    available_tables = ["serial_no_dailydata", "whreceived_serialno", "serialno_check_yes_no"]
                elif report_type == "Loading vs Offloading":
                    available_tables = ["LVO_offloading_vs_loading", "LVO_offloading_loading_staging", "LVO_shopmgrname"]
                elif report_type == "Others":
                    century_tables = {"GEN_sales", "GEN_SIT", "GEN_reorder_level", "GEN_whstock"}
                    serial_tables = {"serial_no_dailydata", "whreceived_serialno", "serialno_check_yes_no"}
                    loading_tables = {"LVO_offloading_vs_loading", "LVO_offloading_loading_staging", "LVO_shopmgrname"}
                    available_tables = [t for t in TABLE_CONFIGS.keys() if t not in century_tables and t not in serial_tables and t not in loading_tables]
                else:  # All Tables
                    available_tables = list(TABLE_CONFIGS.keys())

                selected_table = st.selectbox(
                    "Select Target Table",
                    options=available_tables,
                    help="Choose the database table to upload data to"
                )

            # Table selection metrics
            col1, col2, col3 = st.columns([2, 1, 1])

            with col2:
                # Show current table count
                if selected_table:
                    config = TABLE_CONFIGS[selected_table]
                    target_table = config.get('target_table', selected_table)
                    target_database = config.get('database', 'salesdata')
                    current_count = get_table_count(target_table, target_database)
                    if current_count is not None:
                        st.metric("Current Rows", f"{current_count:,}")

            with col3:
                # Show latest date
                if selected_table and 'date_column' in TABLE_CONFIGS[selected_table]:
                    config = TABLE_CONFIGS[selected_table]
                    target_table = config.get('target_table', selected_table)
                    target_database = config.get('database', 'salesdata')
                    date_col = TABLE_CONFIGS[selected_table]['date_column']
                    if date_col:
                        latest_date = get_latest_date(target_table, date_col, target_database)
                        if latest_date:
                            # Handle both datetime objects and string dates
                            if hasattr(latest_date, 'strftime'):
                                date_str = latest_date.strftime('%Y-%m-%d')
                            else:
                                date_str = str(latest_date)
                            st.metric("Latest Date", date_str)

            st.markdown("---")

            # Display table info
            if selected_table is None or selected_table == "":
                st.info("👆 **Step 1:** Select a Report Category and Table above")
                st.markdown("**📋 Next Steps:**")
                st.markdown("1. Choose a table from the dropdown → table details will appear")
                st.markdown("2. Upload a CSV file with required columns")
                st.markdown("3. Click **🚀 START DATA UPLOAD** to begin")
                st.markdown("4. Monitor progress and view results in real-time")
            elif selected_table:

                config = TABLE_CONFIGS[selected_table]

                with st.expander("📋 Table Information", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        # Show actual upload target (may differ from config key when staging is used)
                        target_tbl = config.get('target_table', selected_table)
                        has_staging = target_tbl != selected_table
                        if has_staging:
                            st.markdown(f"**Upload Target:** `{target_tbl}` *(staging)*")
                            st.markdown(f"**Final Table:** `{selected_table}` *(after post-SQL transfer)*")
                        else:
                            st.markdown(f"**Table Name:** `{selected_table}`")
                        st.markdown(f"**Backup Table:** `{config['backup_table']}`")
                        st.markdown(f"**Date Column:** `{config.get('date_column', 'N/A')}`")
                        upload_mode = config.get('upload_mode', 'incremental')
                        if upload_mode == "incremental":
                            mode_badge = "🔄 INCREMENTAL"
                        elif upload_mode == "truncate":
                            mode_badge = "🗑️ TRUNCATE"
                        elif upload_mode == "append":
                            mode_badge = "➕ APPEND"
                        else:
                            mode_badge = f"📝 {upload_mode.upper()}"
                        st.markdown(f"**Upload Mode:** {mode_badge}")
                    with col2:
                        st.markdown(f"**Required Columns:** {len(config['columns'])}")
                        st.markdown(f"**Indexes:** {len(config['indexes'])}")
                        encoding = config.get('encoding', 'UTF-8')
                        st.markdown(f"**Encoding:** `{encoding}`")
                        if config.get('history_mode'):
                            st.markdown("**Process:** Auto-create table → Update current data → Save full history snapshot")
                        elif upload_mode == "truncate":
                            st.markdown("**Process:** Backup ALL → Truncate → Upload")
                        elif upload_mode == "append" and has_staging:
                            st.markdown(f"**Process:** CSV → `{target_tbl}` (staging) → delete date range from `{selected_table}` → insert new rows")
                        elif upload_mode == "append":
                            st.markdown("**Process:** Upload → Append/History Update")
                        else:
                            st.markdown("**Process:** Drop indexes → Backup NEW → Upload → Create indexes")

                # File upload
                uploaded_files = st.file_uploader(
                    "Choose File (CSV or Excel)",
                    type=['csv', 'xlsx', 'xls'],
                    accept_multiple_files=True,
                    help="Upload one or multiple CSV/XLSX/XLS files with required columns"
                )

                if uploaded_files:
                    try:
                        uploaded_file_list = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]

                        def _uploaded_size_bytes(uf):
                            size = getattr(uf, 'size', None)
                            if isinstance(size, int):
                                return size
                            try:
                                return len(uf.getvalue())
                            except Exception:
                                return 0

                        tiny_files = []
                        filtered_files = []
                        for uf in uploaded_file_list:
                            sz = _uploaded_size_bytes(uf)
                            if sz <= 1024:
                                tiny_files.append(f"{getattr(uf, 'name', 'unknown')} ({sz} bytes)")
                            else:
                                filtered_files.append(uf)

                        if tiny_files:
                            st.warning("Ignoring tiny file(s) <= 1KB: " + ", ".join(tiny_files))

                        uploaded_file_list = filtered_files
                        if not uploaded_file_list:
                            st.info("No valid files to process after ignoring tiny files.")
                            st.stop()

                        if len(uploaded_file_list) > 1 and selected_table != "LVO_offloading_vs_loading":
                            st.warning("Multiple files selected. This table currently uploads one file at a time; using the first file.")
                            uploaded_file_list = uploaded_file_list[:1]

                        uploaded_file = uploaded_file_list[0]
                        file_name = str(getattr(uploaded_file, 'name', '') or '')
                        file_name_lower = file_name.lower()
                        is_excel = file_name_lower.endswith('.xlsx') or file_name_lower.endswith('.xls')
                        lvo_multi_prepared = False

                        # Read CSV with automatic encoding detection
                        encoding = config.get('encoding', 'utf-8')
                        skiprows = config.get('skiprows', 0)
                        read_csv_kwargs = {'skiprows': skiprows} if skiprows else {}
                        if encoding.upper() == 'WIN1252':
                            encoding = 'cp1252'  # pandas uses cp1252 for Windows-1252

                        if selected_table == "LVO_offloading_vs_loading" and len(uploaded_file_list) > 1:
                            chunks = []
                            file_names = []
                            for uf in uploaded_file_list:
                                one_name = str(getattr(uf, 'name', '') or '')
                                one_name_lower = one_name.lower()
                                file_names.append(one_name)
                                if one_name_lower.endswith('.xlsx') or one_name_lower.endswith('.xls'):
                                    uf.seek(0)
                                    one_df = pd.read_excel(uf, skiprows=skiprows if skiprows else 0)
                                else:
                                    try:
                                        uf.seek(0)
                                        one_df = pd.read_csv(uf, encoding=encoding, **read_csv_kwargs)
                                    except Exception:
                                        one_df = None
                                        for enc in ['cp1252', 'latin1', 'iso-8859-1', 'utf-8']:
                                            for sep in [None, ',', ';', '\t', '|']:
                                                try:
                                                    uf.seek(0)
                                                    one_df = pd.read_csv(
                                                        uf,
                                                        encoding=enc,
                                                        sep=sep,
                                                        engine='python',
                                                        on_bad_lines='skip',
                                                        **read_csv_kwargs,
                                                    )
                                                    break
                                                except Exception:
                                                    continue
                                            if one_df is not None:
                                                break
                                        if one_df is None:
                                            raise ValueError(f"Could not detect encoding for file: {one_name}")
                                one_df.columns = one_df.columns.str.strip().str.lower()
                                one_df = _prepare_lvo_offloading_df(one_df, one_name)
                                chunks.append(one_df)

                            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                            file_name = ";".join(file_names)
                            lvo_multi_prepared = True
                            st.info(f"✅ Loaded {len(uploaded_file_list)} files together for append upload")

                        if not lvo_multi_prepared and is_excel:
                            uploaded_file.seek(0)
                            df = pd.read_excel(uploaded_file, skiprows=skiprows if skiprows else 0)

                        # Adaptive CSV read for serialno_check_yes_no:
                        # auto-detect best header row (skiprows) and encoding.
                        elif not lvo_multi_prepared and selected_table == "serialno_check_yes_no":
                            def _compact_header(col_name):
                                header_key = str(col_name).lower().strip().replace('coode', 'code')
                                return re.sub(r'[^a-z0-9]+', '', header_key)

                            expected_headers = {_compact_header(c) for c in config['columns']}

                            encoding_candidates = [encoding, 'utf-8', 'cp1252', 'latin1', 'iso-8859-1']
                            encoding_candidates = list(dict.fromkeys(encoding_candidates))

                            skiprows_candidates = [skiprows, 0, 1, 2, 3, 4, 5]
                            skiprows_candidates = list(dict.fromkeys(skiprows_candidates))

                            delimiter_candidates = [None, ',', ';', '\t', '|']

                            best_df = None
                            best_score = -1
                            best_enc = None
                            best_skiprows = None
                            best_sep = None

                            for enc in encoding_candidates:
                                for skip_val in skiprows_candidates:
                                    for sep in delimiter_candidates:
                                        try:
                                            uploaded_file.seek(0)
                                            trial_df = pd.read_csv(
                                                uploaded_file,
                                                encoding=enc,
                                                skiprows=skip_val,
                                                sep=sep,
                                                engine='python',
                                                on_bad_lines='skip'
                                            )
                                            found_headers = {_compact_header(c) for c in trial_df.columns}
                                            score = len(expected_headers & found_headers)

                                            if score > best_score:
                                                best_score = score
                                                best_df = trial_df
                                                best_enc = enc
                                                best_skiprows = skip_val
                                                best_sep = sep

                                            if score == len(expected_headers):
                                                break
                                        except UnicodeDecodeError:
                                            continue
                                        except Exception:
                                            continue
                                    if best_score == len(expected_headers):
                                        break
                                if best_score == len(expected_headers):
                                    break

                            if best_df is None:
                                raise ValueError("Could not read CSV header. Please verify file format and delimiter.")

                            df = best_df

                            if best_enc != encoding:
                                st.info(f"✅ Auto-detected encoding: {best_enc}")
                            if best_skiprows != skiprows:
                                st.info(f"✅ Auto-detected header row using skiprows={best_skiprows} (configured: {skiprows})")
                            if best_sep not in (None, ','):
                                sep_label = {'\t': 'TAB'}.get(best_sep, best_sep)
                                st.info(f"✅ Auto-detected delimiter: {sep_label}")
                        elif not lvo_multi_prepared:
                            # Try reading with specified encoding, fallback to auto-detection
                            try:
                                df = pd.read_csv(uploaded_file, encoding=encoding, **read_csv_kwargs)
                            except Exception:
                                # Auto-detect encoding (common: utf-8, cp1252, latin1, iso-8859-1)
                                uploaded_file.seek(0)  # Reset file pointer
                                for enc in ['cp1252', 'latin1', 'iso-8859-1', 'utf-8']:
                                    parsed = False
                                    for sep in [None, ',', ';', '\t', '|']:
                                        try:
                                            uploaded_file.seek(0)
                                            df = pd.read_csv(
                                                uploaded_file,
                                                encoding=enc,
                                                sep=sep,
                                                engine='python',
                                                on_bad_lines='skip',
                                                **read_csv_kwargs,
                                            )
                                            st.info(f"✅ Auto-detected encoding: {enc}" + (f", delimiter: {sep if sep is not None else 'auto'}"))
                                            parsed = True
                                            break
                                        except Exception:
                                            continue
                                    if parsed:
                                        break
                                else:
                                    raise ValueError("Could not parse CSV. Check delimiter/header; supported delimiters: comma, semicolon, tab, pipe.")

                        # Show CSV columns for debugging (CRITICAL - shows actual column names)
                        st.warning(f"🔍 **FILE COLUMNS FOUND:** {', '.join(df.columns.tolist())}")
                        st.info(f"🎯 **EXPECTED COLUMNS:** {', '.join(config['columns'])}")

                        # Normalize column names for flexible matching (especially for GEN_whstock)
                        original_columns = df.columns.tolist()

                        # For serial_no_dailydata and whreceived_serialno and serialno_check_yes_no: map CSV columns (uppercase) to match config
                        if selected_table in ["serial_no_dailydata", "whreceived_serialno", "serialno_check_yes_no"]:
                            # Normalize column names: lowercase, normalize separators, fix typos
                            def normalize_col(col):
                                normalized = str(col).lower().strip()
                                normalized = normalized.replace('coode', 'code')
                                normalized = re.sub(r'[^a-z0-9]+', '_', normalized).strip('_')
                                return normalized

                            # Compact form for robust matching when separators are missing
                            # e.g. CASHIERNAME == CASHIER_NAME
                            def compact_col(col):
                                compact = str(col).lower().strip().replace('coode', 'code')
                                compact = re.sub(r'[^a-z0-9]+', '', compact)
                                return compact

                            # Create normalized mapping
                            csv_col_normalized = {}
                            csv_col_compact = {}
                            for col in df.columns:
                                norm_key = normalize_col(col)
                                compact_key = compact_col(col)
                                if norm_key not in csv_col_normalized:
                                    csv_col_normalized[norm_key] = col
                                if compact_key not in csv_col_compact:
                                    csv_col_compact[compact_key] = col
                            config_col_normalized = {normalize_col(col): col for col in config['columns']}

                            # Rename CSV columns to match config expected case
                            rename_map = {}
                            for config_col in config['columns']:
                                config_normalized = normalize_col(config_col)
                                config_compact = compact_col(config_col)
                                # Find matching CSV column (normalized)
                                if config_normalized in csv_col_normalized:
                                    csv_actual = csv_col_normalized[config_normalized]
                                elif config_compact in csv_col_compact:
                                    csv_actual = csv_col_compact[config_compact]
                                else:
                                    csv_actual = None
                                if csv_actual:
                                    if csv_actual != config_col:
                                        rename_map[csv_actual] = config_col

                            if rename_map:
                                df.rename(columns=rename_map, inplace=True)
                                st.info(f"✅ Normalized {len(rename_map)} column names for case consistency")

                            # Parse date columns AFTER normalization (for serial tables)
                            if 'parse_dates' in config:
                                prefer_month_first_cols = {c.lower() for c in config.get('prefer_month_first_dates', [])}
                                for date_col in config['parse_dates']:
                                    if date_col in df.columns:
                                        try:
                                            prefer_month_first = date_col.lower() in prefer_month_first_cols
                                            df[date_col] = parse_mixed_date_series(df[date_col], prefer_month_first=prefer_month_first)
                                            parsed_count = df[date_col].notna().sum()
                                            if prefer_month_first:
                                                st.info(f"📅 Parsed {parsed_count:,} dates in column '{date_col}' (MM/DD priority)")
                                            else:
                                                st.info(f"📅 Parsed {parsed_count:,} dates in column '{date_col}'")
                                        except Exception as e:
                                            st.warning(f"⚠️ Could not parse dates in column '{date_col}': {str(e)}")

                            # Auto-populate LOADED_DATETIME if missing or NULL (for serial_no_dailydata)
                            if selected_table == "serial_no_dailydata" and 'LOADED_DATETIME' in df.columns:
                                # Fill empty/NULL values with current timestamp
                                null_count = df['LOADED_DATETIME'].isna().sum()
                                if null_count > 0:
                                    current_timestamp = datetime.now().strftime('%d-%b-%y %I:%M:%S %p')
                                    df['LOADED_DATETIME'].fillna(current_timestamp, inplace=True)
                                    st.info(f"📅 Auto-filled {null_count} NULL LOADED_DATETIME values with upload time: {current_timestamp}")
                        else:
                            # For other tables, normalize to lowercase
                            df.columns = df.columns.str.strip().str.lower()

                        # LVO table-friendly alias normalization
                        if selected_table == "LVO_shopmgrname":
                            lvo_mgr_alias = {
                                'shopname': 'shop_description',
                                'shop_name': 'shop_description',
                                'managername': 'shop_manager_name',
                                'manager_name': 'shop_manager_name',
                            }
                            rename_map = {}
                            for col in df.columns:
                                key = _normalize_header_key(col)
                                if key in lvo_mgr_alias and col != lvo_mgr_alias[key]:
                                    rename_map[col] = lvo_mgr_alias[key]
                            if rename_map:
                                df.rename(columns=rename_map, inplace=True)

                        if selected_table == "LVO_offloading_loading_staging":
                            lvo_stage_alias = {
                                'shopdescription': 'shop_name',
                                'shopdesc': 'shop_name',
                            }
                            rename_map = {}
                            for col in df.columns:
                                key = _normalize_header_key(col)
                                if key in lvo_stage_alias and col != lvo_stage_alias[key]:
                                    rename_map[col] = lvo_stage_alias[key]
                            if rename_map:
                                df.rename(columns=rename_map, inplace=True)

                        if selected_table == "LVO_offloading_vs_loading" and not lvo_multi_prepared:
                            df = _prepare_lvo_offloading_df(df, file_name)

                        # Column mapping for common variations (case-insensitive)
                        column_variations = {
                            'vc_item_code': ['vc_item_c', 'item_code', 'itemcode', 'vc_item_code'],
                            'wh_code': ['wh_code', 'm_code', 'shop_code', 'warehouse_code'],
                            'wh_name': ['wh_name', 'warehouse_name', 'shop_name', 'location'],
                            'balance_qty': ['balance_qty', 'balance_q', 'qty', 'quantity', 'stock_qty']
                        }

                        # Additional mappings for serial number uploads
                        serial_variations = {
                            'serial_no': ['serial_no', 'vc_serail_no', 'vc_serail_no', 'vc_serial_no', 'shop_serail_no', 'vc_serial', 'serialno'],
                            'vc_item_code': ['vc_item_code', 'vc_item', 'item_code', 'vc_itemcode'],
                            'wh_code': ['vc_wh_code', 'vc_wh_code', 'wh_code', 'vc_shop_code', 'vc_warehouse_code'],
                            'received_date': ['loaded_datetime', 'loadingdate', 'dt_load_date', 'dt_doc_date', 'dt_invoice_date', 'loaded_date']
                        }

                        # Apply column mapping if table is GEN_whstock
                        if selected_table == "GEN_whstock":
                            for target_col, variations in column_variations.items():
                                for var in variations:
                                    if var in df.columns:
                                        if var != target_col:
                                            df.rename(columns={var: target_col}, inplace=True)
                                        break

                        # Apply serial mappings and ensure optional columns exist (only for non-serial_no_dailydata)
                        if selected_table == "whreceived_serialno":
                            for target_col, variations in serial_variations.items():
                                for var in variations:
                                    if var in df.columns:
                                        if var != target_col:
                                            df.rename(columns={var: target_col}, inplace=True)
                                        break

                            # Ensure optional columns exist so validation won't fail
                            for opt_col in ('source', 'note'):
                                if opt_col not in df.columns:
                                    df[opt_col] = ''

                        st.success(f"✅ File loaded: **{len(df):,}** rows")

                        # Validate columns (case-sensitive for serial tables, case-insensitive for others)
                        if selected_table in ["serial_no_dailydata", "whreceived_serialno", "serialno_check_yes_no"]:
                            # Keep original case for validation
                            config_columns = config['columns']
                            df_columns = df.columns.tolist()

                            # Normalized validation (handles spaces, underscores, missing separators, typos)
                            def normalize_col(col):
                                normalized = str(col).lower().strip()
                                normalized = normalized.replace('coode', 'code')
                                normalized = re.sub(r'[^a-z0-9]+', '_', normalized).strip('_')
                                return normalized

                            def compact_col(col):
                                compact = str(col).lower().strip().replace('coode', 'code')
                                compact = re.sub(r'[^a-z0-9]+', '', compact)
                                return compact

                            config_normalized = {normalize_col(col): col for col in config_columns}
                            df_normalized = {normalize_col(col): col for col in df_columns}
                            config_compact = {compact_col(col): col for col in config_columns}
                            df_compact = {compact_col(col): col for col in df_columns}

                            missing_cols = set(config_compact.keys()) - set(df_compact.keys())
                            extra_cols = set(df_compact.keys()) - set(config_compact.keys())

                            # Convert back to actual column names for display
                            missing_cols_display = [config_compact[col] for col in missing_cols]
                            extra_cols_display = [df_compact[col] for col in extra_cols]

                            # Create config_columns_lower for upload (only include columns from config, in config order)
                            config_columns_lower = []
                            for config_col in config_columns:
                                # Find matching column in DataFrame (normalized)
                                config_norm = normalize_col(config_col)
                                config_comp = compact_col(config_col)
                                if config_norm in df_normalized:
                                    config_columns_lower.append(df_normalized[config_norm])
                                elif config_comp in df_compact:
                                    config_columns_lower.append(df_compact[config_comp])
                                else:
                                    # Fallback: try to find by case-insensitive match
                                    for df_col in df_columns:
                                        if df_col.lower().strip() == config_col.lower().strip():
                                            config_columns_lower.append(df_col)
                                            break
                        else:
                            # Update config columns to lowercase for validation and upload
                            config_columns_lower = [col.lower() for col in config['columns']]

                            # Validate columns (check lowercase versions)
                            missing_cols_display = set(config_columns_lower) - set(df.columns)
                            extra_cols_display = set(df.columns) - set(config_columns_lower)
                            missing_cols = missing_cols_display
                            extra_cols = extra_cols_display

                        if missing_cols:
                            st.error(f"❌ Missing required columns: {', '.join(missing_cols_display)}")
                            st.info(f"📋 Available columns in CSV: {', '.join(sorted(df.columns))}")
                        else:
                            st.success(f"✅ All required columns present")

                        if extra_cols_display:
                            st.warning(f"⚠️ Extra columns (will be ignored): {', '.join(extra_cols_display)}")

                        # Partition table validation for GEN_sales
                        if selected_table == "GEN_sales" and config.get('is_partitioned'):
                            if 'date_invoice' in df.columns:
                                df['date_invoice'] = pd.to_datetime(df['date_invoice'])
                                min_date = df['date_invoice'].min()
                                max_date = df['date_invoice'].max()

                                # Get unique months in data
                                df['_month'] = df['date_invoice'].dt.to_period('M')
                                unique_months = df['_month'].unique()
                                month_counts = df['_month'].value_counts().sort_index()

                                st.info(f"📅 **Partitioned Table:** Data will be auto-routed to monthly partitions (e.g., sales_dec2025)")
                                st.success(f"✅ Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

                                if len(unique_months) > 1:
                                    st.info(f"📊 **Multiple months detected:** Data spans {len(unique_months)} months")
                                    for month in month_counts.index:
                                        count = month_counts[month]
                                        month_str = month.strftime('%Y-%m')
                                        partition_name = f"sales_{month.strftime('%b%Y').lower()}"
                                        st.write(f"   • {month_str}: {count:,} rows → `{partition_name}`")
                                else:
                                    month_str = unique_months[0].strftime('%Y-%m')
                                    partition_name = f"sales_{unique_months[0].strftime('%b%Y').lower()}"
                                    st.success(f"✅ Single month: {month_str} → partition `{partition_name}`")

                                df.drop('_month', axis=1, inplace=True)

                                # Check if data goes to end of month
                                last_day_of_month = (max_date.replace(day=28) + pd.Timedelta(days=4)).replace(day=1) - pd.Timedelta(days=1)
                                if max_date.date() < last_day_of_month.date():
                                    st.warning(f"⚠️ **Incomplete month data:** Latest date is {max_date.strftime('%Y-%m-%d')}, month ends on {last_day_of_month.strftime('%Y-%m-%d')}")

                        # Show preview
                        with st.expander("👁️ Preview Data (first 10 rows)", expanded=False):
                            st.dataframe(df.head(10), use_container_width=True)

                        # Upload controls
                        st.markdown("---")

                        col1, col2, col3 = st.columns([1, 2, 1])

                        with col1:
                            if st.session_state.upload_in_progress:
                                if st.button("🛑 CANCEL UPLOAD", type="secondary", use_container_width=True):
                                    cancel_upload()
                                    st.warning("⚠️ Upload cancelled by user")
                                    st.rerun()

                        can_upload = not missing_cols and not st.session_state.upload_in_progress
                        with col2:
                            upload_btn = st.button(
                                "🚀 START DATA UPLOAD",
                                type="primary",
                                use_container_width=True,
                                disabled=not can_upload
                            )

                        with col3:
                            st.markdown("")  # Spacer

                        if missing_cols:
                            st.warning("Upload disabled until all required columns are present.")

                        if upload_btn and not st.session_state.upload_in_progress:
                            st.session_state.upload_cancelled = False
                            st.session_state.upload_in_progress = True

                            # Create containers for better visibility
                            progress_container = st.container()
                            status_container = st.container()
                            details_container = st.container()

                            with progress_container:
                                progress_bar = st.progress(0)
                                status_text = st.empty()

                            with details_container:
                                details_expander = st.expander("📋 Detailed Log", expanded=True)

                            dropped_indexes = []
                            created_indexes = []
                            backed_up = 0
                            uploaded = 0

                            try:
                                # Connect to database (use specific database if configured)
                                status_text.text("🔗 Connecting to database...")
                                progress_bar.progress(5)
                                db_config = DB_CONFIG.copy()
                                target_database = config.get('database', 'salesdata')
                                db_config['database'] = target_database
                                conn = psycopg2.connect(**db_config)
                                st.session_state.current_connection = conn
                                status_container.success(f"✅ Connected to database: {target_database}")

                                # Get target table name (may differ from config key)
                                target_table = config.get('target_table', selected_table)

                                if st.session_state.upload_cancelled:
                                    raise Exception("Upload cancelled by user")

                                # Step 1: Drop indexes
                                status_text.text("🔽 Dropping indexes...")
                                progress_bar.progress(15)
                                dropped_indexes = drop_indexes(conn, config, status_container)
                                status_container.success(f"✅ Dropped {len(dropped_indexes)} indexes")

                                if st.session_state.upload_cancelled:
                                    raise Exception("Upload cancelled by user")

                                # Step 2: Backup existing data
                                upload_mode = config.get('upload_mode', 'incremental')
                                mode_text = "APPEND" if upload_mode == "append" else ("TRUNCATE" if upload_mode == "truncate" else "INCREMENTAL")
                                status_text.text(f"💾 Backing up data ({mode_text} mode)...")
                                progress_bar.progress(35)
                                backed_up = backup_existing_data(conn, target_table, config['backup_table'], upload_mode, status_container, config.get('date_column'))
                                if upload_mode == "truncate":
                                    status_container.success(f"✅ Backed up {backed_up:,} rows and truncated table")
                                elif upload_mode == "append":
                                    status_container.success(f"✅ Append mode: Ready to upload")
                                else:
                                    status_container.success(f"✅ Backed up {backed_up:,} rows")

                                if st.session_state.upload_cancelled:
                                    raise Exception("Upload cancelled by user")

                                # Step 3: Upload new data
                                status_text.text("📤 Uploading data...")
                                progress_bar.progress(55)

                                # Use lowercase column names for selection (after normalization)
                                df_upload = df[config_columns_lower].copy()

                                # For serial tables, convert DataFrame column names to lowercase to match table schema
                                if selected_table in ['serial_no_dailydata', 'whreceived_serialno', 'serialno_check_yes_no']:
                                    df_upload.columns = [col.lower() for col in df_upload.columns]

                                # Apply column mapping if configured
                                if 'column_mapping' in config:
                                    df_upload = df_upload.rename(columns=config['column_mapping'])
                                    mapped_columns = [config['column_mapping'].get(col, col) for col in config['columns']]
                                else:
                                    mapped_columns = config['columns']

                                # For serial tables, ensure mapped_columns are also lowercase
                                if selected_table in ['serial_no_dailydata', 'whreceived_serialno', 'serialno_check_yes_no']:
                                    mapped_columns = [col.lower() for col in mapped_columns]

                                # Parse date columns for serial tables before upload
                                if selected_table in ['serial_no_dailydata', 'whreceived_serialno', 'serialno_check_yes_no'] and 'parse_dates' in config:
                                    dayfirst = config.get('date_parse_dayfirst', True)
                                    prefer_month_first_cols = {c.lower() for c in config.get('prefer_month_first_dates', [])}
                                    # force_day_first_cols: pandas ignores dayfirst=True for ambiguous dates
                                    # (e.g. 01/05 where both ≤12). Use explicit DD/MM/YYYY format instead.
                                    force_day_first_cols = {c.lower() for c in config.get('prefer_day_first_dates', [])}
                                    for date_col in config['parse_dates']:
                                        col_lower = date_col.lower()
                                        if col_lower in df_upload.columns:
                                            try:
                                                prefer_month_first = col_lower in prefer_month_first_cols
                                                force_day_first    = col_lower in force_day_first_cols
                                                if prefer_month_first:
                                                    df_upload[col_lower] = parse_mixed_date_series(df_upload[col_lower], prefer_month_first=True)
                                                elif force_day_first:
                                                    # Pandas silently ignores dayfirst=True for ambiguous dates.
                                                    # Explicitly try DD/MM/YYYY first, then fall back to dayfirst=True.
                                                    s = df_upload[col_lower].astype(str).str.strip()
                                                    parsed = pd.to_datetime(s, format='%d/%m/%Y', errors='coerce')
                                                    still_null = parsed.isna() & s.notna() & (s != 'nan') & (s != 'NaT')
                                                    if still_null.any():
                                                        parsed[still_null] = pd.to_datetime(s[still_null], dayfirst=True, errors='coerce')
                                                    df_upload[col_lower] = parsed
                                                else:
                                                    df_upload[col_lower] = pd.to_datetime(df_upload[col_lower], dayfirst=dayfirst, errors='coerce')
                                                # Convert to string format for PostgreSQL (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
                                                if col_lower == 'loaded_datetime':
                                                    df_upload[col_lower] = df_upload[col_lower].dt.strftime('%Y-%m-%d %H:%M:%S')
                                                else:
                                                    df_upload[col_lower] = df_upload[col_lower].dt.strftime('%Y-%m-%d')
                                                if prefer_month_first:
                                                    status_container.info(f"✅ Parsed date column '{col_lower}' (MM/DD priority)")
                                                elif force_day_first:
                                                    status_container.info(f"✅ Parsed date column '{col_lower}' (DD/MM forced)")
                                                else:
                                                    status_container.info(f"✅ Parsed date column '{col_lower}'")
                                            except Exception as date_err:
                                                status_container.warning(f"⚠️ Could not parse date column '{col_lower}': {str(date_err)[:100]}")

                                # Remove rows with all null values or empty rows
                                rows_before = len(df_upload)
                                df_upload = df_upload.dropna(how='all')  # Drop rows where all columns are null
                                df_upload = df_upload[df_upload.astype(str).ne('').any(axis=1)]  # Drop rows with all empty strings
                                rows_after = len(df_upload)
                                if rows_before > rows_after:
                                    status_container.info(f"ℹ️ Skipped {rows_before - rows_after:,} empty rows")

                                # Apply data type conversions if configured (case-insensitive to match normalized columns)
                                if 'type_conversions' in config:
                                    for col, dtype in config['type_conversions'].items():
                                        matching_cols = [c for c in df_upload.columns if c.lower() == col.lower()]
                                        for actual_col in matching_cols:
                                            numeric_series = pd.to_numeric(df_upload[actual_col], errors='coerce')
                                            if dtype == 'int':
                                                df_upload[actual_col] = numeric_series.fillna(0).astype(int)
                                            elif dtype == 'float':
                                                df_upload[actual_col] = numeric_series

                                # Check if table is partitioned
                                is_partitioned = config.get('is_partitioned', False)
                                add_upload_date = config.get('add_upload_date', False)

                                # Note: For serial_no_dailydata, UPSERT logic handles updates without pre-deletion
                                # CSV data will update existing serial_no records and insert new ones

                                uploaded = upload_data_to_table(
                                    conn,
                                    target_table,
                                    df_upload,
                                    mapped_columns,
                                    status_container,
                                    is_partitioned,
                                    target_database,
                                    add_upload_date,
                                    table_config=config,
                                    source_file_name=file_name,
                                )
                                status_container.success(f"✅ Uploaded {uploaded:,} rows")

                                if st.session_state.upload_cancelled:
                                    raise Exception("Upload cancelled by user")

                                # Step 4: Recreate indexes
                                status_text.text("🔼 Creating indexes...")
                                progress_bar.progress(75)
                                created_indexes = create_indexes(conn, config, status_container)
                                status_container.success(f"✅ Created {len(created_indexes)} indexes")

                                if st.session_state.upload_cancelled:
                                    raise Exception("Upload cancelled by user")

                                # Step 4.5: Refresh materialized views if configured
                                if 'refresh_views' in config and config['refresh_views']:
                                    status_text.text("🔄 Refreshing materialized views...")
                                    progress_bar.progress(85)
                                    try:
                                        cursor = conn.cursor()
                                        for view_name in config['refresh_views']:
                                            status_container.info(f"🔄 Refreshing {view_name}...")
                                            cursor.execute(f"REFRESH MATERIALIZED VIEW {view_name}")
                                            conn.commit()
                                        cursor.close()
                                        status_container.success(f"✅ Refreshed {len(config['refresh_views'])} materialized views")
                                    except Exception as e:
                                        status_container.error(f"❌ View refresh failed: {str(e)}")
                                        raise e

                                    # Step 4.6: Run post-upload SQL if configured
                                if 'post_upload_sql' in config:
                                    status_text.text("🔧 Running post-upload commands...")
                                    progress_bar.progress(90)
                                    try:
                                        cursor = conn.cursor()

                                        # Smart splitter: split by ; but never inside $$...$$
                                        def _split_sql(sql):
                                            stmts, buf, depth = [], [], 0
                                            i = 0
                                            while i < len(sql):
                                                if sql[i] == '$' and not depth:
                                                    j = i + 1
                                                    while j < len(sql) and (sql[j].isalnum() or sql[j] == '_'):
                                                        j += 1
                                                    if j < len(sql) and sql[j] == '$':
                                                        tag = sql[i:j+1]
                                                        buf.append(tag); i = j + 1; depth += 1
                                                        continue
                                                elif depth and sql[i] == '$':
                                                    j = i + 1
                                                    while j < len(sql) and (sql[j].isalnum() or sql[j] == '_'):
                                                        j += 1
                                                    if j < len(sql) and sql[j] == '$':
                                                        tag = sql[i:j+1]
                                                        buf.append(tag); i = j + 1; depth -= 1
                                                        continue
                                                if sql[i] == ';' and not depth:
                                                    s = ''.join(buf).strip()
                                                    if s:
                                                        stmts.append(s)
                                                    buf = []
                                                else:
                                                    buf.append(sql[i])
                                                i += 1
                                            s = ''.join(buf).strip()
                                            if s:
                                                stmts.append(s)
                                            return stmts

                                        for sql_cmd in _split_sql(config['post_upload_sql']):
                                            cursor.execute(sql_cmd)
                                            conn.commit()
                                        cursor.close()
                                        status_container.success("✅ Post-upload commands executed")
                                    except Exception as e:
                                        status_container.warning(f"⚠️ Post-upload command warning: {str(e)}")

                                        # Step 4.7: Compare CSV staging data to uploaded rows in serial_no_dailydata
                                        if target_table == 'serial_no_dailydata':
                                            try:
                                                compare_cursor = conn.cursor()
                                                compare_cursor.execute(
                                                    """
                                                    WITH loading_range AS (
                                                        SELECT
                                                            MIN(loadingdate) AS min_date,
                                                            MAX(loadingdate) AS max_date
                                                        FROM serial_no_dailydata_staging
                                                        WHERE loadingdate IS NOT NULL
                                                    )
                                                    SELECT
                                                        (SELECT COUNT(*) FROM serial_no_dailydata_staging) AS staging_count,
                                                        (SELECT COUNT(*) FROM serial_no_dailydata
                                                         WHERE loadingdate >= loading_range.min_date
                                                           AND loadingdate <= loading_range.max_date) AS main_range_count,
                                                        loading_range.min_date,
                                                        loading_range.max_date
                                                    FROM loading_range;
                                                    """
                                                )
                                                staging_count, main_range_count, min_date, max_date = compare_cursor.fetchone()
                                                compare_cursor.close()

                                                if min_date is not None and max_date is not None:
                                                    status_container.info(
                                                        f"📌 LOADINGDATE range compared: {min_date} to {max_date}. "
                                                        f"CSV rows = {staging_count:,}, main table rows in range = {main_range_count:,}."
                                                    )
                                                    if staging_count == main_range_count:
                                                        status_container.success("✅ CSV staging row count matches uploaded rows in the target LOADINGDATE range.")
                                                    else:
                                                        status_container.warning(
                                                            f"⚠️ Row count mismatch: CSV staging has {staging_count:,} rows, "
                                                            f"but {main_range_count:,} rows exist in serial_no_dailydata for the same LOADINGDATE range."
                                                        )
                                                else:
                                                    status_container.warning("⚠️ Could not determine LOADINGDATE range from CSV staging data for comparison.")
                                            except Exception as e:
                                                status_container.warning(f"⚠️ Could not compare CSV staging data with serial_no_dailydata: {str(e)[:120]}")

                                        # Step 4.8: Ensure uploaded_data_date backfilled for serial_no_dailydata
                                final_count = get_table_count(target_table, target_database)

                                # Complete
                                progress_bar.progress(100)
                                status_text.text("✅ Upload complete!")

                                conn.close()
                                st.session_state.current_connection = None
                                st.session_state.upload_in_progress = False

                                # Summary
                                st.markdown("---")
                                st.balloons()
                                st.success("🎉 **Data Upload Complete!**")

                                # Metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Backed Up", f"{backed_up:,}")
                                with col2:
                                    st.metric("Uploaded", f"{uploaded:,}")
                                with col3:
                                    st.metric("Total in Table", f"{final_count:,}")

                                # Show what was dropped and created
                                with st.expander("🔍 Index Operations Summary", expanded=True):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**🔽 Dropped Indexes:**")
                                        for idx_name in dropped_indexes:
                                            st.markdown(f"- `{idx_name}`")
                                    with col2:
                                        st.markdown("**🔼 Created Indexes:**")
                                        for idx_name in created_indexes:
                                            st.markdown(f"- `{idx_name}`")

                            except Exception as e:
                                st.session_state.upload_in_progress = False
                                st.session_state.current_connection = None
                                st.error(f"❌ Upload failed: {e}")
                                details_expander.error(f"❌ Error: {e}")

                                # Show partial results
                                if dropped_indexes or created_indexes:
                                    with st.expander("⚠️ Partial Operations", expanded=True):
                                        if dropped_indexes:
                                            st.markdown(f"**Dropped:** {', '.join([f'`{i}`' for i in dropped_indexes])}")
                                        if created_indexes:
                                            st.markdown(f"**Created:** {', '.join([f'`{i}`' for i in created_indexes])}")

                                try:
                                    if conn:
                                        conn.close()
                                except:
                                    pass

                    except Exception as e:
                        st.error(f"❌ Error reading file: {e}")

            # ===========================
            # CENTURY STOCKOUT UPSERT AUTOMATION
            # ===========================
            st.markdown("---")
            st.markdown("### 🔄 External DB sync for (Mysql/PHpadmin - postgresSQL) Conumable till Dashbaord data upload")
            st.info("Pull data from MySQL tables ALERTS, ERPDATA, INVOICES, invoices_manager into PostgreSQL WH with table auto-create, incremental sync, snapshot reconciliation, dedupe, and audit logging.")

            with st.expander("⚙️ Configure Source & Run Sync", expanded=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    mysql_host = st.text_input("MySQL Host", value="192.168.0.17", key="mysql_sync_host")
                    mysql_user = st.text_input("MySQL Username", value="misaccount", key="mysql_sync_user")
                with c2:
                    mysql_password = st.text_input("MySQL Password", value="Inv@Central@2024", type="password", key="mysql_sync_password")
                    mysql_database = st.text_input("MySQL Database Name", value="", placeholder="Enter source database name", key="mysql_sync_db")
                with c3:
                    sync_batch_size = st.number_input("Batch Size", min_value=1000, max_value=100000, value=20000, step=1000)
                    sync_verify_days = st.number_input("Post-Sync Verify Days", min_value=1, max_value=30, value=7, step=1)
                    selected_sync_tables = st.multiselect(
                        "Tables to Sync",
                        options=["ALERTS", "ERPDATA", "INVOICES", "invoices_manager"],
                        default=["ALERTS", "ERPDATA", "INVOICES", "invoices_manager"],
                        key="mysql_sync_tables"
                    )

                st.caption("Target is fixed to PostgreSQL WH: localhost:3307 / postgres / WH")

                if st.button("🚀 Run MySQL → PostgreSQL WH Sync", type="primary", use_container_width=True):
                    if not mysql_database.strip():
                        st.error("❌ Please provide MySQL Database Name.")
                    elif not selected_sync_tables:
                        st.error("❌ Please select at least one table to sync.")
                    else:
                        sync_status_box = st.container()
                        with st.spinner("Running cross-database sync..."):
                            try:
                                mysql_cfg = {
                                    'host': mysql_host.strip(),
                                    'user': mysql_user.strip(),
                                    'password': mysql_password,
                                    'database': mysql_database.strip()
                                }

                                pg_cfg = {
                                    'host': 'localhost',
                                    'port': 3307,
                                    'user': 'postgres',
                                    'password': 'hello',
                                    'database': 'WH'
                                }

                                sync_results = sync_mysql_tables_to_postgres(
                                    mysql_config=mysql_cfg,
                                    pg_config=pg_cfg,
                                    source_tables=selected_sync_tables,
                                    status_container=sync_status_box,
                                    batch_size=int(sync_batch_size)
                                )

                                result_df = pd.DataFrame(sync_results)

                                if result_df.empty:
                                    st.warning("No sync results returned.")
                                else:
                                    success_count = int((result_df['status'] == 'success').sum())
                                    failed_count = int((result_df['status'] == 'failed').sum())
                                    total_fetched = int(result_df['fetched_rows'].fillna(0).sum())
                                    total_inserted = int(result_df['inserted_rows'].fillna(0).sum())
                                    total_duplicates = int(result_df['duplicate_rows'].fillna(0).sum())
                                    total_deleted = int(result_df['deleted_rows'].fillna(0).sum())
                                    total_target_rows = int(result_df['target_total_rows'].fillna(0).sum()) if 'target_total_rows' in result_df.columns else 0

                                    st.markdown("#### ✅ Sync Execution Summary")
                                    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
                                    m1.metric("Tables Success", success_count)
                                    m2.metric("Tables Failed", failed_count)
                                    m3.metric("Rows Fetched", f"{total_fetched:,}")
                                    m4.metric("Rows Inserted", f"{total_inserted:,}")
                                    m5.metric("Duplicates Skipped", f"{total_duplicates:,}")
                                    m6.metric("Stale Deleted", f"{total_deleted:,}", help="Snapshot reconciliation: removed rows that no longer exist in MySQL")
                                    m7.metric("Rows In Target", f"{total_target_rows:,}", help="Current total rows in PostgreSQL target tables after sync")

                                    display_cols = [
                                        'source_table', 'target_table', 'strategy',
                                        'fetched_rows', 'inserted_rows', 'duplicate_rows', 'deleted_rows',
                                        'source_total_rows', 'target_total_rows', 'sync_note',
                                        'status', 'error', 'started_at', 'finished_at'
                                    ]
                                    existing_cols = [c for c in display_cols if c in result_df.columns]

                                    st.markdown("#### 📋 Per-Table Detailed Result")
                                    st.dataframe(result_df[existing_cols], use_container_width=True)

                                    st.markdown("#### 🧪 Post-Sync Verification (Date + Shop Counts)")
                                    verify_summary_df, verify_detail_df = verify_mysql_postgres_sync_by_day_shop(
                                        mysql_config=mysql_cfg,
                                        pg_config=pg_cfg,
                                        source_tables=selected_sync_tables,
                                        lookback_days=int(sync_verify_days)
                                    )

                                    if verify_summary_df.empty:
                                        st.warning("No verification summary returned.")
                                    else:
                                        st.dataframe(verify_summary_df, use_container_width=True)

                                    if verify_detail_df.empty:
                                        st.success("✅ Post-sync verification passed: no date/shop count mismatches found.")
                                    else:
                                        st.error(f"❌ Found {len(verify_detail_df):,} date/shop mismatch groups. Review details below.")
                                        st.dataframe(verify_detail_df, use_container_width=True)

                                        st.markdown("#### 🧾 Offloading vs Loading (delta > 0) Dedup Preview")
                                        delta_preview_df = build_offloading_positive_delta_preview(pg_cfg, verify_detail_df)
                                        if delta_preview_df.empty:
                                            st.info("No Offloading vs Loading rows with delta > 0 found for append.")
                                        else:
                                            new_count = int((delta_preview_df['dedup_status'] == 'NEW').sum())
                                            dup_count = int((delta_preview_df['dedup_status'] == 'ALREADY_EXISTS').sum())
                                            c1, c2, c3 = st.columns(3)
                                            c1.metric("Rows in Preview", f"{len(delta_preview_df):,}")
                                            c2.metric("NEW (Will Insert)", f"{new_count:,}")
                                            c3.metric("Already Exists", f"{dup_count:,}")

                                            st.dataframe(delta_preview_df, use_container_width=True)

                                            if new_count > 0:
                                                if st.button("✅ Confirm Append NEW Offloading Delta Rows", key="confirm_append_offloading_delta_rows"):
                                                    appended_rows = append_confirmed_offloading_deltas(pg_cfg, delta_preview_df)
                                                    if appended_rows > 0:
                                                        st.success(f"📝 Inserted {appended_rows:,} NEW row(s) into external_sync_delta_log.")
                                                    else:
                                                        st.warning("No rows inserted. Please retry or check DB connection.")
                                            else:
                                                st.info("All preview rows already exist in PostgreSQL log. Nothing new to insert.")

                                        st.markdown("#### 📥 Invoices* Positive Delta Backfill (MySQL → PostgreSQL)")
                                        invoices_preview_df = build_invoices_positive_delta_group_preview(pg_cfg, verify_detail_df)
                                        if invoices_preview_df.empty:
                                            st.info("No invoices* groups with delta > 0 found for backfill.")
                                        else:
                                            inv_new = int((invoices_preview_df['dedup_status'] == 'NEW_GROUP').sum())
                                            inv_old = int((invoices_preview_df['dedup_status'] == 'ALREADY_REPAIRED').sum())
                                            i1, i2, i3 = st.columns(3)
                                            i1.metric("Invoices Groups in Preview", f"{len(invoices_preview_df):,}")
                                            i2.metric("NEW_GROUP (Will Backfill)", f"{inv_new:,}")
                                            i3.metric("Already Repaired", f"{inv_old:,}")

                                            st.dataframe(invoices_preview_df, use_container_width=True)

                                            if inv_new > 0:
                                                if st.button("✅ Confirm Fetch Missing Invoices Rows (delta > 0)", key="confirm_fetch_missing_invoices_delta"):
                                                    repair_result = repair_invoices_positive_delta_groups(
                                                        mysql_config=mysql_cfg,
                                                        pg_config=pg_cfg,
                                                        preview_df=invoices_preview_df,
                                                        batch_size=int(sync_batch_size),
                                                    )
                                                    st.success(
                                                        f"Invoices backfill done · groups processed: {repair_result['groups_processed']:,}/{repair_result['groups_total']:,}, "
                                                        f"rows fetched: {repair_result['rows_fetched']:,}, rows inserted: {repair_result['rows_inserted']:,}."
                                                    )
                                                    if repair_result['groups_failed'] > 0:
                                                        st.warning(f"{repair_result['groups_failed']:,} group(s) failed. Check external_sync_repair_log for details.")
                                            else:
                                                st.info("All invoices* positive-delta groups were already repaired earlier.")

                                    with st.expander("🧠 DBA Notes / What Happened", expanded=True):
                                        st.markdown("- Source tables were introspected from MySQL schema.")
                                        st.markdown("- Destination tables were created automatically in PostgreSQL WH if missing.")
                                        st.markdown("- Sync uses append-only date logic: pull only MySQL rows with date greater than PostgreSQL max date.")
                                        st.markdown("- No stale-row deletion or source-side reconciliation is performed in this flow.")
                                        st.markdown("- Duplicate prevention is applied via PK conflict handling and source-row hash uniqueness.")
                                        st.markdown("- Post-sync verification compares MySQL vs PostgreSQL counts grouped by date and shop.")
                                        st.markdown("- Sync state and execution logs were written to `external_sync_state` and `external_sync_log`.")

                                    if failed_count > 0:
                                        st.error("Some tables failed. Check the Error column in the detailed report.")
                                    else:
                                        st.success("🎉 All selected tables synchronized successfully.")

                                    # ── Auto-refresh WH materialized views after sync ──
                                    WH_VIEWS_TO_REFRESH = [
                                        "mv_wh_alerts_daily",
                                        "mv_wh_erp_cashier_sessions_daily",
                                        "mv_wh_erp_daily",
                                        "mv_wh_erp_test_bills_cashier_daily",
                                        "mv_wh_invoices_agg_daily",
                                        "mv_wh_manager_handover_daily",
                                    ]
                                    st.markdown("#### 🔄 Refreshing WH Materialized Views")
                                    mv_results = []
                                    try:
                                        mv_conn = psycopg2.connect(
                                            host=pg_cfg['host'], port=pg_cfg['port'],
                                            user=pg_cfg['user'], password=pg_cfg['password'],
                                            dbname=pg_cfg['database']
                                        )
                                        mv_cur = mv_conn.cursor()
                                        for view_name in WH_VIEWS_TO_REFRESH:
                                            try:
                                                mv_cur.execute(f"REFRESH MATERIALIZED VIEW {view_name}")
                                                mv_conn.commit()
                                                mv_results.append({"view": view_name, "status": "✅ refreshed"})
                                            except Exception as ve:
                                                mv_conn.rollback()
                                                mv_results.append({"view": view_name, "status": f"⚠️ {ve}"})
                                        mv_cur.close()
                                        mv_conn.close()
                                    except Exception as mv_err:
                                        st.error(f"❌ Could not connect for MV refresh: {mv_err}")
                                        mv_results = []

                                    if mv_results:
                                        ok = sum(1 for r in mv_results if r["status"].startswith("✅"))
                                        st.dataframe(mv_results, use_container_width=True)
                                        if ok == len(mv_results):
                                            st.success(f"✅ All {ok} WH materialized views refreshed.")
                                        else:
                                            st.warning(f"⚠️ {ok}/{len(mv_results)} views refreshed — see table above for errors.")

                            except Exception as sync_error:
                                st.error(f"❌ Sync failed: {sync_error}")

            st.markdown("---")
            st.markdown("### 🔁 Delete & Re-Fetch from Date")
            st.caption("Permanently deletes PostgreSQL WH rows from the chosen date onwards and re-imports fresh from MySQL.")
            with st.expander("⚙️ Configure & Run Date Reset", expanded=False):
                st.warning("⚠️ **Destructive — irreversible.** Rows deleted from PostgreSQL cannot be recovered from this tool. Confirm MySQL source is accessible before proceeding.")
                rf_c1, rf_c2 = st.columns(2)
                with rf_c1:
                    refetch_from_date = st.date_input(
                        "Re-fetch from date (inclusive)", value=date(2026, 4, 1), key="refetch_from_date"
                    )
                    refetch_tables = st.multiselect(
                        "Tables to reset",
                        options=["ALERTS", "ERPDATA", "INVOICES", "invoices_manager"],
                        default=["ALERTS", "ERPDATA", "INVOICES", "invoices_manager"],
                        key="refetch_tables",
                    )
                with rf_c2:
                    rf_mysql_host     = st.text_input("MySQL Host",     value="192.168.0.17",      key="rf_mysql_host")
                    rf_mysql_user     = st.text_input("MySQL User",     value="misaccount",         key="rf_mysql_user")
                    rf_mysql_password = st.text_input("MySQL Password", type="password",
                                                      value="Inv@Central@2024",                     key="rf_mysql_password")
                    rf_mysql_db       = st.text_input("MySQL Database", value="",
                                                      placeholder="e.g. invcentral",                key="rf_mysql_db")

                pg_cfg_rf = {'host': 'localhost', 'port': 3307, 'user': 'postgres',
                             'password': 'hello', 'database': 'WH'}
                RF_PG_DATE_COLS = {
                    'alerts': 'a_entrytime', 'erpdata': 'invdate',
                    'invoices': 'invdate',   'invoices_manager': 'invdate',
                }

                if st.button("🔍 Preview Rows to Delete", key="btn_preview_delete"):
                    preview_rows = []
                    try:
                        with psycopg2.connect(**pg_cfg_rf) as _pconn:
                            with _pconn.cursor() as _pcur:
                                for _tbl in (refetch_tables or []):
                                    _col = RF_PG_DATE_COLS.get(_tbl.lower(), 'invdate')
                                    try:
                                        _pcur.execute(
                                            f"SELECT COUNT(*) FROM {_quote_ident(_tbl.lower())} "
                                            f"WHERE {_quote_ident(_col)}::date >= %s",
                                            (str(refetch_from_date),)
                                        )
                                        _cnt = _pcur.fetchone()[0]
                                    except Exception as _te:
                                        _cnt = f"Error: {_te}"
                                    preview_rows.append({
                                        'table': _tbl.lower(), 'date_col': _col,
                                        'rows_that_will_be_deleted': _cnt,
                                    })
                        st.dataframe(preview_rows, use_container_width=True)
                    except Exception as _e:
                        st.error(f"Preview failed: {_e}")

                if st.button("✅ CONFIRM: Delete & Re-Fetch", type="primary", key="btn_confirm_delete_refetch"):
                    if not rf_mysql_db.strip():
                        st.error("❌ MySQL Database Name required.")
                    elif not refetch_tables:
                        st.error("❌ Select at least one table.")
                    else:
                        mysql_cfg_rf = {
                            'host': rf_mysql_host.strip(), 'user': rf_mysql_user.strip(),
                            'password': rf_mysql_password, 'database': rf_mysql_db.strip(),
                        }
                        rf_status_box = st.container()
                        with st.spinner("Deleting and re-fetching data from MySQL..."):
                            try:
                                rf_results = delete_and_refetch_tables_from_date(
                                    mysql_config=mysql_cfg_rf,
                                    pg_config=pg_cfg_rf,
                                    from_date_str=str(refetch_from_date),
                                    tables=refetch_tables,
                                    batch_size=20000,
                                    status_container=rf_status_box,
                                )
                                st.success("✅ Delete & Re-Fetch complete.")

                                # Post step requested by user: normalize invoices duplicate flags.
                                selected_refetch_tables = {str(t).strip().lower() for t in refetch_tables}
                                if 'invoices' in selected_refetch_tables:
                                    try:
                                        updated_dup_rows = normalize_invoices_duplicate_flags(
                                            pg_config=pg_cfg_rf,
                                            status_container=rf_status_box,
                                        )
                                        st.success(
                                            f"✅ Invoices duplicate flags refreshed successfully ({updated_dup_rows:,} rows updated)."
                                        )
                                    except Exception as dup_err:
                                        st.error(f"❌ Duplicate flag refresh failed: {dup_err}")

                                st.dataframe(pd.DataFrame(rf_results), use_container_width=True)
                            except Exception as rf_err:
                                st.error(f"❌ Failed: {rf_err}")

            st.markdown("---")
            st.markdown("### 🎯 Century Stockout Daily UPSERT")
            st.info("Automatically run daily stockout tracking UPSERT for Century Penetration dashboard")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("""
                **What this does:**
                - Updates `mv_century_penetration_test` with current stock status
                - Records stockout dates and calculates days out of stock
                - Appends daily snapshot to `century_stockout_daily_snapshot`
                - Refreshes `mv_stockout_analysis` materialized view
                - Enables 7-day trend chart in Century dashboard
                """)

            with col2:
                # Check if tables exist
                try:
                    conn = psycopg2.connect(**DB_CONFIG, database='century_penetration')
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT COUNT(*) FROM information_schema.tables 
                        WHERE table_name = 'mv_century_penetration_test'
                    """)
                    table_exists = cur.fetchone()[0] > 0

                    if table_exists:
                        cur.execute("SELECT COUNT(*) FROM century_stockout_daily_snapshot")
                        snapshot_count = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(DISTINCT snapshot_date) FROM century_stockout_daily_snapshot")
                        days_tracked = cur.fetchone()[0]
                        st.metric("Snapshots", f"{snapshot_count:,}")
                        st.metric("Days Tracked", days_tracked)

                    cur.close()
                    conn.close()
                except Exception as e:
                    st.warning(f"⚠️ Century tables not found: {str(e)[:100]}")

            # UPSERT button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🚀 Run Daily UPSERT", use_container_width=True, type="primary"):
                    with st.spinner("Running Century Stockout UPSERT..."):
                        try:
                            import subprocess
                            import os

                            # Path to the UPSERT script
                            script_path = os.path.join("centurypenetration", "daily_stockout_upsert.py")

                            if os.path.exists(script_path):
                                # Run the script
                                result = subprocess.run(
                                    ["python", script_path],
                                    capture_output=True,
                                    text=True,
                                    timeout=300  # 5 minutes timeout
                                )

                                if result.returncode == 0:
                                    st.success("✅ UPSERT completed successfully!")

                                    # Parse output for statistics
                                    output_lines = result.stdout.split('\\n')
                                    for line in output_lines:
                                        if "Total Items:" in line or "Current Stockouts:" in line or \
                                           "UPSERT completed:" in line or "Daily snapshot appended:" in line:
                                            st.code(line)

                                    # Show full output in expander
                                    with st.expander("📄 Full Output"):
                                        st.code(result.stdout)
                                else:
                                    st.error(f"❌ UPSERT failed with return code {result.returncode}")
                                    st.code(result.stderr)
                            else:
                                st.error(f"❌ Script not found: {script_path}")
                                st.info("💡 Make sure `centurypenetration/daily_stockout_upsert.py` exists")

                        except subprocess.TimeoutExpired:
                            st.error("❌ UPSERT timed out after 5 minutes")
                        except Exception as e:
                            st.error(f"❌ Error running UPSERT: {e}")

            # Schedule automation tip
            with st.expander("⏰ Schedule Daily Automation"):
                st.markdown("""
                **Option 1: Windows Task Scheduler**
                ```powershell
                # Run batch file: centurypenetration/run_daily_stockout_update.bat
                # Schedule: Daily at 1:00 AM
                ```

                **Option 2: Python Script**
                ```python
                import schedule
                import time
                import subprocess

                def run_upsert():
                    subprocess.run(["python", "centurypenetration/daily_stockout_upsert.py"])

                schedule.every().day.at("01:00").do(run_upsert)

                while True:
                    schedule.run_pending()
                    time.sleep(60)
                ```

                **Option 3: Manual**
                - Click "🚀 Run Daily UPSERT" button above once per day
                - Best time: Morning (8 AM) or after midnight (1 AM)
                """)

        # ===========================
        # TAB 3: INDEX MANAGER
        # ===========================

    elif view == "indexmgr":
        if not st.session_state.is_admin:
            st.warning("🔒 Index management is restricted to administrators.")
        else:
            st.markdown('<h2 style="color:#e6edf3;font-weight:800;margin-bottom:4px;">Index Manager</h2>', unsafe_allow_html=True)
            st.markdown('<p style="color:#8b949e;font-size:14px;margin-bottom:24px;">Create and manage database indexes</p>', unsafe_allow_html=True)
            st.markdown("### 🔧 Index Manager")
            st.info("Create database indexes by pasting SQL queries")

            # Table selection
            selected_table_idx = st.selectbox(
                "Select Table",
                options=list(TABLE_CONFIGS.keys()),
                key="idx_table_select",
                help="Choose the table for index operations"
            )

            if selected_table_idx:
                config = TABLE_CONFIGS[selected_table_idx]
                target_table = config.get('target_table', selected_table_idx)
                target_database = config.get('database', 'salesdata')

                # Show current row count and latest date
                col1, col2 = st.columns(2)

                with col1:
                    current_count = get_table_count(target_table, target_database)
                    if current_count is not None:
                        st.metric("Current Row Count", f"{current_count:,}")

                with col2:
                    if 'date_column' in config:
                        latest_date = get_latest_date(target_table, config['date_column'], target_database)
                        if latest_date:
                            st.metric("Latest Date", latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date))

                st.markdown("---")

                # Query input
                st.markdown("**Paste Index Creation Queries:**")
                st.caption("You can paste multiple CREATE INDEX statements separated by semicolons")

                index_queries = st.text_area(
                    "SQL Queries",
                    height=200,
                    placeholder="CREATE INDEX idx_name ON table_name (column_name);\nCREATE INDEX idx_name2 ON table_name (column1, column2);",
                    help="Paste one or more CREATE INDEX statements"
                )

                if index_queries.strip():
                    # Split by semicolon
                    queries = [q.strip() for q in index_queries.split(';') if q.strip()]
                    st.info(f"📝 Found {len(queries)} query(ies)")

                    # Show preview
                    with st.expander("👁️ Preview Queries", expanded=False):
                        for i, q in enumerate(queries, 1):
                            st.code(q, language='sql')

                    # Execute button
                    col1, col2, col3 = st.columns([1, 2, 1])

                    with col2:
                        if st.button("🚀 Execute Queries", type="primary", use_container_width=True):
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            try:
                                status_text.text("🔗 Connecting to database...")
                                conn = psycopg2.connect(**DB_CONFIG)

                                success_count = 0
                                fail_count = 0

                                for i, query in enumerate(queries, 1):
                                    if st.session_state.upload_cancelled:
                                        break

                                    progress = int((i / len(queries)) * 100)
                                    progress_bar.progress(progress)
                                    status_text.text(f"⚙️ Executing query {i}/{len(queries)}...")

                                    result_container = st.empty()
                                    if execute_index_query(conn, query, result_container):
                                        result_container.success(f"✅ Query {i} executed successfully")
                                        success_count += 1
                                    else:
                                        result_container.error(f"❌ Query {i} failed")
                                        fail_count += 1

                                conn.close()
                                progress_bar.progress(100)
                                status_text.text("✅ All queries processed!")

                                # Summary
                                st.markdown("---")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Queries", len(queries))
                                with col2:
                                    st.metric("✅ Successful", success_count)
                                with col3:
                                    st.metric("❌ Failed", fail_count)

                                if success_count == len(queries):
                                    st.balloons()
                                    st.success("🎉 All indexes created successfully!")

                            except Exception as e:
                                st.error(f"❌ Error: {e}")
                                try:
                                    conn.close()
                                except:
                                    pass


    else:
        st.session_state.portal_view = "home"
        st.rerun()

# Footer
st.markdown(
    '<div style="text-align:center;padding:32px 0 16px 0;">'
    '<p style="color:#484f58;font-size:12px;">© 2025 Melcom Group Limited · Analytics Hub · v3.0</p>'
    '</div>',
    unsafe_allow_html=True
)
