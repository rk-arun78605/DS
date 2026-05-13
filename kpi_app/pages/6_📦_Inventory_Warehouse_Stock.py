"""
Inventory Analysis with Warehouse Stock Dashboard
Analyze understock, overstock, critical, and slow-moving items with warehouse-level stock visibility
"""

import streamlit as st
import pandas as pd
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    DB_CONFIG = {
        'host': 'localhost',
        'port': 3307,
        'user': 'postgres',
        'password': 'hello',
        'database': 'salesdata'
    }
    
    PAGE_TITLE = "Inventory Warehouse Stock Analysis"
    PAGE_ICON = "📦"
    LOGO_URL = "https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg"

# ============================================================
# CUSTOM CSS
# ============================================================

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main .block-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 2rem 3rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    h1, h2, h3 {
        color: #1a202c !important;
        font-weight: 700 !important;
    }
    
    h3 {
        font-size: 22px !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: rgba(255, 255, 255, 0.95) !important;
        text-transform: uppercase;
    }
    
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem 1.4rem !important;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
    }
    
    .dataframe {
        font-size: 13px !important;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .dataframe thead th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600;
        padding: 12px !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
</style>
"""

# ============================================================
# DATABASE CONNECTION
# ============================================================

@st.cache_resource
def get_connection_pool():
    """Create connection pool"""
    return psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        **Config.DB_CONFIG
    )

@contextmanager
def get_db_connection():
    """Get connection from pool"""
    _pool = get_connection_pool()
    conn = _pool.getconn()
    try:
        yield conn
    finally:
        _pool.putconn(conn)

# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data(ttl=600)
def get_understock_items_with_whstock():
    """Get understock items with warehouse stock breakdown"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check if whstock table exists and has data
        try:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'whstock'
                )
            """)
            table_exists = cursor.fetchone()[0]
            
            if table_exists:
                cursor.execute("SELECT COUNT(*) as count FROM whstock")
                whstock_count = cursor.fetchone()['count']
            else:
                whstock_count = 0
        except Exception as e:
            whstock_count = 0
        
        max_date = None
        if whstock_count > 0:
            # Get latest upload_date for whstock
            try:
                cursor.execute("SELECT MAX(upload_date)::date as max_date FROM whstock")
                max_date_row = cursor.fetchone()
                if max_date_row and max_date_row.get('max_date'):
                    max_date = max_date_row['max_date']
            except Exception as e:
                max_date = None
        
        if max_date:
            query = """
            WITH latest_whstock AS (
                SELECT DISTINCT ON (vc_item_code, wh_code)
                    vc_item_code,
                    wh_code,
                    wh_name,
                    balance_qty,
                    upload_date
                FROM whstock
                WHERE upload_date = %s
                ORDER BY vc_item_code, wh_code, upload_date DESC
            ),
            item_sales AS (
                SELECT 
                    "ITEM_CODE" as item_code,
                    SUM("QTY") as sales_30d
                FROM sales_2025
                WHERE "DATE_INVOICE" >= CURRENT_DATE - INTERVAL '30 days'
                    AND "DATE_INVOICE" < CURRENT_DATE
                GROUP BY "ITEM_CODE"
            ),
            inventory_summary AS (
                SELECT 
                    im.itemcode,
                    im.item_name,
                    im.dept,
                    im.grp,
                    SUM(im.shopstock) as total_shop_stock,
                    COUNT(DISTINCT im.shop_code) as shop_count,
                    COALESCE(s.sales_30d, 0) as sales_30d
                FROM inventory_master im
                LEFT JOIN item_sales s ON im.itemcode = s.item_code
                GROUP BY im.itemcode, im.item_name, im.dept, im.grp, s.sales_30d
            )
            SELECT 
                i.itemcode,
                i.item_name,
                i.dept,
                i.grp,
                i.total_shop_stock,
                i.shop_count,
                i.sales_30d,
                COALESCE(SUM(w.balance_qty), 0) as total_wh_stock,
                COUNT(DISTINCT w.wh_code) as wh_count,
                ARRAY_AGG(
                    CASE WHEN w.wh_code IS NOT NULL 
                    THEN w.wh_code || ': ' || w.wh_name || ' (' || COALESCE(w.balance_qty, 0) || ')'
                    END
                ) FILTER (WHERE w.wh_code IS NOT NULL) as wh_details
            FROM inventory_summary i
            LEFT JOIN latest_whstock w ON i.itemcode = w.vc_item_code
            WHERE i.total_shop_stock < (i.sales_30d * 0.7)  -- Stock less than 70% of monthly sales
                AND i.sales_30d > 0  -- Has sales demand
            GROUP BY i.itemcode, i.item_name, i.dept, i.grp, i.total_shop_stock, i.shop_count, i.sales_30d
            ORDER BY (i.sales_30d - i.total_shop_stock) DESC
            LIMIT 1000;
            """
            cursor.execute(query, (max_date,))
        else:
            # No whstock data, just return inventory analysis
            query = """
            WITH item_sales AS (
                SELECT 
                    "ITEM_CODE" as item_code,
                    SUM("QTY") as sales_30d
                FROM sales_2025
                WHERE "DATE_INVOICE" >= CURRENT_DATE - INTERVAL '30 days'
                    AND "DATE_INVOICE" < CURRENT_DATE
                GROUP BY "ITEM_CODE"
            ),
            inventory_summary AS (
                SELECT 
                    im.itemcode,
                    im.item_name,
                    im.dept,
                    im.grp,
                    SUM(im.shopstock) as total_shop_stock,
                    COUNT(DISTINCT im.shop_code) as shop_count,
                    COALESCE(s.sales_30d, 0) as sales_30d
                FROM inventory_master im
                LEFT JOIN item_sales s ON im.itemcode = s.item_code
                GROUP BY im.itemcode, im.item_name, im.dept, im.grp, s.sales_30d
            )
            SELECT 
                i.itemcode,
                i.item_name,
                i.dept,
                i.grp,
                i.total_shop_stock,
                i.shop_count,
                i.sales_30d,
                0 as total_wh_stock,
                0 as wh_count,
                NULL as wh_details
            FROM inventory_summary i
            WHERE i.total_shop_stock < (i.sales_30d * 0.7)  -- Stock less than 70% of monthly sales
                AND i.sales_30d > 0  -- Has sales demand
            ORDER BY (i.sales_30d - i.total_shop_stock) DESC
            LIMIT 1000;
            """
            cursor.execute(query)
        
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=600)
def get_overstock_items_with_whstock():
    """Get overstock items with warehouse stock breakdown"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT MAX(upload_date) as max_date FROM whstock")
        max_date_row = cursor.fetchone()
        max_date = max_date_row['max_date'] if max_date_row and max_date_row['max_date'] else datetime.now().date()
        
        query = """
        WITH latest_whstock AS (
            SELECT DISTINCT ON (vc_item_code, wh_code)
                vc_item_code,
                wh_code,
                wh_name,
                balance_qty,
                upload_date
            FROM whstock
            WHERE upload_date = %s
            ORDER BY vc_item_code, wh_code, upload_date DESC
        ),
        item_sales AS (
            SELECT 
                "ITEM_CODE" as item_code,
                SUM("QTY") as sales_30d
            FROM sales_2025
            WHERE "DATE_INVOICE" >= CURRENT_DATE - INTERVAL '30 days'
                AND "DATE_INVOICE" < CURRENT_DATE
            GROUP BY "ITEM_CODE"
        ),
        inventory_summary AS (
            SELECT 
                im.itemcode,
                im.item_name,
                im.dept,
                im.grp,
                SUM(im.shopstock) as total_shop_stock,
                COUNT(DISTINCT im.shop_code) as shop_count,
                COALESCE(s.sales_30d, 0) as sales_30d
            FROM inventory_master im
            LEFT JOIN item_sales s ON im.itemcode = s.item_code
            GROUP BY im.itemcode, im.item_name, im.dept, im.grp, s.sales_30d
        )
        SELECT 
            i.itemcode,
            i.item_name,
            i.dept,
            i.grp,
            i.total_shop_stock,
            i.shop_count,
            i.sales_30d,
            i.total_shop_stock - (i.sales_30d * 2) as excess_stock,
            COALESCE(SUM(w.balance_qty), 0) as total_wh_stock,
            COUNT(DISTINCT w.wh_code) as wh_count,
            ARRAY_AGG(
                CASE WHEN w.wh_code IS NOT NULL 
                THEN w.wh_code || ': ' || w.wh_name || ' (' || COALESCE(w.balance_qty, 0) || ')'
                END
            ) FILTER (WHERE w.wh_code IS NOT NULL) as wh_details
        FROM inventory_summary i
        LEFT JOIN latest_whstock w ON i.itemcode = w.vc_item_code
        WHERE i.total_shop_stock > (i.sales_30d * 2)  -- Stock more than 2x monthly sales
        GROUP BY i.itemcode, i.item_name, i.dept, i.grp, i.total_shop_stock, i.shop_count, i.sales_30d
        ORDER BY (i.total_shop_stock - i.sales_30d * 2) DESC
        LIMIT 1000;
        """
        cursor.execute(query, (max_date,))
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=600)
def get_critical_items_with_whstock():
    """Get critical items (near stockout with high demand) with warehouse stock"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT MAX(upload_date) as max_date FROM whstock")
        max_date_row = cursor.fetchone()
        max_date = max_date_row['max_date'] if max_date_row and max_date_row['max_date'] else datetime.now().date()
        
        query = """
        WITH latest_whstock AS (
            SELECT DISTINCT ON (vc_item_code, wh_code)
                vc_item_code,
                wh_code,
                wh_name,
                balance_qty,
                upload_date
            FROM whstock
            WHERE upload_date = %s
            ORDER BY vc_item_code, wh_code, upload_date DESC
        ),
        item_sales AS (
            SELECT 
                "ITEM_CODE" as item_code,
                SUM("QTY") as sales_7d,
                SUM("QTY") as sales_30d
            FROM sales_2025
            WHERE "DATE_INVOICE" >= CURRENT_DATE - INTERVAL '30 days'
                AND "DATE_INVOICE" < CURRENT_DATE
            GROUP BY "ITEM_CODE"
        ),
        inventory_summary AS (
            SELECT 
                im.itemcode,
                im.item_name,
                im.dept,
                im.grp,
                SUM(im.shopstock) as total_shop_stock,
                COUNT(DISTINCT im.shop_code) as shop_count,
                COALESCE(s.sales_7d, 0) as sales_7d,
                COALESCE(s.sales_30d, 0) as sales_30d,
                CASE 
                    WHEN SUM(im.shopstock) = 0 THEN 0
                    ELSE ROUND(SUM(im.shopstock) / NULLIF(s.sales_30d / 30.0, 0), 1)
                END as days_of_stock
            FROM inventory_master im
            LEFT JOIN item_sales s ON im.itemcode = s.item_code
            GROUP BY im.itemcode, im.item_name, im.dept, im.grp, s.sales_7d, s.sales_30d
        )
        SELECT 
            i.itemcode,
            i.item_name,
            i.dept,
            i.grp,
            i.total_shop_stock,
            i.shop_count,
            i.sales_7d,
            i.sales_30d,
            i.days_of_stock,
            CASE 
                WHEN i.days_of_stock < 3 THEN '🔴 Critical'
                WHEN i.days_of_stock < 7 THEN '🟠 Warning'
                ELSE '🟡 Monitor'
            END as alert_level,
            COALESCE(SUM(w.balance_qty), 0) as total_wh_stock,
            COUNT(DISTINCT w.wh_code) as wh_count,
            ARRAY_AGG(
                CASE WHEN w.wh_code IS NOT NULL 
                THEN w.wh_code || ': ' || w.wh_name || ' (' || COALESCE(w.balance_qty, 0) || ')'
                END
            ) FILTER (WHERE w.wh_code IS NOT NULL) as wh_details
        FROM inventory_summary i
        LEFT JOIN latest_whstock w ON i.itemcode = w.vc_item_code
        WHERE i.sales_30d > 10  -- High demand items
            AND i.days_of_stock < 14  -- Less than 2 weeks stock
        GROUP BY i.itemcode, i.item_name, i.dept, i.grp, i.total_shop_stock, i.shop_count, 
                 i.sales_7d, i.sales_30d, i.days_of_stock
        ORDER BY i.days_of_stock ASC, i.sales_30d DESC
        LIMIT 1000;
        """
        cursor.execute(query, (max_date,))
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=600)
def get_slow_moving_items_with_whstock():
    """Get slow moving items with warehouse stock"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT MAX(upload_date) as max_date FROM whstock")
        max_date_row = cursor.fetchone()
        max_date = max_date_row['max_date'] if max_date_row and max_date_row['max_date'] else datetime.now().date()
        
        query = """
        WITH latest_whstock AS (
            SELECT DISTINCT ON (vc_item_code, wh_code)
                vc_item_code,
                wh_code,
                wh_name,
                balance_qty,
                upload_date
            FROM whstock
            WHERE upload_date = %s
            ORDER BY vc_item_code, wh_code, upload_date DESC
        ),
        item_sales AS (
            SELECT 
                "ITEM_CODE" as item_code,
                SUM("QTY") as sales_30d,
                SUM("QTY") as sales_90d,
                MAX("DATE_INVOICE") as last_sale_date
            FROM sales_2025
            WHERE "DATE_INVOICE" >= CURRENT_DATE - INTERVAL '90 days'
                AND "DATE_INVOICE" < CURRENT_DATE
            GROUP BY "ITEM_CODE"
        ),
        inventory_summary AS (
            SELECT 
                im.itemcode,
                im.item_name,
                im.dept,
                im.grp,
                SUM(im.shopstock) as total_shop_stock,
                COUNT(DISTINCT im.shop_code) as shop_count,
                COALESCE(s.sales_30d, 0) as sales_30d,
                COALESCE(s.sales_90d, 0) as sales_90d,
                s.last_sale_date,
                CASE 
                    WHEN SUM(im.shopstock) = 0 THEN 0
                    WHEN s.sales_90d = 0 THEN 999
                    ELSE ROUND(SUM(im.shopstock) / NULLIF(s.sales_90d / 90.0, 0), 1)
                END as days_of_stock
            FROM inventory_master im
            LEFT JOIN item_sales s ON im.itemcode = s.item_code
            WHERE im.shopstock > 0
            GROUP BY im.itemcode, im.item_name, im.dept, im.grp, s.sales_30d, s.sales_90d, s.last_sale_date
        )
        SELECT 
            i.itemcode,
            i.item_name,
            i.dept,
            i.grp,
            i.total_shop_stock,
            i.shop_count,
            i.sales_30d,
            i.sales_90d,
            i.last_sale_date,
            i.days_of_stock,
            CURRENT_DATE - i.last_sale_date as days_since_last_sale,
            COALESCE(SUM(w.balance_qty), 0) as total_wh_stock,
            COUNT(DISTINCT w.wh_code) as wh_count,
            ARRAY_AGG(
                CASE WHEN w.wh_code IS NOT NULL 
                THEN w.wh_code || ': ' || w.wh_name || ' (' || COALESCE(w.balance_qty, 0) || ')'
                END
            ) FILTER (WHERE w.wh_code IS NOT NULL) as wh_details
        FROM inventory_summary i
        LEFT JOIN latest_whstock w ON i.itemcode = w.vc_item_code
        WHERE i.sales_90d < 5  -- Very low sales in 90 days
            AND i.total_shop_stock > 0  -- Has stock
        GROUP BY i.itemcode, i.item_name, i.dept, i.grp, i.total_shop_stock, i.shop_count,
                 i.sales_30d, i.sales_90d, i.last_sale_date, i.days_of_stock
        ORDER BY i.days_of_stock DESC, i.total_shop_stock DESC
        LIMIT 1000;
        """
        cursor.execute(query, (max_date,))
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=600)
def get_summary_metrics():
    """Get summary metrics for all inventory categories"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
        WITH item_sales AS (
            SELECT 
                "ITEM_CODE" as item_code,
                SUM("QTY") as sales_30d,
                SUM("QTY") as sales_90d
            FROM sales_2025
            WHERE "DATE_INVOICE" >= CURRENT_DATE - INTERVAL '90 days'
                AND "DATE_INVOICE" < CURRENT_DATE
            GROUP BY "ITEM_CODE"
        ),
        inventory_summary AS (
            SELECT 
                im.itemcode,
                SUM(im.shopstock) as total_shop_stock,
                COALESCE(s.sales_30d, 0) as sales_30d,
                COALESCE(s.sales_90d, 0) as sales_90d,
                CASE 
                    WHEN SUM(im.shopstock) = 0 THEN 0
                    WHEN s.sales_30d = 0 THEN 999
                    ELSE ROUND(SUM(im.shopstock) / NULLIF(s.sales_30d / 30.0, 0), 1)
                END as days_of_stock
            FROM inventory_master im
            LEFT JOIN item_sales s ON im.itemcode = s.item_code
            GROUP BY im.itemcode, s.sales_30d, s.sales_90d
        )
        SELECT 
            COUNT(CASE WHEN total_shop_stock < (sales_30d * 0.7) AND sales_30d > 0 THEN 1 END) as understock_count,
            COUNT(CASE WHEN total_shop_stock > (sales_30d * 2) THEN 1 END) as overstock_count,
            COUNT(CASE WHEN sales_30d > 10 AND days_of_stock < 14 THEN 1 END) as critical_count,
            COUNT(CASE WHEN sales_90d < 5 AND total_shop_stock > 0 THEN 1 END) as slow_moving_count,
            SUM(total_shop_stock) as total_stock,
            SUM(sales_30d) as total_sales_30d
        FROM inventory_summary;
        """
        cursor.execute(query)
        result = cursor.fetchone()
        return dict(result) if result else {}

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 2rem;">
        <img src="{Config.LOGO_URL}" style="width: 50px; height: 50px; border-radius: 8px;">
        <div>
            <h1 style="margin: 0;">📦 Inventory Warehouse Stock Analysis</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **Inventory Analysis with Warehouse Stock**
        
        This dashboard integrates warehouse stock data (WHStock) with inventory analysis:
        
        - 📉 **Understock**: Items with shop stock < 70% of monthly sales
        - 📦 **Overstock**: Items with shop stock > 2x monthly sales  
        - 🚨 **Critical**: High-demand items with < 14 days stock
        - 🐌 **Slow Moving**: Items with < 5 sales in 90 days
        
        Each view shows warehouse-level stock breakdown from WHStock table.
        """)
        
        st.markdown("---")
        st.caption("Data refreshes every 10 minutes")
    
    # Get summary metrics
    metrics = get_summary_metrics()
    
    # Display KPIs
    st.markdown("### 📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "⚠️ Understock Items",
            f"{metrics.get('understock_count', 0):,}",
            help="Items with shop stock less than 70% of monthly sales"
        )
    
    with col2:
        st.metric(
            "📦 Overstock Items",
            f"{metrics.get('overstock_count', 0):,}",
            help="Items with shop stock more than 2x monthly sales"
        )
    
    with col3:
        st.metric(
            "🚨 Critical Items",
            f"{metrics.get('critical_count', 0):,}",
            help="High-demand items with less than 14 days stock"
        )
    
    with col4:
        st.metric(
            "🐌 Slow Moving Items",
            f"{metrics.get('slow_moving_count', 0):,}",
            help="Items with less than 5 sales in 90 days"
        )
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚠️ Understock Items",
        "📦 Overstock Items", 
        "🚨 Critical Items",
        "🐌 Slow Moving Items"
    ])
    
    # Tab 1: Understock Items
    with tab1:
        st.markdown("### ⚠️ Understock Items")
        st.caption("Items with shop stock less than 70% of monthly sales (with warehouse stock details)")
        
        df_understock = get_understock_items_with_whstock()
        
        if not df_understock.empty:
            # Check if WH stock data is available
            if df_understock['total_wh_stock'].sum() == 0 and df_understock['wh_count'].sum() == 0:
                st.info("ℹ️ Warehouse stock data not available. Showing shop inventory analysis only. Upload WH stock data via Home Dashboard to see warehouse details.")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Items", f"{len(df_understock):,}")
            with col2:
                st.metric("Total Shop Stock", f"{df_understock['total_shop_stock'].sum():,.0f}")
            with col3:
                st.metric("Total WH Stock", f"{df_understock['total_wh_stock'].sum():,.0f}")
            
            st.markdown("---")
            
            # Display data
            display_df = df_understock.copy()
            display_df['shortage'] = display_df['sales_30d'] - display_df['total_shop_stock']
            display_df['wh_details_str'] = display_df['wh_details'].apply(
                lambda x: '\n'.join(x) if x and isinstance(x, list) else 'No WH Stock'
            )
            
            # Column configuration - Warehouse Stock columns prominently displayed at top
            st.dataframe(
                display_df[[
                    'item_code', 'item_name', 'dept', 'grp',
                    'total_wh_stock', 'wh_count', 'wh_details_str',
                    'total_shop_stock', 'sales_30d', 'shortage'
                ]],
                column_config={
                    'item_code': st.column_config.TextColumn('Item Code', width='medium'),
                    'item_name': st.column_config.TextColumn('Item Name', width='large'),
                    'dept': st.column_config.TextColumn('Department', width='small'),
                    'grp': st.column_config.TextColumn('Group', width='medium'),
                    'total_shop_stock': st.column_config.NumberColumn('Shop Stock', format='%d', width='small'),
                    'sales_30d': st.column_config.NumberColumn('Sales (30d)', format='%d', width='small'),
                    'shortage': st.column_config.NumberColumn('Shortage', format='%d', width='small'),
                    'total_wh_stock': st.column_config.NumberColumn('🏭 WH Stock', format='%d', width='small', help='Total warehouse stock available'),
                    'wh_count': st.column_config.NumberColumn('# WH', format='%d', width='small', help='Number of warehouses with stock'),
                    'wh_details_str': st.column_config.TextColumn('🏭 Warehouse Stock Details', width='large', help='Breakdown by warehouse')
                },
                hide_index=True,
                use_container_width=True,
                height=500
            )
            
            # Export button
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Understock Data",
                csv,
                "understock_items_with_whstock.csv",
                "text/csv"
            )
        else:
            st.info("✅ No understock items found!")
    
    # Tab 2: Overstock Items
    with tab2:
        st.markdown("### 📦 Overstock Items")
        st.caption("Items with shop stock more than 2x monthly sales (with warehouse stock details)")
        
        df_overstock = get_overstock_items_with_whstock()
        
        if not df_overstock.empty:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Items", f"{len(df_overstock):,}")
            with col2:
                st.metric("Total Excess Stock", f"{df_overstock['excess_stock'].sum():,.0f}")
            with col3:
                st.metric("Total WH Stock", f"{df_overstock['total_wh_stock'].sum():,.0f}")
            
            st.markdown("---")
            
            # Display data
            display_df = df_overstock.copy()
            display_df['wh_details_str'] = display_df['wh_details'].apply(
                lambda x: '\n'.join(x) if x and isinstance(x, list) else 'No WH Stock'
            )
            
            # Column configuration - Warehouse Stock columns prominently displayed at top
            st.dataframe(
                display_df[[
                    'item_code', 'item_name', 'dept', 'grp',
                    'total_wh_stock', 'wh_count', 'wh_details_str',
                    'total_shop_stock', 'sales_30d', 'excess_stock'
                ]],
                column_config={
                    'item_code': st.column_config.TextColumn('Item Code', width='medium'),
                    'item_name': st.column_config.TextColumn('Item Name', width='large'),
                    'dept': st.column_config.TextColumn('Department', width='small'),
                    'grp': st.column_config.TextColumn('Group', width='medium'),
                    'total_shop_stock': st.column_config.NumberColumn('Shop Stock', format='%d', width='small'),
                    'sales_30d': st.column_config.NumberColumn('Sales (30d)', format='%d', width='small'),
                    'excess_stock': st.column_config.NumberColumn('Excess Stock', format='%d', width='small'),
                    'total_wh_stock': st.column_config.NumberColumn('🏭 WH Stock', format='%d', width='small', help='Total warehouse stock available'),
                    'wh_count': st.column_config.NumberColumn('# WH', format='%d', width='small', help='Number of warehouses with stock'),
                    'wh_details_str': st.column_config.TextColumn('🏭 Warehouse Stock Details', width='large', help='Breakdown by warehouse')
                },
                hide_index=True,
                use_container_width=True,
                height=500
            )
            
            # Export button
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Overstock Data",
                csv,
                "overstock_items_with_whstock.csv",
                "text/csv"
            )
        else:
            st.info("✅ No overstock items found!")
    
    # Tab 3: Critical Items
    with tab3:
        st.markdown("### 🚨 Critical Items")
        st.caption("High-demand items with less than 14 days of stock (with warehouse stock details)")
        
        df_critical = get_critical_items_with_whstock()
        
        if not df_critical.empty:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                critical_count = len(df_critical[df_critical['alert_level'] == '🔴 Critical'])
                st.metric("Critical (< 3 days)", f"{critical_count:,}")
            with col2:
                warning_count = len(df_critical[df_critical['alert_level'] == '🟠 Warning'])
                st.metric("Warning (< 7 days)", f"{warning_count:,}")
            with col3:
                st.metric("Total WH Stock Available", f"{df_critical['total_wh_stock'].sum():,.0f}")
            
            st.markdown("---")
            
            # Display data
            display_df = df_critical.copy()
            display_df['wh_details_str'] = display_df['wh_details'].apply(
                lambda x: '\n'.join(x) if x and isinstance(x, list) else 'No WH Stock'
            )
            
            # Column configuration - Warehouse Stock columns prominently displayed at top
            st.dataframe(
                display_df[[
                    'alert_level', 'item_code', 'item_name', 'dept', 'grp',
                    'total_wh_stock', 'wh_count', 'wh_details_str',
                    'total_shop_stock', 'sales_30d', 'days_of_stock'
                ]],
                column_config={
                    'alert_level': st.column_config.TextColumn('🚨 Alert', width='small'),
                    'item_code': st.column_config.TextColumn('Item Code', width='medium'),
                    'item_name': st.column_config.TextColumn('Item Name', width='large'),
                    'dept': st.column_config.TextColumn('Department', width='small'),
                    'grp': st.column_config.TextColumn('Group', width='medium'),
                    'total_shop_stock': st.column_config.NumberColumn('Shop Stock', format='%d', width='small'),
                    'sales_30d': st.column_config.NumberColumn('Sales (30d)', format='%d', width='small'),
                    'days_of_stock': st.column_config.NumberColumn('Days Cover', format='%.1f', width='small'),
                    'total_wh_stock': st.column_config.NumberColumn('🏭 WH Stock', format='%d', width='small', help='Total warehouse stock available'),
                    'wh_count': st.column_config.NumberColumn('# WH', format='%d', width='small', help='Number of warehouses with stock'),
                    'wh_details_str': st.column_config.TextColumn('🏭 Warehouse Stock Details', width='large', help='Breakdown by warehouse')
                },
                hide_index=True,
                use_container_width=True,
                height=500
            )
            
            # Export button
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Critical Items Data",
                csv,
                "critical_items_with_whstock.csv",
                "text/csv"
            )
        else:
            st.info("✅ No critical items found!")
    
    # Tab 4: Slow Moving Items
    with tab4:
        st.markdown("### 🐌 Slow Moving Items")
        st.caption("Items with less than 5 sales in last 90 days (with warehouse stock details)")
        
        df_slow = get_slow_moving_items_with_whstock()
        
        if not df_slow.empty:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Items", f"{len(df_slow):,}")
            with col2:
                st.metric("Total Shop Stock Value", f"{df_slow['total_shop_stock'].sum():,.0f}")
            with col3:
                st.metric("Total WH Stock", f"{df_slow['total_wh_stock'].sum():,.0f}")
            
            st.markdown("---")
            
            # Display data
            display_df = df_slow.copy()
            display_df['wh_details_str'] = display_df['wh_details'].apply(
                lambda x: '\n'.join(x) if x and isinstance(x, list) else 'No WH Stock'
            )
            
            # Column configuration - Warehouse Stock columns prominently displayed at top
            st.dataframe(
                display_df[[
                    'item_code', 'item_name', 'dept', 'grp',
                    'total_wh_stock', 'wh_count', 'wh_details_str',
                    'total_shop_stock', 'sales_90d', 'days_of_stock', 'days_since_last_sale'
                ]],
                column_config={
                    'item_code': st.column_config.TextColumn('Item Code', width='medium'),
                    'item_name': st.column_config.TextColumn('Item Name', width='large'),
                    'dept': st.column_config.TextColumn('Department', width='small'),
                    'grp': st.column_config.TextColumn('Group', width='medium'),
                    'total_shop_stock': st.column_config.NumberColumn('Shop Stock', format='%d', width='small'),
                    'sales_90d': st.column_config.NumberColumn('Sales (90d)', format='%d', width='small'),
                    'days_of_stock': st.column_config.NumberColumn('Days Cover', format='%.1f', width='small'),
                    'days_since_last_sale': st.column_config.NumberColumn('Days Since Sale', width='small'),
                    'total_wh_stock': st.column_config.NumberColumn('🏭 WH Stock', format='%d', width='small', help='Total warehouse stock available'),
                    'wh_count': st.column_config.NumberColumn('# WH', format='%d', width='small', help='Number of warehouses with stock'),
                    'wh_details_str': st.column_config.TextColumn('🏭 Warehouse Stock Details', width='large', help='Breakdown by warehouse')
                },
                hide_index=True,
                use_container_width=True,
                height=500
            )
            
            # Export button
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Slow Moving Items Data",
                csv,
                "slow_moving_items_with_whstock.csv",
                "text/csv"
            )
        else:
            st.info("✅ No slow moving items found!")

if __name__ == "__main__":
    main()
