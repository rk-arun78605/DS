# ==================== OVERVIEW COMPLIANCE TREND FUNCTION ====================
def render_compliance_trend_overview():
    """Render compliance % (N) trend for last 7 days in overview tab."""
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    start_date = yesterday - timedelta(days=6)
    end_date = yesterday
    conn = get_db_connection()
    if not conn:
        st.info("No compliance trend data available.")
        return
    query = f"""
    SELECT DATE(bill_date) as date,
           COUNT(*) FILTER (WHERE TRIM(serial_check) = 'Y') as total_yes,
           COUNT(*) as total_serials,
           CASE WHEN COUNT(*) > 0 THEN ROUND(COUNT(*) FILTER (WHERE TRIM(serial_check) = 'Y')::NUMERIC / COUNT(*) * 100, 2) ELSE 0 END as compliance_pct
    FROM serialno_check_yes_no
    WHERE DATE(bill_date) >= '{start_date}' AND DATE(bill_date) <= '{end_date}'
    GROUP BY DATE(bill_date)
    ORDER BY DATE(bill_date)
    """
    try:
        df = pd.read_sql(query, conn)
        conn.close()
        if df is not None and not df.empty:
            fig = px.line(
                df,
                x='date',
                y='compliance_pct',
                title='Compliance % Trend (Last 7 Days)',
                markers=True
            )
            fig.update_layout(height=250, yaxis_title="Compliance %", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No compliance trend data available.")
    except Exception as e:
        st.error(f"Error loading compliance trend: {e}")
        conn.close()
# ====================== UNiQUENESS TREND FUNCTION ======================
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

def get_wh_uniqueness_trend(mode='received'):
    """Return last 7 days uniqueness % for WH Received or WH Loaded."""
    conn = get_db_connection()
    if not conn:
        return None
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    start_date = yesterday - timedelta(days=6)
    end_date = yesterday
    if mode == 'received':
        query = f"""
        SELECT DATE(grn_date) as date,
               COUNT(*) as received_total,
               COUNT(DISTINCT serial_no) as received_unique,
               CASE WHEN COUNT(*) > 0 THEN ROUND(COUNT(DISTINCT serial_no)::NUMERIC / COUNT(*) * 100, 2) ELSE 0 END as uniqueness_pct
        FROM whreceived_serialno
        WHERE DATE(grn_date) >= '{start_date}' AND DATE(grn_date) <= '{end_date}'
        GROUP BY DATE(grn_date)
        ORDER BY DATE(grn_date)
        """
    else:
        query = f"""
        SELECT DATE(dt_doc_date) as date,
               COUNT(*) as loaded_total,
               COUNT(DISTINCT serial_no) as loaded_unique,
               CASE WHEN COUNT(*) > 0 THEN ROUND(COUNT(DISTINCT serial_no)::NUMERIC / COUNT(*) * 100, 2) ELSE 0 END as uniqueness_pct
        FROM serial_no_dailydata
        WHERE DATE(dt_doc_date) >= '{start_date}' AND DATE(dt_doc_date) <= '{end_date}'
        GROUP BY DATE(dt_doc_date)
        ORDER BY DATE(dt_doc_date)
        """
    try:
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading uniqueness trend: {e}")
        conn.close()
        return None


import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta, date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import html as html_lib

try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Serial Tracker | Melcom",
    page_icon="https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== THEME COLORS ======================
MELCOM_BLUE = "#002c6d"
MELCOM_RED = "#ed1b24"
MELCOM_GREEN = "#28a745"
MELCOM_ORANGE = "#fd7e14"
MELCOM_PURPLE = "#6f42c1"
MELCOM_CYAN = "#17a2b8"
SERIAL_CHECK_TABLE = "serialno_check_yes_no"

# ====================== STYLING ======================
st.markdown(f"""
<style>
    /* Remove Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Modern gradient background */
    .stApp {{
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }}
    
    /* Dashboard Header */
    .dashboard-header {{
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: {MELCOM_BLUE};
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,44,109,0.3);
    }}
    
    .dashboard-title {{
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }}
    
    .dashboard-subtitle {{
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        transition: all 0.3s ease;
        border-left: 5px solid {{MELCOM_BLUE}};
        height: 100%;
    }}
    .metric-card.danger {{
        background: #ffeaea !important;
        border-left-color: {{MELCOM_RED}};
    }}
    .metric-card.success {{
        background: #e8f7ef !important;
        border-left-color: {{MELCOM_GREEN}};
    }}
    .metric-card.warning {{
        background: #fffbe6 !important;
        border-left-color: #ffc107;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }}
    
    .metric-card.warning {{
    }}
    
    .metric-card.danger {{
        border-left-color: {MELCOM_RED};
    }}
    
    .metric-card.success {{
        border-left-color: {MELCOM_GREEN};
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {MELCOM_BLUE};
        margin: 0.5rem 0;
    }}
    
    .metric-label {{
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .metric-change {{
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }}
    
    .metric-change.positive {{
        color: {MELCOM_GREEN};
    }}
    
    .metric-change.negative {{
        color: {MELCOM_RED};
    }}
    
    /* Section Headers */
    .section-header {{
        background: linear-gradient(90deg, {MELCOM_BLUE} 0%, {MELCOM_PURPLE} 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }}
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: white;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        border: 2px solid transparent;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: #f0f2f6;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {MELCOM_BLUE};
        color: white;
        border-color: {MELCOM_BLUE};
    }}
    
    /* DataFrames - LARGE FONT SIZE */
    .dataframe {{
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    
    /* AGGRESSIVE FONT SIZE INCREASE - TARGET ALL POSSIBLE SELECTORS - RESPONSIVE TO ZOOM */
    div[data-testid="stDataFrame"] table,
    div[data-testid="stDataFrame"] table *,
    .stDataFrame table,
    .stDataFrame table *,
    [data-testid="stDataFrame"] *,
    .dataframe,
    .dataframe * {{
        font-size: 1.5rem !important;
    }}
    
    div[data-testid="stDataFrame"] thead th,
    .stDataFrame thead th,
    [data-testid="stDataFrame"] thead th {{
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        padding: 1rem !important;
        line-height: 1.5 !important;
    }}
    
    div[data-testid="stDataFrame"] tbody td,
    .stDataFrame tbody td,
    [data-testid="stDataFrame"] tbody td {{
        font-size: 1.5rem !important;
        padding: 0.875rem !important;
        line-height: 1.5 !important;
    }}
    
    /* Column config text */
    div[data-testid="stDataFrame"] [data-testid="stDataFrameCell"],
    [data-testid="stDataFrameCell"] {{
        font-size: 1.5rem !important;
    }}
    
    /* Target the actual data cells */
    div[data-testid="stDataFrame"] div[role="gridcell"],
    div[data-testid="stDataFrame"] div[role="columnheader"] {{
        font-size: 1.5rem !important;
    }}
    
    /* Alerts */
    .alert-box {{
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid;
    }}
    
    .alert-box.critical {{
        background-color: #fee;
        border-left-color: {MELCOM_RED};
        color: #721c24;
    }}
    
    .alert-box.warning {{
        background-color: #fff3cd;
        color: #856404;
    }}
    
    .alert-box.info {{
        background-color: #d1ecf1;
        border-left-color: {MELCOM_CYAN};
        color: #0c5460;
    }}
</style>
""", unsafe_allow_html=True)

# ====================== DATABASE CONNECTION ======================
DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'WH'
}

def get_db_connection():
    """Create database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return None

# ====================== HELPER FUNCTIONS ======================

def format_number(num):
    """Format large numbers with commas"""
    if pd.isna(num):
        return "0"
    return f"{int(num):,}"

def format_percentage(num, decimals=1):
    """Format percentage value with specified decimals"""
    if pd.isna(num):
        return "0.0%"
    return f"{num:.{decimals}f}%"

def get_time_bucket(days):
    """Categorize days into buckets"""
    if pd.isna(days):
        return "Unknown"
    elif days < 0:
        return "Before WH Load"
    elif days <= 3:
        return "0-3 days"
    elif days <= 5:
        return "3-5 days"
    elif days <= 7:
        return "5-7 days"
    elif days <= 10:
        return "7-10 days"
    else:
        return ">10 days"

def get_global_date_range():
    """Return the globally selected date range (defaults to last 30 days, ending yesterday)."""
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    default_start = yesterday - timedelta(days=29)
    start_date = st.session_state.get("global_start_date", default_start)
    end_date = st.session_state.get("global_end_date", yesterday)
    return start_date, end_date

def get_global_date_strs():
    start_date, end_date = get_global_date_range()
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def normalize_date_selection(selection):
    """Normalize Streamlit date_input output into an ordered (start, end) tuple."""
    if isinstance(selection, tuple):
        start, end = selection
    else:
        start = end = selection

    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end).date()

    if start > end:
        start, end = end, start
    return start, end

# ====================== DATA LOADING FUNCTIONS ======================
@st.cache_data(ttl=300)
def get_warehouse_metrics(received_range=None, loaded_range=None):
    """Get warehouse metrics filtered by optional GRN/Doc date ranges."""
    conn = get_db_connection()
    if not conn:
        return None
    
    received_filter = ""
    loaded_filter = ""
    params = []

    if received_range:
        received_filter = " AND grn_date BETWEEN %s AND %s"
        params.extend(received_range)

    if loaded_range:
        loaded_filter = " AND DATE(dt_doc_date) BETWEEN %s AND %s"
        params.extend(loaded_range)

    query = f"""
    WITH wh_received AS (
        -- Received metrics from whreceived_serialno table
        SELECT 
            COUNT(*) as received_total,
            COUNT(DISTINCT serial_no) as received_unique
        FROM whreceived_serialno
        WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
        {received_filter}
    ),
    wh_loaded AS (
        SELECT 
            -- Loaded metrics from serial_no_dailydata
            COUNT(*) as loaded_total,
            COUNT(DISTINCT serial_no) as loaded_unique,
            
            -- Duplicate serials
            COUNT(*) - COUNT(DISTINCT serial_no) as duplicate_serials_count,
            
            -- Unique items with duplicate serials
            COUNT(DISTINCT vc_item_code) FILTER (
                WHERE serial_no IN (
                    SELECT serial_no 
                    FROM serial_no_dailydata 
                    WHERE serial_no IS NOT NULL 
                    GROUP BY serial_no 
                    HAVING COUNT(*) > 1
                )
            ) as unique_items_with_dup_serials
            
        FROM serial_no_dailydata
        WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
        {loaded_filter}
    )
    SELECT 
        r.received_total,
        r.received_unique,
        l.loaded_total,
        l.loaded_unique,
        l.duplicate_serials_count,
        l.unique_items_with_dup_serials,
        CASE 
            WHEN l.loaded_total > 0 
            THEN ROUND((l.loaded_unique::NUMERIC / l.loaded_total * 100), 2)
            ELSE 0 
        END as unique_percentage,
        CASE 
            WHEN l.loaded_total > 0 
            THEN ROUND(((l.loaded_total - l.loaded_unique)::NUMERIC / l.loaded_total * 100), 2)
            ELSE 0 
        END as duplicate_percentage
    FROM wh_received r
    CROSS JOIN wh_loaded l
    """
    
    try:
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df.iloc[0].to_dict() if not df.empty else None
    except Exception as e:
        st.error(f"Error loading warehouse metrics: {e}")
        conn.close()
        return None

@st.cache_data(ttl=300)
def get_wh_metrics_by_date(target_date=None):
    """Get warehouse receiving metrics for a specific date"""
    conn = get_db_connection()
    if not conn:
        return None, None

    if target_date is None:
        target_date = (datetime.today() - timedelta(days=1)).date()
    else:
        target_date = pd.to_datetime(target_date).date()

    query_overall = """
    SELECT 
        COUNT(*) as total_received,
        COUNT(DISTINCT serial_no) as unique_serials,
        COUNT(*) - COUNT(DISTINCT serial_no) as duplicate_count,
        CASE 
            WHEN COUNT(*) > 0 
            THEN ROUND((COUNT(DISTINCT serial_no)::NUMERIC / COUNT(*) * 100), 2)
            ELSE 0 
        END as unique_percentage
    FROM whreceived_serialno
    WHERE serial_no IS NOT NULL 
      AND LENGTH(TRIM(serial_no::text)) > 0
      AND grn_date = %s
    """

    query_by_wh = """
    SELECT 
        COALESCE(warehouse_name, 'Unknown') as warehouse,
        COUNT(*) as total_received,
        COUNT(DISTINCT serial_no) as unique_serials,
        COUNT(*) - COUNT(DISTINCT serial_no) as duplicate_count,
        CASE 
            WHEN COUNT(*) > 0 
            THEN ROUND((COUNT(DISTINCT serial_no)::NUMERIC / COUNT(*) * 100), 2)
            ELSE 0 
        END as unique_percentage
    FROM whreceived_serialno
    WHERE serial_no IS NOT NULL 
      AND LENGTH(TRIM(serial_no::text)) > 0
      AND grn_date = %s
    GROUP BY warehouse_name
    ORDER BY warehouse_name
    """

    try:
        df_overall = pd.read_sql(query_overall, conn, params=(target_date,))
        df_by_wh = pd.read_sql(query_by_wh, conn, params=(target_date,))
        conn.close()

        overall = df_overall.iloc[0].to_dict() if not df_overall.empty else None
        return overall, df_by_wh
    except Exception as e:
        st.error(f"Error loading receiving metrics: {e}")
        if conn:
            conn.close()
        return None, None

@st.cache_data(ttl=300)
def get_wh_10day_trend(warehouse='All'):
    """Get 10-day unique percentage trend for warehouse"""
    conn = get_db_connection()
    if not conn:
        return None
    
    end_date = (datetime.today() - timedelta(days=1)).date()
    start_date = end_date - timedelta(days=9)
    
    wh_filter = ""
    params = [start_date, end_date]
    
    if warehouse != 'All':
        wh_filter = "AND warehouse_name = %s"
        params.append(warehouse)
    
    query = f"""
    SELECT 
        grn_date,
        COALESCE(warehouse_name, 'Unknown') as warehouse,
        COUNT(*) as total_received,
        COUNT(DISTINCT serial_no) as unique_serials,
        COUNT(*) - COUNT(DISTINCT serial_no) as duplicate_count,
        CASE 
            WHEN COUNT(*) > 0 
            THEN ROUND((COUNT(DISTINCT serial_no)::NUMERIC / COUNT(*) * 100), 2)
            ELSE 0 
        END as unique_percentage
    FROM whreceived_serialno
    WHERE serial_no IS NOT NULL 
      AND LENGTH(TRIM(serial_no::text)) > 0
      AND grn_date BETWEEN %s AND %s
      {wh_filter}
    GROUP BY grn_date, warehouse_name
    ORDER BY grn_date
    """
    
    try:
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading 10-day trend: {e}")
        if conn:
            conn.close()
        return None

@st.cache_data(ttl=300)
def get_duplicates_by_date_range(start_date, end_date):
    """Get duplicate serial details for a date range"""
    conn = get_db_connection()
    if not conn:
        return None

    query = """
    WITH serials AS (
        SELECT 
            serial_no,
            COALESCE(warehouse_name, 'Unknown') as warehouse,
            item_code,
            COALESCE(supp_name, 'Unknown') as supp_name,
            grn_date,
            COUNT(*) OVER (PARTITION BY serial_no) as occurrence_count
        FROM whreceived_serialno
        WHERE serial_no IS NOT NULL 
          AND LENGTH(TRIM(serial_no::text)) > 0
          AND grn_date BETWEEN %s AND %s
    )
    SELECT 
        serial_no,
        warehouse,
        item_code,
        supp_name,
        grn_date,
        occurrence_count
    FROM serials
    WHERE occurrence_count > 1
    ORDER BY grn_date DESC, occurrence_count DESC, serial_no
    """

    try:
        df = pd.read_sql(query, conn, params=(start_date, end_date))
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading duplicate details: {e}")
        if conn:
            conn.close()
        return None

DUPLICATE_TABLE_CSS = """
<style>
    #yesterday-duplicate-table-wrapper {
        margin: 0 0 1rem 0;
    }
    #yesterday-duplicate-table {
        border-collapse: separate;
        margin: 0;
        table-layout: auto;
        width: max-content;
        font-size: 0.95rem;
        white-space: nowrap;
    }
    #yesterday-duplicate-table th,
    #yesterday-duplicate-table td {
        text-align: center;
        padding: 6px 10px;
        border-bottom: 1px solid #d1d5db;
    }
    #yesterday-duplicate-table th {
        text-transform: none;
        font-weight: 600;
        background-color: #f8fafc;
    }
</style>
"""

def render_yesterday_duplicate_table(df: pd.DataFrame) -> str:
    """Render yesterday's duplicate serial details with flexible column widths."""
    column_order = [
        "serial_no",
        "warehouse",
        "item_code",
        "supp_name",
        "grn_date",
        "occurrence_count"
    ]
    headers = [
        "Serial Number",
        "Warehouse",
        "Item Code",
        "Supplier Name",
        "GRN Date",
        "Occurrences"
    ]

    header_cells = [f"<th>{html_lib.escape(text)}</th>" for text in headers]
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in column_order:
            value = row.get(col, "")
            cells.append(f"<td>{html_lib.escape(str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    table_html = f"""
    <table id=\"yesterday-duplicate-table\">
        <thead>
            <tr>{''.join(header_cells)}</tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """

    wrapper = f"<div id='yesterday-duplicate-table-wrapper'>{table_html}</div>"
    return DUPLICATE_TABLE_CSS + wrapper


def render_yesterday_trend(yesterday_by_wh, container=None, include_header=True, key_suffix=""):
    """Render the 10-day unique percentage trend within a given container."""
    container = container or st
    if include_header:
        container.markdown("### 📈 Last 10 Days Unique Percentage Trend", unsafe_allow_html=True)

    wh_list = (
        ['All'] + sorted(yesterday_by_wh['warehouse'].unique().tolist())
        if yesterday_by_wh is not None and not yesterday_by_wh.empty else ['All']
    )

    trend_col1, trend_col2 = container.columns([1, 4])

    with trend_col1:
        selected_wh = st.selectbox(
            "Select Warehouse",
            options=wh_list,
            index=0,
            key=f"wh_selector_trend{key_suffix}"
        )

    with trend_col2:
        trend_data = get_wh_10day_trend(selected_wh)

        if trend_data is not None and not trend_data.empty:
            if selected_wh == 'All':
                trend_agg = trend_data.groupby('grn_date').agg({
                    'total_received': 'sum',
                    'unique_serials': 'sum',
                    'duplicate_count': 'sum'
                }).reset_index()
                trend_agg['unique_percentage'] = (
                    trend_agg['unique_serials'] / trend_agg['total_received'] * 100
                ).round(2)
            else:
                trend_agg = trend_data

            fig_trend = go.Figure()

            fig_trend.add_trace(go.Scatter(
                x=trend_agg['grn_date'],
                y=trend_agg['unique_percentage'],
                mode='lines+markers',
                name='Unique %',
                line=dict(color=MELCOM_BLUE, width=3),
                marker=dict(size=10, symbol='circle'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Unique:</b> %{y:.2f}%<br><extra></extra>',
                customdata=trend_agg[['grn_date', 'total_received', 'unique_serials', 'duplicate_count']].values
            ))

            fig_trend.add_hline(
                y=100,
                line_dash="dash",
                line_color="green",
                annotation_text="100% Unique Target",
                annotation_position="right"
            )

            fig_trend.update_layout(
                title=f'Unique Percentage Trend - {selected_wh}',
                xaxis_title='Date',
                yaxis_title='Unique Percentage (%)',
                height=400,
                hovermode='x unified',
                yaxis=dict(range=[0, 105])
            )

            st.plotly_chart(fig_trend, use_container_width=True, key=f"trend_chart{key_suffix}")
        else:
            st.warning(f"No data available for {selected_wh} in the last 10 days.")

@st.cache_data(ttl=300)
def get_loader_breakdown(start_date, end_date):
    """Get breakdown by warehouse loader filtered by date range."""
    conn = get_db_connection()
    if not conn:
        return None

    start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
    end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)

    query = f"""
    SELECT 
        wh_load_user as loader,
        COUNT(*) as total_loads,
        COUNT(DISTINCT serial_no) as unique_serials,
        COUNT(*) - COUNT(DISTINCT serial_no) as duplicate_count,
        COUNT(DISTINCT vc_item_code) as unique_items,
        COUNT(DISTINCT vc_item_code) FILTER (
            WHERE serial_no IN (
                SELECT serial_no 
                FROM serial_no_dailydata 
                GROUP BY serial_no 
                HAVING COUNT(*) > 1
            )
        ) as items_with_dup_serials
    FROM serial_no_dailydata
    WHERE wh_load_user IS NOT NULL 
      AND serial_no IS NOT NULL
      AND DATE(dt_doc_date) >= '{start_str}' AND DATE(dt_doc_date) <= '{end_str}'
    GROUP BY wh_load_user
    ORDER BY total_loads DESC
    """

    try:
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading loader breakdown: {e}")
        conn.close()
        return None

@st.cache_data(ttl=300)
def get_received_duplicate_serials_with_remarks(limit=1000):
    """Get duplicate serials from whreceived_serialno with consolidated remarks"""
    conn = get_db_connection()
    if not conn:
        return None

    def get_column_expr(col_name, default_type="text"):
        if col_name:
            safe = col_name.replace('"', '""')
            return f"\"{safe}\""
        return f"NULL::{default_type}"

    # Resolve actual column names from whreceived_serialno
    try:
        cols_df = pd.read_sql(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'whreceived_serialno'
            """,
            conn
        )
        cols = cols_df['column_name'].tolist()
    except Exception:
        cols = []

    cols_map = {c.lower(): c for c in cols}

    item_code_col = cols_map.get('item_code') or cols_map.get('vc_item_code')
    wh_code_col = (
        cols_map.get('vc_wh_code') or
        cols_map.get('wh_code') or
        cols_map.get('vc_warehouse_code') or
        cols_map.get('vc_warehouse_desc')
    )
    shop_code_col = cols_map.get('vc_shop_code') or cols_map.get('shop_code')
    shop_sold_col = cols_map.get('shop_sold')
    offload_serial_col = cols_map.get('vc_serail_no') or cols_map.get('vc_serial_no')
    sold_serial_col = cols_map.get('shop_serail_no')
    offload_time_col = cols_map.get('dt_mod_date')
    sold_time_col = cols_map.get('dt_invoice_date')

    item_code_expr = get_column_expr(item_code_col)
    wh_code_expr = get_column_expr(wh_code_col)
    shop_code_expr = get_column_expr(shop_code_col)
    shop_sold_expr = get_column_expr(shop_sold_col)
    offload_serial_expr = get_column_expr(offload_serial_col)
    sold_serial_expr = get_column_expr(sold_serial_col)
    offload_time_expr = get_column_expr(offload_time_col, default_type="timestamp")
    sold_time_expr = get_column_expr(sold_time_col, default_type="timestamp")

    query = """
    WITH received AS (
        SELECT
            CASE
                WHEN pg_typeof(serial_no)::text IN (
                    'smallint', 'integer', 'bigint', 'numeric', 'real', 'double precision'
                ) THEN to_char(serial_no::numeric, 'FM999999999999999999999999999999999999')
                ELSE serial_no::text
            END as serial_number,
            LOWER(TRIM(
                CASE
                    WHEN pg_typeof(serial_no)::text IN (
                        'smallint', 'integer', 'bigint', 'numeric', 'real', 'double precision'
                    ) THEN to_char(serial_no::numeric, 'FM999999999999999999999999999999999999')
                    ELSE serial_no::text
                END
            )) as serial_key,
            {item_code_expr} as item_code,
            {wh_code_expr} as vc_wh_code,
            {shop_code_expr} as vc_shop_code,
            {shop_sold_expr} as shop_sold,
            {offload_serial_expr} as vc_serail_no,
            {sold_serial_expr} as shop_serail_no,
            {offload_time_expr} as dt_mod_date,
            {sold_time_expr} as dt_invoice_date
        FROM whreceived_serialno
        WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
    ),
    agg AS (
        SELECT
            serial_key,
            MIN(serial_number) as serial_number,
            COUNT(*) as record_count,
            COUNT(DISTINCT item_code) as item_count,
            STRING_AGG(DISTINCT item_code, ', ' ORDER BY item_code) FILTER (
                WHERE item_code IS NOT NULL AND LENGTH(TRIM(item_code)) > 0
            ) as item_codes,
            COUNT(DISTINCT vc_wh_code) as wh_code_count,
            STRING_AGG(DISTINCT vc_wh_code, ', ' ORDER BY vc_wh_code) FILTER (
                WHERE vc_wh_code IS NOT NULL AND LENGTH(TRIM(vc_wh_code)) > 0
            ) as wh_codes,
            COUNT(DISTINCT vc_shop_code) as shop_count,
            STRING_AGG(DISTINCT vc_shop_code, ', ' ORDER BY vc_shop_code) FILTER (
                WHERE vc_shop_code IS NOT NULL AND LENGTH(TRIM(vc_shop_code)) > 0
            ) as offload_shops,
            COUNT(DISTINCT shop_sold) as sold_shop_count,
            STRING_AGG(DISTINCT shop_sold, ', ' ORDER BY shop_sold) FILTER (
                WHERE shop_sold IS NOT NULL AND LENGTH(TRIM(shop_sold)) > 0
            ) as sold_shops,
            COUNT(DISTINCT dt_mod_date) as offload_time_count,
            COUNT(DISTINCT dt_invoice_date) as sold_time_count,
            COUNT(*) FILTER (
                WHERE vc_serail_no IS NOT NULL AND LENGTH(TRIM(vc_serail_no)) > 0
                  AND shop_serail_no IS NOT NULL AND LENGTH(TRIM(shop_serail_no)) > 0
                  AND LOWER(TRIM(vc_serail_no)) <> LOWER(TRIM(shop_serail_no))
            ) as offload_sold_mismatch_count,
            COUNT(*) FILTER (
                WHERE vc_shop_code IS NOT NULL AND LENGTH(TRIM(vc_shop_code)) > 0
                  AND shop_sold IS NOT NULL AND LENGTH(TRIM(shop_sold)) > 0
                  AND LOWER(TRIM(vc_shop_code)) <> LOWER(TRIM(shop_sold))
            ) as offload_sold_shop_mismatch_count,
            COUNT(*) FILTER (
                WHERE shop_serail_no IS NOT NULL AND LENGTH(TRIM(shop_serail_no)) > 0
            ) as sold_records
        FROM received
        GROUP BY serial_key
        HAVING COUNT(*) > 1
    )
    SELECT
        serial_number,
        record_count,
        item_codes,
        wh_codes,
        offload_shops,
        sold_shops,
        COALESCE(NULLIF(CONCAT_WS('; ',
            CASE WHEN item_count > 1 THEN 'Item mismatch' END,
            CASE WHEN wh_code_count > 1 THEN 'WH code mismatch' END,
            CASE WHEN shop_count > 1 THEN 'Multiple Shop loaded from WH' END,
            CASE WHEN offload_sold_shop_mismatch_count > 0 THEN 'Shop mismatch' END,
            CASE WHEN offload_sold_mismatch_count > 0 THEN 'Offload vs Sold mismatch' END,
            CASE WHEN offload_time_count > 1 THEN 'Offload time diff' END,
            CASE WHEN sold_time_count > 1 THEN 'Sold time diff' END,
            CASE WHEN sold_records = 0 THEN 'Not sold' END
        ), ''), 'Duplicate only') as remark
    FROM agg
    ORDER BY record_count DESC, serial_number
    """

    if limit is not None:
        query += "\nLIMIT %s"

    query = query.format(
        item_code_expr=item_code_expr,
        wh_code_expr=wh_code_expr,
        shop_code_expr=shop_code_expr,
        shop_sold_expr=shop_sold_expr,
        offload_serial_expr=offload_serial_expr,
        sold_serial_expr=sold_serial_expr,
        offload_time_expr=offload_time_expr,
        sold_time_expr=sold_time_expr
    )

    try:
        if limit is None:
            df = pd.read_sql(query, conn)
        else:
            df = pd.read_sql(query, conn, params=(limit,))
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading missing received serials: {e}")
        conn.close()
        return None

@st.cache_data(ttl=300)
def get_shop_offload_metrics(start_date=None, end_date=None):
    """Get shop offloading metrics with optional WH load date filters"""
    conn = get_db_connection()
    if not conn:
        return None

    date_filters = []
    params = []
    if start_date:
        date_filters.append("loaded_datetime::DATE >= %s")
        params.append(start_date)
    if end_date:
        date_filters.append("loaded_datetime::DATE <= %s")
        params.append(end_date)

    date_clause = ""
    if date_filters:
        date_clause = " AND " + " AND ".join(date_filters)

    query = f"""
    WITH parsed AS (
        SELECT
            serial_no,
            vc_serail_no,
            NULLIF(TRIM(dt_mod_date::text), '')::timestamp as mod_dt,
            NULLIF(TRIM(loaded_datetime::text), '')::timestamp as load_dt
        FROM serial_no_dailydata
        WHERE 1=1
          {date_clause}
    ),
    offload_data AS (
        SELECT 
            COUNT(*) FILTER (WHERE serial_no IS NOT NULL) as wh_loaded_total,
            COUNT(DISTINCT serial_no) FILTER (WHERE serial_no IS NOT NULL) as wh_loaded_unique,
            COUNT(*) FILTER (WHERE vc_serail_no IS NOT NULL) as total_offloaded,
            COUNT(DISTINCT vc_serail_no) FILTER (WHERE vc_serail_no IS NOT NULL) as unique_offloaded,
            COUNT(*) FILTER (
                WHERE vc_serail_no IS NOT NULL
                  AND mod_dt IS NULL
            ) as no_offloaded,
            COUNT(*) FILTER (
                WHERE vc_serail_no IS NOT NULL 
                  AND mod_dt IS NOT NULL 
                  AND load_dt IS NOT NULL
                  AND mod_dt::DATE < load_dt::DATE
            ) as offload_before_wh_load,
            COUNT(*) FILTER (
                WHERE vc_serail_no IS NOT NULL 
                  AND mod_dt IS NOT NULL 
                  AND load_dt IS NOT NULL
                  AND (mod_dt::DATE - load_dt::DATE) BETWEEN 0 AND 3
            ) as bucket_0_3_days,
            COUNT(*) FILTER (
                WHERE vc_serail_no IS NOT NULL 
                  AND mod_dt IS NOT NULL 
                  AND load_dt IS NOT NULL
                  AND (mod_dt::DATE - load_dt::DATE) BETWEEN 4 AND 5
            ) as bucket_3_5_days,
            COUNT(*) FILTER (
                WHERE vc_serail_no IS NOT NULL 
                  AND mod_dt IS NOT NULL 
                  AND load_dt IS NOT NULL
                  AND (mod_dt::DATE - load_dt::DATE) BETWEEN 6 AND 7
            ) as bucket_5_7_days,
            COUNT(*) FILTER (
                WHERE vc_serail_no IS NOT NULL 
                  AND mod_dt IS NOT NULL 
                  AND load_dt IS NOT NULL
                  AND (mod_dt::DATE - load_dt::DATE) BETWEEN 8 AND 10
            ) as bucket_7_10_days,
            COUNT(*) FILTER (
                WHERE vc_serail_no IS NOT NULL 
                  AND mod_dt IS NOT NULL 
                  AND load_dt IS NOT NULL
                  AND (mod_dt::DATE - load_dt::DATE) > 10
            ) as bucket_10_plus_days
        FROM parsed
    )
    SELECT * FROM offload_data
    """
    
    try:
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df.iloc[0].to_dict() if not df.empty else None
    except Exception as e:
        st.error(f"Error loading shop offload metrics: {e}")
        conn.close()
        return None

@st.cache_data(ttl=300)
def get_offloading_details_by_shop_and_time(start_date=None, end_date=None):
    """Get detailed offloading records showing shop, dates, and time taken"""
    conn = get_db_connection()
    if not conn:
        return None

    date_filters = []
    params = []
    if start_date:
        date_filters.append("loaded_datetime::DATE >= %s")
        params.append(start_date)
    if end_date:
        date_filters.append("loaded_datetime::DATE <= %s")
        params.append(end_date)

    date_clause = ""
    if date_filters:
        date_clause = " AND " + " AND ".join(date_filters)

    query = f"""
    SELECT 
        vc_shop_code as shop_code,
        serial_no,
        vc_serail_no,
        vc_item_code as item_code,
        vc_item_desc as item_desc,
        loaded_datetime::DATE as wh_loaded_date,
        dt_mod_date as shop_offload_date,
        CASE 
            WHEN dt_mod_date IS NOT NULL AND loaded_datetime IS NOT NULL
            THEN dt_mod_date::DATE - loaded_datetime::DATE
            ELSE NULL
        END as days_to_offload,
        CASE 
            WHEN dt_mod_date IS NULL THEN 'No Offload Date'
            WHEN dt_mod_date IS NOT NULL AND loaded_datetime IS NOT NULL 
                 AND dt_mod_date::DATE - loaded_datetime::DATE BETWEEN 0 AND 3 THEN '0-3 days'
            WHEN dt_mod_date IS NOT NULL AND loaded_datetime IS NOT NULL 
                 AND dt_mod_date::DATE - loaded_datetime::DATE BETWEEN 4 AND 5 THEN '4-5 days'
            WHEN dt_mod_date IS NOT NULL AND loaded_datetime IS NOT NULL 
                 AND dt_mod_date::DATE - loaded_datetime::DATE BETWEEN 6 AND 7 THEN '6-7 days'
            WHEN dt_mod_date IS NOT NULL AND loaded_datetime IS NOT NULL 
                 AND dt_mod_date::DATE - loaded_datetime::DATE BETWEEN 8 AND 10 THEN '8-10 days'
            WHEN dt_mod_date IS NOT NULL AND loaded_datetime IS NOT NULL 
                 AND dt_mod_date::DATE - loaded_datetime::DATE > 10 THEN '>10 days'
            ELSE 'Unknown'
        END as time_bucket
    FROM serial_no_dailydata
    WHERE vc_serail_no IS NOT NULL
        AND vc_shop_code IS NOT NULL
        {date_clause}
    ORDER BY shop_code, days_to_offload DESC NULLS LAST
    """
    
    try:
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading offloading details: {e}")
        if conn:
            conn.close()
        return None

@st.cache_data(ttl=300)
def get_not_offloaded_metrics():
    """Get metrics for items not offloaded with discrepancy breakdown"""
    conn = get_db_connection()
    if not conn:
        return None
    
    query = """
    WITH not_offloaded AS (
        SELECT *,
            CASE 
                WHEN dt_doc_date IS NOT NULL 
                THEN CURRENT_DATE - dt_doc_date::DATE
                ELSE NULL
            END as days_since_wh_load
        FROM serial_no_dailydata
        WHERE serial_no IS NOT NULL 
          AND (vc_serail_no IS NULL OR LENGTH(TRIM(vc_serail_no)) = 0)
          AND (shop_serail_no IS NULL OR LENGTH(TRIM(shop_serail_no)) = 0)
    )
    SELECT 
        COUNT(*) as total_not_offloaded,
        
        -- Discrepancy types
        COUNT(*) FILTER (
            WHERE vc_shop_code IS NOT NULL 
              AND shop_sold IS NOT NULL 
              AND UPPER(TRIM(vc_shop_code)) != UPPER(TRIM(shop_sold))
        ) as shop_mismatch,
        
        COUNT(*) FILTER (
            WHERE serial_no IS NOT NULL 
              AND vc_serail_no IS NOT NULL 
              AND UPPER(TRIM(serial_no)) != UPPER(TRIM(vc_serail_no))
        ) as serial_mismatch,
        
        COUNT(*) FILTER (
            WHERE vc_vehicle_no IS NOT NULL 
              AND "vc_vehicle_no.1" IS NOT NULL 
              AND UPPER(TRIM(vc_vehicle_no)) != UPPER(TRIM("vc_vehicle_no.1"))
        ) as vehicle_mismatch,
        
        -- Time buckets since WH load
        COUNT(*) FILTER (WHERE days_since_wh_load BETWEEN 0 AND 3) as bucket_0_3_days,
        COUNT(*) FILTER (WHERE days_since_wh_load BETWEEN 4 AND 5) as bucket_3_5_days,
        COUNT(*) FILTER (WHERE days_since_wh_load BETWEEN 6 AND 7) as bucket_5_7_days,
        COUNT(*) FILTER (WHERE days_since_wh_load BETWEEN 8 AND 10) as bucket_7_10_days,
        COUNT(*) FILTER (WHERE days_since_wh_load > 10) as bucket_10_plus_days
        
    FROM not_offloaded
    """
    
    try:
        df = pd.read_sql(query, conn)
        conn.close()
        return df.iloc[0].to_dict() if not df.empty else None
    except Exception as e:
        st.error(f"Error loading not offloaded metrics: {e}")
        conn.close()
        return None

@st.cache_data(ttl=300)
def get_not_offloaded_details(bucket_label, limit=1000):
    """Get not-offloaded records for a selected time bucket"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()

    bucket_conditions = {
        '0-3 days': "days_since_wh_load BETWEEN 0 AND 3",
        '3-5 days': "days_since_wh_load BETWEEN 4 AND 5",
        '5-7 days': "days_since_wh_load BETWEEN 6 AND 7",
        '7-10 days': "days_since_wh_load BETWEEN 8 AND 10",
        '>10 days': "days_since_wh_load > 10"
    }

    condition = bucket_conditions.get(bucket_label)
    if not condition:
        conn.close()
        return pd.DataFrame()

    query = f"""
    WITH parsed AS (
        SELECT
            serial_no,
            vc_item_code,
            vc_item_desc,
            vc_shop_code,
            shop_name,
            wh_load_user,
            loaded_datetime,
            NULLIF(TRIM(loaded_datetime::text), '')::timestamp as load_dt,
            NULLIF(TRIM(dt_mod_date::text), '')::timestamp as mod_dt,
            (CURRENT_DATE - loaded_datetime::date) as days_since_wh_load
        FROM serial_no_dailydata
        WHERE serial_no IS NOT NULL
          AND (vc_serail_no IS NULL OR LENGTH(TRIM(vc_serail_no)) = 0)
          AND (shop_serail_no IS NULL OR LENGTH(TRIM(shop_serail_no)) = 0)
    )
    SELECT
        serial_no,
        vc_item_code,
        vc_item_desc,
        vc_shop_code,
        shop_name,
        wh_load_user,
        loaded_datetime,
        days_since_wh_load
    FROM parsed
    WHERE {condition}
    ORDER BY loaded_datetime DESC
    """

    if limit is not None:
        query += "\nLIMIT %s"

    try:
        if limit is None:
            df = pd.read_sql(query, conn)
        else:
            df = pd.read_sql(query, conn, params=(limit,))
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading not-offloaded details: {e}")
        conn.close()
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_offloading_pending_by_shop(bucket_label=None, limit=20):
    """Summary of shops with most offloading pending and mismatch rates"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()

    bucket_conditions = {
        '0-3 days': "days_since_wh_load BETWEEN 0 AND 3",
        '3-5 days': "days_since_wh_load BETWEEN 4 AND 5",
        '5-7 days': "days_since_wh_load BETWEEN 6 AND 7",
        '7-10 days': "days_since_wh_load BETWEEN 8 AND 10",
        '>10 days': "days_since_wh_load > 10"
    }
    bucket_filter = bucket_conditions.get(bucket_label)

    query = """
    WITH parsed AS (
        SELECT
            serial_no,
            vc_item_code,
            vc_item_desc,
            vc_shop_code,
            shop_name,
            vc_serail_no,
            shop_serail_no,
            shop_sold,
            vc_vehicle_no,
            "vc_vehicle_no.1" as shop_vehicle_no,
            loaded_datetime,
            (CURRENT_DATE - loaded_datetime::date) as days_since_wh_load
        FROM serial_no_dailydata
        WHERE vc_shop_code IS NOT NULL AND LENGTH(TRIM(vc_shop_code)) > 0
          AND serial_no IS NOT NULL
    )
    SELECT
        vc_shop_code as shop_code,
        COALESCE(shop_name, 'Unknown') as shop_name,
        COUNT(*) FILTER (
            WHERE vc_serail_no IS NULL OR LENGTH(TRIM(vc_serail_no)) = 0
        ) as pending_offload,
        COUNT(*) as total_loaded,
        ROUND(
            COUNT(*) FILTER (
                WHERE vc_shop_code IS NOT NULL
                  AND shop_sold IS NOT NULL
                  AND UPPER(TRIM(vc_shop_code)) != UPPER(TRIM(shop_sold))
            )::numeric / NULLIF(COUNT(*), 0) * 100, 1
        ) as shop_mismatch_pct,
        ROUND(
            COUNT(*) FILTER (
                WHERE serial_no IS NOT NULL
                  AND vc_serail_no IS NOT NULL
                  AND UPPER(TRIM(serial_no)) != UPPER(TRIM(vc_serail_no))
            )::numeric / NULLIF(COUNT(*), 0) * 100, 1
        ) as serial_mismatch_pct,
        ROUND(
            COUNT(*) FILTER (
                WHERE vc_vehicle_no IS NOT NULL
                  AND shop_vehicle_no IS NOT NULL
                  AND UPPER(TRIM(vc_vehicle_no)) != UPPER(TRIM(shop_vehicle_no))
            )::numeric / NULLIF(COUNT(*), 0) * 100, 1
        ) as vehicle_mismatch_pct
    FROM parsed
    WHERE 1=1
    """

    if bucket_filter:
        query += f"\n  AND {bucket_filter}"

        # Use bill_date and pass selected_date as a function argument or use an existing date variable
        query = f"""
                 SELECT COUNT(*) as total_serials,
                     COUNT(*) FILTER (WHERE serial_check = 'Y') as total_yes
            FROM serialno_check_yes_no
            WHERE DATE(bill_date) >= '{start_date}' AND DATE(bill_date) <= '{end_date}'
        """
        query += "\nLIMIT %s"

    try:
        if limit is None:
            df = pd.read_sql(query, conn)
        else:
            df = pd.read_sql(query, conn, params=(limit,))
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading offloading pending summary: {e}")
        conn.close()
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_discrepancy_summary():
    """Get overall discrepancy counts for mismatch categories"""
    conn = get_db_connection()
    if not conn:
        return None

    start_str, end_str = get_global_date_strs()

    query = """
    SELECT
        COUNT(*) FILTER (
            WHERE vc_shop_code IS NOT NULL
              AND shop_sold IS NOT NULL
              AND UPPER(TRIM(vc_shop_code)) != UPPER(TRIM(shop_sold))
        ) as shop_mismatch,
        COUNT(*) FILTER (
            WHERE serial_no IS NOT NULL
              AND vc_serail_no IS NOT NULL
              AND UPPER(TRIM(serial_no)) != UPPER(TRIM(vc_serail_no))
        ) as serial_mismatch,
        COUNT(*) FILTER (
            WHERE vc_vehicle_no IS NOT NULL
              AND "vc_vehicle_no.1" IS NOT NULL
              AND UPPER(TRIM(vc_vehicle_no)) != UPPER(TRIM("vc_vehicle_no.1"))
        ) as vehicle_mismatch,
        COUNT(*) FILTER (
            WHERE vc_serail_no IS NULL OR LENGTH(TRIM(vc_serail_no)) = 0
        ) as no_offloading,
        COUNT(*) FILTER (
            WHERE (vc_serail_no IS NULL OR LENGTH(TRIM(vc_serail_no)) = 0)
              AND shop_serail_no IS NOT NULL AND LENGTH(TRIM(shop_serail_no)) > 0
        ) as sold_without_offloading
    FROM serial_no_dailydata
    WHERE DATE(dt_doc_date) >= %s AND DATE(dt_doc_date) <= %s
    """

    try:
        df = pd.read_sql(query, conn, params=(start_str, end_str))
        conn.close()
        return df.iloc[0].to_dict() if not df.empty else None
    except Exception as e:
        st.error(f"Error loading discrepancy summary: {e}")
        conn.close()
        return None

@st.cache_data(ttl=300)
def get_sales_metrics(start_date, end_date):
    """Get sales metrics including duplicates, filtered by date range."""
    conn = get_db_connection()
    if not conn:
        return None

    start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
    end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)

    query = f"""
    WITH sales_data AS (
        SELECT 
            COUNT(*) FILTER (WHERE shop_serail_no IS NOT NULL) as total_sold,
            COUNT(DISTINCT shop_serail_no) FILTER (WHERE shop_serail_no IS NOT NULL) as unique_sold,
            -- Serials sold multiple times (fraud detection)
            COUNT(*) FILTER (
                WHERE shop_serail_no IN (
                    SELECT shop_serail_no 
                    FROM serial_no_dailydata 
                    WHERE shop_serail_no IS NOT NULL 
                    AND DATE(dt_doc_date) >= '{start_str}' AND DATE(dt_doc_date) <= '{end_str}'
                    GROUP BY shop_serail_no 
                    HAVING COUNT(*) > 1
                )
            ) as duplicate_sales_count,
            COUNT(DISTINCT shop_serail_no) FILTER (
                WHERE shop_serail_no IN (
                    SELECT shop_serail_no 
                    FROM serial_no_dailydata 
                    WHERE shop_serail_no IS NOT NULL 
                    AND DATE(dt_doc_date) >= '{start_str}' AND DATE(dt_doc_date) <= '{end_str}'
                    GROUP BY shop_serail_no 
                    HAVING COUNT(*) > 1
                )
            ) as unique_duplicate_serials
        FROM serial_no_dailydata
        WHERE DATE(dt_doc_date) >= '{start_str}' AND DATE(dt_doc_date) <= '{end_str}'
    )
    SELECT *,
        total_sold - unique_sold as duplicate_difference
    FROM sales_data
    """
    try:
        df = pd.read_sql(query, conn)
        conn.close()
        if not df.empty:
            result = df.iloc[0].to_dict()
            # Add compliance_pct: unique_sold / total_sold * 100 (safe division)
            total = result.get('total_sold', 0)
            unique = result.get('unique_sold', 0)
            if total > 0:
                result['compliance_pct'] = round((unique / total) * 100, 2)
            else:
                result['compliance_pct'] = 0.0
            return result
        else:
            return None
    except Exception as e:
        st.error(f"Error loading sales metrics: {e}")
        conn.close()
        return None


def clear_overview_caches():
    """Clear cached overview metrics so fresh uploads appear immediately."""
    cached_functions = [
        get_warehouse_metrics,
        get_shop_offload_metrics,
        get_sales_metrics,
        get_not_offloaded_metrics,
        get_discrepancy_summary,
    ]
    for fn in cached_functions:
        try:
            fn.clear()
        except AttributeError:
            pass

def get_serial_check_columns(conn):
    """Resolve column names for serial check queries."""
    try:
        cols_df = pd.read_sql(
            f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = '{SERIAL_CHECK_TABLE}'
            """,
            conn
        )
        cols = cols_df['column_name'].tolist()
    except Exception:
        cols = []

    if not cols:
        try:
            probe_df = pd.read_sql(f"SELECT * FROM {SERIAL_CHECK_TABLE} LIMIT 0", conn)
            cols = probe_df.columns.tolist()
        except Exception:
            cols = []

    cols_map = {c.lower(): c for c in cols}

    return {
        "serial_check_col": cols_map.get('serial_check'),
        "shop_col": (
            cols_map.get('shopcode') or
            cols_map.get('shop_code') or
            cols_map.get('shop') or
            cols_map.get('store_code') or
            cols_map.get('shop_sold')
        ),
        "cashier_col": (
            cols_map.get('cashier_name') or
            cols_map.get('cashier') or
            cols_map.get('user_name') or
            cols_map.get('till_number') or
            cols_map.get('till_no')
        ),
        "bill_col": (
            cols_map.get('bill_n') or
            cols_map.get('bill_no') or
            cols_map.get('bill_number') or
            cols_map.get('invoice_no')
        ),
        "serial_col": (
            cols_map.get('serial_number') or
            cols_map.get('serial_no') or
            cols_map.get('serial')
        ),
        "date_col": (
            cols_map.get('transaction_date') or
            cols_map.get('transaction_dt') or
            cols_map.get('trans_date') or
            cols_map.get('bill_date') or
            cols_map.get('invoice_date') or
            cols_map.get('sale_date') or
            cols_map.get('sales_date') or
            cols_map.get('dt_doc_date') or
            cols_map.get('dt_invoice_date') or
            cols_map.get('date')
        )
    }


@st.cache_data(ttl=300)
def get_serial_check_metrics(start_date=None, end_date=None):
    """Get serial check metrics (overall, shop-wise, cashier-wise) filtered by optional date range."""
    conn = get_db_connection()
    if not conn:
        return None

    cols = get_serial_check_columns(conn)
    serial_check_col = cols.get("serial_check_col")
    shop_col = cols.get("shop_col")
    cashier_col = cols.get("cashier_col")
    bill_col = cols.get("bill_col")
    serial_col = cols.get("serial_col")
    date_col = cols.get("date_col")

    start_obj = None
    end_obj = None
    if start_date:
        if isinstance(start_date, datetime):
            start_obj = start_date.date()
        elif isinstance(start_date, date):
            start_obj = start_date
        else:
            try:
                start_obj = datetime.strptime(str(start_date), '%Y-%m-%d').date()
            except Exception:
                start_obj = None
    if end_date:
        if isinstance(end_date, datetime):
            end_obj = end_date.date()
        elif isinstance(end_date, date):
            end_obj = end_date
        else:
            try:
                end_obj = datetime.strptime(str(end_date), '%Y-%m-%d').date()
            except Exception:
                end_obj = None
    if start_obj and end_obj and start_obj > end_obj:
        start_obj, end_obj = end_obj, start_obj
    start_filter = start_obj.strftime('%Y-%m-%d') if start_obj else None
    end_filter = end_obj.strftime('%Y-%m-%d') if end_obj else None

    if not serial_check_col:
        conn.close()
        return {
            "overall": None,
            "by_shop": None,
            "by_cashier": None,
            "missing_serial_check": True,
            "missing_date": date_col is None
        }

    def qcol(name):
        escaped = name.replace('"', '""')
        return f'"{escaped}"'

    serial_check_expr = qcol(serial_check_col)
    shop_expr = qcol(shop_col) if shop_col else None
    cashier_expr = qcol(cashier_col) if cashier_col else None
    date_expr = qcol(date_col) if date_col else None

    date_condition = None
    date_params = None
    if date_expr and start_filter and end_filter:
        date_condition = f"DATE({date_expr}) BETWEEN %s AND %s"
        date_params = (start_filter, end_filter)

    if date_condition:
        overall_where = f"WHERE {date_condition}"
        overall_params = date_params
    else:
        overall_where = ""
        overall_params = None

    overall_query = f"""
        SELECT
            COUNT(*) as total_serials,
            COUNT(CASE WHEN UPPER(TRIM({serial_check_expr})) = 'Y' THEN 1 END) as total_yes,
            ROUND(
                COUNT(CASE WHEN UPPER(TRIM({serial_check_expr})) = 'Y' THEN 1 END)::numeric /
                NULLIF(COUNT(*), 0) * 100,
                1
            ) as yes_pct
        FROM {SERIAL_CHECK_TABLE}
        {overall_where}
    """

    overall_df = pd.read_sql(overall_query, conn, params=overall_params)
    overall = overall_df.iloc[0].to_dict() if not overall_df.empty else None

    by_shop = None
    if shop_expr:
        shop_filters = [f"{shop_expr} IS NOT NULL AND TRIM({shop_expr}) <> ''"]
        shop_params = None
        if date_condition:
            shop_filters.append(date_condition)
            shop_params = date_params
        shop_where = " AND ".join(shop_filters)
        by_shop_query = f"""
            SELECT
                {shop_expr} as shop_code,
                COUNT(*) as total_serials,
                COUNT(CASE WHEN UPPER(TRIM({serial_check_expr})) = 'Y' THEN 1 END) as total_yes,
                ROUND(
                    COUNT(CASE WHEN UPPER(TRIM({serial_check_expr})) = 'Y' THEN 1 END)::numeric /
                    NULLIF(COUNT(*), 0) * 100,
                    1
                ) as yes_pct
            FROM {SERIAL_CHECK_TABLE}
            WHERE {shop_where}
            GROUP BY {shop_expr}
            ORDER BY yes_pct ASC NULLS FIRST, total_serials DESC
        """
        by_shop = pd.read_sql(by_shop_query, conn, params=shop_params)

    by_cashier = None
    if cashier_expr:
        cashier_filters = [f"{cashier_expr} IS NOT NULL AND TRIM({cashier_expr}) <> ''"]
        cashier_params = None
        if date_condition:
            cashier_filters.append(date_condition)
            cashier_params = date_params
        cashier_where = " AND ".join(cashier_filters)
        by_cashier_query = f"""
            SELECT
                {cashier_expr} as cashier,
                COUNT(*) as total_serials,
                COUNT(CASE WHEN UPPER(TRIM({serial_check_expr})) = 'Y' THEN 1 END) as total_yes,
                ROUND(
                    COUNT(CASE WHEN UPPER(TRIM({serial_check_expr})) = 'Y' THEN 1 END)::numeric /
                    NULLIF(COUNT(*), 0) * 100,
                    1
                ) as yes_pct
            FROM {SERIAL_CHECK_TABLE}
            WHERE {cashier_where}
            GROUP BY {cashier_expr}
            ORDER BY yes_pct ASC NULLS FIRST, total_serials DESC
        """
        by_cashier = pd.read_sql(by_cashier_query, conn, params=cashier_params)

    dup_by_cashier = None
    if cashier_expr and bill_col and serial_col:
        bill_expr = qcol(bill_col)
        serial_expr = qcol(serial_col)
        date_clause = f" AND {date_condition}" if date_condition else ""
        dup_by_cashier_query = f"""
            SELECT
                {cashier_expr} as cashier,
                {serial_expr} as serial_number,
                COUNT(DISTINCT {bill_expr}) as bill_count,
                STRING_AGG(DISTINCT {bill_expr}::text, ', ' ORDER BY {bill_expr}::text) as bills
            FROM {SERIAL_CHECK_TABLE}
            WHERE {cashier_expr} IS NOT NULL AND TRIM({cashier_expr}) <> ''
              AND {serial_expr} IS NOT NULL AND TRIM({serial_expr}) <> ''
              AND {bill_expr} IS NOT NULL AND TRIM({bill_expr}::text) <> ''
              {date_clause}
            GROUP BY {cashier_expr}, {serial_expr}
            HAVING COUNT(DISTINCT {bill_expr}) > 1
            ORDER BY bill_count DESC, cashier
        """
        dup_by_cashier = pd.read_sql(dup_by_cashier_query, conn, params=date_params if date_condition else None)

    by_date_last11 = None
    if date_expr:
        end_date = datetime.today().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=10)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        by_date_query = f"""
            WITH base AS (
                SELECT
                    DATE({date_expr}) as activity_date,
                    CASE WHEN UPPER(TRIM({serial_check_expr})) = 'Y' THEN 1 ELSE 0 END as is_yes
                FROM {SERIAL_CHECK_TABLE}
                WHERE {date_expr} IS NOT NULL
                  AND DATE({date_expr}) BETWEEN %s AND %s
            )
            SELECT
                activity_date,
                COUNT(*) as total_serials,
                SUM(is_yes) as total_yes,
                ROUND(
                    SUM(is_yes)::numeric / NULLIF(COUNT(*), 0) * 100,
                    1
                ) as yes_pct
            FROM base
            GROUP BY activity_date
            ORDER BY activity_date ASC
        """
        by_date_last11 = pd.read_sql(by_date_query, conn, params=(start_str, end_str))

    conn.close()
    return {
        "overall": overall,
        "by_shop": by_shop,
        "by_cashier": by_cashier,
        "dup_by_cashier": dup_by_cashier,
        "missing_serial_check": False,
        "missing_shop": shop_expr is None,
        "missing_cashier": cashier_expr is None,
        "missing_bill": bill_col is None,
        "missing_serial": serial_col is None,
        "missing_date": date_expr is None,
        "by_date_last11": by_date_last11
    }


@st.cache_data(ttl=300)
def get_shop_cashier_compliance_trend(shop_code, days=7):
    """Return last N days compliance % (Y) for a shop and its cashiers."""
    conn = get_db_connection()
    if not conn:
        return None

    cols = get_serial_check_columns(conn)
    serial_check_col = cols.get("serial_check_col")
    shop_col = cols.get("shop_col")
    cashier_col = cols.get("cashier_col")
    date_col = cols.get("date_col")

    if not serial_check_col or not shop_col or not date_col:
        conn.close()
        return {
            "data": None,
            "missing_serial_check": serial_check_col is None,
            "missing_shop": shop_col is None,
            "missing_date": date_col is None,
            "missing_cashier": cashier_col is None
        }

    def qcol(name):
        escaped = name.replace('"', '""')
        return f'"{escaped}"'

    serial_check_expr = qcol(serial_check_col)
    shop_expr = qcol(shop_col)
    cashier_expr = qcol(cashier_col) if cashier_col else None
    date_expr = qcol(date_col)

    end_date = datetime.today().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)

    cashier_select = cashier_expr if cashier_expr else "NULL"
    group_by_clause = f"GROUP BY DATE({date_expr}), {cashier_expr}" if cashier_expr else f"GROUP BY DATE({date_expr})"

    query = f"""
        SELECT
            DATE({date_expr}) as activity_date,
            {cashier_select} as cashier,
            COUNT(*) as total_serials,
            SUM(CASE WHEN UPPER(TRIM({serial_check_expr})) = 'Y' THEN 1 ELSE 0 END) as total_yes
        FROM {SERIAL_CHECK_TABLE}
        WHERE UPPER(TRIM({shop_expr})) = UPPER(TRIM(%s))
          AND DATE({date_expr}) BETWEEN %s AND %s
        {group_by_clause}
        ORDER BY activity_date ASC
    """

    try:
        df = pd.read_sql(query, conn, params=(shop_code, start_date, end_date))
        conn.close()
        return {
            "data": df,
            "start_date": start_date,
            "end_date": end_date,
            "missing_serial_check": False,
            "missing_shop": False,
            "missing_date": False,
            "missing_cashier": cashier_col is None
        }
    except Exception as e:
        st.error(f"Error loading shop compliance trend: {e}")
        conn.close()
        return None

@st.cache_data(ttl=120)
def get_cashier_date_detail_serials(shop_code, cashier, activity_date):
    """Get detailed serial records for a specific cashier and date."""
    conn = get_db_connection()
    if not conn:
        return None

    cols = get_serial_check_columns(conn)
    serial_check_col = cols.get("serial_check_col")
    shop_col = cols.get("shop_col")
    cashier_col = cols.get("cashier_col")
    date_col = cols.get("date_col")

    if not serial_check_col or not shop_col or not date_col:
        conn.close()
        return None

    def qcol(name):
        escaped = name.replace('"', '""')
        return f'"{escaped}"'

    # Build query to get all serials for this cashier on this date
    query = f"""
        SELECT
            {qcol(serial_check_col)} as serial_check,
            item_code,
            item_name,
            serial_number,
            DATE({qcol(date_col)}) as bill_date,
            {qcol(shop_col)} as shop_code,
            {qcol(cashier_col) if cashier_col else 'NULL'} as till_number
        FROM {SERIAL_CHECK_TABLE}
        WHERE UPPER(TRIM({qcol(shop_col)})) = UPPER(TRIM(%s))
          AND DATE({qcol(date_col)}) = %s
    """
    
    # Add cashier filter only if cashier is not the shop itself
    if cashier_col and cashier != shop_code:
        query += f" AND UPPER(TRIM({qcol(cashier_col)})) = UPPER(TRIM(%s))"
        params = (shop_code, activity_date, cashier)
    else:
        params = (shop_code, activity_date)
    
    # Sort: N first, then Y
    query += f"""
        ORDER BY 
            CASE WHEN UPPER(TRIM({qcol(serial_check_col)})) = 'N' THEN 0 ELSE 1 END,
            item_code
    """

    try:
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        
        if df is not None and not df.empty:
            # Clean up serial_check column
            df['serial_check'] = df['serial_check'].str.strip().str.upper()
        else:
            df = pd.DataFrame()
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading cashier detail serials: {e}\n\nQuery: {query}\n\nParams: {params}")
        if conn:
            conn.close()
        return None
    
@st.cache_data(ttl=300)
def get_duplicate_sales_details():
    """Get details of serials sold multiple times"""
    conn = get_db_connection()
    if not conn:
        return None

    start_str, end_str = get_global_date_strs()
    
    query = """
    SELECT 
        shop_serail_no as serial_number,
        COUNT(*) as times_sold,
        STRING_AGG(DISTINCT shop_sold, ', ') as shops_sold,
        STRING_AGG(DISTINCT vc_item_code, ', ') as item_codes,
        STRING_AGG(DISTINCT vc_item_desc, ', ') as item_names,
        MIN(dt_invoice_date) as first_sale_date,
        MAX(dt_invoice_date) as last_sale_date
    FROM serial_no_dailydata
        WHERE shop_serail_no IS NOT NULL
            AND DATE(dt_doc_date) >= %s AND DATE(dt_doc_date) <= %s
    GROUP BY shop_serail_no
    HAVING COUNT(*) > 1
    ORDER BY times_sold DESC, last_sale_date DESC
    LIMIT 100
    """
    
    try:
        df = pd.read_sql(query, conn, params=(start_str, end_str))
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading duplicate sales details: {e}")
        conn.close()
        return None

@st.cache_data(ttl=300)
def get_duplicate_serials_by_item():
    """Get items with duplicate serial numbers at WH level"""
    conn = get_db_connection()
    if not conn:
        return None

@st.cache_data(ttl=300)
def get_duplicate_serials_with_remarks(limit=1000):
    """Get duplicate serials with consolidated remark based on column mismatches"""
    conn = get_db_connection()
    if not conn:
        return None

    query = """
    WITH base AS (
        SELECT 
            LOWER(TRIM(serial_no)) as wh_serial,
            LOWER(TRIM(vc_serail_no)) as offload_serial,
            LOWER(TRIM(shop_serail_no)) as sold_serial,
            vc_item_code,
            vc_item_desc,
            vc_wh_code,
            vc_shop_code,
            shop_sold,
            dt_mod_date,
            dt_invoice_date
        FROM serial_no_dailydata
        WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
    ),
    dup AS (
        SELECT
            wh_serial as serial_key,
            COUNT(*) as duplicate_count,
            COUNT(DISTINCT vc_item_code) as item_count,
            STRING_AGG(DISTINCT vc_item_code, ', ' ORDER BY vc_item_code) FILTER (
                WHERE vc_item_code IS NOT NULL AND LENGTH(TRIM(vc_item_code)) > 0
            ) as item_codes,
            COUNT(DISTINCT vc_wh_code) as wh_code_count,
            STRING_AGG(DISTINCT vc_wh_code, ', ' ORDER BY vc_wh_code) FILTER (
                WHERE vc_wh_code IS NOT NULL AND LENGTH(TRIM(vc_wh_code)) > 0
            ) as wh_codes,
            COUNT(DISTINCT vc_shop_code) as shop_count,
            STRING_AGG(DISTINCT vc_shop_code, ', ' ORDER BY vc_shop_code) FILTER (
                WHERE vc_shop_code IS NOT NULL AND LENGTH(TRIM(vc_shop_code)) > 0
            ) as offload_shops,
            COUNT(DISTINCT shop_sold) as sold_shop_count,
            STRING_AGG(DISTINCT shop_sold, ', ' ORDER BY shop_sold) FILTER (
                WHERE shop_sold IS NOT NULL AND LENGTH(TRIM(shop_sold)) > 0
            ) as sold_shops,
            COUNT(DISTINCT dt_mod_date) as offload_time_count,
            COUNT(DISTINCT dt_invoice_date) as sold_time_count,
            COUNT(DISTINCT offload_serial) FILTER (
                WHERE offload_serial IS NOT NULL AND LENGTH(TRIM(offload_serial)) > 0
            ) as offload_serial_count,
            COUNT(DISTINCT sold_serial) FILTER (
                WHERE sold_serial IS NOT NULL AND LENGTH(TRIM(sold_serial)) > 0
            ) as sold_serial_count,
            COUNT(*) FILTER (
                WHERE offload_serial IS NOT NULL AND LENGTH(TRIM(offload_serial)) > 0
                  AND offload_serial <> wh_serial
            ) as wh_offload_mismatch_count,
            COUNT(*) FILTER (
                WHERE sold_serial IS NOT NULL AND LENGTH(TRIM(sold_serial)) > 0
                  AND sold_serial <> wh_serial
            ) as wh_sold_mismatch_count,
            COUNT(*) FILTER (
                WHERE offload_serial IS NOT NULL AND LENGTH(TRIM(offload_serial)) > 0
                  AND sold_serial IS NOT NULL AND LENGTH(TRIM(sold_serial)) > 0
                  AND offload_serial <> sold_serial
            ) as offload_sold_mismatch_count,
                        COUNT(*) FILTER (
                                WHERE vc_shop_code IS NOT NULL AND LENGTH(TRIM(vc_shop_code)) > 0
                                    AND shop_sold IS NOT NULL AND LENGTH(TRIM(shop_sold)) > 0
                                    AND LOWER(TRIM(vc_shop_code)) <> LOWER(TRIM(shop_sold))
                        ) as offload_sold_shop_mismatch_count,
            COUNT(*) FILTER (
                WHERE sold_serial IS NOT NULL AND LENGTH(TRIM(sold_serial)) > 0
            ) as sold_records
        FROM base
        GROUP BY wh_serial
        HAVING COUNT(*) > 1
    )
    SELECT
        serial_key as serial_number,
        duplicate_count,
        item_codes,
        offload_shops,
        sold_shops,
        item_count,
        wh_code_count,
        shop_count,
        sold_shop_count,
        offload_time_count,
        sold_time_count,
        wh_codes,
        COALESCE(NULLIF(CONCAT_WS('; ',
            CASE WHEN wh_offload_mismatch_count > 0 THEN 'WH vs Offload mismatch' END,
            CASE WHEN wh_sold_mismatch_count > 0 THEN 'WH vs Sold mismatch' END,
            CASE WHEN offload_sold_mismatch_count > 0 THEN 'Offload vs Sold mismatch' END,
            CASE WHEN item_count > 1 THEN 'Item mismatch' END,
            CASE WHEN wh_code_count > 1 THEN 'WH code mismatch' END,
            CASE WHEN shop_count > 1 THEN 'Multiple Shop loaded from WH' END,
            CASE WHEN offload_sold_shop_mismatch_count > 0 THEN 'Shop mismatch' END,
            CASE WHEN sold_shop_count > 1 THEN 'Shop sold mismatch' END,
            CASE WHEN offload_time_count > 1 THEN 'Offload time diff' END,
            CASE WHEN sold_time_count > 1 THEN 'Sold time diff' END,
            CASE 
                WHEN sold_records = 0
                 AND wh_offload_mismatch_count = 0
                 AND wh_sold_mismatch_count = 0
                 AND offload_sold_mismatch_count = 0
                 AND offload_sold_shop_mismatch_count = 0
                 AND item_count <= 1
                 AND wh_code_count <= 1
                 AND shop_count <= 1
                 AND sold_shop_count <= 1
                 AND offload_time_count <= 1
                 AND sold_time_count <= 1
                THEN 'Not sold'
            END
        ), ''), 'Duplicate only') as remark
    FROM dup
    ORDER BY duplicate_count DESC, serial_key
    LIMIT %s
    """

    try:
        df = pd.read_sql(query, conn, params=(limit,))
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading duplicate serials with remarks: {e}")
        conn.close()
        return None
    
    query = """
    SELECT 
        vc_item_code as item_code,
        vc_item_desc as item_name,
        COUNT(DISTINCT serial_no) as unique_serials,
        COUNT(*) as total_records,
        COUNT(*) - COUNT(DISTINCT serial_no) as duplicate_count,
        STRING_AGG(DISTINCT wh_load_user, ', ') as loaders,
        MIN(loaded_datetime) as first_load,
        MAX(loaded_datetime) as last_load
    FROM serial_no_dailydata
    WHERE serial_no IS NOT NULL
      AND serial_no IN (
          SELECT serial_no 
          FROM serial_no_dailydata 
          GROUP BY serial_no 
          HAVING COUNT(*) > 1
      )
    GROUP BY vc_item_code, vc_item_desc
    ORDER BY duplicate_count DESC
    LIMIT 50
    """
    
    try:
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading duplicate serials by item: {e}")
        conn.close()
        return None

@st.cache_data(ttl=300)
def get_shop_performance():
    """Get shop-level performance metrics"""
    conn = get_db_connection()
    if not conn:
        return None
    
    query = """
    SELECT 
        COALESCE(vc_shop_code, shop_sold, 'Unknown') as shop_code,
        COUNT(*) as total_items,
        COUNT(*) FILTER (WHERE vc_serail_no IS NOT NULL) as offloaded_count,
        COUNT(*) FILTER (WHERE shop_serail_no IS NOT NULL) as sold_count,
        
        -- Discrepancies
        COUNT(*) FILTER (
            WHERE serial_no IS NOT NULL 
              AND (vc_serail_no IS NULL OR LENGTH(TRIM(vc_serail_no)) = 0)
              AND (shop_serail_no IS NULL OR LENGTH(TRIM(shop_serail_no)) = 0)
        ) as not_offloaded,
        
        COUNT(*) FILTER (
            WHERE vc_shop_code IS NOT NULL 
              AND shop_sold IS NOT NULL 
              AND UPPER(TRIM(vc_shop_code)) != UPPER(TRIM(shop_sold))
        ) as shop_mismatch,
        
        COUNT(*) FILTER (
            WHERE serial_no IS NOT NULL 
              AND shop_serail_no IS NOT NULL
              AND UPPER(TRIM(serial_no)) != UPPER(TRIM(shop_serail_no))
        ) as serial_mismatch,
        
        -- Average offload time (dt_mod_date - dt_doc_date)
        ROUND(AVG(
            CASE 
                WHEN dt_mod_date IS NOT NULL AND dt_doc_date IS NOT NULL 
                THEN dt_mod_date::DATE - dt_doc_date::DATE 
                ELSE NULL 
            END
        ), 1) as avg_offload_days
        
    FROM serial_no_dailydata
    WHERE serial_no IS NOT NULL
    GROUP BY COALESCE(vc_shop_code, shop_sold, 'Unknown')
    ORDER BY total_items DESC
    """
    
    try:
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading shop performance: {e}")
        conn.close()
        return None

# ====================== SERIAL JOURNEY FUNCTIONS ======================
def _get_table_columns(conn, table_name):
    try:
        cols_df = pd.read_sql(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            """,
            conn,
            params=(table_name,)
        )
        cols = cols_df['column_name'].tolist()
        return {c.lower(): c for c in cols}
    except Exception:
        return {}

def _qcol(name):
    escaped = name.replace('"', '""')
    return f'"{escaped}"'

def _col_or_null(name, cast_type="text"):
    return _qcol(name) if name else f"NULL::{cast_type}"

@st.cache_data(ttl=300)
def get_serial_journey_overview(start_date, end_date):
    """Get overview metrics for serial journey"""
    conn = get_db_connection()
    if not conn:
        return {}

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    try:
        wh_query = """
            SELECT COUNT(DISTINCT serial_no) as received_count
            FROM whreceived_serialno
        """
        wh_df = pd.read_sql(wh_query, conn)
        received_count = int(wh_df.iloc[0]['received_count']) if not wh_df.empty else 0

        loaded_query = """
            SELECT 
                COUNT(DISTINCT serial_no) as loaded_count,
                COUNT(DISTINCT CASE WHEN vc_serail_no IS NOT NULL AND vc_serail_no != '' THEN serial_no END) as offloaded_count,
                COUNT(DISTINCT CASE WHEN shop_serail_no IS NOT NULL AND shop_serail_no != '' THEN serial_no END) as sold_count,
                COUNT(DISTINCT vc_shop_code) as shops_sent_to,
                COUNT(DISTINCT CASE WHEN vc_shop_code != shop_sold THEN serial_no END) as shop_mismatch_count,
                COUNT(DISTINCT CASE WHEN vc_vehicle_no != "vc_vehicle_no.1" THEN serial_no END) as vehicle_mismatch_count
            FROM serial_no_dailydata
        """
        loaded_df = pd.read_sql(loaded_query, conn)
        loaded_row = loaded_df.iloc[0] if not loaded_df.empty else {}

        serial_cols = _get_table_columns(conn, SERIAL_CHECK_TABLE)
        serial_check_col = serial_cols.get('serial_check')
        bill_date_col = (
            serial_cols.get('bill_date') or
            serial_cols.get('date_invoice') or
            serial_cols.get('dt_invoice_date') or
            serial_cols.get('invoice_date')
        )

        in_main_db = 0
        not_in_main_db = 0
        total_checked = 0

        if serial_check_col:
            serial_check_expr = _qcol(serial_check_col)
            if bill_date_col:
                bill_date_expr = _qcol(bill_date_col)
                db_query = f"""
                    SELECT 
                        COUNT(*) as total_checked,
                        SUM(CASE WHEN UPPER(TRIM({serial_check_expr})) = 'Y' THEN 1 ELSE 0 END) as in_main_db,
                        SUM(CASE WHEN UPPER(TRIM({serial_check_expr})) != 'Y' OR {serial_check_expr} IS NULL THEN 1 ELSE 0 END) as not_in_main_db
                    FROM {SERIAL_CHECK_TABLE}
                    WHERE {bill_date_expr} >= %s AND {bill_date_expr} <= %s
                """
                db_df = pd.read_sql(db_query, conn, params=(start_str, end_str))
            else:
                db_query = f"""
                    SELECT 
                        COUNT(*) as total_checked,
                        SUM(CASE WHEN UPPER(TRIM({serial_check_expr})) = 'Y' THEN 1 ELSE 0 END) as in_main_db,
                        SUM(CASE WHEN UPPER(TRIM({serial_check_expr})) != 'Y' OR {serial_check_expr} IS NULL THEN 1 ELSE 0 END) as not_in_main_db
                    FROM {SERIAL_CHECK_TABLE}
                """
                db_df = pd.read_sql(db_query, conn)

            if not db_df.empty:
                total_checked = int(db_df.iloc[0]['total_checked'] or 0)
                in_main_db = int(db_df.iloc[0]['in_main_db'] or 0)
                not_in_main_db = int(db_df.iloc[0]['not_in_main_db'] or 0)

        result = {
            "received_count": received_count,
            "loaded_count": int(loaded_row.get('loaded_count', 0) or 0),
            "offloaded_count": int(loaded_row.get('offloaded_count', 0) or 0),
            "sold_count": int(loaded_row.get('sold_count', 0) or 0),
            "shops_sent_to": int(loaded_row.get('shops_sent_to', 0) or 0),
            "shop_mismatch_count": int(loaded_row.get('shop_mismatch_count', 0) or 0),
            "vehicle_mismatch_count": int(loaded_row.get('vehicle_mismatch_count', 0) or 0),
            "total_checked": total_checked,
            "in_main_db": in_main_db,
            "not_in_main_db": not_in_main_db
        }

        result["not_loaded"] = result["received_count"] - result["loaded_count"]
        result["not_offloaded"] = result["loaded_count"] - result["offloaded_count"]
        result["not_sold"] = result["offloaded_count"] - result["sold_count"]

        return result
    except Exception as e:
        st.error(f"Error loading serial journey overview: {e}")
        return {}
    finally:
        conn.close()

@st.cache_data(ttl=300)
def search_serial_number(serial_no):
    """Get complete journey of a specific serial number"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()

    try:
        serial_cols = _get_table_columns(conn, SERIAL_CHECK_TABLE)
        wh_cols = _get_table_columns(conn, "whreceived_serialno")

        sc_serial_col = (
            serial_cols.get('serial_number') or
            serial_cols.get('serial_no') or
            serial_cols.get('serial')
        )
        sc_check_col = serial_cols.get('serial_check')
        sc_bill_date_col = (
            serial_cols.get('bill_date') or
            serial_cols.get('date_invoice') or
            serial_cols.get('dt_invoice_date') or
            serial_cols.get('invoice_date')
        )
        sc_bill_no_col = (
            serial_cols.get('bill_no') or
            serial_cols.get('bill_n') or
            serial_cols.get('bill_number') or
            serial_cols.get('invoice_no')
        )
        sc_shop_col = (
            serial_cols.get('shop_code') or
            serial_cols.get('shopcode') or
            serial_cols.get('shop') or
            serial_cols.get('store_code') or
            serial_cols.get('shop_sold')
        )
        sc_till_col = (
            serial_cols.get('till_number') or
            serial_cols.get('till_no') or
            serial_cols.get('till')
        )
        sc_cashier_col = (
            serial_cols.get('cashier_name') or
            serial_cols.get('cashier') or
            serial_cols.get('user_name')
        )

        wh_serial_col = wh_cols.get('serial_no') or wh_cols.get('serial') or 'serial_no'
        wh_grn_col = wh_cols.get('grn_date')
        wh_warehouse_col = (
            wh_cols.get('warehouse_name') or
            wh_cols.get('wh_received_warehouse') or
            wh_cols.get('warehouse')
        )
        wh_supplier_col = (
            wh_cols.get('supp_name') or
            wh_cols.get('supplier_name') or
            wh_cols.get('supplier')
        )
        wh_inbound_col = wh_cols.get('inbound_type')

        sc_join = f"LEFT JOIN {SERIAL_CHECK_TABLE} sc ON 1=0"
        if sc_serial_col:
            sc_join = f"FULL OUTER JOIN {SERIAL_CHECK_TABLE} sc ON UPPER(TRIM(COALESCE(sd.shop_serail_no, sd.vc_serail_no, sd.serial_no))) = UPPER(TRIM(sc.{_qcol(sc_serial_col)}))"

        query = f"""
            WITH serial_data AS (
                SELECT 
                    { _col_or_null(wh_grn_col, 'date') } as wh_grn_date,
                    { _col_or_null(wh_warehouse_col) } as wh_received_warehouse,
                    { _col_or_null(wh_supplier_col) } as supplier_name,
                    { _col_or_null(wh_inbound_col) } as inbound_type,
                    sd.dt_doc_date as wh_doc_date,
                    sd.loaded_datetime,
                    sd.vc_shop_code as sent_to_shop,
                    sd.shop_name as sent_to_shop_name,
                    sd.vc_item_code,
                    sd.vc_item_desc,
                    sd.nu_selling_price,
                    sd.serial_no as wh_serial,
                    sd.vc_serail_no as offload_serial,
                    sd.shop_serail_no as sold_serial,
                    sd.vc_vehicle_no as wh_vehicle,
                    sd."vc_vehicle_no.1" as shop_vehicle,
                    sd.shop_sold,
                    sd.vc_invoice_no as sale_invoice,
                    {"COALESCE(sd.dt_invoice_date, sc." + _qcol(sc_bill_date_col) + ")" if sc_bill_date_col else "sd.dt_invoice_date"} as sale_date,
                    { _col_or_null(sc_bill_date_col, 'date') } as bill_date,
                    { _col_or_null(sc_bill_no_col) } as bill_no,
                    { _col_or_null(sc_shop_col) } as sold_at_shop,
                    { _col_or_null(sc_till_col) } as till_number,
                    { _col_or_null(sc_cashier_col) } as cashier_name,
                    { _col_or_null(sc_check_col) } as in_main_db,
                    CASE 
                        WHEN sd.vc_shop_code IS NULL THEN 'Not Loaded to Shop'
                        WHEN sd.vc_serail_no IS NULL OR sd.vc_serail_no = '' THEN 'Not Offloaded'
                        WHEN sd.shop_serail_no IS NULL OR sd.shop_serail_no = '' THEN 'Not Sold'
                        ELSE 'Sold'
                    END as status,
                    CASE WHEN sd.vc_shop_code != sd.shop_sold THEN 'YES' ELSE 'NO' END as has_shop_mismatch,
                    CASE WHEN sd.vc_vehicle_no != sd."vc_vehicle_no.1" THEN 'YES' ELSE 'NO' END as has_vehicle_mismatch,
                    {"CASE WHEN UPPER(TRIM(sc." + _qcol(sc_check_col) + ")) = 'Y' THEN 'YES' ELSE 'NO' END" if sc_check_col else "'NO'"} as verified_in_main_db,
                    CASE 
                        WHEN { _col_or_null(sc_serial_col) } IS NOT NULL THEN 'YES'
                        ELSE 'NO'
                    END as sold_status
                FROM whreceived_serialno wr
                FULL OUTER JOIN serial_no_dailydata sd ON UPPER(TRIM(wr.{_qcol(wh_serial_col)})) = UPPER(TRIM(sd.serial_no))
                {sc_join}
                WHERE UPPER(TRIM(COALESCE(wr.{_qcol(wh_serial_col)}, sd.serial_no, {"sc." + _qcol(sc_serial_col) if sc_serial_col else "''"}))) = UPPER(TRIM(%s))
            )
            SELECT 
                wh_grn_date, wh_received_warehouse, supplier_name, inbound_type,
                wh_doc_date, loaded_datetime, sent_to_shop, sent_to_shop_name,
                vc_item_code, vc_item_desc, nu_selling_price,
                wh_serial, offload_serial, sold_serial,
                wh_vehicle, shop_vehicle,
                STRING_AGG(DISTINCT shop_sold, ', ') as shop_sold,
                sale_invoice,
                MAX(sale_date) as sale_date,
                MAX(bill_date) as bill_date,
                STRING_AGG(DISTINCT bill_no::text, ', ') as bill_no,
                STRING_AGG(DISTINCT sold_at_shop::text, ', ') as sold_at_shop,
                STRING_AGG(DISTINCT till_number::text, ', ') as till_number,
                STRING_AGG(DISTINCT cashier_name::text, ', ') as cashier_name,
                MAX(in_main_db) as in_main_db,
                MAX(status) as status,
                MAX(has_shop_mismatch) as has_shop_mismatch,
                MAX(has_vehicle_mismatch) as has_vehicle_mismatch,
                MAX(verified_in_main_db) as verified_in_main_db,
                MAX(sold_status) as sold_status
            FROM serial_data
            GROUP BY 
                wh_grn_date, wh_received_warehouse, supplier_name, inbound_type,
                wh_doc_date, loaded_datetime, sent_to_shop, sent_to_shop_name,
                vc_item_code, vc_item_desc, nu_selling_price,
                wh_serial, offload_serial, sold_serial,
                wh_vehicle, shop_vehicle, sale_invoice
        """

        results = pd.read_sql(query, conn, params=(serial_no,))
        return results if results is not None else pd.DataFrame()
    except Exception as e:
        st.error(f"Error searching serial number: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_gap_analysis_by_shop(start_date, end_date):
    """Get gap analysis showing loaded → offloaded → sold by shop"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    query = """
        SELECT 
            vc_shop_code,
            shop_name,
            COUNT(DISTINCT serial_no) as loaded_count,
            COUNT(DISTINCT CASE WHEN vc_serail_no IS NOT NULL AND vc_serail_no != '' THEN serial_no END) as offloaded_count,
            COUNT(DISTINCT CASE WHEN shop_serail_no IS NOT NULL AND shop_serail_no != '' THEN serial_no END) as sold_count,
            COUNT(DISTINCT CASE WHEN vc_shop_code != shop_sold THEN serial_no END) as shop_mismatch,
            COUNT(DISTINCT CASE WHEN vc_vehicle_no != "vc_vehicle_no.1" THEN serial_no END) as vehicle_mismatch,
            COUNT(DISTINCT CASE WHEN vc_serail_no IS NULL OR vc_serail_no = '' THEN serial_no END) as not_offloaded,
            COUNT(DISTINCT CASE WHEN vc_serail_no IS NOT NULL AND (shop_serail_no IS NULL OR shop_serail_no = '') THEN serial_no END) as offloaded_not_sold
        FROM serial_no_dailydata
        WHERE DATE(dt_doc_date) >= %s AND DATE(dt_doc_date) <= %s
          AND vc_shop_code IS NOT NULL
        GROUP BY vc_shop_code, shop_name
        ORDER BY loaded_count DESC
    """

    try:
        df = pd.read_sql(query, conn, params=(start_str, end_str))
        return df
    except Exception as e:
        st.error(f"Error loading gap analysis: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_supplier_analysis(start_date, end_date):
    """Get supplier-wise serial performance analysis"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    wh_cols = _get_table_columns(conn, "whreceived_serialno")
    supp_col = (
        wh_cols.get('supp_name') or
        wh_cols.get('supplier_name') or
        wh_cols.get('supplier')
    )
    supp_expr = _col_or_null(supp_col)

    query = f"""
        SELECT 
            COALESCE({supp_expr}, 'Unknown Supplier') as supplier_name,
            COUNT(DISTINCT sd.serial_no) as total_serials,
            COUNT(DISTINCT CASE WHEN sd.vc_serail_no IS NOT NULL AND sd.vc_serail_no != '' THEN sd.serial_no END) as offloaded_count,
            COUNT(DISTINCT CASE WHEN sd.shop_serail_no IS NOT NULL AND sd.shop_serail_no != '' THEN sd.serial_no END) as sold_count,
            COUNT(DISTINCT CASE WHEN sd.vc_shop_code != sd.shop_sold THEN sd.serial_no END) as shop_mismatches,
            COUNT(DISTINCT CASE WHEN sd.vc_vehicle_no != sd."vc_vehicle_no.1" THEN sd.serial_no END) as vehicle_mismatches,
            ROUND(COUNT(DISTINCT CASE WHEN sd.vc_serail_no IS NOT NULL THEN sd.serial_no END)::numeric / NULLIF(COUNT(DISTINCT sd.serial_no), 0) * 100, 2) as offload_rate,
            ROUND(COUNT(DISTINCT CASE WHEN sd.shop_serail_no IS NOT NULL THEN sd.serial_no END)::numeric / NULLIF(COUNT(DISTINCT sd.serial_no), 0) * 100, 2) as sale_rate
        FROM serial_no_dailydata sd
        LEFT JOIN whreceived_serialno wr ON UPPER(TRIM(sd.serial_no)) = UPPER(TRIM(wr.serial_no))
        WHERE DATE(sd.dt_doc_date) >= %s AND DATE(sd.dt_doc_date) <= %s
        GROUP BY COALESCE({supp_expr}, 'Unknown Supplier')
        ORDER BY total_serials DESC
    """

    try:
        df = pd.read_sql(query, conn, params=(start_str, end_str))
        return df
    except Exception as e:
        st.error(f"Error loading supplier analysis: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_brand_analysis(start_date, end_date):
    """Get brand-wise serial performance analysis"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    query = """
        SELECT 
            SPLIT_PART(vc_item_desc, ' ', 1) as brand,
            COUNT(DISTINCT serial_no) as total_serials,
            COUNT(DISTINCT CASE WHEN vc_serail_no IS NOT NULL AND vc_serail_no != '' THEN serial_no END) as offloaded_count,
            COUNT(DISTINCT CASE WHEN shop_serail_no IS NOT NULL AND shop_serail_no != '' THEN serial_no END) as sold_count,
            COUNT(DISTINCT CASE WHEN vc_shop_code != shop_sold THEN serial_no END) as shop_mismatches,
            COUNT(DISTINCT CASE WHEN vc_vehicle_no != "vc_vehicle_no.1" THEN serial_no END) as vehicle_mismatches,
            COUNT(DISTINCT vc_shop_code) as shops_sent_to,
            ROUND(COUNT(DISTINCT CASE WHEN vc_serail_no IS NOT NULL THEN serial_no END)::numeric / NULLIF(COUNT(DISTINCT serial_no), 0) * 100, 2) as offload_rate,
            ROUND(COUNT(DISTINCT CASE WHEN shop_serail_no IS NOT NULL THEN serial_no END)::numeric / NULLIF(COUNT(DISTINCT serial_no), 0) * 100, 2) as sale_rate,
            ROUND(AVG(nu_selling_price), 2) as avg_selling_price
        FROM serial_no_dailydata
        WHERE DATE(dt_doc_date) >= %s AND DATE(dt_doc_date) <= %s
          AND vc_item_desc IS NOT NULL
        GROUP BY SPLIT_PART(vc_item_desc, ' ', 1)
        HAVING COUNT(DISTINCT serial_no) >= 5
        ORDER BY total_serials DESC
    """

    try:
        df = pd.read_sql(query, conn, params=(start_str, end_str))
        return df
    except Exception as e:
        st.error(f"Error loading brand analysis: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_shop_serial_details(start_date, end_date, shop_code):
    """Get detailed serial journey for a specific shop"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    try:
        query = """
            SELECT 
                sd.serial_no as wh_serial,
                sd.vc_item_code,
                sd.vc_item_desc,
                sd.nu_selling_price,
                sd.dt_doc_date as doc_date,
                sd.loaded_datetime as when_loaded,
                sd.vc_vehicle_no as loading_vehicle,
                sd.vc_shop_code as destination_shop,
                sd.shop_name as destination_shop_name,
                sd.vc_serail_no as offload_serial,
                sd."vc_vehicle_no.1" as offload_vehicle,
                CASE 
                    WHEN sd.vc_serail_no IS NOT NULL AND sd.vc_serail_no != '' 
                    THEN sd.loaded_datetime 
                END as when_offloaded,
                sd.shop_serail_no as sold_serial,
                sd.shop_sold as selling_shop,
                sd.dt_invoice_date as when_sold,
                sd.vc_invoice_no as invoice_no,
                sd.remarks2 as issue_category,
                CASE WHEN sd.vc_shop_code != sd.shop_sold THEN 'YES' ELSE 'NO' END as shop_mismatch,
                CASE WHEN sd.vc_vehicle_no != sd."vc_vehicle_no.1" THEN 'YES' ELSE 'NO' END as vehicle_mismatch
            FROM serial_no_dailydata sd
            WHERE DATE(sd.dt_doc_date) >= %s AND DATE(sd.dt_doc_date) <= %s
              AND sd.vc_shop_code = %s
            ORDER BY sd.loaded_datetime DESC
        """

        result_df = pd.read_sql(query, conn, params=(start_str, end_str, shop_code))

        if not result_df.empty:
            unique_serials = result_df['wh_serial'].dropna().unique().tolist()
            if unique_serials:
                batch_size = 1000
                wh_dfs = []
                for i in range(0, len(unique_serials), batch_size):
                    batch = unique_serials[i:i + batch_size]
                    serial_list = "', '".join([str(s).replace("'", "''") for s in batch])
                    wh_query = f"""
                        SELECT serial_no, grn_date as wh_grn_date, supp_name as supplier
                        FROM whreceived_serialno
                        WHERE serial_no IN ('{serial_list}')
                    """
                    batch_df = pd.read_sql(wh_query, conn)
                    if not batch_df.empty:
                        wh_dfs.append(batch_df)

                if wh_dfs:
                    wh_df = pd.concat(wh_dfs, ignore_index=True)
                    result_df = result_df.merge(wh_df, left_on='wh_serial', right_on='serial_no', how='left')
                    result_df.drop('serial_no', axis=1, inplace=True, errors='ignore')

            serial_cols = _get_table_columns(conn, SERIAL_CHECK_TABLE)
            sc_serial_col = (
                serial_cols.get('serial_number') or
                serial_cols.get('serial_no') or
                serial_cols.get('serial')
            )
            sc_cashier_col = (
                serial_cols.get('cashier_name') or
                serial_cols.get('cashier') or
                serial_cols.get('user_name')
            )
            sc_till_col = (
                serial_cols.get('till_number') or
                serial_cols.get('till_no') or
                serial_cols.get('till')
            )
            sc_bill_col = (
                serial_cols.get('bill_no') or
                serial_cols.get('bill_n') or
                serial_cols.get('bill_number') or
                serial_cols.get('invoice_no')
            )
            sc_check_col = serial_cols.get('serial_check')

            if sc_serial_col:
                all_serials = []
                for col in ['wh_serial', 'offload_serial', 'sold_serial']:
                    if col in result_df.columns:
                        all_serials.extend(result_df[col].dropna().unique().tolist())
                all_serials = list(set(all_serials))

                if all_serials:
                    batch_size = 1000
                    personnel_dfs = []
                    for i in range(0, len(all_serials), batch_size):
                        batch = all_serials[i:i + batch_size]
                        serial_list = "', '".join([str(s).replace("'", "''") for s in batch])
                        personnel_query = f"""
                            SELECT 
                                {_qcol(sc_serial_col)} as serial_number,
                                {_col_or_null(sc_cashier_col)} as cashier_name,
                                {_col_or_null(sc_till_col)} as till_number,
                                {_col_or_null(sc_bill_col)} as bill_no,
                                {_col_or_null(sc_check_col)} as serial_check
                            FROM {SERIAL_CHECK_TABLE}
                            WHERE UPPER(TRIM({_qcol(sc_serial_col)})) IN ('{serial_list}')
                        """
                        batch_df = pd.read_sql(personnel_query, conn)
                        if not batch_df.empty:
                            personnel_dfs.append(batch_df)

                    personnel_df = pd.concat(personnel_dfs, ignore_index=True) if personnel_dfs else pd.DataFrame()

                    if not personnel_df.empty:
                        if 'sold_serial' in result_df.columns:
                            result_df = result_df.merge(
                                personnel_df,
                                left_on='sold_serial',
                                right_on='serial_number',
                                how='left'
                            )
                            result_df.drop('serial_number', axis=1, inplace=True, errors='ignore')
                        if 'serial_check' not in result_df.columns and 'offload_serial' in result_df.columns:
                            result_df = result_df.merge(
                                personnel_df,
                                left_on='offload_serial',
                                right_on='serial_number',
                                how='left'
                            )
                            result_df.drop('serial_number', axis=1, inplace=True, errors='ignore')
                        if 'serial_check' not in result_df.columns:
                            result_df = result_df.merge(
                                personnel_df,
                                left_on='wh_serial',
                                right_on='serial_number',
                                how='left'
                            )
                            result_df.drop('serial_number', axis=1, inplace=True, errors='ignore')

        return result_df
    except Exception as e:
        st.error(f"Error loading shop serial details: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=300)
def get_unverified_serials(start_date, end_date):
    """Get serial numbers not verified in main DB"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    try:
        serial_cols = _get_table_columns(conn, SERIAL_CHECK_TABLE)
        serial_col = (
            serial_cols.get('serial_number') or
            serial_cols.get('serial_no') or
            serial_cols.get('serial')
        )
        item_code_col = serial_cols.get('item_code') or serial_cols.get('vc_item_code')
        item_name_col = serial_cols.get('item_name') or serial_cols.get('vc_item_desc')
        shop_col = (
            serial_cols.get('shop_code') or
            serial_cols.get('shopcode') or
            serial_cols.get('shop') or
            serial_cols.get('store_code') or
            serial_cols.get('shop_sold')
        )
        bill_col = (
            serial_cols.get('bill_no') or
            serial_cols.get('bill_n') or
            serial_cols.get('bill_number') or
            serial_cols.get('invoice_no')
        )
        bill_date_col = (
            serial_cols.get('bill_date') or
            serial_cols.get('date_invoice') or
            serial_cols.get('dt_invoice_date') or
            serial_cols.get('invoice_date')
        )
        till_col = (
            serial_cols.get('till_number') or
            serial_cols.get('till_no') or
            serial_cols.get('till')
        )
        cashier_col = (
            serial_cols.get('cashier_name') or
            serial_cols.get('cashier') or
            serial_cols.get('user_name')
        )
        serial_check_col = serial_cols.get('serial_check')

        if not serial_check_col:
            return pd.DataFrame()

        serial_check_expr = _qcol(serial_check_col)
        bill_date_expr = _qcol(bill_date_col) if bill_date_col else None

        where_clause = ""
        params = []
        if bill_date_expr:
            where_clause = f"WHERE {bill_date_expr} >= %s AND {bill_date_expr} <= %s"
            params = [start_str, end_str]

        query = f"""
            SELECT 
                {_col_or_null(serial_col)} as serial_number,
                {_col_or_null(item_code_col)} as item_code,
                {_col_or_null(item_name_col)} as item_name,
                {_col_or_null(shop_col)} as shop_code,
                {_col_or_null(bill_col)} as bill_no,
                {_col_or_null(bill_date_col, 'date')} as bill_date,
                {_col_or_null(till_col)} as till_number,
                {_col_or_null(cashier_col)} as cashier_name,
                {serial_check_expr} as serial_check
            FROM {SERIAL_CHECK_TABLE}
            {where_clause}
              {"AND" if where_clause else "WHERE"} (UPPER(TRIM({serial_check_expr})) = 'N' OR {serial_check_expr} IS NULL OR TRIM({serial_check_expr}) = '')
            ORDER BY {_col_or_null(bill_date_col, 'date')} DESC
            LIMIT 1000
        """

        df = pd.read_sql(query, conn, params=params if params else None)
        return df
    except Exception as e:
        st.error(f"Error loading unverified serials: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# ====================== MAIN DASHBOARD ======================
def inject_r_reload_hotkey():
    components.html(
        """
        <script>
        const isTypingTarget = (el) => {
            if (!el) return false;
            const tag = (el.tagName || '').toLowerCase();
            return tag === 'input' || tag === 'textarea' || el.isContentEditable;
        };
        window.addEventListener('keydown', function(e) {
            if (isTypingTarget(document.activeElement)) return;
            if (e.key === 'r' || e.key === 'R') {
                e.preventDefault();
                window.location.reload();
            }
        }, { passive: false });
        </script>
        """,
        height=0,
        width=0,
    )

def main():
    inject_r_reload_hotkey()
    
    # FORCE LARGE TABLE FONTS - ADDITIONAL CSS INJECTION - RESPONSIVE TO ZOOM
    st.markdown("""
    <style>
    /* FORCE ALL TABLES TO HAVE LARGE FONTS - RESPONSIVE WITH REM UNITS */
    table, table *, 
    [data-testid="stDataFrame"], [data-testid="stDataFrame"] *,
    .dataframe, .dataframe *,
    div[class*="stDataFrame"], div[class*="stDataFrame"] * {
        font-size: 1.5rem !important;
    }
    
    /* Headers even larger */
    th, thead th, [role="columnheader"] {
        font-size: 1.75rem !important;
        font-weight: bold !important;
    }
    
    /* Data cells */
    td, tbody td, [role="gridcell"] {
        font-size: 1.5rem !important;
        padding: 0.875rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">
            <img src="https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg" alt="Melcom" style="height:36px; vertical-align:middle; margin-right:10px;">Melcom Serial Health Dashboard
        </div>
        <div class="dashboard-subtitle">Simple view of serial movement and issues</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">📅 Warehouse Date Filter</div>', unsafe_allow_html=True)
    yesterday = (datetime.today() - timedelta(days=1)).date()
    min_data_date = datetime(2024, 1, 1).date()

    # Default to yesterday for both start and end if not set
    default_wh_range = st.session_state.get(
        'warehouse_date_range',
        (yesterday, yesterday)
    )
    # Defensive: ensure default_wh_range is a tuple of two dates
    if not isinstance(default_wh_range, tuple):
        default_wh_range = (yesterday, yesterday)
    elif len(default_wh_range) == 1:
        # If user selected a single date, duplicate it for start/end
        default_wh_range = (default_wh_range[0], default_wh_range[0])
    elif len(default_wh_range) != 2:
        default_wh_range = (yesterday, yesterday)
    elif isinstance(default_wh_range[0], tuple):
        default_wh_range = default_wh_range[0]
    # Ensure default range is valid
    # Defensive: if still not two elements, fallback
    if not isinstance(default_wh_range, tuple) or len(default_wh_range) != 2:
        default_wh_range = (yesterday, yesterday)
    default_wh_start = max(default_wh_range[0], min_data_date)
    default_wh_end = min(default_wh_range[1], yesterday)

    # Add CSS styling for white background on date inputs
    st.markdown("""
        <style>
        div[data-testid="stDateInput"] > div > div {
            background-color: white !important;
            border: 2px solid #4CAF50 !important;
            border-radius: 5px !important;
            padding: 5px !important;
        }
        div[data-testid="stDateInput"] input {
            background-color: white !important;
            color: #000000 !important;
            font-weight: 500 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    date_col, _ = st.columns([0.2, 0.8])
    with date_col:
        warehouse_selection = st.date_input(
            "Select GRN Date Range",
            value=(default_wh_start, default_wh_end),
            min_value=min_data_date,
            max_value=yesterday,
            key="warehouse_date_filter",
            help="Pick a GRN date range to drive warehouse received metrics and the corresponding DT_DOC_DATE window for loaded metrics."
        )
    # Always store as a tuple (start, end)
    if isinstance(warehouse_selection, tuple) and len(warehouse_selection) == 2:
        warehouse_range = warehouse_selection
    else:
        warehouse_range = (warehouse_selection, warehouse_selection)
    st.session_state['warehouse_date_range'] = warehouse_range
    st.session_state['warehouse_grn_date'] = warehouse_range[0]

    if warehouse_range[0] == warehouse_range[1]:
        st.caption(
            f"Selected GRN date → {warehouse_range[0]:%d %b %Y} (applies to GRN + DT_DOC_DATE filters)"
        )
    else:
        st.caption(
            f"Active window → {warehouse_range[0]:%d %b %Y} to {warehouse_range[1]:%d %b %Y} (GRN + DT_DOC_DATE)"
        )

    received_range = warehouse_range
    loaded_range = warehouse_range

    tab1, tab2, tab3, tab_journey, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🏭 Warehouse Metrics",
        "🏪 Shop Metrics",
        "🧭 Serial Journey",
        "🚨 Discrepancies",
        "📈 Analytics"
    ])
    
    # ==================== TAB 1: OVERVIEW ====================
    with tab1:
        st.markdown('<div class="section-header">📊 Overall Summary</div>', unsafe_allow_html=True)
        refresh_cols = st.columns([1, 4])
        with refresh_cols[0]:
            if st.button(
                "🔄 Refresh Overview Data",
                key="refresh_overview_button",
                help="Clear cached KPI queries and reload the latest warehouse/shop metrics."
            ):
                clear_overview_caches()
                st.session_state['overview_last_refresh'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.success("Overview metrics refreshed from the database.")
        with refresh_cols[1]:
            last_refresh = st.session_state.get('overview_last_refresh')
            if last_refresh:
                st.caption(f"Manual refresh: {last_refresh}")
            else:
                st.caption("Metrics auto-refresh every 5 minutes; use Refresh to force an update.")

        # Load all metrics
        wh_metrics = get_warehouse_metrics(received_range, loaded_range)
        # Apply warehouse date selection to Shop Offloaded metrics using VC_MOD_date
        shop_offload_start, shop_offload_end = warehouse_range
        shop_metrics = get_shop_offload_metrics(shop_offload_start, shop_offload_end)
        # Use warehouse_range for compliance KPI (single date selection drives all metrics)
        sales_metrics = None
        compliance_start, compliance_end = warehouse_range
        def get_overview_compliance_pct(start_date, end_date):
            conn = get_db_connection()
            if not conn:
                return 0.0
            start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
            end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
            query = f"""
                  SELECT COUNT(*) as total_serials,
                      COUNT(*) FILTER (WHERE TRIM(serial_check) = 'Y') as total_yes
                FROM serialno_check_yes_no
                WHERE DATE(bill_date) >= '{start_str}' AND DATE(bill_date) <= '{end_str}'
            """
            try:
                df = pd.read_sql(query, conn)
                conn.close()
                if not df.empty:
                    total = df.iloc[0]['total_serials']
                    total_yes = df.iloc[0]['total_yes']
                    return round((total_yes / total) * 100, 2) if total > 0 else 0.0
                else:
                    return 0.0
            except Exception as e:
                st.error(f"Error loading overview compliance: {e}")
                conn.close()
                return 0.0

        compliance_pct = get_overview_compliance_pct(compliance_start, compliance_end)
        not_offloaded = get_not_offloaded_metrics()
        discrepancy_summary = get_discrepancy_summary()

        # Always show metrics, use compliance_pct directly if sales_metrics is None
        if wh_metrics and shop_metrics:
            # Top KPIs
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card success" title="Total records in whreceived_serialno with non-blank serial_no">
                    <div class="metric-label">WH Received</div>
                    <div class="metric-value">{format_number(wh_metrics['received_total'])}</div>
                    <div class="metric-change positive">✓ {format_number(wh_metrics['received_unique'])} Unique</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" title="Total records in serial_no_dailydata with non-blank serial_no">
                    <div class="metric-label">WH Loaded</div>
                    <div class="metric-value">{format_number(wh_metrics['loaded_total'])}</div>
                    <div class="metric-change">{format_percentage(wh_metrics['unique_percentage'])} Unique</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card" title="Total rows with vc_serail_no present in serial_no_dailydata">
                    <div class="metric-label">Shop Offloaded</div>
                    <div class="metric-value">{format_number(shop_metrics['total_offloaded'])}</div>
                    <div class="metric-change">✓ {format_number(shop_metrics['unique_offloaded'])} Unique</div>
                </div>
                """, unsafe_allow_html=True)

            # Uniqueness % Trend Graphs
            trend_col1, trend_col2 = st.columns(2)
            trend_col1, trend_col2, trend_col3 = st.columns(3)
            with trend_col1:
                st.markdown("#### WH Received Uniqueness % (Last 7 Days)")
                wh_received_trend = get_wh_uniqueness_trend('received')
                if wh_received_trend is not None and not wh_received_trend.empty:
                    fig = px.line(
                        wh_received_trend,
                        x='date',
                        y='uniqueness_pct',
                        title='WH Received Uniqueness %',
                        markers=True
                    )
                    fig.update_layout(height=250, yaxis_title="Uniqueness %", xaxis_title="Date")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No trend data available.")
            with trend_col2:
                st.markdown("#### WH Loaded Uniqueness % (Last 7 Days)")
                wh_loaded_trend = get_wh_uniqueness_trend('loaded')
                if wh_loaded_trend is not None and not wh_loaded_trend.empty:
                    fig = px.line(
                        wh_loaded_trend,
                        x='date',
                        y='uniqueness_pct',
                        title='WH Loaded Uniqueness %',
                        markers=True
                    )
                    yesterday = datetime.today().date() - timedelta(days=1)
                    last7dates = pd.date_range(yesterday - timedelta(days=6), yesterday).strftime('%Y-%m-%d').tolist()
                    fig.update_layout(
                        height=250,
                        yaxis_title="Uniqueness %",
                        xaxis_title="Date",
                        xaxis=dict(
                            tickformat="%Y-%m-%d",
                            tickmode="array",
                            tickvals=last7dates
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No trend data available.")
            with trend_col3:
                            st.markdown("#### Compliance % (Last 7 Days)")
                            # Query last 7 days compliance % from serialno_check_yes_no
                            today = datetime.today().date()
                            yesterday = today - timedelta(days=1)
                            start_date = yesterday - timedelta(days=6)
                            end_date = yesterday
                            conn = get_db_connection()
                            if conn:
                                # 'serial_check' should be exactly what it is as per your request
                                query = f"""
                                    SELECT DATE(bill_date) as date,
                                        COUNT(*) FILTER (WHERE TRIM(serial_check) = 'Y') as total_yes,
                                        COUNT(*) as total_serials,
                                        CASE WHEN COUNT(*) > 0 THEN ROUND(COUNT(*) FILTER (WHERE TRIM(serial_check) = 'Y')::NUMERIC / COUNT(*) * 100, 2) ELSE 0 END as compliance_pct
                                    FROM serialno_check_yes_no
                                    WHERE DATE(bill_date) >= '{start_date}' AND DATE(bill_date) <= '{end_date}'
                                    GROUP BY DATE(bill_date)
                                    ORDER BY DATE(bill_date)
                                """
                                try:
                                    df = pd.read_sql(query, conn)
                                    conn.close()
                                    if df is not None and not df.empty:
                                        # Format date as 'dd mmm' (e.g., '10 Feb')
                                        df['date'] = pd.to_datetime(df['date']).dt.strftime('%d %b')
                                        
                                        # CHANGED: px.bar -> px.line for a line graph
                                        fig = px.line(
                                            df,
                                            x='date',
                                            y='compliance_pct',
                                            title='Compliance % (Last 7 Days)',
                                            markers=True, # Added markers for better visibility
                                            labels={'compliance_pct': 'Compliance %'}
                                        )
                                        
                                        # Update traces for line graph
                                        fig.update_traces(
                                            line=dict(color="#002c6d", width=3),
                                            marker=dict(size=15),
                                            text=df['compliance_pct'].apply(lambda x: f'{x:.1f}%'),
                                            textposition="top center"
                                        )
                                        
                                        fig.update_layout(
                                            height=250, 
                                            yaxis_title="Compliance %", 
                                            xaxis_title="Date", 
                                            margin=dict(l=20, r=20, t=40, b=20),
                                            # --- ADD THIS SECTION TO CHANGE FONT SIZES ---
                                            xaxis=dict(
                                                title_font=dict(size=16),  # Font size for 'Date' label
                                                tickfont=dict(size=16)     # Font size for the dates (10 Feb, etc)
                                            ),
                                            yaxis=dict(
                                                title_font=dict(size=16),  # Font size for 'Compliance %' label
                                                tickfont=dict(size=16)     # Font size for the percentages (0, 20, 40...)
                                            )
                                            # ---------------------------------------------
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                    else:
                                        st.info("No compliance data available.")
                                except Exception as e:
                                    st.error(f"Error loading compliance trend: {e}")
                                    if conn: conn.close()
                            else:
                                st.info("No compliance data available.")            
            with col4:
                if sales_metrics:
                    st.markdown(f"""
                    <div class="metric-card success" title="Total rows with shop_serail_no present in serial_no_dailydata">
                        <div class="metric-label">Sold</div>
                        <div class="metric-value">{format_number(sales_metrics['total_sold'])}</div>
                        <div class="metric-change positive">✓ {format_number(sales_metrics['unique_sold'])} Unique</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card" title="Sold metrics unavailable">
                        <div class="metric-label">Sold</div>
                        <div class="metric-value">N/A</div>
                        <div class="metric-change">No data</div>
                    </div>
                    """, unsafe_allow_html=True)

            with col5:
                # Use compliance_pct from get_overview_compliance_pct if sales_metrics is None
                comp_val = sales_metrics.get('compliance_pct', 0) if sales_metrics else compliance_pct
                card_class = "success" if comp_val >= 90 else "danger"
                st.markdown(f"""
                <div class="metric-card {card_class}" title="Percentage of sold serial numbers that are unique (compliance)">
                    <div class="metric-label">Shop Sold Serial Compliance %</div>
                    <div class="metric-value">{format_percentage(comp_val)}</div>
                    <div class="metric-change">Compliance: Unique serial Sold matched in Serial DB / Total Sold</div>
                </div>
                """, unsafe_allow_html=True)
            
            # ==================== YESTERDAY'S SUMMARY SECTION ====================
            st.markdown('<div class="section-header">📅 Warehouse Receiving Summary</div>', unsafe_allow_html=True)

            summary_date = warehouse_range[0]
            st.caption(f"Summary reflects GRN activity on {summary_date:%Y-%m-%d} (change using the calendar above).")

            summary_overall, summary_by_wh = get_wh_metrics_by_date(summary_date)

            active_duplicate_date = st.session_state.get('receiving_duplicates_for')
            if active_duplicate_date and active_duplicate_date != summary_date:
                st.session_state['show_receiving_duplicates'] = False

            if summary_overall and summary_overall['total_received'] > 0:
                summary_date_str = summary_date.strftime('%Y-%m-%d')
                col_total, col_unique, col_dupe, col_status = st.columns(4)

                is_100_unique = summary_overall['unique_percentage'] == 100.0

                with col_total:
                    st.markdown(f"""
                    <div class="metric-card" title="Total serials received on {summary_date_str}">
                        <div class="metric-label">Total Received</div>
                        <div class="metric-value">{format_number(summary_overall['total_received'])}</div>
                        <div class="metric-change">{summary_date_str}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_unique:
                    st.markdown(f"""
                    <div class="metric-card success" title="Unique serials received on {summary_date_str}">
                        <div class="metric-label">Unique Serials</div>
                        <div class="metric-value">{format_number(summary_overall['unique_serials'])}</div>
                        <div class="metric-change positive">✓ {format_percentage(summary_overall['unique_percentage'])}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_dupe:
                    card_class = "metric-card" if is_100_unique else "metric-card danger"
                    duplicate_icon = "✓" if is_100_unique else "⚠"
                    change_class = "positive" if is_100_unique else "negative"

                    if summary_overall['duplicate_count'] > 0:
                        if st.button(
                            f"🔍 {format_number(summary_overall['duplicate_count'])} Duplicates",
                            key=f"dup_btn_{summary_date_str}",
                            use_container_width=True
                        ):
                            st.session_state['show_receiving_duplicates'] = True
                            st.session_state['receiving_duplicates_for'] = summary_date
                    else:
                        st.markdown(f"""
                        <div class="{card_class}" title="Duplicate serials received on {summary_date_str}">
                            <div class="metric-label">Duplicates</div>
                            <div class="metric-value">{format_number(summary_overall['duplicate_count'])}</div>
                            <div class="metric-change {change_class}">{duplicate_icon} Perfect!</div>
                        </div>
                        """, unsafe_allow_html=True)

                with col_status:
                    uniqueness_status = "🟢 Perfect" if is_100_unique else "🔴 Action Needed"
                    status_card = "metric-card success" if is_100_unique else "metric-card danger"

                    st.markdown(f"""
                    <div class="{status_card}" title="Uniqueness status for {summary_date_str}">
                        <div class="metric-label">Status</div>
                        <div class="metric-value" style="font-size: 1.5rem;">{uniqueness_status}</div>
                        <div class="metric-change">{format_percentage(summary_overall['unique_percentage'])}% Unique</div>
                    </div>
                    """, unsafe_allow_html=True)

                trend_rendered_inline = False
                if st.session_state.get('show_receiving_duplicates', False) and st.session_state.get('receiving_duplicates_for') == summary_date:
                    st.markdown("### 🔍 Duplicate Serial Details")
                    range_end_default = summary_date
                    min_available_date = datetime(2024, 1, 1).date()
                    range_start_default = max(summary_date.replace(day=1), min_available_date)
                    range_key = f"duplicate_range_{summary_date_str}"
                    date_range = st.date_input(
                        "Select Date Range",
                        value=(range_start_default, range_end_default),
                        min_value=min_available_date,
                        max_value=range_end_default,
                        key=range_key,
                        help="Pick a range (e.g., 1st to your chosen date) and the table will update."
                    )

                    if isinstance(date_range, tuple) and len(date_range) == 2:
                        range_start, range_end = date_range
                    else:
                        range_start = range_end = date_range

                    st.caption(f"Showing duplicates from {range_start:%Y-%m-%d} through {range_end:%Y-%m-%d}")
                    dup_details = get_duplicates_by_date_range(range_start, range_end)

                    if dup_details is not None and not dup_details.empty:
                        table_col, trend_col, compliance_col = st.columns([2.5, 1.8, 1.8], gap="medium")
                        with table_col:
                            st.markdown(
                                render_yesterday_duplicate_table(dup_details),
                                unsafe_allow_html=True
                            )
                            csv = dup_details.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Duplicates (CSV)",
                                data=csv,
                                file_name=f"duplicates_{range_start:%Y%m%d}_{range_end:%Y%m%d}.csv",
                                mime="text/csv"
                            )
                            if st.button("✖ Close Duplicate Details", key=f"close_dup_{summary_date_str}"):
                                st.session_state['show_receiving_duplicates'] = False
                                st.session_state['receiving_duplicates_for'] = None
                                st.session_state.pop(range_key, None)
                                st.rerun()
                        with trend_col:
                            render_yesterday_trend(summary_by_wh, container=trend_col, key_suffix="_1")
                        with compliance_col:
                            render_compliance_trend_overview()
                        if not trend_rendered_inline:
                            col_trend, col_compliance = st.columns([1.8, 1.8], gap="medium")
                            with col_trend:
                                render_yesterday_trend(summary_by_wh, container=col_trend, key_suffix="_2", include_header=False)
                            with col_compliance:
                                render_compliance_trend_overview()

        # ==================== OVERVIEW COMPLIANCE TREND FUNCTION ====================
                    with col_trend:
                        render_yesterday_trend(summary_by_wh, container=col_trend, key_suffix="_3", include_header=False)
                    with col_compliance:
                        render_compliance_trend_overview()
            else:
                st.info(f"No data received on {summary_date.strftime('%Y-%m-%d')}")
            
            # ==================== END YESTERDAY'S SUMMARY ====================
            
            # Key Alerts
            col1, col2 = st.columns(2)
            
            with col1:
                if wh_metrics['duplicate_percentage'] > 5:
                    st.markdown(f"""
                    <div class="alert-box critical">
                        <strong>🚨 High Duplicate Rate at Warehouse</strong><br>
                        {format_percentage(wh_metrics['duplicate_percentage'])} of loaded serials are duplicates ({format_number(wh_metrics['loaded_total'] - wh_metrics['loaded_unique'])} records)
                    </div>
                    """, unsafe_allow_html=True)
                
                if shop_metrics['offload_before_wh_load'] > 0:
                    st.markdown(f"""
                    <div class="alert-box warning">
                        <strong>⚠ Time Anomaly Detected</strong><br>
                        {format_number(shop_metrics['offload_before_wh_load'])} items offloaded before WH loading date
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if sales_metrics and sales_metrics.get('duplicate_sales_count', 0) > 0:
                    st.markdown(f"""
                    <div class="alert-box critical">
                        <strong>🚨 Potential Fraud: Duplicate Sales</strong><br>
                        {format_number(sales_metrics.get('unique_duplicate_serials', 0))} serial numbers sold multiple times ({format_number(sales_metrics.get('duplicate_sales_count', 0))} total sales)
                    </div>
                    """, unsafe_allow_html=True)
                
                if not_offloaded and not_offloaded.get('total_not_offloaded', 0) > 100:
                    st.markdown(f"""
                    <div class="alert-box warning">
                        <strong>⚠ High Not-Offloaded Count</strong><br>
                        {format_number(not_offloaded.get('total_not_offloaded', 0))} items loaded but not offloaded yet
                    </div>
                    """, unsafe_allow_html=True)
    
    # ==================== TAB 2: WAREHOUSE METRICS ====================
    with tab2:
        st.markdown('<div class="section-header">🏭 Warehouse Performance</div>', unsafe_allow_html=True)
        
        wh_metrics = get_warehouse_metrics(received_range, loaded_range)
        loader_breakdown = get_loader_breakdown(*warehouse_range)
        dup_items = get_duplicate_serials_by_item()
        dup_serials_with_remarks = get_duplicate_serials_with_remarks()
        
        if wh_metrics:

            # Warehouse KPIs
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" title="Total records in whreceived_serialno with non-blank serial_no">
                    <div class="metric-label">Received Total</div>
                    <div class="metric-value">{format_number(wh_metrics['received_total'])}</div>
                    <div class="metric-change">Warehouse receipts captured</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card success" title="Distinct serial_no in whreceived_serialno">
                    <div class="metric-label">Received Unique</div>
                    <div class="metric-value">{format_number(wh_metrics['received_unique'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card" title="Total records in serial_no_dailydata with non-blank serial_no">
                    <div class="metric-label">Loaded Total</div>
                    <div class="metric-value">{format_number(wh_metrics['loaded_total'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card success" title="Distinct serial_no in serial_no_dailydata">
                    <div class="metric-label">Loaded Unique</div>
                    <div class="metric-value">{format_number(wh_metrics['loaded_unique'])}</div>
                    <div class="metric-change">{format_percentage(wh_metrics['unique_percentage'])} of Total</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Duplicate Analysis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card warning" title="Count of rows whose serial_no appears more than once in serial_no_dailydata">
                    <div class="metric-label">Duplicate Serials</div>
                    <div class="metric-value">{format_number(wh_metrics['duplicate_serials_count'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card warning" title="Distinct vc_item_code values that have duplicate serial_no">
                    <div class="metric-label">Items with Dup Serials</div>
                    <div class="metric-value">{format_number(wh_metrics['unique_items_with_dup_serials'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Calculate unique percentage
                if wh_metrics['loaded_total'] > 0:
                    unique_pct = (wh_metrics['loaded_unique'] / wh_metrics['loaded_total']) * 100
                else:
                    unique_pct = 0
                
                card_class = "success" if unique_pct > 95 else "warning" if unique_pct > 90 else "danger"
                st.markdown(f"""
                <div class="metric-card {card_class}" title="(Loaded Unique ÷ Loaded Total) × 100">
                    <div class="metric-label">Uniqueness Rate</div>
                    <div class="metric-value">{format_percentage(unique_pct)}</div>
                    <div class="metric-change">Target: >95%</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Loader Breakdown
        if loader_breakdown is not None and not loader_breakdown.empty:
            st.markdown('<div class="section-header">👷 Loader Performance Breakdown</div>', unsafe_allow_html=True)

            loader_breakdown = loader_breakdown.copy()
            loader_breakdown['duplicacy_pct_total'] = (
                loader_breakdown['duplicate_count'] / loader_breakdown['total_loads'] * 100
            ).replace([pd.NA, pd.NaT, float('inf'), -float('inf')], 0).fillna(0).round(2)

            loader_breakdown['duplicacy_pct_unique'] = (
                loader_breakdown['duplicate_count'] / loader_breakdown['unique_serials'] * 100
            ).replace([pd.NA, pd.NaT, float('inf'), -float('inf')], 0).fillna(0).round(2)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Loaders", len(loader_breakdown))
            
            with col2:
                st.metric("Total Loads", format_number(loader_breakdown['total_loads'].sum()))
            
            with col3:
                st.metric("Avg Loads per Loader", format_number(loader_breakdown['total_loads'].mean()))
            
            with col4:
                total_dups = loader_breakdown['duplicate_count'].sum()
                st.metric("Total Duplicates", format_number(total_dups))
            
            # Loader table
            st.dataframe(
                loader_breakdown,
                use_container_width=True,
                column_config={
                    "loader": st.column_config.TextColumn("Loader", width="medium"),
                    "total_loads": st.column_config.NumberColumn("Total Loads", format="%d"),
                    "unique_serials": st.column_config.NumberColumn("Unique Serials", format="%d"),
                    "duplicate_count": st.column_config.NumberColumn("Duplicates", format="%d"),
                    "duplicacy_pct_total": st.column_config.NumberColumn("Duplicacy % of Total", format="%.2f%%"),
                    "duplicacy_pct_unique": st.column_config.NumberColumn("Duplicacy % of Unique", format="%.2f%%"),
                    "unique_items": st.column_config.NumberColumn("Unique Items", format="%d"),
                    "items_with_dup_serials": st.column_config.NumberColumn("Items w/ Dup Serials", format="%d")
                },
                hide_index=True
            )
            
            # Chart: Loader comparison
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Total Loads',
                x=loader_breakdown['loader'],
                y=loader_breakdown['total_loads'],
                marker_color=MELCOM_BLUE
            ))
            
            fig.add_trace(go.Bar(
                name='Duplicates',
                x=loader_breakdown['loader'],
                y=loader_breakdown['duplicate_count'],
                marker_color=MELCOM_RED
            ))
            
            fig.update_layout(
                title="Loader Performance: Total Loads vs Duplicates",
                barmode='group',
                height=400,
                xaxis_title="Loader",
                yaxis_title="Count"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Items with Duplicate Serials
        if dup_items is not None and not dup_items.empty:
            st.markdown('<div class="section-header">📦 Items with Duplicate Serial Numbers</div>', unsafe_allow_html=True)
            
            st.dataframe(
                dup_items,
                use_container_width=True,
                column_config={
                    "item_code": st.column_config.TextColumn("Item Code", width="small"),
                    "item_name": st.column_config.TextColumn("Item Name", width="large"),
                    "unique_serials": st.column_config.NumberColumn("Unique Serials", format="%d"),
                    "total_records": st.column_config.NumberColumn("Total Records", format="%d"),
                    "duplicate_count": st.column_config.NumberColumn("Duplicate Count", format="%d"),
                    "loaders": st.column_config.TextColumn("Loaders", width="medium"),
                    "first_load": st.column_config.DatetimeColumn("First Load"),
                    "last_load": st.column_config.DatetimeColumn("Last Load")
                },
                hide_index=True
            )

        # Duplicate Serials with Consolidated Remarks
        if dup_serials_with_remarks is not None and not dup_serials_with_remarks.empty:
            st.markdown('<div class="section-header">🔁 Duplicate Serials (Consolidated Remarks)</div>', unsafe_allow_html=True)
            st.caption("Remarks consolidate all detected mismatches for each duplicated WH serial number(tbl-serial_no_dailydata).")

            col_a, col_b = st.columns([0.1, 0.9])
            with col_a:
                remark_limit = st.selectbox("Row Limit", [200, 500, 1000, 2000], index=1, key="dup_remark_limit")
            with col_b:
                st.markdown("")

            if remark_limit != len(dup_serials_with_remarks):
                dup_serials_with_remarks = get_duplicate_serials_with_remarks(remark_limit)

            st.dataframe(
                dup_serials_with_remarks,
                use_container_width=True,
                column_config={
                    "serial_number": st.column_config.TextColumn("Serial Number", width="medium"),
                    "duplicate_count": st.column_config.NumberColumn("Duplicate Count", format="%d"),
                    "wh_codes": st.column_config.TextColumn("WH Codes", width="small"),
                    "item_codes": st.column_config.TextColumn("Item Codes", width="medium"),
                    "offload_shops": st.column_config.TextColumn("Offload Shops", width="medium"),
                    "sold_shops": st.column_config.TextColumn("Sold Shops", width="medium"),
                    "remark": st.column_config.TextColumn("Consolidated Remark", width="large")
                },
                hide_index=True
            )

            # Graph: Duplicates by Remark
            remark_counts = dup_serials_with_remarks['remark'].value_counts().reset_index()
            remark_counts.columns = ['remark', 'serial_count']

            fig = px.bar(
                remark_counts,
                x='remark',
                y='serial_count',
                title='Duplicate Serials by Consolidated Remark',
                color='serial_count',
                color_continuous_scale=[MELCOM_GREEN, MELCOM_ORANGE, MELCOM_RED]
            )
            fig.update_layout(height=400, xaxis_title="Remark", yaxis_title="# of Duplicate Serials")
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== SHOP COMPLIANCE SECTION - 7 DAY ROLLING ====================
    st.markdown('<div class="section-header">📅 Shop Compliance - Selected Date Range</div>', unsafe_allow_html=True)
    
    # Use warehouse_range for compliance metrics (same as Shop Metrics tab)
    compliance_start_range = warehouse_range[0]
    compliance_end_range = warehouse_range[1]
    
    # Calculate 7-day range from end date
    compliance_start_7d = compliance_end_range - timedelta(days=6)  # 7 days including end date
    compliance_end_7d = compliance_end_range
    
    # Initialize session state for shop compliance drill-down
    if "overview_compliance_selected_shop" not in st.session_state:
        st.session_state["overview_compliance_selected_shop"] = None
    
    # Initialize session state for cashier date drill-down
    if "overview_compliance_selected_cashier" not in st.session_state:
        st.session_state["overview_compliance_selected_cashier"] = None
    if "overview_compliance_selected_date" not in st.session_state:
        st.session_state["overview_compliance_selected_date"] = None
    
    selected_shop_overview_compliance = st.session_state.get("overview_compliance_selected_shop")
    
    # Get 7-day shop compliance data (calculated from end date)
    overview_7day_metrics = get_serial_check_metrics(compliance_start_7d, compliance_end_7d)
    
    if not overview_7day_metrics:
        st.warning("Unable to load Shop Compliance data.")
    elif overview_7day_metrics.get("missing_serial_check"):
        st.warning(f"Serial check column not found in {SERIAL_CHECK_TABLE}.")
    else:
        # Get end-date metrics (single day)
        enddate_metrics = get_serial_check_metrics(compliance_end_range, compliance_end_range)
        
        # Show overview metrics and table with end-date on left and 7-day on right
        overview_compliance_cols = st.columns([1, 1, 2.5], gap="large")
        
        # End-date metrics (left column)
        with overview_compliance_cols[0]:
            if enddate_metrics and not enddate_metrics.get("missing_serial_check"):
                overall_enddate = enddate_metrics.get("overall")
                if overall_enddate:
                    st.metric("Total Serials", format_number(overall_enddate.get('total_serials', 0)))
                    st.metric("Total Yes", format_number(overall_enddate.get('total_yes', 0)))
                    st.metric("Compliance %", f"{overall_enddate.get('yes_pct', 0)}%")
                    st.caption(f"Selected Date: {compliance_end_range:%Y/%m/%d}")
        
        # 7-day metrics (middle column)
        with overview_compliance_cols[1]:
            overall_7day = overview_7day_metrics.get("overall")
            if overall_7day:
                st.metric("Total Serials (7d)", format_number(overall_7day.get('total_serials', 0)))
                st.metric("Total Yes (7d)", format_number(overall_7day.get('total_yes', 0)))
                st.metric("Compliance % (7d)", f"{overall_7day.get('yes_pct', 0)}%")
                st.caption(f"Last 7 days: {compliance_start_7d:%Y/%m/%d} → {compliance_end_7d:%Y/%m/%d}")
        
        with overview_compliance_cols[2]:
            by_shop_7day = overview_7day_metrics.get("by_shop")
            
            if by_shop_7day is not None and not by_shop_7day.empty:
                st.markdown(
                    f"""
                    <div style="padding:0.6rem 0.8rem; background-color:rgba(34,197,94,0.08); border-left:4px solid #3ecc71; border-radius:8px; font-weight:600; margin-bottom:0.5rem;">
                        Click on any shop to view last 7 days drill-down with cashier details<br>
                        <span style="font-size:0.85rem; font-weight:500;">Light Red = lowest Compliance %, Light Green = highest</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                if selected_shop_overview_compliance:
                    st.markdown(f"#### 7-Day Drill-Down: {selected_shop_overview_compliance}")
                    
                    # Single unified back button that adapts to context
                    selected_cashier = st.session_state.get("overview_compliance_selected_cashier")
                    selected_date_label = st.session_state.get("overview_compliance_selected_date")
                    
                    if selected_cashier and selected_date_label:
                        # In cashier detail view - back goes to 7-day trend
                        back_label = "⬅ Back to 7-Day View"
                        if st.button(back_label, key="overview_compliance_back"):
                            st.session_state["overview_compliance_selected_cashier"] = None
                            st.session_state["overview_compliance_selected_date"] = None
                            st.rerun()
                    else:
                        # In 7-day trend view - back goes to shop summary
                        back_label = "⬅ Back to Shop Summary"
                        if st.button(back_label, key="overview_compliance_back"):
                            st.session_state["overview_compliance_selected_shop"] = None
                            st.session_state["overview_compliance_selected_cashier"] = None
                            st.session_state["overview_compliance_selected_date"] = None
                            st.rerun()
                    
                    # Get 7-day trend for selected shop
                    trend_result = get_shop_cashier_compliance_trend(selected_shop_overview_compliance, days=7)
                    
                    if not trend_result or trend_result.get("data") is None:
                        st.info("No compliance trend data available.")
                    elif trend_result.get("missing_date"):
                        st.info(f"Date column not found in {SERIAL_CHECK_TABLE}.")
                    else:
                        trend_df = trend_result.get("data")
                        end_date = trend_result.get("end_date")
                        start_date = trend_result.get("start_date")
                        
                        date_list = [start_date + timedelta(days=i) for i in range(7)]
                        date_labels = [f"{d.day}{d.strftime('%b').lower()}" for d in date_list]
                        
                        if trend_df is not None and not trend_df.empty:
                            trend_df['activity_date'] = pd.to_datetime(trend_df['activity_date']).dt.date
                            trend_df['cashier'] = trend_df['cashier'].fillna('Unknown')
                            trend_df['total_serials'] = trend_df['total_serials'].fillna(0)
                            trend_df['total_yes'] = trend_df['total_yes'].fillna(0)
                            trend_df['compliance_pct'] = (
                                trend_df['total_yes'].astype(float) /
                                trend_df['total_serials'].replace({0: pd.NA}).astype(float)
                            ) * 100
                            trend_df['compliance_pct'] = trend_df['compliance_pct'].fillna(0).round(2)
                            trend_df['has_data'] = trend_df['total_serials'] > 0
                            
                            shop_total = (
                                trend_df
                                .groupby('activity_date', as_index=False)
                                .agg({'total_serials': 'sum', 'total_yes': 'sum', 'has_data': 'first'})
                            )
                            shop_total['compliance_pct'] = (
                                shop_total['total_yes'].astype(float) /
                                shop_total['total_serials'].replace({0: pd.NA}).astype(float)
                            ) * 100
                            shop_total['compliance_pct'] = shop_total['compliance_pct'].fillna(0).round(2)
                            shop_total['cashier'] = selected_shop_overview_compliance
                            shop_total['has_data'] = True
                            
                            cashier_pivot = trend_df.pivot_table(
                                index='cashier',
                                columns='activity_date',
                                values='compliance_pct',
                                aggfunc='mean'
                            )
                            
                            cashier_has_data = trend_df.pivot_table(
                                index='cashier',
                                columns='activity_date',
                                values='has_data',
                                aggfunc='any'
                            )
                            
                            shop_pivot = shop_total.pivot_table(
                                index='cashier',
                                columns='activity_date',
                                values='compliance_pct',
                                aggfunc='mean'
                            )
                            
                            combined = pd.concat([shop_pivot, cashier_pivot])
                            combined_has_data = pd.concat([
                                pd.DataFrame(True, index=[selected_shop_overview_compliance], columns=cashier_has_data.columns),
                                cashier_has_data
                            ])
                        else:
                            combined = pd.DataFrame(index=pd.Index([selected_shop_overview_compliance], name='cashier'))
                            combined_has_data = pd.DataFrame(False, index=pd.Index([selected_shop_overview_compliance], name='cashier'), columns=date_list)
                        
                        combined = combined.reindex(columns=date_list)
                        combined.columns = date_labels
                        
                        combined_has_data = combined_has_data.reindex(columns=date_list)
                        combined_has_data = combined_has_data.fillna(False)
                        
                        trend_display = combined.reset_index().rename(columns={'cashier': 'Shop/Cashier'})
                        data_availability = combined_has_data.reset_index().rename(columns={'cashier': 'Shop/Cashier'})
                        
                        # Trim spaces from Shop/Cashier column to avoid matching issues
                        if 'Shop/Cashier' in trend_display.columns:
                            trend_display['Shop/Cashier'] = trend_display['Shop/Cashier'].astype(str).str.strip()
                        if 'Shop/Cashier' in data_availability.columns:
                            data_availability['Shop/Cashier'] = data_availability['Shop/Cashier'].astype(str).str.strip()
                        
                        # Fill NaN smartly
                        if 'Shop/Cashier' in trend_display.columns and len(trend_display) > 0:
                            for idx in range(len(trend_display)):
                                try:
                                    shop_or_cashier = trend_display.at[idx, 'Shop/Cashier']
                                    
                                    if shop_or_cashier == selected_shop_overview_compliance:
                                        for col in date_labels:
                                            if col in trend_display.columns and pd.isna(trend_display.at[idx, col]):
                                                trend_display.at[idx, col] = 0.0
                                    else:
                                        for col in date_labels:
                                            if col in trend_display.columns:
                                                has_data = False
                                                if col in data_availability.columns:
                                                    try:
                                                        has_data = data_availability.at[idx, col]
                                                    except:
                                                        has_data = False
                                                
                                                if has_data and pd.isna(trend_display.at[idx, col]):
                                                    trend_display.at[idx, col] = 0.0
                                except (KeyError, TypeError, IndexError):
                                    continue
                        
                        def _row_gradient_7day(row):
                            styles = [''] * len(row)
                            
                            shop_cashier = None
                            if 'Shop/Cashier' in row.index:
                                try:
                                    shop_cashier = row.loc['Shop/Cashier']
                                except:
                                    pass
                            
                            is_shop_row = (shop_cashier == selected_shop_overview_compliance)
                            
                            # Use fixed 0-100 scale for conditional formatting
                            min_val = 0.0
                            max_val = 100.0
                            span = 100.0
                            
                            for idx, col in enumerate(row.index):
                                if col not in date_labels:
                                    if is_shop_row:
                                        styles[idx] = 'font-weight:700; background-color:#dcfce7; font-size: 165%; padding: 6px;'
                                    continue
                                
                                val = row.loc[col]
                                
                                if pd.isna(val):
                                    styles[idx] = 'background-color: #ffffff; color: #999999; font-style: italic; font-size: 165%; padding: 6px;'
                                else:
                                    # Calculate ratio on 0-100 scale (100% is always green)
                                    ratio = float(val) / 100.0
                                    ratio = max(0, min(1, ratio))  # Clamp between 0 and 1
                                    
                                    # Light red → light yellow → light green gradient
                                    # 0% = Light Red, 50% = Light Yellow, 100% = Light Green
                                    if ratio < 0.5:
                                        # Light red to light yellow (0-50%)
                                        local_ratio = ratio * 2
                                        r = int(254)  # Stay red
                                        g = int(226 + (243 - 226) * local_ratio)  # 226 → 243
                                        b = int(226 - (226 - 199) * local_ratio)  # 226 → 199
                                    else:
                                        # Light yellow to light green (50-100%)
                                        local_ratio = (ratio - 0.5) * 2
                                        r = int(254 - (254 - 134) * local_ratio)  # 254 → 134
                                        g = int(243 - (243 - 239) * local_ratio)  # 243 → 239
                                        b = int(199 - (199 - 172) * local_ratio)  # 199 → 172
                                    
                                    styles[idx] = f'background-color: rgb({r},{g},{b}); color: #000000; font-weight:600; font-size: 165%; padding: 6px;'
                                    if is_shop_row:
                                        styles[idx] += ' border-left: 4px solid #3ecc71;'
                            
                            return styles
                        
                        styled = (
                            trend_display
                            .style
                            .format({col: '{:.1f}%' for col in date_labels}, na_rep='-')
                            .apply(_row_gradient_7day, axis=1)
                        )
                        
                        # Check if user has drilled down to cashier+date detail
                        selected_cashier = st.session_state.get("overview_compliance_selected_cashier")
                        selected_date_label = st.session_state.get("overview_compliance_selected_date")
                        
                        if selected_cashier and selected_date_label:
                            # User clicked on a specific cashier+date cell - show detail table
                            # Map date label back to actual date
                            selected_date_obj = None
                            for i, label in enumerate(date_labels):
                                if label == selected_date_label:
                                    selected_date_obj = date_list[i]
                                    break
                            
                            if selected_date_obj:
                                st.markdown(f"#### 📋 Serial Details: {selected_cashier} on {selected_date_obj:%d %b %Y}")
                                st.caption(f"🔍 Shop={selected_shop_overview_compliance} | Cashier={selected_cashier} | Date={selected_date_obj:%Y-%m-%d}")
                                
                                # Fetch detail records
                                detail_df = get_cashier_date_detail_serials(
                                    selected_shop_overview_compliance,
                                    selected_cashier,
                                    selected_date_obj
                                )
                                
                                if detail_df is not None and not detail_df.empty:
                                    st.caption(f"Total records: {len(detail_df)} | Sorted: N values first (highlighted in red)")
                                    
                                    # Display detail table with conditional formatting
                                    def _highlight_n_values(row):
                                        styles = [''] * len(row)
                                        if 'serial_check' in row.index:
                                            check_val = str(row['serial_check']).strip().upper()
                                            if check_val == 'N':
                                                # Highlight entire row in red for N values
                                                styles = ['background-color: #fee2e2; color: #991b1b; font-weight: 600;'] * len(row)
                                        return styles
                                    
                                    display_detail = detail_df.copy()
                                    display_detail = display_detail.rename(columns={
                                        'serial_check': 'Serial Check',
                                        'item_code': 'Item Code',
                                        'item_name': 'Item Name',
                                        'serial_number': 'Serial Number',
                                        'bill_date': 'Bill Date',
                                        'shop_code': 'Shop Code',
                                        'till_number': 'Till Number'
                                    })
                                    
                                    styled_detail = (
                                        display_detail
                                        .style
                                        .apply(_highlight_n_values, axis=1)
                                    )
                                    
                                    st.dataframe(styled_detail, use_container_width=True, hide_index=True)
                                    
                                    # Download button
                                    csv = detail_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Detail (CSV)",
                                        data=csv,
                                        file_name=f"serial_detail_{selected_shop_overview_compliance}_{selected_cashier}_{selected_date_obj:%Y%m%d}.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    st.info("No serial data found for this cashier and date.")
                            else:
                                st.error("Unable to map selected date label to actual date.")
                        else:
                            # Show clickable trend table with gradient styling
                            st.caption("💡 Click on any compliance % cell to view serial details for that cashier and date")
                            
                            # Display styled dataframe with clickable events
                            event = st.dataframe(
                                styled,
                                use_container_width=True,
                                hide_index=True,
                                on_select="rerun",
                                selection_mode="single-cell"
                            )
                            
                            # Handle cell selection using cells format
                            if event and hasattr(event, 'selection'):
                                selection = event.selection
                                cells = getattr(selection, 'cells', [])
                                
                                if len(cells) > 0:
                                    # Extract row index and column name from cells array
                                    # Streamlit returns list of tuples: [(row_idx, col_name), ...]
                                    first_cell = cells[0]
                                    row_idx = first_cell[0]
                                    col_name = first_cell[1]
                                    
                                    # Get cashier name from selected row - use iloc for safer access
                                    if row_idx < len(trend_display) and 'Shop/Cashier' in trend_display.columns:
                                        cashier_name = str(trend_display['Shop/Cashier'].iloc[row_idx]).strip()
                                        
                                        # Only drill down if it's a date column (not Shop/Cashier column)
                                        if col_name in date_labels:
                                            st.session_state["overview_compliance_selected_cashier"] = cashier_name
                                            st.session_state["overview_compliance_selected_date"] = col_name
                                            st.rerun()
                else:
                    # Show clickable shop summary for 7 days
                    display_df_7day = by_shop_7day.copy()
                    display_df_7day = display_df_7day.rename(columns={
                        'shop_code': 'Shop Code',
                        'total_serials': 'Total Serials',
                        'total_yes': 'Total Yes',
                        'yes_pct': 'Compliance %'
                    })
                    
                    # Trim spaces from Shop Code column to avoid matching issues
                    if 'Shop Code' in display_df_7day.columns:
                        display_df_7day['Shop Code'] = display_df_7day['Shop Code'].astype(str).str.strip()
                    
                    pct_min_7day = display_df_7day['Compliance %'].min()
                    pct_max_7day = display_df_7day['Compliance %'].max()
                    
                    def _compliance_gradient_7day_summary(col: pd.Series):
                        styles = []
                        
                        # Apply styling to all columns
                        if col.name == 'Compliance %':
                            # Gradient for compliance column
                            if pd.isna(pct_min_7day) or pd.isna(pct_max_7day):
                                return [''] * len(col)
                            span_7day = pct_max_7day - pct_min_7day if pct_max_7day != pct_min_7day else 1
                            for val in col:
                                if pd.isna(val):
                                    styles.append('')
                                    continue
                                ratio = (val - pct_min_7day) / span_7day
                                ratio = max(0, min(1, ratio))
                                # Light red → light yellow → light green gradient
                                if ratio < 0.5:
                                    # Light red to light yellow
                                    local_ratio = ratio * 2
                                    r = int(254)  # Stay red
                                    g = int(226 + (243 - 226) * local_ratio)  # 226 → 243
                                    b = int(226 - (226 - 199) * local_ratio)  # 226 → 199
                                else:
                                    # Light yellow to light green
                                    local_ratio = (ratio - 0.5) * 2
                                    r = int(254 - (254 - 134) * local_ratio)  # 254 → 134
                                    g = int(243 - (243 - 239) * local_ratio)  # 243 → 239
                                    b = int(199 - (199 - 172) * local_ratio)  # 199 → 172
                                styles.append(f'background-color: rgb({r},{g},{b}); color: #000000; font-weight:600; font-size: 297%;')
                        else:
                            # Regular styling for other columns (Shop Code, Total Serials, Total Yes)
                            for val in col:
                                styles.append(f'font-weight:600; font-size: 297%; padding: 4px;')
                        
                        return styles
                    
                    styled_7day = (
                        display_df_7day
                        .style
                        .format({'Total Serials': '{:d}', 'Total Yes': '{:d}', 'Compliance %': '{:.1f}%'})
                        .apply(_compliance_gradient_7day_summary)
                    )
                    
                    # Display clickable table
                    st.caption("💡 Click on any shop row to view 7-day drill-down with cashier breakdown")
                    
                    # Use st.dataframe with on_select for clickable rows
                    event_7day = st.dataframe(
                        styled_7day,
                        use_container_width=True,
                        hide_index=True,
                        on_select="rerun",
                        selection_mode="single-row",
                        column_config={
                            "Shop Code": st.column_config.TextColumn("Shop Code", width="small"),
                            "Total Serials": st.column_config.NumberColumn("Total Serials", format="%d"),
                            "Total Yes": st.column_config.NumberColumn("Total Yes", format="%d"),
                            "Compliance %": st.column_config.NumberColumn("Compliance %", format="%.1f%%")
                        }
                    )
                    
                    # Handle row selection
                    if event_7day and hasattr(event_7day, 'selection') and hasattr(event_7day.selection, 'rows'):
                        selected_rows = event_7day.selection.rows
                        if len(selected_rows) > 0:
                            selected_idx = selected_rows[0]
                            if 0 <= selected_idx < len(display_df_7day):
                                selected_shop_code = str(display_df_7day.iloc[selected_idx]["Shop Code"]).strip()
                                if st.session_state.get("overview_compliance_selected_shop") != selected_shop_code:
                                    st.session_state["overview_compliance_selected_shop"] = selected_shop_code
                                    st.rerun()
            else:
                st.info("No shop-wise compliance data available for last 7 days.")
    
    # ==================== TAB 3: SHOP METRICS ====================
    with tab3:
        st.markdown('<div class="section-header">🏪 Shop Offloading Performance</div>', unsafe_allow_html=True)
        
        # Use warehouse_range from session state for consistency across all tabs
        offload_start, offload_end = warehouse_range
        
        st.caption(f"📅 Using Warehouse Date Range: {offload_start:%Y/%m/%d} → {offload_end:%Y/%m/%d}")

        shop_metrics = get_shop_offload_metrics(offload_start, offload_end)
        shop_perf = get_shop_performance()
        
        if shop_metrics:
            # Offload Summary
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate offload percentage
            if shop_metrics['wh_loaded_total'] > 0:
                offload_pct = (shop_metrics['total_offloaded'] / shop_metrics['wh_loaded_total']) * 100
            else:
                offload_pct = 0
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" title="Total loaded records in serial_no_dailydata; unique = distinct serial_no">
                    <div class="metric-label">Loaded at WH</div>
                    <div class="metric-value">{format_number(shop_metrics['wh_loaded_total'])}</div>
                    <div class="metric-change">✓ {format_number(shop_metrics['wh_loaded_unique'])} Unique</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" title="Total rows with vc_serail_no present; % = Total Offloaded ÷ WH Loaded">
                    <div class="metric-label">Total Offloaded</div>
                    <div class="metric-value">{format_number(shop_metrics['total_offloaded'])}</div>
                    <div class="metric-change">{format_percentage(offload_pct)} of WH Loaded</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card success" title="Distinct vc_serail_no values in serial_no_dailydata">
                    <div class="metric-label">Unique Offloaded</div>
                    <div class="metric-value">{format_number(shop_metrics['unique_offloaded'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                # Shop Sold Serial Compliance % (same as overview)
                sales_metrics = get_sales_metrics(offload_start, offload_end)
                compliance_pct = sales_metrics.get('compliance_pct', 0) if sales_metrics else 0
                card_class = "success" if compliance_pct >= 90 else "danger"
                st.markdown(f"""
                <div class="metric-card {card_class}" title="Shop Sold Serial Compliance %">
                    <div class="metric-label">Shop Sold Serial Compliance %</div>
                    <div class="metric-value">{format_percentage(compliance_pct)}</div>
                    <div class="metric-change">Compliance: Unique Sold / Total Sold</div>
                </div>
                """, unsafe_allow_html=True)

            # Serial Check Y/N Analysis
            st.markdown('<div class="section-header">✅ Serial Check Y/N Analysis</div>', unsafe_allow_html=True)
            
            st.markdown("#### Shop-wise Serial Check %")
            st.caption(f"📅 Using Warehouse Date Range: {warehouse_range[0]:%Y/%m/%d} → {warehouse_range[1]:%Y/%m/%d}")
            
            # Use warehouse_range for serial check analysis consistency
            serial_check_start, serial_check_end = warehouse_range

            serial_check_metrics = get_serial_check_metrics(serial_check_start, serial_check_end)

            summary_col, table_col, chart_col = st.columns([1, 2, 1.2], gap="large")

            if not serial_check_metrics:
                with summary_col:
                    st.warning("Unable to load Serial Check Y/N metrics.")
                with table_col:
                    st.info("No Serial Check activity for the selected date range.")
                with chart_col:
                    st.markdown("###### Last 11 Days Compliance %")
                    st.caption("Daily Compliance % for the last 11 days (yesterday is the latest point)")
                    st.info("Daily trend unavailable.")
            elif serial_check_metrics.get("missing_serial_check"):
                with summary_col:
                    st.warning(f"Serial check column not found in {SERIAL_CHECK_TABLE}.")
                with table_col:
                    st.info("Shop-wise summary not available.")
                with chart_col:
                    st.markdown("###### Last 11 Days Compliance %")
                    st.caption("Daily Compliance % for the last 11 days (yesterday is the latest point)")
                    st.info("Daily trend unavailable.")
            else:
                with summary_col:
                    overall = serial_check_metrics.get("overall")
                    if overall:
                        st.metric("Total Serials", format_number(overall.get('total_serials', 0)))
                        st.metric("Total Yes", format_number(overall.get('total_yes', 0)))
                        st.metric("Compliance %", f"{overall.get('yes_pct', 0)}%")
                        st.caption(f"Filtering {serial_check_start:%Y/%m/%d} → {serial_check_end:%Y/%m/%d}")

                with table_col:
                    if serial_check_metrics.get("missing_shop"):
                        st.info(f"Shop-wise summary not available (shop column missing in {SERIAL_CHECK_TABLE}).")
                    else:
                        by_shop = serial_check_metrics.get("by_shop")
                        if by_shop is not None and not by_shop.empty:
                            st.markdown(
                                f"""
                                <div style="padding:0.6rem 0.8rem; background-color:rgba(0,44,109,0.08); border-left:4px solid {MELCOM_BLUE}; border-radius:8px; font-weight:600; margin-bottom:0.5rem;">
                                    Showing data for {serial_check_start:%Y/%m/%d} → {serial_check_end:%Y/%m/%d}<br>
                                    <span style="font-size:0.85rem; font-weight:500;">Red = lowest Compliance %, Green = highest</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            if "serial_check_selected_shop" not in st.session_state:
                                st.session_state["serial_check_selected_shop"] = None

                            selected_shop = st.session_state.get("serial_check_selected_shop")
                            if selected_shop and selected_shop not in by_shop['shop_code'].tolist():
                                st.session_state["serial_check_selected_shop"] = None
                                selected_shop = None

                            if selected_shop:
                                st.markdown(
                                    f"#### 7-Day Compliance Trend: {selected_shop}")
                                if st.button("Back to shop summary", key="serial_check_back_to_summary"):
                                    st.session_state["serial_check_selected_shop"] = None
                                    st.rerun()

                                trend_result = get_shop_cashier_compliance_trend(selected_shop, days=7)
                                if not trend_result or trend_result.get("data") is None:
                                    st.info("No compliance trend data available for the selected shop.")
                                elif trend_result.get("missing_date"):
                                    st.info(f"Date column not found in {SERIAL_CHECK_TABLE}, unable to render trend.")
                                else:
                                    trend_df = trend_result.get("data")
                                    end_date = trend_result.get("end_date")
                                    start_date = trend_result.get("start_date")

                                    date_list = [start_date + timedelta(days=i) for i in range(7)]
                                    date_labels = [f"{d.day}{d.strftime('%b').lower()}" for d in date_list]

                                    if trend_df is not None and not trend_df.empty:
                                        trend_df['activity_date'] = pd.to_datetime(trend_df['activity_date']).dt.date
                                        trend_df['cashier'] = trend_df['cashier'].fillna('Unknown')
                                        trend_df['total_serials'] = trend_df['total_serials'].fillna(0)
                                        trend_df['total_yes'] = trend_df['total_yes'].fillna(0)
                                        trend_df['compliance_pct'] = (
                                            trend_df['total_yes'].astype(float) /
                                            trend_df['total_serials'].replace({0: pd.NA}).astype(float)
                                        ) * 100
                                        trend_df['compliance_pct'] = trend_df['compliance_pct'].fillna(0).round(2)
                                        
                                        # Mark which rows have data vs no data (for dash formatting)
                                        trend_df['has_data'] = trend_df['total_serials'] > 0

                                        shop_total = (
                                            trend_df
                                            .groupby('activity_date', as_index=False)
                                            .agg({'total_serials': 'sum', 'total_yes': 'sum', 'has_data': 'first'})
                                        )
                                        shop_total['compliance_pct'] = (
                                            shop_total['total_yes'].astype(float) /
                                            shop_total['total_serials'].replace({0: pd.NA}).astype(float)
                                        ) * 100
                                        shop_total['compliance_pct'] = shop_total['compliance_pct'].fillna(0).round(2)
                                        shop_total['cashier'] = selected_shop
                                        shop_total['has_data'] = True  # Shop totals always show data

                                        cashier_pivot = trend_df.pivot_table(
                                            index='cashier',
                                            columns='activity_date',
                                            values='compliance_pct',
                                            aggfunc='mean'
                                        )
                                        
                                        # Also track which cells have data
                                        cashier_has_data = trend_df.pivot_table(
                                            index='cashier',
                                            columns='activity_date',
                                            values='has_data',
                                            aggfunc='any'
                                        )
                                        
                                        shop_pivot = shop_total.pivot_table(
                                            index='cashier',
                                            columns='activity_date',
                                            values='compliance_pct',
                                            aggfunc='mean'
                                        )

                                        combined = pd.concat([shop_pivot, cashier_pivot])
                                        combined_has_data = pd.concat([
                                            pd.DataFrame(True, index=[selected_shop], columns=cashier_has_data.columns),
                                            cashier_has_data
                                        ])
                                    else:
                                        combined = pd.DataFrame(index=pd.Index([selected_shop], name='cashier'))
                                        combined_has_data = pd.DataFrame(False, index=pd.Index([selected_shop], name='cashier'), columns=date_list)

                                    combined = combined.reindex(columns=date_list)
                                    combined.columns = date_labels
                                    
                                    # Also reindex data tracking dataframe
                                    combined_has_data = combined_has_data.reindex(columns=date_list)
                                    combined_has_data = combined_has_data.fillna(False)

                                    trend_display = combined.reset_index().rename(columns={'cashier': 'Shop/Cashier'})
                                    data_availability = combined_has_data.reset_index().rename(columns={'cashier': 'Shop/Cashier'})
                                    
                                    # Trim spaces from Shop/Cashier column to avoid matching issues
                                    if 'Shop/Cashier' in trend_display.columns:
                                        trend_display['Shop/Cashier'] = trend_display['Shop/Cashier'].astype(str).str.strip()
                                    if 'Shop/Cashier' in data_availability.columns:
                                        data_availability['Shop/Cashier'] = data_availability['Shop/Cashier'].astype(str).str.strip()

                                    # Only fill NaN with 0 for the shop row, keep NaN for cashier rows without data
                                    # This way, NaN will display as "-" for missing cashier data
                                    if 'Shop/Cashier' in trend_display.columns and len(trend_display) > 0:
                                        for idx in range(len(trend_display)):
                                            try:
                                                shop_or_cashier = trend_display.at[idx, 'Shop/Cashier']
                                                
                                                if shop_or_cashier == selected_shop:
                                                    # Shop total row: fill all NaN with 0
                                                    for col in date_labels:
                                                        if col in trend_display.columns and pd.isna(trend_display.at[idx, col]):
                                                            trend_display.at[idx, col] = 0.0
                                                else:
                                                    # Cashier row: keep NaN for cells without data, fill 0 only for dates with data
                                                    for col in date_labels:
                                                        if col in trend_display.columns:
                                                            has_data = False
                                                            if col in data_availability.columns:
                                                                try:
                                                                    has_data = data_availability.at[idx, col]
                                                                except:
                                                                    has_data = False
                                                            
                                                            # If this cell has data but shows NaN, fill with 0
                                                            # If this cell has no data, keep it as NaN (displays as "-")
                                                            if has_data and pd.isna(trend_display.at[idx, col]):
                                                                trend_display.at[idx, col] = 0.0
                                            except (KeyError, TypeError, IndexError):
                                                continue

                                    def _row_gradient(row):
                                        styles = [''] * len(row)
                                        
                                        # Get shop/cashier identifier for row matching (safe access)
                                        shop_cashier = None
                                        if 'Shop/Cashier' in row.index:
                                            try:
                                                shop_cashier = row.loc['Shop/Cashier']
                                            except:
                                                pass
                                        
                                        is_shop_row = (shop_cashier == selected_shop)
                                        
                                        # Get date column values for gradient calculation (only for non-NaN values)
                                        date_vals = []
                                        for col in date_labels:
                                            if col in row.index:
                                                val = row.loc[col]
                                                if pd.notna(val):
                                                    date_vals.append(float(val))
                                        
                                        # Calculate min/max only from non-NaN values
                                        if len(date_vals) > 0:
                                            min_val = min(date_vals)
                                            max_val = max(date_vals)
                                            span = max_val - min_val if max_val != min_val else 1
                                        else:
                                            min_val = max_val = span = 0
                                        
                                        for idx, col in enumerate(row.index):
                                            if col not in date_labels:
                                                # Non-date column (Shop/Cashier column)
                                                if is_shop_row:
                                                    styles[idx] = 'font-weight:400; background-color:#e0f2fe; color: #000000; font-size: 338%; padding: 8px;'
                                                continue
                                            
                                            # Get value for this cell
                                            val = row.loc[col]
                                            
                                            if pd.isna(val):
                                                # No data - show "-" in gray italic
                                                styles[idx] = 'background-color: #ffffff; color: #999999; font-style: italic; font-size: 338%; padding: 8px;'
                                            else:
                                                # Has data - apply gradient
                                                if span > 0:
                                                    ratio = (float(val) - min_val) / span
                                                else:
                                                    ratio = 0.5  # Default middle if all values are same
                                                ratio = max(0, min(1, ratio))
                                                
                                                # Light red → light yellow → light green gradient
                                                if ratio < 0.5:
                                                    # Light red to light yellow
                                                    local_ratio = ratio * 2
                                                    r = int(254)  # Stay red
                                                    g = int(226 + (243 - 226) * local_ratio)  # 226 → 243
                                                    b = int(226 - (226 - 199) * local_ratio)  # 226 → 199
                                                else:
                                                    # Light yellow to light green
                                                    local_ratio = (ratio - 0.5) * 2
                                                    r = int(254 - (254 - 134) * local_ratio)  # 254 → 134
                                                    g = int(243 - (243 - 239) * local_ratio)  # 243 → 239
                                                    b = int(199 - (199 - 172) * local_ratio)  # 199 → 172
                                                
                                                styles[idx] = f'background-color: rgb({r},{g},{b}); color: #000000; font-weight:700; font-size: 338%; padding: 8px;'
                                                if is_shop_row:
                                                    styles[idx] += ' border-left: 4px solid #0284c7;'
                                        
                                        return styles

                                    styled = (
                                        trend_display
                                        .style
                                        .format({col: '{:.1f}%' for col in date_labels}, na_rep='-')
                                        .apply(_row_gradient, axis=1)
                                    )

                                    st.dataframe(styled, use_container_width=True, hide_index=True)
                            else:
                                display_df = by_shop.copy()
                                display_df = display_df.rename(columns={
                                    'shop_code': 'Shop Code',
                                    'total_serials': 'Total Serials',
                                    'total_yes': 'Total Yes',
                                    'yes_pct': 'Compliance %'
                                })
                                pct_min = display_df['Compliance %'].min()
                                pct_max = display_df['Compliance %'].max()

                                st.caption("💡 Click on any row to view 7-day compliance trends for that shop and its cashiers")

                                # Create color-coded dataframe using column_config for styling
                                display_for_select = display_df.copy()
                                
                                # Use st.dataframe with on_select for clickable rows
                                event = st.dataframe(
                                    display_for_select,
                                    use_container_width=True,
                                    hide_index=True,
                                    on_select="rerun",
                                    selection_mode="single-row",
                                    column_config={
                                        "Shop Code": st.column_config.TextColumn("Shop Code", width="small"),
                                        "Total Serials": st.column_config.NumberColumn("Total Serials", format="%d"),
                                        "Total Yes": st.column_config.NumberColumn("Total Yes", format="%d"),
                                        "Compliance %": st.column_config.NumberColumn("Compliance %", format="%.1f%%")
                                    }
                                )
                                
                                # Handle row selection
                                if event and hasattr(event, 'selection') and hasattr(event.selection, 'rows'):
                                    selected_rows = event.selection.rows
                                    if len(selected_rows) > 0:
                                        selected_idx = selected_rows[0]
                                        if 0 <= selected_idx < len(display_df):
                                            selected_shop_code = display_df.iloc[selected_idx]["Shop Code"]
                                            if st.session_state.get("serial_check_selected_shop") != selected_shop_code:
                                                st.session_state["serial_check_selected_shop"] = selected_shop_code
                                                st.rerun()
                        else:
                            st.info("No Serial Check activity for the selected date range.")

                with chart_col:
                    st.markdown("###### Last 11 Days Compliance %")
                    st.caption("Daily Compliance % for the last 11 days (yesterday is the latest point)")
                    if serial_check_metrics.get("missing_date"):
                        st.info(f"Date column not found in {SERIAL_CHECK_TABLE}, unable to render trend.")
                    else:
                        last11 = serial_check_metrics.get("by_date_last11")
                        if last11 is not None and not last11.empty:
                            recent_df = last11.copy()
                            try:
                                recent_df['activity_date'] = pd.to_datetime(recent_df['activity_date']).dt.date
                            except Exception:
                                pass
                            if {'total_yes', 'total_serials'}.issubset(recent_df.columns):
                                recent_df['yes_pct_precise'] = (
                                    recent_df['total_yes'].astype(float) /
                                    recent_df['total_serials'].replace({0: pd.NA}).astype(float)
                                ) * 100
                                recent_df['yes_pct_precise'] = recent_df['yes_pct_precise'].fillna(0).round(2)
                            else:
                                recent_df['yes_pct_precise'] = recent_df['yes_pct'].astype(float)
                            fig = px.line(
                                recent_df,
                                x='activity_date',
                                y='yes_pct_precise',
                                markers=True,
                                title=None,
                                labels={'activity_date': 'Date', 'yes_pct_precise': 'Compliance %'}
                            )
                            fig.update_traces(hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Compliance %: %{y:.2f}%<extra></extra>")
                            fig.update_layout(height=280, yaxis_ticksuffix='%', margin=dict(l=20, r=10, t=10, b=30))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No recent data available for the last 11 days.")

                if serial_check_metrics.get("missing_cashier"):
                    st.info(f"Cashier-wise summary not available (cashier column missing in {SERIAL_CHECK_TABLE}).")
                else:
                    by_cashier = serial_check_metrics.get("by_cashier")
                    if by_cashier is not None and not by_cashier.empty:
                        st.markdown("#### Cashier-wise Serial Check %")
                        st.dataframe(
                            by_cashier,
                            use_container_width=True,
                            column_config={
                                "cashier": st.column_config.TextColumn("Cashier", width="medium"),
                                "total_serials": st.column_config.NumberColumn("Total Serials", format="%d"),
                                "total_yes": st.column_config.NumberColumn("Total Yes", format="%d"),
                                "yes_pct": st.column_config.NumberColumn("Compliance %", format="%.1f")
                            },
                            hide_index=True
                        )

                if serial_check_metrics.get("missing_cashier") or serial_check_metrics.get("missing_bill") or serial_check_metrics.get("missing_serial"):
                    st.info(f"Cashier duplicate-serials by bill not available (cashier/bill/serial column missing in {SERIAL_CHECK_TABLE}).")
                else:
                    dup_by_cashier = serial_check_metrics.get("dup_by_cashier")
                    if dup_by_cashier is not None and not dup_by_cashier.empty:
                        st.markdown("#### Cashiers with Same Serial Across Multiple Bills")
                        st.dataframe(
                            dup_by_cashier,
                            use_container_width=True,
                            column_config={
                                "cashier": st.column_config.TextColumn("Cashier", width="medium"),
                                "serial_number": st.column_config.TextColumn("Serial Number", width="medium"),
                                "bill_count": st.column_config.NumberColumn("Bill Count", format="%d"),
                                "bills": st.column_config.TextColumn("Bills", width="large")
                            },
                            hide_index=True
                        )
            
            # Time Bucket Analysis
            st.markdown('<div class="section-header">⏱ Offloading Time Analysis</div>', unsafe_allow_html=True)
            st.info("**Time Calculation**: Shop Offload Date (dt_mod_date) - WH Loaded Date (loaded_datetime::DATE)")
            
            bucket_data = {
                'Time Bucket': ['0-3 days', '4-5 days', '6-7 days', '8-10 days', '>10 days', 'No Offload Date'],
                'Count': [
                    shop_metrics['bucket_0_3_days'],
                    shop_metrics['bucket_3_5_days'],
                    shop_metrics['bucket_5_7_days'],
                    shop_metrics['bucket_7_10_days'],
                    shop_metrics['bucket_10_plus_days'],
                    shop_metrics['no_offloaded']
                ]
            }
            
            bucket_df = pd.DataFrame(bucket_data)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    bucket_df,
                    x='Time Bucket',
                    y='Count',
                    title='Shop Offloading Time Distribution (Days Post WH Loading)',
                    color='Count',
                    color_continuous_scale=[MELCOM_GREEN, MELCOM_ORANGE, MELCOM_RED]
                )
                fig.update_layout(height=400)
                fig.add_annotation(
                    text="UNDER VALIDATION",
                    font=dict(size=36, color="rgba(0,0,0,0.18)", family="Arial Black"),
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    textangle=-25,
                    opacity=0.6
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display as metrics
                for idx, row in bucket_df.iterrows():
                    total = bucket_df['Count'].sum()
                    pct = (row['Count'] / total * 100) if total > 0 else 0
                    st.metric(
                        label=row['Time Bucket'],
                        value=format_number(row['Count']),
                        delta=f"{pct:.1f}%"
                    )
            
            # Detailed offloading records with shop and dates
            st.markdown('<div class="section-header">📋 Detailed Offloading Records (Where & When)</div>', unsafe_allow_html=True)
            
            offload_details = get_offloading_details_by_shop_and_time(offload_start, offload_end)
            
            if offload_details is not None and not offload_details.empty:
                # Shop-level summary - handle NULL dates properly
                def safe_date_range(dates):
                    """Get date range handling NaT/NULL values"""
                    valid_dates = dates.dropna()
                    if len(valid_dates) == 0:
                        return "No dates"
                    min_date = valid_dates.min()
                    max_date = valid_dates.max()
                    if min_date == max_date:
                        return str(min_date)
                    return f"{min_date} to {max_date}"
                
                shop_summary = offload_details.groupby('shop_code').agg({
                    'serial_no': 'count',
                    'days_to_offload': 'mean',
                    'wh_loaded_date': safe_date_range,
                    'shop_offload_date': safe_date_range
                }).reset_index()
                shop_summary.columns = ['Shop Code', 'Total Items Offloaded', 'Avg Days to Offload', 'WH Loading Date Range', 'Shop Offload Date Range']
                shop_summary['Avg Days to Offload'] = shop_summary['Avg Days to Offload'].round(1)
                
                st.markdown("### 🏪 Shop-Level Offloading Summary")
                st.dataframe(
                    shop_summary,
                    use_container_width=True,
                    column_config={
                        "Shop Code": st.column_config.TextColumn("Shop Code", width="small"),
                        "Total Items Offloaded": st.column_config.NumberColumn("Total Items", format="%d"),
                        "Avg Days to Offload": st.column_config.NumberColumn("Avg Days", format="%.1f"),
                        "WH Loading Date Range": st.column_config.TextColumn("WH Loading Date Range", width="medium"),
                        "Shop Offload Date Range": st.column_config.TextColumn("Shop Offload Date Range", width="medium")
                    },
                    hide_index=True
                )
                
                # Filters for detailed view
                st.markdown("### 🔍 Filter Detailed Records")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    selected_shops = st.multiselect(
                        "Select Shops",
                        options=['All'] + sorted(offload_details['shop_code'].unique().tolist()),
                        default=['All']
                    )
                
                with col2:
                    selected_time_bucket = st.selectbox(
                        "Select Time Bucket",
                        options=['All'] + sorted(offload_details['time_bucket'].unique().tolist())
                    )
                
                with col3:
                    limit_records = st.number_input("Display Limit", min_value=10, max_value=10000, value=100, step=10)
                
                # Apply filters
                filtered_details = offload_details.copy()
                
                if 'All' not in selected_shops:
                    filtered_details = filtered_details[filtered_details['shop_code'].isin(selected_shops)]
                
                if selected_time_bucket != 'All':
                    filtered_details = filtered_details[filtered_details['time_bucket'] == selected_time_bucket]
                
                # Display filtered records
                st.markdown(f"### 📊 Showing {min(len(filtered_details), limit_records):,} of {len(filtered_details):,} Records")
                
                display_df = filtered_details.head(limit_records)[
                    ['shop_code', 'serial_no', 'item_code', 'item_desc', 
                     'wh_loaded_date', 'shop_offload_date', 'days_to_offload', 'time_bucket']
                ]
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    column_config={
                        "shop_code": st.column_config.TextColumn("Shop", width="small"),
                        "serial_no": st.column_config.TextColumn("Serial No", width="medium"),
                        "item_code": st.column_config.TextColumn("Item Code", width="small"),
                        "item_desc": st.column_config.TextColumn("Item Description", width="large"),
                        "wh_loaded_date": st.column_config.DateColumn("WH Loaded Date", format="YYYY-MM-DD"),
                        "shop_offload_date": st.column_config.DateColumn("Shop Offload Date", format="YYYY-MM-DD"),
                        "days_to_offload": st.column_config.NumberColumn("Days to Offload", format="%d"),
                        "time_bucket": st.column_config.TextColumn("Time Bucket", width="small")
                    },
                    hide_index=True
                )
                
                # Download button for filtered data
                csv = filtered_details.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Filtered Records (CSV)",
                    data=csv,
                    file_name=f"offloading_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No offloading details available")
        
        # Shop Performance Table
        if shop_perf is not None and not shop_perf.empty:
            st.markdown('<div class="section-header">🏪 Shop-wise Performance</div>', unsafe_allow_html=True)
            
            st.dataframe(
                shop_perf,
                use_container_width=True,
                column_config={
                    "shop_code": st.column_config.TextColumn("Shop Code", width="small"),
                    "total_items": st.column_config.NumberColumn("Total Items", format="%d"),
                    "offloaded_count": st.column_config.NumberColumn("Offloaded", format="%d"),
                    "sold_count": st.column_config.NumberColumn("Sold", format="%d"),
                    "not_offloaded": st.column_config.NumberColumn("Not Offloaded", format="%d"),
                    "shop_mismatch": st.column_config.NumberColumn("Shop Mismatch", format="%d"),
                    "serial_mismatch": st.column_config.NumberColumn("Serial Mismatch", format="%d"),
                    "avg_offload_days": st.column_config.NumberColumn("Avg Offload Days", format="%.1f")
                },
                hide_index=True
            )
            
            # Top/Bottom performers
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🏆 Shops Offloaded before WH dispatch**")
                top_shops = shop_perf.nsmallest(5, 'avg_offload_days')[['shop_code', 'avg_offload_days', 'total_items']]
                st.dataframe(top_shops, hide_index=True, use_container_width=True)
            
            with col2:
                st.markdown("**🐌 Slowest Offloading Shops**")
                bottom_shops = shop_perf.nlargest(5, 'avg_offload_days')[['shop_code', 'avg_offload_days', 'total_items']]
                st.dataframe(bottom_shops, hide_index=True, use_container_width=True)

    # ==================== TAB 4: SERIAL JOURNEY ====================
    with tab_journey:
        st.markdown('<div class="section-header">🧭 Serial Journey</div>', unsafe_allow_html=True)
        st.markdown("## 🔍 Complete Serial Number Journey Tracker")
        st.info("💡 Track serial numbers from warehouse receipt through to final sale, with verification against main database")

        # Use warehouse_range for date filtering (selected at top of dashboard)
        start_date = warehouse_range[0]
        end_date = warehouse_range[1]
        
        st.caption(f"📅 Date range: {start_date:%Y/%m/%d} to {end_date:%Y/%m/%d} (selected in Warehouse Date Filter above)")

        # ============================================================
        # SERIAL NUMBER SEARCH
        # ============================================================
        st.markdown("### 🔎 Search Serial Number")
        
        # Add CSS styling for white background on search input
        st.markdown("""
            <style>
            div[data-testid="stTextInput"] > div > div {
                background-color: white !important;
                border: 2px solid #4CAF50 !important;
                border-radius: 5px !important;
                padding: 5px !important;
            }
            div[data-testid="stTextInput"] input {
                background-color: white !important;
                color: #000000 !important;
                font-weight: 500 !important;
            }
            </style>
        """, unsafe_allow_html=True)

        col_input, col_btn, col_empty = st.columns([0.20, 0.10, 0.80])
        with col_input:
            search_serial = st.text_input(
                "Enter Serial Number",
                placeholder="Type serial number to search...",
                key="serial_search",
                label_visibility="collapsed"
            )
        with col_btn:
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

        if search_clicked and search_serial:
            with st.spinner(f"Searching for serial: {search_serial}..."):
                journey_df = search_serial_number(search_serial)

                if not journey_df.empty:
                    st.success(f"✅ Found {len(journey_df)} record(s) for serial: **{search_serial}**")

                    st.markdown("#### 📋 Complete Journey")

                    for idx, row in journey_df.iterrows():
                        issues_found = []
                        if row.get('has_shop_mismatch') == 'YES':
                            issues_found.append("🔴 Shop Mismatch")
                        if row.get('has_vehicle_mismatch') == 'YES':
                            issues_found.append("🔴 Vehicle Mismatch")
                        if row.get('verified_in_main_db') == 'NO':
                            issues_found.append("🔴 Not in Main DB")
                        if row.get('status') in ['Not Sold', 'Not Offloaded', 'Not Loaded to Shop']:
                            issues_found.append(f"⚠️ {row.get('status')}")

                        title = f"🔍 Journey Record {idx + 1}"
                        if issues_found:
                            title += f" ⚠️ {len(issues_found)} Issue(s) Detected"

                        with st.expander(title, expanded=False):
                            if issues_found:
                                st.error("**⚠️ Issues Detected in This Journey:**")
                                for issue in issues_found:
                                    st.markdown(f"- {issue}")
                                st.markdown("---")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.markdown("**🏭 Warehouse Receipt**")
                                st.write(f"**GRN Date:** {row.get('wh_grn_date') or 'N/A'}")
                                st.write(f"**Warehouse:** {row.get('wh_received_warehouse') or 'N/A'}")
                                st.write(f"**Supplier:** {row.get('supplier_name') or 'N/A'}")
                                st.write(f"**Inbound Type:** {row.get('inbound_type') or 'N/A'}")

                            with col2:
                                st.markdown("**🚚 Distribution**")
                                st.write(f"**Doc Date:** {row.get('wh_doc_date') or 'N/A'}")
                                st.write(f"**Loaded:** {row.get('loaded_datetime') or 'N/A'}")

                                shop_text = f"**Sent to Shop:** {row.get('sent_to_shop') or 'N/A'}"
                                if row.get('has_shop_mismatch') == 'YES':
                                    st.markdown(f":red[{shop_text}] ⚠️")
                                else:
                                    st.write(shop_text)

                                st.write(f"**Shop Name:** {row.get('sent_to_shop_name') or 'N/A'}")

                                wh_vehicle_text = f"**WH Vehicle:** {row.get('wh_vehicle') or 'N/A'}"
                                shop_vehicle_text = f"**Shop Vehicle:** {row.get('shop_vehicle') or 'N/A'}"
                                if row.get('has_vehicle_mismatch') == 'YES':
                                    st.markdown(f":red[{wh_vehicle_text}] ⚠️")
                                    st.markdown(f":red[{shop_vehicle_text}] ⚠️")
                                else:
                                    st.write(wh_vehicle_text)
                                    st.write(shop_vehicle_text)

                            with col3:
                                st.markdown("**🛒 Sale**")
                                sold_shop_text = f"**Sold at Shop:** {row.get('shop_sold') or 'N/A'}"
                                if row.get('has_shop_mismatch') == 'YES':
                                    st.markdown(f":red[{sold_shop_text}] ⚠️")
                                else:
                                    st.write(sold_shop_text)

                                st.write(f"**Sale Date:** {row.get('sale_date') or 'N/A'}")
                                st.write(f"**Invoice:** {row.get('sale_invoice') or 'N/A'}")
                                st.write(f"**Bill No:** {row.get('bill_no') or 'N/A'}")
                                st.write(f"**Cashier:** {row.get('cashier_name') or 'N/A'}")
                                st.write(f"**Till:** {row.get('till_number') or 'N/A'}")

                            st.markdown("---")
                            col_status1, col_status2, col_status3, col_status4 = st.columns(4)

                            with col_status1:
                                status_color = {
                                    'Sold': '🟢',
                                    'Not Sold': '🟡',
                                    'Not Offloaded': '🟠',
                                    'Not Loaded to Shop': '🔴'
                                }.get(row.get('status'), '⚪')
                                st.metric("Status", f"{status_color} {row.get('status')}")

                            with col_status2:
                                mismatch_color = '🔴' if row.get('has_shop_mismatch') == 'YES' else '🟢'
                                st.metric("Shop Match", f"{mismatch_color} {row.get('has_shop_mismatch')}")

                            with col_status3:
                                vehicle_color = '🔴' if row.get('has_vehicle_mismatch') == 'YES' else '🟢'
                                st.metric("Vehicle Match", f"{vehicle_color} {row.get('has_vehicle_mismatch')}")

                            with col_status4:
                                db_color = '🟢' if row.get('verified_in_main_db') == 'YES' else '🔴'
                                st.metric("In Main DB", f"{db_color} {row.get('verified_in_main_db')}")

                            st.markdown("**📦 Item Information**")
                            st.write(f"**Item Code:** {row.get('vc_item_code') or 'N/A'}")
                            st.write(f"**Description:** {row.get('vc_item_desc') or 'N/A'}")
                            st.write(f"**Selling Price:** {row.get('nu_selling_price') or 'N/A'}")

                            st.markdown("**🔢 Serial Numbers**")
                            st.write(f"**WH Serial:** {row.get('wh_serial') or 'N/A'}")
                            st.write(f"**Offload Serial:** {row.get('offload_serial') or 'N/A'}")
                            st.write(f"**Sold Serial:** {row.get('sold_serial') or 'N/A'}")
                else:
                    st.warning(f"⚠️ No records found for serial number: **{search_serial}**")

        st.markdown("---")

        # ============================================================
        # JOURNEY OVERVIEW METRICS
        # ============================================================
        st.markdown("### 📊 Journey Overview (All Serials)")

        overview_data = get_serial_journey_overview(start_date, end_date)

        if overview_data:
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(
                    "🏭 WH Received",
                    f"{overview_data.get('received_count', 0):,}",
                    help="Serial numbers received at warehouse (from GRN)"
                )

            with col2:
                st.metric(
                    "📦 Loaded to Shops",
                    f"{overview_data.get('loaded_count', 0):,}",
                    delta=f"-{overview_data.get('not_loaded', 0):,} not loaded" if overview_data.get('not_loaded', 0) > 0 else None,
                    delta_color="inverse"
                )

            with col3:
                st.metric(
                    "🚚 Offloaded",
                    f"{overview_data.get('offloaded_count', 0):,}",
                    delta=f"-{overview_data.get('not_offloaded', 0):,} not offloaded" if overview_data.get('not_offloaded', 0) > 0 else None,
                    delta_color="inverse"
                )

            with col4:
                st.metric(
                    "🛒 Sold",
                    f"{overview_data.get('sold_count', 0):,}",
                    delta=f"-{overview_data.get('not_sold', 0):,} not sold" if overview_data.get('not_sold', 0) > 0 else None,
                    delta_color="inverse"
                )

            with col5:
                st.metric(
                    "🏪 Shops Involved",
                    f"{overview_data.get('shops_sent_to', 0):,}",
                    help="Number of unique shops that received serials"
                )

            st.markdown("#### ⚠️ Issues Detected")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "🏪 Shop Mismatches",
                    f"{overview_data.get('shop_mismatch_count', 0):,}",
                    help="Offloaded at one shop, sold at another"
                )

            with col2:
                st.metric(
                    "🚛 Vehicle Mismatches",
                    f"{overview_data.get('vehicle_mismatch_count', 0):,}",
                    help="Different vehicles at warehouse vs shop"
                )

            with col3:
                raw_in_db = overview_data.get('in_main_db')
                raw_not_in_db = overview_data.get('not_in_main_db')
                try:
                    in_db = int(raw_in_db) if raw_in_db is not None else 0
                except Exception:
                    try:
                        in_db = int(float(raw_in_db))
                    except Exception:
                        in_db = 0
                try:
                    not_in_db = int(raw_not_in_db) if raw_not_in_db is not None else 0
                except Exception:
                    try:
                        not_in_db = int(float(raw_not_in_db))
                    except Exception:
                        not_in_db = 0
                st.metric(
                    "✅ Verified in Main DB",
                    f"{in_db:,}",
                    help="Serial numbers verified in main database"
                )

            with col4:
                st.metric(
                    "❌ Not in Main DB",
                    f"{not_in_db:,}",
                    delta=f"{(not_in_db / (in_db + not_in_db) * 100):.1f}%" if (in_db + not_in_db) > 0 else "0%",
                    delta_color="inverse",
                    help="Serial numbers not found in main database"
                )

        st.markdown("---")

        # ============================================================
        # GAP ANALYSIS BY SHOP (WITH DRILL-DOWN)
        # ============================================================
        st.markdown("### 📈 Shop-wise Gap Analysis with Drill-Down")
        st.info("💡 Select a shop to see detailed serial journey with personnel info")

        gap_df = get_gap_analysis_by_shop(start_date, end_date)

        if not gap_df.empty:
            gap_df['offload_rate'] = (
                gap_df['offloaded_count'] / gap_df['loaded_count'].replace(0, pd.NA) * 100
            ).round(1).fillna(0)
            gap_df['sale_rate'] = (
                gap_df['sold_count'] / gap_df['loaded_count'].replace(0, pd.NA) * 100
            ).round(1).fillna(0)

            selected_gap_shop = st.selectbox(
                "Select Shop for Details",
                ['All'] + gap_df['vc_shop_code'].tolist(),
                key='gap_shop_selector'
            )
            # Save selected shop to session state for use in other sections
            st.session_state['selected_shop'] = selected_gap_shop

            if selected_gap_shop == 'All':
                fig = go.Figure()

                fig.add_trace(go.Bar(
                    name='Loaded',
                    x=gap_df['vc_shop_code'],
                    y=gap_df['loaded_count'],
                    marker_color='#3498db'
                ))

                fig.add_trace(go.Bar(
                    name='Offloaded',
                    x=gap_df['vc_shop_code'],
                    y=gap_df['offloaded_count'],
                    marker_color='#f39c12'
                ))

                fig.add_trace(go.Bar(
                    name='Sold',
                    x=gap_df['vc_shop_code'],
                    y=gap_df['sold_count'],
                    marker_color='#27ae60'
                ))

                fig.update_layout(
                    barmode='group',
                    title="Serial Flow by Shop: Loaded → Offloaded → Sold",
                    xaxis_title="Shop Code",
                    yaxis_title="Count",
                    height=450,
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### 📋 Shop-wise Summary")
                display_gap_df = gap_df[[
                    'vc_shop_code', 'shop_name', 'loaded_count', 'offloaded_count', 
                    'sold_count', 'offload_rate', 'sale_rate', 'shop_mismatch', 'vehicle_mismatch'
                ]].copy()

                display_gap_df.columns = [
                    'Shop Code', 'Shop Name', 'Loaded', 'Offloaded', 
                    'Sold', 'Offload %', 'Sale %', 'Shop Mismatch', 'Vehicle Mismatch'
                ]

                st.dataframe(display_gap_df, use_container_width=True, height=400)

                csv = gap_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Gap Analysis",
                    data=csv,
                    file_name=f"gap_analysis_{start_date}_{end_date}.csv",
                    mime="text/csv",
                    key="download_gap_analysis"
                )
            else:
                shop_data = gap_df[gap_df['vc_shop_code'] == selected_gap_shop].iloc[0]

                st.markdown(f"### 🏪 {selected_gap_shop} - {shop_data['shop_name']}")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📦 Loaded", f"{shop_data['loaded_count']:,}")
                with col2:
                    st.metric("🚚 Offloaded", f"{shop_data['offloaded_count']:,}", 
                             delta=f"{shop_data['offload_rate']}%")
                with col3:
                    st.metric("🛒 Sold", f"{shop_data['sold_count']:,}",
                             delta=f"{shop_data['sale_rate']}%")
                with col4:
                    st.metric("⚠️ Issues", f"{shop_data['shop_mismatch'] + shop_data['vehicle_mismatch']:,}")

                st.markdown("---")

                with st.spinner(f"Loading detailed journey for {selected_gap_shop}..."):
                    shop_serials_df = get_shop_serial_details(start_date, end_date, selected_gap_shop)

                if not shop_serials_df.empty:
                    st.markdown(f"#### 📋 Complete Serial Journey - {len(shop_serials_df):,} Records")
                    st.caption(f"Showing all serials sent to {selected_gap_shop} with complete journey details including personnel info")

                    col_filter1, col_filter2 = st.columns(2)

                    with col_filter1:
                        if 'issue_category' in shop_serials_df.columns:
                            issue_filter = st.multiselect(
                                "Filter by Issue Category",
                                options=shop_serials_df['issue_category'].dropna().unique().tolist(),
                                default=None,
                                key=f'issue_filter_{selected_gap_shop}'
                            )
                        else:
                            issue_filter = []
                            st.info("Issue category not available")

                    with col_filter2:
                        if 'supplier' in shop_serials_df.columns:
                            supplier_filter = st.multiselect(
                                "Filter by Supplier",
                                options=shop_serials_df['supplier'].dropna().unique().tolist(),
                                default=None,
                                key=f'supplier_filter_{selected_gap_shop}'
                            )
                        else:
                            supplier_filter = []
                            st.info("Supplier data not available")

                    filtered_df = shop_serials_df.copy()

                    if issue_filter and 'issue_category' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['issue_category'].isin(issue_filter)]

                    if supplier_filter and 'supplier' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['supplier'].isin(supplier_filter)]

                    st.info(f"Showing {len(filtered_df):,} of {len(shop_serials_df):,} records")

                    all_possible_cols = {
                        'wh_serial': 'WH Serial',
                        'vc_item_code': 'Item Code',
                        'vc_item_desc': 'Item Description',
                        'nu_selling_price': 'Price (₵)',
                        'when_loaded': '📅 Loaded',
                        'loading_vehicle': '🚚 Load Vehicle',
                        'when_offloaded': '📅 Offloaded',
                        'offload_serial': 'Offload Serial',
                        'offload_vehicle': '🚚 Offload Vehicle',
                        'when_sold': '📅 Sold',
                        'sold_serial': 'Sold Serial',
                        'selling_shop': '🏪 Selling Shop',
                        'invoice_no': 'Invoice',
                        'issue_category': '⚠️ Issue',
                        'shop_mismatch': '🏪 Shop Mismatch',
                        'vehicle_mismatch': '🚛 Vehicle Mismatch',
                        'supplier': 'Supplier',
                        'wh_grn_date': 'WH GRN Date',
                        'cashier_name': '👤 Cashier',
                        'till_number': '🖥️ Till',
                        'bill_no': '📄 Bill No',
                        'serial_check': '✅ Serial Check'
                    }

                    display_cols = []
                    display_names = []

                    for col, name in all_possible_cols.items():
                        if col in filtered_df.columns:
                            display_cols.append(col)
                            display_names.append(name)

                    if display_cols:
                        display_df = filtered_df[display_cols].copy()
                        display_df.columns = display_names
                    else:
                        st.error("No columns available to display")
                        display_df = pd.DataFrame()

                    timestamp_cols = ['📅 Loaded', '📅 Offloaded', '📅 Sold', 'WH GRN Date']
                    for col in timestamp_cols:
                        if col in display_df.columns:
                            display_df[col] = pd.to_datetime(display_df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')

                    if not display_df.empty and len(display_df.columns) > 0:
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            height=500
                        )
                    else:
                        st.warning("No data available to display after filtering")

                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Complete Serial Journey",
                        data=csv,
                        file_name=f"serial_journey_{selected_gap_shop}_{start_date}_{end_date}.csv",
                        mime="text/csv",
                        key="download_serial_journey"
                    )
                else:
                    st.warning(f"No serial data found for {selected_gap_shop}")
        else:
            st.info("No data available for the selected date range")

        st.markdown("---")

        # ============================================================
        # UNVERIFIED SERIALS
        # ============================================================
        st.markdown("### ❌ Serial Numbers Not in Main Database")

        unverified_df = get_unverified_serials(start_date, end_date)

        if not unverified_df.empty:
            # Use shop selected from Gap Analysis section, or show all if 'All' selected
            selected_shop_filter = st.session_state.get('selected_shop', 'All')
            
            if selected_shop_filter != 'All':
                filtered_df = unverified_df[unverified_df['shop_code'] == selected_shop_filter]
                st.info(f"📍 Showing data for selected shop: **{selected_shop_filter}** | {len(filtered_df):,} unverified serials")
            else:
                filtered_df = unverified_df
                st.info(f"💡 Select a shop above in the Gap Analysis section to filter. Showing **{len(filtered_df):,} total unverified** across all shops.")
            
            st.warning(f"⚠️ Found **{len(filtered_df):,}** serial numbers not verified in main database")

            st.dataframe(filtered_df, use_container_width=True, height=400)

            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download Unverified Serials",
                data=csv,
                file_name=f"unverified_serials_{start_date}_{end_date}.csv",
                mime="text/csv",
                key="download_unverified_serials"
            )
        else:
            st.success("✅ All serial numbers verified in main database!")
    
    # ==================== TAB 4: DISCREPANCIES ====================
    with tab4:
        st.markdown('<div class="section-header">🚨 Discrepancy Detection</div>', unsafe_allow_html=True)
        
        discrepancy_summary = get_discrepancy_summary()
        not_offloaded = get_not_offloaded_metrics()
        # Use warehouse_range for date filtering, or fallback to yesterday if not available
        try:
            sales_start, sales_end = warehouse_range
        except Exception:
            sales_start = sales_end = (datetime.today() - timedelta(days=1)).date()
        sales_metrics = get_sales_metrics(sales_start, sales_end)
        dup_sales = get_duplicate_sales_details()

        if discrepancy_summary:
            st.markdown("### 🔎 Mismatch Overview")
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Shop Mismatch", format_number(discrepancy_summary.get('shop_mismatch', 0)))
            with col2:
                st.metric("Serial Mismatch", format_number(discrepancy_summary.get('serial_mismatch', 0)))
            with col3:
                st.metric("Vehicle Mismatch", format_number(discrepancy_summary.get('vehicle_mismatch', 0)))
            with col4:
                st.metric("No Offloading", format_number(discrepancy_summary.get('no_offloading', 0)))
            with col5:
                st.metric("Sold w/o Offload", format_number(discrepancy_summary.get('sold_without_offloading', 0)))

            mismatch_df = pd.DataFrame({
                'Issue': [
                    'Shop Mismatch',
                    'Serial Mismatch',
                    'Vehicle Mismatch',
                    'No Offloading',
                    'Sold w/o Offload'
                ],
                'Count': [
                    discrepancy_summary.get('shop_mismatch', 0),
                    discrepancy_summary.get('serial_mismatch', 0),
                    discrepancy_summary.get('vehicle_mismatch', 0),
                    discrepancy_summary.get('no_offloading', 0),
                    discrepancy_summary.get('sold_without_offloading', 0)
                ]
            })

            col_a, col_b = st.columns([2, 1])
            with col_a:
                fig_bar = px.bar(
                    mismatch_df,
                    x='Issue',
                    y='Count',
                    title='Mismatch Counts by Category',
                    color='Count',
                    color_continuous_scale=[MELCOM_GREEN, MELCOM_ORANGE, MELCOM_RED]
                )
                fig_bar.update_layout(height=380, xaxis_title="", yaxis_title="Count")
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_b:
                fig_pie = px.pie(
                    mismatch_df,
                    values='Count',
                    names='Issue',
                    title='Mismatch Share',
                    color_discrete_sequence=[MELCOM_GREEN, MELCOM_CYAN, MELCOM_ORANGE, MELCOM_PURPLE, MELCOM_RED]
                )
                fig_pie.update_layout(height=380)
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")
        
        if not_offloaded:
            # Time Since WH Load
            st.markdown("### Days Since WH Loading (Not Yet Offloaded)")
            
            bucket_data = {
                'Days Since WH Load': ['0-3 days', '3-5 days', '5-7 days', '7-10 days', '>10 days'],
                'Count': [
                    not_offloaded['bucket_0_3_days'],
                    not_offloaded['bucket_3_5_days'],
                    not_offloaded['bucket_5_7_days'],
                    not_offloaded['bucket_7_10_days'],
                    not_offloaded['bucket_10_plus_days']
                ]
            }
            
            bucket_df = pd.DataFrame(bucket_data)
            
            fig = px.pie(
                bucket_df,
                values='Count',
                names='Days Since WH Load',
                title='Not Offloaded - Time Distribution',
                color_discrete_sequence=[MELCOM_GREEN, MELCOM_CYAN, MELCOM_ORANGE, MELCOM_RED, '#8B0000']
            )
            fig.update_layout(height=400)
            if plotly_events:
                selected = plotly_events(
                    fig,
                    click_event=True,
                    select_event=False,
                    hover_event=False,
                    override_height=400,
                    override_width="100%",
                    key="not_offloaded_pie"
                )
            else:
                st.plotly_chart(fig, use_container_width=True)
                selected = []

            bucket_options = bucket_df['Days Since WH Load'].tolist()
            select_options = ['Select a bucket'] + bucket_options if bucket_options else []
            selected_bucket = st.selectbox(
                "Select a bucket to view details",
                select_options,
                index=0,
                key="not_offloaded_bucket_select"
            ) if select_options else None

            if selected:
                selected_bucket = selected[0].get("label") or selected_bucket

            if selected_bucket and selected_bucket != 'Select a bucket':
                st.markdown(f"#### 📋 Not Offloaded Details: {selected_bucket}")
                detail_df = get_not_offloaded_details(selected_bucket, limit=1000)
                if not detail_df.empty:
                    st.dataframe(detail_df, use_container_width=True, height=400)
                    csv = detail_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Not Offloaded Details",
                        data=csv,
                        file_name=f"not_offloaded_{selected_bucket.replace(' ', '_')}.csv",
                        mime="text/csv",
                        key="download_not_offloaded_details"
                    )
                else:
                    st.info("No records for the selected bucket.")

            st.markdown("---")
            st.markdown("### 🏪 Shops with Most Offloading Pending")

            pending_by_shop = get_offloading_pending_by_shop(
                bucket_label=selected_bucket if selected_bucket != 'Select a bucket' else None,
                limit=20
            )
            if pending_by_shop is not None and not pending_by_shop.empty:
                st.dataframe(
                    pending_by_shop,
                    use_container_width=True,
                    column_config={
                        "shop_code": st.column_config.TextColumn("Shop Code", width="small"),
                        "shop_name": st.column_config.TextColumn("Shop Name", width="medium"),
                        "pending_offload": st.column_config.NumberColumn("Pending Offload", format="%d"),
                        "total_loaded": st.column_config.NumberColumn("Total Loaded", format="%d"),
                        "shop_mismatch_pct": st.column_config.NumberColumn("Shop Mismatch %", format="%.1f"),
                        "serial_mismatch_pct": st.column_config.NumberColumn("Serial Mismatch %", format="%.1f"),
                        "vehicle_mismatch_pct": st.column_config.NumberColumn("Vehicle Mismatch %", format="%.1f")
                    },
                    height=380,
                    hide_index=True
                )

                fig_pending = px.bar(
                    pending_by_shop,
                    x="shop_code",
                    y="pending_offload",
                    title="Top Shops by Offloading Pending",
                    color="pending_offload",
                    color_continuous_scale=[MELCOM_GREEN, MELCOM_ORANGE, MELCOM_RED]
                )
                fig_pending.update_layout(height=350, xaxis_title="Shop", yaxis_title="Pending Offload")
                st.plotly_chart(fig_pending, use_container_width=True)
            else:
                st.info("No pending offloading data available by shop.")
        
        # Duplicate Sales
        if sales_metrics and dup_sales is not None:
            st.markdown("---")
            st.markdown("### 🚨 Duplicate Sales (Potential Fraud)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card danger" title="Distinct shop_serail_no that appear more than once">
                    <div class="metric-label">Serials Sold Multiple Times</div>
                    <div class="metric-value">{format_number(sales_metrics['unique_duplicate_serials'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card danger" title="Count of rows where shop_serail_no appears more than once">
                    <div class="metric-label">Total Duplicate Sales</div>
                    <div class="metric-value">{format_number(sales_metrics['duplicate_sales_count'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card danger" title="Total Sold minus Unique Sold">
                    <div class="metric-label">Extra Sales Count</div>
                    <div class="metric-value">{format_number(sales_metrics['duplicate_difference'])}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if not dup_sales.empty:
                st.markdown("#### Detailed Duplicate Sales Report")
                st.warning("⚠ These serial numbers have been sold multiple times across different shops/dates")
                
                st.dataframe(
                    dup_sales,
                    use_container_width=True,
                    column_config={
                        "serial_number": st.column_config.TextColumn("Serial Number", width="medium"),
                        "times_sold": st.column_config.NumberColumn("Times Sold", format="%d"),
                        "shops_sold": st.column_config.TextColumn("Shops", width="medium"),
                        "item_codes": st.column_config.TextColumn("Item Codes", width="small"),
                        "item_names": st.column_config.TextColumn("Item Names", width="large"),
                        "first_sale_date": st.column_config.DateColumn("First Sale"),
                        "last_sale_date": st.column_config.DateColumn("Last Sale")
                    },
                    hide_index=True
                )
    
    # ==================== TAB 5: ANALYTICS ====================
            with col5:
                if discrepancy_summary:
                    total_issues = (
                        discrepancy_summary.get('shop_mismatch', 0) +
                        discrepancy_summary.get('serial_mismatch', 0) +
                        discrepancy_summary.get('vehicle_mismatch', 0)
                    )
                else:
                    total_issues = (not_offloaded['shop_mismatch'] + 
                                   not_offloaded['serial_mismatch'] + 
                                   not_offloaded['vehicle_mismatch'])
        shop_perf = get_shop_performance()

        # Shop Performance Heatmap
        if shop_perf is not None and not shop_perf.empty:
            st.markdown("---")
            st.markdown("### 🏪 Shop Performance Heatmap")
            
            # Calculate issue rate
            shop_perf['issue_rate'] = (
                (shop_perf['not_offloaded'] + shop_perf['shop_mismatch'] + shop_perf['serial_mismatch']) / 
                shop_perf['total_items'] * 100
            ).fillna(0).round(2)
            
            # Top 20 shops by total items
            top_shops = shop_perf.nlargest(20, 'total_items')
            
            # Create heatmap data
            heatmap_data = top_shops[['shop_code', 'not_offloaded', 'shop_mismatch', 'serial_mismatch']].set_index('shop_code')
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values.T,
                x=heatmap_data.index,
                y=['Not Offloaded', 'Shop Mismatch', 'Serial Mismatch'],
                colorscale=[[0, MELCOM_GREEN], [0.5, MELCOM_ORANGE], [1, MELCOM_RED]],
                text=heatmap_data.values.T,
                texttemplate='%{text}',
                textfont={"size": 14}
            ))
            
            fig.update_layout(
                title='Shop Discrepancy Heatmap (Top 20 Shops by Volume)',
                height=400,
                xaxis_title='Shop Code',
                yaxis_title='Issue Type'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Issue Distribution
            total_not_off = shop_perf['not_offloaded'].sum()
            total_shop_mis = shop_perf['shop_mismatch'].sum()
            total_serial_mis = shop_perf['serial_mismatch'].sum()

            issue_breakdown = pd.DataFrame({
                'Issue Type': ['Not Offloaded', 'Shop Mismatch', 'Serial Mismatch'],
                'Count': [total_not_off, total_shop_mis, total_serial_mis]
            })

            fig = px.pie(
                issue_breakdown,
                values='Count',
                names='Issue Type',
                title='Overall Issue Breakdown',
                color_discrete_sequence=[MELCOM_RED, MELCOM_ORANGE, MELCOM_PURPLE]
            )
            st.plotly_chart(fig, use_container_width=True)

    # ==================== TAB 5: ANALYTICS ====================
    with tab5:
        st.markdown('<div class="section-header">📈 Advanced Analytics</div>', unsafe_allow_html=True)
        
        st.caption(f"📅 Using Warehouse Date Range: {warehouse_range[0]:%Y/%m/%d} → {warehouse_range[1]:%Y/%m/%d}")
        
        # Get all metrics using warehouse_range for consistency
        wh_metrics = get_warehouse_metrics(warehouse_range, warehouse_range)
        shop_metrics = get_shop_offload_metrics(warehouse_range[0], warehouse_range[1])
        serial_check_metrics = get_serial_check_metrics(warehouse_range[0], warehouse_range[1])
        sales_metrics = get_sales_metrics(warehouse_range[0], warehouse_range[1])
        discrepancy_summary = get_discrepancy_summary()
        
        # Overall Performance Summary
        st.markdown("### 📊 Overall Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if wh_metrics:
                wh_load_rate = (wh_metrics['loaded_unique'] / wh_metrics['received_unique'] * 100) if wh_metrics['received_unique'] > 0 else 0
                card_class = "success" if wh_load_rate > 90 else "warning" if wh_load_rate > 75 else "danger"
                st.markdown(f"""
                <div class="metric-card {card_class}">
                    <div class="metric-label">WH Load Rate</div>
                    <div class="metric-value">{format_percentage(wh_load_rate)}</div>
                    <div class="metric-change">{format_number(wh_metrics['loaded_unique'])} / {format_number(wh_metrics['received_unique'])}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if shop_metrics:
                shop_offload_rate = (shop_metrics['total_offloaded'] / shop_metrics['wh_loaded_total'] * 100) if shop_metrics['wh_loaded_total'] > 0 else 0
                card_class = "success" if shop_offload_rate > 85 else "warning" if shop_offload_rate > 70 else "danger"
                st.markdown(f"""
                <div class="metric-card {card_class}">
                    <div class="metric-label">Shop Offload Rate</div>
                    <div class="metric-value">{format_percentage(shop_offload_rate)}</div>
                    <div class="metric-change">{format_number(shop_metrics['total_offloaded'])} / {format_number(shop_metrics['wh_loaded_total'])}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if serial_check_metrics and not serial_check_metrics.get('missing_serial_check'):
                shop_rows = serial_check_metrics.get('by_shop')
                if isinstance(shop_rows, pd.DataFrame) and not shop_rows.empty:
                    total_yes = shop_rows['total_yes'].sum()
                    total_all = shop_rows['total_serials'].sum()
                else:
                    total_yes = 0
                    total_all = 0
                compliance_rate = (total_yes / total_all * 100) if total_all > 0 else 0
                card_class = "success" if compliance_rate > 90 else "warning" if compliance_rate > 75 else "danger"
                st.markdown(f"""
                <div class="metric-card {card_class}">
                    <div class="metric-label">Serial Compliance Rate</div>
                    <div class="metric-value">{format_percentage(compliance_rate)}</div>
                    <div class="metric-change">{format_number(total_yes)} / {format_number(total_all)} checked</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if sales_metrics:
                total_sold = sales_metrics.get('total_sold', 0)
                unique_sold = sales_metrics.get('unique_sold', 0)
                sale_match_rate = (unique_sold / total_sold * 100) if total_sold > 0 else 0
                card_class = "success" if sale_match_rate > 85 else "warning" if sale_match_rate > 70 else "danger"
                st.markdown(f"""
                <div class="metric-card {card_class}">
                    <div class="metric-label">Sales Match Rate</div>
                    <div class="metric-value">{format_percentage(sale_match_rate)}</div>
                    <div class="metric-change">{format_number(unique_sold)} / {format_number(total_sold)}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # End-to-End Pipeline Funnel
        st.markdown("### 🔄 End-to-End Serial Pipeline Funnel")
        
        if wh_metrics and shop_metrics and sales_metrics:
            funnel_data = pd.DataFrame({
                'Stage': ['WH Received', 'WH Loaded', 'Shop Offloaded', 'Sales Matched'],
                'Count': [
                    wh_metrics['received_unique'],
                    wh_metrics['loaded_unique'],
                    shop_metrics['unique_offloaded'],
                    sales_metrics.get('unique_sold', 0)
                ]
            })
            
            fig = px.funnel(
                funnel_data,
                x='Count',
                y='Stage',
                title='Serial Number Journey Pipeline',
                color='Stage',
                color_discrete_sequence=[MELCOM_BLUE, MELCOM_GREEN, MELCOM_ORANGE, MELCOM_PURPLE]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Pipeline conversion rates
            col1, col2, col3 = st.columns(3)
            
            with col1:
                load_conversion = (wh_metrics['loaded_unique'] / wh_metrics['received_unique'] * 100) if wh_metrics['received_unique'] > 0 else 0
                st.metric("Received → Loaded Conversion", f"{load_conversion:.1f}%")
            
            with col2:
                offload_conversion = (shop_metrics['unique_offloaded'] / wh_metrics['loaded_unique'] * 100) if wh_metrics['loaded_unique'] > 0 else 0
                st.metric("Loaded → Offloaded Conversion", f"{offload_conversion:.1f}%")
            
            with col3:
                unique_sold = sales_metrics.get('unique_sold', 0)
                sale_conversion = (unique_sold / shop_metrics['unique_offloaded'] * 100) if shop_metrics['unique_offloaded'] > 0 else 0
                st.metric("Offloaded → Sold Conversion", f"{sale_conversion:.1f}%")
        
        st.markdown("---")
        
        # Issue Summary
        if discrepancy_summary:
            st.markdown("### ⚠️ Issue Summary")
            
            issue_data = pd.DataFrame({
                'Issue Type': [
                    'Shop Mismatch',
                    'Serial Mismatch',
                    'Vehicle Mismatch',
                    'Not Offloaded',
                    'Sold w/o Offload'
                ],
                'Count': [
                    discrepancy_summary.get('shop_mismatch', 0),
                    discrepancy_summary.get('serial_mismatch', 0),
                    discrepancy_summary.get('vehicle_mismatch', 0),
                    discrepancy_summary.get('no_offloading', 0),
                    discrepancy_summary.get('sold_without_offloading', 0)
                ]
            })
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    issue_data,
                    x='Issue Type',
                    y='Count',
                    title='Discrepancy Breakdown by Type',
                    color='Count',
                    color_continuous_scale=[MELCOM_GREEN, MELCOM_ORANGE, MELCOM_RED]
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                total_issues = issue_data['Count'].sum()
                st.metric("Total Issues", format_number(total_issues))
                
                if wh_metrics:
                    issue_rate = (total_issues / wh_metrics['received_unique'] * 100) if wh_metrics['received_unique'] > 0 else 0
                    st.metric("Issue Rate", f"{issue_rate:.2f}%")
                    
                    health_score = max(0, 100 - issue_rate)
                    health_class = "🟢" if health_score > 90 else "🟡" if health_score > 75 else "🔴"
                    st.metric("System Health Score", f"{health_class} {health_score:.1f}/100")

# ====================== RUN DASHBOARD ======================
if __name__ == "__main__":
    main()
