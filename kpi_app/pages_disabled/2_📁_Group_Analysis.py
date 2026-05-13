# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
from datetime import datetime, timedelta
from contextlib import contextmanager
import plotly.graph_objects as go
import io

# ====================== MELCOM THEME ======================
MELCOM_BLUE = "#002c6d"
MELCOM_RED = "#ed1b24"
MELCOM_LIGHT = "#f3f6fa"

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Group Analysis - Melcom",
    page_icon="📁",
    layout="wide"
)

# ====================== STYLING ======================
st.markdown(f"""
<style>
    .dashboard-header {{
        background: linear-gradient(135deg, {MELCOM_BLUE} 0%, #003d8f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,44,109,0.3);
    }}
    
    .kpi-card {{
        background: linear-gradient(135deg, white 0%, {MELCOM_LIGHT} 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 5px solid {MELCOM_BLUE};
    }}
    
    .kpi-title {{
        font-size: 0.9rem;
        color: #666;
        font-weight: 600;
        text-transform: uppercase;
    }}
    
    .kpi-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {MELCOM_BLUE};
    }}
</style>
""", unsafe_allow_html=True)

# ====================== DATABASE CONNECTION ======================
@st.cache_resource
def get_sales_pool():
    return psycopg2.pool.SimpleConnectionPool(
        minconn=1, maxconn=10,
        host="localhost", user="postgres", password="hello",
        database="salesdata", port=3307
    )

@contextmanager
def get_sales_connection():
    pool = get_sales_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)

# ====================== HELPER FUNCTIONS ======================
def format_number(value):
    try:
        value = float(value)
        if abs(value) >= 1_000_000:
            return f"GHS {value/1_000_000:.2f}M"
        elif abs(value) >= 1_000:
            return f"GHS {value/1_000:.2f}K"
        else:
            return f"GHS {value:.2f}"
    except:
        return "GHS 0"

def get_current_month_dates():
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    first_day = yesterday.replace(day=1)
    return first_day.date(), yesterday.date()

def get_mtd_dates():
    """Get Month-to-Date: First day of current month to yesterday"""
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    first_day = yesterday.replace(day=1)
    return first_day.date(), yesterday.date()

def get_ytd_dates():
    """Get Year-to-Date: Jan 1 to yesterday"""
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    first_day = datetime(yesterday.year, 1, 1)
    return first_day.date(), yesterday.date()

# ====================== DATA LOADING FUNCTIONS ======================
@st.cache_data(ttl=1800)
def get_departments():
    """Get list of departments"""
    with get_sales_connection() as conn:
        query = 'SELECT DISTINCT "DEPT" FROM sales_2025 ORDER BY "DEPT"'
        df = pd.read_sql(query, conn)
    return ['All Departments'] + df['DEPT'].tolist()

@st.cache_data(ttl=3600)
def get_group_data(start_date, end_date, dept=None, group=None):
    """Get group data for date range"""
    with get_sales_connection() as conn:
        # Base query
        where_clause = 'WHERE "DATE_INVOICE"::date BETWEEN %s AND %s'
        params_2024 = [start_date.replace(year=2024), end_date.replace(year=2024)]
        params_2025 = [start_date, end_date]
        
        if dept and dept != 'All Departments':
            where_clause += ' AND "DEPT" = %s'
            params_2024.append(dept)
            params_2025.append(dept)
        
        if group and group != 'All Groups':
            where_clause += ' AND "GROUPS" = %s'
            params_2024.append(group)
            params_2025.append(group)
        
        # 2024 data
        query_2024 = f"""
            SELECT 
                "GROUPS" as group_name,
                SUM("QTY") as qty_2024,
                SUM("NET_SALES") as net_sales_2024
            FROM sales_2024
            {where_clause}
            GROUP BY "GROUPS"
        """
        df_2024 = pd.read_sql(query_2024, conn, params=params_2024)
        
        # 2025 data
        query_2025 = f"""
            SELECT 
                "GROUPS" as group_name,
                SUM("QTY") as qty_2025,
                SUM("NET_SALES") as net_sales_2025
            FROM sales_2025
            {where_clause}
            GROUP BY "GROUPS"
        """
        df_2025 = pd.read_sql(query_2025, conn, params=params_2025)
    
    # Merge datasets
    df = pd.merge(df_2024, df_2025, on='group_name', how='outer').fillna(0)
    
    # Calculate metrics
    total_2024 = df['net_sales_2024'].sum()
    total_2025 = df['net_sales_2025'].sum()
    
    df['contribution_2024'] = (df['net_sales_2024'] / total_2024 * 100) if total_2024 > 0 else 0
    df['contribution_2025'] = (df['net_sales_2025'] / total_2025 * 100) if total_2025 > 0 else 0
    df['yoyg'] = ((df['net_sales_2025'] - df['net_sales_2024']) / df['net_sales_2024'].replace(0, 1) * 100)
    df['yoyg'] = df['yoyg'].replace([float('inf'), -float('inf')], 0)
    df['contri_delta'] = df['contribution_2025'] - df['contribution_2024']
    
    return df.sort_values('net_sales_2025', ascending=False)

@st.cache_data(ttl=3600)
def get_group_data_dual(start_date_2024, end_date_2024, start_date_2025, end_date_2025, dept=None, group=None):
    """Get group data for separate date ranges for 2024 and 2025"""
    with get_sales_connection() as conn:
        # Base where clause builder
        def build_where_clause():
            where_parts = []
            params = []
            if dept and dept != 'All Departments':
                where_parts.append('"DEPT" = %s')
                params.append(dept)
            if group and group != 'All Groups':
                where_parts.append('"GROUPS" = %s')
                params.append(group)
            return (' AND ' + ' AND '.join(where_parts)) if where_parts else '', params
        
        where_extra, extra_params = build_where_clause()
        
        # 2024 data with custom date range
        query_2024 = f"""
            SELECT 
                "GROUPS" as group_name,
                SUM("QTY") as qty_2024,
                SUM("NET_SALES") as net_sales_2024
            FROM sales_2024
            WHERE "DATE_INVOICE"::date BETWEEN %s AND %s{where_extra}
            GROUP BY "GROUPS"
        """
        params_2024 = [start_date_2024, end_date_2024] + extra_params
        df_2024 = pd.read_sql(query_2024, conn, params=params_2024)
        
        # 2025 data with custom date range
        query_2025 = f"""
            SELECT 
                "GROUPS" as group_name,
                SUM("QTY") as qty_2025,
                SUM("NET_SALES") as net_sales_2025
            FROM sales_2025
            WHERE "DATE_INVOICE"::date BETWEEN %s AND %s{where_extra}
            GROUP BY "GROUPS"
        """
        params_2025 = [start_date_2025, end_date_2025] + extra_params
        df_2025 = pd.read_sql(query_2025, conn, params=params_2025)
    
    # Merge datasets
    df = pd.merge(df_2024, df_2025, on='group_name', how='outer').fillna(0)
    
    # Calculate metrics
    total_2024 = df['net_sales_2024'].sum()
    total_2025 = df['net_sales_2025'].sum()
    
    df['contribution_2024'] = (df['net_sales_2024'] / total_2024 * 100) if total_2024 > 0 else 0
    df['contribution_2025'] = (df['net_sales_2025'] / total_2025 * 100) if total_2025 > 0 else 0
    df['yoyg'] = ((df['net_sales_2025'] - df['net_sales_2024']) / df['net_sales_2024'].replace(0, 1) * 100)
    df['yoyg'] = df['yoyg'].replace([float('inf'), -float('inf')], 0)
    df['contri_delta'] = df['contribution_2025'] - df['contribution_2024']
    
    return df.sort_values('net_sales_2025', ascending=False)

@st.cache_data(ttl=3600)
def get_monthly_group_comparison(start_date, end_date, dept=None, group=None):
    """Get month-on-month comparison for groups within selected date range"""
    with get_sales_connection() as conn:
        where_conditions = []
        params = []
        
        if dept and dept != 'All Departments':
            where_conditions.append('"DEPT" = %s')
            params.append(dept)
        
        if group and group != 'All Groups':
            where_conditions.append('"GROUPS" = %s')
            params.append(group)
        
        where_clause = ' AND '.join(where_conditions) if where_conditions else '1=1'
        
        if where_conditions:
            query = f"""
                SELECT 
                    EXTRACT(MONTH FROM "DATE_INVOICE"::date) as month,
                    EXTRACT(YEAR FROM "DATE_INVOICE"::date) as year,
                    SUM("NET_SALES") as total_sales
                FROM sales_2024
                WHERE "DATE_INVOICE"::date BETWEEN %s AND %s AND {where_clause}
                GROUP BY month, year
                
                UNION ALL
                
                SELECT 
                    EXTRACT(MONTH FROM "DATE_INVOICE"::date) as month,
                    EXTRACT(YEAR FROM "DATE_INVOICE"::date) as year,
                    SUM("NET_SALES") as total_sales
                FROM sales_2025
                WHERE "DATE_INVOICE"::date BETWEEN %s AND %s AND {where_clause}
                GROUP BY month, year
                ORDER BY year, month
            """
            df = pd.read_sql(query, conn, params=(
                start_date.replace(year=2024), end_date.replace(year=2024),
                *params,
                start_date, end_date,
                *params
            ))
        else:
            query = """
                SELECT 
                    EXTRACT(MONTH FROM "DATE_INVOICE"::date) as month,
                    EXTRACT(YEAR FROM "DATE_INVOICE"::date) as year,
                    SUM("NET_SALES") as total_sales
                FROM sales_2024
                WHERE "DATE_INVOICE"::date BETWEEN %s AND %s
                GROUP BY month, year
                
                UNION ALL
                
                SELECT 
                    EXTRACT(MONTH FROM "DATE_INVOICE"::date) as month,
                    EXTRACT(YEAR FROM "DATE_INVOICE"::date) as year,
                    SUM("NET_SALES") as total_sales
                FROM sales_2025
                WHERE "DATE_INVOICE"::date BETWEEN %s AND %s
                GROUP BY month, year
                ORDER BY year, month
            """
            df = pd.read_sql(query, conn, params=(
                start_date.replace(year=2024), end_date.replace(year=2024),
                start_date, end_date
            ))
    
    return df

# ====================== MAIN APP ======================
def main():
    # Check authentication
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        st.warning("Please login from the Home page first")
        st.stop()
    
    # Header
    st.markdown("<div class='dashboard-header'>📁 Group Analysis</div>", unsafe_allow_html=True)
    
    # Department Filter
    st.sidebar.header("🔍 Filters")
    departments = get_departments()
    selected_dept = st.sidebar.selectbox("Select Department", departments)
    
    # Group Filter - load groups based on selected department
    @st.cache_data(ttl=1800)
    def get_groups(dept=None):
        """Get list of groups for selected department"""
        with get_sales_connection() as conn:
            if dept and dept != 'All Departments':
                query = 'SELECT DISTINCT "GROUPS" FROM sales_2025 WHERE "DEPT" = %s ORDER BY "GROUPS"'
                df = pd.read_sql(query, conn, params=(dept,))
            else:
                query = 'SELECT DISTINCT "GROUPS" FROM sales_2025 ORDER BY "GROUPS"'
                df = pd.read_sql(query, conn)
        return ['All Groups'] + df['GROUPS'].tolist()
    
    groups = get_groups(selected_dept)
    selected_group = st.sidebar.selectbox("Select Group", groups)
    
    # Date Selection
    st.sidebar.header("📅 Date Selection")
    default_start, default_end = get_current_month_dates()
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        max_value=datetime.today().date()
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=default_end,
        max_value=datetime.today().date()
    )
    
    # Quick presets
    col1, col2 = st.sidebar.columns(2)
    if col1.button("This Month", use_container_width=True):
        start_date, end_date = get_current_month_dates()
        st.rerun()
    
    if col2.button("This Year", use_container_width=True):
        today = datetime.today()
        start_date = datetime(today.year, 1, 1).date()
        end_date = (today - timedelta(days=1)).date()
        st.rerun()
    
    # Show context
    dept_text = selected_dept if selected_dept != 'All Departments' else 'All Departments'
    group_text = selected_group if selected_group != 'All Groups' else 'All Groups'
    st.sidebar.info(f"📊 **Department**: {dept_text}\n\n📁 **Group**: {group_text}\n\n📆 **Period**: {start_date} to {end_date}")
    
    # Data explanation
    if end_date.year == 2025 and end_date.month < 12:
        st.sidebar.warning(f"ℹ️ **Note**: 2025 data includes only {end_date.month} months vs full year 2024. Use same date ranges for fair comparison!")
    
    # Load data
    df = get_group_data(start_date, end_date, selected_dept, selected_group)
    
    if df.empty:
        st.warning("No data available for selected filters")
        return
    
    # Calculate metrics
    total_2024 = df['net_sales_2024'].sum()
    total_2025 = df['net_sales_2025'].sum()
    total_qty_2024 = df['qty_2024'].sum()
    total_qty_2025 = df['qty_2025'].sum()
    sales_growth = ((total_2025 - total_2024) / total_2024 * 100) if total_2024 > 0 else 0
    qty_growth = ((total_qty_2025 - total_qty_2024) / total_qty_2024 * 100) if total_qty_2024 > 0 else 0
    
    # Get MTD and YTD data with separate date ranges
    mtd_start, mtd_end = get_mtd_dates()
    ytd_start, ytd_end = get_ytd_dates()
    df_mtd = get_group_data_dual(mtd_start.replace(year=2024), mtd_end.replace(year=2024), mtd_start, mtd_end, selected_dept, selected_group)
    df_ytd = get_group_data_dual(ytd_start.replace(year=2024), ytd_end.replace(year=2024), ytd_start, ytd_end, selected_dept, selected_group)
    
    mtd_2024 = df_mtd['net_sales_2024'].sum()
    mtd_2025 = df_mtd['net_sales_2025'].sum()
    mtd_growth = ((mtd_2025 - mtd_2024) / mtd_2024 * 100) if mtd_2024 > 0 else 0
    
    ytd_2024 = df_ytd['net_sales_2024'].sum()
    ytd_2025 = df_ytd['net_sales_2025'].sum()
    ytd_growth = ((ytd_2025 - ytd_2024) / ytd_2024 * 100) if ytd_2024 > 0 else 0
    
    # KPI Cards Row
    filter_display = f"{dept_text} → {group_text}" if group_text != 'All Groups' else dept_text
    st.markdown(f"### 📊 {filter_display}")
    col1, col2, col3, col4, col5 = st.columns([1.2, 1.2, 1.2, 1.2, 0.2])
    
    with col1:
        arrow = '▲' if sales_growth >= 0 else '▼'
        color = '#2ecc71' if sales_growth >= 0 else '#e74c3c'
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>NS Growth</div>
            <div style='display: flex; justify-content: space-between; align-items: flex-end; margin: 15px 0 5px 0;'>
                <div style='text-align: left; flex: 1;'>
                    <div style='font-size: 0.65rem; color: #888; margin-bottom: 3px;'>2024</div>
                    <div style='font-size: 1.1rem; font-weight: 700; color: #2c3e50;'>{format_number(total_2024)}</div>
                </div>
                <div style='text-align: center; flex: 0.7;'>
                    <div style='font-size: 1rem; font-weight: 700; color: {color};'>{abs(sales_growth):.1f}%</div>
                </div>
                <div style='text-align: right; flex: 1;'>
                    <div style='font-size: 0.65rem; color: #888; margin-bottom: 3px;'>2025</div>
                    <div style='font-size: 1.1rem; font-weight: 700; color: #2c3e50;'>{format_number(total_2025)}</div>
                </div>
            </div>
            <div style='text-align: right; margin-top: 5px;'>
                <span style='font-size: 1.2rem; color: {color};'>{arrow}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        arrow = '▲' if qty_growth >= 0 else '▼'
        color = '#2ecc71' if qty_growth >= 0 else '#e74c3c'
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>QTY Growth</div>
            <div style='display: flex; justify-content: space-between; align-items: flex-end; margin: 15px 0 5px 0;'>
                <div style='text-align: left; flex: 1;'>
                    <div style='font-size: 0.65rem; color: #888; margin-bottom: 3px;'>2024</div>
                    <div style='font-size: 1.1rem; font-weight: 700; color: #2c3e50;'>{int(total_qty_2024):,}</div>
                </div>
                <div style='text-align: center; flex: 0.7;'>
                    <div style='font-size: 1rem; font-weight: 700; color: {color};'>{abs(qty_growth):.1f}%</div>
                </div>
                <div style='text-align: right; flex: 1;'>
                    <div style='font-size: 0.65rem; color: #888; margin-bottom: 3px;'>2025</div>
                    <div style='font-size: 1.1rem; font-weight: 700; color: #2c3e50;'>{int(total_qty_2025):,}</div>
                </div>
            </div>
            <div style='text-align: right; margin-top: 5px;'>
                <span style='font-size: 1.2rem; color: {color};'>{arrow}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        mtd_2024_val = df_mtd['net_sales_2024'].sum()
        mtd_2025_val = df_mtd['net_sales_2025'].sum()
        mtd_growth_val = ((mtd_2025_val - mtd_2024_val) / mtd_2024_val * 100) if mtd_2024_val > 0 else 0
        arrow = '▲' if mtd_growth_val >= 0 else '▼'
        st.markdown(f"""
        <div class='kpi-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
            <div class='kpi-title' style='color: white;'>MTD Net Sales</div>
            <div style='display: flex; justify-content: space-between; align-items: flex-end; margin: 15px 0 5px 0;'>
                <div style='text-align: left; flex: 1;'>
                    <div style='font-size: 0.65rem; color: #f0f0f0; margin-bottom: 3px;'>2024</div>
                    <div style='font-size: 1.1rem; font-weight: 700; color: white;'>{format_number(mtd_2024_val)}</div>
                </div>
                <div style='text-align: center; flex: 0.7;'>
                    <div style='font-size: 1rem; font-weight: 700; color: white;'>{abs(mtd_growth_val):.1f}%</div>
                </div>
                <div style='text-align: right; flex: 1;'>
                    <div style='font-size: 0.65rem; color: #f0f0f0; margin-bottom: 3px;'>2025</div>
                    <div style='font-size: 1.1rem; font-weight: 700; color: white;'>{format_number(mtd_2025_val)}</div>
                </div>
            </div>
            <div style='text-align: right; margin-top: 5px;'>
                <span style='font-size: 1.2rem; color: white;'>{arrow}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ytd_2024_val = df_ytd['net_sales_2024'].sum()
        ytd_2025_val = df_ytd['net_sales_2025'].sum()
        ytd_growth_val = ((ytd_2025_val - ytd_2024_val) / ytd_2024_val * 100) if ytd_2024_val > 0 else 0
        arrow = '▲' if ytd_growth_val >= 0 else '▼'
        st.markdown(f"""
        <div class='kpi-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
            <div class='kpi-title' style='color: white;'>YTD Net Sales</div>
            <div style='display: flex; justify-content: space-between; align-items: flex-end; margin: 15px 0 5px 0;'>
                <div style='text-align: left; flex: 1;'>
                    <div style='font-size: 0.65rem; color: #f0f0f0; margin-bottom: 3px;'>2024</div>
                    <div style='font-size: 1.1rem; font-weight: 700; color: white;'>{format_number(ytd_2024_val)}</div>
                </div>
                <div style='text-align: center; flex: 0.7;'>
                    <div style='font-size: 1rem; font-weight: 700; color: white;'>{abs(ytd_growth_val):.1f}%</div>
                </div>
                <div style='text-align: right; flex: 1;'>
                    <div style='font-size: 0.65rem; color: #f0f0f0; margin-bottom: 3px;'>2025</div>
                    <div style='font-size: 1.1rem; font-weight: 700; color: white;'>{format_number(ytd_2025_val)}</div>
                </div>
            </div>
            <div style='text-align: right; margin-top: 5px;'>
                <span style='font-size: 1.2rem; color: white;'>{arrow}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 2024 vs 2025 - Month on Month Trend")
    
    monthly_df = get_monthly_group_comparison(start_date, end_date, selected_dept, selected_group)
    
    fig = go.Figure()
    
    df_2024 = monthly_df[monthly_df['year'] == 2024]
    fig.add_trace(go.Scatter(
        x=df_2024['month'],
        y=df_2024['total_sales'],
        name='2024',
        line=dict(color=MELCOM_BLUE, width=3),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    df_2025 = monthly_df[monthly_df['year'] == 2025]
    fig.add_trace(go.Scatter(
        x=df_2025['month'],
        y=df_2025['total_sales'],
        name='2025',
        line=dict(color=MELCOM_RED, width=3),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Total Sales (GHS)",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=month_names),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 10 Groups Comparison
    st.subheader("🏆 Top 10 Groups - Sales Comparison")
    
    top10 = df.head(10)
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Bar(
        name='2024',
        x=top10['group_name'],
        y=top10['net_sales_2024'],
        marker_color=MELCOM_BLUE,
        text=top10['net_sales_2024'].apply(lambda x: format_number(x)),
        textposition='outside'
    ))
    
    fig2.add_trace(go.Bar(
        name='2025',
        x=top10['group_name'],
        y=top10['net_sales_2025'],
        marker_color=MELCOM_RED,
        text=top10['net_sales_2025'].apply(lambda x: format_number(x)),
        textposition='outside'
    ))
    
    fig2.update_layout(
        barmode='group',
        xaxis_title="Group",
        yaxis_title="Sales (GHS)",
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # QTY Comparison Chart
    st.subheader("📦 Quantity Sold - 2024 vs 2025 Comparison")
    
    fig_qty = go.Figure()
    
    fig_qty.add_trace(go.Bar(
        name='2024 Quantity',
        x=top10['group_name'],
        y=top10['qty_2024'],
        marker_color='#3b82f6',
        text=top10['qty_2024'].apply(lambda x: f"{int(x):,}"),
        textposition='outside'
    ))
    
    fig_qty.add_trace(go.Bar(
        name='2025 Quantity',
        x=top10['group_name'],
        y=top10['qty_2025'],
        marker_color='#ef4444',
        text=top10['qty_2025'].apply(lambda x: f"{int(x):,}"),
        textposition='outside'
    ))
    
    fig_qty.update_layout(
        barmode='group',
        xaxis_title="Group",
        yaxis_title="Quantity Sold",
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_qty, use_container_width=True)
    
    # Sales vs QTY Analysis
    st.subheader("💰 Sales & Quantity Analysis")
    col_analysis1, col_analysis2 = st.columns(2)
    
    with col_analysis1:
        # Average price per unit 2024 vs 2025
        avg_price_2024 = total_2024 / total_qty_2024 if total_qty_2024 > 0 else 0
        avg_price_2025 = total_2025 / total_qty_2025 if total_qty_2025 > 0 else 0
        price_change = ((avg_price_2025 - avg_price_2024) / avg_price_2024 * 100) if avg_price_2024 > 0 else 0
        
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Average Price Per Unit</div>
            <div style='font-size: 1.2rem; color: #666; margin: 0.5rem 0;'>
                2024: {format_number(avg_price_2024)}<br>
                2025: {format_number(avg_price_2025)}
            </div>
            <div style='font-size: 1rem; color: {MELCOM_BLUE if price_change >= 0 else MELCOM_RED}; font-weight: 600;'>
                {'▲' if price_change >= 0 else '▼'} {abs(price_change):.1f}% change
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_analysis2:
        # Sales per group
        avg_sales_per_group_2024 = total_2024 / len(df) if len(df) > 0 else 0
        avg_sales_per_group_2025 = total_2025 / len(df) if len(df) > 0 else 0
        
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Average Sales Per Group</div>
            <div style='font-size: 1.2rem; color: #666; margin: 0.5rem 0;'>
                2024: {format_number(avg_sales_per_group_2024)}<br>
                2025: {format_number(avg_sales_per_group_2025)}
            </div>
            <div style='font-size: 1rem; color: {MELCOM_BLUE}; font-weight: 600;'>
                {len(df)} active groups
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Top Growth Leaders")
        growth_leaders = df[df['yoyg'] > 0].nlargest(5, 'yoyg')[['group_name', 'yoyg', 'net_sales_2025']]
        if not growth_leaders.empty:
            for idx, row in growth_leaders.iterrows():
                st.markdown(f"**{row['group_name']}**: {row['yoyg']:.1f}% ({format_number(row['net_sales_2025'])})")
        else:
            st.info("No growth leaders")
    
    with col2:
        st.subheader("⚠️ Declining Groups")
        decliners = df[df['yoyg'] < 0].nsmallest(5, 'yoyg')[['group_name', 'yoyg', 'net_sales_2025']]
        if not decliners.empty:
            for idx, row in decliners.iterrows():
                st.markdown(f"**{row['group_name']}**: {row['yoyg']:.1f}% ({format_number(row['net_sales_2025'])})")
        else:
            st.info("No declining groups")
    
    # Detailed Data Table
    st.subheader("📋 Complete Group Performance Data")
    
    # Add calculated columns
    df['qty_growth'] = ((df['qty_2025'] - df['qty_2024']) / df['qty_2024'].replace(0, 1) * 100).replace([float('inf'), -float('inf')], 0)
    df['sales_diff'] = df['net_sales_2025'] - df['net_sales_2024']
    df['avg_price_2024'] = (df['net_sales_2024'] / df['qty_2024'].replace(0, 1)).replace([float('inf'), -float('inf')], 0)
    df['avg_price_2025'] = (df['net_sales_2025'] / df['qty_2025'].replace(0, 1)).replace([float('inf'), -float('inf')], 0)
    
    # Format for display
    display_df = df[['group_name', 'qty_2024', 'qty_2025', 'qty_growth', 
                     'net_sales_2024', 'net_sales_2025', 'yoyg', 'sales_diff',
                     'avg_price_2024', 'avg_price_2025',
                     'contribution_2024', 'contribution_2025', 'contri_delta']].copy()
    
    display_df.columns = ['Group', 'QTY 2024', 'QTY 2025', 'QTY Growth %', 
                          'Sales 2024', 'Sales 2025', 'Sales Growth %', 'Sales Diff',
                          'Avg Price 2024', 'Avg Price 2025',
                          'Contrib % 2024', 'Contrib % 2025', 'Contrib Δ']
    
    st.dataframe(
        display_df.style.format({
            'QTY 2024': '{:,.0f}',
            'QTY 2025': '{:,.0f}',
            'QTY Growth %': '{:+.1f}%',
            'Sales 2024': 'GHS {:,.2f}',
            'Sales 2025': 'GHS {:,.2f}',
            'Sales Growth %': '{:+.1f}%',
            'Sales Diff': 'GHS {:+,.2f}',
            'Avg Price 2024': 'GHS {:.2f}',
            'Avg Price 2025': 'GHS {:.2f}',
            'Contrib % 2024': '{:.2f}%',
            'Contrib % 2025': '{:.2f}%',
            'Contrib Δ': '{:+.2f}%'
        }).background_gradient(subset=['Sales Growth %', 'QTY Growth %'], cmap='RdYlGn', vmin=-50, vmax=50),
        hide_index=True,
        use_container_width=True,
        height=500
    )
    
    # Export to Excel
    if st.button("📥 Export to Excel", type="primary"):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Group Analysis', index=False)
        
        st.download_button(
            label="Download Excel File",
            data=buffer.getvalue(),
            file_name=f"group_analysis_{selected_dept}_{start_date}_to_{end_date}.xlsx",
            mime="application/vnd.ms-excel"
        )

if __name__ == "__main__":
    main()
