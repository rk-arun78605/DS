# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
from datetime import datetime, timedelta
from contextlib import contextmanager
import plotly.graph_objects as go
import plotly.express as px
import io

# ====================== MELCOM THEME ======================
MELCOM_BLUE = "#002c6d"
MELCOM_RED = "#ed1b24"
MELCOM_LIGHT = "#f3f6fa"
MELCOM_GRAY = "#e8edf2"

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Department Analysis - Melcom",
    page_icon="📊",
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
    
    .insight-box {{
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        border-left: 4px solid {MELCOM_RED};
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
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
    """Get first day of current month to yesterday"""
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
@st.cache_data(ttl=3600)
def get_dept_data(start_date, end_date):
    """Get department data for date range comparing 2024 vs 2025"""
    with get_sales_connection() as conn:
        # 2024 data (same dates but in 2024)
        query_2024 = """
            SELECT 
                "DEPT",
                SUM("QTY") as qty_2024,
                SUM("NET_SALES") as net_sales_2024
            FROM sales_2024
            WHERE "DATE_INVOICE"::date BETWEEN %s AND %s
            GROUP BY "DEPT"
        """
        start_2024 = start_date.replace(year=2024)
        end_2024 = end_date.replace(year=2024)
        df_2024 = pd.read_sql(query_2024, conn, params=(start_2024, end_2024))
        
        # 2025 data
        query_2025 = """
            SELECT 
                "DEPT",
                SUM("QTY") as qty_2025,
                SUM("NET_SALES") as net_sales_2025
            FROM sales_2025
            WHERE "DATE_INVOICE"::date BETWEEN %s AND %s
            GROUP BY "DEPT"
        """
        df_2025 = pd.read_sql(query_2025, conn, params=(start_date, end_date))
    
    # Merge datasets
    df = pd.merge(df_2024, df_2025, on='DEPT', how='outer').fillna(0)
    
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
def get_dept_data_dual(start_date_2024, end_date_2024, start_date_2025, end_date_2025):
    """Get department data for separate date ranges for 2024 and 2025"""
    with get_sales_connection() as conn:
        # 2024 data with custom date range
        query_2024 = """
            SELECT 
                "DEPT",
                SUM("QTY") as qty_2024,
                SUM("NET_SALES") as net_sales_2024
            FROM sales_2024
            WHERE "DATE_INVOICE"::date BETWEEN %s AND %s
            GROUP BY "DEPT"
        """
        df_2024 = pd.read_sql(query_2024, conn, params=(start_date_2024, end_date_2024))
        
        # 2025 data with custom date range
        query_2025 = """
            SELECT 
                "DEPT",
                SUM("QTY") as qty_2025,
                SUM("NET_SALES") as net_sales_2025
            FROM sales_2025
            WHERE "DATE_INVOICE"::date BETWEEN %s AND %s
            GROUP BY "DEPT"
        """
        df_2025 = pd.read_sql(query_2025, conn, params=(start_date_2025, end_date_2025))
    
    # Merge datasets
    df = pd.merge(df_2024, df_2025, on='DEPT', how='outer').fillna(0)
    
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
def get_monthly_dept_comparison(start_date, end_date):
    """Get month-on-month comparison for 2024 vs 2025 within selected date range"""
    with get_sales_connection() as conn:
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
    st.markdown("<div class='dashboard-header'>📊 Department Analysis</div>", unsafe_allow_html=True)
    
    # Date Selection
    st.sidebar.header("📅 Date Selection")
    
    # Get default dates (YTD by default)
    ytd_start_default, ytd_end_default = get_ytd_dates()
    
    # 2024 Date Range
    st.sidebar.markdown("**2024 Period**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date_2024 = st.date_input(
            "Start",
            value=ytd_start_default.replace(year=2024),
            max_value=datetime(2024, 12, 31).date(),
            key="start_2024"
        )
    with col2:
        end_date_2024 = st.date_input(
            "End",
            value=ytd_end_default.replace(year=2024),
            max_value=datetime(2024, 12, 31).date(),
            key="end_2024"
        )
    
    # 2025 Date Range
    st.sidebar.markdown("**2025 Period**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date_2025 = st.date_input(
            "Start",
            value=ytd_start_default,
            max_value=datetime.today().date(),
            key="start_2025"
        )
    with col2:
        end_date_2025 = st.date_input(
            "End",
            value=ytd_end_default,
            max_value=datetime.today().date(),
            key="end_2025"
        )
    
    # Quick presets
    st.sidebar.markdown("**Quick Presets**")
    col1, col2 = st.sidebar.columns(2)
    if col1.button("This Month", use_container_width=True):
        st.rerun()
    
    if col2.button("This Year", use_container_width=True):
        st.rerun()
    
    # Show selected periods
    st.sidebar.info(f"📆 **Comparing:**\n\n**2024**: {start_date_2024} to {end_date_2024}\n\n**2025**: {start_date_2025} to {end_date_2025}")
    
    # Load data with separate date ranges
    df = get_dept_data_dual(start_date_2024, end_date_2024, start_date_2025, end_date_2025)
    
    if df.empty:
        st.warning("No data available for selected period")
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
    df_mtd = get_dept_data_dual(mtd_start.replace(year=2024), mtd_end.replace(year=2024), mtd_start, mtd_end)
    df_ytd = get_dept_data_dual(ytd_start.replace(year=2024), ytd_end.replace(year=2024), ytd_start, ytd_end)
    
    mtd_2024 = df_mtd['net_sales_2024'].sum()
    mtd_2025 = df_mtd['net_sales_2025'].sum()
    mtd_growth = ((mtd_2025 - mtd_2024) / mtd_2024 * 100) if mtd_2024 > 0 else 0
    
    ytd_2024 = df_ytd['net_sales_2024'].sum()
    ytd_2025 = df_ytd['net_sales_2025'].sum()
    ytd_growth = ((ytd_2025 - ytd_2024) / ytd_2024 * 100) if ytd_2024 > 0 else 0
    
    # KPI Cards Row
    st.markdown("### 📊 Performance Metrics")
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
        arrow = '▲' if mtd_growth >= 0 else '▼'
        st.markdown(f"""
        <div class='kpi-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
            <div class='kpi-title' style='color: white;'>MTD Net Sales</div>
            <div style='display: flex; justify-content: space-between; align-items: flex-end; margin: 15px 0 5px 0;'>
                <div style='text-align: left; flex: 1;'>
                    <div style='font-size: 0.65rem; color: #f0f0f0; margin-bottom: 3px;'>2024</div>
                    <div style='font-size: 1.1rem; font-weight: 700; color: white;'>{format_number(mtd_2024)}</div>
                </div>
                <div style='text-align: center; flex: 0.7;'>
                    <div style='font-size: 1rem; font-weight: 700; color: white;'>{abs(mtd_growth):.1f}%</div>
                </div>
                <div style='text-align: right; flex: 1;'>
                    <div style='font-size: 0.65rem; color: #f0f0f0; margin-bottom: 3px;'>2025</div>
                    <div style='font-size: 1.1rem; font-weight: 700; color: white;'>{format_number(mtd_2025)}</div>
                </div>
            </div>
            <div style='text-align: right; margin-top: 5px;'>
                <span style='font-size: 1.2rem; color: white;'>{arrow}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        arrow = '▲' if ytd_growth >= 0 else '▼'
        st.markdown(f"""
        <div class='kpi-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
            <div class='kpi-title' style='color: white;'>YTD Net Sales</div>
            <div style='display: flex; justify-content: space-between; align-items: flex-end; margin: 15px 0 5px 0;'>
                <div style='text-align: left; flex: 1;'>
                    <div style='font-size: 0.65rem; color: #f0f0f0; margin-bottom: 3px;'>2024</div>
                    <div style='font-size: 1.1rem; font-weight: 700; color: white;'>{format_number(ytd_2024)}</div>
                </div>
                <div style='text-align: center; flex: 0.7;'>
                    <div style='font-size: 1rem; font-weight: 700; color: white;'>{abs(ytd_growth):.1f}%</div>
                </div>
                <div style='text-align: right; flex: 1;'>
                    <div style='font-size: 0.65rem; color: #f0f0f0; margin-bottom: 3px;'>2025</div>
                    <div style='font-size: 1.1rem; font-weight: 700; color: white;'>{format_number(ytd_2025)}</div>
                </div>
            </div>
            <div style='text-align: right; margin-top: 5px;'>
                <span style='font-size: 1.2rem; color: white;'>{arrow}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Multi-Chart Dashboard Layout
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 1: Two charts side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Month-on-Month Trend")
        monthly_df = get_monthly_dept_comparison(start_date_2025, end_date_2025)
        
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
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏆 Top 10 Departments - Sales")
        top_10 = df.nlargest(10, 'net_sales_2025')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_10['DEPT'],
            y=top_10['net_sales_2024'],
            name='2024',
            marker_color=MELCOM_BLUE
        ))
        fig.add_trace(go.Bar(
            x=top_10['DEPT'],
            y=top_10['net_sales_2025'],
            name='2025',
            marker_color=MELCOM_RED
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="",
            yaxis_title="Sales"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 2: Two more charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Top 10 Departments - Quantity")
        top_10_qty = df.nlargest(10, 'qty_2025')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_10_qty['DEPT'],
            y=top_10_qty['qty_2024'],
            name='2024',
            marker_color=MELCOM_BLUE
        ))
        fig.add_trace(go.Bar(
            x=top_10_qty['DEPT'],
            y=top_10_qty['qty_2025'],
            name='2025',
            marker_color=MELCOM_RED
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="",
            yaxis_title="Quantity"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Growth Rate Analysis")
        top_growth = df.nlargest(10, 'yoyg')
        
        colors = [MELCOM_BLUE if x >= 0 else MELCOM_RED for x in top_growth['yoyg']]
        
        fig = go.Figure(go.Bar(
            x=top_growth['yoyg'],
            y=top_growth['DEPT'],
            orientation='h',
            marker_color=colors,
            text=top_growth['yoyg'].apply(lambda x: f"{x:.1f}%"),
            textposition='outside'
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Growth %",
            yaxis_title="",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Data Table with Growth Metrics
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Detailed Department Performance")
    
    # Prepare comprehensive table with growth columns
    display_df = df.copy()
    display_df['ns_growth_%'] = display_df['yoyg']
    display_df['qty_growth_%'] = ((display_df['qty_2025'] - display_df['qty_2024']) / display_df['qty_2024'] * 100).replace([float('inf'), -float('inf')], 0).fillna(0)
    
    # Select and reorder columns for better readability
    table_cols = ['DEPT', 'net_sales_2024', 'net_sales_2025', 'ns_growth_%', 'qty_2024', 'qty_2025', 'qty_growth_%', 'contribution_2024', 'contribution_2025', 'contri_delta']
    display_table = display_df[table_cols].copy()
    
    # Configure column display with clear labels
    col_config_table = {
        'DEPT': st.column_config.TextColumn('Department', width=120),
        'net_sales_2024': st.column_config.NumberColumn('Net Sales 2024', format='%.2f', width=120),
        'net_sales_2025': st.column_config.NumberColumn('Net Sales 2025', format='%.2f', width=120),
        'ns_growth_%': st.column_config.NumberColumn('NS Growth %', format='%.2f%%', width=100),
        'qty_2024': st.column_config.NumberColumn('Qty 2024', format='%,d', width=100),
        'qty_2025': st.column_config.NumberColumn('Qty 2025', format='%,d', width=100),
        'qty_growth_%': st.column_config.NumberColumn('Qty Growth %', format='%.2f%%', width=110),
        'contribution_2024': st.column_config.NumberColumn('Contribution 2024', format='%.2f%%', width=120),
        'contribution_2025': st.column_config.NumberColumn('Contribution 2025', format='%.2f%%', width=120),
        'contri_delta': st.column_config.NumberColumn('Contribution Δ', format='%.2f%%', width=110)
    }
    
    st.dataframe(
        display_table,
        column_config=col_config_table,
        hide_index=True,
        use_container_width=True,
        height=400
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Visual Performance Analysis")
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Bar(
        name='2024',
        x=top10['DEPT'],
        y=top10['net_sales_2024'],
        marker_color=MELCOM_BLUE,
        text=top10['net_sales_2024'].apply(lambda x: format_number(x)),
        textposition='outside'
    ))
    
    fig2.add_trace(go.Bar(
        name='2025',
        x=top10['DEPT'],
        y=top10['net_sales_2025'],
        marker_color=MELCOM_RED,
        text=top10['net_sales_2025'].apply(lambda x: format_number(x)),
        textposition='outside'
    ))
    
    fig2.update_layout(
        barmode='group',
        xaxis_title="Department",
        yaxis_title="Sales (GHS)",
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # QTY Comparison Chart
    st.subheader("📦 Top 10 Departments - Quantity Comparison")
    
    fig_qty = go.Figure()
    
    fig_qty.add_trace(go.Bar(
        name='2024 Quantity',
        x=top10['DEPT'],
        y=top10['qty_2024'],
        marker_color='#3b82f6',
        text=top10['qty_2024'].apply(lambda x: f"{int(x):,}"),
        textposition='outside'
    ))
    
    fig_qty.add_trace(go.Bar(
        name='2025 Quantity',
        x=top10['DEPT'],
        y=top10['qty_2025'],
        marker_color='#ef4444',
        text=top10['qty_2025'].apply(lambda x: f"{int(x):,}"),
        textposition='outside'
    ))
    
    fig_qty.update_layout(
        barmode='group',
        xaxis_title="Department",
        yaxis_title="Quantity Sold",
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_qty, use_container_width=True)
    
    # Performance Metrics
    st.subheader("📉 Key Performance Indicators")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        avg_price_2024 = total_2024 / total_qty_2024 if total_qty_2024 > 0 else 0
        avg_price_2025 = total_2025 / total_qty_2025 if total_qty_2025 > 0 else 0
        price_change = ((avg_price_2025 - avg_price_2024) / avg_price_2024 * 100) if avg_price_2024 > 0 else 0
        
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Average Price Per Unit</div>
            <div style='font-size: 1.3rem; color: #666; margin: 0.5rem 0;'>
                2024: {format_number(avg_price_2024)}<br>
                2025: {format_number(avg_price_2025)}
            </div>
            <div style='font-size: 1rem; color: {MELCOM_BLUE if price_change >= 0 else MELCOM_RED}; font-weight: 600;'>
                {'▲' if price_change >= 0 else '▼'} {abs(price_change):.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Total Revenue Difference</div>
            <div style='font-size: 1.3rem; color: {MELCOM_BLUE if (total_2025-total_2024) >= 0 else MELCOM_RED}; margin: 0.5rem 0;'>
                {format_number(abs(total_2025 - total_2024))}
            </div>
            <div style='font-size: 1rem; color: #666; font-weight: 600;'>
                {'▲ Increase' if (total_2025-total_2024) >= 0 else '▼ Decrease'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        qty_diff = int(total_qty_2025 - total_qty_2024)
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Quantity Difference</div>
            <div style='font-size: 1.3rem; color: {MELCOM_BLUE if qty_diff >= 0 else MELCOM_RED}; margin: 0.5rem 0;'>
                {qty_diff:+,} units
            </div>
            <div style='font-size: 1rem; color: #666; font-weight: 600;'>
                {qty_growth:+.1f}% change
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Growth Leaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Top Growth Leaders")
        growth_leaders = df[df['yoyg'] > 0].nlargest(5, 'yoyg')[['DEPT', 'yoyg', 'net_sales_2025']]
        if not growth_leaders.empty:
            for idx, row in growth_leaders.iterrows():
                st.markdown(f"**{row['DEPT']}**: {row['yoyg']:.1f}% growth ({format_number(row['net_sales_2025'])})")
        else:
            st.info("No growth leaders in this period")
    
    with col2:
        st.subheader("⚠️ Declining Departments")
        decliners = df[df['yoyg'] < 0].nsmallest(5, 'yoyg')[['DEPT', 'yoyg', 'net_sales_2025']]
        if not decliners.empty:
            for idx, row in decliners.iterrows():
                st.markdown(f"**{row['DEPT']}**: {row['yoyg']:.1f}% decline ({format_number(row['net_sales_2025'])})")
        else:
            st.info("No declining departments")
    
    # Detailed Data Table
    st.subheader("📋 Complete Department Performance Data")
    
    # Calculate additional metrics
    df['qty_growth'] = ((df['qty_2025'] - df['qty_2024']) / df['qty_2024'].replace(0, 1) * 100).replace([float('inf'), -float('inf')], 0)
    df['sales_diff'] = df['net_sales_2025'] - df['net_sales_2024']
    df['qty_diff'] = df['qty_2025'] - df['qty_2024']
    df['avg_price_2024'] = (df['net_sales_2024'] / df['qty_2024'].replace(0, 1)).replace([float('inf'), -float('inf')], 0)
    df['avg_price_2025'] = (df['net_sales_2025'] / df['qty_2025'].replace(0, 1)).replace([float('inf'), -float('inf')], 0)
    
    # Select and reorder columns
    display_df = df[['DEPT', 'qty_2024', 'qty_2025', 'qty_diff', 'qty_growth',
                     'net_sales_2024', 'net_sales_2025', 'sales_diff', 'yoyg',
                     'avg_price_2024', 'avg_price_2025',
                     'contribution_2024', 'contribution_2025', 'contri_delta']].copy()
    
    display_df.columns = ['Department', 'QTY 2024', 'QTY 2025', 'QTY Diff', 'QTY Growth %',
                          'Sales 2024', 'Sales 2025', 'Sales Diff', 'Sales Growth %',
                          'Avg Price 2024', 'Avg Price 2025',
                          'Contrib % 2024', 'Contrib % 2025', 'Contrib Δ']
    
    st.dataframe(
        display_df.style.format({
            'QTY 2024': '{:,.0f}',
            'QTY 2025': '{:,.0f}',
            'QTY Diff': '{:+,.0f}',
            'QTY Growth %': '{:+.1f}%',
            'Sales 2024': 'GHS {:,.2f}',
            'Sales 2025': 'GHS {:,.2f}',
            'Sales Diff': 'GHS {:+,.2f}',
            'Sales Growth %': '{:+.1f}%',
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
            df.to_excel(writer, sheet_name='Department Analysis', index=False)
        
        st.download_button(
            label="Download Excel File",
            data=buffer.getvalue(),
            file_name=f"department_analysis_{start_date}_to_{end_date}.xlsx",
            mime="application/vnd.ms-excel"
        )

if __name__ == "__main__":
    main()
