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
    page_title="Shop Analysis - Melcom",
    page_icon="🏪",
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
    
    .filter-info {{
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid {MELCOM_RED};
        margin-bottom: 1rem;
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

@st.cache_data(ttl=1800)
def get_groups(dept=None):
    """Get list of groups, optionally filtered by department"""
    with get_sales_connection() as conn:
        if dept and dept != 'All Departments':
            query = 'SELECT DISTINCT "GROUPS" FROM sales_2025 WHERE "DEPT" = %s ORDER BY "GROUPS"'
            df = pd.read_sql(query, conn, params=(dept,))
        else:
            query = 'SELECT DISTINCT "GROUPS" FROM sales_2025 ORDER BY "GROUPS"'
            df = pd.read_sql(query, conn)
    return ['All Groups'] + df['GROUPS'].tolist()

@st.cache_data(ttl=600)
def get_shop_data(start_date, end_date, dept=None, group=None):
    """Get shop-wise data with optional department and group filters"""
    with get_sales_connection() as conn:
        # Build WHERE clause
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
                "SHOP_CODE" as shop_code,
                SUM("QTY") as qty_2024,
                SUM("NET_SALES") as net_sales_2024
            FROM sales_2024
            {where_clause}
            GROUP BY "SHOP_CODE"
        """
        df_2024 = pd.read_sql(query_2024, conn, params=params_2024)
        
        # 2025 data
        query_2025 = f"""
            SELECT 
                "SHOP_CODE" as shop_code,
                SUM("QTY") as qty_2025,
                SUM("NET_SALES") as net_sales_2025
            FROM sales_2025
            {where_clause}
            GROUP BY "SHOP_CODE"
        """
        df_2025 = pd.read_sql(query_2025, conn, params=params_2025)
    
    # Merge datasets
    df = pd.merge(df_2024, df_2025, on='shop_code', how='outer').fillna(0)
    
    # Calculate metrics
    total_2024 = df['net_sales_2024'].sum()
    total_2025 = df['net_sales_2025'].sum()
    
    df['contribution_2024'] = (df['net_sales_2024'] / total_2024 * 100) if total_2024 > 0 else 0
    df['contribution_2025'] = (df['net_sales_2025'] / total_2025 * 100) if total_2025 > 0 else 0
    df['yoyg'] = ((df['net_sales_2025'] - df['net_sales_2024']) / df['net_sales_2024'].replace(0, 1) * 100)
    df['yoyg'] = df['yoyg'].replace([float('inf'), -float('inf')], 0)
    df['contri_delta'] = df['contribution_2025'] - df['contribution_2024']
    df['sales_diff'] = df['net_sales_2025'] - df['net_sales_2024']
    
    return df.sort_values('net_sales_2025', ascending=False)

@st.cache_data(ttl=600)
def get_monthly_shop_comparison(dept=None, group=None):
    """Get month-on-month comparison for shops"""
    with get_sales_connection() as conn:
        filters = []
        if dept and dept != 'All Departments':
            filters.append(f'"DEPT" = \'{dept}\'')
        if group and group != 'All Groups':
            filters.append(f'"GROUPS" = \'{group}\'')
        
        filter_clause = ' AND ' + ' AND '.join(filters) if filters else ''
        
        query = f"""
            SELECT 
                EXTRACT(MONTH FROM "DATE_INVOICE"::date) as month,
                EXTRACT(YEAR FROM "DATE_INVOICE"::date) as year,
                SUM("NET_SALES") as total_sales
            FROM sales_2024
            WHERE EXTRACT(YEAR FROM "DATE_INVOICE"::date) = 2024 {filter_clause}
            GROUP BY month, year
            
            UNION ALL
            
            SELECT 
                EXTRACT(MONTH FROM "DATE_INVOICE"::date) as month,
                EXTRACT(YEAR FROM "DATE_INVOICE"::date) as year,
                SUM("NET_SALES") as total_sales
            FROM sales_2025
            WHERE EXTRACT(YEAR FROM "DATE_INVOICE"::date) = 2025 {filter_clause}
            GROUP BY month, year
            ORDER BY year, month
        """
        df = pd.read_sql(query, conn)
    
    return df

@st.cache_data(ttl=600)
def get_shop_group_breakdown(start_date, end_date, shop_code):
    """Get group-level breakdown for a specific shop"""
    with get_sales_connection() as conn:
        query_2024 = """
            SELECT 
                "GROUPS" as group_name,
                SUM("QTY") as qty_2024,
                SUM("NET_SALES") as net_sales_2024
            FROM sales_2024
            WHERE "SHOP_CODE" = %s AND "DATE_INVOICE"::date BETWEEN %s AND %s
            GROUP BY "GROUPS"
        """
        df_2024 = pd.read_sql(query_2024, conn, params=(shop_code, start_date.replace(year=2024), end_date.replace(year=2024)))
        
        query_2025 = """
            SELECT 
                "GROUPS" as group_name,
                SUM("QTY") as qty_2025,
                SUM("NET_SALES") as net_sales_2025
            FROM sales_2025
            WHERE "SHOP_CODE" = %s AND "DATE_INVOICE"::date BETWEEN %s AND %s
            GROUP BY "GROUPS"
        """
        df_2025 = pd.read_sql(query_2025, conn, params=(shop_code, start_date, end_date))
    
    df = pd.merge(df_2024, df_2025, on='group_name', how='outer').fillna(0)
    
    total_2024 = df['net_sales_2024'].sum()
    total_2025 = df['net_sales_2025'].sum()
    
    df['contribution_2024'] = (df['net_sales_2024'] / total_2024 * 100) if total_2024 > 0 else 0
    df['contribution_2025'] = (df['net_sales_2025'] / total_2025 * 100) if total_2025 > 0 else 0
    df['yoyg'] = ((df['net_sales_2025'] - df['net_sales_2024']) / df['net_sales_2024'].replace(0, 1) * 100)
    df['yoyg'] = df['yoyg'].replace([float('inf'), -float('inf')], 0)
    
    return df.sort_values('net_sales_2025', ascending=False)

# ====================== MAIN APP ======================
def main():
    # Check authentication
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        st.warning("Please login from the Home page first")
        st.stop()
    
    # Header
    st.markdown("<div class='dashboard-header'>🏪 Shop with Group Analysis</div>", unsafe_allow_html=True)
    
    # Filters
    st.sidebar.header("🔍 Filters")
    
    # Department filter
    departments = get_departments()
    selected_dept = st.sidebar.selectbox("Select Department", departments, key='shop_dept')
    
    # Group filter (depends on department)
    groups = get_groups(selected_dept)
    selected_group = st.sidebar.selectbox("Select Group", groups, key='shop_group')
    
    # Date Selection
    st.sidebar.header("📅 Date Selection")
    default_start, default_end = get_current_month_dates()
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        max_value=datetime.today().date(),
        key='shop_start'
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=default_end,
        max_value=datetime.today().date(),
        key='shop_end'
    )
    
    # Quick presets
    col1, col2 = st.sidebar.columns(2)
    if col1.button("This Month", use_container_width=True, key='shop_month'):
        start_date, end_date = get_current_month_dates()
        st.rerun()
    
    if col2.button("This Year", use_container_width=True, key='shop_year'):
        today = datetime.today()
        start_date = datetime(today.year, 1, 1).date()
        end_date = (today - timedelta(days=1)).date()
        st.rerun()
    
    # Show active filters
    dept_text = selected_dept if selected_dept != 'All Departments' else 'All Departments'
    group_text = selected_group if selected_group != 'All Groups' else 'All Groups'
    
    st.markdown(f"""
    <div class='filter-info'>
        <strong>📊 Analyzing:</strong> {dept_text} → {group_text}<br>
        <strong>📆 Period:</strong> {start_date} to {end_date}
    </div>
    """, unsafe_allow_html=True)
    
    # Data explanation
    if end_date.year == 2025 and end_date.month < 12:
        st.sidebar.warning(f"ℹ️ **Note**: 2025 data includes only {end_date.month} months vs full year 2024. Use same date ranges for fair comparison!")
    
    # Load data
    df = get_shop_data(start_date, end_date, selected_dept, selected_group)
    
    if df.empty:
        st.warning("No data available for selected filters")
        return
    
    # Executive Summary KPIs
    st.subheader("📈 Shop Performance Overview")
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    
    total_2024 = df['net_sales_2024'].sum()
    total_2025 = df['net_sales_2025'].sum()
    total_qty_2024 = df['qty_2024'].sum()
    total_qty_2025 = df['qty_2025'].sum()
    total_growth = ((total_2025 - total_2024) / total_2024 * 100) if total_2024 > 0 else 0
    qty_growth = ((total_qty_2025 - total_qty_2024) / total_qty_2024 * 100) if total_qty_2024 > 0 else 0
    
    with kpi1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>2025 Total Sales</div>
            <div class='kpi-value'>{format_number(total_2025)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>2024 Total Sales</div>
            <div class='kpi-value'>{format_number(total_2024)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi3:
        growth_color = MELCOM_BLUE if total_growth >= 0 else MELCOM_RED
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>YoY Growth</div>
            <div class='kpi-value' style='color: {growth_color};'>{total_growth:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi4:
        shop_count = len(df)
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Active Shops</div>
            <div class='kpi-value'>{shop_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi5:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>2025 Total QTY</div>
            <div class='kpi-value'>{int(total_qty_2025):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi6:
        qty_color = MELCOM_BLUE if qty_growth >= 0 else MELCOM_RED
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>QTY Growth</div>
            <div class='kpi-value' style='color: {qty_color};'>{qty_growth:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # MTD & YTD Comparison Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 MTD & YTD Comparison")
    
    # Get MTD and YTD data
    mtd_start, mtd_end = get_mtd_dates()
    ytd_start, ytd_end = get_ytd_dates()
    
    df_mtd = get_shop_data(mtd_start, mtd_end, selected_dept, selected_group)
    df_ytd = get_shop_data(ytd_start, ytd_end, selected_dept, selected_group)
    
    mtd_2024 = df_mtd['net_sales_2024'].sum()
    mtd_2025 = df_mtd['net_sales_2025'].sum()
    mtd_growth = ((mtd_2025 - mtd_2024) / mtd_2024 * 100) if mtd_2024 > 0 else 0
    
    ytd_2024 = df_ytd['net_sales_2024'].sum()
    ytd_2025 = df_ytd['net_sales_2025'].sum()
    ytd_growth = ((ytd_2025 - ytd_2024) / ytd_2024 * 100) if ytd_2024 > 0 else 0
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class='kpi-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
            <div class='kpi-title' style='color: #f0f0f0;'>MTD 2025</div>
            <div class='kpi-value' style='color: white;'>{format_number(mtd_2025)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='kpi-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
            <div class='kpi-title' style='color: #f0f0f0;'>MTD 2024</div>
            <div class='kpi-value' style='color: white;'>{format_number(mtd_2024)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        mtd_color = MELCOM_BLUE if mtd_growth >= 0 else MELCOM_RED
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>MTD Growth</div>
            <div class='kpi-value' style='color: {mtd_color};'>{mtd_growth:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='kpi-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
            <div class='kpi-title' style='color: #f0f0f0;'>YTD 2025</div>
            <div class='kpi-value' style='color: white;'>{format_number(ytd_2025)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class='kpi-card' style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);'>
            <div class='kpi-title' style='color: #f0f0f0;'>YTD 2024</div>
            <div class='kpi-value' style='color: white;'>{format_number(ytd_2024)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        ytd_color = MELCOM_BLUE if ytd_growth >= 0 else MELCOM_RED
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>YTD Growth</div>
            <div class='kpi-value' style='color: {ytd_color};'>{ytd_growth:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Month-on-Month Comparison Chart
    st.subheader("📊 2024 vs 2025 - Month on Month Trend")
    
    monthly_df = get_monthly_shop_comparison(start_date, end_date, selected_dept, selected_group)
    
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
    
    # Top Performing Shops
    st.subheader("🏆 Top 10 Performing Shops - Sales Comparison")
    
    top10 = df.head(10)
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Bar(
        name='2024',
        x=top10['shop_code'],
        y=top10['net_sales_2024'],
        marker_color=MELCOM_BLUE,
        text=top10['net_sales_2024'].apply(lambda x: format_number(x)),
        textposition='outside'
    ))
    
    fig2.add_trace(go.Bar(
        name='2025',
        x=top10['shop_code'],
        y=top10['net_sales_2025'],
        marker_color=MELCOM_RED,
        text=top10['net_sales_2025'].apply(lambda x: format_number(x)),
        textposition='outside'
    ))
    
    fig2.update_layout(
        barmode='group',
        xaxis_title="Shop Code",
        yaxis_title="Sales (GHS)",
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Shop Performance Insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Top Growth Shops")
        growth_leaders = df[df['yoyg'] > 0].nlargest(5, 'yoyg')[['shop_code', 'yoyg', 'net_sales_2025']]
        if not growth_leaders.empty:
            for idx, row in growth_leaders.iterrows():
                st.markdown(f"**{row['shop_code']}**: {row['yoyg']:.1f}% growth ({format_number(row['net_sales_2025'])})")
        else:
            st.info("No shops with positive growth")
    
    with col2:
        st.subheader("⚠️ Declining Shops")
        decliners = df[df['yoyg'] < 0].nsmallest(5, 'yoyg')[['shop_code', 'yoyg', 'net_sales_2025']]
        if not decliners.empty:
            for idx, row in decliners.iterrows():
                st.markdown(f"**{row['shop_code']}**: {row['yoyg']:.1f}% decline ({format_number(row['net_sales_2025'])})")
        else:
            st.info("No declining shops")
    
    # Sales Contribution Pie Chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 2024 Shop Contribution")
        fig_pie_2024 = go.Figure(data=[go.Pie(
            labels=top10['shop_code'],
            values=top10['net_sales_2024'],
            hole=0.4
        )])
        fig_pie_2024.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig_pie_2024, use_container_width=True)
    
    with col2:
        st.subheader("📊 2025 Shop Contribution")
        fig_pie_2025 = go.Figure(data=[go.Pie(
            labels=top10['shop_code'],
            values=top10['net_sales_2025'],
            hole=0.4
        )])
        fig_pie_2025.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig_pie_2025, use_container_width=True)
    
    # Detailed Shop Performance Table
    st.subheader("📋 Detailed Shop Performance Data")
    
    # Format display
    display_df = df.copy()
    display_df['net_sales_2024'] = display_df['net_sales_2024'].apply(lambda x: f"{x:,.2f}")
    display_df['net_sales_2025'] = display_df['net_sales_2025'].apply(lambda x: f"{x:,.2f}")
    display_df['contribution_2024'] = display_df['contribution_2024'].apply(lambda x: f"{x:.2f}%")
    display_df['contribution_2025'] = display_df['contribution_2025'].apply(lambda x: f"{x:.2f}%")
    display_df['yoyg'] = display_df['yoyg'].apply(lambda x: f"{x:.2f}%")
    display_df['contri_delta'] = display_df['contri_delta'].apply(lambda x: f"{x:+.2f}%")
    display_df['sales_diff'] = display_df['sales_diff'].apply(lambda x: f"{x:+,.2f}")
    
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        height=400
    )
    
    # Shop Drill-down
    st.markdown("---")
    st.subheader("🔍 Shop Group Breakdown")
    st.info("Select a shop to see its group-level performance breakdown")
    
    selected_shop = st.selectbox(
        "Select Shop for Detailed Analysis",
        options=df['shop_code'].tolist(),
        key='shop_drill'
    )
    
    if selected_shop:
        shop_breakdown = get_shop_group_breakdown(start_date, end_date, selected_shop)
        
        if not shop_breakdown.empty:
            st.markdown(f"#### 🏪 {selected_shop} - Group Performance")
            
            # Shop breakdown chart
            fig_breakdown = go.Figure()
            
            fig_breakdown.add_trace(go.Bar(
                name='2024',
                x=shop_breakdown['group_name'],
                y=shop_breakdown['net_sales_2024'],
                marker_color=MELCOM_BLUE
            ))
            
            fig_breakdown.add_trace(go.Bar(
                name='2025',
                x=shop_breakdown['group_name'],
                y=shop_breakdown['net_sales_2025'],
                marker_color=MELCOM_RED
            ))
            
            fig_breakdown.update_layout(
                barmode='group',
                xaxis_title="Product Group",
                yaxis_title="Sales (GHS)",
                template='plotly_white',
                height=350,
                xaxis={'tickangle': -45}
            )
            
            st.plotly_chart(fig_breakdown, use_container_width=True)
            
            # Shop breakdown table
            breakdown_display = shop_breakdown.copy()
            breakdown_display['net_sales_2024'] = breakdown_display['net_sales_2024'].apply(lambda x: f"{x:,.2f}")
            breakdown_display['net_sales_2025'] = breakdown_display['net_sales_2025'].apply(lambda x: f"{x:,.2f}")
            breakdown_display['contribution_2024'] = breakdown_display['contribution_2024'].apply(lambda x: f"{x:.2f}%")
            breakdown_display['contribution_2025'] = breakdown_display['contribution_2025'].apply(lambda x: f"{x:.2f}%")
            breakdown_display['yoyg'] = breakdown_display['yoyg'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(breakdown_display, hide_index=True, use_container_width=True)
    
    # Export to Excel
    st.markdown("---")
    if st.button("📥 Export Shop Analysis to Excel", type="primary"):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Shop Performance', index=False)
            if selected_shop and not shop_breakdown.empty:
                shop_breakdown.to_excel(writer, sheet_name=f'Shop {selected_shop} Breakdown', index=False)
        
        filename = f"shop_analysis_{dept_text}_{group_text}_{start_date}_to_{end_date}.xlsx"
        filename = filename.replace(' ', '_').replace('→', '-')
        
        st.download_button(
            label="Download Excel File",
            data=buffer.getvalue(),
            file_name=filename,
            mime="application/vnd.ms-excel"
        )

if __name__ == "__main__":
    main()
