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
    page_title="SubGroup Analysis - Melcom",
    page_icon="📑",
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
def get_groups():
    """Get list of groups"""
    with get_sales_connection() as conn:
        query = 'SELECT DISTINCT "GROUPS" FROM sales_2025 ORDER BY "GROUPS"'
        df = pd.read_sql(query, conn)
    return ['All Groups'] + df['GROUPS'].tolist()

@st.cache_data(ttl=600)
def get_subgroup_data(start_date, end_date, group=None):
    """Get subgroup data for date range"""
    with get_sales_connection() as conn:
        where_clause = 'WHERE "DATE_INVOICE"::date BETWEEN %s AND %s'
        params_2024 = [start_date.replace(year=2024), end_date.replace(year=2024)]
        params_2025 = [start_date, end_date]
        
        if group and group != 'All Groups':
            where_clause += ' AND "GROUPS" = %s'
            params_2024.append(group)
            params_2025.append(group)
        
        # 2024 data
        query_2024 = f"""
            SELECT 
                "SUB_GROUP" as subgroup_name,
                SUM("QTY") as qty_2024,
                SUM("NET_SALES") as net_sales_2024
            FROM sales_2024
            {where_clause}
            GROUP BY "SUB_GROUP"
        """
        df_2024 = pd.read_sql(query_2024, conn, params=params_2024)
        
        # 2025 data
        query_2025 = f"""
            SELECT 
                "SUB_GROUP" as subgroup_name,
                SUM("QTY") as qty_2025,
                SUM("NET_SALES") as net_sales_2025
            FROM sales_2025
            {where_clause}
            GROUP BY "SUB_GROUP"
        """
        df_2025 = pd.read_sql(query_2025, conn, params=params_2025)
    
    # Merge datasets
    df = pd.merge(df_2024, df_2025, on='subgroup_name', how='outer').fillna(0)
    
    # Calculate metrics
    total_2024 = df['net_sales_2024'].sum()
    total_2025 = df['net_sales_2025'].sum()
    
    df['contribution_2024'] = (df['net_sales_2024'] / total_2024 * 100) if total_2024 > 0 else 0
    df['contribution_2025'] = (df['net_sales_2025'] / total_2025 * 100) if total_2025 > 0 else 0
    df['yoyg'] = ((df['net_sales_2025'] - df['net_sales_2024']) / df['net_sales_2024'].replace(0, 1) * 100)
    df['yoyg'] = df['yoyg'].replace([float('inf'), -float('inf')], 0)
    df['contri_delta'] = df['contribution_2025'] - df['contribution_2024']
    
    return df.sort_values('net_sales_2025', ascending=False)

@st.cache_data(ttl=600)
def get_monthly_subgroup_comparison(start_date, end_date, group=None):
    """Get month-on-month comparison for subgroups within selected date range"""
    with get_sales_connection() as conn:
        if group and group != 'All Groups':
            query = """
                SELECT 
                    EXTRACT(MONTH FROM "DATE_INVOICE"::date) as month,
                    EXTRACT(YEAR FROM "DATE_INVOICE"::date) as year,
                    SUM("NET_SALES") as total_sales
                FROM sales_2024
                WHERE "DATE_INVOICE"::date BETWEEN %s AND %s AND "GROUPS" = %s
                GROUP BY month, year
                
                UNION ALL
                
                SELECT 
                    EXTRACT(MONTH FROM "DATE_INVOICE"::date) as month,
                    EXTRACT(YEAR FROM "DATE_INVOICE"::date) as year,
                    SUM("NET_SALES") as total_sales
                FROM sales_2025
                WHERE "DATE_INVOICE"::date BETWEEN %s AND %s AND "GROUPS" = %s
                GROUP BY month, year
                ORDER BY year, month
            """
            df = pd.read_sql(query, conn, params=(
                start_date.replace(year=2024), end_date.replace(year=2024), group,
                start_date, end_date, group
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
    st.markdown("<div class='dashboard-header'>📑 Sub-Group Analysis</div>", unsafe_allow_html=True)
    
    # Group Filter
    st.sidebar.header("🔍 Filters")
    groups = get_groups()
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
    group_text = selected_group if selected_group != 'All Groups' else 'All Groups'
    st.sidebar.info(f"📊 **Analyzing**: {group_text}\n\n📆 **Period**: {start_date} to {end_date}")
    
    # Data explanation
    if end_date.year == 2025 and end_date.month < 12:
        st.sidebar.warning(f"ℹ️ **Note**: 2025 data includes only {end_date.month} months vs full year 2024. Use same date ranges for fair comparison!")
    
    # Load data
    df = get_subgroup_data(start_date, end_date, selected_group)
    
    if df.empty:
        st.warning("No data available for selected filters")
        return
    
    # Executive Summary KPIs
    st.subheader(f"📈 {group_text} - Sub-Group Performance")
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
        subgroup_count = len(df)
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Sub-Groups</div>
            <div class='kpi-value'>{subgroup_count}</div>
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
    
    df_mtd = get_subgroup_data(mtd_start, mtd_end, selected_group)
    df_ytd = get_subgroup_data(ytd_start, ytd_end, selected_group)
    
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
    
    monthly_df = get_monthly_subgroup_comparison(start_date, end_date, selected_group)
    
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
    
    # Top 15 SubGroups Comparison
    st.subheader("🏆 Top 15 Sub-Groups - Sales Comparison")
    
    top15 = df.head(15)
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Bar(
        name='2024',
        x=top15['subgroup_name'],
        y=top15['net_sales_2024'],
        marker_color=MELCOM_BLUE,
        text=top15['net_sales_2024'].apply(lambda x: format_number(x)),
        textposition='outside'
    ))
    
    fig2.add_trace(go.Bar(
        name='2025',
        x=top15['subgroup_name'],
        y=top15['net_sales_2025'],
        marker_color=MELCOM_RED,
        text=top15['net_sales_2025'].apply(lambda x: format_number(x)),
        textposition='outside'
    ))
    
    fig2.update_layout(
        barmode='group',
        xaxis_title="Sub-Group",
        yaxis_title="Sales (GHS)",
        template='plotly_white',
        height=500,
        xaxis={'tickangle': -45},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Top Growth Leaders")
        growth_leaders = df[df['yoyg'] > 0].nlargest(5, 'yoyg')[['subgroup_name', 'yoyg', 'net_sales_2025']]
        if not growth_leaders.empty:
            for idx, row in growth_leaders.iterrows():
                st.markdown(f"**{row['subgroup_name']}**: {row['yoyg']:.1f}% ({format_number(row['net_sales_2025'])})")
        else:
            st.info("No growth leaders")
    
    with col2:
        st.subheader("⚠️ Declining Sub-Groups")
        decliners = df[df['yoyg'] < 0].nsmallest(5, 'yoyg')[['subgroup_name', 'yoyg', 'net_sales_2025']]
        if not decliners.empty:
            for idx, row in decliners.iterrows():
                st.markdown(f"**{row['subgroup_name']}**: {row['yoyg']:.1f}% ({format_number(row['net_sales_2025'])})")
        else:
            st.info("No declining sub-groups")
    
    # Detailed Data Table
    st.subheader("📋 Detailed Sub-Group Data")
    
    display_df = df.copy()
    display_df['net_sales_2024'] = display_df['net_sales_2024'].apply(lambda x: f"{x:,.2f}")
    display_df['net_sales_2025'] = display_df['net_sales_2025'].apply(lambda x: f"{x:,.2f}")
    display_df['contribution_2024'] = display_df['contribution_2024'].apply(lambda x: f"{x:.2f}%")
    display_df['contribution_2025'] = display_df['contribution_2025'].apply(lambda x: f"{x:.2f}%")
    display_df['yoyg'] = display_df['yoyg'].apply(lambda x: f"{x:.2f}%")
    display_df['contri_delta'] = display_df['contri_delta'].apply(lambda x: f"{x:+.2f}%")
    
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        height=400
    )
    
    # Export to Excel
    if st.button("📥 Export to Excel", type="primary"):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='SubGroup Analysis', index=False)
        
        st.download_button(
            label="Download Excel File",
            data=buffer.getvalue(),
            file_name=f"subgroup_analysis_{selected_group}_{start_date}_to_{end_date}.xlsx",
            mime="application/vnd.ms-excel"
        )

if __name__ == "__main__":
    main()
