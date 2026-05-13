"""
KPI Dashboard - Insights & Story-Driven Version
A narrative-based dashboard that tells the business story and provides actionable insights
"""

import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
from contextlib import contextmanager
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Melcom KPI Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CONSTANTS ======================
MELCOM_BLUE = '#1e3a8a'
MELCOM_RED = '#dc2626'
MELCOM_GREEN = '#16a34a'
MELCOM_ORANGE = '#ea580c'

DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'salesdata'
}

# ====================== STYLING ======================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a8a 0%, #dc2626 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .insight-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #1e3a8a;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .dept-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    
    .dept-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .performance-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .badge-positive {
        background: #d1fae5;
        color: #065f46;
    }
    
    .badge-negative {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .badge-neutral {
        background: #e0e7ff;
        color: #3730a3;
    }
    
    .metric-big {
        font-size: 3rem;
        font-weight: 800;
        color: #1e3a8a;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .story-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 1.5rem 0;
    }
    
    .recommendation {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# ====================== DATABASE CONNECTION ======================
@st.cache_resource
def get_connection_pool():
    return psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        dbname=DB_CONFIG['database']
    )

@contextmanager
def get_db_connection():
    pool = get_connection_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)

# ====================== DATA LOADING FUNCTIONS ======================
@st.cache_data(ttl=600)
def get_business_snapshot():
    """Get overall business performance snapshot"""
    yesterday = (datetime.today() - timedelta(days=1)).date()
    mtd_start = yesterday.replace(day=1)
    ytd_start = datetime(yesterday.year, 1, 1).date()
        
    # Get same period last year
    mtd_start_ly = mtd_start.replace(year=mtd_start.year - 1)
    mtd_end_ly = yesterday.replace(year=yesterday.year - 1)
    ytd_start_ly = ytd_start.replace(year=ytd_start.year - 1)
    ytd_end_ly = yesterday.replace(year=yesterday.year - 1)
    
    # Determine which tables to query (2026 uses lowercase, 2024/2025 use uppercase)
    current_year = yesterday.year
    last_year = current_year - 1
    
    if current_year >= 2026:
        # sales_2026+ uses lowercase columns
        current_query = f"""
        SELECT 
            'MTD' as period,
            SUM(net_sales) as sales,
            SUM(qty) as qty
        FROM sales_{current_year}
        WHERE date_invoice BETWEEN %s AND %s
        UNION ALL
        SELECT 
            'YTD' as period,
            SUM(net_sales) as sales,
            SUM(qty) as qty
        FROM sales_{current_year}
        WHERE date_invoice BETWEEN %s AND %s
        """
    else:
        # sales_2024/2025 use uppercase columns
        current_query = f"""
        SELECT 
            'MTD' as period,
            SUM(\"NET_SALES\") as sales,
            SUM(\"QTY\") as qty
        FROM sales_{current_year}
        WHERE \"DATE_INVOICE\" BETWEEN %s AND %s
        UNION ALL
        SELECT 
            'YTD' as period,
            SUM(\"NET_SALES\") as sales,
            SUM(\"QTY\") as qty
        FROM sales_{current_year}
        WHERE \"DATE_INVOICE\" BETWEEN %s AND %s
        """
    
    if last_year >= 2026:
        last_query = f"""
        SELECT 
            'MTD' as period,
            SUM(net_sales) as sales_ly,
            SUM(qty) as qty_ly
        FROM sales_{last_year}
        WHERE date_invoice BETWEEN %s AND %s
        UNION ALL
        SELECT 
            'YTD' as period,
            SUM(net_sales) as sales_ly,
            SUM(qty) as qty_ly
        FROM sales_{last_year}
        WHERE date_invoice BETWEEN %s AND %s
        """
    else:
        last_query = f"""
        SELECT 
            'MTD' as period,
            SUM(\"NET_SALES\") as sales_ly,
            SUM(\"QTY\") as qty_ly
        FROM sales_{last_year}
        WHERE \"DATE_INVOICE\" BETWEEN %s AND %s
        UNION ALL
        SELECT 
            'YTD' as period,
            SUM(\"NET_SALES\") as sales_ly,
            SUM(\"QTY\") as qty_ly
        FROM sales_{last_year}
        WHERE \"DATE_INVOICE\" BETWEEN %s AND %s
        """
    
    query = f"""
    WITH current_year AS ({current_query}),
    last_year AS ({last_query})
    SELECT 
        cy.period,
        cy.sales,
        cy.qty,
        ly.sales_ly,
        ly.qty_ly,
        CASE WHEN ly.sales_ly > 0 
            THEN ((cy.sales - ly.sales_ly) / ly.sales_ly * 100)
            ELSE 0 
        END as sales_growth,
        CASE WHEN ly.qty_ly > 0 
            THEN ((cy.qty - ly.qty_ly) / ly.qty_ly * 100)
            ELSE 0 
        END as qty_growth
    FROM current_year cy
    JOIN last_year ly ON cy.period = ly.period
    """
    
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=[
            mtd_start, yesterday, ytd_start, yesterday,
            mtd_start_ly, mtd_end_ly, ytd_start_ly, ytd_end_ly
        ])
    
    return df

@st.cache_data(ttl=600)
def get_top_bottom_performers():
    """Get top and bottom performing departments"""
    yesterday = (datetime.today() - timedelta(days=1)).date()
    ytd_start = datetime(yesterday.year, 1, 1).date()
    ytd_start_ly = ytd_start.replace(year=ytd_start.year - 1)
    ytd_end_ly = yesterday.replace(year=yesterday.year - 1)
    
    # Determine which tables to query based on year
    current_year = yesterday.year
    last_year = current_year - 1
    
    if current_year >= 2026:
        # sales_2026+ uses lowercase columns
        current_query = f"""
        SELECT 
            dept,
            SUM(net_sales) as sales,
            SUM(qty) as qty
        FROM sales_{current_year}
        WHERE date_invoice BETWEEN %s AND %s
        GROUP BY dept
        """
    else:
        # sales_2024/2025 use uppercase quoted columns
        current_query = f"""
        SELECT 
            \"DEPT\" as dept,
            SUM(\"NET_SALES\") as sales,
            SUM(\"QTY\") as qty
        FROM sales_{current_year}
        WHERE \"DATE_INVOICE\" BETWEEN %s AND %s
        GROUP BY \"DEPT\"
        """
    
    if last_year >= 2026:
        # sales_2026+ uses lowercase columns
        last_query = f"""
        SELECT 
            dept,
            SUM(net_sales) as sales_ly,
            SUM(qty) as qty_ly
        FROM sales_{last_year}
        WHERE date_invoice BETWEEN %s AND %s
        GROUP BY dept
        """
    else:
        # sales_2024/2025 use uppercase quoted columns
        last_query = f"""
        SELECT 
            \"DEPT\" as dept,
            SUM(\"NET_SALES\") as sales_ly,
            SUM(\"QTY\") as qty_ly
        FROM sales_{last_year}
        WHERE \"DATE_INVOICE\" BETWEEN %s AND %s
        GROUP BY \"DEPT\"
        """
    
    query = f"""
    WITH current_year AS (
        {current_query}
    ),
    last_year AS (
        {last_query}
    )
    SELECT 
        cy.dept as \"DEPT\",
        cy.sales,
        cy.qty,
        COALESCE(ly.sales_ly, 0) as sales_ly,
        COALESCE(ly.qty_ly, 0) as qty_ly,
        CASE WHEN COALESCE(ly.sales_ly, 0) > 0 
            THEN ((cy.sales - ly.sales_ly) / ly.sales_ly * 100)
            ELSE 100 
        END as growth,
        (cy.sales / NULLIF((SELECT SUM(sales) FROM current_year), 0) * 100) as contribution
    FROM current_year cy
    LEFT JOIN last_year ly ON cy.dept = ly.dept
    WHERE cy.dept NOT IN ('OVERALL', 'TOTAL', 'GRAND TOTAL')
    ORDER BY cy.sales DESC
    """
    
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=[
            ytd_start, yesterday, ytd_start_ly, ytd_end_ly
        ])
    
    return df

@st.cache_data(ttl=600)
def get_weekly_trend():
    """Get weekly sales trend for last 8 weeks"""
    yesterday = (datetime.today() - timedelta(days=1)).date()
    start_date = yesterday - timedelta(days=56)  # 8 weeks
    
    # Determine which tables to query based on year
    current_year = yesterday.year
    last_year = current_year - 1
    
    if current_year >= 2026:
        # sales_2026+ uses lowercase columns
        current_query = f"""
        SELECT 
            DATE_TRUNC('week', date_invoice)::date as week_start,
            SUM(net_sales) as sales,
            SUM(qty) as qty
        FROM sales_{current_year}
        WHERE date_invoice BETWEEN %s AND %s
        GROUP BY DATE_TRUNC('week', date_invoice)
        """
    else:
        # sales_2024/2025 use uppercase quoted columns
        current_query = f"""
        SELECT 
            DATE_TRUNC('week', \"DATE_INVOICE\")::date as week_start,
            SUM(\"NET_SALES\") as sales,
            SUM(\"QTY\") as qty
        FROM sales_{current_year}
        WHERE \"DATE_INVOICE\" BETWEEN %s AND %s
        GROUP BY DATE_TRUNC('week', \"DATE_INVOICE\")
        """
    
    if last_year >= 2026:
        # sales_2026+ uses lowercase columns
        last_query = f"""
        SELECT 
            DATE_TRUNC('week', date_invoice)::date as week_start,
            SUM(net_sales) as sales_ly
        FROM sales_{last_year}
        WHERE date_invoice BETWEEN %s AND %s
        GROUP BY DATE_TRUNC('week', date_invoice)
        """
    else:
        # sales_2024/2025 use uppercase quoted columns
        last_query = f"""
        SELECT 
            DATE_TRUNC('week', \"DATE_INVOICE\")::date as week_start,
            SUM(\"NET_SALES\") as sales_ly
        FROM sales_{last_year}
        WHERE \"DATE_INVOICE\" BETWEEN %s AND %s
        GROUP BY DATE_TRUNC('week', \"DATE_INVOICE\")
        """
    
    query = f"""
    WITH weeks_current AS (
        {current_query}
    ),
    weeks_last AS (
        {last_query}
    )
    SELECT 
        wc.week_start,
        wc.sales,
        wc.qty,
        COALESCE(wl.sales_ly, 0) as sales_ly
    FROM weeks_current wc
    LEFT JOIN weeks_last wl ON wc.week_start = wl.week_start + INTERVAL '1 year'
    ORDER BY wc.week_start
    """
    
    start_date_ly = start_date.replace(year=start_date.year - 1)
    yesterday_ly = yesterday.replace(year=yesterday.year - 1)
    
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=[
            start_date, yesterday, start_date_ly, yesterday_ly
        ])
    
    return df

@st.cache_data(ttl=600)
def get_shop_performance():
    """Get shop-level performance YTD"""
    yesterday = (datetime.today() - timedelta(days=1)).date()
    ytd_start = datetime(yesterday.year, 1, 1).date()
    
    # Determine which tables to query based on year
    current_year = yesterday.year
    last_year = current_year - 1
    
    if current_year >= 2026:
        # sales_2026+ uses lowercase columns
        current_query = f"""
        SELECT 
            shop_code,
            SUM(net_sales) as sales,
            SUM(qty) as qty
        FROM sales_{current_year}
        WHERE date_invoice BETWEEN %s AND %s
        GROUP BY shop_code
        """
    else:
        # sales_2024/2025 use uppercase quoted columns
        current_query = f"""
        SELECT 
            \"SHOP_CODE\" as shop_code,
            SUM(\"NET_SALES\") as sales,
            SUM(\"QTY\") as qty
        FROM sales_{current_year}
        WHERE \"DATE_INVOICE\" BETWEEN %s AND %s
        GROUP BY \"SHOP_CODE\"
        """
    
    if last_year >= 2026:
        # sales_2026+ uses lowercase columns
        last_query = f"""
        SELECT 
            shop_code,
            SUM(net_sales) as sales_ly
        FROM sales_{last_year}
        WHERE date_invoice BETWEEN %s AND %s
        GROUP BY shop_code
        """
    else:
        # sales_2024/2025 use uppercase quoted columns
        last_query = f"""
        SELECT 
            \"SHOP_CODE\" as shop_code,
            SUM(\"NET_SALES\") as sales_ly
        FROM sales_{last_year}
        WHERE \"DATE_INVOICE\" BETWEEN %s AND %s
        GROUP BY \"SHOP_CODE\"
        """
    
    query = f"""
    WITH current_year AS (
        {current_query}
    ),
    last_year AS (
        {last_query}
    )
    SELECT 
        cy.shop_code,
        cy.sales,
        cy.qty,
        COALESCE(ly.sales_ly, 0) as sales_ly,
        CASE WHEN COALESCE(ly.sales_ly, 0) > 0 
            THEN ((cy.sales - ly.sales_ly) / ly.sales_ly * 100)
            ELSE 100 
        END as growth
    FROM current_year cy
    LEFT JOIN last_year ly ON cy.shop_code = ly.shop_code
    ORDER BY cy.sales DESC
    """
    
    ytd_start_ly = ytd_start.replace(year=ytd_start.year - 1)
    ytd_end_ly = yesterday.replace(year=yesterday.year - 1)
    
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=[
            ytd_start, yesterday, ytd_start_ly, ytd_end_ly
        ])
    
    return df

# ====================== HELPER FUNCTIONS ======================
def format_currency(value):
    """Format as Ghanaian Cedis"""
    if value is None or pd.isna(value):
        return "GH₵0"
    if value >= 1_000_000:
        return f"GH₵{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"GH₵{value/1_000:.1f}K"
    else:
        return f"GH₵{value:.0f}"

def format_number(value):
    """Format large numbers"""
    if value is None or pd.isna(value):
        return "0"
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:.0f}"

def get_growth_color(growth):
    """Get color based on growth rate"""
    if growth is None or pd.isna(growth):
        return MELCOM_BLUE
    if growth >= 10:
        return MELCOM_GREEN
    elif growth >= 0:
        return MELCOM_BLUE
    elif growth >= -5:
        return MELCOM_ORANGE
    else:
        return MELCOM_RED

def get_growth_icon(growth):
    """Get icon based on growth"""
    if growth is None or pd.isna(growth):
        return "📊"
    if growth >= 5:
        return "🚀"
    elif growth >= 0:
        return "📈"
    elif growth >= -5:
        return "⚠️"
    else:
        return "🔻"

# ====================== MAIN APP ======================
def main():
    # Header
    st.markdown('<div class="main-header">📊 Melcom Performance Insights</div>', unsafe_allow_html=True)
    st.caption(f"📅 Data as of: {(datetime.today() - timedelta(days=1)).strftime('%B %d, %Y')}")
    
    # Add cache control
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    with st.spinner("📊 Loading business insights..."):
        try:
            snapshot_df = get_business_snapshot()
            dept_df = get_top_bottom_performers()
            weekly_df = get_weekly_trend()
            shop_df = get_shop_performance()
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            st.info("💡 Try refreshing the page or clearing cache (press 'C' then 'Clear cache')")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
            st.stop()
    
    # Check if we have data
    if snapshot_df.empty or dept_df.empty:
        yesterday = (datetime.today() - timedelta(days=1)).date()
        current_year = yesterday.year
        last_year = current_year - 1
        st.error(f"❌ No sales data available. Please ensure sales_{current_year} and sales_{last_year} tables have data.")
        st.info("💡 Use the Home Portal to upload sales data first.")
        st.stop()
    
    # ====================== EXECUTIVE OVERVIEW ======================
    st.markdown("---")
    st.markdown("## 📊 Performance Overview")
    
    # Extract key metrics
    mtd_data = snapshot_df[snapshot_df['period'] == 'MTD'].iloc[0]
    ytd_data = snapshot_df[snapshot_df['period'] == 'YTD'].iloc[0]
    
    # Handle null values
    for col in ['sales', 'qty', 'sales_ly', 'qty_ly', 'sales_growth', 'qty_growth']:
        if pd.isna(mtd_data[col]):
            mtd_data[col] = 0
        if pd.isna(ytd_data[col]):
            ytd_data[col] = 0
    
    # Performance metrics
    yesterday = (datetime.today() - timedelta(days=1)).date()
    current_year = yesterday.year
    last_year = current_year - 1
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        growth_color = get_growth_color(ytd_data['sales_growth'])
        growth_icon = get_growth_icon(ytd_data['sales_growth'])
        st.markdown(f"""
        <div class="insight-card">
            <div class="metric-label">Year-to-Date Sales</div>
            <div class="metric-big">{format_currency(ytd_data['sales'])}</div>
            <div style="font-size: 1rem; margin-top: 0.5rem; color: #666;">
                {format_currency(ytd_data['sales_ly'])} ({last_year})
            </div>
            <div style="font-size: 1.2rem; margin-top: 0.5rem;">
                <span style="color: {growth_color}; font-weight: 700;">
                    {growth_icon} {ytd_data['sales_growth']:+.1f}% YoY
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        growth_color = get_growth_color(mtd_data['sales_growth'])
        growth_icon = get_growth_icon(mtd_data['sales_growth'])
        st.markdown(f"""
        <div class="insight-card">
            <div class="metric-label">Month-to-Date Sales</div>
            <div class="metric-big">{format_currency(mtd_data['sales'])}</div>
            <div style="font-size: 1rem; margin-top: 0.5rem; color: #666;">
                {format_currency(mtd_data['sales_ly'])} ({last_year})
            </div>
            <div style="font-size: 1.2rem; margin-top: 0.5rem;">
                <span style="color: {growth_color}; font-weight: 700;">
                    {growth_icon} {mtd_data['sales_growth']:+.1f}% YoY
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_ticket = ytd_data['sales'] / ytd_data['qty'] if ytd_data['qty'] > 0 else 0
        avg_ticket_ly = ytd_data['sales_ly'] / ytd_data['qty_ly'] if ytd_data['qty_ly'] > 0 else 0
        ticket_growth = ((avg_ticket - avg_ticket_ly) / avg_ticket_ly * 100) if avg_ticket_ly > 0 else 0
        
        growth_color = get_growth_color(ticket_growth)
        growth_icon = get_growth_icon(ticket_growth)
        st.markdown(f"""
        <div class="insight-card">
            <div class="metric-label">Average Transaction Value</div>
            <div class="metric-big">{format_currency(avg_ticket)}</div>
            <div style="font-size: 1.2rem; margin-top: 0.5rem;">
                <span style="color: {growth_color}; font-weight: 700;">
                    {growth_icon} {ticket_growth:+.1f}% vs 2024
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ====================== DEPARTMENT INSIGHTS ======================
    st.markdown("---")
    st.markdown("## 📈 Department Performance Analysis")
    
    # Calculate department statistics
    top_performers = dept_df.nlargest(3, 'growth')
    bottom_performers = dept_df.nsmallest(3, 'growth')
    top_revenue = dept_df.nlargest(3, 'sales')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🚀 Highest Growth")
        for idx, row in top_performers.iterrows():
            badge_class = "badge-positive" if row['growth'] > 0 else "badge-negative"
            st.markdown(f"""
            <div class="dept-card">
                <h4 style="margin: 0; color: {MELCOM_BLUE};">{row['DEPT']}</h4>
                <p style="font-size: 1.3rem; font-weight: 700; margin: 0.5rem 0;">
                    {format_currency(row['sales'])}
                </p>
                <span class="performance-badge {badge_class}">
                    {row['growth']:+.1f}% YoY
                </span>
                <p style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">
                    {row['contribution']:.1f}% of total revenue
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💰 Largest Revenue")
        for idx, row in top_revenue.iterrows():
            badge_class = "badge-positive" if row['growth'] > 0 else "badge-negative"
            st.markdown(f"""
            <div class="dept-card">
                <h4 style="margin: 0; color: {MELCOM_BLUE};">{row['DEPT']}</h4>
                <p style="font-size: 1.3rem; font-weight: 700; margin: 0.5rem 0;">
                    {format_currency(row['sales'])}
                </p>
                <span class="performance-badge {badge_class}">
                    {row['growth']:+.1f}% YoY
                </span>
                <p style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">
                    {row['contribution']:.1f}% of total revenue
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 📉 Declining Departments")
        for idx, row in bottom_performers.iterrows():
            st.markdown(f"""
            <div class="dept-card">
                <h4 style="margin: 0; color: {MELCOM_BLUE};">{row['DEPT']}</h4>
                <p style="font-size: 1.3rem; font-weight: 700; margin: 0.5rem 0;">
                    {format_currency(row['sales'])}
                </p>
                <span class="performance-badge badge-negative">
                    {row['growth']:+.1f}% YoY
                </span>
                <p style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">
                    {row['contribution']:.1f}% of total revenue
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # ====================== TREND ANALYSIS ======================
    st.markdown("---")
    st.markdown("## 📈 Weekly Momentum")
    
    # Weekly trend chart
    if not weekly_df.empty:
        # Add numeric week column for trend calculation
        weekly_df['week_num'] = range(len(weekly_df))
        
        # Calculate trend using linear regression
        X = weekly_df['week_num'].values.reshape(-1, 1)
        y = weekly_df['sales'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict current trend
        weekly_df['trend'] = model.predict(X)
        
        # Forecast next 4 weeks
        future_weeks = np.array(range(len(weekly_df), len(weekly_df) + 4)).reshape(-1, 1)
        future_predictions = model.predict(future_weeks)
        
        # Create future dates
        last_date = weekly_df['week_start'].iloc[-1]
        future_dates = [last_date + timedelta(days=7*i) for i in range(1, 5)]
        
        fig = go.Figure()
        
        # Add current year line
        fig.add_trace(go.Scatter(
            x=weekly_df['week_start'],
            y=weekly_df['sales'],
            name=f'{current_year} Actual',
            mode='lines+markers',
            line=dict(color=MELCOM_RED, width=4),
            marker=dict(size=10),
            fill='tonexty',
            fillcolor='rgba(220, 38, 38, 0.1)'
        ))
        
        # Add last year line
        fig.add_trace(go.Scatter(
            x=weekly_df['week_start'],
            y=weekly_df['sales_ly'],
            name=f'{last_year} Actual',
            mode='lines+markers',
            line=dict(color=MELCOM_BLUE, width=3, dash='dot'),
            marker=dict(size=8)
        ))
        
        # Add trend line
        fig.add_trace(go.Scatter(
            x=weekly_df['week_start'],
            y=weekly_df['trend'],
            name='Trend (ML)',
            mode='lines',
            line=dict(color='purple', width=2, dash='dash'),
            opacity=0.7
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions,
            name='Forecast (4 weeks)',
            mode='lines+markers',
            line=dict(color='orange', width=2, dash='dash'),
            marker=dict(size=8, symbol='diamond'),
            opacity=0.8
        ))
        
        # Calculate trend direction
        trend_slope = model.coef_[0]
        trend_direction = "📈 Upward" if trend_slope > 0 else "📉 Downward"
        trend_pct = (trend_slope / weekly_df['sales'].mean() * 100) if weekly_df['sales'].mean() > 0 else 0
        
        fig.update_layout(
            title=f'Weekly Sales Trend: {current_year} vs {last_year} (ML Trend: {trend_direction} {abs(trend_pct):.1f}%/week)',
            xaxis_title='Week Starting',
            yaxis_title='Sales (GH₵)',
            height=450,
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend analysis
        last_4_weeks = weekly_df.tail(4)
        recent_avg = last_4_weeks['sales'].mean()
        prev_4_weeks = weekly_df.iloc[-8:-4]
        prev_avg = prev_4_weeks['sales'].mean()
        momentum = ((recent_avg - prev_avg) / prev_avg * 100) if prev_avg > 0 else 0
        
        # Momentum summary with ML insights
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Last 4 Weeks Avg", format_currency(recent_avg), 
                     f"{momentum:+.1f}% vs Prior 4 Weeks")
        with col2:
            st.metric("Previous 4 Weeks Avg", format_currency(prev_avg))
        with col3:
            momentum_status = "🚀 Accelerating" if momentum > 5 else ("⚠️ Decelerating" if momentum < -5 else "📊 Stable")
            st.metric("Momentum Status", momentum_status)
        with col4:
            next_week_forecast = future_predictions[0]
            current_week = weekly_df['sales'].iloc[-1]
            forecast_change = ((next_week_forecast - current_week) / current_week * 100) if current_week > 0 else 0
            st.metric("Next Week Forecast", format_currency(next_week_forecast),
                     f"{forecast_change:+.1f}%")
    
    # ====================== DEPARTMENT DEEP DIVE ======================
    st.markdown("---")
    st.markdown("## 🎯 Department Performance Matrix")
    
    # Create performance quadrant
    dept_df['sales_share'] = (dept_df['sales'] / dept_df['sales'].sum() * 100)
    
    # Scatter plot
    fig = px.scatter(
        dept_df,
        x='growth',
        y='sales_share',
        size='sales',
        color='growth',
        hover_name='DEPT',
        hover_data={'sales': ':,.0f', 'growth': ':.1f', 'sales_share': ':.1f'},
        labels={'growth': 'YoY Growth %', 'sales_share': 'Revenue Share %'},
        color_continuous_scale=['red', 'yellow', 'green'],
        title='Department Performance Matrix - Size = Sales Volume'
    )
    
    # Add quadrant lines
    fig.add_hline(y=dept_df['sales_share'].median(), line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=20, y=dept_df['sales_share'].max() * 0.9, text="⭐ Stars", showarrow=False, font=dict(size=14, color="green"))
    fig.add_annotation(x=-20, y=dept_df['sales_share'].max() * 0.9, text="❓ Question Marks", showarrow=False, font=dict(size=14, color="orange"))
    fig.add_annotation(x=20, y=dept_df['sales_share'].min(), text="💎 Cash Cows", showarrow=False, font=dict(size=14, color="blue"))
    fig.add_annotation(x=-20, y=dept_df['sales_share'].min(), text="⚠️ Concerns", showarrow=False, font=dict(size=14, color="red"))
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Quadrant explanations
    col1, col2, col3, col4 = st.columns(4)
    
    stars = dept_df[(dept_df['growth'] > 0) & (dept_df['sales_share'] > dept_df['sales_share'].median())]
    questions = dept_df[(dept_df['growth'] < 0) & (dept_df['sales_share'] > dept_df['sales_share'].median())]
    cows = dept_df[(dept_df['growth'] > 0) & (dept_df['sales_share'] <= dept_df['sales_share'].median())]
    concerns = dept_df[(dept_df['growth'] < 0) & (dept_df['sales_share'] <= dept_df['sales_share'].median())]
    
    with col1:
        st.markdown(f"""
        **⭐ Stars** ({len(stars)})  
        High growth + High revenue  
        *Invest more resources*
        """)
    
    with col2:
        st.markdown(f"""
        **❓ Question Marks** ({len(questions)})  
        Declining but large revenue  
        *Urgent intervention needed*
        """)
    
    with col3:
        st.markdown(f"""
        **💎 Cash Cows** ({len(cows)})  
        Growing but smaller revenue  
        *Scale up opportunities*
        """)
    
    with col4:
        st.markdown(f"""
        **⚠️ Concerns** ({len(concerns)})  
        Small revenue + Declining  
        *Consider restructuring*
        """)
    
    # ====================== SHOP PERFORMANCE ======================
    st.markdown("---")
    st.markdown("## 🏪 Shop Performance Leaderboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Top shops chart
        top_shops = shop_df.head(15)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_shops['sales'],
            y=top_shops['shop_code'],
            orientation='h',
            marker=dict(
                color=top_shops['growth'],
                colorscale=['red', 'yellow', 'green'],
                showscale=True,
                colorbar=dict(title="Growth %")
            ),
            text=top_shops['sales'].apply(lambda x: format_currency(x)),
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Top 15 Shops by YTD Sales',
            xaxis_title='Sales (GH₵)',
            yaxis_title='Shop Code',
            height=500,
            yaxis=dict(autorange="reversed")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Shop Statistics")
        
        top_shop = shop_df.iloc[0]
        st.markdown(f"""
        <div class="dept-card">
            <div class="metric-label">Highest Sales</div>
            <h2 style="color: {MELCOM_BLUE};">{top_shop['shop_code']}</h2>
            <p style="font-size: 1.2rem;">{format_currency(top_shop['sales'])}</p>
            <span class="performance-badge {'badge-positive' if top_shop['growth'] > 0 else 'badge-negative'}">
                {top_shop['growth']:+.1f}% YoY
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        fastest_growing = shop_df.nlargest(1, 'growth').iloc[0]
        st.markdown(f"""
        <div class="dept-card">
            <div class="metric-label">Highest Growth</div>
            <h2 style="color: {MELCOM_GREEN};">{fastest_growing['shop_code']}</h2>
            <p style="font-size: 1.2rem;">{format_currency(fastest_growing['sales'])}</p>
            <span class="performance-badge badge-positive">
                {fastest_growing['growth']:+.1f}% YoY
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        declining_shops = len(shop_df[shop_df['growth'] < 0])
        positive_shops = len(shop_df[shop_df['growth'] > 0])
        st.markdown(f"""
        <div class="dept-card">
            <div class="metric-label">Growth Distribution</div>
            <p style="font-size: 1.1rem; margin: 0.5rem 0;">
                <span style="color: {MELCOM_GREEN}; font-weight: 700;">▲ {positive_shops}</span> Growing
            </p>
            <p style="font-size: 1.1rem; margin: 0.5rem 0;">
                <span style="color: {MELCOM_RED}; font-weight: 700;">▼ {declining_shops}</span> Declining
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ====================== SUMMARY STATISTICS ======================
    st.markdown("---")
    st.markdown("## 📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        growing_depts = len(dept_df[dept_df['growth'] > 0])
        total_depts = len(dept_df)
        st.metric("Departments Growing", f"{growing_depts}/{total_depts}", 
                 f"{(growing_depts/total_depts*100):.0f}%")
    
    with col2:
        growing_shops = len(shop_df[shop_df['growth'] > 0])
        total_shops = len(shop_df)
        st.metric("Shops Growing", f"{growing_shops}/{total_shops}",
                 f"{(growing_shops/total_shops*100):.0f}%")
    
    with col3:
        avg_dept_growth = dept_df['growth'].mean()
        st.metric("Avg Department Growth", f"{avg_dept_growth:+.1f}%")
    
    with col4:
        avg_shop_growth = shop_df['growth'].mean()
        st.metric("Avg Shop Growth", f"{avg_shop_growth:+.1f}%")
    
    # Footer
    st.markdown("---")
    st.caption(f"📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data as of: {yesterday.strftime('%Y-%m-%d')}")

# ====================== RUN APP ======================
if __name__ == "__main__":
    main()
