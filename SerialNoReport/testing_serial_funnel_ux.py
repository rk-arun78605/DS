"""
Serial Number Tracking Dashboard - UX Optimized Version
Modern analytics dashboard implementing best UX practices
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import logging

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Serial Tracking Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM STYLING - ACCESSIBILITY & VISUAL HIERARCHY
# ============================================================
st.markdown("""
<style>
    /* Accessible color palette with proper contrast */
    :root {
        --primary-color: #2E86AB;
        --success-color: #06A77D;
        --warning-color: #F77F00;
        --danger-color: #D62828;
        --neutral-color: #6C757D;
    }
    
    /* Clear visual hierarchy */
    .main-metric {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid var(--primary-color);
        padding-bottom: 0.5rem;
    }
    
    /* Tooltip styling */
    .tooltip-icon {
        display: inline-block;
        margin-left: 0.5rem;
        color: var(--primary-color);
        cursor: help;
    }
    
    /* Accessible buttons */
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Data freshness indicator */
    .data-freshness {
        background: #e8f5e9;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        border-left: 4px solid var(--success-color);
        margin-bottom: 1rem;
    }
    
    /* Alert styling */
    .alert-critical {
        background: #ffebee;
        border-left: 4px solid var(--danger-color);
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: #fff3e0;
        border-left: 4px solid var(--warning-color);
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATABASE CONNECTION (CONSISTENT PATTERN)
# ============================================================
@contextmanager
def get_db_connection():
    """Context manager for database connections - simplified pattern"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=3307,
            user='postgres',
            password='hello',
            database='WH'
        )
        yield conn
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

# ============================================================
# DATA LOADING WITH ERROR HANDLING
# ============================================================
@st.cache_data(ttl=600, show_spinner="Loading funnel data...")
def load_funnel_metrics(start_date: datetime.date, end_date: datetime.date) -> dict:
    """
    Load core funnel metrics with transparent data source
    Returns: Dictionary with metrics and metadata
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            WITH wh_received AS (
                SELECT COUNT(DISTINCT serial_no) as count
                FROM whreceived_serialno
                WHERE grn_date BETWEEN %s AND %s
                  AND serial_no IS NOT NULL 
                  AND LENGTH(TRIM(serial_no)) > 0
            ),
            daily_data AS (
                SELECT 
                    COUNT(CASE WHEN serial_no IS NOT NULL AND TRIM(serial_no) != '' THEN 1 END) as loaded,
                    COUNT(CASE WHEN vc_serail_no IS NOT NULL AND TRIM(vc_serail_no) != '' THEN 1 END) as offloaded,
                    COUNT(CASE WHEN shop_serail_no IS NOT NULL AND TRIM(shop_serail_no) != '' THEN 1 END) as sold
                FROM serial_no_dailydata
                WHERE loaded_datetime BETWEEN %s AND %s
            )
            SELECT 
                w.count as wh_received,
                d.loaded,
                d.offloaded,
                d.sold,
                CURRENT_TIMESTAMP as data_timestamp
            FROM wh_received w, daily_data d
            """
            
            cursor.execute(query, (start_date, end_date, start_date, end_date))
            result = cursor.fetchone()
            
            if result:
                return {
                    'wh_received': result['wh_received'],
                    'loaded': result['loaded'],
                    'offloaded': result['offloaded'],
                    'sold': result['sold'],
                    'data_timestamp': result['data_timestamp'],
                    'date_range': f"{start_date} to {end_date}",
                    'success': True,
                    'error': None
                }
            else:
                return {'success': False, 'error': 'No data found'}
                
    except Exception as e:
        return {'success': False, 'error': str(e)}

@st.cache_data(ttl=600)
def load_issue_summary(start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """Load issue breakdown with clear categorization"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT 
                'No Offloading' as issue_type,
                'Process' as category,
                COUNT(*) as count,
                'Items loaded at WH but not offloaded at shop' as description
            FROM serial_no_dailydata
            WHERE loaded_datetime BETWEEN %s AND %s
              AND serial_no IS NOT NULL AND TRIM(serial_no) != ''
              AND (vc_serail_no IS NULL OR TRIM(vc_serail_no) = '')
            
            UNION ALL
            
            SELECT 
                'Shop Mismatch' as issue_type,
                'Routing' as category,
                COUNT(*) as count,
                'Items offloaded to one shop but sold at different shop' as description
            FROM serial_no_dailydata
            WHERE loaded_datetime BETWEEN %s AND %s
              AND vc_shop_code IS NOT NULL 
              AND shop_sold IS NOT NULL
              AND UPPER(TRIM(vc_shop_code)) != UPPER(TRIM(shop_sold))
            
            ORDER BY count DESC
            """
            
            cursor.execute(query, (start_date, end_date, start_date, end_date))
            return pd.DataFrame(cursor.fetchall())
            
    except Exception as e:
        st.error(f"Error loading issues: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_daily_trends(start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """Load daily funnel metrics for trend analysis"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            WITH daily_metrics AS (
                SELECT 
                    DATE(loaded_datetime) as date,
                    COUNT(CASE WHEN serial_no IS NOT NULL AND TRIM(serial_no) != '' THEN 1 END) as loaded,
                    COUNT(CASE WHEN vc_serail_no IS NOT NULL AND TRIM(vc_serail_no) != '' THEN 1 END) as offloaded,
                    COUNT(CASE WHEN shop_serail_no IS NOT NULL AND TRIM(shop_serail_no) != '' THEN 1 END) as sold
                FROM serial_no_dailydata
                WHERE loaded_datetime BETWEEN %s AND %s
                GROUP BY DATE(loaded_datetime)
            ),
            wh_daily AS (
                SELECT 
                    DATE(grn_date) as date,
                    COUNT(DISTINCT serial_no) as wh_received
                FROM whreceived_serialno
                WHERE grn_date BETWEEN %s AND %s
                  AND serial_no IS NOT NULL 
                  AND LENGTH(TRIM(serial_no)) > 0
                GROUP BY DATE(grn_date)
            )
            SELECT 
                COALESCE(d.date, w.date) as date,
                COALESCE(w.wh_received, 0) as wh_received,
                COALESCE(d.loaded, 0) as loaded,
                COALESCE(d.offloaded, 0) as offloaded,
                COALESCE(d.sold, 0) as sold
            FROM daily_metrics d
            FULL OUTER JOIN wh_daily w ON d.date = w.date
            ORDER BY date
            """
            
            cursor.execute(query, (start_date, end_date, start_date, end_date))
            return pd.DataFrame(cursor.fetchall())
            
    except Exception as e:
        st.error(f"Error loading daily trends: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_shop_performance(start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """Load shop-level performance metrics"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT 
                COALESCE(UPPER(TRIM(vc_shop_code)), 'Unknown') as shop_code,
                COUNT(CASE WHEN vc_serail_no IS NOT NULL AND TRIM(vc_serail_no) != '' THEN 1 END) as offloaded,
                COUNT(CASE WHEN shop_serail_no IS NOT NULL AND TRIM(shop_serail_no) != '' THEN 1 END) as sold,
                COUNT(CASE WHEN vc_shop_code IS NOT NULL AND shop_sold IS NOT NULL 
                           AND UPPER(TRIM(vc_shop_code)) != UPPER(TRIM(shop_sold)) THEN 1 END) as mismatched,
                CASE 
                    WHEN COUNT(CASE WHEN vc_serail_no IS NOT NULL AND TRIM(vc_serail_no) != '' THEN 1 END) > 0
                    THEN ROUND(100.0 * COUNT(CASE WHEN shop_serail_no IS NOT NULL AND TRIM(shop_serail_no) != '' THEN 1 END) / 
                               COUNT(CASE WHEN vc_serail_no IS NOT NULL AND TRIM(vc_serail_no) != '' THEN 1 END), 1)
                    ELSE 0
                END as sell_through_rate
            FROM serial_no_dailydata
            WHERE loaded_datetime BETWEEN %s AND %s
              AND vc_shop_code IS NOT NULL
            GROUP BY UPPER(TRIM(vc_shop_code))
            HAVING COUNT(CASE WHEN vc_serail_no IS NOT NULL AND TRIM(vc_serail_no) != '' THEN 1 END) > 0
            ORDER BY offloaded DESC
            """
            
            cursor.execute(query, (start_date, end_date))
            return pd.DataFrame(cursor.fetchall())
            
    except Exception as e:
        st.error(f"Error loading shop performance: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_serial_verification_by_shop(start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """Load serial verification metrics by shop from main database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            WITH shop_serials AS (
                SELECT 
                    COALESCE(UPPER(TRIM(shop_sold)), 'Unknown') as shop_code,
                    UPPER(TRIM(shop_serail_no)) as serial_no
                FROM serial_no_dailydata
                WHERE loaded_datetime BETWEEN %s AND %s
                  AND shop_serail_no IS NOT NULL
                  AND LENGTH(TRIM(shop_serail_no)) > 0
                  AND shop_sold IS NOT NULL
            ),
            verified_serials AS (
                SELECT DISTINCT UPPER(TRIM(serial_number)) as serial_no
                FROM serialnocheck_indb
                WHERE serial_check = 'Y'
            )
            SELECT 
                s.shop_code,
                COUNT(DISTINCT s.serial_no) as total_serials,
                COUNT(DISTINCT v.serial_no) as matched_serials,
                COUNT(DISTINCT s.serial_no) - COUNT(DISTINCT v.serial_no) as unmatched_serials,
                ROUND(100.0 * COUNT(DISTINCT v.serial_no) / 
                      NULLIF(COUNT(DISTINCT s.serial_no), 0), 1) as match_rate_pct
            FROM shop_serials s
            LEFT JOIN verified_serials v ON s.serial_no = v.serial_no
            GROUP BY s.shop_code
            ORDER BY match_rate_pct ASC NULLS FIRST
            """
            
            cursor.execute(query, (start_date, end_date))
            return pd.DataFrame(cursor.fetchall())
            
    except Exception as e:
        st.error(f"Error loading serial verification: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_brand_performance(start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """Load brand-level performance metrics"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            WITH brand_data AS (
                SELECT 
                    SPLIT_PART(vc_item_desc, ' ', 1) as brand,
                    COUNT(CASE WHEN serial_no IS NOT NULL AND TRIM(serial_no) != '' THEN 1 END) as loaded,
                    COUNT(CASE WHEN vc_serail_no IS NOT NULL AND TRIM(vc_serail_no) != '' THEN 1 END) as offloaded,
                    COUNT(CASE WHEN shop_serail_no IS NOT NULL AND TRIM(shop_serail_no) != '' THEN 1 END) as sold
                FROM serial_no_dailydata
                WHERE loaded_datetime BETWEEN %s AND %s
                  AND vc_item_desc IS NOT NULL
                GROUP BY SPLIT_PART(vc_item_desc, ' ', 1)
                HAVING COUNT(CASE WHEN serial_no IS NOT NULL AND TRIM(serial_no) != '' THEN 1 END) > 10
            )
            SELECT 
                brand,
                loaded,
                offloaded,
                sold,
                ROUND(100.0 * offloaded / NULLIF(loaded, 0), 1) as offload_rate,
                ROUND(100.0 * sold / NULLIF(loaded, 0), 1) as sell_through_rate
            FROM brand_data
            ORDER BY loaded DESC
            LIMIT 15
            """
            
            cursor.execute(query, (start_date, end_date))
            return pd.DataFrame(cursor.fetchall())
            
    except Exception as e:
        st.error(f"Error loading brand performance: {e}")
        return pd.DataFrame()

# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    # ============================================================
    # SIDEBAR - EFFICIENT FILTERING
    # ============================================================
    with st.sidebar:
        st.image("https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg", width=100)
        st.title("📦 Serial Tracking")
        st.caption("UX-Optimized Analytics")
        
        st.markdown("---")
        
        # Date range with smart defaults
        st.markdown("### 📅 Date Range")
        
        col1, col2 = st.columns(2)
        with col1:
            quick_range = st.selectbox(
                "Quick Select",
                ["Last 7 Days", "Last 30 Days", "Last 90 Days", "This Month", "Custom"],
                help="Select a predefined date range or choose Custom"
            )
        
        today = datetime.today().date()
        yesterday = today - timedelta(days=1)
        
        if quick_range == "Last 7 Days":
            start_date = yesterday - timedelta(days=6)
            end_date = yesterday
        elif quick_range == "Last 30 Days":
            start_date = yesterday - timedelta(days=29)
            end_date = yesterday
        elif quick_range == "Last 90 Days":
            start_date = yesterday - timedelta(days=89)
            end_date = yesterday
        elif quick_range == "This Month":
            start_date = yesterday.replace(day=1)
            end_date = yesterday
        else:  # Custom
            with col2:
                st.write("")  # Spacing
            start_date = st.date_input(
                "Start Date",
                yesterday - timedelta(days=29),
                help="Select the start date for analysis"
            )
            end_date = st.date_input(
                "End Date",
                yesterday,
                help="Select the end date (yesterday is latest complete data)"
            )
        
        st.info(f"📊 Analyzing {(end_date - start_date).days + 1} days of data")
        
        st.markdown("---")
        
        # Help & Guidance
        with st.expander("❓ How to Use", expanded=False):
            st.markdown("""
            **Quick Start:**
            1. Select date range above
            2. Review funnel metrics
            3. Investigate issues in detail tabs
            
            **Key Metrics:**
            - **WH Received**: Items received from suppliers
            - **Loaded**: Items loaded for distribution
            - **Offloaded**: Items delivered to shops
            - **Sold**: Items sold to customers
            
            **Taking Action:**
            - Red alerts = Immediate action needed
            - Yellow warnings = Monitor closely
            - Green status = Normal operations
            """)
        
        with st.expander("📖 Data Sources", expanded=False):
            st.markdown("""
            **Primary Tables:**
            - `whreceived_serialno` - Warehouse receipts
            - `serial_no_dailydata` - Daily tracking data
            
            **Refresh Rate:** Every 10 minutes
            
            **Data Completeness:** 
            Yesterday is the latest complete date
            (today's data updates throughout the day)
            """)
    
    # ============================================================
    # MAIN CONTENT - PROGRESSIVE DISCLOSURE
    # ============================================================
    
    # Header with data freshness indicator
    st.title("📦 Serial Number Tracking Analytics")
    
    # Load data with error handling
    metrics = load_funnel_metrics(start_date, end_date)
    
    if not metrics.get('success'):
        st.error(f"""
        ### ⚠️ Unable to Load Data
        
        **Error:** {metrics.get('error', 'Unknown error')}
        
        **What to do:**
        1. Check database connection
        2. Verify date range has data
        3. Contact support if issue persists
        """)
        st.stop()
    
    # Data freshness indicator
    data_time = metrics['data_timestamp']
    if data_time.tzinfo is not None:
        data_time = data_time.replace(tzinfo=None)  # Remove timezone for comparison
    minutes_ago = int((datetime.now() - data_time).total_seconds() / 60)
    
    st.markdown(f"""
    <div class="data-freshness">
        🔄 <strong>Data Refreshed:</strong> {minutes_ago} minutes ago | 
        📅 <strong>Period:</strong> {metrics['date_range']} | 
        ⚡ <strong>Status:</strong> Live
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # LEVEL 1: HIGH-LEVEL OVERVIEW (PROGRESSIVE DISCLOSURE)
    # ============================================================
    st.markdown('<div class="section-header">📊 Funnel Overview</div>', unsafe_allow_html=True)
    
    # Primary KPIs with clear hierarchy
    col1, col2, col3, col4 = st.columns(4)
    
    wh = metrics['wh_received']
    loaded = metrics['loaded']
    offloaded = metrics['offloaded']
    sold = metrics['sold']
    
    # Debug: Show what we're getting from database
    st.info(f"📊 **Debug Data:** WH={wh:,} | Loaded={loaded:,} | Offloaded={offloaded:,} | Sold={sold:,}")
    
    # Calculate conversion rates
    load_rate = (loaded / wh * 100) if wh > 0 else 0
    offload_rate = (offloaded / loaded * 100) if loaded > 0 else 0
    sell_rate = (sold / wh * 100) if wh > 0 else 0
    
    with col1:
        st.metric(
            "📦 WH Received",
            f"{wh:,}",
            help="Unique serial numbers received at warehouse from suppliers"
        )
        st.caption("Starting point")
    
    with col2:
        delta_color = "normal" if load_rate >= 95 else "inverse"
        st.metric(
            "🏭 Loaded",
            f"{loaded:,}",
            f"{load_rate:.1f}%",
            delta_color=delta_color,
            help="Items loaded at warehouse for distribution (can exceed 100% if same serial sent multiple times)"
        )
        if load_rate < 95:
            st.caption("⚠️ Below target")
        else:
            st.caption("✅ On track")
    
    with col3:
        delta_color = "normal" if offload_rate >= 90 else "inverse"
        st.metric(
            "🚚 Offloaded",
            f"{offloaded:,}",
            f"{offload_rate:.1f}%",
            delta_color=delta_color,
            help="Items offloaded at shops (% of loaded items)"
        )
        if offload_rate < 90:
            st.caption("⚠️ Action needed")
        else:
            st.caption("✅ Good")
    
    with col4:
        delta_color = "normal" if sell_rate >= 5 else "inverse"
        st.metric(
            "✅ Sold",
            f"{sold:,}",
            f"{sell_rate:.1f}%",
            delta_color=delta_color,
            help="Items sold to customers (% of WH received - overall conversion)"
        )
        if sell_rate < 5:
            st.caption("📉 Monitor")
        else:
            st.caption("📈 Healthy")
    
    # Visual funnel
    st.markdown("### 📊 Conversion Funnel")
    
    col_funnel, col_insights = st.columns([2, 1])
    
    with col_funnel:
        fig = go.Figure()
        
        stages = ['WH Received', 'Loaded at WH', 'Offloaded at Shop', 'Sold to Customer']
        values = [wh, loaded, offloaded, sold]
        colors = ['#2E86AB', '#06A77D', '#F77F00', '#D62828']
        
        # Text with percentages
        texts = [
            f"{wh:,}",
            f"{loaded:,}<br>{load_rate:.1f}%",
            f"{offloaded:,}<br>{offload_rate:.1f}%",
            f"{sold:,}<br>{sell_rate:.1f}%"
        ]
        
        fig.add_trace(go.Funnel(
            y=stages,
            x=values,
            text=texts,
            textposition="inside",
            textfont=dict(size=16, family="Arial", color="white"),
            marker=dict(color=colors, line=dict(width=2, color='white')),
            connector=dict(line=dict(color="#e0e0e0", width=2))
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="main_funnel")
    
    with col_insights:
        st.markdown("#### 💡 Key Insights")
        
        # Actionable insights based on data
        if sell_rate >= 10:
            st.success("🎉 **Excellent conversion!** Overall sell-through is healthy.")
        elif sell_rate >= 5:
            st.info("📊 **Good performance** - Sell-through within normal range.")
        else:
            st.warning("📉 **Low conversion** - Review shop operations and demand.")
        
        st.markdown("---")
        
        # Drop-off analysis
        dropoff_1_2 = wh - loaded if wh > loaded else 0
        dropoff_2_3 = loaded - offloaded
        dropoff_3_4 = offloaded - sold
        
        st.markdown("**Drop-off Points:**")
        
        if dropoff_1_2 > 0:
            st.markdown(f"• Loading: {dropoff_1_2:,} items")
        if dropoff_2_3 > 0:
            st.markdown(f"• Offloading: {dropoff_2_3:,} items")
        if dropoff_3_4 > 0:
            st.markdown(f"• Selling: {dropoff_3_4:,} items")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("**Quick Actions:**")
        if st.button("📥 Export Data", key="export_main"):
            # Create downloadable data
            export_df = pd.DataFrame({
                'Metric': stages,
                'Count': values,
                'Percentage': [100, load_rate, offload_rate, sell_rate]
            })
            csv = export_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                f"funnel_data_{start_date}_{end_date}.csv",
                "text/csv",
                key="download_main_data"
            )
    
    # ============================================================
    # LEVEL 2: DRILL-DOWN TABS (PROGRESSIVE DISCLOSURE)
    # ============================================================
    st.markdown('<div class="section-header">🔍 Detailed Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚨 Issues", "📈 Trends", "🏪 Shops", "🏷️ Brands"])
    
    with tab1:
        st.markdown("### 🚨 Issue Summary")
        st.caption("Identify and prioritize issues requiring attention")
        
        issues_df = load_issue_summary(start_date, end_date)
        
        if not issues_df.empty:
            # Calculate priorities
            total_items = loaded
            issues_df['impact_%'] = (issues_df['count'] / total_items * 100).round(1)
            issues_df['priority'] = issues_df['impact_%'].apply(
                lambda x: '🔴 Critical' if x > 10 else ('🟠 High' if x > 5 else '🟡 Medium')
            )
            
            # Display as cards
            for idx, row in issues_df.iterrows():
                impact = row['impact_%']
                
                if impact > 10:
                    alert_class = "alert-critical"
                    icon = "🔴"
                elif impact > 5:
                    alert_class = "alert-warning"
                    icon = "🟠"
                else:
                    alert_class = "alert-warning"
                    icon = "🟡"
                
                st.markdown(f"""
                <div class="{alert_class}">
                    <h4>{icon} {row['issue_type']}</h4>
                    <p><strong>Category:</strong> {row['category']}</p>
                    <p>{row['description']}</p>
                    <p><strong>Count:</strong> {row['count']:,} items ({impact}% impact)</p>
                    <p><strong>Priority:</strong> {row['priority']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Actionable recommendations
                if row['issue_type'] == 'No Offloading':
                    with st.expander("💡 Recommended Actions"):
                        st.markdown("""
                        - Review shop receiving processes
                        - Check vehicle routing efficiency
                        - Verify staff training on offloading procedures
                        - Investigate system integration gaps
                        """)
                
                elif row['issue_type'] == 'Shop Mismatch':
                    with st.expander("💡 Recommended Actions"):
                        st.markdown("""
                        - Audit inter-shop transfers
                        - Review POS system shop code configuration
                        - Verify vehicle routing compliance
                        - Implement stricter shop verification
                        """)
            
            # Export issues
            if st.button("📥 Export Issues", key="export_issues"):
                csv = issues_df.to_csv(index=False)
                st.download_button(
                    "Download Issues Report",
                    csv,
                    f"issues_report_{start_date}_{end_date}.csv",
                    "text/csv",
                    key="download_issues_data"
                )
        
        else:
            st.success("✅ No significant issues detected in this period!")
    
    with tab2:
        st.markdown("### 📈 Performance Trends")
        st.caption("Track daily performance to identify patterns and anomalies")
        
        trends_df = load_daily_trends(start_date, end_date)
        
        if not trends_df.empty:
            # Time series chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=trends_df['date'],
                y=trends_df['wh_received'],
                name='📦 WH Received',
                line=dict(color='#2E86AB', width=3),
                mode='lines+markers'
            ))
            
            fig.add_trace(go.Scatter(
                x=trends_df['date'],
                y=trends_df['loaded'],
                name='🏭 Loaded',
                line=dict(color='#06A77D', width=3),
                mode='lines+markers'
            ))
            
            fig.add_trace(go.Scatter(
                x=trends_df['date'],
                y=trends_df['offloaded'],
                name='🚚 Offloaded',
                line=dict(color='#F77F00', width=3),
                mode='lines+markers'
            ))
            
            fig.add_trace(go.Scatter(
                x=trends_df['date'],
                y=trends_df['sold'],
                name='✅ Sold',
                line=dict(color='#D62828', width=3),
                mode='lines+markers'
            ))
            
            fig.update_layout(
                title="Daily Funnel Metrics Over Time",
                xaxis_title="Date",
                yaxis_title="Count",
                height=500,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True, key="trends_chart")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_offload_rate = (trends_df['offloaded'].sum() / trends_df['loaded'].sum() * 100) if trends_df['loaded'].sum() > 0 else 0
                st.metric("Avg Offload Rate", f"{avg_offload_rate:.1f}%")
            
            with col2:
                best_day = trends_df.loc[trends_df['sold'].idxmax()]
                st.metric("Best Sales Day", f"{best_day['sold']:,.0f}", f"{best_day['date']}")
            
            with col3:
                trend = "📈 Improving" if trends_df['sold'].iloc[-3:].mean() > trends_df['sold'].iloc[:3].mean() else "📉 Declining"
                st.metric("Trend Direction", trend)
            
            # Export trends
            if st.button("📥 Export Trends", key="export_trends"):
                csv = trends_df.to_csv(index=False)
                st.download_button(
                    "Download Trends Data",
                    csv,
                    f"trends_data_{start_date}_{end_date}.csv",
                    "text/csv",
                    key="download_trends_data"
                )
        else:
            st.warning("⚠️ No trend data available for selected period")
    
    with tab3:
        st.markdown("### 🏪 Shop Performance")
        st.caption("Compare shop-by-shop performance and identify top/bottom performers")
        
        shops_df = load_shop_performance(start_date, end_date)
        serial_verification_df = load_serial_verification_by_shop(start_date, end_date)
        
        if not shops_df.empty:
            # Convert numeric columns to proper types
            shops_df['offloaded'] = pd.to_numeric(shops_df['offloaded'], errors='coerce').fillna(0).astype(int)
            shops_df['sold'] = pd.to_numeric(shops_df['sold'], errors='coerce').fillna(0).astype(int)
            shops_df['mismatched'] = pd.to_numeric(shops_df['mismatched'], errors='coerce').fillna(0).astype(int)
            shops_df['sell_through_rate'] = pd.to_numeric(shops_df['sell_through_rate'], errors='coerce').fillna(0)
            
            # Serial Verification Analysis Section
            st.markdown("#### 📊 Serial Number Verification in Main Database")
            st.info("💡 Analysis of how many serial numbers from each shop are verified (found) in the main database")
            
            if not serial_verification_df.empty:
                # Convert numeric columns
                serial_verification_df['match_rate_pct'] = pd.to_numeric(serial_verification_df['match_rate_pct'], errors='coerce').fillna(0)
                serial_verification_df['total_serials'] = pd.to_numeric(serial_verification_df['total_serials'], errors='coerce').fillna(0).astype(int)
                serial_verification_df['matched_serials'] = pd.to_numeric(serial_verification_df['matched_serials'], errors='coerce').fillna(0).astype(int)
                serial_verification_df['unmatched_serials'] = pd.to_numeric(serial_verification_df['unmatched_serials'], errors='coerce').fillna(0).astype(int)
                
                # Overall verification metrics
                total_all_serials = serial_verification_df['total_serials'].sum()
                total_matched = serial_verification_df['matched_serials'].sum()
                total_unmatched = serial_verification_df['unmatched_serials'].sum()
                overall_match_rate = (total_matched / total_all_serials * 100) if total_all_serials > 0 else 0
                
                col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                
                with col_v1:
                    st.metric(
                        "📦 Total Serials Verified",
                        f"{total_all_serials:,}",
                        help="Total unique serial numbers checked in main database"
                    )
                
                with col_v2:
                    st.metric(
                        "✅ Matched in DB",
                        f"{total_matched:,}",
                        delta=f"{overall_match_rate:.1f}%",
                        delta_color="normal",
                        help="Serial numbers found and verified in main database"
                    )
                
                with col_v3:
                    unmatched_rate = (total_unmatched / total_all_serials * 100) if total_all_serials > 0 else 0
                    st.metric(
                        "❌ Not Matched",
                        f"{total_unmatched:,}",
                        delta=f"{unmatched_rate:.1f}%",
                        delta_color="inverse",
                        help="Serial numbers not found in main database"
                    )
                
                with col_v4:
                    avg_match_rate = serial_verification_df['match_rate_pct'].mean()
                    st.metric(
                        "📈 Avg Match Rate",
                        f"{avg_match_rate:.1f}%",
                        help="Average match rate across all shops"
                    )
                
                # Bottom 5 and Top 5 shops side by side
                col_bottom, col_top = st.columns(2)
                
                with col_bottom:
                    st.markdown("##### 🔴 Bottom 5 Shops - Lowest Verification Rates")
                    
                    bottom_5 = serial_verification_df.head(5)
                    
                    if not bottom_5.empty:
                        # Color code based on severity
                        colors = []
                        for rate in bottom_5['match_rate_pct']:
                            if rate == 0:
                                colors.append('#c0392b')  # Critical
                            elif rate < 50:
                                colors.append('#e74c3c')  # High
                            elif rate < 70:
                                colors.append('#e67e22')  # Medium
                            else:
                                colors.append('#f39c12')  # Low
                        
                        fig_bottom = go.Figure()
                        fig_bottom.add_trace(go.Bar(
                            x=bottom_5['shop_code'],
                            y=bottom_5['match_rate_pct'],
                            marker_color=colors,
                            text=bottom_5['match_rate_pct'].round(1),
                            texttemplate='%{text}%',
                            textposition='outside',
                            hovertemplate='<b>%{x}</b><br>Match Rate: %{y:.1f}%<extra></extra>'
                        ))
                        
                        fig_bottom.update_layout(
                            xaxis_title="Shop Code",
                            yaxis_title="Match Rate (%)",
                            height=350,
                            yaxis=dict(range=[0, 100]),
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig_bottom, use_container_width=True, key="verification_bottom5")
                
                with col_top:
                    st.markdown("##### 🟢 Top 5 Shops - Highest Verification Rates")
                    
                    top_5 = serial_verification_df.nlargest(5, 'match_rate_pct')
                    
                    if not top_5.empty:
                        fig_top = go.Figure()
                        fig_top.add_trace(go.Bar(
                            x=top_5['shop_code'],
                            y=top_5['match_rate_pct'],
                            marker_color='#27ae60',
                            text=top_5['match_rate_pct'].round(1),
                            texttemplate='%{text}%',
                            textposition='outside',
                            hovertemplate='<b>%{x}</b><br>Match Rate: %{y:.1f}%<extra></extra>'
                        ))
                        
                        fig_top.update_layout(
                            xaxis_title="Shop Code",
                            yaxis_title="Match Rate (%)",
                            height=350,
                            yaxis=dict(range=[0, 100]),
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        st.plotly_chart(fig_top, use_container_width=True, key="verification_top5")
            
            st.markdown("---")
            st.markdown("#### 📊 Shop Volume & Sell-Through Performance")
            
            # Top performers
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Shop performance chart
                fig = go.Figure()
                
                shops_top15 = shops_df.head(15)
                
                fig.add_trace(go.Bar(
                    x=shops_top15['shop_code'],
                    y=shops_top15['offloaded'],
                    name='Offloaded',
                    marker_color='#06A77D'
                ))
                
                fig.add_trace(go.Bar(
                    x=shops_top15['shop_code'],
                    y=shops_top15['sold'],
                    name='Sold',
                    marker_color='#2E86AB'
                ))
                
                fig.update_layout(
                    title="Top 15 Shops by Volume",
                    xaxis_title="Shop Code",
                    yaxis_title="Count",
                    height=400,
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True, key="shop_performance")
            
            with col2:
                st.markdown("#### 🏆 Top Performers")
                
                # Best sell-through
                best_sellthrough = shops_df.nlargest(5, 'sell_through_rate')
                for idx, row in best_sellthrough.iterrows():
                    st.markdown(f"""
                    **{row['shop_code']}**  
                    Sell-through: {row['sell_through_rate']:.1f}%  
                    Sold: {row['sold']:,}
                    """)
                    st.markdown("---")
            
            # Issues by shop
            st.markdown("#### ⚠️ Shops with Issues")
            
            issues_shops = shops_df[shops_df['mismatched'] > 0].sort_values('mismatched', ascending=False).head(10)
            
            if not issues_shops.empty:
                st.dataframe(
                    issues_shops[['shop_code', 'offloaded', 'sold', 'mismatched', 'sell_through_rate']],
                    use_container_width=True,
                    column_config={
                        'shop_code': st.column_config.TextColumn('Shop', width='small'),
                        'offloaded': st.column_config.NumberColumn('Offloaded', format='%d'),
                        'sold': st.column_config.NumberColumn('Sold', format='%d'),
                        'mismatched': st.column_config.NumberColumn('Mismatched', format='%d'),
                        'sell_through_rate': st.column_config.NumberColumn('Sell-through %', format='%.1f%%')
                    }
                )
            else:
                st.success("✅ No shop mismatch issues detected!")
            
            # Export shop data
            if st.button("📥 Export Shop Data", key="export_shops"):
                csv = shops_df.to_csv(index=False)
                st.download_button(
                    "Download Shop Performance",
                    csv,
                    f"shop_performance_{start_date}_{end_date}.csv",
                    "text/csv",
                    key="download_shop_data"
                )
        else:
            st.warning("⚠️ No shop performance data available for selected period")
    
    with tab4:
        st.markdown("### 🏷️ Brand Performance Analysis")
        st.caption("Track brand-level serial number flow and conversion rates")
        
        brand_df = load_brand_performance(start_date, end_date)
        
        if not brand_df.empty:
            # Convert numeric columns
            brand_df['loaded'] = pd.to_numeric(brand_df['loaded'], errors='coerce').fillna(0).astype(int)
            brand_df['offloaded'] = pd.to_numeric(brand_df['offloaded'], errors='coerce').fillna(0).astype(int)
            brand_df['sold'] = pd.to_numeric(brand_df['sold'], errors='coerce').fillna(0).astype(int)
            brand_df['offload_rate'] = pd.to_numeric(brand_df['offload_rate'], errors='coerce').fillna(0)
            brand_df['sell_through_rate'] = pd.to_numeric(brand_df['sell_through_rate'], errors='coerce').fillna(0)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_brands = len(brand_df)
                st.metric("📊 Brands Tracked", f"{total_brands}", help="Brands with >10 serial numbers")
            
            with col2:
                avg_sellthrough = brand_df['sell_through_rate'].mean()
                st.metric("📈 Avg Sell-Through", f"{avg_sellthrough:.1f}%", help="Average sell-through across all brands")
            
            with col3:
                best_brand = brand_df.nlargest(1, 'sell_through_rate').iloc[0]
                st.metric("🏆 Best Performer", best_brand['brand'], f"{best_brand['sell_through_rate']:.1f}%")
            
            # Dual-axis chart: Volume vs Sell-Through
            fig_brand = go.Figure()
            
            fig_brand.add_trace(go.Bar(
                name='Total Loaded',
                x=brand_df['brand'],
                y=brand_df['loaded'],
                marker_color='#3498db',
                yaxis='y',
                hovertemplate='<b>%{x}</b><br>Loaded: %{y:,}<extra></extra>'
            ))
            
            fig_brand.add_trace(go.Scatter(
                name='Sell-Through %',
                x=brand_df['brand'],
                y=brand_df['sell_through_rate'],
                marker_color='#27ae60',
                yaxis='y2',
                mode='lines+markers',
                line=dict(width=3),
                hovertemplate='<b>%{x}</b><br>Sell-Through: %{y:.1f}%<extra></extra>'
            ))
            
            fig_brand.update_layout(
                title="Brand Performance: Volume vs Sell-Through Rate",
                xaxis=dict(title="Brand", tickangle=-45),
                yaxis=dict(title="Total Loaded", side='left'),
                yaxis2=dict(title="Sell-Through %", overlaying='y', side='right', range=[0, 100]),
                height=500,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_brand, use_container_width=True, key="brand_performance")
            
            # Detailed table
            st.markdown("#### 📋 Detailed Brand Performance")
            
            display_brand = brand_df.copy()
            display_brand.columns = ['Brand', 'Loaded', 'Offloaded', 'Sold', 'Offload %', 'Sell-Through %']
            
            # Add performance indicator
            def performance_indicator(rate):
                if rate >= 80:
                    return '🟢'
                elif rate >= 50:
                    return '🟡'
                else:
                    return '🔴'
            
            display_brand.insert(0, 'Status', display_brand['Sell-Through %'].apply(performance_indicator))
            
            st.dataframe(
                display_brand,
                use_container_width=True,
                height=400,
                column_config={
                    'Status': st.column_config.TextColumn('📊', width='small'),
                    'Brand': st.column_config.TextColumn('Brand', width='medium'),
                    'Loaded': st.column_config.NumberColumn('Loaded', format='%d'),
                    'Offloaded': st.column_config.NumberColumn('Offloaded', format='%d'),
                    'Sold': st.column_config.NumberColumn('Sold', format='%d'),
                    'Offload %': st.column_config.NumberColumn('Offload %', format='%.1f%%'),
                    'Sell-Through %': st.column_config.NumberColumn('Sell-Through %', format='%.1f%%')
                }
            )
            
            # Export
            if st.button("📥 Export Brand Analysis", key="export_brands"):
                csv = brand_df.to_csv(index=False)
                st.download_button(
                    "Download Brand Performance",
                    csv,
                    f"brand_performance_{start_date}_{end_date}.csv",
                    "text/csv",
                    key="download_brand_data"
                )
        else:
            st.warning("⚠️ No brand performance data available for selected period")
    
    # ============================================================
    # FOOTER - TRANSPARENCY & TRUST
    # ============================================================
    st.markdown("---")
    st.caption("""
    **About This Dashboard** | 
    Data Source: Serial Tracking Database (WH) | 
    Refresh: Every 10 minutes | 
    Version: 1.0 (UX Optimized) | 
    [Documentation](#) | [Support](#)
    """)

# ============================================================
# RUN APPLICATION
# ============================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"""
        ### ⚠️ Application Error
        
        An unexpected error occurred: {str(e)}
        
        Please refresh the page or contact support if the issue persists.
        """)
        logging.error(f"Application error: {e}", exc_info=True)
