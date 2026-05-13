import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import psycopg2
from psycopg2 import pool, Error
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import logging
from typing import Optional, Dict

# ============================================================
# ISSUE DETECTION SQL - NEW BUSINESS LOGIC
# ============================================================
# Helper function to clean serial numbers: remove leading/trailing non-alphanumeric characters
def clean_serial_sql(field):
    """Generate SQL to clean a serial number field"""
    return "REGEXP_REPLACE(" + field + ", '^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', 'g')"

# NEW LOGIC: No Offloading
# Item loaded at warehouse (serial_no has value) but NOT offloaded (vc_serail_no blank) and NOT sold yet (shop_serail_no blank)
SQL_NO_OFFLOADING = (
    "(serial_no IS NOT NULL "
    " AND LENGTH(" + clean_serial_sql('serial_no') + ") > 0"
    " AND (vc_serail_no IS NULL OR LENGTH(" + clean_serial_sql("COALESCE(vc_serail_no, '')") + ") = 0)"
    " AND (shop_serail_no IS NULL OR LENGTH(" + clean_serial_sql("COALESCE(shop_serail_no, '')") + ") = 0))"
)

# NEW LOGIC: Sold Without Offloading
# All 3 serial columns have values (loaded, offloaded, sold) BUT serial_no != shop_serail_no (mismatch between warehouse and final sale)
SQL_SOLD_WITHOUT_OFFLOADING = (
    "(serial_no IS NOT NULL AND vc_serail_no IS NOT NULL AND shop_serail_no IS NOT NULL"
    " AND LENGTH(" + clean_serial_sql('serial_no') + ") > 0"
    " AND LENGTH(" + clean_serial_sql('vc_serail_no') + ") > 0"
    " AND LENGTH(" + clean_serial_sql('shop_serail_no') + ") > 0"
    " AND UPPER(" + clean_serial_sql('serial_no') + ") != UPPER(" + clean_serial_sql('shop_serail_no') + "))"
)

# Shop mismatch: offloaded to one shop, sold by different shop
SQL_SHOP_MISMATCH = (
    "(vc_shop_code IS NOT NULL AND shop_sold IS NOT NULL"
    " AND LENGTH(TRIM(vc_shop_code)) > 0 AND LENGTH(TRIM(shop_sold)) > 0"
    " AND UPPER(TRIM(vc_shop_code)) != UPPER(TRIM(shop_sold)))"
)

# Vehicle mismatch: vehicle number mismatch between warehouse and shop offloading
SQL_VEHICLE_MISMATCH = (
    "(vc_vehicle_no IS NOT NULL AND \"vc_vehicle_no.1\" IS NOT NULL"
    " AND vc_vehicle_no != \"vc_vehicle_no.1\")"
)

# NEW LOGIC: Serial Mismatch
# Serial numbers don't match across stages (WH → Offload → Sold)
# Check if any of the three serial fields are different from each other
SQL_SERIAL_MISMATCH = (
    "(("
    # Condition 1: serial_no != vc_serail_no (when both exist)
    "  (serial_no IS NOT NULL AND vc_serail_no IS NOT NULL"
    "   AND LENGTH(" + clean_serial_sql('serial_no') + ") > 0"
    "   AND LENGTH(" + clean_serial_sql('vc_serail_no') + ") > 0"
    "   AND UPPER(" + clean_serial_sql('serial_no') + ") != UPPER(" + clean_serial_sql('vc_serail_no') + "))"
    " OR"
    # Condition 2: vc_serail_no != shop_serail_no (when both exist)
    "  (vc_serail_no IS NOT NULL AND shop_serail_no IS NOT NULL"
    "   AND LENGTH(" + clean_serial_sql('vc_serail_no') + ") > 0"
    "   AND LENGTH(" + clean_serial_sql('shop_serail_no') + ") > 0"
    "   AND UPPER(" + clean_serial_sql('vc_serail_no') + ") != UPPER(" + clean_serial_sql('shop_serail_no') + "))"
    " OR"
    # Condition 3: shop_serail_no != serial_no (when both exist)
    "  (shop_serail_no IS NOT NULL AND serial_no IS NOT NULL"
    "   AND LENGTH(" + clean_serial_sql('shop_serail_no') + ") > 0"
    "   AND LENGTH(" + clean_serial_sql('serial_no') + ") > 0"
    "   AND UPPER(" + clean_serial_sql('shop_serail_no') + ") != UPPER(" + clean_serial_sql('serial_no') + "))"
    "))"
)

# Any issue (updated with new categories)
SQL_ANY_ISSUE = "(" + SQL_NO_OFFLOADING + " OR " + SQL_SOLD_WITHOUT_OFFLOADING + " OR " + SQL_VEHICLE_MISMATCH + " OR " + SQL_SHOP_MISMATCH + " OR " + SQL_SERIAL_MISMATCH + ")"

# ============================================================
# LOGGING & SESSION STATE
# ============================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None

# Custom CSS
st.markdown("""
<style>
    /* Base Responsive Setup - Full Width */
    * {
        box-sizing: border-box;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Remove default padding to use full width */
    .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }
    
    @media (min-width: 768px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    
    @media (min-width: 1400px) {
        .block-container {
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            max-width: 100% !important;
        }
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin-bottom: 15px;
        text-align: center;
        padding: 15px 25px;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 3.2rem;
        font-weight: bold;
        color: #667eea;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 1.1rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .dropoff-badge {
        background: #e74c3c;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 1rem;
        font-weight: bold;
        display: inline-block;
        margin-top: 5px;
    }
    
    /* Sticky Issue Badge */
    .issue-badge-sticky {
        position: fixed;
        bottom: 30px;
        right: 30px;
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 20px 25px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(231, 76, 60, 0.4);
        z-index: 9999;
        cursor: pointer;
        animation: shake 2s infinite, pulse 2s infinite;
        border: 3px solid white;
        text-align: center;
        min-width: 180px;
    }
    
    .issue-badge-sticky:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 30px rgba(231, 76, 60, 0.6);
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0) rotate(0deg); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-2px) rotate(-1deg); }
        20%, 40%, 60%, 80% { transform: translateX(2px) rotate(1deg); }
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 8px 20px rgba(231, 76, 60, 0.4); }
        50% { box-shadow: 0 8px 30px rgba(231, 76, 60, 0.8); }
    }
    
    .issue-count {
        font-size: 3rem;
        font-weight: bold;
        margin: 5px 0;
    }
    
    .issue-label {
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.95;
    }
    
    /* Top Shop Alert Popup */
    .shop-alert-popup {
        position: fixed;
        top: 80px;
        right: 30px;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(238, 90, 111, 0.5);
        z-index: 10000;
        max-width: 350px;
        animation: slideInRight 0.5s ease-out, shake 2s infinite;
        border: 2px solid white;
    }
    
    .shop-alert-popup.hidden {
        display: none;
    }
    
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .close-btn {
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(255,255,255,0.3);
        border: none;
        color: white;
        font-size: 20px;
        cursor: pointer;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        line-height: 1;
        font-weight: bold;
    }
    
    .close-btn:hover {
        background: rgba(255,255,255,0.5);
    }
    
    .shop-alert-title {
        font-size: 1.6rem;
        font-weight: bold;
        margin-bottom: 10px;
        padding-right: 30px;
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .shop-list {
        margin-top: 10px;
        font-size: 1.2rem;
    }
    
    .shop-item {
        background: rgba(255,255,255,0.25);
        padding: 10px 12px;
        margin: 6px 0;
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        font-size: 1.2rem;
        color: #ffffff;
        font-weight: 500;
    }
    
    .shop-item strong {
        font-size: 1.35rem;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATABASE CONNECTION
# ============================================================

class Config:
    """Database configuration"""
    DB_CONFIG = {
        'host': 'localhost',
        'user': 'postgres',
        'password': 'hello',
        'port': 3307
        
    }
    
    # UI Configuration
    PAGE_ICON = "https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg"
    LOGO_URL = "https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg"

@st.cache_resource
def get_connection_pool(dbname='WH'):
    """Create connection pool for database"""
    try:
        return SimpleConnectionPool(
            minconn=1, maxconn=10,
            dbname=dbname,
            connect_timeout=5,
            **Config.DB_CONFIG
        )
    except Error as e:
        logger.error(f"Connection pool error for {dbname}: {e}")
        st.error(f"❌ Cannot create connection pool for {dbname}: {e}")
        return None

@contextmanager
def get_db_connection(dbname='WH'):
    """Context manager for safe database connections"""
    pool = get_connection_pool(dbname)
    if not pool:
        raise Exception(f"Connection pool not available for {dbname}")
    
    conn = None
    try:
        conn = pool.getconn()
        yield conn
    except Error as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            pool.putconn(conn)

# ============================================================
# DATAFRAME CONFIGURATION
# ============================================================
def get_column_config(columns):
    """Generate streamlit column config for center alignment - fastest method"""
    config = {}
    for col in columns:
        config[col] = st.column_config.TextColumn(
            col,
            width="medium",
            help=None
        )
    return config

# ============================================================
# AUTHENTICATION
# ============================================================
@st.cache_data(ttl=3600, show_spinner=True)
def authenticate_user(employee_id: str, password: str) -> Optional[Dict]:
    """Authenticate user from PostgreSQL users database (cached for speed)"""
    try:
        with get_db_connection('users') as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # First check if user exists
                cursor.execute("""
                    SELECT employee_id, full_name, db_access, table_access, is_active, password
                    FROM users
                    WHERE employee_id = %s
                """, (employee_id,))
                user_check = cursor.fetchone()
                
                if not user_check:
                    logger.warning(f"Failed login - User not found: {employee_id}")
                    return None
                
                # Check if active
                if user_check['is_active'].lower() != 'true':
                    logger.warning(f"Failed login - Account inactive: {employee_id}")
                    return None
                
                # Check password
                if user_check['password'] != password:
                    logger.warning(f"Failed login - Wrong password: {employee_id}")
                    return None
                
                logger.info(f"✅ Successful login: {employee_id} ({user_check['full_name']}) (cached)")
                # Return without password field
                return {
                    'employee_id': user_check['employee_id'],
                    'full_name': user_check['full_name'],
                    'db_access': user_check['db_access'],
                    'table_access': user_check['table_access'],
                    'is_active': user_check['is_active']
                }
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None

def check_table_access(user: Dict, required_table: str) -> bool:
    """Check if user has access to required table"""
    if not user or 'table_access' not in user:
        return False
    
    user_tables = [t.strip().lower() for t in (user.get('table_access') or '').split(',')]
    return required_table.lower() in user_tables or 'all' in user_tables

@st.cache_data(ttl=3600)
def get_required_tables(dbname: str = 'WH') -> list:
    """
    Dynamically get the main tables used by this dashboard from the database
    Returns list of table names that this dashboard queries
    """
    try:
        with get_db_connection(dbname) as conn:
            with conn.cursor() as cursor:
                # Get the main table used by this dashboard
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name LIKE '%serial%'
                    ORDER BY table_name
                """)
                tables = [row[0] for row in cursor.fetchall()]
                
                if not tables:
                    # Fallback: return known table
                    return ['serial_no_dailydata']
                
                logger.info(f"Found tables for access validation: {tables}")
                return tables
    except Exception as e:
        logger.error(f"Error getting table list: {e}")
        # Fallback to known main table
        return ['serial_no_dailydata']

def validate_user_access(user: Dict) -> bool:
    """
    Validate if user has access to any of the tables used by this dashboard
    Checks against actual database tables instead of hardcoded names
    """
    if not user:
        return False
    
    # Check for 'all' access first
    if check_table_access(user, 'all'):
        return True
    
    # Get required tables from database
    required_tables = get_required_tables('WH')
    
    # Check if user has access to any of the required tables
    for table in required_tables:
        if check_table_access(user, table):
            logger.info(f"User {user.get('employee_id')} has access via table: {table}")
            return True
    
    return False

def show_login_page():
    """Display login page"""
    st.markdown(f"""
        <div style="text-align: center; padding: 30px 20px 20px 20px;">
            <img src="{Config.LOGO_URL}" style="width: 80px; height: 80px; border-radius: 12px; margin-bottom: 20px;">
            <h1 style="margin: 10px 0; color: #667eea;">🔐 SERIAL Number Funnel</h1>
            <p style="color: #666; font-size: 1.4rem;">Secure Login</p>
            <p style="color: #999; font-size: 1.1rem;">Enter your credentials to access the dashboard</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    
    with col2:
        with st.form("login_form"):
            employee_id = st.text_input("👤 Employee ID", placeholder="Enter your Employee ID")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("🚀 Login", use_container_width=True)
            
            if submit:
                if not employee_id or not password:
                    st.error("⚠️ Please enter both Employee ID and Password")
                else:
                    with st.spinner("🔄 Authenticating..."):
                        user = authenticate_user(employee_id, password)
                        
                        if user:
                            # Validate access against actual database tables
                            has_access = validate_user_access(user)
                            
                            if not has_access:
                                user_access = user.get('table_access', 'None')
                                required_tables = get_required_tables('WH')
                                st.error(f"❌ Access Denied: {user['full_name']} does not have permission")
                                st.info(f"**Your current access:** `{user_access}`\n\n**Required table access:** `{', '.join(required_tables)}` or `all`")
                                logger.warning(f"Access denied for {employee_id} - table_access: {user_access}, required: {required_tables}")
                            else:
                                st.session_state.logged_in = True
                                st.session_state.user = user
                                logger.info(f"✅ User logged in: {user['employee_id']} - {user['full_name']}")
                                st.success(f"✅ Welcome, {user['full_name']}!")
                                st.rerun()
                        else:
                            st.error("❌ Invalid credentials or account inactive")
                            logger.warning(f"Failed login attempt: {employee_id}")

# ============================================================
# DATA FUNCTIONS
# ============================================================
@st.cache_data(ttl=300)
def get_latest_data_date():
    """Get the latest loaded_datetime date from the database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Get max date only from loaded_datetime column
            cursor.execute("""
                SELECT MAX(DATE(loaded_datetime)) 
                FROM serial_no_dailydata 
                WHERE loaded_datetime IS NOT NULL
            """)
            result = cursor.fetchone()
            if result and result[0]:
                # If result is already a date object
                if hasattr(result[0], 'strftime'):
                    return result[0].strftime('%d-%b-%Y')
                # If it's a string, parse and format
                else:
                    date_obj = datetime.strptime(str(result[0]), '%Y-%m-%d')
                    return date_obj.strftime('%d-%b-%Y')
            return datetime.today().strftime('%d-%b-%Y')
    except Exception as e:
        logger.error(f"Error getting latest data date: {e}")
        # Return today's date as fallback
        return datetime.today().strftime('%d-%b-%Y')

@st.cache_data(ttl=300)
def get_available_date_range():
    """Get the date range of available data in the database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    MIN(DATE(loaded_datetime)) as min_date,
                    MAX(DATE(loaded_datetime)) as max_date,
                    COUNT(*) as total_records
                FROM serial_no_dailydata 
                WHERE loaded_datetime IS NOT NULL
            """)
            result = cursor.fetchone()
            if result and result[0] and result[1]:
                return {
                    'min_date': result[0],
                    'max_date': result[1],
                    'total_records': result[2] or 0
                }
            return None
    except Exception as e:
        logger.error(f"Error getting available date range: {e}")
        return None

@st.cache_data(ttl=300)
def get_funnel_data(start_date, end_date):
    """Get funnel metrics with drop-offs - starting from WH received"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Convert dates to strings and make end_date inclusive (end of day)
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
        
        query = f"""
            WITH wh_received AS (
                -- Stage 0: Items received at WH (from whreceived_serialno table)
                SELECT COUNT(DISTINCT serial_no) as received_count
                FROM whreceived_serialno
                WHERE grn_date >= '{start_str}' AND grn_date <= '{end_str}'
                  AND serial_no IS NOT NULL 
                  AND LENGTH(TRIM(serial_no)) > 0
            ),
            daily_data AS (
                SELECT 
                    -- Stage 1: Loaded at Warehouse (COUNT RECORDS - same serial can be sent multiple times)
                    COUNT(CASE WHEN serial_no IS NOT NULL 
                          AND TRIM(serial_no) != ''
                          THEN 1 END) as stage1_loaded,
                    
                    -- Stage 2: Offloaded to Shop (COUNT RECORDS where offload serial exists)
                    COUNT(CASE WHEN vc_serail_no IS NOT NULL 
                          AND TRIM(vc_serail_no) != ''
                          THEN 1 END) as stage2_offloaded,
                    
                    -- Stage 3: Sold (COUNT RECORDS where sold serial exists)
                    COUNT(CASE WHEN shop_serail_no IS NOT NULL 
                          AND TRIM(shop_serail_no) != ''
                          THEN 1 END) as stage3_sold,
                    
                    -- Issues (calculate from actual data with special char cleanup)
                    -- 1. No Offloading: WH sent (serial_no exists) but shop didn't offload (vc_serail_no blank)
                    COUNT(CASE 
                        WHEN {SQL_NO_OFFLOADING}
                        THEN 1 
                    END) as no_offloading,
                    
                    -- 2. Sold without offloading: All 3 serials exist but WH serial != Sold serial
                    COUNT(CASE 
                        WHEN {SQL_SOLD_WITHOUT_OFFLOADING}
                        THEN 1 
                    END) as sold_without_offload,
                    
                    -- 3. Vehicle mismatch: vehicle number mismatch between warehouse and shop
                    COUNT(CASE 
                        WHEN {SQL_VEHICLE_MISMATCH}
                        THEN 1 
                    END) as vehicle_mismatch,
                    
                    -- 4. Shop mismatch: offloaded to one shop, sold by different shop
                    COUNT(CASE 
                        WHEN {SQL_SHOP_MISMATCH}
                        THEN 1 
                    END) as shop_mismatch,
                    
                    -- 5. Serial mismatch: serial numbers don't match across stages
                    -- Checks if serial_no != vc_serail_no OR vc_serail_no != shop_serail_no OR shop_serail_no != serial_no
                    COUNT(CASE 
                        WHEN {SQL_SERIAL_MISMATCH}
                        THEN 1 
                    END) as serial_mismatch,
                    
                    -- Total records
                    COUNT(*) as total_records
                FROM serial_no_dailydata
                WHERE DATE(loaded_datetime) >= '{start_str}' AND DATE(loaded_datetime) <= '{end_str}'
            )
            SELECT 
                COALESCE(wh_received.received_count, 0) as stage0_wh_received,
                daily_data.*
            FROM wh_received, daily_data
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        
        # Ensure all values are not None and calculate total issues
        if result:
            data = dict(result)
            # Set defaults for any None values
            for key in ['stage0_wh_received', 'stage1_loaded', 'stage2_offloaded', 'stage3_sold',
                       'no_offloading', 'sold_without_offload', 'vehicle_mismatch', 'shop_mismatch', 'serial_mismatch', 'total_records']:
                if data.get(key) is None:
                    data[key] = 0
            
            # Calculate total issues as sum of all issue types
            data['total_issues'] = (
                data['no_offloading'] +
                data['sold_without_offload'] + 
                data['vehicle_mismatch'] +
                data['shop_mismatch'] + 
                data['serial_mismatch']
            )
            
            return data
        else:
            # Return empty dataset with all zeros
            return {
                'stage0_wh_received': 0, 'stage1_loaded': 0, 'stage2_offloaded': 0, 'stage3_sold': 0,
                'total_issues': 0, 'no_offloading': 0, 'sold_without_offload': 0, 'vehicle_mismatch': 0,
                'shop_mismatch': 0, 'serial_mismatch': 0, 'total_records': 0
            }

@st.cache_data(ttl=300)
def get_last_7_days_offloading(end_date):
    """Get last 7 days WH sent vs Shop offloaded data"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Calculate start date (7 days before end_date)
        from datetime import timedelta
        start_date = end_date - timedelta(days=6)  # 6 days before + today = 7 days
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        query = f"""
            SELECT 
                DATE(loaded_datetime) as date,
                COUNT(CASE WHEN serial_no IS NOT NULL AND TRIM(serial_no) != '' THEN 1 END) as wh_sent,
                COUNT(CASE WHEN vc_serail_no IS NOT NULL AND TRIM(vc_serail_no) != '' THEN 1 END) as shop_offloaded
            FROM serial_no_dailydata
            WHERE DATE(loaded_datetime) >= '{start_str}'
              AND DATE(loaded_datetime) <= '{end_str}'
            GROUP BY DATE(loaded_datetime)
            ORDER BY date
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        if results:
            df = pd.DataFrame(results)
            df['not_offloaded'] = df['wh_sent'] - df['shop_offloaded']
            # Store date as string and also as date object for lookup
            if 'date' in df.columns:
                # Keep original date from database
                df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                df['date'] = pd.to_datetime(df['date_str']).dt.date
            return df
        else:
            return pd.DataFrame(columns=['date', 'date_str', 'wh_sent', 'shop_offloaded', 'not_offloaded'])

@st.cache_data(ttl=300)
def get_shop_funnel_by_date(selected_date):
    """Get shop-wise funnel data for a specific date"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Convert date to string
        date_str = selected_date.strftime('%Y-%m-%d') if hasattr(selected_date, 'strftime') else str(selected_date)
        
        query = f"""
            SELECT 
                vc_shop_code as shop_code,
                COUNT(CASE WHEN serial_no IS NOT NULL 
                      AND LENGTH(REGEXP_REPLACE(serial_no, '^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', 'g')) > 0
                      THEN 1 END) as loaded,
                COUNT(CASE WHEN vc_serail_no IS NOT NULL 
                      AND LENGTH(REGEXP_REPLACE(vc_serail_no, '^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', 'g')) > 0
                      THEN 1 END) as offloaded,
                COUNT(CASE WHEN shop_serail_no IS NOT NULL 
                      AND LENGTH(REGEXP_REPLACE(shop_serail_no, '^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', 'g')) > 0
                      THEN 1 END) as sold
            FROM serial_no_dailydata
            WHERE DATE(loaded_datetime) = '{date_str}'
              AND vc_shop_code IS NOT NULL
            GROUP BY vc_shop_code
            HAVING COUNT(*) > 0
            ORDER BY loaded DESC
            LIMIT 15
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        if results:
            df = pd.DataFrame(results)
            return df
        else:
            return pd.DataFrame(columns=['shop_code', 'loaded', 'offloaded', 'sold'])

@st.cache_data(ttl=300)
def get_brand_funnel(start_date, end_date):
    """Get funnel by brand"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Convert dates to strings
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
        
        query = f"""
            SELECT 
                vc_item_code as item_code,
                vc_item_desc as item_name,
                COUNT(CASE WHEN serial_no IS NOT NULL 
                      AND LENGTH(REGEXP_REPLACE(serial_no, '^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', 'g')) > 0
                      THEN 1 END) as loaded,
                COUNT(CASE WHEN vc_serail_no IS NOT NULL 
                      AND LENGTH(REGEXP_REPLACE(vc_serail_no, '^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', 'g')) > 0
                      THEN 1 END) as offloaded,
                COUNT(CASE WHEN shop_serail_no IS NOT NULL 
                      AND LENGTH(REGEXP_REPLACE(shop_serail_no, '^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', 'g')) > 0
                      THEN 1 END) as sold,
                COUNT(CASE WHEN {SQL_ANY_ISSUE} THEN 1 END) as issues
            FROM serial_no_dailydata
            WHERE DATE(loaded_datetime) >= '{start_str}' AND DATE(loaded_datetime) <= '{end_str}'
            GROUP BY vc_item_code, vc_item_desc
            HAVING COUNT(*) > 10
            ORDER BY loaded DESC
            LIMIT 20
        """
        
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=300)
def get_issue_details(start_date, end_date, issue_type=None):
    """Get detailed issue records"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Convert dates to strings
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
        
        # Build issue filter based on issue_type
        issue_filter = SQL_ANY_ISSUE
        if issue_type == 'No offloading':
            issue_filter = SQL_NO_OFFLOADING
        elif issue_type == 'Sold without offloading':
            issue_filter = SQL_SOLD_WITHOUT_OFFLOADING
        elif issue_type == 'Vehicle mismatch':
            issue_filter = "(vc_vehicle_no != \"vc_vehicle_no.1\" AND vc_vehicle_no IS NOT NULL AND \"vc_vehicle_no.1\" IS NOT NULL)"
        elif issue_type == 'Shop mismatch':
            issue_filter = SQL_SHOP_MISMATCH
        elif issue_type == 'Serial mismatch':
            issue_filter = SQL_SERIAL_MISMATCH
        
        query = f"""
            SELECT 
                vc_item_code, vc_item_desc,
                serial_no, vc_serail_no, shop_serail_no,
                -- Show cleaned serial numbers for comparison
                REGEXP_REPLACE(serial_no, '^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', 'g') as serial_no_cleaned,
                REGEXP_REPLACE(COALESCE(vc_serail_no, ''), '^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', 'g') as vc_serail_no_cleaned,
                REGEXP_REPLACE(COALESCE(shop_serail_no, ''), '^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', 'g') as shop_serail_no_cleaned,
                vc_shop_code, shop_sold,
                vc_vehicle_no, "vc_vehicle_no.1" as vc_vehicle_no_2,
                loaded_datetime,
                CASE 
                    WHEN {SQL_NO_OFFLOADING} THEN '⚠️ No offloading'
                    WHEN {SQL_SOLD_WITHOUT_OFFLOADING} THEN '⚠️ Sold without offloading'
                    WHEN (vc_vehicle_no != "vc_vehicle_no.1" AND vc_vehicle_no IS NOT NULL AND "vc_vehicle_no.1" IS NOT NULL) THEN '⚠️ Vehicle mismatch'
                    WHEN {SQL_SHOP_MISMATCH} THEN '⚠️ Shop mismatch'
                    WHEN {SQL_SERIAL_MISMATCH} THEN '⚠️ Serial mismatch'
                    ELSE 'Unknown'
                END as remarks2
            FROM serial_no_dailydata
            WHERE DATE(loaded_datetime) >= '{start_str}' AND DATE(loaded_datetime) <= '{end_str}'
              AND {issue_filter}
            ORDER BY loaded_datetime DESC
        """
        
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=300)
def get_top_issue_shops(start_date, end_date, issue_type=None):
    """Get shops with most issues, optionally filtered by issue type"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Convert dates to strings
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
        
        # Build issue filter
        issue_filter = SQL_ANY_ISSUE
        if issue_type == 'No offloading':
            issue_filter = SQL_NO_OFFLOADING
        elif issue_type == 'Sold without offloading':
            issue_filter = SQL_SOLD_WITHOUT_OFFLOADING
        elif issue_type == 'Vehicle mismatch':
            issue_filter = "(vc_vehicle_no != \"vc_vehicle_no.1\" AND vc_vehicle_no IS NOT NULL AND \"vc_vehicle_no.1\" IS NOT NULL)"
        elif issue_type == 'Shop mismatch':
            issue_filter = SQL_SHOP_MISMATCH
        elif issue_type == 'Serial mismatch':
            issue_filter = SQL_SERIAL_MISMATCH
        
        # Base query
        query = f"""
            SELECT 
                vc_shop_code,
                COUNT(*) as issue_count,
                SUM(CASE WHEN {SQL_NO_OFFLOADING} THEN 1 ELSE 0 END) as no_offloading,
                SUM(CASE WHEN {SQL_SOLD_WITHOUT_OFFLOADING} THEN 1 ELSE 0 END) as sold_without_offload,
                SUM(CASE WHEN {SQL_SHOP_MISMATCH} THEN 1 ELSE 0 END) as shop_mismatch,
                SUM(CASE WHEN {SQL_SERIAL_MISMATCH} THEN 1 ELSE 0 END) as serial_mismatch
            FROM serial_no_dailydata
            WHERE DATE(loaded_datetime) >= '{start_str}' AND DATE(loaded_datetime) <= '{end_str}'
              AND {issue_filter}
              AND vc_shop_code IS NOT NULL
        """
        
        query += """
            GROUP BY vc_shop_code
            ORDER BY issue_count DESC
            LIMIT 10
        """
        
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=300)
def get_shop_issues_by_brand(start_date, end_date, shop_code, issue_type=None):
    """Get issue breakdown by brand for a specific shop"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
        
        # Build issue filter
        issue_filter = SQL_ANY_ISSUE
        if issue_type:
            if 'no offloading' in issue_type.lower():
                issue_filter = SQL_NO_OFFLOADING
            elif 'without offload' in issue_type.lower():
                issue_filter = SQL_SOLD_WITHOUT_OFFLOADING
            elif 'vehicle mismatch' in issue_type.lower():
                issue_filter = "(vc_vehicle_no != \"vc_vehicle_no.1\" AND vc_vehicle_no IS NOT NULL AND \"vc_vehicle_no.1\" IS NOT NULL)"
            elif 'shop mismatch' in issue_type.lower():
                issue_filter = SQL_SHOP_MISMATCH
            elif 'serial mismatch' in issue_type.lower():
                issue_filter = SQL_SERIAL_MISMATCH
        
        query = f"""
            SELECT 
                vc_item_code as item_code,
                vc_item_desc as item_name,
                COUNT(*) as issue_count
            FROM serial_no_dailydata
            WHERE DATE(loaded_datetime) >= '{start_str}' AND DATE(loaded_datetime) <= '{end_str}'
              AND {issue_filter}
              AND vc_shop_code = '{shop_code}'
        """
        
        query += """
            GROUP BY vc_item_code, vc_item_desc
            ORDER BY issue_count DESC
            LIMIT 15
        """
        
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=300)
def get_shop_issues_by_brand_date(start_date, end_date, shop_code, issue_type=None):
    """Get issue breakdown by brand and date for a specific shop, optionally filtered by issue type"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Convert dates to strings
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
        
        # Build issue filter
        issue_filter = SQL_ANY_ISSUE
        if issue_type:
            if 'no offloading' in issue_type.lower():
                issue_filter = SQL_NO_OFFLOADING
            elif 'without offload' in issue_type.lower():
                issue_filter = SQL_SOLD_WITHOUT_OFFLOADING
            elif 'vehicle mismatch' in issue_type.lower():
                issue_filter = "(vc_vehicle_no != \"vc_vehicle_no.1\" AND vc_vehicle_no IS NOT NULL AND \"vc_vehicle_no.1\" IS NOT NULL)"
            elif 'shop mismatch' in issue_type.lower():
                issue_filter = SQL_SHOP_MISMATCH
            elif 'serial mismatch' in issue_type.lower():
                issue_filter = SQL_SERIAL_MISMATCH
        
        query = f"""
            SELECT 
                DATE(loaded_datetime) as load_date,
                vc_item_code as item_code,
                COUNT(*) as issue_count
            FROM serial_no_dailydata
            WHERE DATE(loaded_datetime) >= '{start_str}' AND DATE(loaded_datetime) <= '{end_str}'
              AND {issue_filter}
              AND vc_shop_code = '{shop_code}'
        """
        
        query += """
            GROUP BY DATE(loaded_datetime), vc_item_code
            ORDER BY load_date, issue_count DESC
        """
        
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=300)
def get_issue_details_by_date(start_date, end_date, shop_code, selected_date, issue_type=None):
    """Get detailed issue records for a specific date and shop"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Convert dates to strings
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
        selected_date_str = selected_date.strftime('%Y-%m-%d') if hasattr(selected_date, 'strftime') else str(selected_date)
        
        # Build issue filter
        issue_filter = SQL_ANY_ISSUE
        if issue_type:
            if 'no offloading' in issue_type.lower():
                issue_filter = SQL_NO_OFFLOADING
            elif 'without offload' in issue_type.lower():
                issue_filter = SQL_SOLD_WITHOUT_OFFLOADING
            elif 'vehicle mismatch' in issue_type.lower():
                issue_filter = "(vc_vehicle_no != \"vc_vehicle_no.1\" AND vc_vehicle_no IS NOT NULL AND \"vc_vehicle_no.1\" IS NOT NULL)"
            elif 'shop mismatch' in issue_type.lower():
                issue_filter = SQL_SHOP_MISMATCH
            elif 'serial mismatch' in issue_type.lower():
                issue_filter = SQL_SERIAL_MISMATCH
        
        query = f"""
            SELECT 
                loaded_datetime,
                brand,
                "group",
                subgroup,
                serial_no,
                vc_serail_no,
                shop_serail_no,
                -- Show cleaned serial numbers for comparison
                REGEXP_REPLACE(serial_no, '^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', 'g') as serial_no_cleaned,
                REGEXP_REPLACE(COALESCE(vc_serail_no, ''), '^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', 'g') as vc_serail_no_cleaned,
                REGEXP_REPLACE(COALESCE(shop_serail_no, ''), '^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', 'g') as shop_serail_no_cleaned,
                vc_shop_code,
                shop_sold,
                vc_vehicle_no,
                "vc_vehicle_no.1" as vc_vehicle_no_2,
                CASE 
                    WHEN {SQL_NO_OFFLOADING} THEN '⚠️ No offloading'
                    WHEN {SQL_SOLD_WITHOUT_OFFLOADING} THEN '⚠️ Sold without offloading'
                    WHEN (vc_vehicle_no != "vc_vehicle_no.1" AND vc_vehicle_no IS NOT NULL AND "vc_vehicle_no.1" IS NOT NULL) THEN '⚠️ Vehicle mismatch'
                    WHEN {SQL_SHOP_MISMATCH} THEN '⚠️ Shop mismatch'
                    WHEN {SQL_SERIAL_MISMATCH} THEN '⚠️ Serial mismatch'
                    ELSE 'Unknown'
                END as remarks2
            FROM serial_no_dailydata
            WHERE DATE(loaded_datetime) >= '{start_str}' AND DATE(loaded_datetime) <= '{end_str}'
              AND DATE(loaded_datetime) = '{selected_date_str}'
              AND {issue_filter}
              AND vc_shop_code = '{shop_code}'
        """
        
        query += """
            ORDER BY loaded_datetime DESC, brand
        """
        
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=300)
def get_vehicle_mismatches(start_date, end_date, shop_code=None):
    """Get vehicle number mismatches between warehouse departure and shop arrival"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
        
        query = f"""
            SELECT 
                vc_shop_code,
                vc_item_code,
                vc_item_desc,
                serial_no,
                vc_vehicle_no as warehouse_vehicle,
                "vc_vehicle_no.1" as shop_vehicle,
                loaded_datetime,
                CASE 
                    WHEN vc_vehicle_no != "vc_vehicle_no.1" THEN '⚠️ Vehicle mismatch'
                    ELSE 'OK'
                END as issue_type
            FROM serial_no_dailydata
            WHERE DATE(loaded_datetime) >= '{start_str}' AND DATE(loaded_datetime) <= '{end_str}'
              AND vc_vehicle_no IS NOT NULL 
              AND "vc_vehicle_no.1" IS NOT NULL
              AND TRIM(vc_vehicle_no) != ''
              AND TRIM("vc_vehicle_no.1") != ''
              AND vc_vehicle_no != "vc_vehicle_no.1"
        """
        
        if shop_code and shop_code != 'All':
            query += f" AND vc_shop_code = '{shop_code}'"
        
        query += " ORDER BY loaded_datetime DESC"
        
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=300)
def get_vehicle_mismatch_summary(start_date, end_date):
    """Get summary of vehicle mismatches by shop"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
        
        query = f"""
            SELECT 
                vc_shop_code,
                COUNT(*) as vehicle_mismatch_count,
                COUNT(DISTINCT vc_vehicle_no) as unique_wh_vehicles,
                COUNT(DISTINCT "vc_vehicle_no.1") as unique_shop_vehicles,
                COUNT(DISTINCT vc_item_code) as affected_items
            FROM serial_no_dailydata
            WHERE DATE(loaded_datetime) >= '{start_str}' AND DATE(loaded_datetime) <= '{end_str}'
              AND vc_vehicle_no IS NOT NULL 
              AND "vc_vehicle_no.1" IS NOT NULL
              AND TRIM(vc_vehicle_no) != ''
              AND TRIM("vc_vehicle_no.1") != ''
              AND vc_vehicle_no != "vc_vehicle_no.1"
              AND vc_shop_code IS NOT NULL
            GROUP BY vc_shop_code
            ORDER BY vehicle_mismatch_count DESC
            LIMIT 20
        """
        
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=300)
def get_transit_accuracy_metrics(start_date, end_date):
    """Calculate overall transit accuracy and vehicle tracking metrics"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
        
        query = f"""
            SELECT 
                COUNT(*) as total_with_vehicle_data,
                SUM(CASE WHEN vc_vehicle_no = "vc_vehicle_no.1" THEN 1 ELSE 0 END) as vehicle_match,
                SUM(CASE WHEN vc_vehicle_no != "vc_vehicle_no.1" THEN 1 ELSE 0 END) as vehicle_mismatch,
                SUM(CASE WHEN vc_vehicle_no IS NULL OR TRIM(vc_vehicle_no) = '' THEN 1 ELSE 0 END) as missing_wh_vehicle,
                SUM(CASE WHEN "vc_vehicle_no.1" IS NULL OR TRIM("vc_vehicle_no.1") = '' THEN 1 ELSE 0 END) as missing_shop_vehicle
            FROM serial_no_dailydata
            WHERE DATE(loaded_datetime) >= '{start_str}' AND DATE(loaded_datetime) <= '{end_str}'
              AND (vc_vehicle_no IS NOT NULL OR "vc_vehicle_no.1" IS NOT NULL)
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        return dict(result) if result else {}

@st.cache_data(ttl=300)
def get_root_cause_patterns(start_date, end_date, shop_code='All'):
    """Identify patterns that indicate root causes of issues"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
        
        # Build shop filter
        shop_filter = ""
        if shop_code != 'All':
            shop_filter = f"AND vc_shop_code = '{shop_code}'"
        
        query = f"""
            SELECT 
                vc_shop_code,
                -- Process Issues (No Offloading + Sold Without Offloading)
                SUM(CASE WHEN {SQL_NO_OFFLOADING} THEN 1 ELSE 0 END) as no_offloading,
                SUM(CASE WHEN {SQL_SOLD_WITHOUT_OFFLOADING} THEN 1 ELSE 0 END) as process_bypass,
                -- Data Entry Issues
                SUM(CASE WHEN {SQL_SERIAL_MISMATCH} THEN 1 ELSE 0 END) as data_entry_errors,
                -- Routing Issues
                SUM(CASE WHEN {SQL_SHOP_MISMATCH} THEN 1 ELSE 0 END) as routing_errors,
                -- Vehicle Issues (count separately, not part of main total to avoid double-counting)
                SUM(CASE WHEN vc_vehicle_no != "vc_vehicle_no.1" 
                    AND vc_vehicle_no IS NOT NULL 
                    AND "vc_vehicle_no.1" IS NOT NULL 
                    THEN 1 ELSE 0 END) as vehicle_mismatches,
                -- Total Issues (all four main categories, not vehicle)
                SUM(CASE WHEN {SQL_ANY_ISSUE} THEN 1 ELSE 0 END) as total_issues
            FROM serial_no_dailydata
            WHERE DATE(loaded_datetime) >= '{start_str}' AND DATE(loaded_datetime) <= '{end_str}'
              AND vc_shop_code IS NOT NULL
              {shop_filter}
            GROUP BY vc_shop_code
            HAVING SUM(CASE WHEN {SQL_ANY_ISSUE} THEN 1 ELSE 0 END) > 0
            ORDER BY total_issues DESC
            LIMIT 15
        """
        
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

# ============================================================
# SERIAL JOURNEY TRACKING FUNCTIONS
# ============================================================
@st.cache_data(ttl=300)
def get_serial_journey_overview(start_date, end_date):
    """Get overview metrics for serial number journey across all 3 tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        query = f"""
            WITH wh_received AS (
                SELECT COUNT(DISTINCT serial_no) as received_count
                FROM whreceived_serialno
                WHERE grn_date >= '{start_str}' AND grn_date <= '{end_str}'
            ),
            loaded_data AS (
                SELECT 
                    COUNT(DISTINCT serial_no) as loaded_count,
                    COUNT(DISTINCT CASE WHEN vc_serail_no IS NOT NULL AND vc_serail_no != '' THEN serial_no END) as offloaded_count,
                    COUNT(DISTINCT CASE WHEN shop_serail_no IS NOT NULL AND shop_serail_no != '' THEN serial_no END) as sold_count,
                    COUNT(DISTINCT vc_shop_code) as shops_sent_to,
                    COUNT(DISTINCT CASE WHEN vc_shop_code != shop_sold THEN serial_no END) as shop_mismatch_count,
                    COUNT(DISTINCT CASE WHEN vc_vehicle_no != "vc_vehicle_no.1" THEN serial_no END) as vehicle_mismatch_count
                FROM serial_no_dailydata
                WHERE DATE(loaded_datetime) >= '{start_str}' AND DATE(loaded_datetime) <= '{end_str}'
            ),
            db_check AS (
                SELECT 
                    COUNT(*) as total_checked,
                    SUM(CASE WHEN UPPER(serial_check) = 'Y' THEN 1 ELSE 0 END) as in_main_db,
                    SUM(CASE WHEN UPPER(serial_check) != 'Y' THEN 1 ELSE 0 END) as not_in_main_db
                FROM serialno_check_yes_no
                WHERE bill_date >= '{start_str}' AND bill_date <= '{end_str}'
            )
            SELECT 
                w.received_count,
                l.loaded_count,
                l.offloaded_count,
                l.sold_count,
                l.shops_sent_to,
                l.shop_mismatch_count,
                l.vehicle_mismatch_count,
                d.total_checked,
                d.in_main_db,
                d.not_in_main_db,
                (w.received_count - l.loaded_count) as not_loaded,
                (l.loaded_count - l.offloaded_count) as not_offloaded,
                (l.offloaded_count - l.sold_count) as not_sold
            FROM wh_received w
            CROSS JOIN loaded_data l
            CROSS JOIN db_check d
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        return dict(result) if result else {}

@st.cache_data(ttl=300)
def search_serial_number(serial_no):
    """Get complete journey of a specific serial number"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = f"""
            WITH serial_data AS (
                SELECT 
                    -- WH Received Info
                    wr.grn_date as wh_grn_date,
                    wr.warehouse_name as wh_received_warehouse,
                    wr.supp_name as supplier_name,
                    wr.inbound_type,
                    
                    -- Daily Data Info
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
                    COALESCE(sd.dt_invoice_date, sc.bill_date) as sale_date,
                    
                    -- DB Check Info
                    sc.bill_date,
                    sc.bill_no,
                    sc.shop_code as sold_at_shop,
                    sc.till_number,
                    sc.cashier_name,
                    sc.serial_check as in_main_db,
                    
                    -- Calculated Flags
                    CASE 
                        WHEN sd.vc_shop_code IS NULL THEN 'Not Loaded to Shop'
                        WHEN sd.vc_serail_no IS NULL OR sd.vc_serail_no = '' THEN 'Not Offloaded'
                        WHEN sd.shop_serail_no IS NULL OR sd.shop_serail_no = '' THEN 'Not Sold'
                        ELSE 'Sold'
                    END as status,
                    
                    CASE WHEN sd.vc_shop_code != sd.shop_sold THEN 'YES' ELSE 'NO' END as has_shop_mismatch,
                    CASE WHEN sd.vc_vehicle_no != sd."vc_vehicle_no.1" THEN 'YES' ELSE 'NO' END as has_vehicle_mismatch,
                    CASE WHEN UPPER(sc.serial_check) = 'Y' THEN 'YES' ELSE 'NO' END as verified_in_main_db,
                    CASE 
                        WHEN sc.serial_number IS NOT NULL THEN 'YES'
                        ELSE 'NO'
                    END as sold_status
                    
                FROM whreceived_serialno wr
                FULL OUTER JOIN serial_no_dailydata sd ON UPPER(TRIM(wr.serial_no)) = UPPER(TRIM(sd.serial_no))
                FULL OUTER JOIN serialno_check_yes_no sc ON UPPER(TRIM(COALESCE(sd.shop_serail_no, sd.vc_serail_no, sd.serial_no))) = UPPER(TRIM(sc.serial_number))
                WHERE UPPER(TRIM(COALESCE(wr.serial_no, sd.serial_no, sc.serial_number))) = UPPER(TRIM('{serial_no}'))
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
                STRING_AGG(DISTINCT bill_no, ', ') as bill_no,
                STRING_AGG(DISTINCT sold_at_shop, ', ') as sold_at_shop,
                STRING_AGG(DISTINCT till_number, ', ') as till_number,
                STRING_AGG(DISTINCT cashier_name, ', ') as cashier_name,
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
        
        cursor.execute(query)
        results = cursor.fetchall()
        return pd.DataFrame(results) if results else pd.DataFrame()

@st.cache_data(ttl=300)
def get_gap_analysis_by_shop(start_date, end_date):
    """Get gap analysis showing received → loaded → offloaded → sold by shop"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        query = f"""
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
            WHERE DATE(loaded_datetime) >= '{start_str}' AND DATE(loaded_datetime) <= '{end_str}'
              AND vc_shop_code IS NOT NULL
            GROUP BY vc_shop_code, shop_name
            ORDER BY loaded_count DESC
        """
        
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=300)
def get_mismatch_details(start_date, end_date, mismatch_type, shop_code=None):
    """Get detailed records for shop or vehicle mismatches"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        shop_filter = f"AND vc_shop_code = '{shop_code}'" if shop_code else ""
        
        if mismatch_type == 'shop':
            condition = "vc_shop_code != shop_sold AND vc_shop_code IS NOT NULL AND shop_sold IS NOT NULL"
        else:  # vehicle
            condition = 'vc_vehicle_no != "vc_vehicle_no.1" AND vc_vehicle_no IS NOT NULL AND "vc_vehicle_no.1" IS NOT NULL'
        
        query = f"""
            SELECT 
                serial_no,
                vc_item_code,
                vc_item_desc,
                loaded_datetime,
                vc_shop_code as sent_to_shop,
                shop_name as sent_to_shop_name,
                vc_serail_no as offload_serial,
                shop_sold,
                shop_serail_no as sold_serial,
                vc_vehicle_no as wh_vehicle,
                "vc_vehicle_no.1" as shop_vehicle,
                dt_invoice_date as sale_date
            FROM serial_no_dailydata
            WHERE DATE(loaded_datetime) >= '{start_str}' AND DATE(loaded_datetime) <= '{end_str}'
              AND {condition}
              {shop_filter}
            ORDER BY loaded_datetime DESC
            LIMIT 1000
        """
        
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=300)
def get_supplier_analysis(start_date, end_date):
    """Get supplier-wise serial performance analysis"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        query = f"""
            SELECT 
                COALESCE(wr.supp_name, 'Unknown Supplier') as supplier_name,
                COUNT(DISTINCT sd.serial_no) as total_serials,
                COUNT(DISTINCT CASE WHEN sd.vc_serail_no IS NOT NULL AND sd.vc_serail_no != '' THEN sd.serial_no END) as offloaded_count,
                COUNT(DISTINCT CASE WHEN sd.shop_serail_no IS NOT NULL AND sd.shop_serail_no != '' THEN sd.serial_no END) as sold_count,
                COUNT(DISTINCT CASE WHEN sd.vc_shop_code != sd.shop_sold THEN sd.serial_no END) as shop_mismatches,
                COUNT(DISTINCT CASE WHEN sd.vc_vehicle_no != sd."vc_vehicle_no.1" THEN sd.serial_no END) as vehicle_mismatches,
                ROUND(COUNT(DISTINCT CASE WHEN sd.vc_serail_no IS NOT NULL THEN sd.serial_no END)::numeric / NULLIF(COUNT(DISTINCT sd.serial_no), 0) * 100, 2) as offload_rate,
                ROUND(COUNT(DISTINCT CASE WHEN sd.shop_serail_no IS NOT NULL THEN sd.serial_no END)::numeric / NULLIF(COUNT(DISTINCT sd.serial_no), 0) * 100, 2) as sale_rate
            FROM serial_no_dailydata sd
            LEFT JOIN whreceived_serialno wr ON UPPER(TRIM(sd.serial_no)) = UPPER(TRIM(wr.serial_no))
            WHERE DATE(sd.loaded_datetime) >= '{start_str}' AND DATE(sd.loaded_datetime) <= '{end_str}'
            GROUP BY COALESCE(wr.supp_name, 'Unknown Supplier')
            ORDER BY total_serials DESC
        """
        
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=300)
def get_brand_analysis(start_date, end_date):
    """Get brand-wise serial performance analysis"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        query = f"""
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
            WHERE DATE(loaded_datetime) >= '{start_str}' AND DATE(loaded_datetime) <= '{end_str}'
              AND vc_item_desc IS NOT NULL
            GROUP BY SPLIT_PART(vc_item_desc, ' ', 1)
            HAVING COUNT(DISTINCT serial_no) >= 5
            ORDER BY total_serials DESC
        """
        
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=300)
def get_shop_serial_details(start_date, end_date, shop_code):
    """Get detailed serial journey for a specific shop with timestamps and personnel - OPTIMIZED"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Optimized query - removed expensive LEFT JOINs, made them subqueries
        query = f"""
            SELECT 
                sd.serial_no as wh_serial,
                sd.vc_item_code,
                sd.vc_item_desc,
                sd.nu_selling_price,
                
                -- Loading Info (direct from serial_no_dailydata)
                sd.dt_doc_date as doc_date,
                sd.loaded_datetime as when_loaded,
                sd.vc_vehicle_no as loading_vehicle,
                sd.vc_shop_code as destination_shop,
                sd.shop_name as destination_shop_name,
                
                -- Offloading Info
                sd.vc_serail_no as offload_serial,
                sd."vc_vehicle_no.1" as offload_vehicle,
                CASE 
                    WHEN sd.vc_serail_no IS NOT NULL AND sd.vc_serail_no != '' 
                    THEN sd.loaded_datetime 
                END as when_offloaded,
                
                -- Sale Info
                sd.shop_serail_no as sold_serial,
                sd.shop_sold as selling_shop,
                sd.dt_invoice_date as when_sold,
                sd.vc_invoice_no as invoice_no,
                
                -- Status Flags (from serial_no_dailydata only)
                sd.remarks2 as issue_category,
                CASE WHEN sd.vc_shop_code != sd.shop_sold THEN 'YES' ELSE 'NO' END as shop_mismatch,
                CASE WHEN sd.vc_vehicle_no != sd."vc_vehicle_no.1" THEN 'YES' ELSE 'NO' END as vehicle_mismatch
                
            FROM serial_no_dailydata sd
            WHERE DATE(sd.loaded_datetime) >= '{start_str}' AND DATE(sd.loaded_datetime) <= '{end_str}'
              AND sd.vc_shop_code = '{shop_code}'
            ORDER BY sd.loaded_datetime DESC
        """
        
        cursor.execute(query)
        result_df = pd.DataFrame(cursor.fetchall())
        
        # If we have results, enrich with WH and personnel data in separate queries
        if not result_df.empty and len(result_df) > 0:
            # Get WH data for unique serials (batch lookup) - NO LIMIT, process all
            unique_serials = result_df['wh_serial'].dropna().unique().tolist()
            if unique_serials:
                # Process in batches of 1000 to avoid SQL query length limits
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
                    cursor.execute(wh_query)
                    batch_df = pd.DataFrame(cursor.fetchall())
                    if not batch_df.empty:
                        wh_dfs.append(batch_df)
                
                if wh_dfs:
                    wh_df = pd.concat(wh_dfs, ignore_index=True)
                    # Merge WH data
                    result_df = result_df.merge(wh_df, left_on='wh_serial', right_on='serial_no', how='left')
                    result_df.drop('serial_no', axis=1, inplace=True, errors='ignore')
            
            # Get personnel data for ALL serials (not just sold ones) - check against all serial columns
            all_serials = []
            for col in ['wh_serial', 'offload_serial', 'sold_serial']:
                if col in result_df.columns:
                    all_serials.extend(result_df[col].dropna().unique().tolist())
            
            # Remove duplicates
            all_serials = list(set(all_serials))
            
            if all_serials:
                # Process in batches of 1000 to avoid SQL query length limits - NO LIMIT, get all
                batch_size = 1000
                personnel_dfs = []
                
                for i in range(0, len(all_serials), batch_size):
                    batch = all_serials[i:i + batch_size]
                    serial_list = "', '".join([str(s).replace("'", "''") for s in batch])
                    
                    personnel_query = f"""
                        SELECT 
                            serial_number,
                            cashier_name,
                            till_number,
                            bill_no,
                            serial_check
                        FROM serialnocheck_indb
                        WHERE UPPER(TRIM(serial_number)) IN ('{serial_list}')
                    """
                    cursor.execute(personnel_query)
                    batch_df = pd.DataFrame(cursor.fetchall())
                    if not batch_df.empty:
                        personnel_dfs.append(batch_df)
                
                personnel_df = pd.concat(personnel_dfs, ignore_index=True) if personnel_dfs else pd.DataFrame()
                
                if not personnel_df.empty:
                    # Try matching against all three serial columns
                    # First try sold_serial (most likely to match)
                    if 'sold_serial' in result_df.columns:
                        result_df = result_df.merge(
                            personnel_df, 
                            left_on='sold_serial', 
                            right_on='serial_number', 
                            how='left'
                        )
                        result_df.drop('serial_number', axis=1, inplace=True, errors='ignore')
                    
                    # If still missing, try offload_serial
                    if 'serial_check' not in result_df.columns and 'offload_serial' in result_df.columns:
                        result_df = result_df.merge(
                            personnel_df, 
                            left_on='offload_serial', 
                            right_on='serial_number', 
                            how='left'
                        )
                        result_df.drop('serial_number', axis=1, inplace=True, errors='ignore')
                    
                    # If still missing, try wh_serial
                    if 'serial_check' not in result_df.columns:
                        result_df = result_df.merge(
                            personnel_df, 
                            left_on='wh_serial', 
                            right_on='serial_number', 
                            how='left'
                        )
                        result_df.drop('serial_number', axis=1, inplace=True, errors='ignore')
        
        return result_df

@st.cache_data(ttl=300)
def get_shop_offloading_performance(start_date, end_date):
    """Get shop-wise offloading performance (worst performers by offloading %)"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        query = f"""
            SELECT 
                vc_shop_code,
                shop_name,
                COUNT(DISTINCT serial_no) as loaded_count,
                COUNT(DISTINCT CASE WHEN vc_serail_no IS NOT NULL AND vc_serail_no != '' THEN serial_no END) as offloaded_count,
                ROUND(
                    COUNT(DISTINCT CASE WHEN vc_serail_no IS NOT NULL AND vc_serail_no != '' THEN serial_no END)::numeric / 
                    NULLIF(COUNT(DISTINCT serial_no), 0) * 100, 
                    1
                ) as offloading_pct
            FROM serial_no_dailydata
            WHERE DATE(loaded_datetime) >= '{start_str}' AND DATE(loaded_datetime) <= '{end_str}'
              AND vc_shop_code IS NOT NULL
            GROUP BY vc_shop_code, shop_name
            HAVING COUNT(DISTINCT serial_no) >= 10
            ORDER BY offloading_pct ASC NULLS FIRST, loaded_count DESC
            LIMIT 10
        """
        
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=300)
def get_management_summary(start_date, end_date):
    """Get executive summary with actionable insights - includes serial check analysis"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        query = f"""
            WITH base_data AS (
                SELECT 
                    sd.*,
                    wr.supp_name,
                    SPLIT_PART(sd.vc_item_desc, ' ', 1) as brand
                FROM serial_no_dailydata sd
                LEFT JOIN whreceived_serialno wr ON UPPER(TRIM(sd.serial_no)) = UPPER(TRIM(wr.serial_no))
                WHERE DATE(sd.loaded_datetime) >= '{start_str}' AND DATE(sd.loaded_datetime) <= '{end_str}'
            ),
            serial_check_data AS (
                SELECT 
                    COUNT(*) as total_serials_checked,
                    COUNT(CASE WHEN UPPER(TRIM(serial_check)) = 'N' OR serial_check IS NULL OR TRIM(serial_check) = '' THEN 1 END) as serial_check_n_count,
                    COUNT(CASE WHEN UPPER(TRIM(serial_check)) = 'Y' THEN 1 END) as serial_check_y_count,
                    ROUND(
                        COUNT(CASE WHEN UPPER(TRIM(serial_check)) = 'N' OR serial_check IS NULL OR TRIM(serial_check) = '' THEN 1 END)::numeric / 
                        NULLIF(COUNT(*), 0) * 100, 
                        1
                    ) as serial_check_n_pct
                FROM serialno_check_yes_no
                WHERE bill_date >= '{start_str}' AND bill_date <= '{end_str}'
            )
            SELECT 
                -- Top Issue Categories
                b.remarks2 as issue_category,
                COUNT(*) as issue_count,
                ROUND(COUNT(*)::numeric / SUM(COUNT(*)) OVER () * 100, 2) as issue_percentage,
                
                -- Top Affected Shops
                (SELECT STRING_AGG(DISTINCT vc_shop_code, ', ') 
                 FROM (SELECT vc_shop_code FROM base_data WHERE remarks2 = b.remarks2 
                       ORDER BY vc_shop_code LIMIT 5) sub) as top_shops,
                
                -- Top Affected Brands
                (SELECT STRING_AGG(DISTINCT brand, ', ') 
                 FROM (SELECT brand FROM base_data WHERE remarks2 = b.remarks2 
                       AND brand IS NOT NULL ORDER BY brand LIMIT 5) sub) as top_brands,
                
                -- Avg Days to Resolution
                AVG(CASE 
                    WHEN b.dt_invoice_date IS NOT NULL AND b.loaded_datetime IS NOT NULL 
                    THEN EXTRACT(DAY FROM (b.dt_invoice_date - b.loaded_datetime))
                END) as avg_days_to_sale,
                
                -- Serial Check Analysis (same for all rows, aggregated later)
                MAX(sc.total_serials_checked) as total_serials_checked,
                MAX(sc.serial_check_n_count) as serial_check_n_count,
                MAX(sc.serial_check_y_count) as serial_check_y_count,
                MAX(sc.serial_check_n_pct) as serial_check_n_pct
                
            FROM base_data b
            CROSS JOIN serial_check_data sc
            GROUP BY b.remarks2
            ORDER BY issue_count DESC
        """
        
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

@st.cache_data(ttl=300)
def get_wh_data_quality_kpis(start_date, end_date):
    """Get WH serial number data quality KPIs from whreceived_serialno table"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        query = f"""
            WITH wh_data AS (
                SELECT 
                    serial_no,
                    grn_date
                FROM whreceived_serialno
                WHERE grn_date >= '{start_str}' AND grn_date <= '{end_str}'
            ),
            quality_checks AS (
                SELECT 
                    COUNT(*) as total_serials,
                    COUNT(DISTINCT serial_no) as unique_serials,
                    COUNT(*) - COUNT(DISTINCT serial_no) as duplicate_count,
                    COUNT(CASE WHEN serial_no IS NULL OR TRIM(serial_no) = '' THEN 1 END) as blank_serials,
                    COUNT(CASE WHEN LENGTH(REGEXP_REPLACE(serial_no, '[^a-zA-Z0-9]', '', 'g')) = 1 THEN 1 END) as single_digit,
                    COUNT(CASE WHEN LENGTH(REGEXP_REPLACE(serial_no, '[^a-zA-Z0-9]', '', 'g')) <= 3 THEN 1 END) as short_serials,
                    COUNT(CASE WHEN serial_no ~ '^[0-9]+\$' AND LENGTH(serial_no) <= 5 THEN 1 END) as numeric_only_short,
                    COUNT(CASE WHEN serial_no LIKE '%TEST%' OR serial_no LIKE '%DUMMY%' OR serial_no LIKE '%XXX%' THEN 1 END) as test_serials,
                    COUNT(CASE WHEN LENGTH(serial_no) > 50 THEN 1 END) as too_long_serials
                FROM wh_data
            ),
            duplicate_details AS (
                SELECT 
                    serial_no,
                    COUNT(*) as occurrence_count
                FROM wh_data
                WHERE serial_no IS NOT NULL AND TRIM(serial_no) != ''
                GROUP BY serial_no
                HAVING COUNT(*) > 1
                ORDER BY COUNT(*) DESC
                LIMIT 10
            )
            SELECT 
                qc.*,
                COALESCE(ARRAY_AGG(dd.serial_no || ' (' || dd.occurrence_count || 'x)'), ARRAY[]::text[]) as top_duplicates
            FROM quality_checks qc
            LEFT JOIN duplicate_details dd ON true
            GROUP BY qc.total_serials, qc.unique_serials, qc.duplicate_count, qc.blank_serials,
                     qc.single_digit, qc.short_serials, qc.numeric_only_short, qc.test_serials, qc.too_long_serials
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        return dict(result) if result else {}

@st.cache_data(ttl=300)
def get_serial_check_by_shop(start_date, end_date):
    """Get serial check match rate by shop - directly from serialno_check_yes_no table"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        query = f"""
            SELECT 
                shop_code,
                COUNT(*) as total_serials,
                COUNT(CASE WHEN UPPER(TRIM(serial_check)) = 'Y' THEN 1 END) as matched_serials,
                COUNT(CASE WHEN UPPER(TRIM(serial_check)) = 'N' OR serial_check IS NULL OR TRIM(serial_check) = '' THEN 1 END) as unmatched_serials,
                ROUND(
                    COUNT(CASE WHEN UPPER(TRIM(serial_check)) = 'Y' THEN 1 END)::numeric / 
                    NULLIF(COUNT(*), 0) * 100, 
                    1
                ) as match_rate_pct
            FROM serialno_check_yes_no
            WHERE bill_date >= '{start_str}' AND bill_date <= '{end_str}'
              AND shop_code IS NOT NULL
              AND TRIM(shop_code) != ''
            GROUP BY shop_code
            ORDER BY match_rate_pct ASC NULLS FIRST, total_serials DESC
        """
        
        cursor.execute(query)
        df = pd.DataFrame(cursor.fetchall())
        
        # Convert match_rate_pct to float to fix dtype issues
        if not df.empty and 'match_rate_pct' in df.columns:
            df['match_rate_pct'] = df['match_rate_pct'].astype(float)
        
        return df

@st.cache_data(ttl=300)
def get_unverified_serials(start_date, end_date):
    """Get serial numbers not found in main DB"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        query = f"""
            SELECT 
                serial_number,
                item_code,
                item_name,
                shop_code,
                bill_no,
                bill_date,
                till_number,
                cashier_name,
                serial_check
            FROM serialno_check_yes_no
            WHERE bill_date >= '{start_str}' AND bill_date <= '{end_str}'
              AND (UPPER(TRIM(serial_check)) = 'N' OR serial_check IS NULL OR TRIM(serial_check) = '')
            ORDER BY bill_date DESC
            LIMIT 1000
        """
        
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())

# ============================================================
# MAIN DASHBOARD
# ============================================================
def main():
    """Main dashboard function"""
    
    # ============================================================
    # PAGE CONFIGURATION (must be first Streamlit command)
    # ============================================================
    st.set_page_config(
        page_title="Serial Number Funnel",
        page_icon=Config.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # ============================================================
    # CHECK AUTHENTICATION
    # ============================================================
    if not st.session_state.logged_in:
        show_login_page()
        return
    
    # ============================================================
    # SIDEBAR - USER INFO & LOGOUT
    # ============================================================
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 👤 User Info")
        
        st.write(f"**Name:** {st.session_state.user['full_name']}")
        st.write(f"**ID:** {st.session_state.user['employee_id']}")
        
        if st.button("🚪 Logout", use_container_width=True, type="primary"):
            user_id = st.session_state.user['employee_id'] if st.session_state.user else 'Unknown'
            st.session_state.logged_in = False
            st.session_state.user = None
            logger.info(f"User logged out: {user_id}")
            st.rerun()
        st.markdown("---")
    
    # Get latest data date
    latest_date = get_latest_data_date()
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; flex-direction: column; align-items: center; text-align: center; width: 100%;">
            <div style="display: flex; align-items: center; gap: 15px; justify-content: center;">
                <img src="{Config.LOGO_URL}" style="width: 50px; height: 50px; border-radius: 8px;">
                <div>
                    <h1 style="margin: 0;">📦 Serial Number  Funnel</h1>
                    <p style="margin: 5px 0 0 0; font-size: 1.2rem;">Track item flow from Warehouse → Shop → Customer with drop-off analysis</p>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <span style="font-size: 1rem; color: rgba(255,255,255,0.8);">Updated Till: </span>
                <span style="font-size: 1.2rem; font-weight: bold;">{latest_date}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Date Selection in Sidebar
    with st.sidebar:
        st.markdown("### 📅 Date Range Selection")
        start_date = st.date_input(
            "Start Date",
            value=datetime.today() - timedelta(days=30),
            max_value=datetime.today()
        )
        
        end_date = st.date_input(
            "End Date",
            value=datetime.today(),
            max_value=datetime.today()
        )
        
        st.markdown("---")
        
        # Show available date range info
        date_range_info = get_available_date_range()
        if date_range_info and date_range_info['total_records'] > 0:
            st.success(f"""
            📅 **Data Available:**  
            {date_range_info['min_date'].strftime('%d-%b-%Y')} to {date_range_info['max_date'].strftime('%d-%b-%Y')}  
            📊 {date_range_info['total_records']:,} records
            """)
        
        st.info(f"📊 Analyzing {(end_date - start_date).days + 1} days of data")
    
    # Check available date range
    date_range_info = get_available_date_range()
    
    # Get data
    funnel_data = get_funnel_data(start_date, end_date)
    
    # Check if data exists for selected date range
    total_records = funnel_data.get('total_records', 0)
    
    if total_records == 0:
        # Show warning with available date range
        if date_range_info and date_range_info['total_records'] > 0:
            st.error(f"""
            ⚠️ **No data available for the selected date range ({start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')})**
            
            📅 **Available Data Range:**
            - **From:** {date_range_info['min_date'].strftime('%d-%b-%Y')}
            - **To:** {date_range_info['max_date'].strftime('%d-%b-%Y')}
            - **Total Records:** {date_range_info['total_records']:,}
            
            Please select dates within the available range.
            """)
        else:
            st.error("⚠️ **No data available in the database.** Please load data first.")
        st.stop()
    
    # Calculate drop-offs with WH received as starting point
    wh_received = funnel_data['stage0_wh_received']
    loaded = funnel_data['stage1_loaded']
    offloaded = funnel_data['stage2_offloaded']
    sold = funnel_data['stage3_sold']
    
    dropoff_0_1 = wh_received - loaded  # Received but not loaded
    dropoff_1_2 = loaded - offloaded     # Loaded but not offloaded
    dropoff_2_3 = offloaded - sold       # Offloaded but not sold
    
    dropoff_0_1_pct = (dropoff_0_1 / wh_received * 100) if wh_received > 0 else 0
    dropoff_1_2_pct = (dropoff_1_2 / loaded * 100) if loaded > 0 else 0
    dropoff_2_3_pct = (dropoff_2_3 / offloaded * 100) if offloaded > 0 else 0
    
    conversion_loaded = (loaded / wh_received * 100) if wh_received > 0 else 0
    conversion_offload = (offloaded / loaded * 100) if loaded > 0 else 0
    conversion_sold = (sold / offloaded * 100) if offloaded > 0 else 0
    conversion_overall = (sold / wh_received * 100) if wh_received > 0 else 0
    
    # Get top issue shops (with date range)
    top_shops = get_top_issue_shops(start_date, end_date)
    
    # Initialize session state for issue type selection
    if 'selected_issue_type' not in st.session_state:
        st.session_state.selected_issue_type = None
    
    # ============================================================
    # DASHBOARD FLOW GUIDE
    # ============================================================
    with st.expander("🗺️ Dashboard Navigation Guide - How to Use This Dashboard", expanded=False):
        st.markdown("""
        ### 📊 3-Step Analysis Workflow
        
        ---
        
        #### 📊 STEP 1: Overview Tab - High-Level Summary
        
        **Purpose:** Get a bird's-eye view of serial number flow
        
        **Key Metrics:**
        - 📦 WH Received → 🚚 Loaded → 🚛 Offloaded → 🛒 Sold (funnel stages)
        - Conversion rates at each stage
        - Issue distribution by type and shops
        - **🏭 WH Data Quality KPIs (NEW!)** - Blanks, duplicates, invalid serials
        
        **What to Look For:**
        - 🔴 Red flags: Low conversion rates (<70%)
        - 📈 Biggest drop-offs between stages
        - 🏪 Shops with most issues (click pie chart to filter)
        - ⚠️ Data quality issues in WH serials
        
        **Next Action:** Click on shop/issue in pie chart → Auto-filter Tab 2
        
        ---
        
        #### 🏪 STEP 2: Shop Drill-Down Tab - Root Cause Analysis
        
        **Purpose:** Deep-dive into shop-specific issues
        
        **Two-Part Analysis:**
        
        1. **Serial Check Verification (NEW!)**
           - 📊 Overall match rate (serials verified in main DB)
           - 🔴 Bottom 5 shops: Lowest match rates (action required)
           - 🟢 Top 5 shops: Highest match rates (best practices)
           - 📋 Complete shop comparison table
        
        2. **Shop Issue Drill-Down**
           - Select shop from dropdown → See detailed issues
           - View brand distribution, item analysis, root causes
           - Transit accuracy, vehicle mismatches
        
        **What to Look For:**
        - 🔴 Shops with <50% match rate → URGENT investigation needed
        - ⚠️ Recurring issue types (process vs data entry problems)
        - 🏷️ Specific brands/items causing issues
        
        **Next Action:** Select specific shop → Go to Tab 3 for serial-level tracking
        
        ---
        
        #### 🔍 STEP 3: Serial Journey Tab - Serial-Level Tracking
        
        **Purpose:** Track individual serials + Management insights
        
        **Four-Part Analysis:**
        
        1. **Search Serial Number:** Complete journey from WH to sale
        
        2. **Management Analysis (NEW!)**
           - 📊 Executive summary with severity levels
           - 🎯 Immediate/Short-term/Long-term action plans
           - 📥 Download executive summary
        
        3. **Supplier & Brand Analysis (NEW!)**
           - 🏭 Supplier performance comparison
           - 🏷️ Brand tracking efficiency
           - 📈 Dual-axis charts (volume vs sale rate)
        
        4. **Shop Serial Journey (ENHANCED!)**
           - Select shop → See complete serial history (500 records)
           - 👤 Personnel tracking: Cashier, Till, Timestamps
           - ⚠️ Filter by issue, supplier, DB status
           - 📊 Quick stats per shop
        
        **What to Look For:**
        - 🔴 Serials stuck at specific stages
        - 👤 Personnel accountability (who handled the serial)
        - ⏱️ Time gaps between stages
        - 🏭 Supplier/brand patterns
        
        **Download:** Complete serial journey CSV with all timestamps + personnel
        
        ---
        
        """)
        
        st.markdown("""
        ### 💡 Quick Tips for Management
        
        **Daily Monitoring:**
        1. Start with Tab 1 → Check overall conversion rate (target: >80%)
        2. Check WH Data Quality KPIs → Identify blank/duplicate serials
        3. Go to Tab 2 → Review serial check verification (identify shops <50% match rate)
        4. Review Tab 2 shop drill-down for recurring issues
        
        **Weekly Review:**
        1. Tab 3 → Management Analysis → Review top 3 issues with action plans
        2. Tab 3 → Supplier/Brand Analysis → Identify underperformers
        3. Download executive summary → Share with ops team
        
        **Issue Investigation:**
        1. Tab 1 → Click on issue type in pie chart
        2. Tab 2 → Auto-filtered to issue + Shop drill-down
        3. Tab 3 → Search specific serial → View complete journey with personnel
        
        **Best Practice Identification:**
        1. Tab 2 → Review top 5 shops with highest match rates
        2. Interview those shop managers
        3. Document and share processes with low-performing shops
        
        ---
        
        ### 📥 Available Downloads
        - **Tab 1:** WH Data Quality Report (blanks, duplicates, invalid serials)
        - **Tab 2:** Serial check analysis CSV (shop-wise match rates)
        - **Tab 2:** Shop issue reports (filtered by shop/issue)
        - **Tab 3:** Executive summary (management insights)
        - **Tab 3:** Supplier/brand analysis
        - **Tab 3:** Complete serial journey (with timestamps + personnel)
        """)
    
    # ============================================================
    # TABS
    # ============================================================
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🏪 Shop Drill-down", "🔍 Serial Journey"])
    
    # ============================================================
    # TAB 1: OVERVIEW
    # ============================================================
    with tab1:
        # Custom CSS for compact metrics
        st.markdown("""
        <style>
            div[data-testid="stMetricValue"] {
                font-size: 1.3rem !important;
            }
            div[data-testid="stMetricLabel"] {
                font-size: 0.75rem !important;
            }
            div[data-testid="stMetricDelta"] {
                font-size: 0.7rem !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # ============================================================
        # TOP ROW: 4-STAGE METRICS (WH Received → Loaded → Offloaded → Sold)
        # ============================================================
        st.markdown("### 📊 Serial Number Flow")
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric(
                "📦 WH Received",
                f"{wh_received:,}",
                help="Items received at warehouse from suppliers"
            )
        
        with m2:
            # Loaded vs WH Received (can exceed 100% if same serial sent to multiple shops)
            loaded_vs_received = (loaded / wh_received * 100) if wh_received > 0 else 0
            st.metric(
                "🏭 Loaded at WH",
                f"{loaded:,}",
                f"{loaded_vs_received:.1f}%" if wh_received > 0 else "0%",
                help="Items loaded at warehouse for distribution (% vs WH Received)"
            )
        
        with m3:
            # Offloaded vs Loaded
            offloaded_vs_loaded = (offloaded / loaded * 100) if loaded > 0 else 0
            st.metric(
                "🚚 Offloaded to Shop",
                f"{offloaded:,}",
                f"{offloaded_vs_loaded:.1f}%" if loaded > 0 else "0%",
                help="Items offloaded to shops (% vs Loaded)"
            )
        
        with m4:
            # Sold vs WH Received (overall conversion)
            sold_vs_received = (sold / wh_received * 100) if wh_received > 0 else 0
            st.metric(
                "✅ Sold to Customer",
                f"{sold:,}",
                f"{sold_vs_received:.1f}%" if wh_received > 0 else "0%",
                help="Items sold to customers (% vs WH Received - overall conversion)"
            )
        
        st.markdown("---")
        
        # ============================================================
        # WH DATA QUALITY KPIs
        # ============================================================
        st.markdown("### 🏭 Warehouse Serial Number Data Quality")
        st.info("💡 Analysis of serial number quality in warehouse receipts - Identifying blanks, duplicates, and invalid entries")
        
        wh_quality = get_wh_data_quality_kpis(start_date, end_date)
        
        if wh_quality and wh_quality.get('total_serials', 0) > 0:
            # Quality metrics row
            col_q1, col_q2, col_q3, col_q4, col_q5 = st.columns(5)
            
            total_serials = wh_quality['total_serials']
            unique_serials = wh_quality['unique_serials']
            duplicate_count = wh_quality['duplicate_count']
            blank_serials = wh_quality['blank_serials']
            
            with col_q1:
                st.metric(
                    "📦 Total Received",
                    f"{total_serials:,}",
                    help="Total serial numbers received at warehouse"
                )
            
            with col_q2:
                uniqueness_rate = (unique_serials / total_serials * 100) if total_serials > 0 else 0
                st.metric(
                    "✅ Unique Serials",
                    f"{unique_serials:,}",
                    delta=f"{uniqueness_rate:.1f}%",
                    delta_color="normal",
                    help="Unique serial numbers (no duplicates)"
                )
            
            with col_q3:
                duplicate_rate = (duplicate_count / total_serials * 100) if total_serials > 0 else 0
                st.metric(
                    "⚠️ Duplicates",
                    f"{duplicate_count:,}",
                    delta=f"{duplicate_rate:.1f}%",
                    delta_color="inverse",
                    help="Duplicate serial numbers (same serial received multiple times)"
                )
            
            with col_q4:
                blank_rate = (blank_serials / total_serials * 100) if total_serials > 0 else 0
                st.metric(
                    "❌ Blank/Empty",
                    f"{blank_serials:,}",
                    delta=f"{blank_rate:.1f}%",
                    delta_color="inverse",
                    help="Blank or empty serial numbers"
                )
            
            with col_q5:
                valid_serials = total_serials - blank_serials - wh_quality['single_digit'] - wh_quality['test_serials']
                validity_rate = (valid_serials / total_serials * 100) if total_serials > 0 else 0
                st.metric(
                    "🎯 Valid Serials",
                    f"{valid_serials:,}",
                    delta=f"{validity_rate:.1f}%",
                    delta_color="normal",
                    help="Valid serial numbers (not blank/invalid/test)"
                )
            
            # Second row - Quality issues
            col_q6, col_q7, col_q8, col_q9 = st.columns(4)
            
            with col_q6:
                single_digit = wh_quality['single_digit']
                st.metric(
                    "🔢 Single Digit",
                    f"{single_digit:,}",
                    delta=f"{(single_digit/total_serials*100):.1f}%" if total_serials > 0 else "0%",
                    delta_color="inverse",
                    help="Serial numbers with only 1 alphanumeric character"
                )
            
            with col_q7:
                short_serials = wh_quality['short_serials']
                st.metric(
                    "📏 Too Short (≤3 chars)",
                    f"{short_serials:,}",
                    delta=f"{(short_serials/total_serials*100):.1f}%" if total_serials > 0 else "0%",
                    delta_color="inverse",
                    help="Serial numbers with 3 or fewer characters"
                )
            
            with col_q8:
                test_serials = wh_quality['test_serials']
                st.metric(
                    "🧪 Test/Dummy",
                    f"{test_serials:,}",
                    delta=f"{(test_serials/total_serials*100):.1f}%" if total_serials > 0 else "0%",
                    delta_color="inverse",
                    help="Test serials (contains TEST, DUMMY, XXX)"
                )
            
            with col_q9:
                too_long = wh_quality['too_long_serials']
                st.metric(
                    "📐 Too Long (>50 chars)",
                    f"{too_long:,}",
                    delta=f"{(too_long/total_serials*100):.1f}%" if total_serials > 0 else "0%",
                    delta_color="inverse",
                    help="Serial numbers exceeding 50 characters"
                )
            
            # Create layout: Data Quality (30%) + Serial Verification (70%)
            col_quality_section, col_verification_section = st.columns([0.3, 0.7])
            
            # LEFT COLUMN: Data Quality Breakdown (30%)
            with col_quality_section:
                st.markdown("#### 📊 Data Quality Breakdown")
                quality_categories = ['Valid', 'Duplicates', 'Blank', 'Single Digit', 'Too Short', 'Test/Dummy']
                quality_values = [
                    valid_serials,
                    duplicate_count,
                    blank_serials,
                    single_digit,
                    short_serials - single_digit,  # Exclude single digit from short
                    test_serials
                ]
                
                fig_quality = go.Figure()
                
                fig_quality.add_trace(go.Pie(
                    labels=quality_categories,
                    values=quality_values,
                    hole=0.4,
                    marker=dict(
                        colors=['#27ae60', '#e67e22', '#e74c3c', '#c0392b', '#f39c12', '#9b59b6']
                    ),
                    textinfo='label+percent',
                    textposition='auto'
                ))
                
                fig_quality.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
                )
                
                st.plotly_chart(fig_quality, use_container_width=True)
            
            # RIGHT COLUMN: Serial Numbers Verification (70%)
            with col_verification_section:
                st.markdown("#### ✅ Serial Numbers Verification in Main Database")
                st.info("💡 Analysis of serial numbers verified in main database - Identifying match and no match rates across all shops")
                
                # Get serial check data aggregated across all shops  
                serial_check_df = get_serial_check_by_shop(start_date, end_date)
                
                if not serial_check_df.empty and len(serial_check_df) > 0:
                    # Calculate overall metrics
                    total_serials_checked = serial_check_df['total_serials'].sum()
                    total_matched = serial_check_df['matched_serials'].sum()
                    total_unmatched = serial_check_df['unmatched_serials'].sum()
                    
                    overall_match_rate = (total_matched / total_serials_checked * 100) if total_serials_checked > 0 else 0
                    overall_no_match_rate = (total_unmatched / total_serials_checked * 100) if total_serials_checked > 0 else 0
                    
                    # Display KPI cards
                    col_sc1, col_sc2, col_sc3, col_sc4, col_sc5 = st.columns(5)
                    
                    with col_sc1:
                        st.metric(
                            "📊 Total Checked",
                            f"{total_serials_checked:,}",
                            help="Total serial numbers checked in main database"
                        )
                    
                    with col_sc2:
                        # Match rate color coding
                        if overall_match_rate >= 85:
                            match_icon = "🟪"
                        elif overall_match_rate >= 70:
                            match_icon = "🟡"
                        elif overall_match_rate >= 50:
                            match_icon = "🟠"
                        else:
                            match_icon = "🔴"
                        
                        st.metric(
                            f"{match_icon} Match Rate",
                            f"{overall_match_rate:.1f}%",
                            delta=f"{total_matched:,} verified",
                            delta_color="normal",
                            help="Percentage of serials verified in main database (serial_check = Y)"
                        )
                    
                    with col_sc3:
                        # No match rate color coding (inverse)
                        if overall_no_match_rate >= 50:
                            no_match_icon = "🔴"
                        elif overall_no_match_rate >= 30:
                            no_match_icon = "🟠"
                        elif overall_no_match_rate >= 15:
                            no_match_icon = "🟡"
                        else:
                            no_match_icon = "🟪"
                        
                        st.metric(
                            f"{no_match_icon} No Match Rate",
                            f"{overall_no_match_rate:.1f}%",
                            delta=f"{total_unmatched:,} unverified",
                            delta_color="inverse",
                            help="Percentage of serials NOT verified in main database (serial_check = N)"
                        )
                    
                    with col_sc4:
                        # Average shop match rate
                        avg_shop_match_rate = serial_check_df['match_rate_pct'].mean()
                        st.metric(
                            "📊 Avg Shop Rate",
                            f"{avg_shop_match_rate:.1f}%",
                            help="Average match rate across all shops"
                        )
                    
                    with col_sc5:
                        # Number of shops below 50% (critical)
                        critical_shops = (serial_check_df['match_rate_pct'] < 50).sum()
                        st.metric(
                            "🚨 Critical Shops",
                            f"{critical_shops}",
                            delta="<50% match rate",
                            delta_color="inverse",
                            help="Shops with match rate below 50% (urgent action required)"
                        )
                    
                    # Visual breakdown
                    col_chart_sc1, col_chart_sc2 = st.columns(2)
                    
                    with col_chart_sc1:
                        st.markdown("##### 📊 Overall Verification Status")
                        
                        verification_categories = ['Matched (Y)', 'Not Matched (N)']
                        verification_values = [total_matched, total_unmatched]
                        verification_colors = ['#2ecc71', '#e74c3c']
                        
                        fig_verification = go.Figure(data=[go.Pie(
                            labels=verification_categories,
                            values=verification_values,
                            marker=dict(colors=verification_colors),
                            textinfo='label+percent+value',
                            texttemplate='<b>%{label}</b><br>%{percent}<br>(%{value:,})',
                            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
                        )])
                        
                        fig_verification.update_layout(
                            height=250,
                            margin=dict(l=20, r=20, t=20, b=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_verification, use_container_width=True)
                    
                    with col_chart_sc2:
                        st.markdown("##### 🏪 Bottom 10 Shops by Match Rate")
                        
                        # Show bottom 10 shops with worst match rates
                        bottom_10_shops = serial_check_df.nsmallest(10, 'match_rate_pct')
                        
                        fig_shop_match = go.Figure()
                        
                        # Color coding for bars (mostly red/orange for worst performers)
                        bar_colors = []
                        for rate in bottom_10_shops['match_rate_pct']:
                            if rate >= 85:
                                bar_colors.append('#2ecc71')  # Green
                            elif rate >= 70:
                                bar_colors.append('#f39c12')  # Yellow
                            elif rate >= 50:
                                bar_colors.append('#e67e22')  # Orange
                            else:
                                bar_colors.append('#e74c3c')  # Red
                        
                        fig_shop_match.add_trace(go.Bar(
                            y=bottom_10_shops['shop_code'],
                            x=bottom_10_shops['match_rate_pct'],
                            orientation='h',
                            marker_color=bar_colors,
                            text=bottom_10_shops['match_rate_pct'].apply(lambda x: f"{x:.1f}%"),
                            textposition='outside',
                            hovertemplate='<b>%{y}</b><br>Match Rate: %{x:.1f}%<br>Total: %{customdata[0]:,}<br>Matched: %{customdata[1]:,}<extra></extra>',
                            customdata=bottom_10_shops[['total_serials', 'matched_serials']].values
                        ))
                        
                        fig_shop_match.update_layout(
                            height=250,
                            xaxis_title="Match Rate (%)",
                            yaxis_title="Shop Code",
                            margin=dict(l=20, r=20, t=20, b=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            showlegend=False,
                            xaxis=dict(range=[0, 100])
                        )
                        
                        st.plotly_chart(fig_shop_match, use_container_width=True)
                    
                    # Download button
                    csv_serial_check = serial_check_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Serial Verification Report",
                        data=csv_serial_check,
                        file_name=f"serial_verification_{start_date}_{end_date}.csv",
                        mime="text/csv",
                        key="download_serial_verification"
                    )
                else:
                    st.warning("No serial verification data available for the selected period")
            
 
        #st.markdown("---")
        
        # (Removed) Top 10 Shops with Lowest Offloading % section
        # This section was intentionally removed from the Overview tab.
        st.markdown("---")
        
        # ============================================================
        # ROW 1: LEFT SIDE (METRICS) + RIGHT SIDE (FUNNEL)
        # ============================================================
        
        # Create two main columns: left for metrics, right for funnel
        left_col, right_col = st.columns([1.2, 1])
        
        with left_col:
            # ============================================================
            # ISSUE BREAKDOWN (COMPACT HORIZONTAL LAYOUT)
            # ============================================================
            st.markdown("### 🚨 Issue Breakdown")
            
            # Calculate percentages for all 5 issue types
            no_offload_pct = (funnel_data['no_offloading']/funnel_data['total_issues']*100) if funnel_data['total_issues'] > 0 else 0
            sold_wo_offload_pct = (funnel_data['sold_without_offload']/funnel_data['total_issues']*100) if funnel_data['total_issues'] > 0 else 0
            vehicle_mismatch_pct = (funnel_data['vehicle_mismatch']/funnel_data['total_issues']*100) if funnel_data['total_issues'] > 0 else 0
            shop_mismatch_pct = (funnel_data['shop_mismatch']/funnel_data['total_issues']*100) if funnel_data['total_issues'] > 0 else 0
            serial_mismatch_pct = (funnel_data['serial_mismatch']/funnel_data['total_issues']*100) if funnel_data['total_issues'] > 0 else 0
            
            # 5 columns for horizontal layout
            ib1, ib2, ib3, ib4, ib5 = st.columns(5)
            
            with ib1:
                st.metric(
                    "📦 No Offload",
                    f"{funnel_data['no_offloading']:,}",
                    f"{no_offload_pct:.1f}%",
                    help=f"Items loaded but never offloaded"
                )
            
            with ib2:
                st.metric(
                    "🚫 Sold W/O",
                    f"{funnel_data['sold_without_offload']:,}",
                    f"{sold_wo_offload_pct:.1f}%",
                    help=f"Items sold without offloading"
                )
            
            with ib3:
                st.metric(
                    "🚗 Vehicle",
                    f"{funnel_data['vehicle_mismatch']:,}",
                    f"{vehicle_mismatch_pct:.1f}%",
                    help=f"Mismatched vehicle numbers"
                )
            
            with ib4:
                st.metric(
                    "🏪 Shop",
                    f"{funnel_data['shop_mismatch']:,}",
                    f"{shop_mismatch_pct:.1f}%",
                    help=f"Sold by different shop"
                )
            
            with ib5:
                st.metric(
                    "🔢 Serial",
                    f"{funnel_data['serial_mismatch']:,}",
                    f"{serial_mismatch_pct:.1f}%",
                    help=f"Serial inconsistencies"
                )
            
            # ============================================================
            # LAST 7 DAYS TREND (UNDER ISSUE BREAKDOWN)
            # ============================================================
            st.markdown("---")
            
            # Initialize session state for selected date
            if 'selected_trend_date' not in st.session_state:
                st.session_state.selected_trend_date = None
            
            # Check if we're showing shop-wise view or trend view
            if st.session_state.selected_trend_date is not None:
                # SHOP-WISE FUNNEL VIEW FOR SELECTED DATE
                selected_date = st.session_state.selected_trend_date
                st.markdown(f"### 🏪 Shop-wise Funnel - {selected_date.strftime('%Y-%m-%d')}")
                
                # Back button to return to trend view
                if st.button("⬅️ Back to 7-Day Trend", key="back_to_trend"):
                    st.session_state.selected_trend_date = None
                    st.rerun()
                
                # Get shop-wise data
                shop_df = get_shop_funnel_by_date(selected_date)
                
                if not shop_df.empty:
                    # Simple grouped bar chart showing Loaded, Offloaded, Sold per shop
                    fig_shop = go.Figure()
                    
                    fig_shop.add_trace(go.Bar(
                        name='Loaded',
                        y=shop_df['shop_code'],
                        x=shop_df['loaded'],
                        orientation='h',
                        marker=dict(color='#667eea', line=dict(color='white', width=1)),
                        text=shop_df['loaded'],
                        textposition='auto',
                        textfont=dict(size=11),
                        hovertemplate='<b>%{y}</b><br>Loaded: %{x:,}<extra></extra>'
                    ))
                    
                    fig_shop.add_trace(go.Bar(
                        name='Offloaded',
                        y=shop_df['shop_code'],
                        x=shop_df['offloaded'],
                        orientation='h',
                        marker=dict(color='#4facfe', line=dict(color='white', width=1)),
                        text=shop_df['offloaded'],
                        textposition='auto',
                        textfont=dict(size=11),
                        hovertemplate='<b>%{y}</b><br>Offloaded: %{x:,}<extra></extra>'
                    ))
                    
                    fig_shop.add_trace(go.Bar(
                        name='Sold',
                        y=shop_df['shop_code'],
                        x=shop_df['sold'],
                        orientation='h',
                        marker=dict(color='#00d2ff', line=dict(color='white', width=1)),
                        text=shop_df['sold'],
                        textposition='auto',
                        textfont=dict(size=11),
                        hovertemplate='<b>%{y}</b><br>Sold: %{x:,}<extra></extra>'
                    ))
                    
                    fig_shop.update_layout(
                        barmode='group',
                        height=550,
                        margin=dict(l=80, r=40, t=30, b=50),
                        xaxis=dict(
                            title=dict(text="Count", font=dict(size=13)),
                            tickfont=dict(size=11),
                            gridcolor='rgba(200,200,200,0.3)'
                        ),
                        yaxis=dict(
                            title=dict(text="Shop Code", font=dict(size=13)),
                            tickfont=dict(size=11),
                            showgrid=False,
                            categoryorder='total descending'
                        ),
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=11)
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(250,250,250,0.5)'
                    )
                    
                    st.plotly_chart(fig_shop, use_container_width=True)
                else:
                    st.info(f"No shop data found for {selected_date.strftime('%Y-%m-%d')}")
            
            else:
                # TREND VIEW (DEFAULT) - LAST 7 DAYS
                st.markdown("### 📅 Last 7 Days Offloading Trend")
                st.caption("💡 Click on any date bar to see shop-wise breakdown")
                
                last_7_days_df = get_last_7_days_offloading(end_date)
                
                if not last_7_days_df.empty:
                    # Trend chart with click configuration
                    fig_7days = go.Figure()
                    
                    # Use date_str for x-axis to ensure consistent formatting
                    # Vertical bars for each metric
                    fig_7days.add_trace(go.Bar(
                        name='WH Sent',
                        x=last_7_days_df['date_str'],
                        y=last_7_days_df['wh_sent'],
                        marker=dict(color='#667eea', line=dict(color='white', width=1)),
                        text=last_7_days_df['wh_sent'],
                        textposition='outside',
                        textfont=dict(size=12),
                        hovertemplate='<b>%{x}</b><br>WH Sent: %{y:,}<extra></extra>',
                        customdata=last_7_days_df['date_str']
                    ))
                    
                    fig_7days.add_trace(go.Bar(
                        name='Offloaded',
                        x=last_7_days_df['date_str'],
                        y=last_7_days_df['shop_offloaded'],
                        marker=dict(color='#4facfe', line=dict(color='white', width=1)),
                        text=last_7_days_df['shop_offloaded'],
                        textposition='outside',
                        textfont=dict(size=12),
                        hovertemplate='<b>%{x}</b><br>Offloaded: %{y:,}<extra></extra>',
                        customdata=last_7_days_df['date_str']
                    ))
                    
                    fig_7days.add_trace(go.Bar(
                        name='Not Offloaded',
                        x=last_7_days_df['date_str'],
                        y=last_7_days_df['not_offloaded'],
                        marker=dict(color='#f5576c', line=dict(color='white', width=1)),
                        text=last_7_days_df['not_offloaded'],
                        textposition='outside',
                        textfont=dict(size=12),
                        hovertemplate='<b>%{x}</b><br>Not Offloaded: %{y:,}<extra></extra>',
                        customdata=last_7_days_df['date_str']
                    ))
                    
                    fig_7days.update_layout(
                        barmode='group',
                        height=350,
                        margin=dict(l=40, r=40, t=30, b=70),
                        xaxis=dict(
                            title=dict(text="Date", font=dict(size=13)),
                            tickangle=-45,
                            tickfont=dict(size=11),
                            showgrid=False
                        ),
                        yaxis=dict(
                            title=dict(text="Count", font=dict(size=13)),
                            tickfont=dict(size=11),
                            gridcolor='rgba(200,200,200,0.3)'
                        ),
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=11)
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(250,250,250,0.5)',
                        clickmode='event+select'
                    )
                    
                    # Display chart and capture click events
                    chart_click = st.plotly_chart(fig_7days, use_container_width=True, key="trend_chart", on_select="rerun")
                    
                    # Handle click events
                    if chart_click and 'selection' in chart_click and 'points' in chart_click['selection']:
                        points = chart_click['selection']['points']
                        if points and len(points) > 0:
                            # Use customdata which contains the exact date string
                            clicked_date_str = points[0].get('customdata', points[0].get('x', None))
                            
                            if clicked_date_str:
                                try:
                                    # clicked_date_str is already in YYYY-MM-DD format from customdata
                                    if isinstance(clicked_date_str, str):
                                        # Direct conversion - no timezone issues
                                        clicked_date = datetime.strptime(clicked_date_str, '%Y-%m-%d').date()
                                    else:
                                        # Shouldn't happen but handle it
                                        clicked_date = pd.to_datetime(str(clicked_date_str)).date()
                                    
                                    st.session_state.selected_trend_date = clicked_date
                                    st.rerun()
                                        
                                except Exception as e:
                                    st.error(f"Error parsing date: {clicked_date_str} - {e}")
                                    logger.error(f"Date parse error: {clicked_date_str} - {e}")
                else:
                    st.info("No data for last 7 days")
        
        with right_col:
            # ============================================================
            # FUNNEL CHART (RIGHT SIDE) - Updated with WH Received
            # ============================================================
            st.markdown("### 📊 Serial Number Funnel")
            
            # Create funnel chart
            fig = go.Figure()
            
            # Funnel data with WH Received as first stage
            stages = ['WH Received', 'Loaded at WH', 'Offloaded at Shop', 'Sold to Customer']
            
            # Ensure values are integers
            values = [int(wh_received), int(loaded), int(offloaded), int(sold)]
            
            colors = ['#2E86AB', '#06A77D', '#F77F00', '#D62828']
            
            # Calculate percentages
            load_rate = (loaded / wh_received * 100) if wh_received > 0 else 0
            offload_rate = (offloaded / loaded * 100) if loaded > 0 else 0
            sell_rate = (sold / wh_received * 100) if wh_received > 0 else 0
            
            # Text with percentages - ensure formatting
            texts = [
                f"{int(wh_received):,}",
                f"{int(loaded):,}<br>{load_rate:.1f}%",
                f"{int(offloaded):,}<br>{offload_rate:.1f}%",
                f"{int(sold):,}<br>{sell_rate:.1f}%"
            ]
            
            fig.add_trace(go.Funnel(
                y=stages,
                x=values,
                text=texts,
                textposition="inside",
                textfont=dict(size=18, family="Arial Black", color="white"),
                marker=dict(color=colors, line=dict(width=2, color='white')),
                connector=dict(line=dict(color="#e0e0e0", width=2)),
                texttemplate='%{text}',
                hovertemplate='<b>%{y}</b><br>Count: %{x:,}<extra></extra>'
            ))
            
            fig.update_layout(
                height=450,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True, key="funnel_chart")
        
        # Shop Alert Popup (Top Right)
        if not top_shops.empty:
            shop_items = []
            for idx, shop in top_shops.head(3).iterrows():
                shop_items.append(f"""
                <div class="shop-item">
                    <span><strong>{shop['vc_shop_code']}</strong></span>
                    <span style="background: rgba(255,255,255,0.4); padding: 4px 10px; border-radius: 12px; font-weight: 600;">
                        {int(shop['issue_count'])} issues
                    </span>
                </div>""")
            
            shop_list_html = "".join(shop_items)
            
            popup_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <style>
                .shop-alert-popup {{
                    position: fixed;
                    top: clamp(10px, 2vh, 20px);
                    right: clamp(10px, 2vw, 20px);
                    background: linear-gradient(135deg, #ff4757 0%, #dc3545 100%);
                    color: white;
                    padding: clamp(15px, 2.5vw, 25px);
                    border-radius: 15px;
                    box-shadow: 0 8px 20px rgba(220, 53, 69, 0.4);
                    z-index: 10000;
                    width: clamp(280px, 90vw, 400px);
                    max-width: 400px;
                    animation: slideInRight 0.5s ease-out;
                    font-size: clamp(0.85rem, 1.5vw, 1rem);
                }}
                
                /* Mobile devices */
                @media (max-width: 480px) {{
                    .shop-alert-popup {{
                        top: 10px;
                        right: 5px;
                        left: 5px;
                        width: auto;
                        max-width: none;
                        padding: 15px;
                    }}
                }}
                
                /* Tablets */
                @media (min-width: 481px) and (max-width: 1024px) {{
                    .shop-alert-popup {{
                        top: 15px;
                        right: 15px;
                        width: clamp(300px, 40vw, 380px);
                    }}
                }}
                
                /* Large screens and TVs */
                @media (min-width: 1920px) {{
                    .shop-alert-popup {{
                        top: 30px;
                        right: 30px;
                        width: 420px;
                        padding: 25px 30px;
                    }}
                }}
                
                @keyframes slideInRight {{
                    from {{ transform: translateX(400px); opacity: 0; }}
                    to {{ transform: translateX(0); opacity: 1; }}
                }}
                
                .close-btn {{
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    background: rgba(255,255,255,0.3);
                    border: none;
                    color: white;
                    font-size: 20px;
                    cursor: pointer;
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    line-height: 1;
                    font-weight: bold;
                    transition: all 0.3s ease;
                }}
                
                .close-btn:hover {{
                    background: rgba(255,255,255,0.5);
                    transform: scale(1.1);
                }}
                
                .shop-alert-title {{
                    font-size: clamp(1.1rem, 2.5vw, 1.3rem);
                    font-weight: bold;
                    margin-bottom: 10px;
                    padding-right: 30px;
                    color: #ffffff;
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
                }}
                
                .shop-list {{
                    margin-top: 10px;
                    font-size: 1rem;
                }}
                
                .shop-item {{
                    background: rgba(255,255,255,0.25);
                    padding: clamp(8px, 1.5vw, 10px) clamp(10px, 2vw, 12px);
                    margin: 6px 0;
                    border-radius: 8px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: clamp(0.9rem, 1.8vw, 1.05rem);
                    color: #ffffff;
                    font-weight: 500;
                    flex-wrap: wrap;
                    gap: 8px;
                }}
                
                .shop-item strong {{
                    font-size: clamp(1rem, 2vw, 1.15rem);
                    letter-spacing: 0.5px;
                }}
            </style>
            </head>
            <body>
            <div class="shop-alert-popup" id="shopAlert">
                <button class="close-btn" onclick="closePopup()">×</button>
                <div class="shop-alert-title">🚨 TOP ISSUE SHOPS</div>
                <div style="font-size: 1.2rem; margin-bottom: 10px; color: #fff; font-weight: 500;">
                    Most mismatches from:
                </div>
                <div class="shop-list">{shop_list_html}</div>
                <div style="font-size: 1rem; margin-top: 12px; opacity: 0.9; color: #fff;">
                    Auto-closes in 30s or click × to dismiss
                </div>
            </div>
            
            <script>
                function closePopup() {{
                    document.getElementById('shopAlert').style.display = 'none';
                }}
                
                // Auto-hide after 30 seconds and stop animation
                setTimeout(function() {{
                    var popup = document.getElementById('shopAlert');
                    if (popup) {{
                        popup.style.animation = 'none';
                        popup.style.display = 'none';
                    }}
                }}, 30000);
            </script>
            </body>
            </html>
            """
            
            components.html(popup_html, height=0, scrolling=False)
        
        # Sticky Issue Badge (Bottom Right) with close button
        badge_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            .issue-badge-sticky {{
                position: fixed;
                bottom: clamp(10px, 1.5vh, 15px);
                right: clamp(10px, 2vw, 20px);
                background: linear-gradient(135deg, #ff4757 0%, #dc3545 100%);
                color: white;
                padding: clamp(12px, 2vw, 18px) clamp(18px, 3vw, 25px);
                border-radius: 15px;
                box-shadow: 0 8px 20px rgba(220, 53, 69, 0.4);
                z-index: 9999;
                min-width: clamp(140px, 20vw, 180px);
                text-align: center;
                animation: pulse 2s ease-in-out infinite;
            }}
            
            /* Mobile devices */
            @media (max-width: 480px) {{
                .issue-badge-sticky {{
                    bottom: 10px;
                    right: 5px;
                    left: 5px;
                    width: auto;
                    min-width: auto;
                    padding: 12px 18px;
                }}
            }}
            
            /* Tablets */
            @media (min-width: 481px) and (max-width: 1024px) {{
                .issue-badge-sticky {{
                    bottom: 12px;
                    right: 15px;
                    min-width: 160px;
                }}
            }}
            
            /* Large screens and TVs */
            @media (min-width: 1920px) {{
                .issue-badge-sticky {{
                    bottom: 20px;
                    right: 30px;
                    min-width: 200px;
                    padding: 20px 30px;
                }}
            }}
            
            @keyframes pulse {{
                0%, 100% {{ box-shadow: 0 8px 20px rgba(220, 53, 69, 0.4); }}
                50% {{ box-shadow: 0 8px 30px rgba(220, 53, 69, 0.7); }}
            }}
            
            .issue-label {{
                font-size: clamp(0.75rem, 1.5vw, 0.85rem);
                font-weight: 600;
                margin-bottom: 5px;
                opacity: 0.95;
            }}
            
            .issue-count {{
                font-size: clamp(1.5rem, 4vw, 2rem);
                font-weight: bold;
                margin: 8px 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }}
            
            .close-btn-badge {{
                position: absolute;
                top: 5px;
                right: 5px;
                background: rgba(255,255,255,0.3);
                border: none;
                color: white;
                font-size: 16px;
                cursor: pointer;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                line-height: 1;
                font-weight: bold;
                transition: all 0.3s ease;
            }}
            
            .close-btn-badge:hover {{
                background: rgba(255,255,255,0.5);
                transform: scale(1.1);
            }}
        </style>
        </head>
        <body>
        <div class="issue-badge-sticky" id="issueBadge">
            <button class="close-btn-badge" onclick="closeBadge()">×</button>
            <div class="issue-label">⚠️ TOTAL ISSUES</div>
            <div class="issue-count">{funnel_data['total_issues']:,}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Process Violations</div>
        </div>
        
        <script>
            function closeBadge() {{
                document.getElementById('issueBadge').style.display = 'none';
            }}
            
            // Stop animations after 30 seconds
            setTimeout(function() {{
                var badge = document.getElementById('issueBadge');
                if (badge) {{
                    badge.style.animation = 'none';
                }}
            }}, 30000);
        </script>
        </body>
        </html>
        """
        
        components.html(badge_html, height=0, scrolling=False)
        
        # ============================================================
        # ISSUE DISTRIBUTION (2 COLUMNS: BY SHOP + BY ISSUE TYPE)
        # ============================================================
        st.markdown("---")
        st.markdown("### 📊 Issue Distribution")
        
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            st.markdown("#### By Shop")
            if not top_shops.empty:
                # Pie chart of top 5 shops
                fig_pie_shop = go.Figure()
                
                top_5_shops = top_shops.head(5)
                
                fig_pie_shop.add_trace(go.Pie(
                    labels=top_5_shops['vc_shop_code'],
                    values=top_5_shops['issue_count'],
                    hole=0.4,
                    marker=dict(
                        colors=['#ff4757', '#ff6b81', '#ff7f50', '#ffa502', '#ffb142']
                    ),
                    textinfo='label+percent',
                    textposition='auto',
                    textfont=dict(size=14)
                ))
                
                fig_pie_shop.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                
                st.plotly_chart(fig_pie_shop, use_container_width=True)
            else:
                st.info("✅ No shop issues found in selected date range")
        
        with dist_col2:
            st.markdown("#### By Issue Type")
            if funnel_data['total_issues'] > 0:
                # Create issue type data with all 5 types
                issue_types = [
                    'No Offloading',
                    'Sold W/O Offload',
                    'Vehicle Mismatch',
                    'Shop Mismatch',
                    'Serial Mismatch'
                ]
                issue_values = [
                    funnel_data['no_offloading'],
                    funnel_data['sold_without_offload'],
                    funnel_data['vehicle_mismatch'],
                    funnel_data['shop_mismatch'],
                    funnel_data['serial_mismatch']
                ]
                
                fig_pie_issue = go.Figure()
                
                fig_pie_issue.add_trace(go.Pie(
                    labels=issue_types,
                    values=issue_values,
                    hole=0.4,
                    marker=dict(
                        colors=['#e74c3c', '#f39c12', '#9b59b6', '#3498db', '#1abc9c']
                    ),
                    textinfo='label+percent',
                    textposition='auto',
                    textfont=dict(size=14)
                ))
                
                fig_pie_issue.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                
                st.plotly_chart(fig_pie_issue, use_container_width=True, key='issue_type_pie')
            else:
                st.info("✅ No issues found in selected date range")
        # Management Analysis & Actionable Insights removed from Overview tab
    
    # ============================================================
    # TAB 2: SHOP DRILL-DOWN
    # ============================================================
    with tab2:
        # ============================================================
        # SERIAL CHECK VERIFICATION ANALYSIS
        # ============================================================
        st.markdown("## 📊 Serial Number Verification in Main Database")
        st.info("💡 Analysis of how many serial numbers from each shop are verified (found) in the main database")
        
        # Load serial check data
        serial_check_df = get_serial_check_by_shop(start_date, end_date)
        
        if not serial_check_df.empty:
            # Overall summary metrics
            total_all_serials = serial_check_df['total_serials'].sum()
            total_matched = serial_check_df['matched_serials'].sum()
            total_unmatched = serial_check_df['unmatched_serials'].sum()
            overall_match_rate = (total_matched / total_all_serials * 100) if total_all_serials > 0 else 0
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric(
                    "📦 Total Serial Numbers",
                    f"{total_all_serials:,}",
                    help="Total unique serial numbers sold across all shops"
                )
            
            with col_m2:
                st.metric(
                    "✅ Matched in Main DB",
                    f"{total_matched:,}",
                    delta=f"{overall_match_rate:.1f}%",
                    delta_color="normal",
                    help="Serial numbers verified (Y) in main database"
                )
            
            with col_m3:
                st.metric(
                    "❌ Not Matched",
                    f"{total_unmatched:,}",
                    delta=f"{(total_unmatched/total_all_serials*100):.1f}%" if total_all_serials > 0 else "0%",
                    delta_color="inverse",
                    help="Serial numbers not found (N) or missing in main database"
                )
            
            with col_m4:
                avg_match_rate = serial_check_df['match_rate_pct'].mean()
                st.metric(
                    "📈 Avg Match Rate",
                    f"{avg_match_rate:.1f}%",
                    help="Average match rate across all shops"
                )
            
            st.markdown("---")
            
            # Visual analysis section
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 🔴 Top 5 Shops with LOWEST Match Rates (Action Required)")
                
                # Get bottom 5 shops
                bottom_5 = serial_check_df.head(5)
                
                if not bottom_5.empty:
                    # Bar chart for bottom 5
                    fig_bottom = go.Figure()
                    
                    # Color code based on severity
                    colors = []
                    for rate in bottom_5['match_rate_pct']:
                        if pd.isna(rate) or rate == 0:
                            colors.append('#c0392b')  # Critical - dark red
                        elif rate < 50:
                            colors.append('#e74c3c')  # High - red
                        elif rate < 70:
                            colors.append('#e67e22')  # Medium - orange
                        else:
                            colors.append('#f39c12')  # Low - yellow
                    
                    fig_bottom.add_trace(go.Bar(
                        x=bottom_5['shop_code'],
                        y=bottom_5['match_rate_pct'].fillna(0).astype(float),
                        marker_color=colors,
                        text=bottom_5['match_rate_pct'].fillna(0).astype(float).round(1),
                        texttemplate='%{text}%',
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>' +
                                     'Match Rate: %{y:.1f}%<br>' +
                                     'Matched: ' + bottom_5['matched_serials'].astype(str) + '<br>' +
                                     'Total: ' + bottom_5['total_serials'].astype(str) +
                                     '<extra></extra>'
                    ))
                    
                    fig_bottom.update_layout(
                        title="Critical: Lowest Verification Rates",
                        xaxis_title="Shop Code",
                        yaxis_title="Match Rate (%)",
                        height=400,
                        yaxis=dict(range=[0, 100]),
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_bottom, use_container_width=True)
                    
                    # Detailed table for bottom 5
                    st.markdown("##### 📋 Action Required - Shop Details")
                    display_bottom = bottom_5[['shop_code', 'total_serials', 'matched_serials', 'unmatched_serials', 'match_rate_pct']].copy()
                    display_bottom.columns = ['Shop Code', 'Total Serials', '✅ Matched', '❌ Unmatched', 'Match Rate %']
                    
                    # Add action recommendation
                    display_bottom['⚠️ Action'] = display_bottom['Match Rate %'].apply(
                        lambda x: '🔴 URGENT - Investigate immediately' if pd.isna(x) or x == 0
                        else '🔴 HIGH - Review processes' if x < 50
                        else '🟠 MEDIUM - Monitor closely' if x < 70
                        else '🟡 LOW - Minor review needed'
                    )
                    
                    st.dataframe(display_bottom, use_container_width=True, height=250)
                else:
                    st.success("✅ All shops have good match rates!")
            
            with col_chart2:
                st.markdown("#### 🟢 Top 5 Shops with HIGHEST Match Rates (Best Performers)")
                
                # Get top 5 shops
                top_5 = serial_check_df.nlargest(5, 'match_rate_pct')
                
                if not top_5.empty:
                    # Bar chart for top 5
                    fig_top = go.Figure()
                    
                    fig_top.add_trace(go.Bar(
                        x=top_5['shop_code'],
                        y=top_5['match_rate_pct'],
                        marker_color='#27ae60',
                        text=top_5['match_rate_pct'].round(1),
                        texttemplate='%{text}%',
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>' +
                                     'Match Rate: %{y:.1f}%<br>' +
                                     'Matched: ' + top_5['matched_serials'].astype(str) + '<br>' +
                                     'Total: ' + top_5['total_serials'].astype(str) +
                                     '<extra></extra>'
                    ))
                    
                    fig_top.update_layout(
                        title="Excellence: Highest Verification Rates",
                        xaxis_title="Shop Code",
                        yaxis_title="Match Rate (%)",
                        height=400,
                        yaxis=dict(range=[0, 100]),
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_top, use_container_width=True)
                    
                    # Detailed table for top 5
                    st.markdown("##### 🏆 Best Practices - Shop Details")
                    display_top = top_5[['shop_code', 'total_serials', 'matched_serials', 'unmatched_serials', 'match_rate_pct']].copy()
                    display_top.columns = ['Shop Code', 'Total Serials', '✅ Matched', '❌ Unmatched', 'Match Rate %']
                    display_top['🎯 Status'] = '🟢 Excellent - Share best practices'
                    
                    st.dataframe(display_top, use_container_width=True, height=250)
                else:
                    st.warning("No data available")
            
            # Complete shop comparison table
            st.markdown("---")
            st.markdown("#### 📊 Complete Shop-wise Serial Check Comparison")
            
            # Add visual indicators
            def match_rate_indicator(rate):
                if pd.isna(rate) or rate == 0:
                    return '🔴'
                elif rate < 50:
                    return '🔴'
                elif rate < 70:
                    return '🟠'
                elif rate < 85:
                    return '🟡'
                else:
                    return '🟢'
            
            display_all = serial_check_df.copy()
            # Add indicator next to shop code instead of separate column
            display_all['Shop Code'] = display_all.apply(
                lambda row: f"{match_rate_indicator(row['match_rate_pct'])} {row['shop_code']}", 
                axis=1
            )
            display_all = display_all[['Shop Code', 'total_serials', 'matched_serials', 'unmatched_serials', 'match_rate_pct']]
            display_all.columns = ['Shop Code', 'Total Serials', '✅ Matched', '❌ Unmatched', 'Match Rate %']
            
            st.dataframe(display_all, use_container_width=True, height=400)
            
            # Download button
            csv_serial_check = serial_check_df.to_csv(index=False)
            st.download_button(
                "📥 Download Serial Check Analysis",
                data=csv_serial_check,
                file_name=f"serial_check_analysis_{start_date}_{end_date}.csv",
                mime="text/csv",
                key="download_serial_check_analysis"
            )
        else:
            st.warning("No serial check data available for the selected period")
        
        st.markdown("---")
        
        # ============================================================
        # SHOP DRILL-DOWN SECTION
        # ============================================================
        st.markdown("## 🏪 Shop-wise Issue Analysis & Drill-Down")
        
        # Sticky filter bar at top
        filter_container = st.container()
        
        with filter_container:
            # Check if user clicked on issue type pie chart
            if 'selected_issue_type' in st.session_state and st.session_state.selected_issue_type:
                selected_issue_display = st.session_state.selected_issue_type
                
                # Add custom CSS for sticky filter bar
                st.markdown("""
                <style>
                    .filter-sticky-bar {
                        position: sticky;
                        top: 0;
                        z-index: 999;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 15px 20px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    }
                    .filter-sticky-bar h3 {
                        color: white;
                        margin: 0 0 10px 0;
                        font-size: 1.3rem;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="filter-sticky-bar">', unsafe_allow_html=True)
                st.markdown(f"### 🏪 Shop Analysis - Filtered by: **{selected_issue_display}**")
                
                # Filter controls in columns
                col_a, col_b, col_c = st.columns([2, 2, 1])
                with col_a:
                    st.info(f"📌 Active Filter: **{selected_issue_display}**")
                with col_b:
                    pass  # Spacer
                with col_c:
                    if st.button("🔄 Clear", use_container_width=True, key="clear_filter_top"):
                        st.session_state.selected_issue_type = None
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Get filtered shop data
                filtered_top_shops = get_top_issue_shops(start_date, end_date, selected_issue_display)
            else:
                st.markdown("### 🏪 Top Issue Shops - Detailed Analysis")
                selected_issue_display = None
                filtered_top_shops = top_shops
            
            # Two filters side by side with sticky positioning
            st.markdown("""
            <style>
                .element-container:has(> div.row-widget.stSelectbox) {
                    position: sticky;
                    top: 80px;
                    z-index: 998;
                    background: white;
                    padding: 15px 0;
                    margin-bottom: 10px;
                    border-bottom: 2px solid #f0f0f0;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Initialize session state for filters to prevent tab switching
            if 'selected_shop_filter' not in st.session_state:
                st.session_state.selected_shop_filter = 'All'
            if 'selected_issue_filter' not in st.session_state:
                st.session_state.selected_issue_filter = 'All Issues'
            
            col_filter1, col_filter2 = st.columns(2)
            
            with col_filter1:
                # Shop selector for drill-down with 'All' option
                if not filtered_top_shops.empty and 'vc_shop_code' in filtered_top_shops.columns:
                    shop_options = ['All'] + filtered_top_shops['vc_shop_code'].tolist()
                else:
                    shop_options = ['All']
                
                # Get default index
                try:
                    shop_default_idx = shop_options.index(st.session_state.selected_shop_filter)
                except ValueError:
                    shop_default_idx = 0
                    st.session_state.selected_shop_filter = 'All'
                
                selected_shop = st.selectbox(
                    "🏪 Select Shop:",
                    options=shop_options,
                    index=shop_default_idx,
                    key='shop_selector_widget',
                    help="Choose a specific shop or 'All' to view all shops"
                )
                
                # Update session state
                if selected_shop != st.session_state.selected_shop_filter:
                    st.session_state.selected_shop_filter = selected_shop
            
            with col_filter2:
                # Issue type filter
                issue_options = [
                    'All Issues',
                    'No Offloading',
                    'Sold Without Offloading',
                    'Vehicle Mismatch',
                    'Shop Mismatch',
                    'Serial Mismatch'
                ]
                
                # Set default based on session state or selected_issue_display
                default_idx = 0
                if selected_issue_display:
                    if selected_issue_display == 'No offloading':
                        default_idx = 1
                    elif selected_issue_display == 'Sold without offloading':
                        default_idx = 2
                    elif selected_issue_display == 'Vehicle mismatch':
                        default_idx = 3
                    elif selected_issue_display == 'Shop mismatch':
                        default_idx = 4
                    elif selected_issue_display == 'Serial mismatch':
                        default_idx = 5
                else:
                    try:
                        default_idx = issue_options.index(st.session_state.selected_issue_filter)
                    except ValueError:
                        default_idx = 0
                
                selected_issue_filter = st.selectbox(
                    "⚠️ Filter by Issue Type:",
                    options=issue_options,
                    index=default_idx,
                    key='issue_type_selector_widget',
                    help="Filter data by specific issue type"
                )
                
                # Update session state
                if selected_issue_filter != st.session_state.selected_issue_filter:
                    st.session_state.selected_issue_filter = selected_issue_filter
                
                # Map display names to SQL filter logic
                issue_filter_mapping = {
                    'All Issues': None,
                    'No Offloading': 'No offloading',
                    'Sold Without Offloading': 'Sold without offloading',
                    'Vehicle Mismatch': 'Vehicle mismatch',
                    'Shop Mismatch': 'Shop mismatch',
                    'Serial Mismatch': 'Serial mismatch'
                }
                
                # Update filtered data based on dropdown selection
                selected_issue_display = issue_filter_mapping[selected_issue_filter]
                if selected_issue_filter != 'All Issues':
                    filtered_top_shops = get_top_issue_shops(start_date, end_date, selected_issue_display)
        
        if not filtered_top_shops.empty:
            # Create dynamic bar chart based on shop selection
            fig_shops = go.Figure()
            
            if selected_shop == 'All':
                # Show shop-wise breakdown when 'All' is selected
                display_shops = filtered_top_shops
                shop_title = "All Shops"
                
                fig_shops.add_trace(go.Bar(
                    x=display_shops['vc_shop_code'],
                    y=display_shops['issue_count'],
                    text=display_shops['issue_count'],
                    textposition='auto',
                    marker=dict(
                        color=display_shops['issue_count'],
                        colorscale='Reds',
                        line=dict(color='white', width=2)
                    ),
                    hovertemplate='<b>%{x}</b><br>Total Issues: %{y}<extra></extra>'
                ))
                
                chart_title = f"Issue Count by Shop"
                xlabel = "Shop Code"
            else:
                # Show item-wise breakdown when specific shop is selected
                brand_data = get_shop_issues_by_brand(start_date, end_date, selected_shop, selected_issue_display)
                shop_title = selected_shop
                
                if not brand_data.empty:
                    # Show full item code and item name in hover
                    hover_text = brand_data.apply(
                        lambda row: f"<b>{row['item_code']}</b><br>{row.get('item_name', '')}<br>Issues: {row['issue_count']}",
                        axis=1
                    )
                    
                    fig_shops.add_trace(go.Bar(
                        x=brand_data['item_code'],
                        y=brand_data['issue_count'],
                        text=brand_data['issue_count'],
                        textposition='auto',
                        marker=dict(
                            color=brand_data['issue_count'],
                            colorscale='Reds',
                            line=dict(color='white', width=2)
                        ),
                        hovertemplate='%{customdata}<extra></extra>',
                        customdata=hover_text
                    ))
                    
                    chart_title = f"Issue Count by Item - {shop_title}"
                    xlabel = "Item Code"
                else:
                    st.warning(f"📊 **Item data missing** - Cannot show item breakdown for {selected_shop}.")
                    chart_title = f"Issue Count - {shop_title}"
                    xlabel = "Item Code"
            
            if selected_issue_display:
                chart_title += f" ({selected_issue_display})"
            
            fig_shops.update_layout(
                height=320,
                margin=dict(l=30, r=30, t=40, b=120),
                xaxis=dict(
                    title=dict(text=xlabel, font=dict(size=15)), 
                    tickangle=-45, 
                    tickfont=dict(size=11),
                    tickmode='linear',
                    type='category'  # Force categorical to prevent scientific notation
                ),
                yaxis=dict(title=dict(text="Issue Count", font=dict(size=15)), tickfont=dict(size=13)),
                title=dict(text=chart_title, x=0.5, xanchor='center', font=dict(size=16)),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(250,250,250,0.5)',
                showlegend=False
            )
            
            st.plotly_chart(fig_shops, use_container_width=True, key='shop_issues_detailed')
            
            # Show detailed breakdown by item & date
            if selected_shop != 'All':
                subtitle = f"#### 📊 {selected_shop} - Issues by Item & Date"
                if selected_issue_display:
                    subtitle += f" ({selected_issue_display})"
                st.markdown(subtitle)
                
                shop_details = get_shop_issues_by_brand_date(start_date, end_date, selected_shop, selected_issue_display)
                
                if not shop_details.empty:
                    # Create line chart showing issues over time by item
                    fig_timeline = go.Figure()
                    
                    # Get unique items
                    items = shop_details['item_code'].unique()
                    
                    for item in items[:10]:  # Top 10 items
                        item_data = shop_details[shop_details['item_code'] == item]
                        fig_timeline.add_trace(go.Scatter(
                            x=item_data['load_date'],
                            y=item_data['issue_count'],
                            mode='lines+markers',
                            name=item,
                            line=dict(width=2),
                            marker=dict(size=8),
                            hovertemplate=f'<b>{item}</b><br>Date: %{{x}}<br>Issues: %{{y}}<extra></extra>',
                            customdata=item_data['load_date']  # Store date for click events
                        ))
                    
                    fig_timeline.update_layout(
                        height=400,
                        margin=dict(l=30, r=30, t=30, b=80),
                        xaxis=dict(title=dict(text="Date (Click on a point to see details)", font=dict(size=15)), tickangle=-45, tickfont=dict(size=13)),
                        yaxis=dict(title=dict(text="Issue Count", font=dict(size=15)), tickfont=dict(size=13)),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(250,250,250,0.5)',
                        hovermode='x unified',
                        clickmode='event',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.4,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=13)
                        )
                    )
                    
                    # Display chart with click events
                    clicked_point = st.plotly_chart(
                        fig_timeline, 
                        use_container_width=True, 
                        key=f'timeline_{selected_shop}',
                        on_select="rerun"
                    )
                    
                    # Handle click events
                    if clicked_point and clicked_point.selection and clicked_point.selection.points:
                        # Get the clicked date
                        clicked_date = clicked_point.selection.points[0]['x']
                        
                        # Convert to date object if it's a string
                        if isinstance(clicked_date, str):
                            clicked_date = datetime.strptime(clicked_date.split('T')[0], '%Y-%m-%d').date()
                        
                        # Get detailed records for clicked date
                        date_details = get_issue_details_by_date(
                            start_date, end_date, selected_shop, clicked_date, selected_issue_display
                        )
                        
                        if not date_details.empty:
                            # Extract brand from item description (characters before first space)
                            date_details['brand'] = date_details['vc_item_desc'].str.split(' ').str[0] if 'vc_item_desc' in date_details.columns else ''
                            
                            # Display metric for clicked date
                            col_metric1, col_metric2 = st.columns([2, 1])
                            with col_metric2:
                                st.metric(
                                    "Total Issues",
                                    f"{len(date_details):,}",
                                    f"on {clicked_date.strftime('%Y-%m-%d') if hasattr(clicked_date, 'strftime') else str(clicked_date)}"
                                )
                            
                            st.markdown(f"##### 📋 Issue Details for {clicked_date.strftime('%Y-%m-%d') if hasattr(clicked_date, 'strftime') else str(clicked_date)}")
                            
                            # Display the detailed records
                            display_cols = ['loaded_datetime', 'brand', 'vc_item_code', 
                                           'serial_no', 'vc_serail_no', 'shop_serail_no',
                                           'vc_shop_code', 'shop_sold', 'vc_vehicle_no', 
                                           'vc_vehicle_no_2', 'remarks2']
                            
                            # Prepare display dataframe with renamed columns
                            display_data = date_details[display_cols].copy()
                            display_data.columns = ['Date & Time', 'Brand', 'Item Code', 
                                                   'WH Serial', 'Shop Serial', 'Sold Serial',
                                                   'Offload Shop', 'Sold Shop', 'WH Vehicle', 
                                                   'Shop Vehicle', 'Issues']
                            
                            st.dataframe(
                                display_data,
                                use_container_width=True,
                                height=600
                            )
                            
                            # Download button for date-specific data
                            csv_date = date_details.to_csv(index=False)
                            st.download_button(
                                label=f"📥 Download Issues for {clicked_date.strftime('%Y-%m-%d') if hasattr(clicked_date, 'strftime') else str(clicked_date)}",
                                data=csv_date,
                                file_name=f"issues_{clicked_date.strftime('%Y%m%d') if hasattr(clicked_date, 'strftime') else str(clicked_date)}.csv",
                                mime="text/csv",
                                key="download_date_issues"
                            )
                        else:
                            st.info(f"No issue details found for {clicked_date}")
                    else:
                        st.info("💡 Click on any point in the timeline chart above to see detailed records for that date")
                else:
                    st.info(f"No detailed data available for {selected_shop}")
            
            # ============================================================
            # VEHICLE MISMATCH ANALYSIS
            # ============================================================
            st.markdown("---")
            st.markdown("### 🚛 Vehicle Tracking & Transit Analysis")
            
            # Get transit accuracy metrics
            transit_metrics = get_transit_accuracy_metrics(start_date, end_date)
            
            if transit_metrics:
                col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                
                total_tracked = transit_metrics.get('total_with_vehicle_data', 0)
                vehicle_match = transit_metrics.get('vehicle_match', 0)
                vehicle_mismatch = transit_metrics.get('vehicle_mismatch', 0)
                
                match_rate = (vehicle_match / total_tracked * 100) if total_tracked > 0 else 0
                mismatch_rate = (vehicle_mismatch / total_tracked * 100) if total_tracked > 0 else 0
                
                with col_t1:
                    st.metric(
                        "🎯 Vehicle Match Rate",
                        f"{match_rate:.1f}%",
                        f"{vehicle_match:,} items",
                        delta_color="normal",
                        help="Percentage of items where warehouse vehicle matches shop vehicle (same vehicle throughout)"
                    )
                
                with col_t2:
                    mismatch_color = "inverse" if mismatch_rate > 10 else "normal"
                    st.metric(
                        "⚠️ Vehicle Mismatch Rate",
                        f"{mismatch_rate:.1f}%",
                        f"{vehicle_mismatch:,} items",
                        delta_color=mismatch_color,
                        help="Items transferred between vehicles during transit (potential cross-loading or data errors)"
                    )
                
                with col_t3:
                    missing_wh = transit_metrics.get('missing_wh_vehicle', 0)
                    st.metric(
                        "📦 no of Item w/o WH vehical Number",
                        f"{missing_wh:,}",
                        f"{(missing_wh/total_tracked*100):.1f}%" if total_tracked > 0 else "0%",
                        delta_color="inverse" if missing_wh > 0 else "normal",
                        help="Items without warehouse departure vehicle number (data entry gap)"
                    )
                
                with col_t4:
                    missing_shop = transit_metrics.get('missing_shop_vehicle', 0)
                    st.metric(
                        "🏪 Missing Shop VehicleNO",
                        f"{missing_shop:,}",
                        f"{(missing_shop/total_tracked*100):.1f}%" if total_tracked > 0 else "0%",
                        delta_color="inverse" if missing_shop > 0 else "normal",
                        help="Items without shop arrival vehicle number (offload not recorded)"
                    )
            
            # Vehicle mismatch details
            if selected_shop != 'All':
                st.markdown(f"#### 🚛 Vehicle Mismatches - {selected_shop}")
                vehicle_mismatches = get_vehicle_mismatches(start_date, end_date, selected_shop)
                
                if not vehicle_mismatches.empty:
                    st.warning(f"⚠️ Found {len(vehicle_mismatches):,} vehicle mismatches for {selected_shop}")
                    
                    # Show sample of mismatches
                    # Prepare display dataframe with renamed columns
                    display_data = vehicle_mismatches[['loaded_datetime', 'brand', 'serial_no', 
                                      'warehouse_vehicle', 'shop_vehicle', 'remarks2']].head(50).copy()
                    display_data.columns = ['Date', 'Brand', 'Serial No', 'WH Vehicle', 'Shop Vehicle', 'Issues']
                    
                    st.dataframe(
                        display_data,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download vehicle mismatches
                    csv_vehicle = vehicle_mismatches.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Vehicle Mismatches",
                        data=csv_vehicle,
                        file_name=f"vehicle_mismatches_{start_date}_{end_date}.csv",
                        mime="text/csv",
                        key="download_vehicle_mismatches"
                    )
                    st.success(f"✅ No vehicle mismatches found for {selected_shop}")
            else:
                # Show vehicle mismatch summary for all shops
                st.markdown("#### 🚛 Vehicle Mismatches by Shop")
                vehicle_summary = get_vehicle_mismatch_summary(start_date, end_date)
                
                if not vehicle_summary.empty:
                    # Create bar chart
                    fig_vehicle = go.Figure()
                    
                    fig_vehicle.add_trace(go.Bar(
                        x=vehicle_summary['vc_shop_code'],
                        y=vehicle_summary['vehicle_mismatch_count'],
                        text=vehicle_summary['vehicle_mismatch_count'],
                        textposition='auto',
                        marker=dict(
                            color=vehicle_summary['vehicle_mismatch_count'],
                            colorscale='OrRd',
                            line=dict(color='white', width=2)
                        ),
                        hovertemplate='<b>%{x}</b><br>Mismatches: %{y}<br>' +
                                    'WH Vehicles: ' + vehicle_summary['unique_wh_vehicles'].astype(str) + '<br>' +
                                    'Shop Vehicles: ' + vehicle_summary['unique_shop_vehicles'].astype(str) + '<extra></extra>'
                    ))
                    
                    fig_vehicle.update_layout(
                        height=350,
                        margin=dict(l=30, r=30, t=40, b=70),
                        xaxis=dict(title=dict(text="Shop Code", font=dict(size=15)), tickangle=-45, tickfont=dict(size=13)),
                        yaxis=dict(title=dict(text="Vehicle Mismatch Count", font=dict(size=15)), tickfont=dict(size=13)),
                        title=dict(text="Vehicle Mismatches by Shop", x=0.5, xanchor='center', font=dict(size=16)),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(250,250,250,0.5)',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_vehicle, use_container_width=True, key='vehicle_summary_chart')
                else:
                    st.success("✅ No vehicle mismatches found")
            
            # ============================================================
            # ROOT CAUSE PATTERN ANALYSIS
            # ============================================================
            st.markdown("---")
            shop_display = selected_shop if selected_shop != 'All' else 'All Shops'
            st.markdown(f"### 🔍 Root Cause Analysis - Issue Pattern Matrix ({shop_display})")
            st.info("💡 This analysis shows the PRIMARY cause of serial number issues. All values shown as percentages (%) of total issues per shop.")
            
            root_cause_df = get_root_cause_patterns(start_date, end_date, selected_shop)
            
            if not root_cause_df.empty:
                # Calculate percentages for heatmap (using total_issues for main categories)
                root_cause_df['no_offload_pct'] = (root_cause_df['no_offloading'] / root_cause_df['total_issues'] * 100).round(1)
                root_cause_df['process_pct'] = (root_cause_df['process_bypass'] / root_cause_df['total_issues'] * 100).round(1)
                root_cause_df['data_pct'] = (root_cause_df['data_entry_errors'] / root_cause_df['total_issues'] * 100).round(1)
                root_cause_df['routing_pct'] = (root_cause_df['routing_errors'] / root_cause_df['total_issues'] * 100).round(1)
                # Vehicle percentage: calculate as percentage of items that have vehicle data
                root_cause_df['vehicle_pct'] = root_cause_df.apply(
                    lambda row: round(row['vehicle_mismatches'] / row['total_issues'] * 100, 1) if row['total_issues'] > 0 else 0,
                    axis=1
                )
                
                # Identify primary issue for each shop (from all four main categories)
                def get_primary_issue(row):
                    issues = {
                        'No Offloading': row['no_offload_pct'],
                        'Process Bypass': row['process_pct'],
                        'Data Entry': row['data_pct'],
                        'Routing': row['routing_pct']
                    }
                    return max(issues, key=issues.get)
                
                root_cause_df['primary_issue'] = root_cause_df.apply(get_primary_issue, axis=1)
                
                # Create heatmap
                col_rc1, col_rc2 = st.columns([2, 1])
                
                with col_rc1:
                    fig_heatmap = go.Figure()
                    
                    # Prepare data for heatmap (all categories as %, including vehicle)
                    shops = root_cause_df['vc_shop_code'].tolist()
                    categories = ['No\nOffloading\n(%)', 'Process\nBypass\n(%)', 'Data\nEntry\n(%)', 'Routing\nErrors\n(%)', 'Vehicle\nIssues\n(%)']
                    
                    heatmap_data = [
                        root_cause_df['no_offload_pct'].tolist(),
                        root_cause_df['process_pct'].tolist(),
                        root_cause_df['data_pct'].tolist(),
                        root_cause_df['routing_pct'].tolist(),
                        root_cause_df['vehicle_pct'].tolist()  # Now showing as percentage
                    ]
                    
                    fig_heatmap.add_trace(go.Heatmap(
                        z=heatmap_data,
                        x=shops,
                        y=categories,
                        colorscale='RdYlGn_r',
                        text=heatmap_data,
                        texttemplate='%{text:.1f}',  # Show decimal for percentages
                        textfont={"size": 14},
                        colorbar=dict(title="Percentage")
                    ))
                    
                    fig_heatmap.update_layout(
                        height=400,
                        margin=dict(l=30, r=30, t=40, b=100),
                        xaxis=dict(title=dict(text="Shop Code", font=dict(size=15)), tickangle=-45, tickfont=dict(size=13)),
                        yaxis=dict(title=dict(text="Issue Category", font=dict(size=15)), tickfont=dict(size=14)),
                        title=dict(text="Issue Distribution by Root Cause (%)", x=0.5, xanchor='center', font=dict(size=16)),
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True, key='root_cause_heatmap')
                    
                    # Issue explanations
                    st.markdown("""
                    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 10px; font-size: 17px; line-height: 1.8;'>
                        <b style='font-size: 19px;'>📖 Issue Definitions:</b><br>
                        <b style='color: #ff6b6b;'>• No Offloading:</b> Items sent from WH but not offloaded at shop (vc_serail_no blank)<br>
                        <b style='color: #e74c3c;'>• Process Bypass:</b> Items sold with serial mismatch between WH and shop<br>
                        <b style='color: #f39c12;'>• Data Entry:</b> Serial number mismatches or invalid serials (≤5 digits, matches item code)<br>
                        <b style='color: #9b59b6;'>• Vehicle Issues:</b> Vehicle number changed during transit (cross-loading detected)<br>
                        <b style='color: #3498db;'>• Routing Errors:</b> Items delivered to wrong shop (shop code mismatch)
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_rc2:
                    st.markdown("#### 🎯 Primary Issue Summary")
                    
                    # Count primary issues
                    primary_counts = root_cause_df['primary_issue'].value_counts()
                    
                    # Pie chart
                    fig_primary = go.Figure()
                    
                    colors_primary = {
                        'No Offloading': '#ff6b6b',
                        'Process Bypass': '#e74c3c',
                        'Data Entry': '#f39c12',
                        'Routing': '#3498db',
                        'Vehicle': '#9b59b6'
                    }
                    
                    fig_primary.add_trace(go.Pie(
                        labels=primary_counts.index,
                        values=primary_counts.values,
                        hole=0.4,
                        marker=dict(colors=[colors_primary.get(x, '#95a5a6') for x in primary_counts.index]),
                        textinfo='label+value',
                        textposition='auto'
                    ))
                    
                    fig_primary.update_layout(
                        height=300,
                        margin=dict(l=10, r=10, t=30, b=10),
                        title=dict(text="Shops by Primary Issue", x=0.5, xanchor='center'),
                        paper_bgcolor='rgba(0,0,0,0)',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_primary, use_container_width=True, key='primary_issue_pie')
                    
                    # Recommendations with download buttons
                    st.markdown("#### 📋 Recommendations")
                    for issue_type, count in primary_counts.items():
                        # Filter shops by primary issue
                        shops_with_issue = root_cause_df[root_cause_df['primary_issue'] == issue_type].copy()
                        shops_with_issue['recommendation'] = ''
                        
                        col_rec1, col_rec2 = st.columns([3, 1])
                        
                        with col_rec1:
                            if issue_type == 'Process Bypass':
                                st.error(f"**{count} shops:** Enforce offload process before sales")
                                shops_with_issue['recommendation'] = 'Enforce offload process before sales'
                            elif issue_type == 'Data Entry':
                                st.warning(f"**{count} shops:** Improve serial number data entry training")
                                shops_with_issue['recommendation'] = 'Improve serial number data entry training'
                            elif issue_type == 'Routing':
                                st.info(f"**{count} shops:** Review and optimize delivery routes")
                                shops_with_issue['recommendation'] = 'Review and optimize delivery routes'
                            elif issue_type == 'Vehicle':
                                st.warning(f"**{count} shops:** Audit vehicle tracking & cross-loading")
                                shops_with_issue['recommendation'] = 'Audit vehicle tracking & cross-loading'
                        
                        with col_rec2:
                            if not shops_with_issue.empty:
                                # Prepare download data
                                download_df = shops_with_issue[[
                                    'vc_shop_code', 'total_issues', 'process_bypass', 
                                    'data_entry_errors', 'routing_errors', 'vehicle_mismatches',
                                    'recommendation'
                                ]].copy()
                                download_df.columns = ['Shop Code', 'Total Issues', 'Process Bypass', 
                                                      'Data Entry Errors', 'Routing Errors', 'Vehicle Mismatches',
                                                      'Action Required']
                                
                                csv_recommendation = download_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download",
                                    data=csv_recommendation,
                                    file_name=f"shop_recommendations_{start_date}_{end_date}.csv",
                                    mime="text/csv",
                                    key="download_shop_recommendations"
                                )

                
                # Detailed table
                st.markdown("#### 📊 Detailed Root Cause Breakdown")
                display_df = root_cause_df[[
                    'vc_shop_code', 'total_issues', 'primary_issue',
                    'no_offloading', 'process_bypass', 'data_entry_errors', 'routing_errors', 'vehicle_mismatches'
                ]].copy()
                
                # Prepare display dataframe with renamed columns
                styled_display_df = display_df.copy()
                styled_display_df.columns = ['Shop', 'Total Issues', 'Primary Cause', 
                                            'No Offload', 'Process', 'Data Entry', 
                                            'Routing', 'Vehicle']
                
                st.dataframe(
                    styled_display_df,
                    use_container_width=True,
                    height=500
                )
                
                # Download root cause analysis
                csv_root = root_cause_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Root Cause Analysis",
                    data=csv_root,
                    file_name=f"root_cause_analysis_{start_date}_{end_date}.csv",
                    mime="text/csv",
                    key="download_root_cause"
                )

            else:
                st.success("✅ No issues found for root cause analysis")
        else:
            st.info("No shop-specific issues found in the selected date range.")
    
    # ============================================================
    # ISSUE DETAILS
    # ============================================================
    st.markdown("### 🔍 Issue Details")
    
    issue_filter = st.selectbox(
        "Filter by issue type:",
        ['All Issues', 'No offloading', 'Sold without offloading', 'Vehicle mismatch', 'Shop mismatch', 'Serial mismatch'],
        key='overview_issue_filter'
    )
    
    issue_type_map = {
        'All Issues': None,
        'No offloading': 'No offloading',
        'Sold without offloading': 'Sold without offloading',
        'Vehicle mismatch': 'Vehicle mismatch',
        'Shop mismatch': 'Shop mismatch',
        'Serial mismatch': 'Serial mismatch'
    }
    
    issue_df = get_issue_details(start_date, end_date, issue_type_map[issue_filter])
    
    if not issue_df.empty:
        st.markdown(f"**Found {len(issue_df):,} issues** (showing all records)")
        
        # Extract brand from item description (characters before first space)
        issue_df['brand'] = issue_df['vc_item_desc'].str.split(' ').str[0] if 'vc_item_desc' in issue_df.columns else ''
        
        # Display issues
        display_cols = ['loaded_datetime', 'brand', 'vc_item_code', 'serial_no', 'vc_serail_no', 
                       'shop_serail_no', 'vc_shop_code', 'shop_sold', 'remarks2']
        
        # Prepare display dataframe with renamed columns
        display_data = issue_df[display_cols].copy()
        display_data.columns = ['Date', 'Brand', 'Item Code', 'WH Serial', 'Shop Serial', 
                               'Sold Serial', 'Offload Shop', 'Sold Shop', 'Issue']
        
        st.dataframe(
            display_data,
            use_container_width=True,
            height=600
        )
        
        # Download button
        csv = issue_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Issue Report",
            data=csv,
            file_name=f"serial_issues_{start_date}_{end_date}.csv",
            mime="text/csv",
            key="download_full_issues"
        ),

    else:
        st.success("✅ No issues found for the selected period!")
    
    # ============================================================
    # TAB 3: SERIAL JOURNEY TRACKER
    # ============================================================
    with tab3:
        st.markdown("## 🔍 Complete Serial Number Journey Tracker")
        st.info("💡 Track serial numbers from warehouse receipt through to final sale, with verification against main database")
        
        # ============================================================
        # SERIAL NUMBER SEARCH
        # ============================================================
        st.markdown("### 🔎 Search Serial Number")
        
        col_search, col_btn = st.columns([3, 1])
        with col_search:
            search_serial = st.text_input(
                "Enter Serial Number",
                placeholder="Type serial number to search...",
                key="serial_search"
            )
        with col_btn:
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
        
        if search_clicked and search_serial:
            with st.spinner(f"Searching for serial: {search_serial}..."):
                journey_df = search_serial_number(search_serial)
                
                if not journey_df.empty:
                    st.success(f"✅ Found {len(journey_df)} record(s) for serial: **{search_serial}**")
                    
                    # Display journey timeline
                    st.markdown("#### 📋 Complete Journey")
                    
                    for idx, row in journey_df.iterrows():
                        # Detect issues for this record
                        issues_found = []
                        if row['has_shop_mismatch'] == 'YES':
                            issues_found.append("🔴 Shop Mismatch")
                        if row['has_vehicle_mismatch'] == 'YES':
                            issues_found.append("🔴 Vehicle Mismatch")
                        if row['verified_in_main_db'] == 'NO':
                            issues_found.append("🔴 Not in Main DB")
                        if row['status'] in ['Not Sold', 'Not Offloaded', 'Not Loaded to Shop']:
                            issues_found.append(f"⚠️ {row['status']}")
                        
                        # Title with issue indicator
                        title = f"🔍 Journey Record {idx + 1}"
                        if issues_found:
                            title += f" ⚠️ {len(issues_found)} Issue(s) Detected"
                        
                        with st.expander(title, expanded=False):
                            # Show issues at the top if any
                            if issues_found:
                                st.error("**⚠️ Issues Detected in This Journey:**")
                                for issue in issues_found:
                                    st.markdown(f"- {issue}")
                                st.markdown("---")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**🏭 Warehouse Receipt**")
                                st.write(f"**GRN Date:** {row['wh_grn_date'] or 'N/A'}")
                                st.write(f"**Warehouse:** {row['wh_received_warehouse'] or 'N/A'}")
                                st.write(f"**Supplier:** {row['supplier_name'] or 'N/A'}")
                                st.write(f"**Inbound Type:** {row['inbound_type'] or 'N/A'}")
                            
                            with col2:
                                st.markdown("**🚚 Distribution**")
                                st.write(f"**Doc Date:** {row['wh_doc_date'] or 'N/A'}")
                                st.write(f"**Loaded:** {row['loaded_datetime'] or 'N/A'}")
                                
                                # Highlight shop mismatch
                                shop_text = f"**Sent to Shop:** {row['sent_to_shop'] or 'N/A'}"
                                if row['has_shop_mismatch'] == 'YES':
                                    st.markdown(f":red[{shop_text}] ⚠️")
                                else:
                                    st.write(shop_text)
                                
                                st.write(f"**Shop Name:** {row['sent_to_shop_name'] or 'N/A'}")
                                
                                # Highlight vehicle mismatch
                                wh_vehicle_text = f"**WH Vehicle:** {row['wh_vehicle'] or 'N/A'}"
                                shop_vehicle_text = f"**Shop Vehicle:** {row['shop_vehicle'] or 'N/A'}"
                                if row['has_vehicle_mismatch'] == 'YES':
                                    st.markdown(f":red[{wh_vehicle_text}] ⚠️")
                                    st.markdown(f":red[{shop_vehicle_text}] ⚠️")
                                else:
                                    st.write(wh_vehicle_text)
                                    st.write(shop_vehicle_text)
                            
                            with col3:
                                st.markdown("**🛒 Sale**")
                                # Highlight if sold at different shop
                                sold_shop_text = f"**Sold at Shop:** {row['shop_sold'] or 'N/A'}"
                                if row['has_shop_mismatch'] == 'YES':
                                    st.markdown(f":red[{sold_shop_text}] ⚠️")
                                else:
                                    st.write(sold_shop_text)
                                
                                st.write(f"**Sale Date:** {row['sale_date'] or 'N/A'}")
                                st.write(f"**Invoice:** {row['sale_invoice'] or 'N/A'}")
                                st.write(f"**Bill No:** {row['bill_no'] or 'N/A'}")
                                st.write(f"**Cashier:** {row['cashier_name'] or 'N/A'}")
                                st.write(f"**Till:** {row['till_number'] or 'N/A'}")
                            
                            # Status and verification
                            st.markdown("---")
                            col_status1, col_status2, col_status3, col_status4 = st.columns(4)
                            
                            with col_status1:
                                status_color = {
                                    'Sold': '🟢',
                                    'Not Sold': '🟡',
                                    'Not Offloaded': '🟠',
                                    'Not Loaded to Shop': '🔴'
                                }.get(row['status'], '⚪')
                                st.metric("Status", f"{status_color} {row['status']}")
                            
                            with col_status2:
                                mismatch_color = '🔴' if row['has_shop_mismatch'] == 'YES' else '🟢'
                                st.metric("Shop Match", f"{mismatch_color} {row['has_shop_mismatch']}")
                            
                            with col_status3:
                                vehicle_color = '🔴' if row['has_vehicle_mismatch'] == 'YES' else '🟢'
                                st.metric("Vehicle Match", f"{vehicle_color} {row['has_vehicle_mismatch']}")
                            
                            with col_status4:
                                db_color = '🟢' if row['verified_in_main_db'] == 'YES' else '🔴'
                                st.metric("In Main DB", f"{db_color} {row['verified_in_main_db']}")
                            
                            # Item details
                            st.markdown("**📦 Item Information**")
                            st.write(f"**Item Code:** {row['vc_item_code'] or 'N/A'}")
                            st.write(f"**Description:** {row['vc_item_desc'] or 'N/A'}")
                            st.write(f"**Selling Price:** {row['nu_selling_price'] or 'N/A'}")
                            
                            # Serial numbers at different stages
                            st.markdown("**🔢 Serial Numbers**")
                            st.write(f"**WH Serial:** {row['wh_serial'] or 'N/A'}")
                            st.write(f"**Offload Serial:** {row['offload_serial'] or 'N/A'}")
                            st.write(f"**Sold Serial:** {row['sold_serial'] or 'N/A'}")
                else:
                    st.warning(f"⚠️ No records found for serial number: **{search_serial}**")
        
        st.markdown("---")
        
        # ============================================================
        # JOURNEY OVERVIEW METRICS
        # ============================================================
        st.markdown("### 📊 Journey Overview (All Serials)")
        
        overview_data = get_serial_journey_overview(start_date, end_date)
        
        if overview_data:
            # Top metrics row
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
            
            # Second metrics row - Issues
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
                # Coerce values to integers safely (handle None or non-numeric values)
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
        # SUPPLIER ANALYSIS
        # ============================================================
        with st.expander("🏭 Supplier Performance Analysis", expanded=False):
            supplier_df = get_supplier_analysis(start_date, end_date)
            
            if not supplier_df.empty:
                st.markdown("#### 📊 Supplier-wise Serial Tracking Performance")
                
                # Visual chart
                fig_supplier = go.Figure()
                
                fig_supplier.add_trace(go.Bar(
                    name='Total Serials',
                    x=supplier_df['supplier_name'],
                    y=supplier_df['total_serials'],
                    marker_color='#3498db',
                    yaxis='y',
                    offsetgroup=1
                ))
                
                fig_supplier.add_trace(go.Scatter(
                    name='Sale Rate %',
                    x=supplier_df['supplier_name'],
                    y=supplier_df['sale_rate'],
                    marker_color='#27ae60',
                    yaxis='y2',
                    mode='lines+markers',
                    line=dict(width=3)
                ))
                
                fig_supplier.update_layout(
                    title="Supplier Performance: Volume vs Sale Rate",
                    xaxis=dict(title="Supplier", tickangle=-45),
                    yaxis=dict(title="Total Serials", side='left'),
                    yaxis2=dict(title="Sale Rate %", overlaying='y', side='right', range=[0, 100]),
                    height=400,
                    hovermode='x unified',
                    barmode='group'
                )
                
                st.plotly_chart(fig_supplier, use_container_width=True)
                
                # Data table with styling
                display_supplier_df = supplier_df[[
                    'supplier_name', 'total_serials', 'offloaded_count', 'sold_count',
                    'offload_rate', 'sale_rate', 'shop_mismatches', 'vehicle_mismatches'
                ]].copy()
                
                display_supplier_df.columns = [
                    'Supplier', 'Total Serials', 'Offloaded', 'Sold',
                    'Offload %', 'Sale %', 'Shop Issues', 'Vehicle Issues'
                ]
                
                st.dataframe(display_supplier_df, use_container_width=True, height=300)
                
                # Download
                csv_supplier = supplier_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Supplier Analysis",
                    data=csv_supplier,
                    file_name=f"supplier_analysis_{start_date}_{end_date}.csv",
                    mime="text/csv",
                    key="download_supplier_analysis"
                ),

            else:
                st.info("No supplier data available for the selected period")
        
        # ============================================================
        # BRAND ANALYSIS
        # ============================================================
        with st.expander("🏷️ Brand Performance Analysis", expanded=False):
            brand_df = get_brand_analysis(start_date, end_date)
            
            if not brand_df.empty:
                st.markdown("#### 📊 Brand-wise Serial Tracking Performance")
                
                # Visual chart
                fig_brand = go.Figure()
                
                fig_brand.add_trace(go.Bar(
                    name='Total Serials',
                    x=brand_df['brand'],
                    y=brand_df['total_serials'],
                    marker_color='#9b59b6'
                ))
                
                fig_brand.add_trace(go.Scatter(
                    name='Sale Rate %',
                    x=brand_df['brand'],
                    y=brand_df['sale_rate'],
                    marker_color='#e74c3c',
                    yaxis='y2',
                    mode='lines+markers',
                    line=dict(width=3)
                ))
                
                fig_brand.update_layout(
                    title="Brand Performance: Volume vs Sale Rate",
                    xaxis=dict(title="Brand", tickangle=-45),
                    yaxis=dict(title="Total Serials", side='left'),
                    yaxis2=dict(title="Sale Rate %", overlaying='y', side='right', range=[0, 100]),
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_brand, use_container_width=True)
                
                # Data table
                display_brand_df = brand_df[[
                    'brand', 'total_serials', 'offloaded_count', 'sold_count',
                    'offload_rate', 'sale_rate', 'shops_sent_to', 'avg_selling_price'
                ]].copy()
                
                display_brand_df.columns = [
                    'Brand', 'Total Serials', 'Offloaded', 'Sold',
                    'Offload %', 'Sale %', 'Shops', 'Avg Price (₵)'
                ]
                
                st.dataframe(display_brand_df, use_container_width=True, height=300)
                
                # Download
                csv_brand = brand_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Brand Analysis",
                    data=csv_brand,
                    file_name=f"brand_analysis_{start_date}_{end_date}.csv",
                    mime="text/csv",
                    key="download_brand_analysis"
                )

            else:
                st.info("No brand data available for the selected period")
        
        st.markdown("---")
        
        # ============================================================
        # GAP ANALYSIS BY SHOP (WITH DRILL-DOWN)
        # ============================================================
        st.markdown("### 📈 Shop-wise Gap Analysis with Drill-Down")
        st.info("💡 Select a shop to see detailed serial journey with personnel info")
        
        gap_df = get_gap_analysis_by_shop(start_date, end_date)
        
        if not gap_df.empty:
            # Add calculated columns
            gap_df['offload_rate'] = (gap_df['offloaded_count'] / gap_df['loaded_count'] * 100).round(1)
            gap_df['sale_rate'] = (gap_df['sold_count'] / gap_df['loaded_count'] * 100).round(1)
            
            # Shop selector
            selected_gap_shop = st.selectbox(
                "Select Shop for Details",
                ['All'] + gap_df['vc_shop_code'].tolist(),
                key='gap_shop_selector'
            )
            
            if selected_gap_shop == 'All':
                # Show summary chart
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
                
                # Data table
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
                
                # Download
                csv = gap_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Gap Analysis",
                    data=csv,
                    file_name=f"gap_analysis_{start_date}_{end_date}.csv",
                    mime="text/csv",
                    key="download_gap_analysis"
                ),

            else:
                # ============================================================
                # DETAILED SERIAL JOURNEY FOR SELECTED SHOP
                # ============================================================
                shop_data = gap_df[gap_df['vc_shop_code'] == selected_gap_shop].iloc[0]
                
                st.markdown(f"### 🏪 {selected_gap_shop} - {shop_data['shop_name']}")
                
                # Summary metrics
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
                
                # Load detailed serial data
                with st.spinner(f"Loading detailed journey for {selected_gap_shop}..."):
                    shop_serials_df = get_shop_serial_details(start_date, end_date, selected_gap_shop)
                
                if not shop_serials_df.empty:
                    st.markdown(f"#### 📋 Complete Serial Journey - {len(shop_serials_df):,} Records")
                    st.caption(f"Showing all serials sent to {selected_gap_shop} with complete journey details including personnel info")
                    
                    # Interactive filters for drill-down
                    col_filter1, col_filter2 = st.columns(2)
                    
                    with col_filter1:
                        # Check if issue_category column exists
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
                        # Check if supplier column exists
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
                    
                    # Apply filters (only if column exists)
                    filtered_df = shop_serials_df.copy()
                    
                    if issue_filter and 'issue_category' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['issue_category'].isin(issue_filter)]
                    
                    if supplier_filter and 'supplier' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['supplier'].isin(supplier_filter)]
                    
                    st.info(f"Showing {len(filtered_df):,} of {len(shop_serials_df):,} records")
                    
                    # Display detailed journey table - handle missing columns gracefully
                    # Define all possible columns with their display names
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
                    
                    # Build display columns list - only include columns that exist
                    display_cols = []
                    display_names = []
                    
                    for col, name in all_possible_cols.items():
                        if col in filtered_df.columns:
                            display_cols.append(col)
                            display_names.append(name)
                    
                    # Create display dataframe with only available columns
                    if display_cols:
                        display_df = filtered_df[display_cols].copy()
                        display_df.columns = display_names
                    else:
                        st.error("No columns available to display")
                        display_df = pd.DataFrame()
                    
                    # Format timestamps
                    timestamp_cols = ['📅 Loaded', '📅 Offloaded', '📅 Sold', 'WH GRN Date']
                    for col in timestamp_cols:
                        if col in display_df.columns:
                            display_df[col] = pd.to_datetime(display_df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
                    
                    # Display the dataframe if we have columns
                    if not display_df.empty and len(display_df.columns) > 0:
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            height=500
                        )
                    else:
                        st.warning("No data available to display after filtering")
                    
                    # Analysis summary for this shop
                    st.markdown("#### 📊 Quick Stats for This Shop")
                    
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        if 'issue_category' in filtered_df.columns:
                            issue_counts = filtered_df['issue_category'].value_counts()
                            st.markdown("**Top Issue:**")
                            if not issue_counts.empty:
                                st.write(f"{issue_counts.index[0]}: {issue_counts.values[0]:,}")
                            else:
                                st.write("No issues")
                        else:
                            st.markdown("**Top Issue:**")
                            st.write("N/A")
                    
                    with col_stat2:
                        if 'supplier' in filtered_df.columns:
                            top_supplier = filtered_df['supplier'].value_counts().head(1)
                            st.markdown("**Top Supplier:**")
                            if not top_supplier.empty:
                                st.write(f"{top_supplier.index[0]}: {top_supplier.values[0]:,}")
                            else:
                                st.write("N/A")
                        else:
                            st.markdown("**Top Supplier:**")
                            st.write("N/A")
                    
                    with col_stat3:
                        if 'nu_selling_price' in filtered_df.columns:
                            avg_price = filtered_df['nu_selling_price'].mean()
                            st.markdown("**Avg Price:**")
                            st.write(f"₵{avg_price:,.2f}" if not pd.isna(avg_price) else "N/A")
                        else:
                            st.markdown("**Avg Price:**")
                            st.write("N/A")
                    
                    with col_stat4:
                        if 'serial_check' in filtered_df.columns:
                            verified_count = (filtered_df['serial_check'] == 'Y').sum()
                            verified_pct = (verified_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
                            st.markdown("**Verified Rate:**")
                            st.write(f"{verified_pct:.1f}% ({verified_count:,}/{len(filtered_df):,})")
                        else:
                            st.markdown("**Verified Rate:**")
                            st.write("N/A")
                    
                    # Download button
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
            st.warning(f"⚠️ Found **{len(unverified_df):,}** serial numbers not verified in main database")
            
            st.dataframe(unverified_df, use_container_width=True, height=400)
            
            # Download
            csv = unverified_df.to_csv(index=False)
            st.download_button(
                "📥 Download Unverified Serials",
                data=csv,
                file_name=f"unverified_serials_{start_date}_{end_date}.csv",
                mime="text/csv",
                key="download_unverified_serials"
            )

        else:
            st.success("✅ All serial numbers verified in main database!")

if __name__ == "__main__":
    main()

