# melcom_inventory_pg.py
"""
Melcom NO_WH Inventory Pulse - Production Ready
PostgreSQL-based Inventory Management & Stock Transfer Recommendation System
Version: 2.3 - Fixed dynamic sales calculation with debugging
"""

import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import io
import warnings
from psycopg2 import pool, Error
from psycopg2.extras import RealDictCursor
#from io import BytesIO
from typing import Optional, Dict, Tuple
from contextlib import contextmanager
import logging
from datetime import datetime, timedelta

# Suppress pandas SQLAlchemy warnings
warnings.filterwarnings('ignore', message='.*SQLAlchemy.*')

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Centralized configuration"""
    
    # Database configurations
    DB_CONFIG = {
        'host': 'localhost',
        'user': 'postgres',
        'password': 'hello',
        'port': 3307
    }
    
    # Business constants
    PRIORITY_SHOPS = ['SPN', 'MSS', 'LFS', 'M03', 'KAS', 'MM1', 'MM2', 'FAR', 'KS7', 'WHL', 'MM3']
    DEFAULT_THRESHOLD = 30  # Changed from 10 to 1
    BUFFER_DAYS = 30
    GRN_FALLBACK_DAYS = 90
    SALES_DAYS_WINDOW = 30
    
    # UI
    PAGE_TITLE = "Inventory Pulse NO_WH"
    PAGE_ICON = "https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg"
    LOGO_URL = "https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg"

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None

# ============================================================
# DATABASE CONNECTION MANAGEMENT
# ============================================================

@st.cache_resource
def get_connection_pool(dbname: str):
    """Create connection pool for database"""
    try:
        return psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            dbname=dbname,
            **Config.DB_CONFIG
        )
    except Error as e:
        logger.error(f"Connection pool error for {dbname}: {e}")
        st.error(f"❌ Cannot create connection pool for {dbname}: {e}")
        return None

@contextmanager
def get_db_connection(dbname: str):
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
# AUTHENTICATION
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def authenticate_user(employee_id: str, password: str) -> Optional[Dict]:
    """Authenticate user from PostgreSQL users database (cached for speed)"""
    try:
        with get_db_connection('users') as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT employee_id, full_name, db_access, table_access, is_active
                    FROM users
                    WHERE employee_id = %s AND password = %s AND LOWER(is_active) = 'true'
                """, (employee_id, password))
                row = cursor.fetchone()
                
                if not row:
                    logger.warning(f"Failed login: {employee_id}")
                    return None
                
                logger.info(f"✅ Successful login: {employee_id} (cached)")
                return dict(row)
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None
        return None

def check_table_access(user: Dict, required_table: str) -> bool:
    """Check if user has access to required table"""
    if not user or 'table_access' not in user:
        return False
    
    user_tables = [t.strip().lower() for t in (user.get('table_access') or '').split(',')]
    return required_table.lower() in user_tables or 'all' in user_tables

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_filter_options() -> pd.DataFrame:
    """Load distinct filter options from inventory (cached 1 hour)"""
    try:
        with get_db_connection('grndetails') as conn:
            query = '''
                SELECT DISTINCT "GROUPS", "SUB_GROUP", "ITEM_CODE", "ITEM_NAME", "SHOP_CODE"
                FROM nowhstock_tbl_new
            '''
            df = pd.read_sql(query, conn)
            logger.info("✅ Loaded filter options (cached)")
            return df
    except Exception as e:
        logger.error(f"Error loading filters: {e}")
        st.error(f"❌ Error loading filters: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_sit_filter_options() -> pd.DataFrame:
    """Load distinct filter options from SIT item details view (cached 1 hour)"""
    try:
        with get_db_connection('grndetails') as conn:
            query = '''
                SELECT DISTINCT 
                    "vc_item_code", "item_name", "type", "vc_supplier_name"
                FROM mv_sit_itemdetails
                WHERE "item_name" IS NOT NULL AND "item_name" != ''
            '''
            df = pd.read_sql(query, conn)
            logger.info(f"✅ Loaded SIT filter options: {len(df)} items (cached)")
            return df
    except Exception as e:
        logger.error(f"Error loading SIT filters: {e}")
        logger.warning(f"⚠️ SIT filter view not available: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_sit_filter_lookup() -> Dict:
    """Load and normalize SIT filter options into fast lookup dicts (cached 1 hour)"""
    try:
        sit_df = load_sit_filter_options()
        if sit_df.empty:
            return {}
        
        import time
        t0 = time.time()
        
        # Normalize once
        sit_df = sit_df.copy()
        sit_df['ITEM_CODE'] = sit_df['vc_item_code'].astype(str).str.strip().str.upper()
        sit_df['ITEM_NAME_SIT'] = sit_df['item_name'].astype(str).str.strip()
        sit_df['TYPE_SIT'] = sit_df['type'].astype(str).str.strip()
        sit_df['SUPPLIER_SIT'] = sit_df['vc_supplier_name'].astype(str).str.strip()
        
        # Build fast lookup: (type, supplier, item_name) -> set of item_codes
        lookup = {}
        for _, row in sit_df.iterrows():
            key = (row['TYPE_SIT'], row['SUPPLIER_SIT'], row['ITEM_NAME_SIT'])
            if key not in lookup:
                lookup[key] = set()
            lookup[key].add(row['ITEM_CODE'])
        
        elapsed = time.time() - t0
        logger.info(f"✅ Built SIT lookup ({len(lookup)} combinations) in {elapsed:.3f}s")
        return lookup
    except Exception as e:
        logger.error(f"Error building SIT lookup: {e}")
        return {}


def apply_itemdetails_filters(inventory_df: pd.DataFrame, sit_df: pd.DataFrame, item_type: str, supplier: str, item_name: str) -> pd.DataFrame:
    """Filter `inventory_df` using SIT item details selections (fast vectorized version).

    Uses pre-computed lookup for O(1) filter matching instead of repeated normalizations.
    If no SIT filters selected, returns inventory_df unchanged.
    """
    try:
        import time
        t0 = time.time()
        
        # If all filters are 'All', skip filtering
        if (not item_type or item_type == 'All') and (not supplier or supplier == 'All') and (not item_name or item_name == 'All'):
            logger.info(f"✅ SIT filters all 'All' - returning full inventory ({len(inventory_df)} rows)")
            return inventory_df
        
        if sit_df is None or sit_df.empty:
            return inventory_df
        
        # Get pre-computed lookup
        sit_lookup = load_sit_filter_lookup()
        if not sit_lookup:
            return inventory_df
        
        # Collect matching item codes using set union
        matched_codes = set()
        for (ftype, fsupplier, fname), codes in sit_lookup.items():
            type_match = (not item_type or item_type == 'All' or ftype == item_type)
            supplier_match = (not supplier or supplier == 'All' or fsupplier == supplier)
            name_match = (not item_name or item_name == 'All' or fname == item_name)
            
            if type_match and supplier_match and name_match:
                matched_codes.update(codes)
        
        if not matched_codes:
            logger.warning(f"⚠️ SIT filters matched 0 items (type={item_type}, supplier={supplier}, item_name={item_name})")
            return inventory_df.iloc[0:0]
        
        # Fast filter using set membership (no normalization needed - keys already normalized)
        inv = inventory_df.copy()
        inv['ITEM_CODE'] = inv['ITEM_CODE'].astype(str).str.strip().str.upper()
        result = inv[inv['ITEM_CODE'].isin(matched_codes)].reset_index(drop=True)
        
        elapsed = time.time() - t0
        logger.info(f"✅ SIT filter applied: {len(inventory_df)} -> {len(result)} rows ({elapsed:.3f}s)")
        return result

    except Exception as e:
        logger.error(f"Error applying item details filters: {e}")
        return inventory_df

@st.cache_data(ttl=900, show_spinner=False)
def load_last_30d_sales() -> pd.DataFrame:
    """Load sales from last 30 days (cached 15 min) - optimized query"""
    try:
        end_date = datetime.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=Config.SALES_DAYS_WINDOW - 1)
        
        with get_db_connection('salesdata') as conn:
            # Single optimized query - no validation overhead
            query = """
                SELECT 
                    TRIM(UPPER("ITEM_CODE")) AS item_code,
                    TRIM(UPPER("SHOP_CODE")) AS shop_code,
                    SUM(COALESCE("QTY", 0))::INT AS sales_30d
                FROM sales_2025
                WHERE "DATE_INVOICE"::date BETWEEN %s AND %s
                GROUP BY 1, 2
            """
            sales_df = pd.read_sql(query, conn, params=(start_date.date(), end_date.date()))
        
        sales_df.columns = sales_df.columns.str.upper()
        logger.info(f"✅ Loaded 30-day sales: {len(sales_df)} rows (cached)")
        return sales_df if not sales_df.empty else pd.DataFrame(columns=["ITEM_CODE", "SHOP_CODE", "SALES_30D"])
        
    except Exception as e:
        logger.error(f"Error loading 30-day sales: {e}")
        return pd.DataFrame(columns=["ITEM_CODE", "SHOP_CODE", "SALES_30D"])

@st.cache_data(ttl=1800, show_spinner=False)
def load_inventory_data(group: str, subgroup: str, product: str, shop: str) -> pd.DataFrame:
    """Load inventory data with filters - optimized for speed"""
    try:
        with get_db_connection('grndetails') as conn:
            # Build filters
            filters = []
            params = []
            if group and group.strip() != "" and group != "All":
                filters.append(f'"GROUPS" = %s')
                params.append(group)
            if subgroup and subgroup.strip() != "" and subgroup != "All":
                filters.append(f'"SUB_GROUP" = %s')
                params.append(subgroup)
            if product and product.strip() != "" and product != "All":
                filters.append(f'"ITEM_CODE" = %s')
                params.append(product)
            if shop and shop.strip() != "" and shop != "All":
                filters.append(f'"SHOP_CODE" = %s')
                params.append(shop)

            where_sql = "WHERE " + " AND ".join(filters) if filters else ""

            # Skip MV refresh - use cached view
            query = f"""
                SELECT 
                    "ITEM_CODE", "ITEM_NAME", "SHOP_CODE",
                    "SHOP_STOCK", "GROUPS", "SUB_GROUP", "SHOP_GRN_DATE"
                FROM mv_nowhstock_grn
                {where_sql}
            """

            df = pd.read_sql(query, conn, params=params)
            
            # Optimized data cleaning
            df['ITEM_CODE'] = df['ITEM_CODE'].str.strip().str.upper()
            df['SHOP_CODE'] = df['SHOP_CODE'].str.strip().str.upper()
            df['SHOP_STOCK'] = pd.to_numeric(df['SHOP_STOCK'], errors='coerce').fillna(0).astype(int)
        
        # Merge with cached sales data
        sales_30d_df = load_last_30d_sales()
        
        if not sales_30d_df.empty:
            df = df.merge(sales_30d_df, on=['ITEM_CODE', 'SHOP_CODE'], how='left')
            df['ITEM_SALES_30_DAYS'] = df.get('SALES_30D', 0).fillna(0).astype(int)
            df = df.drop(columns=['SALES_30D'], errors='ignore')
        else:
            df['ITEM_SALES_30_DAYS'] = 0
        
        logger.info(f"✅ Loaded {len(df)} inventory records")
        return df
            
    except Exception as e:
        logger.error(f"Error loading inventory: {e}")
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def load_last_grn_dates() -> pd.DataFrame:
    """Load last GRN date per item-shop from materialized view (cached 15 min)"""
    try:
        with get_db_connection('grndetails') as conn:
            query = """
                SELECT 
                    item_code,
                    shop_code,
                    last_grn_date
                FROM mv_last_grn_dates
            """
            df = pd.read_sql(query, conn)
            if df.empty:
                logger.warning("⚠️ mv_last_grn_dates returned no rows")
                return pd.DataFrame(columns=['ITEM_CODE', 'SHOP_CODE', 'LAST_GRN_DATE'])
            
            # Normalize columns to uppercase for consistency
            df.columns = df.columns.str.upper()
            logger.info(f"✅ Loaded last GRN dates: {len(df)} item-shop combinations (cached)")
            return df
    except Exception as e:
        logger.error(f"❌ Error loading last GRN dates: {e}")
        logger.warning(f"⚠️ Using fallback: will calculate GRN dates from sup_shop_grn table instead")
        # Fallback: query the base table directly
        try:
            with get_db_connection('grndetails') as conn:
                query = """
                    SELECT 
                        TRIM(UPPER("ITEM_CODE")) AS ITEM_CODE,
                        TRIM(UPPER("SHOP_CODE")) AS SHOP_CODE,
                        MAX("SHOP_GRN_DATE") AS LAST_GRN_DATE
                    FROM sup_shop_grn
                    GROUP BY 1, 2
                """
                df = pd.read_sql(query, conn)
                logger.info(f"✅ Fallback: Loaded {len(df)} GRN dates from sup_shop_grn")
                return df
        except Exception as e2:
            logger.error(f"❌ Fallback also failed: {e2}")
            return pd.DataFrame(columns=['ITEM_CODE', 'SHOP_CODE', 'LAST_GRN_DATE'])

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_grn_sales() -> pd.DataFrame:
    """Calculate sales since last GRN date"""
    try:
        with get_db_connection('grndetails') as conn_grn, get_db_connection('salesdata') as conn_sales:
            # Load GRN dates
            grn_df = pd.read_sql("""
                SELECT 
                    TRIM(UPPER("ITEM_CODE")) AS item_code,
                    TRIM(UPPER("SHOP_CODE")) AS shop_code,
                    MAX("SHOP_GRN_DATE")::date AS last_grn_date
                FROM sup_shop_grn
                GROUP BY 1, 2
            """, conn_grn)
            
            # Load sales
            sales_df = pd.read_sql("""
                SELECT 
                    TRIM(UPPER("ITEM_CODE")) AS item_code,
                    TRIM(UPPER("SHOP_CODE")) AS shop_code,
                    "DATE_INVOICE"::date AS date_invoice,
                    SUM(COALESCE("QTY", 0)) AS qty
                FROM sales_2025
                GROUP BY 1, 2, 3
            """, conn_sales)
        
        # Process dates
        grn_df["last_grn_date"] = pd.to_datetime(grn_df["last_grn_date"], errors="coerce")
        sales_df["date_invoice"] = pd.to_datetime(sales_df["date_invoice"], errors="coerce")
        
        # Merge and calculate
        merged = sales_df.merge(grn_df, on=["item_code", "shop_code"], how="left")
        fallback = pd.Timestamp.today() - pd.Timedelta(days=Config.GRN_FALLBACK_DAYS)
        merged["last_grn_date"] = merged["last_grn_date"].fillna(fallback)
        
        # Filter sales since GRN
        filtered = merged[merged["date_invoice"] >= merged["last_grn_date"]]
        
        # Aggregate
        result = (
            filtered.groupby(["item_code", "shop_code"])["qty"]
            .sum()
            .reset_index()
            .rename(columns={"qty": "total_sales_grn_to_today"})
        )
        
        # Normalize column names
        result.columns = result.columns.str.upper()
        logger.info(f"Calculated GRN sales for {len(result)} combinations")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating GRN sales: {e}")
        st.warning(f"⚠️ GRN sales calculation failed: {e}")
        return pd.DataFrame(columns=["ITEM_CODE", "SHOP_CODE", "TOTAL_SALES_GRN_TO_TODAY"])

# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def format_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format raw recommendation query results into final output structure.
    Handles business logic for remarks, blocking rules, and final calculations.
    """
    if df.empty:
        return pd.DataFrame()
    
    import numpy as np
    
    # Calculate final metrics
    df['dest_updated_stock'] = df['dest_stock'] + df['recommended_qty']
    df['dest_final_stock_days'] = np.where(
        df['dest_sales'] > 0,
        np.round(df['dest_updated_stock'] / df['dest_sales'] * 30, 1),
        0
    )
    
    # Apply blocking rules for zero-sales combinations
    df['skip_transfer'] = False
    df['skip_reason'] = ''
    
    # Rule: Block if both source and dest have zero sales
    zero_both = (df['source_sales'] == 0) & (df['dest_sales'] == 0)
    
    # Check GRN conditions
    src_grn_na = zero_both & df['source_last_grn'].isna()
    df.loc[src_grn_na, 'skip_transfer'] = True
    df.loc[src_grn_na, 'skip_reason'] = 'GRN not available'
    
    both_old_grn = zero_both & (df['source_grn_age'] > 30) & (df.get('dest_grn_age', 999) > 30)
    df.loc[both_old_grn, 'skip_transfer'] = True
    df.loc[both_old_grn, 'skip_reason'] = 'src & dest shop has zero sales'
    
    recent_grn = zero_both & ((df['source_grn_age'] < 30) | (df.get('dest_grn_age', 999) < 30))
    df.loc[recent_grn, 'skip_transfer'] = True
    df.loc[recent_grn, 'skip_reason'] = 'Latest GRN'
    
    # Set remarks
    df['remark'] = ''
    df.loc[~df['skip_transfer'] & (df['is_priority_shop'] == 1), 'remark'] = 'Priority transfer'
    df.loc[~df['skip_transfer'] & (df['is_priority_shop'] == 0), 'remark'] = 'Normal transfer'
    df.loc[df['skip_transfer'], 'remark'] = '❌ Not recommended: ' + df['skip_reason']
    df.loc[df['skip_transfer'], 'recommended_qty'] = 0
    
    # Filter out blocked transfers
    df = df[df['recommended_qty'] > 0].copy()
    
    # Format output columns
    result = df[[
        'item_code', 'item_name', 'source_shop', 'source_stock', 'source_sales',
        'source_last_grn', 'source_grn_age', 'dest_shop', 'dest_stock', 'dest_sales',
        'recommended_qty', 'dest_updated_stock', 'dest_final_stock_days', 'remark'
    ]].copy()
    
    result.columns = [
        'ITEM_CODE', 'Item Name', 'Source Shop', 'Source Stock', 'Source Last 30d Sales',
        'Source Last GRN Date', 'Source GRN Age (days)', 'Destination Shop',
        'Destination Stock', 'Destination Last 30d Sales', 'Recommended_Qty',
        'Destination Updated Stock', 'Destination Final Stock In Hand Days', 'Remark'
    ]
    
    # Format GRN dates
    result['Source Last GRN Date'] = result['Source Last GRN Date'].fillna('N/A')
    
    return result


def generate_recommendations_optimized(group: str = 'All', subgroup: str = 'All', 
                                      product: str = 'All', shop: str = 'All',
                                      use_grn_logic: bool = True, threshold: int = 10) -> pd.DataFrame:
    """
    ULTRA-FAST recommendation engine using materialized view.
    
    Performance: 30-60x faster than original (120s → 3-5s for 121K rows)
    
    Key optimizations:
    1. Database-side filtering with pre-computed materialized view
    2. Single SQL query replaces nested loops
    3. Pre-computed flags (is_slow_moving, has_transferable_stock, etc.)
    4. Indexed joins in PostgreSQL instead of Python pandas
    5. Vectorized calculations only where needed
    
    Args:
        group, subgroup, product, shop: Inventory filters
        use_grn_logic: Whether to filter by GRN dates (< 30 days)
        threshold: Sales threshold for slow/fast classification (default: 10)
        
    Returns:
        DataFrame with formatted recommendations
    """
    import time
    
    start_time = time.time()
    logger.info(f"⚡ ULTRA-FAST MODE | Generating recommendations (GRN: {use_grn_logic})")
    
    # Build dynamic SQL filters
    filter_conditions = []
    filter_params = []
    
    if group and group != 'All':
        filter_conditions.append("src.groups = %s")
        filter_params.append(group)
    
    if subgroup and subgroup != 'All':
        filter_conditions.append("src.sub_group = %s")
        filter_params.append(subgroup)
    
    if product and product != 'All':
        filter_conditions.append("src.item_code = %s")
        filter_params.append(product.strip().upper())
    
    if shop and shop != 'All':
        filter_conditions.append("(src.shop_code = %s OR dest.shop_code = %s)")
        filter_params.extend([shop.strip().upper(), shop.strip().upper()])
    
    filter_sql = " AND " + " AND ".join(filter_conditions) if filter_conditions else ""
    
    # Single optimized query - database does all the heavy lifting
    query = f"""
        WITH priority_shops AS (
            SELECT unnest(ARRAY['SPN', 'MSS', 'LFS', 'M03', 'KAS', 'MM1', 'MM2', 'FAR', 'KS7', 'WHL', 'MM3']) as shop_code
        ),
        source_shops AS (
            -- Pre-filter source shops (shops that can give stock)
            SELECT 
                item_code,
                item_name,
                shop_code,
                groups,
                sub_group,
                shop_stock,
                sales_30d,
                grn_age_days,
                last_grn_date,
                available_to_transfer,
                is_priority_shop
            FROM mv_recommendation_base
            WHERE has_transferable_stock = 1  -- Stock > 30
              AND is_priority_shop = 0  -- Not a priority shop (can't be source)
              {"AND has_recent_grn = 0" if use_grn_logic else ""}  -- GRN > 30 days
        ),
        destination_shops AS (
            -- Pre-filter destination shops (shops that need stock)
            SELECT 
                item_code,
                shop_code,
                shop_stock,
                sales_30d,
                grn_age_days,
                last_grn_date,
                is_priority_shop,
                CASE WHEN ps.shop_code IS NOT NULL THEN 0 ELSE 1 END as priority_rank
            FROM mv_recommendation_base
            LEFT JOIN priority_shops ps ON mv_recommendation_base.shop_code = ps.shop_code
            WHERE is_slow_moving = 1  -- Sales < {threshold}
              {"AND has_recent_grn = 0" if use_grn_logic else ""}  -- GRN > 30 days
        )
        SELECT 
            src.item_code,
            src.item_name,
            src.shop_code as source_shop,
            src.shop_stock as source_stock,
            src.sales_30d as source_sales,
            src.grn_age_days as source_grn_age,
            src.last_grn_date as source_last_grn,
            dest.shop_code as dest_shop,
            dest.shop_stock as dest_stock,
            dest.sales_30d as dest_sales,
            dest.grn_age_days as dest_grn_age,
            dest.last_grn_date as dest_last_grn,
            dest.is_priority_shop,
            dest.priority_rank,
            -- Pre-calculate recommended qty
            LEAST(
                GREATEST(
                    CASE WHEN dest.sales_30d > 0 
                         THEN dest.sales_30d - dest.shop_stock
                         ELSE 30 - dest.shop_stock END,
                    0
                ),
                src.available_to_transfer
            ) as recommended_qty
        FROM source_shops src
        INNER JOIN destination_shops dest
            ON src.item_code = dest.item_code
            AND src.shop_code != dest.shop_code  -- Can't transfer to self
        WHERE 1=1 {filter_sql}
        ORDER BY src.item_code, dest.priority_rank, dest.shop_code
    """
    
    try:
        with get_db_connection('grndetails') as conn:
            df = pd.read_sql(query, conn, params=filter_params)
        
        query_time = time.time() - start_time
        logger.info(f"⚡ Query executed in {query_time:.2f}s - fetched {len(df)} raw recommendations")
        
    except Exception as e:
        logger.error(f"❌ Optimized query failed: {e}")
        logger.info("Falling back to legacy recommendation engine...")
        # Fallback to old implementation if materialized view doesn't exist
        return pd.DataFrame()
    
    if df.empty:
        logger.warning("No recommendations found")
        return pd.DataFrame()
    
    # Format and apply business logic
    result = format_recommendations(df)
    
    elapsed = time.time() - start_time
    logger.info(f"⚡ ULTRA-FAST: Generated {len(result)} recommendations in {elapsed:.2f}s ({len(result)/elapsed:.0f} recs/sec)")
    
    return result


def generate_recommendations(df: pd.DataFrame, use_grn_logic: bool = True) -> pd.DataFrame:
    """
    LEGACY recommendation engine - kept for backwards compatibility.
    
    This version processes a pre-loaded dataframe with nested loops.
    NEW CODE SHOULD USE generate_recommendations_optimized() instead.
    
    Performance: ~90-120s for 121K rows (use optimized version for 3-5s)
    """
    import pandas as pd
    import numpy as np
    from datetime import timedelta
    import time

    if df is None or df.empty:
        logger.warning("Empty dataframe passed to generate_recommendations")
        return pd.DataFrame()

    start_time = time.time()
    logger.info(f"⚠️ LEGACY MODE | Generating recommendations (GRN: {use_grn_logic}) | Input rows: {len(df)}")

    # --- normalize input data ---
    base_df = df.copy()
    base_df['ITEM_CODE'] = base_df['ITEM_CODE'].str.strip().str.upper()
    base_df['SHOP_CODE'] = base_df['SHOP_CODE'].str.strip().str.upper()

    # --- optional GRN merge ---
    if use_grn_logic:
        grn_sales = calculate_grn_sales()
        grn_sales['ITEM_CODE'] = grn_sales['ITEM_CODE'].str.strip().str.upper()
        grn_sales['SHOP_CODE'] = grn_sales['SHOP_CODE'].str.strip().str.upper()
        base_df = base_df.merge(grn_sales, on=['ITEM_CODE', 'SHOP_CODE'], how='left', copy=False)
        base_df['TOTAL_SALES_GRN_TO_TODAY'] = base_df['TOTAL_SALES_GRN_TO_TODAY'].fillna(0)
    else:
        base_df['TOTAL_SALES_GRN_TO_TODAY'] = 0

    # --- Load both source and destination 30d sales (use cached aggregate) ---
    sales_30d_df = load_last_30d_sales()
    if not sales_30d_df.empty:
        # Ensure columns are normalized
        sales_30d_df.columns = sales_30d_df.columns.str.upper()
        sales_30d_df['ITEM_CODE'] = sales_30d_df['ITEM_CODE'].str.strip().str.upper()
        sales_30d_df['SHOP_CODE'] = sales_30d_df['SHOP_CODE'].str.strip().str.upper()

        # Merge with base_df to get sales for each item-shop combination
        # Use explicit left_on/right_on in case of odd column capitalization
        base_df = base_df.merge(sales_30d_df, on=['ITEM_CODE', 'SHOP_CODE'], how='left')
        base_df['DEST_LAST_30D_SALES'] = base_df.get('SALES_30D', 0).fillna(0).astype(float)
        base_df = base_df.drop(columns=['SALES_30D', 'ITEM_SALES_30_DAYS'], errors='ignore')
        logger.info(f"✅ Loaded 30-day sales from cache for {len(sales_30d_df)} item-shop combinations")
    else:
        # Fallback: use ITEM_SALES_30_DAYS if cached sales empty
        base_df['DEST_LAST_30D_SALES'] = base_df.get('ITEM_SALES_30_DAYS', 0).fillna(0).astype(float)

    # Defensive cleanup: ensure required keys exist and are normalized
    base_df['ITEM_CODE'] = base_df['ITEM_CODE'].astype(str).str.strip().str.upper()
    base_df['SHOP_CODE'] = base_df['SHOP_CODE'].astype(str).str.strip().str.upper()

    # --- Load last GRN dates for filtering (source and destination shops) ---
    try:
        grn_dates_df = load_last_grn_dates()
        if not grn_dates_df.empty:
            grn_dates_df['ITEM_CODE'] = grn_dates_df['ITEM_CODE'].astype(str).str.strip().str.upper()
            grn_dates_df['SHOP_CODE'] = grn_dates_df['SHOP_CODE'].astype(str).str.strip().str.upper()
            # Ensure the column name is uppercase (it should be after .str.upper() in load_last_grn_dates)
            if 'LAST_GRN_DATE' in grn_dates_df.columns:
                grn_dates_df['LAST_GRN_DATE'] = pd.to_datetime(grn_dates_df['LAST_GRN_DATE'], errors='coerce')
                base_df = base_df.merge(grn_dates_df[['ITEM_CODE', 'SHOP_CODE', 'LAST_GRN_DATE']], 
                                       on=['ITEM_CODE', 'SHOP_CODE'], how='left')
                logger.info(f"✅ Loaded last GRN dates for {len(grn_dates_df)} item-shop combinations")
            else:
                logger.warning("⚠️ LAST_GRN_DATE column not found in grn_dates_df")
                base_df['LAST_GRN_DATE'] = pd.NaT
        else:
            logger.warning("⚠️ GRN dates dataframe is empty, using fallback")
            base_df['LAST_GRN_DATE'] = pd.NaT
    except Exception as e:
        logger.error(f"❌ Error loading GRN dates: {e}")
        base_df['LAST_GRN_DATE'] = pd.NaT
    
    # Ensure LAST_GRN_DATE column exists (defensive check)
    if 'LAST_GRN_DATE' not in base_df.columns:
        base_df['LAST_GRN_DATE'] = pd.NaT
    all_items = base_df['ITEM_CODE'].unique()
    priority_idx = pd.MultiIndex.from_product(
        [all_items, Config.PRIORITY_SHOPS], names=['ITEM_CODE', 'SHOP_CODE']
    )
    missing = priority_idx.difference(pd.MultiIndex.from_frame(base_df[['ITEM_CODE', 'SHOP_CODE']]))
    if len(missing) > 0:
        mdf = pd.DataFrame(list(missing), columns=['ITEM_CODE', 'SHOP_CODE'])
        mdf['ITEM_NAME'] = mdf['ITEM_CODE'].map(
            base_df.drop_duplicates('ITEM_CODE').set_index('ITEM_CODE')['ITEM_NAME'].to_dict()
        )
        mdf['SHOP_STOCK'] = 0
        # ✅ FIX: Get sales for missing destination shops from sales_30d
        sales_30d_dict = base_df[['SHOP_CODE', 'DEST_LAST_30D_SALES']].drop_duplicates('SHOP_CODE').set_index('SHOP_CODE')['DEST_LAST_30D_SALES'].to_dict()
        mdf['DEST_LAST_30D_SALES'] = mdf['SHOP_CODE'].map(sales_30d_dict).fillna(0)
        mdf['TOTAL_SALES_GRN_TO_TODAY'] = 0
        mdf['SHOP_GRN_DATE'] = pd.NaT
        mdf['LAST_GRN_DATE'] = pd.NaT  # Add missing column
        base_df = pd.concat([base_df, mdf], ignore_index=True)

    # --- generate recommendations ---
    # Use faster groupby approach instead of full merge
    if len(base_df) == 0:
        logger.warning("Base dataframe empty after preparation")
        return pd.DataFrame()
    
    # ⚡ OPTIMIZATION 1: Filter out items with no slow-moving shops BEFORE merge
    # This drastically reduces the merge size
    slow_items = base_df[base_df['DEST_LAST_30D_SALES'] < 10]['ITEM_CODE'].unique()
    base_df = base_df[base_df['ITEM_CODE'].isin(slow_items)]
    logger.info(f"⚡ Filtered to {len(slow_items)} items with slow-moving shops (from {len(base_df)} rows)")
    
    if base_df.empty:
        logger.warning("No slow-moving shops found")
        return pd.DataFrame()
    
    # ⚡ OPTIMIZATION 2: Pre-filter source shops with available stock
    # Skip shops with stock <= 30 (they have nothing to transfer)
    base_df = base_df[base_df['SHOP_STOCK'] > 30]
    logger.info(f"⚡ Filtered to shops with stock > 30 ({len(base_df)} rows)")
    
    if base_df.empty:
        logger.warning("No shops with sufficient stock")
        return pd.DataFrame()
    
    # Pre-compute shop combinations for each item
    merged = base_df.merge(base_df, on='ITEM_CODE', suffixes=('_SRC', '_DST'), how='inner')
    merged = merged[merged['SHOP_CODE_SRC'] != merged['SHOP_CODE_DST']]
    
    if merged.empty:
        logger.warning("No valid shop combinations found")
        return pd.DataFrame()
    
    logger.info(f"⚡ Processing {len(merged)} shop-to-shop combinations")

    recs = []
    items_processed = 0
    
    # Track total recommended qty per (item, dest_shop) to enforce 30-unit cap
    destination_totals = {}
    
    # ⚡ OPTIMIZATION 3: Pre-compute today's date once
    today = pd.Timestamp.today().date()

    for (item, src_shop), g in merged.groupby(['ITEM_CODE', 'SHOP_CODE_SRC'], observed=True):
        src_row = g.iloc[0]
        src_stock = int(src_row['SHOP_STOCK_SRC'] or 0)
        src_sales = float(src_row.get('DEST_LAST_30D_SALES_SRC', 0) or 0)
        item_name = src_row.get('ITEM_NAME_SRC', item)

        src_needed_30d = src_sales if src_sales > 0 else 30
        
        # ⚡ OPTIMIZATION 4: Calculate available once and skip early if nothing to transfer
        available = max(src_stock - src_needed_30d, 0)
        if available <= 0:
            continue

        # --- Check if SOURCE shop GRN is recent (within last 30 days) ---
        src_last_grn = pd.NaT
        if 'LAST_GRN_DATE_SRC' in src_row.index:
            src_last_grn = src_row['LAST_GRN_DATE_SRC']
        elif 'LAST_GRN_DATE' in src_row.index:
            src_last_grn = src_row['LAST_GRN_DATE']
        
        src_grn_is_recent = False
        src_grn_age_days = 0
        
        if pd.notna(src_last_grn):
            src_grn_age_days = (today - pd.Timestamp(src_last_grn).date()).days
            if src_grn_age_days < 30:
                src_grn_is_recent = True
        
        # If source GRN is recent, skip recommendations for this source shop
        if use_grn_logic and src_grn_is_recent:
            continue  # ⚡ Skip early - no logging to reduce overhead

        # ⚡ OPTIMIZATION 5: Pre-filter destination shops in this group
        # Only process shops with low sales and no recent GRN
        dest_candidates = g[g['DEST_LAST_30D_SALES_DST'] < 10].copy()
        
        if use_grn_logic:
            # Filter out destinations with recent GRN
            dest_candidates['dest_grn_age'] = dest_candidates.apply(
                lambda row: (today - pd.Timestamp(row.get('LAST_GRN_DATE_DST', pd.NaT)).date()).days 
                if pd.notna(row.get('LAST_GRN_DATE_DST')) else 999,
                axis=1
            )
            dest_candidates = dest_candidates[dest_candidates['dest_grn_age'] >= 30]
        
        if dest_candidates.empty:
            continue
        
        # Priority shop order
        dest_order = list(Config.PRIORITY_SHOPS) + [
            x for x in dest_candidates['SHOP_CODE_DST'].unique() if x not in Config.PRIORITY_SHOPS
        ]

        for dest_shop in dest_order:
            drow = g[g['SHOP_CODE_DST'] == dest_shop]
            dest_stock = int(drow['SHOP_STOCK_DST'].iloc[0] or 0) if not drow.empty else 0
            dest_sales = float(drow['DEST_LAST_30D_SALES_DST'].iloc[0] or 0) if not drow.empty else 0

            # Ensure numeric values and fill NaNs
            dest_stock = 0 if pd.isna(dest_stock) else dest_stock
            dest_sales = 0 if pd.isna(dest_sales) else dest_sales

            # Get destination last GRN date
            dest_last_grn = pd.NaT
            if not drow.empty:
                if 'LAST_GRN_DATE_DST' in drow.columns:
                    dest_last_grn = drow['LAST_GRN_DATE_DST'].iloc[0]
                elif 'LAST_GRN_DATE' in drow.columns:
                    dest_last_grn = drow['LAST_GRN_DATE'].iloc[0]
            
            # Calculate destination GRN age
            dest_grn_age_days = 0
            if pd.notna(dest_last_grn):
                dest_grn_age_days = (today - pd.Timestamp(dest_last_grn).date()).days
            
            # 🚫 NEW RULE 1: Block if source and destination are the same shop
            skip_transfer = False
            skip_reason = ''
            
            if src_shop == dest_shop:
                skip_transfer = True
                skip_reason = ''  # Keep blank as requested
                logger.info(f"⏭️ Blocking {item} {src_shop}→{dest_shop}: Same shop")
            
            # 🚫 NEW RULE 2: Block if source shop is a priority shop
            if not skip_transfer and src_shop in Config.PRIORITY_SHOPS:
                skip_transfer = True
                skip_reason = 'These are priority shop'
                logger.info(f"⏭️ Blocking {item} {src_shop}→{dest_shop}: Source is priority shop")
            
            # 🚫 NEW RULE 3: Block if both sales are zero
            if not skip_transfer and src_sales == 0 and dest_sales == 0:
                # Check if source GRN is N/A
                if pd.isna(src_last_grn):
                    skip_transfer = True
                    skip_reason = 'GRN not available'
                    logger.info(f"⏭️ Blocking {item} {src_shop}→{dest_shop}: Source GRN N/A")
                elif src_grn_age_days > 30 and dest_grn_age_days > 30:
                    skip_transfer = True
                    skip_reason = 'src & dest shop has zero sales'
                    logger.info(f"⏭️ Blocking {item} {src_shop}→{dest_shop}: {skip_reason}")
                elif src_grn_age_days < 30 or dest_grn_age_days < 30:
                    skip_transfer = True
                    skip_reason = 'Latest GRN'
                    logger.info(f"⏭️ Blocking {item} {src_shop}→{dest_shop}: {skip_reason} (src GRN: {src_grn_age_days}d, dest GRN: {dest_grn_age_days}d)")
            
            # Calculate base recommended qty using existing logic
            base_recommended_qty = 0
            
            # === NEW LOGIC: Track actual destination capacity including allocated ===
            dest_key = (item, dest_shop)
            if dest_key not in destination_totals:
                destination_totals[dest_key] = 0
            
            already_allocated = destination_totals[dest_key]
            current_total_stock = dest_stock + already_allocated  # Current stock + what's already being sent
            
            if dest_sales > 0:
                if dest_stock == 0:
                    base_recommended_qty = dest_sales
                else:
                    current_stock_days = dest_stock / dest_sales * 30
                    base_recommended_qty = max(0, dest_sales - dest_stock) if current_stock_days <= 30 else 0
            else:  # dest_sales == 0
                # For zero-sales destinations, cap at 30 units TOTAL (current + allocated)
                if current_total_stock < 30:
                    base_recommended_qty = 30 - current_total_stock
                else:
                    base_recommended_qty = 0
            
            if pd.notna(base_recommended_qty) and base_recommended_qty < 0:
                base_recommended_qty = 0
            
            base_recommended_qty = max(0, base_recommended_qty)
            
            # Maximum allowed to destination = 30 units total (for zero-sales shops)
            # For shops with sales, allow based on sales calculation
            if dest_sales == 0:
                max_allowed_to_dest = 30
                available_capacity = max_allowed_to_dest - current_total_stock
            else:
                # For shops with sales, use sales-based calculation (no hard 30-unit cap)
                available_capacity = base_recommended_qty
            
            # If no capacity left or would cause overstocking, mark it
            will_overstock = False
            if available_capacity <= 0 or (dest_sales == 0 and current_total_stock >= 30):
                will_overstock = True
                logger.info(f"⚠️ {item} → {dest_shop}: Would cause overstocking (current: {dest_stock}, allocated: {already_allocated}, total: {current_total_stock}/30)")
            
            # Cap recommended qty to available capacity and available stock
            # If transfer is blocked by new rule, set qty to 0
            if skip_transfer:
                capped_recommended_qty = 0
            elif will_overstock:
                capped_recommended_qty = 0
            else:
                capped_recommended_qty = min(base_recommended_qty, available_capacity, available)
            
            # === WEIGHTED ALLOCATION BY GRN AGE ===
            # If we need to allocate less than base due to capacity, use GRN age as weight
            # Priority: Older GRN (higher age_days) gets to send more stock
            if capped_recommended_qty > 0 and capped_recommended_qty < base_recommended_qty and src_grn_age_days > 0:
                # Weight factor: normalize GRN age to 0-1 scale (max 365 days = full priority)
                grn_weight = min(src_grn_age_days / 365.0, 1.0)  # 0 = recent, 1.0 = very old
                weighted_qty = int(capped_recommended_qty * grn_weight)
                logger.info(f"📊 {item} {src_shop}→{dest_shop}: GRN age {src_grn_age_days}d, weight={grn_weight:.2f}, qty capped {base_recommended_qty} → {weighted_qty}")
                capped_recommended_qty = weighted_qty
            
            # Update running total for this destination
            recommended_qty = capped_recommended_qty
            destination_totals[dest_key] += recommended_qty
            
            dest_updated_stock = dest_stock + already_allocated + recommended_qty
            final_stock_inhand_days = (dest_updated_stock / dest_sales * 30) if dest_sales > 0 else 0
            final_stock_inhand_days = np.round(final_stock_inhand_days, 1)

            # Set remark: show transfer type, capacity info, or blocking reason
            if skip_transfer:
                remark = f'❌ Not recommended: {skip_reason}'
            elif will_overstock:
                remark = '❌ Transfer will cause overstocking'
            elif recommended_qty == 0:
                remark = ''
            else:
                transfer_type = 'Priority transfer' if dest_shop in Config.PRIORITY_SHOPS else 'Normal transfer'
                remark = f"{transfer_type}"

            recs.append({
                'ITEM_CODE': item,
                'Item Name': item_name,
                'Source Shop': src_shop,
                'Source Stock': src_stock,
                'Source Last 30d Sales': src_sales,
                'Source Last GRN Date': src_last_grn if pd.notna(src_last_grn) else 'N/A',
                'Source GRN Age (days)': src_grn_age_days,
                'Destination Shop': dest_shop,
                'Destination Stock': dest_stock,
                'Destination Last 30d Sales': dest_sales,
                'Recommended_Qty': np.round(recommended_qty, 0),
                'Destination Updated Stock': dest_updated_stock,
                'Destination Final Stock In Hand Days': final_stock_inhand_days,
                'Remark': remark
            })

            if recommended_qty > 0:
                available -= recommended_qty

    result = pd.DataFrame(recs)
    if result.empty:
        logger.warning("No recommendations generated")
        return result

    # --- sort by priority ---
    rank = {s: i for i, s in enumerate(Config.PRIORITY_SHOPS)}
    result['_priority'] = result['Destination Shop'].map(rank).fillna(999)
    result = result.sort_values(['ITEM_CODE', '_priority', 'Destination Shop'], ignore_index=True)
    result = result.drop(columns=['_priority'])

    elapsed = time.time() - start_time
    logger.info(f"⚡ Generated {len(result)} recommendations in fast mode (took {elapsed:.2f}s)")
    return result


@st.cache_data(ttl=300, show_spinner=False)
def cached_generate_recommendations(group: str, subgroup: str, product: str, item_type: str, supplier: str, item_name: str, use_grn_logic: bool, threshold: int = 10) -> pd.DataFrame:
    """
    Cacheable wrapper for ultra-fast recommendations.
    
    **NEW**: Attempts to use optimized materialized view first.
    Falls back to legacy implementation if MV not available.
    
    Performance:
    - Optimized: 3-5s for 121K rows (30-60x faster)
    - Legacy fallback: 90-120s for 121K rows
    """
    import time
    t0 = time.time()
    
    # Try optimized version first (uses mv_recommendation_base)
    try:
        logger.info("🚀 Attempting ULTRA-FAST optimized recommendation engine...")
        result = generate_recommendations_optimized(
            group=group,
            subgroup=subgroup,
            product=product,
            shop='All',
            use_grn_logic=use_grn_logic,
            threshold=threshold
        )
        
        if not result.empty:
            # Apply SIT filters if specified
            if item_type != 'All' or supplier != 'All' or item_name != 'All':
                logger.info("Applying SIT filters to optimized results...")
                sit_df = load_sit_filter_options()
                if not sit_df.empty:
                    # Filter by item codes that match SIT criteria
                    matched_codes = set()
                    
                    for _, row in sit_df.iterrows():
                        type_match = (item_type == 'All' or row['type'] == item_type)
                        supplier_match = (supplier == 'All' or row['vc_supplier_name'] == supplier)
                        name_match = (item_name == 'All' or row['item_name'] == item_name)
                        
                        if type_match and supplier_match and name_match:
                            matched_codes.add(row['vc_item_code'].strip().upper())
                    
                    if matched_codes:
                        result = result[result['ITEM_CODE'].isin(matched_codes)]
            
            total_time = time.time() - t0
            logger.info(f"⚡ OPTIMIZED SUCCESS: {len(result)} recommendations in {total_time:.2f}s")
            return result
            
    except Exception as e:
        logger.warning(f"⚠️ Optimized engine unavailable: {e}")
        logger.info("Falling back to legacy recommendation engine...")
    
    # Legacy fallback - load dataframe and process with nested loops
    logger.info("🐌 Using LEGACY recommendation engine (slower)...")
    
    # Load the full inventory for the selected product/group/subgroup (all shops)
    t1 = time.time()
    full_df = load_inventory_data(group, subgroup, product, 'All')
    load_time = time.time() - t1
    logger.info(f"⏱️ load_inventory_data: {load_time:.3f}s ({len(full_df)} rows)")

    # Apply SIT filters if present
    t2 = time.time()
    sit_df = load_sit_filter_options()
    full_df = apply_itemdetails_filters(full_df, sit_df, item_type, supplier, item_name)
    sit_time = time.time() - t2
    logger.info(f"⏱️ apply_itemdetails_filters: {sit_time:.3f}s ({len(full_df)} rows)")
    
    if full_df.empty:
        logger.warning("⚠️ Filtered dataframe empty after SIT filters")
        return pd.DataFrame()

    # Call the non-cached generator (its result is cached by this wrapper)
    t3 = time.time()
    result = generate_recommendations(full_df, use_grn_logic)
    gen_time = time.time() - t3
    logger.info(f"⏱️ generate_recommendations: {gen_time:.3f}s ({len(result)} recommendations)")
    
    total_time = time.time() - t0
    logger.info(f"⏱️ TOTAL cached_generate_recommendations: {total_time:.3f}s")
    
    return result



# ============================================================
# EXPORT UTILITIES
# ============================================================

def convert_to_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV"""
    return df.to_csv(index=False).encode('utf-8')

def convert_to_excel(slow_df, fast_df, final_df, filters_used):
    """Export the data to Excel with timezone-safe datetimes."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        
        # 🧹 Remove timezone info (Excel doesn’t support tz-aware datetimes)
        for df in [slow_df, fast_df, final_df]:
            for col in df.select_dtypes(include=['datetimetz']).columns:
                df[col] = df[col].dt.tz_localize(None)

        # Write sheets
        slow_df.to_excel(writer, index=False, sheet_name='Slow_Moving_Shops')
        fast_df.to_excel(writer, index=False, sheet_name='Fast_Moving_Shops')
        final_df.to_excel(writer, index=False, sheet_name='Final_Recs')

        # Add filters or metadata if needed
        meta_df = pd.DataFrame(list(filters_used.items()), columns=["Filter", "Value"])
        meta_df.to_excel(writer, index=False, sheet_name='Filters')

    output.seek(0)
    return output


# ============================================================
# UI COMPONENTS
# ============================================================

def render_header():
    """Render header with logo centered - responsive"""
    st.markdown(f"""
        <style>
        /* Main app background */
        .stApp {{
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        }}
        
        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        ::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 10px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #764ba2;
        }}
        
        /* Metric cards enhancement */
        [data-testid="stMetricValue"] {{
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
        }}
        
        /* Selectbox/Dropdown styling - Fix visibility and size */
        .stSelectbox label {{
            font-size: 16px;
            font-weight: 600;
            color: #333;
        }}
        
        .stSelectbox > div > div {{
            background-color: white;
            border: 2px solid #667eea;
            border-radius: 8px;
            font-size: 15px;
        }}
        
        .stSelectbox [data-baseweb="select"] {{
            background-color: white;
        }}
        
        .stSelectbox [data-baseweb="select"] > div {{
            background-color: white;
            border-color: #667eea;
            font-size: 15px;
            color: #333;
        }}
        
        /* Number input styling */
        .stNumberInput label {{
            font-size: 16px;
            font-weight: 600;
            color: #333;
        }}
        
        .stNumberInput input {{
            font-size: 15px;
            border: 2px solid #667eea;
            border-radius: 8px;
        }}
        
        /* Checkbox styling */
        .stCheckbox label {{
            font-size: 15px;
            font-weight: 500;
            color: #333;
        }}
        
        /* Text input styling */
        .stTextInput label {{
            font-size: 16px;
            font-weight: 600;
            color: #333;
        }}
        
        .stTextInput input {{
            font-size: 15px;
            border: 2px solid #667eea;
            border-radius: 8px;
        }}
        
        /* Caption text - make larger */
        .stCaptionContainer {{
            font-size: 14px !important;
        }}
        
        /* Improve main content width */
        .main .block-container {{
            max-width: 1400px;
            padding-left: 2rem;
            padding-right: 2rem;
        }}
        
        /* Button styling */
        .stButton > button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }}
        
        /* Download button */
        .stDownloadButton > button {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        }}
        
        .stDownloadButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
        }}
        
        .melcom-logo {{
            width: 60px;
            height: 60px;
            margin-right: 15px;
            filter: drop-shadow(0 0 10px rgba(255,255,255,0.8));
            animation: pulse 2s ease-in-out infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        
        .title-container {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            border-radius: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-family: 'Segoe UI', sans-serif;
            font-weight: 700;
            font-size: 28px;
            margin-bottom: 30px;
            flex-wrap: wrap;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        @media (max-width: 768px) {{
            .melcom-logo {{
                width: 40px;
                height: 40px;
                margin-right: 8px;
            }}
            .title-container {{
                font-size: 18px;
                padding: 15px;
                margin-bottom: 15px;
            }}
        }}
        </style>
        <div class="title-container">
            <img src="{Config.LOGO_URL}" alt="Logo" class="melcom-logo" />
             MELCOM Inventory Pulse NO_WH
        </div>
    """, unsafe_allow_html=True)

def render_login():
    """Render login page: Employee ID above Password in a centered narrow column.

    The Login button is left-aligned under the password field.
    Press Enter to login functionality enabled.
    """
    
    # Custom styling for login page
    st.markdown("""
        <style>
        .login-header {
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        .login-header h1 {
            color: white;
            font-size: 32px;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .login-header p {
            color: rgba(255,255,255,0.9);
            font-size: 14px;
            margin-top: 10px;
        }
        </style>
        <div class="login-header">
            <h1>🔐 Melcom Inventory Pulse</h1>
            <p>Secure Access Portal</p>
        </div>
    """, unsafe_allow_html=True)

    # Create a centered narrow column to reduce input width
    left_col, center_col, right_col = st.columns([1, 0.6, 1])
    with center_col:
        # Use form to enable Enter key submission
        with st.form(key="login_form", clear_on_submit=False):
            employee_id = st.text_input("👤 Employee ID")
            password = st.text_input("🔑 Password", type="password")

            # Left-aligned login button beneath the inputs
            btn_col1, btn_col2 = st.columns([1, 3])
            with btn_col1:
                login_clicked = st.form_submit_button("Login", width='stretch')
            with btn_col2:
                st.write("")
            
            # Handle login when button clicked or Enter pressed
            if login_clicked:
                if not employee_id or not password:
                    st.warning("⚠️ Enter both Employee ID and Password")
                else:
                    with st.spinner("Authenticating..."):
                        user = authenticate_user(employee_id, password)

                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user = user
                        st.success(f"✅ Welcome, {user['full_name']}")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")

def render_base_filters(filter_df: pd.DataFrame) -> Tuple[str, str, str, str]:
    """Render base filters in the sidebar (Groups, Sub Group, Product Code, Shop Code)"""
    st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 12px; border-radius: 8px; text-align: center; color: white; margin-bottom: 15px;">
            <div style="font-size: 16px; font-weight: bold;">🔍 Inventory Filters</div>
        </div>
    """, unsafe_allow_html=True)
    
    def add_all(options):
        return ['All'] + sorted(list(set([str(x).strip() for x in options if x and str(x).strip()])))
    
    # Original filters
    group = st.sidebar.selectbox("Groups", add_all(filter_df['GROUPS'].dropna().unique()))
    
    subgroup_opts = filter_df['SUB_GROUP'].dropna().unique() if group == 'All' else \
                    filter_df.loc[filter_df['GROUPS'] == group, 'SUB_GROUP'].dropna().unique()
    subgroup = st.sidebar.selectbox("Sub Group", add_all(subgroup_opts))
    
    if subgroup == 'All' and group == 'All':
        product_opts = filter_df['ITEM_CODE'].dropna().unique()
    else:
        mask = ((filter_df['GROUPS'] == group) | (group == 'All')) & \
               ((filter_df['SUB_GROUP'] == subgroup) | (subgroup == 'All'))
        product_opts = filter_df.loc[mask, 'ITEM_CODE'].dropna().unique()
    product = st.sidebar.selectbox("Product Code", add_all(product_opts))
    
    if group == 'All' and subgroup == 'All' and product == 'All':
        shop_opts = filter_df['SHOP_CODE'].dropna().unique()
    else:
        mask = ((filter_df['GROUPS'] == group) | (group == 'All')) & \
               ((filter_df['SUB_GROUP'] == subgroup) | (subgroup == 'All')) & \
               ((filter_df['ITEM_CODE'] == product) | (product == 'All'))
        shop_opts = filter_df.loc[mask, 'SHOP_CODE'].dropna().unique()
    shop = st.sidebar.selectbox("Shop Code", add_all(shop_opts))
    
    return group, subgroup, product, shop

def render_sit_filters(sit_filter_df: pd.DataFrame) -> Tuple[str, str, str]:
    """Render SIT item-details filters on the main page - responsive"""
    st.markdown("""
        <h3 style="color: #667eea; margin: 25px 0 15px 0; font-size: 22px;">📦 Item Details Filters</h3>
    """, unsafe_allow_html=True)
    
    def add_all(options):
        return ['All'] + sorted(list(set([str(x).strip() for x in options if x and str(x).strip()])))
    
    # Use responsive columns - will stack on mobile
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        item_type_opts = sit_filter_df['type'].dropna().unique() if not sit_filter_df.empty else []
        item_type = st.selectbox("Type", add_all(item_type_opts), key="sit_type")
    
    with col2:
        supplier_opts = sit_filter_df['vc_supplier_name'].dropna().unique() if not sit_filter_df.empty else []
        supplier = st.selectbox("Supplier Name", add_all(supplier_opts), key="sit_supplier")
    
    with col3:
        item_name_opts = sit_filter_df['item_name'].dropna().unique() if not sit_filter_df.empty else []
        item_name = st.selectbox("Item Name", add_all(item_name_opts), key="sit_item_name")
    
    return item_type, supplier, item_name

def render_filter_summary(group: str, subgroup: str, product: str, shop: str, item_type: str, supplier: str, item_name: str):
    """Render a heading showing currently applied filters"""
    active_filters = []
    if group and group != 'All':
        active_filters.append(f"Group: **{group}**")
    if subgroup and subgroup != 'All':
        active_filters.append(f"Sub-Group: **{subgroup}**")
    if product and product != 'All':
        active_filters.append(f"Product: **{product}**")
    if shop and shop != 'All':
        active_filters.append(f"Shop: **{shop}**")
    if item_type and item_type != 'All':
        active_filters.append(f"Type: **{item_type}**")
    if supplier and supplier != 'All':
        active_filters.append(f"Supplier: **{supplier}**")
    if item_name and item_name != 'All':
        active_filters.append(f"Item: **{item_name}**")
    
    if active_filters:
        filter_text = " | ".join(active_filters)
        st.markdown(f"### 🎯 Active Filters: {filter_text}")
    else:
        st.markdown("### 🎯 Active Filters: None (Showing All Data)")

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main application"""
    
    # Page config
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout="wide"
    )
    
    # Add responsive CSS
    st.markdown("""
        <style>
        /* Mobile-first responsive design */
        @media (max-width: 768px) {
            .stButton > button {
                width: 100%;
                margin: 5px 0;
            }
            .row-widget.stSelectbox {
                width: 100%;
            }
            div[data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }
            .stDataFrame {
                overflow-x: auto;
            }
            h1, h2, h3 {
                font-size: 1.2rem !important;
            }
        }
        
        @media (min-width: 769px) and (max-width: 1024px) {
            /* Tablet adjustments */
            div[data-testid="column"] {
                padding: 0 0.5rem;
            }
        }
        
        /* Make tables horizontally scrollable */
        .dataframe-container {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
        
        /* Improve button visibility on mobile */
        .stDownloadButton > button {
            width: 100%;
        }
        
        /* Format dataframe columns properly */
        .stDataFrame div[data-testid="stDataFrameResizable"] table tbody tr td,
        .stDataFrame div[data-testid="stDataFrameResizable"] table thead tr th {
            text-align: center !important;
        }
        
        /* Left align text columns (Item Name, Shop codes, Remark) */
        .stDataFrame div[data-testid="stDataFrameResizable"] table tbody tr td:nth-child(1),
        .stDataFrame div[data-testid="stDataFrameResizable"] table tbody tr td:nth-child(2) {
            text-align: left !important;
            padding-left: 10px !important;
        }
        
        /* Right align numeric columns */
        .stDataFrame div[data-testid="stDataFrameResizable"] table tbody tr td:has(div:matches('[0-9]+')) {
            text-align: right !important;
            padding-right: 10px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user = None
    
    # Login flow
    if not st.session_state.logged_in:
        render_login()
        st.stop()
    
    # Access control
    user = st.session_state.user
    if not check_table_access(user, "nowhstock_tbl_new"):
        st.error("🚫 Access denied")
        st.stop()
    
    # Header
    render_header()
    
    # Sidebar
    st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center; color: white;">
            <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">👋 Welcome</div>
            <div style="font-size: 14px;">{}</div>
            <div style="font-size: 12px; opacity: 0.9; margin-top: 3px;">ID: {}</div>
        </div>
    """.format(user['full_name'], user['employee_id']), unsafe_allow_html=True)
    
    if st.sidebar.button("🚪 Logout", key="logout_btn", width='stretch'):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.rerun()
    st.sidebar.divider()
    
    # Load filters
    filter_df = load_filter_options()
    sit_filter_df = load_sit_filter_options()
    
    if filter_df.empty:
        st.error("❌ Cannot load filters")
        st.stop()
    
    # Render base filters in sidebar
    group, subgroup, product, shop = render_base_filters(filter_df)
    
    # Additional controls in sidebar
    threshold = st.sidebar.number_input("Sales threshold (Fast vs Slow)", min_value=0, value=Config.DEFAULT_THRESHOLD)
    use_grn_logic = st.sidebar.checkbox("📦 Use GRN Date Logic", value=True)
    show_grn_info = st.sidebar.checkbox("🔍 Show GRN Info", value=True)
    
    # Calculate dates for display
    end_date = datetime.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)
    st.sidebar.caption(f"✨ Sales period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Render SIT filters on main page
    item_type, supplier, item_name = render_sit_filters(sit_filter_df)
    
    # Show active filters heading
    render_filter_summary(group, subgroup, product, shop, item_type, supplier, item_name)
    
    st.divider()
    
    # Load data
    inventory_df = load_inventory_data(group, subgroup, product, shop)

    # Apply Item Details (SIT) filters if selected
    inventory_df = apply_itemdetails_filters(inventory_df, sit_filter_df, item_type, supplier, item_name)

    if inventory_df.empty:
        st.warning("No records found")
        st.stop()
    
    # ✅ Show diagnostic information about sales distribution
    st.sidebar.divider()
    st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 12px; border-radius: 8px; text-align: center; color: white; margin-bottom: 15px;">
            <div style="font-size: 16px; font-weight: bold;">📊 Sales Statistics</div>
        </div>
    """, unsafe_allow_html=True)
    
    total_items = len(inventory_df)
    items_with_sales = (inventory_df['ITEM_SALES_30_DAYS'] > 0).sum()
    max_sales = inventory_df['ITEM_SALES_30_DAYS'].max()
    avg_sales = inventory_df['ITEM_SALES_30_DAYS'].mean()
    total_sales_qty = inventory_df['ITEM_SALES_30_DAYS'].sum()
    
    st.sidebar.metric("📦 Total Records", total_items)
    st.sidebar.metric("✅ Items with Sales", f"{items_with_sales} ({items_with_sales/total_items*100:.1f}%)")
    st.sidebar.metric("📈 Total Sales Qty", f"{int(total_sales_qty):,}")
    st.sidebar.metric("🎯 Max 30d Sales", int(max_sales))
    st.sidebar.metric("📊 Avg 30d Sales", f"{avg_sales:.1f}")
    
    # Classify
    inventory_df['Sales_Status'] = np.where(
        inventory_df['ITEM_SALES_30_DAYS'] >= threshold,
        'Fast',
        'Slow'
    )
    slow_shops = inventory_df[inventory_df['Sales_Status'] == 'Slow']
    fast_shops = inventory_df[inventory_df['Sales_Status'] == 'Fast']
    
    # Calculate trend metrics with unique shop and item counts
    slow_unique_shops = slow_shops['SHOP_CODE'].nunique() if not slow_shops.empty else 0
    fast_unique_shops = fast_shops['SHOP_CODE'].nunique() if not fast_shops.empty else 0
    slow_unique_items = slow_shops['ITEM_CODE'].nunique() if not slow_shops.empty else 0
    fast_unique_items = fast_shops['ITEM_CODE'].nunique() if not fast_shops.empty else 0
    slow_total_stock = int(slow_shops['SHOP_STOCK'].sum()) if not slow_shops.empty else 0
    fast_total_stock = int(fast_shops['SHOP_STOCK'].sum()) if not fast_shops.empty else 0
    total_unique_shops = inventory_df['SHOP_CODE'].nunique()
    slow_pct = (slow_unique_shops / total_unique_shops * 100) if total_unique_shops > 0 else 0
    fast_pct = (fast_unique_shops / total_unique_shops * 100) if total_unique_shops > 0 else 0
    
    # Display Current Inventory Trends at the top - LEFT ALIGNED with smallest width
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: -10px 15px; border-radius: 12px; border-left: 5px solid #667eea; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.07); max-width: 600px;">
            <div style="text-align: center; font-size: 18px; font-weight: bold; color: #333; margin-bottom: 12px;">📊 Current Inventory Trends</div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                <div style="background: white; padding: 12px 10px; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-top: 3px solid #f5576c;">
                    <div style="font-size: 12px; color: #444; margin-bottom: 5px; font-weight: 600;">Slow Moving Shops</div>
                    <div style="font-size: 28px; font-weight: bold; color: #f5576c; margin-bottom: 3px;">{slow_unique_shops}</div>
                    <div style="font-size: 13px; color: #555; margin-bottom: 6px;">{slow_pct:.1f}% of all shops</div>
                    <div style="font-size: 12px; color: #666; padding-top: 5px; border-top: 1px solid #f0f0f0;">
                        <div><strong>{slow_unique_items}</strong> unique items</div>
                        <div style="margin-top: 2px;"><strong>{slow_total_stock:,}</strong> total stock units</div>
                    </div>
                </div>
                <div style="background: white; padding: 12px 10px; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-top: 3px solid #10b981;">
                    <div style="font-size: 12px; color: #444; margin-bottom: 5px; font-weight: 600;">Fast Moving Shops</div>
                    <div style="font-size: 28px; font-weight: bold; color: #10b981; margin-bottom: 3px;">{fast_unique_shops}</div>
                    <div style="font-size: 13px; color: #555; margin-bottom: 6px;">{fast_pct:.1f}% of all shops</div>
                    <div style="font-size: 12px; color: #666; padding-top: 5px; border-top: 1px solid #f0f0f0;">
                        <div><strong>{fast_unique_items}</strong> unique items</div>
                        <div style="margin-top: 2px;"><strong>{fast_total_stock:,}</strong> total stock units</div>
                    </div>
                </div>
                <div style="background: white; padding: 12px 10px; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-top: 3px solid #667eea;">
                    <div style="font-size: 12px; color: #444; margin-bottom: 5px; font-weight: 600;">Transfer Potential</div>
                    <div style="font-size: 28px; font-weight: bold; color: #667eea; margin-bottom: 3px;">{len(slow_shops)}</div>
                    <div style="font-size: 13px; color: #555; margin-bottom: 6px;">slow moving records</div>
                    <div style="font-size: 12px; color: #666; padding-top: 5px; border-top: 1px solid #f0f0f0;">
                        <div>Ready for reallocation</div>
                        <div style="margin-top: 2px;">to fast-moving shops</div>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Multi-page tabs for Slow Moving and Fast Moving with custom font size and active tab highlighting
    st.markdown("""
        <style>
        /* Increase tab font size */
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 18px;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        /* Highlight active/selected tab */
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px 8px 0 0;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {
            color: white;
        }
        /* Inactive tab styling */
        .stTabs [data-baseweb="tab-list"] button[aria-selected="false"] {
            background-color: #f0f0f0;
            border-radius: 8px 8px 0 0;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="false"]:hover {
            background-color: #e0e0e0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs([f"📉 Slow Moving Shops (Sales < {threshold})", f"📈 Fast Moving Shops (Sales >= {threshold})"])
    
    with tab1:
        st.caption(f"{len(slow_shops)} records | Total Sales: {int(slow_shops['ITEM_SALES_30_DAYS'].sum()):,} | Total Stock: {slow_total_stock:,}")
        if not slow_shops.empty:
            st.dataframe(slow_shops, width='stretch', height=400)
        else:
            st.info("✅ No slow-moving items found")
    
    with tab2:
        st.caption(f"{len(fast_shops)} records | Total Sales: {int(fast_shops['ITEM_SALES_30_DAYS'].sum()):,} | Total Stock: {fast_total_stock:,}")
        if not fast_shops.empty:
            st.dataframe(fast_shops, width='stretch', height=400)
        else:
            st.info(f"ℹ️ No items with sales >= {threshold}. Try lowering the threshold.")
    
    st.divider()

    # Generate recommendations button - centered and compact
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1, 1.5])
    with btn_col2:
        gen_clicked = st.button("🎯 Generate Smart Recommendations", key="gen_rec_btn", width='stretch', type="primary")
    st.markdown("""
        <div style="text-align: center; margin-top: 8px; margin-bottom: 15px; font-size: 13px; color: #666;">
            Analyze inventory and suggest optimal transfers
        </div>
    """, unsafe_allow_html=True)

    if gen_clicked:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("📊 Loading data...")
            progress_bar.progress(25)

            if product != "All":
                full_df = load_inventory_data(group, subgroup, product, "All")
                # Apply SIT filters to the full set used for recommendations
                full_df = apply_itemdetails_filters(full_df, sit_filter_df, item_type, supplier, item_name)
            else:
                full_df = inventory_df

            status_text.text("⚡ Generating recommendations...")
            progress_bar.progress(60)

            # Use cached wrapper which tries optimized version first, falls back to legacy
            recommendations = cached_generate_recommendations(
                group=group,
                subgroup=subgroup,
                product=product,
                item_type=item_type,
                supplier=supplier,
                item_name=item_name,
                use_grn_logic=use_grn_logic,
                threshold=threshold
            )

            if shop != "All":
                recommendations = recommendations[
                    (recommendations['Source Shop'] == shop) | 
                    (recommendations['Destination Shop'] == shop)
                ]

            progress_bar.progress(100)
            status_text.text("✅ Complete!")

        except Exception as e:
            st.error(f"Error: {e}")
            logger.error(f"Recommendation generation error: {e}")
            return

        if not recommendations.empty:
            status_text.text("📋 Formatting output...")

            preferred_cols = [
                'ITEM_CODE', 'Item Name',
                'Source Shop', 'Source Stock', 'Source Last 30d Sales', 'Source Last GRN Date', 'Source GRN Age (days)',
                'Destination Shop', 'Destination Stock', 'Destination Last 30d Sales',
                'Destination Updated Stock', 'Destination Final Stock In Hand Days',
                'Recommended_Qty', 'Remark'
            ]
            final_recs = recommendations[[c for c in preferred_cols if c in recommendations.columns]].copy()

            if not show_grn_info:
                final_recs = final_recs.drop(columns=['Destination GRN Sales', 'GRN Date', 'Source Last GRN Date', 'Source GRN Age (days)'], errors='ignore')

            st.subheader(f"🚚 {len(final_recs)} Transfer Recommendations")
            grn_status = "ON" if use_grn_logic else "OFF"
            st.caption(f"GRN: {grn_status} | Sales: {start_date.strftime('%m-%d')} to {end_date.strftime('%m-%d')}")

            st.dataframe(final_recs, width='stretch', height=400)

            # Download buttons - stack on mobile
            col1, col2 = st.columns([1, 1])
            with col1:
                st.download_button(
                    "📥 Download CSV",
                    convert_to_csv(final_recs),
                    "recommendations.csv",
                    "text/csv",
                    key="rec_csv",
                    width='stretch'
                )
            with col2:
                st.info(f"✅ {len(final_recs)} recommendations ready")
        else:
            st.info("ℹ️ No recommendations for current filters")

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        filters_used = {
            "Groups": group,
            "Sub Group": subgroup,
            "Product": product,
            "Shop": shop,
            "Threshold": threshold,
            "GRN Logic": use_grn_logic,
            "Sales Period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        }

        excel_data = convert_to_excel(slow_shops, fast_shops, final_recs, filters_used)
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        st.download_button(
            "📊 Download Excel Report (All Sheets)",
            excel_data,
            f"Inventory_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="rec_excel",
            width='stretch'
        )

if __name__ == "__main__":
    main()
