"""
Barcode-to-Item Code Matcher Application
Streamlit app for uploading barcodes and getting matched item codes
"""
import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from contextlib import contextmanager
import io
from datetime import datetime
import json
import os

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    DB_CONFIG = {
        'host': 'localhost',
        'port': 3307,
        'database': 'salesdata',
        'user': 'postgres',
        'password': 'hello'
    }
    PAGE_TITLE = "Barcode Matcher"
    PAGE_ICON = "🔍"
    LOGO_URL = "https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg"
    MAX_UPLOAD_SIZE_MB = 10

# ============================================================
# DATABASE CONNECTION
# ============================================================

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = psycopg2.connect(**Config.DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()

# ============================================================
# CORE FUNCTIONS
# ============================================================

def load_barcode_master():
    """Load active barcode mappings into memory (for quick lookup)"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT vc_item_barcode, vc_item_code
            FROM barcode_item_master
            WHERE is_active = TRUE
        """)
        rows = cursor.fetchall()
        cursor.close()
    
    # Convert to dict for O(1) lookup
    barcode_dict = {row['vc_item_barcode']: row['vc_item_code'] for row in rows}
    
    return barcode_dict

@st.cache_data(ttl=3600)
def get_barcode_master_cached():
    """Cached version of barcode master (1 hour TTL)"""
    return load_barcode_master()

def process_uploaded_file(uploaded_file, barcode_column_name='barcode'):
    """
    Process uploaded Excel/CSV file with SINGLE barcode column
    Returns: DataFrame with barcode, item_code, match_status
    """
    try:
        # Read file based on extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        
        # Validate barcode column exists
        if barcode_column_name not in df.columns:
            st.error(f"Column '{barcode_column_name}' not found in file. Available columns: {', '.join(df.columns)}")
            return None
        
        # Store original barcode and clean for matching
        df['barcode_original'] = df[barcode_column_name]
        
        # Clean barcodes: convert to string, strip spaces, handle numeric values
        df['barcode_clean'] = df[barcode_column_name].apply(lambda x: 
            str(x).strip() if pd.notna(x) else ''
        )
        
        # Remove .0 suffix from numeric barcodes (e.g., "123.0" -> "123")
        df['barcode_clean'] = df['barcode_clean'].str.replace(r'\.0$', '', regex=True)
        
        # Load barcode master
        barcode_dict = get_barcode_master_cached()
        
        # Match barcodes with fallback logic for leading zeros
        def match_barcode(barcode_clean):
            # Try exact match first
            item_code = barcode_dict.get(barcode_clean, '')
            if item_code:
                return item_code
            
            # If no match and barcode is numeric, try with leading zero (for UPC-12 -> EAN-13)
            if barcode_clean.isdigit() and len(barcode_clean) == 11:
                # Try adding single leading zero for 11-digit codes (UPC without leading zero)
                barcode_with_zero = '0' + barcode_clean
                item_code = barcode_dict.get(barcode_with_zero, '')
                if item_code:
                    return item_code
            
            return ''
        
        df['item_code'] = df['barcode_clean'].apply(match_barcode)
        df['match_status'] = df['item_code'].apply(lambda x: 'Matched' if x else 'Not Found')
        
        # Final columns: barcode (original), item_code, match_status
        df = df[['barcode_original', 'item_code', 'match_status']].copy()
        df.rename(columns={'barcode_original': 'barcode'}, inplace=True)
        
        return df
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def save_upload_to_database(df, filename, uploaded_by='system', update_master=False):
    """Save upload history and details to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Calculate stats (convert numpy types to Python native types)
            total_rows = int(len(df))
            matched_rows = int((df['match_status'] == 'Matched').sum())
            unmatched_rows = int(total_rows - matched_rows)
            new_mappings_added = 0
            
            # If update_master is True, add new barcode-item mappings to master table
            if update_master:
                # Get existing barcodes from master
                cursor.execute("SELECT vc_item_barcode FROM barcode_item_master WHERE is_active = TRUE")
                existing_barcodes = {row[0] for row in cursor.fetchall()}
                
                # Find matched records with barcodes not in master
                new_mappings = df[(df['match_status'] == 'Matched') & 
                                  (~df['barcode'].isin(existing_barcodes))].copy()
                
                if not new_mappings.empty:
                    # Insert new mappings (convert to string to avoid numpy type issues)
                    for _, row in new_mappings.iterrows():
                        cursor.execute("""
                            INSERT INTO barcode_item_master (vc_item_barcode, vc_item_code)
                            VALUES (%s, %s)
                            ON CONFLICT (vc_item_barcode) DO NOTHING
                        """, (str(row['barcode']), str(row['item_code'])))
                    
                    new_mappings_added = int(len(new_mappings))
            
            # Insert upload history
            cursor.execute("""
                INSERT INTO barcode_upload_history 
                (filename, uploaded_by, total_rows, matched_rows, unmatched_rows, status)
                VALUES (%s, %s, %s, %s, %s, 'completed')
                RETURNING upload_id
            """, (filename, uploaded_by, total_rows, matched_rows, unmatched_rows))
            
            upload_id = cursor.fetchone()[0]
            
            # Prepare detail records (convert all values to native Python types)
            detail_records = []
            for idx, row in df.iterrows():
                detail_records.append((
                    int(upload_id),
                    int(idx + 1),  # row_number (1-based)
                    str(row['barcode']) if pd.notna(row['barcode']) else '',
                    str(row['item_code']) if pd.notna(row['item_code']) and row['item_code'] else None,
                    str(row['match_status'])
                ))
            
            # Bulk insert details
            execute_values(
                cursor,
                """
                INSERT INTO barcode_upload_details 
                (upload_id, row_number, barcode, item_code, match_status)
                VALUES %s
                """,
                detail_records
            )
            
            # Track unmatched barcodes
            unmatched_df = df[df['match_status'] == 'Not Found']
            if not unmatched_df.empty:
                unmatched_barcodes = unmatched_df['barcode'].unique()
                for barcode in unmatched_barcodes:
                    cursor.execute("SELECT track_unmatched_barcode(%s)", (str(barcode),))
            
            conn.commit()
            cursor.close()
            
            return upload_id, new_mappings_added
    
    except Exception as e:
        st.error(f"Error saving to database: {str(e)}")
        return None, 0

def get_upload_history(limit=10):
    """Get recent upload history"""
    with get_db_connection() as conn:
        df = pd.read_sql("""
            SELECT * FROM v_recent_uploads
            ORDER BY uploaded_at DESC
            LIMIT %s
        """, conn, params=(limit,))
    return df

def get_unmatched_barcodes():
    """Get pending unmatched barcodes"""
    with get_db_connection() as conn:
        df = pd.read_sql("""
            SELECT * FROM v_unmatched_barcodes_pending
            ORDER BY occurrence_count DESC
            LIMIT 100
        """, conn)
    return df

def add_barcode_mapping(barcode, item_code, item_name=None):
    """Add new barcode mapping"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT add_barcode_mapping(%s, %s, %s)
            """, (barcode, item_code, item_name))
            conn.commit()
            cursor.close()
        return True
    except Exception as e:
        st.error(f"Error adding mapping: {str(e)}")
        return False

# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    # Page config
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout="wide"
    )
    
    # Header
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(Config.LOGO_URL, width=80)
    with col2:
        st.title("🔍 Barcode-to-Item Code Matcher")
        st.caption("Upload barcodes and instantly get matched item codes")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        **How to use:**
        1. Upload CSV/Excel with `barcode` column
        2. Get matched item codes instantly
        3. Download results
        
        **File Requirements:**
        - Single column: `barcode`
        - Formats: CSV, XLSX, XLS
        - Max size: 10MB
        
        **⚠️ Important:**
        - Format column as TEXT in Excel (preserves zeros)
        - Save as XLSX (best) or CSV UTF-8
        - Trailing zeros are REAL (EAN-13/UPC codes)
        
        **Output:**
        - Matched: Item code filled
        - Not Found: Blank item code
        """)
        
        st.divider()
        
        # Statistics
        barcode_dict = get_barcode_master_cached()
        st.metric("Total Barcodes in Database", f"{len(barcode_dict):,}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Upload & Match", "🔍 Search Barcode", "📊 History", "🔄 Refresh Master Data", "ℹ️ Format Help"])
    
    # ==================== TAB 1: UPLOAD ====================
    with tab1:
        st.header("Upload Barcode File")
        
        # Column name input
        barcode_col = st.text_input(
            "Barcode Column Name",
            value="barcode",
            help="Enter the exact column name containing barcodes"
        )
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help=f"Max file size: {Config.MAX_UPLOAD_SIZE_MB}MB"
        )
        
        if uploaded_file is not None:
            st.info(f"📁 File: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
            
            # Process file
            with st.spinner("Processing file..."):
                df_result = process_uploaded_file(uploaded_file, barcode_col)
            
            if df_result is not None:
                # Show statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", len(df_result))
                with col2:
                    matched = (df_result['match_status'] == 'Matched').sum()
                    st.metric("Matched", matched, delta=f"{matched/len(df_result)*100:.1f}%")
                with col3:
                    not_found = (df_result['match_status'] == 'Not Found').sum()
                    st.metric("Not Found", not_found, delta=f"{not_found/len(df_result)*100:.1f}%")
                with col4:
                    match_rate = matched / len(df_result) * 100
                    st.metric("Match Rate", f"{match_rate:.1f}%")
                
                st.divider()
                
                # Display results
                st.subheader("Results Preview")
                
                # Filter options
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    filter_status = st.selectbox("Filter by Status", ["All", "Matched", "Not Found"])
                with filter_col2:
                    show_rows = st.slider("Rows to display", 10, 100, 20)
                
                # Apply filter
                if filter_status != "All":
                    df_display = df_result[df_result['match_status'] == filter_status].head(show_rows)
                else:
                    df_display = df_result.head(show_rows)
                
                # Color-code results
                def highlight_status(row):
                    if row['match_status'] == 'Matched':
                        return ['background-color: #d4edda'] * len(row)
                    else:
                        return ['background-color: #f8d7da'] * len(row)
                
                st.dataframe(
                    df_display.style.apply(highlight_status, axis=1),
                    use_container_width=True,
                    height=400
                )
                
                st.caption(f"Showing {len(df_display)} of {len(df_result)} rows")
                
                # Download options
                st.divider()
                st.subheader("Download Results")
                
                col1, col2, col3 = st.columns(3)
                
                # CSV download
                with col1:
                    csv = df_result.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"barcode_matched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Excel download
                with col2:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_result.to_excel(writer, index=False, sheet_name='Results')
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"barcode_matched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Save to database with option to update master
                st.divider()
                st.subheader("💾 Save to Database")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    update_master = st.checkbox(
                        "➕ Also add NEW barcode mappings to Master Table",
                        value=False,
                        help="If checked, matched barcodes not in master will be added to barcode_item_master table"
                    )
                    
                    if update_master:
                        st.info("""
                        ℹ️ **What will be added:**
                        - Only MATCHED barcodes (with item codes)
                        - Only NEW barcodes (not already in master)
                        - Skips if barcode already exists (no duplicates)
                        """)
                
                with col2:
                    if st.button("💾 Save Upload", type="primary", use_container_width=True):
                        upload_id, new_mappings = save_upload_to_database(
                            df_result, 
                            uploaded_file.name,
                            update_master=update_master
                        )
                        if upload_id:
                            st.success(f"✅ Saved! Upload ID: {upload_id}")
                            if update_master and new_mappings > 0:
                                st.success(f"➕ Added {new_mappings} new barcode mappings to master table!")
                                st.balloons()
                            elif update_master and new_mappings == 0:
                                st.info("ℹ️ No new mappings to add (all barcodes already in master)")
                            # Clear cache to refresh history and master data
                            st.cache_data.clear()
    
    # ==================== TAB 2: SEARCH BARCODE ====================
    with tab2:
        st.header("🔍 Search Barcode Database")
        st.caption("Search for specific barcodes in the master database")
        
        # Search input
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "Enter Barcode or Item Code",
                placeholder="e.g., 033844004019 or D446",
                help="Search by exact barcode or item code, or use wildcards (%)"
            )
        with col2:
            search_type = st.selectbox("Search Type", ["Exact Match", "Contains", "Starts With", "Ends With"])
        
        if search_query:
            # Build query based on search type
            if search_type == "Exact Match":
                sql_condition = f"WHERE vc_item_barcode = '{search_query}' OR vc_item_code = '{search_query}'"
            elif search_type == "Contains":
                sql_condition = f"WHERE vc_item_barcode LIKE '%{search_query}%' OR vc_item_code LIKE '%{search_query}%'"
            elif search_type == "Starts With":
                sql_condition = f"WHERE vc_item_barcode LIKE '{search_query}%' OR vc_item_code LIKE '{search_query}%'"
            else:  # Ends With
                sql_condition = f"WHERE vc_item_barcode LIKE '%{search_query}' OR vc_item_code LIKE '%{search_query}'"
            
            try:
                with get_db_connection() as conn:
                    search_df = pd.read_sql(f"""
                        SELECT 
                            vc_item_barcode,
                            vc_item_code,
                            created_at,
                            updated_at
                        FROM barcode_item_master
                        {sql_condition}
                        AND is_active = TRUE
                        ORDER BY vc_item_barcode
                        LIMIT 1000
                    """, conn)
                
                if not search_df.empty:
                    st.success(f"✅ Found {len(search_df)} result(s)")
                    
                    # Display results
                    st.dataframe(
                        search_df,
                        use_container_width=True,
                        column_config={
                            "vc_item_barcode": st.column_config.TextColumn("Barcode", width="medium"),
                            "vc_item_code": st.column_config.TextColumn("Item Code", width="medium"),
                            "created_at": st.column_config.DatetimeColumn("Created", format="DD/MM/YYYY HH:mm"),
                            "updated_at": st.column_config.DatetimeColumn("Updated", format="DD/MM/YYYY HH:mm")
                        }
                    )
                    
                    # Download search results
                    csv = search_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Search Results",
                        data=csv,
                        file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning(f"❌ No results found for: {search_query}")
                    st.info("💡 Try using 'Contains' search type or check spelling")
            
            except Exception as e:
                st.error(f"Search error: {str(e)}")
        
        st.divider()
        
        # Advanced search
        with st.expander("🔎 Advanced Search Options"):
            st.subheader("Bulk Search")
            st.caption("Search multiple barcodes at once")
            
            bulk_search = st.text_area(
                "Enter barcodes (one per line)",
                placeholder="033844004019\n4607048108727\nG524",
                height=150
            )
            
            if st.button("🔍 Search All", type="primary") and bulk_search:
                barcodes_list = [b.strip() for b in bulk_search.split('\n') if b.strip()]
                
                if barcodes_list:
                    # Create SQL IN clause
                    barcodes_sql = "', '".join(barcodes_list)
                    
                    try:
                        with get_db_connection() as conn:
                            bulk_df = pd.read_sql(f"""
                                SELECT 
                                    vc_item_barcode,
                                    vc_item_code,
                                    CASE 
                                        WHEN vc_item_barcode IN ('{barcodes_sql}') THEN 'Found'
                                        ELSE 'Not Found'
                                    END as status
                                FROM barcode_item_master
                                WHERE vc_item_barcode IN ('{barcodes_sql}')
                                AND is_active = TRUE
                            """, conn)
                        
                        # Find missing barcodes
                        found_barcodes = set(bulk_df['vc_item_barcode'].tolist())
                        missing_barcodes = set(barcodes_list) - found_barcodes
                        
                        # Add missing to dataframe
                        if missing_barcodes:
                            missing_df = pd.DataFrame({
                                'vc_item_barcode': list(missing_barcodes),
                                'vc_item_code': [''] * len(missing_barcodes),
                                'status': ['Not Found'] * len(missing_barcodes)
                            })
                            bulk_df = pd.concat([bulk_df, missing_df], ignore_index=True)
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Searched", len(barcodes_list))
                        with col2:
                            st.metric("Found", len(found_barcodes))
                        with col3:
                            st.metric("Not Found", len(missing_barcodes))
                        
                        # Display results
                        st.dataframe(bulk_df, use_container_width=True)
                        
                        # Download
                        csv = bulk_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Bulk Search Results",
                            data=csv,
                            file_name=f"bulk_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    except Exception as e:
                        st.error(f"Bulk search error: {str(e)}")
        
        # Quick stats
        st.divider()
        st.subheader("📊 Database Statistics")
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Total count
                cursor.execute("SELECT COUNT(*) FROM barcode_item_master WHERE is_active = TRUE")
                total = cursor.fetchone()[0]
                
                # Numeric vs alphanumeric
                cursor.execute("""
                    SELECT 
                        COUNT(*) FILTER (WHERE vc_item_barcode ~ '^[0-9]+$') as numeric_count,
                        COUNT(*) FILTER (WHERE vc_item_barcode ~ '[A-Za-z]') as alphanumeric_count
                    FROM barcode_item_master 
                    WHERE is_active = TRUE
                """)
                stats = cursor.fetchone()
                
                cursor.close()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Barcodes", f"{total:,}")
                with col2:
                    st.metric("Numeric Only", f"{stats[0]:,}")
                with col3:
                    st.metric("Alphanumeric", f"{stats[1]:,}")
        
        except Exception as e:
            st.error(f"Stats error: {str(e)}")
    
    # ==================== TAB 3: HISTORY ====================
    with tab3:
        st.header("Upload History")
        
        df_history = get_upload_history(20)
        
        if not df_history.empty:
            st.dataframe(
                df_history,
                use_container_width=True,
                column_config={
                    "uploaded_at": st.column_config.DatetimeColumn("Upload Time", format="DD/MM/YYYY HH:mm"),
                    "match_rate_pct": st.column_config.ProgressColumn("Match Rate %", min_value=0, max_value=100)
                }
            )
        else:
            st.info("No upload history yet.")
    
    # ==================== TAB 4: REFRESH MASTER ====================
    with tab4:
        st.header("🔄 Refresh Master Data")
        st.caption("Upload barcode master file to add NEW entries to database")
        
        st.info("""
        📌 **This will:**
        - Load barcode data from your file (CSV/Excel)
        - Compare with existing barcodes in database
        - Add ONLY NEW barcodes (skips existing ones)
        - Show you what was added
        """)
        
        # File uploader
        master_file = st.file_uploader(
            "Choose Master Barcode File",
            type=['csv', 'xlsx', 'xls'],
            key="master_upload",
            help="File should have columns: VC_ITEM_BARCODE, VC_ITEM_CODE"
        )
        
        if master_file:
            try:
                # Read file
                if master_file.name.endswith('.csv'):
                    df_master = pd.read_csv(master_file, dtype=str, encoding='utf-8')
                else:
                    df_master = pd.read_excel(master_file, dtype=str)
                
                st.success(f"✅ File loaded: {len(df_master):,} rows")
                
                # Show preview
                with st.expander("📄 Preview Data (first 10 rows)"):
                    st.dataframe(df_master.head(10))
                
                # Validate columns
                required_cols = ['VC_ITEM_BARCODE', 'VC_ITEM_CODE']
                missing_cols = [col for col in required_cols if col not in df_master.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                    st.info(f"Available columns: {', '.join(df_master.columns)}")
                else:
                    # Clean data
                    df_master = df_master[required_cols].copy()
                    df_master = df_master.dropna(subset=['VC_ITEM_BARCODE'])
                    
                    # Remove .0 suffix from numeric barcodes
                    df_master['VC_ITEM_BARCODE'] = df_master['VC_ITEM_BARCODE'].astype(str).str.strip()
                    df_master['VC_ITEM_BARCODE'] = df_master['VC_ITEM_BARCODE'].str.replace(r'\.0$', '', regex=True)
                    df_master['VC_ITEM_CODE'] = df_master['VC_ITEM_CODE'].astype(str).str.strip()
                    
                    # Show stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Rows in File", f"{len(df_master):,}")
                    with col2:
                        st.metric("Unique Barcodes", f"{df_master['VC_ITEM_BARCODE'].nunique():,}")
                    
                    # Load button
                    if st.button("🚀 Load to Master Database", type="primary", use_container_width=True):
                        with st.spinner("Loading new barcodes to database..."):
                            try:
                                with get_db_connection() as conn:
                                    cursor = conn.cursor()
                                    
                                    # Get existing barcodes
                                    cursor.execute("SELECT vc_item_barcode FROM barcode_item_master WHERE is_active = TRUE")
                                    existing_barcodes = {row[0] for row in cursor.fetchall()}
                                    
                                    st.info(f"📊 Current database has {len(existing_barcodes):,} barcodes")
                                    
                                    # Filter to only NEW barcodes
                                    df_new = df_master[~df_master['VC_ITEM_BARCODE'].isin(existing_barcodes)].copy()
                                    
                                    if df_new.empty:
                                        st.warning("⚠️ No new barcodes to add. All barcodes already exist in database.")
                                    else:
                                        st.info(f"🆕 Found {len(df_new):,} NEW barcodes to add")
                                        
                                        # Show what will be added
                                        with st.expander(f"📋 Preview NEW Barcodes ({len(df_new)} items)", expanded=True):
                                            st.dataframe(
                                                df_new.head(50), 
                                                use_container_width=True,
                                                column_config={
                                                    "VC_ITEM_BARCODE": "Barcode",
                                                    "VC_ITEM_CODE": "Item Code"
                                                }
                                            )
                                            if len(df_new) > 50:
                                                st.caption(f"Showing first 50 of {len(df_new)} new barcodes")
                                        
                                        # Insert new barcodes
                                        insert_count = 0
                                        failed_count = 0
                                        
                                        progress_bar = st.progress(0)
                                        status_text = st.empty()
                                        
                                        for idx, row in df_new.iterrows():
                                            try:
                                                cursor.execute("""
                                                    INSERT INTO barcode_item_master (vc_item_barcode, vc_item_code)
                                                    VALUES (%s, %s)
                                                    ON CONFLICT (vc_item_barcode) DO NOTHING
                                                """, (str(row['VC_ITEM_BARCODE']), str(row['VC_ITEM_CODE'])))
                                                insert_count += 1
                                            except Exception as e:
                                                failed_count += 1
                                            
                                            # Update progress
                                            if idx % 100 == 0:
                                                progress = (idx + 1) / len(df_new)
                                                progress_bar.progress(progress)
                                                status_text.text(f"Processing: {idx + 1}/{len(df_new)}")
                                        
                                        conn.commit()
                                        cursor.close()
                                        
                                        progress_bar.progress(1.0)
                                        status_text.empty()
                                        
                                        # Show results
                                        st.success(f"✅ Successfully added {insert_count:,} new barcodes to master table!")
                                        
                                        if failed_count > 0:
                                            st.warning(f"⚠️ Failed to insert {failed_count} barcodes")
                                        
                                        # Summary
                                        st.divider()
                                        st.subheader("📊 Load Summary")
                                        
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("File Rows", f"{len(df_master):,}")
                                        with col2:
                                            st.metric("Already Existed", f"{len(df_master) - len(df_new):,}")
                                        with col3:
                                            st.metric("✅ Added", f"{insert_count:,}", delta=f"+{insert_count}")
                                        with col4:
                                            st.metric("❌ Failed", f"{failed_count:,}")
                                        
                                        st.balloons()
                                        
                                        # Clear cache
                                        st.cache_data.clear()
                                        
                            except Exception as e:
                                st.error(f"❌ Error loading data: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # ==================== TAB 5: FORMAT HELP ====================
    with tab5:
        st.header("📋 CSV/Excel Format Guide")
        
        st.subheader("✅ Correct Format")
        st.code("""barcode
9853963126000
5870000000000
033844004019
03000
G524
087000006935""", language="csv")
        
        st.subheader("🎯 Key Points")
        st.markdown("""
        **Trailing Zeros are REAL:**
        - `9853963126000` ← Valid EAN-13 (13 digits)
        - `5870000000000` ← Valid barcode
        - `03000` ← Valid item code
        
        These zeros are **part of the barcode**, not Excel corruption!
        
        **Format Your Excel:**
        1. Select barcode column
        2. Format Cells → Text → OK
        3. Then paste/type barcodes
        4. Save as XLSX (best) or CSV UTF-8
        
        **Verify Before Upload:**
        ```
        cd barcode_matcher\\batch
        validate_my_csv.bat "C:\\path\\to\\your\\file.csv"
        ```
        
        **Sample Files:**
        - `test_upload_numeric.csv` - Test file
        - `sample_db_format.csv` - Database format reference
        """)
        
        st.subheader("📖 Complete Guides")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**ZEROS_EXPLAINED.md**\nWhy barcodes have trailing zeros")
        with col2:
            st.info("**EXACT_FORMAT_GUIDE.md**\nStep-by-step format guide")
        
        st.divider()
        
        st.subheader("🔍 Database Sample")
        st.caption("These are actual barcodes from your database (with trailing zeros):")
        
        # Show real database samples
        try:
            with get_db_connection() as conn:
                sample_df = pd.read_sql("""
                    SELECT vc_item_barcode, vc_item_code, LENGTH(vc_item_barcode) as barcode_length
                    FROM barcode_item_master 
                    WHERE vc_item_barcode ~ '^[0-9]+000$'
                    LIMIT 10
                """, conn)
            
            if not sample_df.empty:
                st.dataframe(sample_df, use_container_width=True)
                st.caption("✅ All these barcodes end with 000 - this is NORMAL for EAN-13/UPC codes!")
            else:
                st.info("No trailing-zero examples found in database")
        except:
            st.warning("Could not load database samples")
        
        st.divider()
        
        # Show log
        st.subheader("Recent Load Log")
        log_file = "d:/Dashboard Code/NO_WH/DS/barcode_matcher/sample_data/barcode_load_log.txt"
        
        if os.path.exists(log_file):
            try:
                # Try different encodings for log file
                for encoding in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(log_file, 'r', encoding=encoding) as f:
                            log_lines = f.readlines()
                            recent_log = ''.join(log_lines[-30:])  # Last 30 lines
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    st.warning("Could not read log file with any encoding")
                    recent_log = None
                
                if recent_log:
                    st.code(recent_log, language='log')
            except Exception as e:
                st.error(f"Error reading log: {str(e)}")
        else:
            st.info("No log file yet")

if __name__ == '__main__':
    main()
