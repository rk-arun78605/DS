"""
test_serialtrackerdashboard.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Revamped single-view Serial Tracker Dashboard.
Four pipeline stages visible in one scroll:
  WH Receiving → WH Loading → Shop Receiving → Shop Compliance %
"""

import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re
import html
import io
import base64
from datetime import datetime, timedelta, date
from urllib.parse import quote, unquote

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Serial Tracker — Pipeline View",
    page_icon="https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────
DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'postgres',
    'password': 'hello',
    'database': 'WH'
}

DASHBOARD_DB_APP_NAME = "serial_tracker_dashboard"

def get_db_connection():
    try:
        return psycopg2.connect(
            **DB_CONFIG,
            application_name=DASHBOARD_DB_APP_NAME,
            options="-c statement_timeout=120000"
        )
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return None



@st.cache_data(ttl=300)
def get_dept_compliance_data(start_date, end_date):
    """Get compliance % by dept/grp/sub_group from mv_dept_compliance (pre-joined with item_dept_map)."""
    conn = get_db_connection()
    if not conn:
        return None, None

    start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
    end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)

    today = datetime.today().date()
    trend_end = today - timedelta(days=1)
    trend_start = trend_end - timedelta(days=6)

    query_summary = """
        SELECT
            dept, grp, sub_group,
            SUM(total_serials)  AS total_serials,
            SUM(total_yes)      AS total_yes,
            ROUND(SUM(total_yes)::NUMERIC / NULLIF(SUM(total_serials), 0) * 100, 1) AS yes_pct
        FROM mv_dept_compliance
        WHERE activity_date BETWEEN %(s)s AND %(e)s
          AND dept <> 'Unclassified'
        GROUP BY dept, grp, sub_group
        ORDER BY dept, yes_pct ASC NULLS FIRST
    """

    query_trend = """
        SELECT
            activity_date, dept, grp,
            SUM(total_serials) AS total_serials,
            SUM(total_yes)     AS total_yes
        FROM mv_dept_compliance
        WHERE activity_date BETWEEN %(s)s AND %(e)s
          AND dept <> 'Unclassified'
        GROUP BY activity_date, dept, grp
        ORDER BY activity_date, dept, grp
    """

    try:
        df_summary = pd.read_sql(query_summary, conn, params={'s': start_str, 'e': end_str})
        df_trend = pd.read_sql(query_trend, conn, params={'s': str(trend_start), 'e': str(trend_end)})
        conn.close()
        return df_summary, df_trend
    except Exception:
        if conn:
            conn.close()
        return None, None


def render_dept_compliance_section(start_s: str, end_s: str):
    """Render dept/grp/sub_group compliance % bar chart, 7-day trend, and summary table."""
    try:
        start_dt = pd.to_datetime(start_s).date()
        end_dt = pd.to_datetime(end_s).date()
    except Exception:
        st.info("Invalid date range for department compliance.")
        return

    df_dept_summary, df_dept_trend = get_dept_compliance_data(start_dt, end_dt)

    if df_dept_summary is None or df_dept_summary.empty:
        st.info("No department compliance data available for the selected date range.")
        return

    level_choice = st.radio(
        "View by:",
        ["Department", "Group", "Sub-Group"],
        horizontal=True,
        key="dept_compliance_level_test"
    )
    level_col = {"Department": "dept", "Group": "grp", "Sub-Group": "sub_group"}[level_choice]
    trend_color_col = {"Department": "dept", "Group": "grp", "Sub-Group": "grp"}[level_choice]

    agg_summary = (
        df_dept_summary
        .groupby(level_col, as_index=False)
        .agg(total_serials=('total_serials', 'sum'), total_yes=('total_yes', 'sum'))
    )
    agg_summary['yes_pct'] = (
        agg_summary['total_yes'] / agg_summary['total_serials'].replace(0, None) * 100
    ).round(1).fillna(0)
    agg_summary = agg_summary.sort_values('yes_pct')

    dept_col1, dept_col2 = st.columns([1.1, 1.9], gap="large")

    with dept_col1:
        st.markdown(f"##### Compliance % by {level_choice}")
        bar_height = max(380, len(agg_summary) * 28)
        fig_dept_bar = px.bar(
            agg_summary,
            x='yes_pct',
            y=level_col,
            orientation='h',
            color='yes_pct',
            color_continuous_scale=['#ed1b24', '#fd7e14', '#28a745'],
            range_color=[0, 100],
            text='yes_pct',
            labels={'yes_pct': 'Compliance %', level_col: level_choice},
            title=f'{level_choice} Compliance % (Selected Range)',
        )
        fig_dept_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_dept_bar.update_layout(
            height=bar_height,
            xaxis=dict(range=[0, 115], title='Compliance %', ticksuffix='%'),
            yaxis=dict(title='', automargin=True),
            coloraxis_showscale=False,
            showlegend=False,
            plot_bgcolor='rgba(248, 250, 252, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11),
            title_font=dict(size=14),
            margin=dict(l=10, r=70, t=40, b=20)
        )
        st.plotly_chart(fig_dept_bar, use_container_width=True, key="dept_compliance_bar_test")

    with dept_col2:
        st.markdown("##### 7-Day Compliance Trend")
        if df_dept_trend is not None and not df_dept_trend.empty:
            df_dept_trend = df_dept_trend.copy()
            df_dept_trend['yes_pct'] = (
                df_dept_trend['total_yes'].astype(float) /
                df_dept_trend['total_serials'].replace({0: None}).astype(float) * 100
            ).round(1).fillna(0)
            df_dept_trend['activity_date'] = pd.to_datetime(df_dept_trend['activity_date']).dt.strftime('%d %b')

            top_items = agg_summary.nlargest(8, 'total_serials')[level_col].tolist()
            trend_agg = (
                df_dept_trend[df_dept_trend[trend_color_col].isin(top_items)]
                .groupby(['activity_date', trend_color_col], as_index=False)
                .agg(total_serials=('total_serials', 'sum'), total_yes=('total_yes', 'sum'))
            )
            trend_agg['yes_pct'] = (
                trend_agg['total_yes'] / trend_agg['total_serials'].replace({0: None}) * 100
            ).round(1).fillna(0)

            fig_trend = px.line(
                trend_agg,
                x='activity_date',
                y='yes_pct',
                color=trend_color_col,
                markers=True,
                labels={'yes_pct': 'Compliance %', 'activity_date': 'Date', trend_color_col: level_choice},
                title=f'Compliance % Trend by {level_choice} (Last 7 Days)',
            )
            fig_trend.update_traces(line=dict(width=2.5), marker=dict(size=7))
            fig_trend.update_layout(
                height=max(380, bar_height),
                yaxis=dict(ticksuffix='%', range=[0, 105], title='Compliance %'),
                xaxis=dict(title='Date'),
                legend=dict(title=level_choice, orientation='v', x=1.01, y=1, font=dict(size=10)),
                plot_bgcolor='rgba(248, 250, 252, 0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=11),
                title_font=dict(size=14),
                hovermode='x unified',
                margin=dict(l=10, r=10, t=40, b=20)
            )
            st.plotly_chart(fig_trend, use_container_width=True, key="dept_compliance_trend_test")
        else:
            st.info("No 7-day trend data available.")

    st.markdown(f"##### {level_choice} Compliance Summary Table")
    display_dept = agg_summary.rename(columns={
        level_col: level_choice,
        'total_serials': 'Total Serials',
        'total_yes': 'Total Yes (Y)',
        'yes_pct': 'Compliance %'
    }).sort_values('Compliance %')
    st.dataframe(
        display_dept,
        use_container_width=True,
        hide_index=True,
        column_config={
            level_choice: st.column_config.TextColumn(level_choice, width='medium'),
            'Total Serials': st.column_config.NumberColumn('Total Serials', format='%d'),
            'Total Yes (Y)': st.column_config.NumberColumn('Total Yes (Y)', format='%d'),
            'Compliance %': st.column_config.ProgressColumn('Compliance %', format='%.1f%%', min_value=0, max_value=100)
        }
    )


def render_shop_compliance_last10days_chart(start_s: str, end_s: str):
    """Line graph for Shop Compliance % over strict last 10 days."""
    try:
        end_date = pd.to_datetime(end_s).date()
    except Exception:
        return

    trend_start = end_date - timedelta(days=9)
    trend_df = load_shop_selling_trend("Compliance %", trend_start.strftime('%Y-%m-%d'), end_s)

    if trend_df is None or trend_df.empty:
        st.info("No data for Shop Compliance % trend in last 10 days.")
        return

    trend_df = trend_df.copy()
    trend_df["date"] = pd.to_datetime(trend_df["date"], errors="coerce")
    trend_df["total"] = pd.to_numeric(trend_df["total"], errors="coerce")
    trend_df = trend_df.dropna(subset=["date", "total"]).sort_values("date")

    if trend_df.empty:
        st.info("No valid numeric data for Shop Compliance % trend.")
        return

    full_dates = pd.date_range(start=trend_start, end=end_date, freq='D')
    trend_df = (
        trend_df.set_index("date")
        .reindex(full_dates)
        .rename_axis("date")
        .reset_index()
    )
    trend_df["label"] = trend_df["total"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "")

    fig = px.line(
        trend_df,
        x="date",
        y="total",
        text="label",
        markers=True,
    )
    fig.update_traces(
        line=dict(color="#9b5bff", width=3, shape="spline"),
        marker=dict(size=8, color="#9b5bff", line=dict(color="rgba(255,255,255,0.85)", width=1.5)),
        textposition="top center",
        cliponaxis=False,
        hovertemplate="%{x|%d %b %Y}<br>Compliance: %{y:.1f}%<extra></extra>",
    )
    fig.update_layout(
        height=235,
        margin=dict(l=12, r=12, t=10, b=28),
        xaxis_title=None,
        yaxis_title="Compliance %",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,40,71,0.3)",
        font=dict(family="Inter", size=11, color="#FFFFFF"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(26,40,71,0.95)", font_size=12,
                        font_family="Inter", font_color="white"),
        xaxis=dict(showgrid=False, gridcolor="rgba(91,84,255,0.1)"),
        yaxis=dict(autorange=True, showgrid=True, gridcolor="rgba(148,163,184,0.18)"),
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(full_dates),
        ticktext=[d.strftime('%d %b') for d in full_dates],
        tickangle=-25,
    )
    st.plotly_chart(fig, use_container_width=True, key="shop_compliance_last10d")

def cancel_dashboard_db_activity():
    """Cancel active and terminate idle-in-transaction sessions for this dashboard only."""
    conn = get_db_connection()
    if not conn:
        return 0, 0

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH targets AS (
                    SELECT pid, state
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                      AND usename = current_user
                      AND application_name = %s
                      AND pid <> pg_backend_pid()
                      AND state IN ('active', 'idle in transaction')
                )
                SELECT
                    COALESCE(SUM(CASE WHEN state = 'active' AND pg_cancel_backend(pid) THEN 1 ELSE 0 END), 0) AS canceled_count,
                    COALESCE(SUM(CASE WHEN state = 'idle in transaction' AND pg_terminate_backend(pid) THEN 1 ELSE 0 END), 0) AS terminated_count
                FROM targets
                """,
                (DASHBOARD_DB_APP_NAME,)
            )
            row = cur.fetchone() or (0, 0)
            canceled = int(row[0] or 0)
            terminated = int(row[1] or 0)
            return canceled, terminated
    except Exception:
        return 0, 0
    finally:
        try:
            conn.close()
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
# SERIAL NUMBER SEQUENCE VALIDATION MODEL
# ─────────────────────────────────────────────────────────────
def extract_numeric_groups(serial: str) -> list:
    """
    Extract all numeric groups from serial with their positions.
    Example: 'NASD218FL042504495' → [(4, '218'), (9, '042504495')]
    Returns: list of (start_pos, numeric_string) tuples
    """
    if not serial or not isinstance(serial, str):
        return []
    
    groups = []
    current_num = ''
    start_pos = -1
    
    for i, char in enumerate(serial):
        if char.isdigit():
            if not current_num:
                start_pos = i
            current_num += char
        else:
            if current_num:
                groups.append((start_pos, current_num))
                current_num = ''
    
    if current_num:
        groups.append((start_pos, current_num))
    
    return groups

def get_serial_structure(serial: str) -> str:
    """
    Get the structure pattern of serial (digits vs letters vs others).
    Example: 'NASD218FL042504495' → 'LLLL999LL999999999'
    """
    if not serial or not isinstance(serial, str):
        return ''
    
    structure = ''
    for char in serial:
        if char.isdigit():
            structure += '9'
        elif char.isalpha():
            structure += 'L'
        else:
            structure += 'X'
    
    return structure

def compare_alphanumeric_patterns(serials: list) -> dict:
    """
    Analyze a group of serials to identify which numeric group might be sequential.
    
    Returns: {
        'compatible': bool,
        'common_structure': str,
        'sequential_group_index': int,  # Which numeric group changes
        'static_parts': {'prefix': str, 'suffix': str},
        'numeric_groups': [...]
    }
    """
    if not serials or len(serials) < 2:
        return {'compatible': False, 'common_structure': '', 'sequential_group_index': -1}
    
    clean_serials = [s.strip() for s in serials if s and isinstance(s, str)]
    clean_serials = sorted(set(clean_serials))
    
    if len(clean_serials) < 2:
        return {'compatible': False, 'common_structure': '', 'sequential_group_index': -1}
    
    # Check structure compatibility
    structures = [get_serial_structure(s) for s in clean_serials]
    first_structure = structures[0]
    all_same_structure = all(s == first_structure for s in structures)
    
    result = {
        'compatible': all_same_structure,
        'common_structure': first_structure,
        'sequential_group_index': -1,
        'static_parts': {'prefix': '', 'suffix': ''},
        'numeric_groups': []
    }
    
    if not all_same_structure:
        return result
    
    # Extract numeric groups from all serials
    all_numeric_groups = []
    for serial in clean_serials:
        groups = extract_numeric_groups(serial)
        all_numeric_groups.append(groups)
    
    result['numeric_groups'] = all_numeric_groups
    
    # Find which numeric group differs (most likely to be sequential)
    if all_numeric_groups:
        num_groups = len(all_numeric_groups[0])
        for group_idx in range(num_groups):
            # Check if this group varies across serials
            group_values = [int(g[group_idx][1]) if group_idx < len(g) else None 
                          for g in all_numeric_groups]
            group_values = [v for v in group_values if v is not None]
            
            if len(set(group_values)) > 1:  # This group has variations
                result['sequential_group_index'] = group_idx
                break
    
    return result

def find_gaps_in_numeric_sequence(numeric_values: list) -> list:
    """
    Find gaps in a sequence of numeric values.
    Example: [001, 002, 004, 005] → gaps at [003]
    
    Returns: list of gap dicts with missing values and surrounding values
    """
    if not numeric_values or len(numeric_values) < 2:
        return []
    
    gaps = []
    sorted_vals = sorted(numeric_values)
    
    for i in range(len(sorted_vals) - 1):
        curr = sorted_vals[i]
        next_val = sorted_vals[i + 1]
        
        if next_val - curr > 1:
            missing = list(range(curr + 1, next_val))
            gaps.append({
                'missing_count': len(missing),
                'missing_values': missing,
                'between': [curr, next_val],
                'gap_size': next_val - curr
            })
    
    return gaps

def find_missing_serials_in_sequence(serials: list, item_code: str = None, 
                                     item_desc: str = None) -> dict:
    """
    Detect missing serials in a sequence (handles alphanumeric).
    Works with any serial format: numeric, alphanumeric, mixed.
    
    Args:
        serials: List of serial numbers to check
        item_code: Optional item code to help identify batches
        item_desc: Optional item description for additional context
    
    Returns:
        {
            'has_gaps': bool,
            'gaps': [{'missing': [...], 'between': [...], 'confidence': float}],
            'format_info': {...},
            'batch_info': {'item_code': ..., 'item_desc': ..., 'count': ...},
            'confidence': float
        }
    """
    if not serials or len(serials) < 2:
        return {
            'has_gaps': False,
            'gaps': [],
            'format_info': {},
            'batch_info': {'item_code': item_code, 'item_desc': item_desc, 'count': len(serials) if serials else 0},
            'confidence': 0.0
        }
    
    clean_serials = [s.strip() for s in serials if s and isinstance(s, str)]
    clean_serials = sorted(set(clean_serials))
    
    result = {
        'has_gaps': False,
        'gaps': [],
        'format_info': {},
        'batch_info': {'item_code': item_code, 'item_desc': item_desc, 'count': len(clean_serials)},
        'confidence': 0.5
    }
    
    if len(clean_serials) < 2:
        return result
    
    # Analyze pattern compatibility
    pattern_analysis = compare_alphanumeric_patterns(clean_serials)
    result['format_info'] = pattern_analysis
    
    if not pattern_analysis['compatible']:
        result['confidence'] = 0.4
        return result
    
    result['confidence'] = 0.8
    
    # Find the sequential numeric group
    seq_group_idx = pattern_analysis.get('sequential_group_index', -1)
    if seq_group_idx == -1:
        return result
    
    # Extract values from the sequential group
    numeric_values = []
    numeric_to_serial = {}
    
    for serial, groups in zip(clean_serials, pattern_analysis.get('numeric_groups', [])):
        if seq_group_idx < len(groups):
            _, num_str = groups[seq_group_idx]
            try:
                num_val = int(num_str)
                numeric_values.append(num_val)
                numeric_to_serial[num_val] = serial
            except ValueError:
                pass
    
    if len(numeric_values) < 2:
        return result
    
    # Find gaps in sequence
    gaps = find_gaps_in_numeric_sequence(numeric_values)
    
    if gaps:
        result['has_gaps'] = True
        result['confidence'] = 0.95
        
        for gap in gaps:
            # Reconstruct missing serials based on pattern
            result['gaps'].append({
                'missing_count': gap['missing_count'],
                'missing_values': gap['missing_values'],
                'between_serials': [numeric_to_serial.get(gap['between'][0], str(gap['between'][0])),
                                   numeric_to_serial.get(gap['between'][1], str(gap['between'][1]))],
                'gap_size': gap['gap_size'],
                'confidence': 0.95
            })
    
    return result

def validate_serial_not_in_wh(serial: str, item_code: str, 
                               wh_serials_dict: dict) -> dict:
    """
    Check if serial is truly missing or if it's a sequence format issue.
    
    Args:
        serial: Serial from Shop Compliance %
        item_code: Item code being sold
        wh_serials_dict: Dict of {item_code: [serials]} from WH
    
    Returns:
        {
            'is_in_wh': bool,
            'reason': 'found|missing|format_issue|sequence_gap',
            'similar_serials': [...],
            'suggestion': str
        }
    """
    if item_code not in wh_serials_dict:
        return {
            'is_in_wh': False,
            'reason': 'item_not_in_wh',
            'similar_serials': [],
            'suggestion': f'Item code {item_code} not found in WH records'
        }
    
    wh_serials = wh_serials_dict[item_code]
    
    # Exact match
    if serial in wh_serials:
        return {
            'is_in_wh': True,
            'reason': 'found',
            'similar_serials': [],
            'suggestion': 'Serial exists in WH'
        }
    
    # Check sequence analysis
    seq_analysis = find_missing_serials_in_sequence(
        wh_serials + [serial],
        item_code=item_code
    )
    
    if seq_analysis['has_gaps']:
        # Find if this serial fits a gap
        for gap in seq_analysis['gaps']:
            if gap.get('missing_count', 0) > 0:
                between_serials = gap.get('between_serials', [])
                return {
                    'is_in_wh': False,
                    'reason': 'sequence_gap',
                    'similar_serials': between_serials if between_serials else [],
                    'suggestion': f"Serial {serial} matches a sequence gap pattern. "
                                  f"Goes between {between_serials[0] if between_serials else ''} and {between_serials[1] if len(between_serials) > 1 else ''}"
                }
    
    # Find similar serials by comparing structure
    serial_structure = get_serial_structure(serial)
    similar = []
    for wh_serial in wh_serials:
        wh_structure = get_serial_structure(wh_serial)
        # Check if structures are similar (same length, similar patterns)
        if len(serial_structure) == len(wh_structure):
            similar.append(wh_serial)
    
    return {
        'is_in_wh': False,
        'reason': 'missing' if not similar else 'format_issue',
        'similar_serials': similar[:3],
        'suggestion': f'Serial not in WH. Similar serials: {similar[:2] if similar else "none"}'
    }


def fmt(num):
    """Format integer with commas."""
    try:
        return f"{int(num):,}"
    except Exception:
        return "0"

def pct(num, denom):
    """Calculate percentage safely."""
    return round(num / denom * 100, 1) if denom > 0 else 0.0

# ─────────────────────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_wh_receiving(start_str: str, end_str: str):
    """Load WH Receiving aggregated metrics."""
    conn = get_db_connection()
    if not conn:
        return {}
    q = """
    SELECT
        COUNT(*)                                                        AS total_received,
        COUNT(DISTINCT CASE
            WHEN serial_no IS NOT NULL AND TRIM(serial_no) <> '' THEN serial_no
        END)                                                            AS unique_serials,
        COUNT(CASE
            WHEN serial_no IS NOT NULL AND TRIM(serial_no) <> '' THEN 1
        END) - COUNT(DISTINCT CASE
            WHEN serial_no IS NOT NULL AND TRIM(serial_no) <> '' THEN serial_no
        END)                                                            AS duplicate_count,
        COUNT(CASE
            WHEN serial_no IS NULL OR TRIM(serial_no) = '' THEN 1
        END)                                                            AS blank_serials,
        COUNT(CASE
            WHEN serial_no IS NOT NULL AND TRIM(serial_no) <> ''
             AND LENGTH(TRIM(serial_no)) < 6 THEN 1
        END)                                                            AS small_serials,
        COUNT(*) FILTER (
            WHERE serial_no IS NOT NULL AND TRIM(serial_no) != ''
              AND serial_no IN (
                SELECT serial_no FROM whreceived_serialno
                WHERE grn_date BETWEEN %(s)s AND %(e)s
                  AND serial_no IS NOT NULL AND TRIM(serial_no) != ''
                GROUP BY serial_no HAVING COUNT(*) > 1
              )
        )                                                               AS wh_db_duplicates,
        CASE
            WHEN COUNT(*) > 0
            THEN ROUND(COUNT(DISTINCT CASE
                WHEN serial_no IS NOT NULL AND TRIM(serial_no) <> '' THEN serial_no END
            )::NUMERIC / COUNT(*) * 100, 2)
            ELSE 0
        END                                                             AS unique_percentage
    FROM whreceived_serialno
    WHERE grn_date BETWEEN %(s)s AND %(e)s
    """
    try:
        df = pd.read_sql(q, conn, params={'s': start_str, 'e': end_str})
        conn.close()
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception as ex:
        st.error(f"WH Receiving query error: {ex}")
        try: conn.close()
        except: pass
        return {}


@st.cache_data(ttl=300)
def load_wh_receiving_drilldown(start_str: str, end_str: str, metric: str):
    conn = get_db_connection()
    if not conn:
        return None

    metric_filter = ""
    if metric in ["Unique %", "Unique Serials"]:
        metric_filter = "AND serial_no IS NOT NULL AND TRIM(serial_no) != ''"
    elif metric == "In-Batch Dup":
        metric_filter = """
        AND serial_no IN (
            SELECT serial_no
            FROM whreceived_serialno
            WHERE grn_date BETWEEN %(s)s AND %(e)s
              AND serial_no IS NOT NULL AND TRIM(serial_no) != ''
            GROUP BY serial_no HAVING COUNT(*) > 1
        )
        """
    elif metric == "WH DB Dup":
        metric_filter = """
        AND serial_no IN (
            SELECT DISTINCT serial_no
            FROM whreceived_serialno
            WHERE grn_date IS NOT NULL
              AND grn_date::DATE < %(s)s
              AND serial_no IS NOT NULL AND TRIM(serial_no) != ''
            GROUP BY serial_no HAVING COUNT(*) > 1
        )
        AND serial_no NOT IN (
            SELECT DISTINCT serial_no
            FROM whreceived_serialno
            WHERE grn_date IS NOT NULL
              AND grn_date::DATE BETWEEN %(s)s AND %(e)s
              AND serial_no IS NOT NULL AND TRIM(serial_no) != ''
            GROUP BY serial_no HAVING COUNT(*) > 1
        )
        """
    elif metric == "Blank Serials":
        metric_filter = "AND (serial_no IS NULL OR TRIM(serial_no) = '')"
    elif metric == "Small (≤6)":
        metric_filter = "AND serial_no IS NOT NULL AND TRIM(serial_no) != '' AND LENGTH(TRIM(serial_no)) < 6"

    q = f"""
    SELECT
        TRIM(serial_no) AS serial_no,
        COALESCE(NULLIF(TRIM(item_code),''),'Unknown') AS item_code,
        COALESCE(NULLIF(TRIM(item_desc),''),'Unknown') AS item_desc,
        COALESCE(NULLIF(TRIM(vc_inbond_type),''),'Unknown') AS vc_inbond_type,
        grn_date::DATE AS grn_date,
        COALESCE(NULLIF(TRIM(warehouse_name),''),'Unknown') AS warehouse_name
    FROM whreceived_serialno
    WHERE grn_date BETWEEN %(s)s AND %(e)s
        {metric_filter}
    ORDER BY grn_date DESC, serial_no
    """
    try:
        df = pd.read_sql(q, conn, params={'s': start_str, 'e': end_str})
        conn.close()
        return df if df is not None and not df.empty else None
    except Exception as ex:
        st.error(f"WH Receiving drilldown error: {ex}")
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def load_wh_receiving_quality_trend(end_str: str, group_by: str = "daily"):
    """Load WH Receiving quality metrics grouped by day (last 30) or month (YTD)."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        end_dt = pd.to_datetime(end_str, errors="coerce")
        if pd.isna(end_dt):
            conn.close()
            return None
        if group_by == "daily":
            start_str = (end_dt - timedelta(days=29)).strftime("%Y-%m-%d")
            group_expr = "grn_date::DATE"
        else:
            start_str = end_dt.replace(month=1, day=1).strftime("%Y-%m-%d")
            group_expr = "DATE_TRUNC('month', grn_date)::DATE"
        q = f"""
        WITH inbatch_dup_serials AS (
            -- Serials appearing multiple times in the selected date range (today/this period)
            SELECT DISTINCT serial_no
            FROM whreceived_serialno
            WHERE grn_date IS NOT NULL
              AND grn_date::DATE BETWEEN %(s)s AND %(e)s
              AND serial_no IS NOT NULL AND TRIM(serial_no) != ''
            GROUP BY serial_no
            HAVING COUNT(*) > 1
        ),
        historical_dup_serials AS (
            -- Serials appearing multiple times in database BEFORE the selected range
            SELECT DISTINCT serial_no
            FROM whreceived_serialno
            WHERE grn_date IS NOT NULL
              AND grn_date::DATE < %(s)s
              AND serial_no IS NOT NULL AND TRIM(serial_no) != ''
            GROUP BY serial_no
            HAVING COUNT(*) > 1
        ),
        agg AS (
            SELECT
                {group_expr} AS period,
                COUNT(*)                                                                                         AS total,
                COUNT(DISTINCT CASE WHEN serial_no IS NOT NULL AND TRIM(serial_no) != '' THEN serial_no END)     AS unique_cnt,
                COUNT(CASE WHEN serial_no IS NOT NULL AND TRIM(serial_no) != '' THEN 1 END)
                    - COUNT(DISTINCT CASE WHEN serial_no IS NOT NULL AND TRIM(serial_no) != '' THEN serial_no END) AS inbatch_dup,
                COUNT(CASE WHEN serial_no IS NULL OR TRIM(serial_no) = '' THEN 1 END)                            AS blank_cnt,
                COUNT(CASE WHEN serial_no IS NOT NULL AND TRIM(serial_no) != ''
                            AND LENGTH(TRIM(serial_no)) < 6 THEN 1 END)                                         AS small_cnt,
                COUNT(CASE WHEN serial_no IN (SELECT serial_no FROM historical_dup_serials) 
                            AND serial_no NOT IN (SELECT serial_no FROM inbatch_dup_serials)
                            THEN 1 END)                                                                         AS wh_db_dup
            FROM whreceived_serialno
            WHERE grn_date IS NOT NULL
              AND grn_date::DATE BETWEEN %(s)s AND %(e)s
            GROUP BY 1
        )
        SELECT
            period,
            total,
            ROUND(unique_cnt::NUMERIC   / NULLIF(total, 0) * 100, 1) AS unique_pct,
            ROUND(inbatch_dup::NUMERIC  / NULLIF(total, 0) * 100, 1) AS inbatch_dup_pct,
            ROUND(blank_cnt::NUMERIC    / NULLIF(total, 0) * 100, 1) AS blank_pct,
            ROUND(small_cnt::NUMERIC    / NULLIF(total, 0) * 100, 1) AS small_pct,
            ROUND(wh_db_dup::NUMERIC    / NULLIF(total, 0) * 100, 1) AS wh_db_dup_pct,
            (inbatch_dup = 0 AND blank_cnt = 0 AND small_cnt = 0 AND wh_db_dup = 0) AS is_clean
        FROM agg
        ORDER BY period
        """
        df = pd.read_sql(q, conn, params={"s": start_str, "e": end_str})
        if df is not None and group_by == "daily":
            # Force a complete rolling 30-day axis so selected end date is always visible,
            # even when there are no rows for some days.
            full_dates = pd.DataFrame({
                "period": pd.date_range(start=start_str, end=end_str, freq="D")
            })
            df["period"] = pd.to_datetime(df["period"], errors="coerce")
            df = full_dates.merge(df, on="period", how="left")
            no_data_mask = df["total"].isna()
            # For days with zero WH records, show a stable baseline instead of blanks.
            df.loc[no_data_mask, "total"] = 0
            df.loc[no_data_mask, "unique_pct"] = 100.0
            df.loc[no_data_mask, "inbatch_dup_pct"] = 0.0
            df.loc[no_data_mask, "blank_pct"] = 0.0
            df.loc[no_data_mask, "small_pct"] = 0.0
            df.loc[no_data_mask, "wh_db_dup_pct"] = 0.0
            df.loc[no_data_mask, "is_clean"] = True
        conn.close()
        return df if df is not None and not df.empty else None
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None


@st.cache_data(ttl=300)
def load_wh_loading(start_str: str, end_str: str):
    conn = get_db_connection()
    if not conn:
        return {}
    q = """
    WITH base AS (
        SELECT
            serial_no,
            vc_item_code::TEXT AS item_code,
            REGEXP_REPLACE(TRIM(vc_item_code::TEXT), '[^A-Za-z0-9]','','g') AS item_code_clean
        FROM serial_no_dailydata
        WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
    ),
    serial_item AS (
        SELECT
            serial_no,
            COUNT(DISTINCT item_code_clean) AS item_code_count
        FROM base
        WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
        GROUP BY serial_no
    )
    SELECT
        COUNT(*)                                                                 AS loaded_total,
        COUNT(DISTINCT b.serial_no) FILTER (
            WHERE b.serial_no IS NOT NULL AND LENGTH(TRIM(b.serial_no)) > 0
        )                                                                        AS loaded_unique,
        COUNT(*) FILTER (
            WHERE b.serial_no IS NULL OR LENGTH(TRIM(COALESCE(b.serial_no,''))) = 0
        )                                                                        AS loaded_blank,
        COUNT(*) FILTER (
            WHERE b.serial_no IS NOT NULL AND LENGTH(TRIM(b.serial_no)) > 0
                AND LENGTH(TRIM(b.serial_no)) <= 6
        )                                                                        AS loaded_small,
        COUNT(*) FILTER (
            WHERE b.serial_no IS NOT NULL AND LENGTH(TRIM(b.serial_no)) > 0
              AND REGEXP_REPLACE(TRIM(b.serial_no), '[^A-Za-z0-9]','','g') != ''
              AND REGEXP_REPLACE(TRIM(b.serial_no), '[^A-Za-z0-9]','','g') =
                  REGEXP_REPLACE(TRIM(b.item_code), '[^A-Za-z0-9]','','g')
        )                                                                        AS loaded_serial_is_ic,
        COUNT(DISTINCT b.serial_no) FILTER (
            WHERE s.item_code_count > 1
        )                                                                        AS loaded_dup,
        CASE
            WHEN COUNT(*) > 0
            THEN ROUND(COUNT(DISTINCT b.serial_no) FILTER (
                WHERE b.serial_no IS NOT NULL AND LENGTH(TRIM(b.serial_no)) > 0
            )::NUMERIC / COUNT(*) * 100, 2)
            ELSE 0
        END                                                                      AS unique_percentage
    FROM base b
    LEFT JOIN serial_item s ON s.serial_no = b.serial_no
    """
    try:
        df = pd.read_sql(q, conn, params={'s': start_str, 'e': end_str})
        conn.close()
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception as ex:
        st.error(f"WH Loading query error: {ex}")
        try: conn.close()
        except: pass
        return {}


@st.cache_data(ttl=300)
def load_shop_receiving(start_str: str, end_str: str):
    conn = get_db_connection()
    if not conn:
        return {}
    q = """
    SELECT
        -- Total offloaded (mod_date IS NOT NULL)
        COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL)                          AS sr_total,

        -- Not yet offloaded
        COUNT(*) FILTER (WHERE dt_mod_date IS NULL)                              AS not_offloaded,

        -- Unique shop serials
        COUNT(DISTINCT vc_serail_no) FILTER (
            WHERE dt_mod_date IS NOT NULL
              AND vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != ''
        )                                                                         AS sr_unique,

        -- Blank shop serials
        COUNT(*) FILTER (
            WHERE dt_mod_date IS NOT NULL
              AND (vc_serail_no IS NULL OR TRIM(COALESCE(vc_serail_no,'')) = '')
        )                                                                         AS sr_blank,

        -- Small shop serials
        COUNT(*) FILTER (
            WHERE dt_mod_date IS NOT NULL
              AND vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != ''
              AND LENGTH(TRIM(vc_serail_no)) < 6
        )                                                                         AS sr_small,

        -- IC: Serial matches Item Code
        COUNT(*) FILTER (
            WHERE dt_mod_date IS NOT NULL
              AND vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != ''
              AND REGEXP_REPLACE(TRIM(vc_serail_no), '[^A-Za-z0-9]','','g') != ''
              AND REGEXP_REPLACE(TRIM(vc_serail_no), '[^A-Za-z0-9]','','g') =
                  REGEXP_REPLACE(TRIM(vc_item_code::TEXT), '[^A-Za-z0-9]','','g')
        )                                                                         AS sr_ic,

        -- Same-day offload
        COUNT(*) FILTER (
            WHERE dt_mod_date IS NOT NULL AND dt_doc_date IS NOT NULL
              AND dt_mod_date::DATE = dt_doc_date::DATE
        )                                                                         AS same_day,

        -- Old doc date (mod outside selected range)
        COUNT(*) FILTER (
            WHERE dt_mod_date IS NOT NULL
              AND dt_mod_date::DATE NOT BETWEEN %(s)s AND %(e)s
        )                                                                         AS old_doc,


        -- WH→Shop serial match
        COUNT(*) FILTER (
            WHERE dt_mod_date IS NOT NULL
              AND vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != ''
              AND serial_no    IS NOT NULL AND TRIM(COALESCE(serial_no,''))    != ''
              AND TRIM(vc_serail_no) = TRIM(serial_no)
        )                                                                         AS wh_match,

        COUNT(*) FILTER (
            WHERE dt_mod_date IS NOT NULL
              AND vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != ''
              AND (serial_no IS NULL OR TRIM(COALESCE(serial_no,'')) = ''
                   OR TRIM(vc_serail_no) != TRIM(serial_no))
        )                                                                         AS wh_mismatch,


        COUNT(*) FILTER (
            WHERE dt_mod_date IS NOT NULL
              AND vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != ''
        ) - COUNT(DISTINCT vc_serail_no) FILTER (
            WHERE dt_mod_date IS NOT NULL
              AND vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != ''
        )                                                                         AS sr_dup

    FROM serial_no_dailydata
    WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
    """
    try:
        df = pd.read_sql(q, conn, params={'s': start_str, 'e': end_str})
        conn.close()
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception as ex:
        st.error(f"Shop Receiving query error: {ex}")
        try: conn.close()
        except: pass
        return {}


@st.cache_data(ttl=300)
def load_truck_summary(start_str: str, end_str: str):
    """
    Truck + qty summary for selected date range.
    Uses doc-date window so counts align with dashboard date filter semantics.
    Prefers table serial_no_daily_data, falls back to serial_no_dailydata.
    """
    conn = get_db_connection()
    if not conn:
        return {}

    query_tpl = """
    SELECT
        COUNT(DISTINCT TRIM(vc_vehicle_no)) FILTER (
            WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
              AND vc_vehicle_no IS NOT NULL AND TRIM(COALESCE(vc_vehicle_no,'')) <> ''
        ) AS wh_unique_trucks,

        COUNT(DISTINCT TRIM("VC_VEHICLE_NO_1")) FILTER (
                        WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
                            AND dt_mod_date IS NOT NULL
              AND "VC_VEHICLE_NO_1" IS NOT NULL AND TRIM(COALESCE("VC_VEHICLE_NO_1",'')) <> ''
        ) AS shop_unique_trucks,

        COUNT(*) FILTER (
            WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
              AND vc_vehicle_no IS NOT NULL AND TRIM(COALESCE(vc_vehicle_no,'')) <> ''
        ) AS wh_truck_loaded_qty,

        COUNT(*) FILTER (
                        WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
                            AND dt_mod_date IS NOT NULL
              AND "VC_VEHICLE_NO_1" IS NOT NULL AND TRIM(COALESCE("VC_VEHICLE_NO_1",'')) <> ''
        ) AS shop_truck_offloaded_qty
    FROM {table_name}
    """

    tried = []
    for table_name in ("serial_no_daily_data", "serial_no_dailydata"):
        tried.append(table_name)
        try:
            df = pd.read_sql(query_tpl.format(table_name=table_name), conn, params={'s': start_str, 'e': end_str})
            conn.close()
            return df.iloc[0].to_dict() if df is not None and not df.empty else {}
        except Exception:
            continue

    try:
        conn.close()
    except Exception:
        pass
    st.error(f"Truck summary query error: table not found/accessible ({', '.join(tried)})")
    return {}


@st.cache_data(ttl=300)
def load_shop_receiving_pct_trend(end_str: str):
    """Trend: For each loaded date, % with MOD_date present out of WH loaded on same date."""
    conn = get_db_connection()
    if not conn:
        return None


@st.cache_data(ttl=300)
def load_truck_7day_summary(end_str: str):
    """Last 7 doc-dates: WH loaded trucks, shop offloaded trucks, and offload %."""
    conn = get_db_connection()
    if not conn:
        return None
    q = """
    SELECT
        dt_doc_date::DATE AS date,
        COUNT(DISTINCT TRIM(vc_vehicle_no)) FILTER (
            WHERE vc_vehicle_no IS NOT NULL AND TRIM(COALESCE(vc_vehicle_no,'')) <> ''
        ) AS wh_trucks_loaded,
        COUNT(DISTINCT TRIM("VC_VEHICLE_NO_1")) FILTER (
            WHERE dt_mod_date IS NOT NULL
              AND "VC_VEHICLE_NO_1" IS NOT NULL AND TRIM(COALESCE("VC_VEHICLE_NO_1",'')) <> ''
        ) AS shop_trucks_offloaded,
        ROUND(
            COUNT(DISTINCT TRIM("VC_VEHICLE_NO_1")) FILTER (
                WHERE dt_mod_date IS NOT NULL
                  AND "VC_VEHICLE_NO_1" IS NOT NULL AND TRIM(COALESCE("VC_VEHICLE_NO_1",'')) <> ''
            )::NUMERIC
            / NULLIF(
                COUNT(DISTINCT TRIM(vc_vehicle_no)) FILTER (
                    WHERE vc_vehicle_no IS NOT NULL AND TRIM(COALESCE(vc_vehicle_no,'')) <> ''
                ),
                0
            ) * 100,
            1
        ) AS offload_pct
    FROM serial_no_dailydata
    WHERE dt_doc_date IS NOT NULL
      AND dt_doc_date::DATE BETWEEN (%(e)s::DATE - 6) AND %(e)s::DATE
    GROUP BY dt_doc_date::DATE
    ORDER BY dt_doc_date::DATE DESC
    """
    try:
        df = pd.read_sql(q, conn, params={"e": end_str})
        conn.close()
        return df if df is not None and not df.empty else None
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None
    q = f"""
    SELECT
        dt_doc_date::DATE AS date,
        COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL) AS shop_recv,
        COUNT(*) AS wh_loaded,
        CASE
            WHEN COUNT(*) > 0
            THEN ROUND((COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL))::NUMERIC / COUNT(*) * 100, 1)
            ELSE 0
        END AS pct
    FROM serial_no_dailydata
    WHERE dt_doc_date::DATE BETWEEN ('{end_str}'::DATE - 6) AND '{end_str}'::DATE
    GROUP BY dt_doc_date::DATE
    ORDER BY dt_doc_date::DATE
    """
    try:
        df = pd.read_sql(q, conn)
        conn.close()
        return df if df is not None and not df.empty else None
    except Exception:
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def load_shop_receiving_day_breakdown(date_str: str):
    """Selected-day offload breakdown: same day, 1-3 days, >3 days."""
    conn = get_db_connection()
    if not conn:
        return {}
    q = """
    SELECT
        COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL) AS total_offloaded,
        COUNT(*) FILTER (
            WHERE dt_mod_date IS NOT NULL AND dt_doc_date IS NOT NULL
              AND dt_mod_date::DATE = dt_doc_date::DATE
        ) AS same_day,
        COUNT(*) FILTER (
            WHERE dt_mod_date IS NOT NULL AND dt_doc_date IS NOT NULL
              AND (dt_mod_date::DATE - dt_doc_date::DATE) BETWEEN 1 AND 3
        ) AS days_1_3,
        COUNT(*) FILTER (
            WHERE dt_mod_date IS NOT NULL AND dt_doc_date IS NOT NULL
              AND (dt_mod_date::DATE - dt_doc_date::DATE) > 3
        ) AS days_gt_3
    FROM serial_no_dailydata
    WHERE dt_mod_date::DATE = %(d)s
    """
    try:
        df = pd.read_sql(q, conn, params={"d": date_str})
        conn.close()
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception:
        try: conn.close()
        except: pass
        return {}


@st.cache_data(ttl=300)
def load_shop_truck_shopwise_for_doc_date(date_str: str):
    """Shop-wise truck/qty summary for a selected WH loaded DOC date."""
    conn = get_db_connection()
    if not conn:
        return None
    q = """
    SELECT
        COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
        COALESCE(shop_name, 'Unknown Shop') AS shop_name,

        COUNT(DISTINCT TRIM(vc_vehicle_no)) FILTER (
            WHERE vc_vehicle_no IS NOT NULL AND TRIM(COALESCE(vc_vehicle_no,'')) <> ''
        ) AS wh_trucks_loaded,

        COUNT(*) FILTER (
            WHERE vc_vehicle_no IS NOT NULL AND TRIM(COALESCE(vc_vehicle_no,'')) <> ''
        ) AS wh_qty_loaded,

        COUNT(DISTINCT TRIM("VC_VEHICLE_NO_1")) FILTER (
            WHERE dt_mod_date IS NOT NULL
              AND "VC_VEHICLE_NO_1" IS NOT NULL AND TRIM(COALESCE("VC_VEHICLE_NO_1",'')) <> ''
        ) AS shop_trucks_offloaded,

        COUNT(*) FILTER (
            WHERE dt_mod_date IS NOT NULL
              AND "VC_VEHICLE_NO_1" IS NOT NULL AND TRIM(COALESCE("VC_VEHICLE_NO_1",'')) <> ''
        ) AS shop_qty_offloaded

    FROM serial_no_dailydata
    WHERE dt_doc_date::DATE = %(d)s
    GROUP BY vc_shop_code, shop_name
    HAVING COUNT(*) FILTER (
        WHERE vc_vehicle_no IS NOT NULL AND TRIM(COALESCE(vc_vehicle_no,'')) <> ''
    ) > 0
       OR COUNT(*) FILTER (
        WHERE dt_mod_date IS NOT NULL
          AND "VC_VEHICLE_NO_1" IS NOT NULL AND TRIM(COALESCE("VC_VEHICLE_NO_1",'')) <> ''
    ) > 0
    ORDER BY wh_qty_loaded DESC, shop_qty_offloaded DESC, shop_code
    """
    try:
        df = pd.read_sql(q, conn, params={"d": date_str})
        conn.close()
        return df if df is not None and not df.empty else None
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None


@st.cache_data(ttl=300)
def load_shop_selling(start_str: str, end_str: str):
    conn = get_db_connection()
    if not conn:
        return {}
    q = """
    WITH base AS (
                SELECT item_code, serial_number, serial_check, bill_date
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
          AND COALESCE(item_code,'') != 'sales data not available'
    ),
    dup_sn AS (
        SELECT serial_number FROM base
        WHERE serial_number IS NOT NULL AND TRIM(serial_number) != ''
        GROUP BY serial_number HAVING COUNT(*) > 1
    )
    SELECT
        COUNT(*)                                                              AS total_sold,
        COUNT(*) FILTER (WHERE TRIM(serial_check) = 'Y')                     AS compliance_yes,
        COUNT(*) FILTER (
            WHERE serial_number IN (SELECT serial_number FROM dup_sn)
        )                                                                     AS dup_sold,
        COUNT(*) FILTER (
            WHERE serial_number IS NULL OR TRIM(COALESCE(serial_number,'')) = ''
        )                                                                     AS no_serial,
        COUNT(*) FILTER (
            WHERE serial_number IS NOT NULL
              AND LENGTH(TRIM(COALESCE(serial_number,''))) > 0
              AND LENGTH(TRIM(serial_number)) <= 6
        )                                                                     AS small_serial,
        COUNT(*) FILTER (
            WHERE TRIM(serial_check) = 'N'
              AND serial_number IS NOT NULL AND TRIM(COALESCE(serial_number,'')) != ''
        )                                                                     AS not_in_wh,
        COUNT(*) FILTER (
            WHERE TRIM(serial_check) = 'N'
              AND serial_number IS NOT NULL AND TRIM(COALESCE(serial_number,'')) != ''
                            AND (
                                    EXISTS (
                                            SELECT 1
                                            FROM serialno_check_yes_no y_same
                                            WHERE TRIM(y_same.serial_check) = 'Y'
                                                AND TRIM(COALESCE(y_same.item_code,'')) = TRIM(COALESCE(base.item_code,''))
                                                AND DATE(y_same.bill_date) = DATE(base.bill_date)
                                    )
                                    OR EXISTS (
                                            SELECT 1
                                            FROM serialno_check_yes_no y_any
                                            WHERE TRIM(y_any.serial_check) = 'Y'
                                                AND TRIM(COALESCE(y_any.item_code,'')) = TRIM(COALESCE(base.item_code,''))
                                    )
                            )
                )                                                                     AS incorrect_serial_count,
        COUNT(*) FILTER (
            WHERE serial_number IS NOT NULL
              AND TRIM(COALESCE(serial_number,'')) != ''
              AND REGEXP_REPLACE(TRIM(serial_number), '[^A-Za-z0-9]','','g') != ''
              AND REGEXP_REPLACE(TRIM(serial_number), '[^A-Za-z0-9]','','g') = REGEXP_REPLACE(TRIM(item_code), '[^A-Za-z0-9]','','g')
        )                                                                     AS serial_is_ic
    FROM base
    """
    try:
        df = pd.read_sql(q, conn, params={'s': start_str, 'e': end_str})
        conn.close()
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception as ex:
        st.error(f"Shop Compliance % query error: {ex}")
        try: conn.close()
        except: pass
        return {}


@st.cache_data(ttl=300)
def detect_serial_sequence_misses(start_str: str, end_str: str):
    """
    Detect serials in Shop Compliance % (serial_check_yes_no) marked as 'Not in WH'.
    Check if these serials fit within a sequence gap found in whreceived_serialno.
    
    Works with any serial format: numeric, alphanumeric, mixed.
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        # Get all "Not in WH" serials from Shop Compliance % with their item codes
        not_in_wh_q = """
        SELECT 
            TRIM(cs.item_code) AS item_code,
            pii.item_desc,
            TRIM(cs.serial_number) AS serial_number,
            cs.bill_date::DATE AS bill_date
        FROM serialno_check_yes_no cs
        LEFT JOIN prod_item_info pii ON TRIM(pii.item_code) = TRIM(cs.item_code)
        WHERE cs.serial_check = 'N'
          AND cs.serial_number IS NOT NULL AND TRIM(cs.serial_number) != ''
          AND DATE(cs.bill_date) BETWEEN %(s)s AND %(e)s
        ORDER BY cs.item_code, cs.serial_number
        """
        
        not_in_wh_df = pd.read_sql(not_in_wh_q, conn, params={'s': start_str, 'e': end_str})
        
        if not_in_wh_df is None or not_in_wh_df.empty:
            conn.close()
            return None
        
        # Get all WH received serials grouped by item code
        wh_serials_q = """
        SELECT 
            TRIM(w.item_code) AS item_code,
            TRIM(w.serial_no) AS serial_no
        FROM whreceived_serialno w
        WHERE w.serial_no IS NOT NULL AND TRIM(w.serial_no) != ''
          AND w.grn_date::DATE BETWEEN %(s)s::DATE - INTERVAL '90 days' AND %(e)s
        """
        
        wh_serials_df = pd.read_sql(wh_serials_q, conn, params={'s': start_str, 'e': end_str})
        conn.close()
        
        if wh_serials_df is None or wh_serials_df.empty:
            return None
        
        # Group WH serials by item code
        wh_by_item = {}
        for _, row in wh_serials_df.iterrows():
            item_code = row['item_code']
            serial_no = row['serial_no']
            if item_code not in wh_by_item:
                wh_by_item[item_code] = []
            wh_by_item[item_code].append(serial_no)
        
        # Analyze each "Not in WH" serial against WH sequences
        sequence_misses = []
        
        for _, row in not_in_wh_df.iterrows():
            item_code = row['item_code']
            serial_number = row['serial_number']
            item_desc = row['item_desc'] or 'N/A'
            bill_date = row['bill_date']
            
            # Get WH serials for this item code
            wh_serials = wh_by_item.get(item_code, [])
            if not wh_serials or len(wh_serials) < 2:
                continue
            
            # Analyze sequence with the missing serial included
            all_serials_with_missing = wh_serials + [serial_number]
            
            # Run sequence analysis using alphanumeric-aware function
            analysis = find_missing_serials_in_sequence(
                all_serials_with_missing,
                item_code=item_code,
                item_desc=item_desc
            )
            
            # Check if this serial fits within a gap
            if analysis['has_gaps']:
                for gap in analysis['gaps']:
                    gap_start_serial = gap['between_serials'][0]
                    gap_end_serial = gap['between_serials'][1]
                    missing_count = gap['missing_count']
                    
                    # This "missing" serial could be one of the missing ones
                    sequence_misses.append({
                        'item_code': item_code,
                        'item_desc': item_desc,
                        'missing_serial': serial_number,
                        'gap_start': gap_start_serial,
                        'gap_end': gap_end_serial,
                        'gap_size': gap['gap_size'],
                        'total_missing_in_gap': missing_count,
                        'confidence': gap['confidence'],
                        'sale_date': bill_date,
                        'detection_type': 'sequence_gap'
                    })
        
        return pd.DataFrame(sequence_misses) if sequence_misses else None
        
    except Exception as ex:
        st.error(f"Serial sequence miss detection error: {ex}")
        try: conn.close()
        except: pass
        return None

    except Exception as ex:
        st.error(f"Sequence miss detection error: {ex}")
        try: conn.close()
        except: pass
        return None




@st.cache_data(ttl=300)
def load_slab_trend(slab_start_str: str, slab_end_str: str):
    conn = get_db_connection()
    if not conn:
        return None
    q = f"""
    SELECT
        dt_mod_date::DATE                                                              AS mod_date,
        COUNT(*) FILTER (WHERE (dt_mod_date::DATE - dt_doc_date::DATE) < 0)           AS "Before WH Loaded",
        COUNT(*) FILTER (WHERE (dt_mod_date::DATE - dt_doc_date::DATE) = 0)           AS "0 Days",
        COUNT(*) FILTER (WHERE (dt_mod_date::DATE - dt_doc_date::DATE) BETWEEN 1 AND 3)  AS "1-3 Days",
        COUNT(*) FILTER (WHERE (dt_mod_date::DATE - dt_doc_date::DATE) BETWEEN 4 AND 7)  AS "4-7 Days",
        COUNT(*) FILTER (WHERE (dt_mod_date::DATE - dt_doc_date::DATE) BETWEEN 8 AND 10) AS "8-10 Days",
        COUNT(*) FILTER (WHERE (dt_mod_date::DATE - dt_doc_date::DATE) > 10)           AS ">10 Days",
        COUNT(*)                                                                        AS "Total"
    FROM serial_no_dailydata
    WHERE dt_mod_date IS NOT NULL
      AND dt_doc_date IS NOT NULL
      AND dt_mod_date::DATE BETWEEN '{slab_start_str}' AND '{slab_end_str}'
    GROUP BY dt_mod_date::DATE
    ORDER BY dt_mod_date::DATE DESC
    """
    try:
        df = pd.read_sql(q, conn)
        conn.close()
        return df
    except Exception:
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def load_monthly_offloading_pct(start_str: str, end_str: str):
    """Monthly offloading % by WH loaded month."""
    conn = get_db_connection()
    if not conn:
        return None

    def _run_range_query(conn_obj, range_start: str, range_end: str):
        q = """
        WITH monthly AS (
            SELECT
                DATE_TRUNC('month', dt_doc_date)::DATE AS month_start,
                COUNT(*) AS wh_loaded,
                COUNT(*) FILTER (
                    WHERE vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != ''
                ) AS shop_received
            FROM serial_no_dailydata
            WHERE dt_doc_date IS NOT NULL
              AND dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
            GROUP BY 1
        )
        SELECT
            month_start,
            wh_loaded,
            shop_received,
            ROUND(shop_received::NUMERIC / NULLIF(wh_loaded, 0) * 100, 1) AS offloading_pct
        FROM monthly
        ORDER BY month_start
        """
        return pd.read_sql(q, conn_obj, params={"s": range_start, "e": range_end})

    try:
        end_dt = pd.to_datetime(end_str, errors="coerce")
        if pd.isna(end_dt):
            conn.close()
            return None

        ytd_start = end_dt.replace(month=1, day=1).strftime("%Y-%m-%d")
        ytd_end = end_dt.strftime("%Y-%m-%d")
        df = _run_range_query(conn, ytd_start, ytd_end)

        if df is None or df.empty:
            fallback_q = """
            SELECT MAX(dt_doc_date::DATE) AS max_doc_date
            FROM serial_no_dailydata
            WHERE dt_doc_date IS NOT NULL
              AND dt_doc_date::DATE <= %(e)s
            """
            max_df = pd.read_sql(fallback_q, conn, params={"e": ytd_end})
            max_doc_date = None
            if max_df is not None and not max_df.empty:
                max_doc_date = max_df.iloc[0].get("max_doc_date")

            if pd.notna(max_doc_date):
                max_dt = pd.to_datetime(max_doc_date)
                fb_start = max_dt.replace(month=1, day=1).strftime("%Y-%m-%d")
                fb_end = max_dt.strftime("%Y-%m-%d")
                df = _run_range_query(conn, fb_start, fb_end)

        conn.close()
        return df if df is not None and not df.empty else None
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None


def render_wh_receiving_quality_graphs(end_s: str, compact: bool = False):
    """Render WH Receiving quality trend in a compact panel (tabs)."""
    METRIC_COLORS = {
        "Unique %":       "#00D97E",
        "In-Batch Dup %": "#ef4444",
        "WH DB Dup %":    "#f97316",
        "Blank %":        "#eab308",
        "Small (≤6) %":   "#8b5cf6",
    }
    METRIC_COLS = {
        "unique_pct":      "Unique %",
        "inbatch_dup_pct": "In-Batch Dup %",
        "wh_db_dup_pct":   "WH DB Dup %",
        "blank_pct":       "Blank %",
        "small_pct":       "Small (≤6) %",
    }

    def _build_quality_chart(df, period_label: str, chart_key: str):
        if df is None or df.empty:
            st.info(f"No WH Receiving data for this period.")
            return
        plot_df = df[["period"] + list(METRIC_COLS.keys())].copy()
        plot_df = plot_df.rename(columns=METRIC_COLS)
        plot_df["period"] = pd.to_datetime(plot_df["period"])
        plot_df["x_label"] = (
            plot_df["period"].dt.strftime("%d %b")
            if period_label == "daily"
            else plot_df["period"].dt.strftime("%b %Y")
        )

        if period_label == "daily":
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            x_vals = plot_df["x_label"].tolist()
            unique_min = float(plot_df["Unique %"].min())
            unique_max = float(plot_df["Unique %"].max())

            # Tight zoom around unique% so 99.9 vs 100 is visually distinguishable.
            if unique_max - unique_min < 0.8:
                unique_center = (unique_min + unique_max) / 2
                unique_ymin = max(95.0, unique_center - 0.5)
                unique_ymax = min(100.2, unique_center + 0.5)
            else:
                unique_ymin = max(90.0, unique_min - 0.3)
                unique_ymax = min(100.2, unique_max + 0.3)

            # Keep unique % on primary axis; move all issue metrics to secondary axis.
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=plot_df["Unique %"],
                    mode="lines+markers",
                    connectgaps=True,
                    name="Unique %",
                    line=dict(color=METRIC_COLORS["Unique %"], width=2.4),
                    marker=dict(size=6),
                    hovertemplate="%{x}<br>Unique: %{y:.1f}%<extra></extra>",
                ),
                secondary_y=False,
            )

            for metric in ["In-Batch Dup %", "WH DB Dup %", "Blank %", "Small (≤6) %"]:
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=plot_df[metric],
                        mode="lines+markers",
                        connectgaps=True,
                        name=metric,
                        line=dict(color=METRIC_COLORS[metric], width=2),
                        marker=dict(size=5),
                        hovertemplate=f"%{{x}}<br>{metric}: %{{y:.2f}}%<extra></extra>",
                    ),
                    secondary_y=True,
                )

            fig.update_layout(
                height=235 if compact else 265,
                margin=dict(l=10, r=10, t=2 if compact else 8, b=76 if compact else 86),
                plot_bgcolor="rgba(26,40,71,0.3)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", size=10, color="#FFFFFF"),
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.42 if compact else -0.52,
                    xanchor="center", x=0.5,
                    font=dict(size=10, color="#FFFFFF"), bgcolor="rgba(0,0,0,0)",
                ),
                xaxis=dict(
                    gridcolor="rgba(91,84,255,0.1)",
                    showgrid=False,
                    tickangle=-40,
                    tickfont=dict(size=10, color="#FFFFFF"),
                ),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="rgba(26,40,71,0.95)",
                                font_size=11, font_color="white"),
            )
            fig.update_yaxes(
                title_text="Unique %",
                range=[unique_ymin, unique_ymax],
                ticksuffix="%",
                tickformat=".1f",
                showgrid=True,
                gridcolor="rgba(91,84,255,0.1)",
                tickfont=dict(size=10, color="#FFFFFF"),
                title_font=dict(size=10, color="#FFFFFF"),
                secondary_y=False,
            )
            fig.update_yaxes(

                title_text="Issue %",
                range=[-0.2, max(2.5, float(plot_df[["In-Batch Dup %", "WH DB Dup %", "Blank %", "Small (≤6) %"]].max().max()) + 0.5)],
                ticksuffix="%",
                showgrid=False,
                tickfont=dict(size=10, color="#FFFFFF"),
                title_font=dict(size=10, color="#FFFFFF"),
                secondary_y=True,
            )
        else:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            x_vals = plot_df["x_label"].tolist()
            unique_min = float(plot_df["Unique %"].min())
            unique_max = float(plot_df["Unique %"].max())

            # Tight zoom around unique% so tiny monthly changes are visible.
            if unique_max - unique_min < 0.8:
                unique_center = (unique_min + unique_max) / 2
                unique_ymin = max(95.0, unique_center - 0.5)
                unique_ymax = min(100.2, unique_center + 0.5)
            else:
                unique_ymin = max(90.0, unique_min - 0.3)
                unique_ymax = min(100.2, unique_max + 0.3)

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=plot_df["Unique %"],
                    mode="lines+markers",
                    connectgaps=True,
                    name="Unique %",
                    line=dict(color=METRIC_COLORS["Unique %"], width=2.4),
                    marker=dict(size=6),
                    hovertemplate="%{x}<br>Unique: %{y:.1f}%<extra></extra>",
                ),
                secondary_y=False,
            )

            for metric in ["In-Batch Dup %", "WH DB Dup %", "Blank %", "Small (≤6) %"]:
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=plot_df[metric],
                        mode="lines+markers",
                        connectgaps=True,
                        name=metric,
                        line=dict(color=METRIC_COLORS[metric], width=2),
                        marker=dict(size=5),
                        hovertemplate=f"%{{x}}<br>{metric}: %{{y:.2f}}%<extra></extra>",
                    ),
                    secondary_y=True,
                )

            fig.update_layout(
                height=235 if compact else 265,
                margin=dict(l=10, r=10, t=8, b=76 if compact else 86),
                plot_bgcolor="rgba(26,40,71,0.3)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", size=10, color="#FFFFFF"),
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.42 if compact else -0.52,
                    xanchor="center", x=0.5,
                    font=dict(size=10, color="#FFFFFF"), bgcolor="rgba(0,0,0,0)",
                ),
                xaxis=dict(
                    gridcolor="rgba(91,84,255,0.1)",
                    showgrid=False,
                    tickangle=0,
                    tickfont=dict(size=10, color="#FFFFFF"),
                ),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="rgba(26,40,71,0.95)",
                                font_size=11, font_color="white"),
            )
            fig.update_yaxes(
                title_text="Unique %",
                range=[unique_ymin, unique_ymax],
                ticksuffix="%",
                tickformat=".1f",
                showgrid=True,
                gridcolor="rgba(91,84,255,0.1)",
                tickfont=dict(size=10, color="#FFFFFF"),
                title_font=dict(size=10, color="#FFFFFF"),
                secondary_y=False,
            )
            fig.update_yaxes(
                title_text="Issue %",
                range=[-0.2, max(2.5, float(plot_df[["In-Batch Dup %", "WH DB Dup %", "Blank %", "Small (≤6) %"]].max().max()) + 0.5)],
                ticksuffix="%",
                showgrid=False,
                tickfont=dict(size=10, color="#FFFFFF"),
                title_font=dict(size=10, color="#FFFFFF"),
                secondary_y=True,
            )
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

    if not compact:
        st.markdown(
            "<h3 style='text-align:center;margin:0 0 1px 0;'>🏭 WH Received Quality</h3>",
            unsafe_allow_html=True,
        )
        st.caption("Unique % → 100% (green) · In-Batch Dup / WH DB Dup / Blank / Small (≤6) → 0%")

    tab_daily, tab_monthly = st.tabs([
        "🏭 WH Received Quality — Last 30 Days",
        "🏭 WH Received Quality — Monthly (YTD)",
    ])

    with tab_daily:
        _build_quality_chart(
            load_wh_receiving_quality_trend(end_s, "daily"),
            "daily",
            f"wh_recv_quality_daily_{end_s.replace('-', '')}",
        )

    with tab_monthly:
        _build_quality_chart(
            load_wh_receiving_quality_trend(end_s, "monthly"),
            "monthly",
            f"wh_recv_quality_monthly_{end_s.replace('-', '')}",
        )


@st.cache_data(ttl=300)
def load_wh_loading_monthly(end_str: str):
    """Monthly WH Loading quality %: 100% when blank=0, small=0, IC=0."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        end_dt = pd.to_datetime(end_str, errors="coerce")
        if pd.isna(end_dt):
            conn.close()
            return None
        ytd_start = end_dt.replace(month=1, day=1).strftime("%Y-%m-%d")
        q = """
        WITH base AS (
            SELECT
                DATE_TRUNC('month', dt_doc_date)::DATE AS period,
                serial_no,
                vc_item_code::TEXT AS item_code
            FROM serial_no_dailydata
            WHERE dt_doc_date IS NOT NULL
              AND dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
        ),
        agg AS (
            SELECT
                period,
                COUNT(*) AS total,
                COUNT(*) FILTER (
                    WHERE serial_no IS NULL OR LENGTH(TRIM(COALESCE(serial_no,''))) = 0
                ) AS blank_cnt,
                COUNT(*) FILTER (
                    WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
                      AND LENGTH(TRIM(serial_no)) <= 6
                ) AS small_cnt,
                COUNT(*) FILTER (
                    WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
                      AND REGEXP_REPLACE(TRIM(serial_no), '[^A-Za-z0-9]','','g') != ''
                      AND REGEXP_REPLACE(TRIM(serial_no), '[^A-Za-z0-9]','','g') =
                          REGEXP_REPLACE(TRIM(item_code), '[^A-Za-z0-9]','','g')
                ) AS ic_cnt
            FROM base
            GROUP BY period
        )
        SELECT
            period,
            total,
            ROUND(GREATEST(0.0,
                100.0
                - ROUND(blank_cnt::NUMERIC / NULLIF(total,0) * 100, 1)
                - ROUND(small_cnt::NUMERIC / NULLIF(total,0) * 100, 1)
                - ROUND(ic_cnt::NUMERIC    / NULLIF(total,0) * 100, 1)
            ), 1) AS quality_pct
        FROM agg
        ORDER BY period
        """
        df = pd.read_sql(q, conn, params={"s": ytd_start, "e": end_str})
        conn.close()
        return df if df is not None and not df.empty else None
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None


@st.cache_data(ttl=300)
def load_shop_compliance_monthly(end_str: str):
    """Monthly Shop Compliance % YTD."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        end_dt = pd.to_datetime(end_str, errors="coerce")
        if pd.isna(end_dt):
            conn.close()
            return None
        ytd_start = end_dt.replace(month=1, day=1).strftime("%Y-%m-%d")
        q = """
        SELECT
            DATE_TRUNC('month', DATE(bill_date))::DATE AS period,
            COUNT(*) AS total,
            ROUND(
                COUNT(*) FILTER (WHERE TRIM(serial_check) = 'Y')::NUMERIC
                / NULLIF(COUNT(*), 0) * 100, 1
            ) AS compliance_pct
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
          AND COALESCE(item_code,'') != 'sales data not available'
        GROUP BY 1
        ORDER BY 1
        """
        df = pd.read_sql(q, conn, params={"s": ytd_start, "e": end_str})
        conn.close()
        return df if df is not None and not df.empty else None
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None


def render_month_on_month_combined_chart(end_s: str):
    """Grouped bar chart: WH Receiving %, WH Loading %, Shop Compliance % — month on month."""
    wh_recv_df   = load_wh_receiving_quality_trend(end_s, "monthly")
    wh_load_df   = load_wh_loading_monthly(end_s)
    shop_comp_df = load_shop_compliance_monthly(end_s)

    has_data = any(
        df is not None and not (hasattr(df, "empty") and df.empty)
        for df in [wh_recv_df, wh_load_df, shop_comp_df]
    )
    if not has_data:
        st.info("No month-on-month data available.")
        return

    # ── Build a unified month index ────────────────────────────────────────────
    all_vals = []
    traces = []

    def _prep(df, col, label, color):
        if df is None or df.empty:
            return None
        d = df.copy()
        d["period"]  = pd.to_datetime(d["period"])
        d["x_label"] = d["period"].dt.strftime("%b %Y")
        d[col] = pd.to_numeric(d[col], errors="coerce")
        all_vals.extend(d[col].dropna().tolist())
        return dict(x=d["x_label"].tolist(), y=d[col].tolist(), label=label, color=color)

    t_recv = _prep(wh_recv_df,   "unique_pct",    "WH Receiving %",   "#00D97E")
    t_load = _prep(wh_load_df,   "quality_pct",   "WH Loading %",     "#38bdf8")
    t_comp = _prep(shop_comp_df, "compliance_pct", "Shop Compliance %", "#9b5bff")

    for t in [t_recv, t_load, t_comp]:
        if t:
            traces.append(t)

    fig = go.Figure()
    for t in traces:
        fig.add_trace(go.Bar(
            x=t["x"],
            y=t["y"],
            name=t["label"],
            marker_color=t["color"],
            text=[f"{v:.1f}%" if pd.notna(v) else "" for v in t["y"]],
            textposition="outside",
            textfont=dict(size=9, color="#FFFFFF"),
            hovertemplate=f"%{{x}}<br>{t['label']}: %{{y:.1f}}%<extra></extra>",
        ))

    # ── Zoom Y axis to actual data range so differences are visible ────────────
    if all_vals:
        y_min = max(0.0,  min(all_vals) - 3.0)
        y_max = min(105.0, max(all_vals) + 5.0)
    else:
        y_min, y_max = 0, 105

    fig.add_hline(
        y=100, line_dash="dash", line_color="#f59e0b", line_width=1.5,
        annotation_text="100%", annotation_position="top right",
        annotation_font=dict(color="#f59e0b", size=9),
    )

    fig.update_layout(
        barmode="group",
        bargap=0.22,
        bargroupgap=0.06,
        height=310,
        margin=dict(l=10, r=10, t=16, b=70),
        plot_bgcolor="rgba(26,40,71,0.3)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=10, color="#FFFFFF"),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.40,
            xanchor="center", x=0.5,
            font=dict(size=10, color="#FFFFFF"), bgcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            title="Quality %",
            range=[y_min, y_max],
            ticksuffix="%",
            showgrid=True,
            gridcolor="rgba(91,84,255,0.12)",
            tickfont=dict(size=10, color="#FFFFFF"),
            title_font=dict(size=10, color="#FFFFFF"),
            dtick=max(0.5, round((y_max - y_min) / 8, 1)),
        ),
        xaxis=dict(
            showgrid=False,
            tickangle=-30,
            tickfont=dict(size=10, color="#FFFFFF"),
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(26,40,71,0.95)", font_size=11, font_color="white"),
    )
    # Not full-width: put the chart in a centred column so it doesn't stretch
    _, chart_col, _ = st.columns([0.5, 3, 0.5])
    with chart_col:
        st.plotly_chart(fig, use_container_width=True, key=f"mom_combined_{end_s.replace('-','')}")


def render_monthly_offloading_chart(start_s: str, end_s: str):
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    # ── Combined month-on-month quality chart (not full width) ─────────────────
    st.markdown(
        "<div style='text-align:center;font-size:1.05em;font-weight:700;"
        "color:#FFFFFF;margin-bottom:2px;'>"
        "📊 Month-on-Month Quality — WH Receiving / WH Loading / Shop Compliance</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "WH Receiving %: 100% = zero in-batch dup / WH DB dup / blank  "
        "· WH Loading %: 100% = zero blank, small, IC-equal serials  "
        "· Shop Compliance %: serial_check Y ÷ total sold · Y-axis zoomed to show differences"
    )
    render_month_on_month_combined_chart(end_s)

    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

    # ── Existing 3 detail charts (aligned to top) ─────────────────────────────
    quality_col, compliance_col, offload_col = st.columns([1, 1, 1], gap="small")

    # Shared fixed-height header block so all 3 columns start at the same level
    TITLE_H  = "height:36px;overflow:hidden;display:flex;align-items:center;justify-content:center;"
    SUB_H    = "height:30px;overflow:hidden;display:flex;align-items:center;justify-content:center;"
    TITLE_CSS = f"text-align:center;font-size:1.0em;font-weight:700;color:#FFFFFF;margin-bottom:0px;{TITLE_H}"
    SUB_CSS   = f"text-align:center;font-size:0.78em;color:#94a3b8;{SUB_H}"

    with quality_col:
        st.markdown(
            f"<div style='{TITLE_CSS}'>🏭 WH Received Quality</div>",
            unsafe_allow_html=True,
        )
        render_wh_receiving_quality_graphs(end_s, compact=True)

    with compliance_col:
        st.markdown(
            f"<div style='{TITLE_CSS}'>📈 Shop Compliance % — Last 10 Days</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='{SUB_CSS}'>Trend: strict last 10 days ending selected date</div>",
            unsafe_allow_html=True,
        )
        render_shop_compliance_last10days_chart(start_s, end_s)

    with offload_col:
        st.markdown(
            f"<div style='{TITLE_CSS}'>📈 Monthly Offloading % (WH Loaded vs Shop Received)</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='{SUB_CSS}'>Rows with shop serial ÷ WH loaded rows (by DOC date month) × 100</div>",
            unsafe_allow_html=True,
        )

        monthly_df = load_monthly_offloading_pct(start_s, end_s)
        if monthly_df is None or monthly_df.empty:
            st.info("No monthly offloading data for selected year-to-date range.")
        else:
            monthly_df['month_start'] = pd.to_datetime(monthly_df['month_start'])
            monthly_df['month_label'] = monthly_df['month_start'].dt.strftime('%b %Y')

            fig = px.bar(monthly_df, x='month_label', y='offloading_pct', text='offloading_pct')
            fig.update_traces(
                marker=dict(color="#00D97E", line=dict(color="rgba(255,255,255,0.7)", width=1.2)),
                texttemplate='%{text:.1f}%',
                textposition='outside',
                textfont=dict(size=11, color='#FFFFFF'),
                customdata=monthly_df[['wh_loaded', 'shop_received']].values,
                hovertemplate=(
                    "<b>%{x}</b><br>Offloading %: %{y:.1f}%"
                    "<br>WH Loaded: %{customdata[0]}"
                    "<br>Shop Received (shop serial present): %{customdata[1]}<extra></extra>"
                )
            )
            fig.add_hline(
                y=100,
                line_width=2,
                line_dash="dash",
                line_color="#f59e0b",
                annotation_text="Benchmark 100%",
                annotation_position="top left"
            )
            fig.update_layout(
                height=235,
                yaxis_title="Offloading %",
                xaxis_title="Month",
                margin=dict(l=12, r=12, t=10, b=28),
                plot_bgcolor="rgba(26,40,71,0.3)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", size=11, color="#FFFFFF"),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="rgba(26,40,71,0.95)", font_size=12,
                                font_family="Inter", font_color="white"),
                xaxis=dict(gridcolor="rgba(91,84,255,0.1)", showgrid=True),
                yaxis=dict(autorange=True, gridcolor="rgba(91,84,255,0.1)", showgrid=True)
            )
            st.plotly_chart(fig, use_container_width=True, key="monthly_offloading_pct")


@st.cache_data(ttl=300)
def load_wh_loading_drilldown(start_str: str, end_str: str, metric: str):
    """Drilldown: WH Loading by loader + daily breakdown for user-selected date range."""
    conn = get_db_connection()
    if not conn:
        return None
    # ✅ OPTIMIZED: Using DISTINCT ON instead of ROW_NUMBER() window function
    q = f"""
    WITH base AS (
        SELECT
            dt_doc_date::DATE AS doc_date,
            COALESCE(NULLIF(TRIM(wh_load_user),''),'Unknown') AS loader,
            serial_no,
            vc_item_code::TEXT AS item_code,
            REGEXP_REPLACE(TRIM(vc_item_code::TEXT), '[^A-Za-z0-9]','','g') AS item_code_clean
        FROM serial_no_dailydata
        WHERE dt_doc_date::DATE BETWEEN '{start_str}'::DATE AND '{end_str}'::DATE
    ),
    serial_item AS (
        SELECT
            doc_date,
            loader,
            serial_no,
            COUNT(DISTINCT item_code_clean) AS item_code_count
        FROM base
        WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
        GROUP BY doc_date, loader, serial_no
    ),
    daily_data AS (
        SELECT
            b.doc_date,
            b.loader,
            COUNT(*) AS total_records,
            COUNT(DISTINCT b.serial_no) FILTER (
                WHERE b.serial_no IS NOT NULL AND LENGTH(TRIM(b.serial_no)) > 0
            ) AS unique_serials,
            COUNT(*) FILTER (
                WHERE b.serial_no IS NOT NULL AND LENGTH(TRIM(b.serial_no)) > 0
            ) AS loaded_with_serial,
            COUNT(*) FILTER (
                WHERE LENGTH(TRIM(COALESCE(b.serial_no,''))) <= 6
                  AND b.serial_no IS NOT NULL AND LENGTH(TRIM(b.serial_no)) > 0
            ) AS small_serials,
            COUNT(DISTINCT s.serial_no) FILTER (WHERE s.item_code_count > 1) AS duplicates,
            COUNT(*) FILTER (
                WHERE b.serial_no IS NULL OR LENGTH(TRIM(b.serial_no)) = 0
            ) AS blank_count,
            COUNT(*) FILTER (
                WHERE b.serial_no IS NOT NULL AND LENGTH(TRIM(b.serial_no)) > 0
                  AND REGEXP_REPLACE(TRIM(b.serial_no), '[^A-Za-z0-9]','','g') != ''
                  AND REGEXP_REPLACE(TRIM(b.serial_no), '[^A-Za-z0-9]','','g') =
                      REGEXP_REPLACE(TRIM(b.item_code), '[^A-Za-z0-9]','','g')
            ) AS ic_count
        FROM base b
        LEFT JOIN serial_item s
            ON s.doc_date = b.doc_date
           AND s.loader = b.loader
           AND s.serial_no = b.serial_no
        GROUP BY b.doc_date, b.loader
    )
    SELECT
        loader,
        doc_date,
        CASE
            WHEN '{metric}' = 'Unique %' THEN ROUND(unique_serials::NUMERIC / NULLIF(total_records, 0) * 100, 1)::TEXT || '%'
            WHEN '{metric}' = 'Duplicates' THEN duplicates::TEXT
            WHEN '{metric}' = 'Blank Serials' THEN blank_count::TEXT
            WHEN '{metric}' = 'Small (≤6)' THEN small_serials::TEXT
            WHEN '{metric}' = 'IC' THEN ic_count::TEXT
            ELSE total_records::TEXT
        END AS metric_value
    FROM daily_data
    WHERE doc_date BETWEEN '{start_str}'::DATE AND '{end_str}'::DATE
    ORDER BY loader, doc_date DESC
    """
    try:
        df = pd.read_sql(q, conn)
        conn.close()
        if df is not None and not df.empty:
            pivot = df.pivot_table(
                index=['loader'],
                columns='doc_date',
                values='metric_value',
                aggfunc='first'
            )
            pivot = pivot[sorted(pivot.columns, reverse=True)]
            pivot.columns = [d.strftime('%Y-%m-%d') if hasattr(d,'strftime') else str(d) for d in pivot.columns]
            pivot = pivot.reset_index()
            return pivot
        return None
    except Exception as ex:
        st.error(f"WH Loading drilldown error: {ex}")
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def load_wh_loading_shop_drilldown(start_str: str, end_str: str, metric: str):
    """WH Loading drilldown: shop-code wise for user-selected date range."""
    conn = get_db_connection()
    if not conn:
        return None

    q = f"""
    WITH base AS (
        SELECT
            dt_doc_date::DATE AS doc_date,
            COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
            serial_no,
            vc_item_code::TEXT AS item_code,
            REGEXP_REPLACE(TRIM(vc_item_code::TEXT), '[^A-Za-z0-9]','','g') AS item_code_clean
        FROM serial_no_dailydata
        WHERE dt_doc_date::DATE BETWEEN '{start_str}'::DATE AND '{end_str}'::DATE
    ),
    serial_item AS (
        SELECT
            doc_date,
            shop_code,
            serial_no,
            COUNT(DISTINCT item_code_clean) AS item_code_count
        FROM base
        WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
        GROUP BY doc_date, shop_code, serial_no
    ),
    daily_data AS (
        SELECT
            b.doc_date,
            b.shop_code,
            COUNT(*) AS total_records,
            COUNT(DISTINCT b.serial_no) FILTER (
                WHERE b.serial_no IS NOT NULL AND LENGTH(TRIM(b.serial_no)) > 0
            ) AS unique_serials,
            COUNT(*) FILTER (
                WHERE b.serial_no IS NOT NULL AND LENGTH(TRIM(b.serial_no)) > 0
            ) AS loaded_with_serial,
            COUNT(*) FILTER (
                WHERE b.serial_no IS NULL OR LENGTH(TRIM(COALESCE(b.serial_no,''))) = 0
            ) AS blank_count,
            COUNT(*) FILTER (
                WHERE b.serial_no IS NOT NULL AND LENGTH(TRIM(COALESCE(b.serial_no,''))) <= 6
            ) AS small_serials,
            COUNT(DISTINCT s.serial_no) FILTER (WHERE s.item_code_count > 1) AS duplicates,
            COUNT(*) FILTER (
                WHERE b.serial_no IS NOT NULL AND LENGTH(TRIM(b.serial_no)) > 0
                  AND REGEXP_REPLACE(TRIM(b.serial_no), '[^A-Za-z0-9]','','g') != ''
                  AND REGEXP_REPLACE(TRIM(b.serial_no), '[^A-Za-z0-9]','','g') =
                      REGEXP_REPLACE(TRIM(b.item_code), '[^A-Za-z0-9]','','g')
            ) AS ic_count
        FROM base b
        LEFT JOIN serial_item s
            ON s.doc_date = b.doc_date
           AND s.shop_code = b.shop_code
           AND s.serial_no = b.serial_no
        GROUP BY b.doc_date, b.shop_code
    )
    SELECT
        shop_code,
        doc_date,
        CASE
            WHEN '{metric}' = 'Unique %' THEN ROUND(unique_serials::NUMERIC / NULLIF(total_records, 0) * 100, 1)::TEXT || '%'
            WHEN '{metric}' = 'Duplicates' THEN duplicates::TEXT
            WHEN '{metric}' = 'Blank Serials' THEN blank_count::TEXT
            WHEN '{metric}' = 'Small (≤6)' THEN small_serials::TEXT
            WHEN '{metric}' = 'IC' THEN ic_count::TEXT
            ELSE total_records::TEXT
        END AS metric_value
    FROM daily_data
    ORDER BY shop_code, doc_date DESC
    """
    try:
        df = pd.read_sql(q, conn)
        conn.close()
        if df is not None and not df.empty:
            pivot = df.pivot_table(
                index=['shop_code'],
                columns='doc_date',
                values='metric_value',
                aggfunc='first'
            )
            pivot = pivot[sorted(pivot.columns, reverse=True)]
            pivot.columns = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in pivot.columns]
            pivot = pivot.reset_index()
            return pivot
        return None
    except Exception as ex:
        st.error(f"WH Loading shop-wise drilldown error: {ex}")
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def load_wh_loading_monthly_summary(start_str: str, end_str: str, metric: str = "Duplicates"):
    """WH Loading monthly summary for Duplicates, Small, or IC."""
    conn = get_db_connection()
    if not conn:
        return None
    
    # Build the metric-specific logic
    if metric == "Duplicates":
        metric_logic = """
            COUNT(DISTINCT s.serial_no) FILTER (WHERE s.item_code_count > 1) AS metric_value
        """
    elif metric == "Small (≤6)":
        metric_logic = """
            COUNT(*) FILTER (
                WHERE b.serial_no IS NOT NULL 
                  AND LENGTH(TRIM(b.serial_no)) > 0
                  AND LENGTH(TRIM(b.serial_no)) <= 6
            ) AS metric_value
        """
    elif metric == "IC":
        metric_logic = """
            COUNT(*) FILTER (
                WHERE b.serial_no IS NOT NULL 
                  AND LENGTH(TRIM(b.serial_no)) > 0
                  AND REGEXP_REPLACE(TRIM(b.serial_no), '[^A-Za-z0-9]','','g') != ''
                  AND REGEXP_REPLACE(TRIM(b.serial_no), '[^A-Za-z0-9]','','g') =
                      REGEXP_REPLACE(TRIM(b.item_code), '[^A-Za-z0-9]','','g')
            ) AS metric_value
        """
    else:
        metric_logic = "COUNT(*) AS metric_value"
    
    q = f"""
    WITH base AS (
        SELECT
            TO_CHAR(dt_doc_date, 'YYYY-MM') AS month,
            serial_no,
            vc_item_code::TEXT AS item_code,
            REGEXP_REPLACE(TRIM(vc_item_code::TEXT), '[^A-Za-z0-9]','','g') AS item_code_clean
        FROM serial_no_dailydata
    ),
    serial_item AS (
        SELECT
            month,
            serial_no,
            COUNT(DISTINCT item_code_clean) AS item_code_count
        FROM base
        WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
        GROUP BY month, serial_no
    )
    SELECT
        b.month,
        {metric_logic}
    FROM base b
    LEFT JOIN serial_item s 
        ON s.month = b.month 
       AND s.serial_no = b.serial_no
    GROUP BY b.month
    ORDER BY b.month DESC
    """
    
    try:
        df = pd.read_sql(q, conn)
        conn.close()
        if df is not None and not df.empty:
            return df
        return None
    except Exception as ex:
        st.error(f"WH Loading monthly summary error: {ex}")
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def load_shop_receiving_drilldown(start_str: str, end_str: str, metric: str):
    """Drilldown: Shop Receiving by shop + daily breakdown."""
    conn = get_db_connection()
    if not conn:
        return None
    params = None
    if metric == "Old Doc Date":
        q = """
        WITH daily_data AS (
            SELECT
                dt_mod_date::DATE AS mod_date,
                COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
                COALESCE(shop_name, 'Unknown Shop') AS shop_name,
                COUNT(*) AS old_doc_date
            FROM serial_no_dailydata
            WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
              AND dt_mod_date IS NOT NULL
              AND dt_mod_date::DATE NOT BETWEEN %(s)s AND %(e)s
            GROUP BY dt_mod_date::DATE, vc_shop_code, shop_name
        )
        SELECT
            shop_code,
            shop_name,
            mod_date,
            old_doc_date::TEXT AS metric_value
        FROM daily_data
        ORDER BY shop_code, mod_date DESC
        """
        params = {'s': start_str, 'e': end_str}
    else:
        q = f"""
        WITH daily_data AS (
            SELECT
                (dt_mod_date::DATE) AS mod_date,
                COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
                COALESCE(shop_name, 'Unknown Shop') AS shop_name,
                COUNT(*) AS total_offloaded,
                COUNT(*) FILTER (WHERE (dt_mod_date::DATE - dt_doc_date::DATE) = 0) AS same_day,
                COUNT(DISTINCT serial_no) FILTER (WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0) AS unique_serials,
                COUNT(*) FILTER (WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0) AS with_serial,
                COUNT(*) FILTER (WHERE serial_no IS NULL OR LENGTH(TRIM(serial_no)) = 0) AS no_serial,
                COUNT(*) FILTER (
                    WHERE vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != ''
                      AND (serial_no IS NULL OR TRIM(COALESCE(serial_no,'')) = ''
                           OR TRIM(vc_serail_no) != TRIM(serial_no))
                ) AS mismatch_count,
                COUNT(*) FILTER (
                    WHERE vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != ''
                      AND REGEXP_REPLACE(TRIM(vc_serail_no), '[^A-Za-z0-9]','','g') != ''
                      AND REGEXP_REPLACE(TRIM(vc_serail_no), '[^A-Za-z0-9]','','g') =
                          REGEXP_REPLACE(TRIM(vc_item_code::TEXT), '[^A-Za-z0-9]','','g')
                ) AS ic_count
                                ,COUNT(*) FILTER (
                                        WHERE vc_serail_no IS NOT NULL
                                            AND TRIM(COALESCE(vc_serail_no,'')) != ''
                                            AND LENGTH(TRIM(vc_serail_no)) <= 6
                                ) AS small_count
            FROM serial_no_dailydata
            WHERE dt_mod_date IS NOT NULL
              AND dt_mod_date::DATE BETWEEN ('{end_str}'::DATE - 6) AND '{end_str}'::DATE
            GROUP BY dt_mod_date::DATE, vc_shop_code, shop_name
        )
        SELECT
            shop_code,
            shop_name,
            mod_date,
            CASE
                WHEN '{metric}' = 'Total Offloaded' THEN total_offloaded::TEXT
                WHEN '{metric}' = 'Same-Day' THEN same_day::TEXT
                WHEN '{metric}' = 'Unique %' THEN ROUND(unique_serials::NUMERIC / NULLIF(total_offloaded, 0) * 100, 1)::TEXT || '%'
                WHEN '{metric}' = 'Not Offloaded' THEN (total_offloaded - with_serial)::TEXT
                WHEN '{metric}' = 'WH→Shop Mismatch' THEN mismatch_count::TEXT
                WHEN '{metric}' = 'Small (≤6)' THEN small_count::TEXT
                WHEN '{metric}' = 'IC' THEN ic_count::TEXT
                ELSE total_offloaded::TEXT
            END AS metric_value
        FROM daily_data
        ORDER BY shop_code, mod_date DESC
        """
    try:
        df = pd.read_sql(q, conn, params=params) if params else pd.read_sql(q, conn)
        conn.close()
        
        if df is not None and not df.empty:
            pivot = df.pivot_table(
                index=['shop_code', 'shop_name'],
                columns='mod_date',
                values='metric_value',
                aggfunc='first'
            )
            pivot = pivot[sorted(pivot.columns, reverse=True)]
            pivot.columns = [d.strftime('%Y-%m-%d') if hasattr(d,'strftime') else str(d) for d in pivot.columns]
            pivot = pivot.reset_index()
            return pivot
        return None
    except Exception as ex:
        st.error(f"Shop Receiving drilldown error: {ex}")
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def load_shop_receiving_old_doc_detail(start_str: str, end_str: str, shop_code: str, mod_date_str: str):
    """Detail rows for Shop Receiving -> Old Doc Date selected shop/date cell."""
    conn = get_db_connection()
    if not conn:
        return None

    q = """
    SELECT
        COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
        COALESCE(shop_name, 'Unknown Shop') AS shop_name,
        dt_doc_date::DATE AS wh_loading_date,
        dt_mod_date::DATE AS shop_offloading_date,
        serial_no AS wh_loading_serial_no,
        vc_serail_no AS shop_offloading_serial_no,
        vc_item_code AS item_code,
        vc_item_desc AS item_name,
        vc_vehicle_no AS wh_truck_no,
        "VC_VEHICLE_NO_1" AS shop_truck_no,
        COALESCE(NULLIF(TRIM(wh_load_user),''),'Unknown') AS wh_loader
    FROM serial_no_dailydata
    WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
      AND dt_mod_date IS NOT NULL
      AND dt_mod_date::DATE NOT BETWEEN %(s)s AND %(e)s
      AND dt_mod_date::DATE = %(d)s
      AND COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') = %(shop)s
    ORDER BY dt_mod_date::DATE DESC, dt_doc_date::DATE DESC, vc_item_code, vc_serail_no
    """

    try:
        df = pd.read_sql(q, conn, params={'s': start_str, 'e': end_str, 'd': mod_date_str, 'shop': shop_code})
        conn.close()
        return df if df is not None and not df.empty else None
    except Exception as ex:
        st.error(f"Old Doc Date detail error: {ex}")
        try:
            conn.close()
        except Exception:
            pass
        return None


@st.cache_data(ttl=300)
def load_shop_selling_drilldown(start_str: str, end_str: str, metric: str):
    """Drilldown: Shop Compliance % by shop + daily breakdown."""
    conn = get_db_connection()
    if not conn:
        return None
    
    q = f"""
    WITH daily_data AS (
        SELECT
            (dt_invoice_date::DATE) AS invoice_date,
            COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
            COALESCE(shop_name, 'Unknown Shop') AS shop_name,
            COUNT(*) AS total_sold,
            COUNT(*) FILTER (WHERE yn_compliance = 'Y') AS compliant_sold,
            COUNT(*) FILTER (WHERE yn_compliance NOT IN ('Y', 'y')) AS non_compliant_sold,
            COUNT(*) FILTER (WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0) AS with_serial,
            COUNT(*) FILTER (WHERE serial_no IS NULL OR LENGTH(TRIM(serial_no)) = 0) AS no_serial,
            COUNT(DISTINCT serial_no) FILTER (WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0) - COUNT(*) FILTER (WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0) + COUNT(DISTINCT serial_no) FILTER (WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0) AS dup_serials
        FROM serialno_check_yes_no
        WHERE dt_invoice_date IS NOT NULL
          AND dt_invoice_date::DATE BETWEEN ('{end_str}'::DATE - 6) AND '{end_str}'::DATE
        GROUP BY dt_invoice_date::DATE, vc_shop_code, shop_name
    )
    SELECT
        shop_code,
        shop_name,
        invoice_date,
        CASE
            WHEN '{metric}' = 'Compliance %' THEN ROUND(compliant_sold::NUMERIC / NULLIF(total_sold, 0) * 100, 1)::TEXT || '%'
            WHEN '{metric}' = 'Dup Serials' THEN dup_serials::TEXT
            WHEN '{metric}' = 'No Serial' THEN no_serial::TEXT
            WHEN '{metric}' = 'Not in WH' THEN (total_sold - with_serial)::TEXT
            WHEN '{metric}' = 'Serial = IC' THEN '0'
            WHEN '{metric}' = 'Small (≤6)' THEN '0'
            ELSE total_sold::TEXT
        END AS metric_value
    FROM daily_data
    ORDER BY shop_code, invoice_date DESC
    """
    try:
        df = pd.read_sql(q, conn)
        conn.close()
        
        if df is not None and not df.empty:
            pivot = df.pivot_table(
                index=['shop_code', 'shop_name'],
                columns='invoice_date',
                values='metric_value',
                aggfunc='first'
            )
            pivot = pivot[sorted(pivot.columns, reverse=True)]
            pivot.columns = [d.strftime('%Y-%m-%d') if hasattr(d,'strftime') else str(d) for d in pivot.columns]
            pivot = pivot.reset_index()
            return pivot
        return None
    except Exception as ex:
        st.error(f"Shop Compliance % drilldown error: {ex}")
        try: conn.close()
        except: pass
        return None


# ────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
    if not conn:
        return None
    if metric == 'Not Offloaded':
        q = """
        SELECT
            COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
            shop_name,
            COUNT(*) FILTER (WHERE dt_mod_date IS NULL) AS not_offloaded,
            COUNT(*) AS total_items
        FROM serial_no_dailydata WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
        GROUP BY vc_shop_code, shop_name HAVING COUNT(*) FILTER (WHERE dt_mod_date IS NULL) > 0
        ORDER BY not_offloaded DESC
        """
    elif metric == 'Total Offloaded':
        q = """
        SELECT
            COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
            shop_name,
            COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL) AS offloaded,
            COUNT(*) AS total_items
        FROM serial_no_dailydata WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
        GROUP BY vc_shop_code, shop_name ORDER BY offloaded DESC
        """
    elif metric == 'Same-Day':
        q = """
        SELECT
            COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
            shop_name,
            COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL AND dt_doc_date IS NOT NULL AND dt_mod_date::DATE = dt_doc_date::DATE) AS same_day,
            COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL) AS total_offloaded
        FROM serial_no_dailydata WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
        GROUP BY vc_shop_code, shop_name ORDER BY same_day DESC
        """
    elif metric == 'Old Doc Date':
        q = """
        SELECT
            COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
            shop_name,
            COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL AND dt_mod_date::DATE NOT BETWEEN %(s)s AND %(e)s) AS old_doc,
            COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL) AS total_offloaded
        FROM serial_no_dailydata WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
        GROUP BY vc_shop_code, shop_name HAVING COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL AND dt_mod_date::DATE NOT BETWEEN %(s)s AND %(e)s) > 0
        ORDER BY old_doc DESC
        """
    elif metric == 'WH→Shop Mismatch':
        q = """
        SELECT
            COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
            shop_name,
            COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL AND vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != '' AND (serial_no IS NULL OR TRIM(COALESCE(serial_no,'')) = '' OR TRIM(vc_serail_no) != TRIM(serial_no))) AS mismatch,
            COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL) AS total_offloaded
        FROM serial_no_dailydata WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
        GROUP BY vc_shop_code, shop_name HAVING COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL AND vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != '') > 0
        ORDER BY mismatch DESC
        """
    elif metric == 'Unique %':
        q = """
        SELECT
            COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
            shop_name,
            COUNT(DISTINCT vc_serail_no) FILTER (WHERE dt_mod_date IS NOT NULL AND vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != '') AS unique_serials,
            COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL) AS total_offloaded,
            ROUND(COUNT(DISTINCT vc_serail_no)::NUMERIC / NULLIF(COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL),0) * 100, 1) AS unique_pct
        FROM serial_no_dailydata WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
        GROUP BY vc_shop_code, shop_name ORDER BY unique_pct DESC
        """
    elif metric == 'Blank':
        q = """
        SELECT
            COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
            shop_name,
            COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL AND (vc_serail_no IS NULL OR TRIM(COALESCE(vc_serail_no,'')) = '')) AS blank_serials,
            COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL) AS total_offloaded
        FROM serial_no_dailydata WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
        GROUP BY vc_shop_code, shop_name HAVING COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL AND (vc_serail_no IS NULL OR TRIM(COALESCE(vc_serail_no,'')) = '')) > 0
        ORDER BY blank_serials DESC
        """
    elif metric == 'Small (≤6)':
        q = """
        SELECT
            COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
            shop_name,
            COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL AND vc_serail_no IS NOT NULL AND LENGTH(TRIM(vc_serail_no)) <= 6) AS small_serials,
            COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL) AS total_offloaded
        FROM serial_no_dailydata WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
        GROUP BY vc_shop_code, shop_name HAVING COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL AND vc_serail_no IS NOT NULL AND LENGTH(TRIM(vc_serail_no)) <= 6) > 0
        ORDER BY small_serials DESC
        """
    elif metric == 'Duplicates':
        q = """
        SELECT
            COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
            shop_name,
            COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL AND vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != '') - COUNT(DISTINCT vc_serail_no) FILTER (WHERE dt_mod_date IS NOT NULL AND vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != '') AS duplicate_count,
            COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL) AS total_offloaded
        FROM serial_no_dailydata WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
        GROUP BY vc_shop_code, shop_name HAVING COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL AND vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != '') - COUNT(DISTINCT vc_serail_no) FILTER (WHERE dt_mod_date IS NOT NULL AND vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != '') > 0
        ORDER BY duplicate_count DESC
        """
    else:
        return None
    try:
        df = pd.read_sql(q, conn, params={'s': start_str, 'e': end_str})
        conn.close()
        return df if not df.empty else None
    except:
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def load_shop_selling_drilldown(start_str: str, end_str: str, metric: str):
    """Drilldown: Shop Compliance % by shop & cashier for selected metric."""
    conn = get_db_connection()
    if not conn:
        return None

    metric_expr = "COUNT(*)::TEXT"
    extra_cte = ""
    joins = ""

    if metric == "Compliance %":
        metric_expr = "ROUND(COUNT(*) FILTER (WHERE TRIM(b.serial_check) = 'Y')::NUMERIC / NULLIF(COUNT(*),0) * 100, 1)::TEXT || '%'"
    elif metric == "Not in WH":
        metric_expr = "COUNT(*) FILTER (WHERE TRIM(b.serial_check) = 'N' AND b.serial_number IS NOT NULL AND TRIM(COALESCE(b.serial_number,'')) != '')::TEXT"
    elif metric == "Dup Serials":
        metric_expr = "(COUNT(*) FILTER (WHERE b.serial_number IS NOT NULL AND TRIM(COALESCE(b.serial_number,'')) != '') - COUNT(DISTINCT b.serial_number) FILTER (WHERE b.serial_number IS NOT NULL AND TRIM(COALESCE(b.serial_number,'')) != ''))::TEXT"
    elif metric == "No Serial":
        metric_expr = "COUNT(*) FILTER (WHERE b.serial_number IS NULL OR TRIM(COALESCE(b.serial_number,'')) = '')::TEXT"
    elif metric == "Serial = IC":
        metric_expr = "COUNT(*) FILTER (WHERE b.serial_number IS NOT NULL AND TRIM(COALESCE(b.serial_number,'')) != '' AND REGEXP_REPLACE(TRIM(b.serial_number), '[^A-Za-z0-9]','','g') != '' AND REGEXP_REPLACE(TRIM(b.serial_number), '[^A-Za-z0-9]','','g') = REGEXP_REPLACE(TRIM(b.item_code), '[^A-Za-z0-9]','','g'))::TEXT"
    elif metric == "Check Serial No":
        extra_cte = """
        , y_same_item AS (
            SELECT DISTINCT
                DATE(bill_date) AS bill_date,
                TRIM(COALESCE(item_code,'')) AS item_code_norm
            FROM serialno_check_yes_no
            WHERE TRIM(serial_check) = 'Y'
              AND COALESCE(item_code,'') != ''
        ),
        y_any_item AS (
            SELECT DISTINCT
                TRIM(COALESCE(item_code,'')) AS item_code_norm
            FROM serialno_check_yes_no
            WHERE TRIM(serial_check) = 'Y'
              AND COALESCE(item_code,'') != ''
        )
        """
        joins = """
        LEFT JOIN y_same_item ys
          ON ys.bill_date = b.bill_date
         AND ys.item_code_norm = TRIM(COALESCE(b.item_code,''))
        LEFT JOIN y_any_item ya
          ON ya.item_code_norm = TRIM(COALESCE(b.item_code,''))
        """
        metric_expr = "COUNT(*) FILTER (WHERE TRIM(b.serial_check) = 'N' AND b.serial_number IS NOT NULL AND TRIM(COALESCE(b.serial_number,'')) != '' AND (ys.item_code_norm IS NOT NULL OR ya.item_code_norm IS NOT NULL))::TEXT"
    elif metric == "Small (≤6)":
        metric_expr = "COUNT(*) FILTER (WHERE b.serial_number IS NOT NULL AND LENGTH(TRIM(COALESCE(b.serial_number,''))) > 0 AND LENGTH(TRIM(COALESCE(b.serial_number,''))) <= 6)::TEXT"

    q = f"""
    WITH base AS (
        SELECT
            DATE(bill_date) AS bill_date,
            COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
            COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown') AS cashier,
            serial_check,
            serial_number,
            item_code
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
          AND COALESCE(item_code,'') != 'sales data not available'
    )
    {extra_cte}
    SELECT
        b.shop_code,
        b.cashier,
        b.bill_date,
        {metric_expr} AS metric_value
    FROM base b
    {joins}
    GROUP BY b.shop_code, b.cashier, b.bill_date
    ORDER BY b.shop_code, b.cashier, b.bill_date DESC
    """
    try:
        df = pd.read_sql(q, conn, params={'s': start_str, 'e': end_str})
        conn.close()
        if df is not None and not df.empty:
            pivot = df.pivot_table(
                index=['shop_code', 'cashier'],
                columns='bill_date',
                values='metric_value',
                aggfunc='first'
            )
            pivot = pivot[sorted(pivot.columns, reverse=True)]
            pivot.columns = [d.strftime('%Y-%m-%d') if hasattr(d,'strftime') else str(d) for d in pivot.columns]
            pivot = pivot.reset_index()
            return pivot
        return None
    except:
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def load_shop_selling_shop_metric(metric: str, start_str: str, end_str: str):
    """Shop-level metric for last 7 days."""
    conn = get_db_connection()
    if not conn:
        return None

    if metric == "Compliance %":
        q_compliance = """
        WITH base AS (
            SELECT
                COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
                TRIM(COALESCE(serial_check,'')) AS serial_check
            FROM serialno_check_yes_no
            WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
              AND COALESCE(item_code,'') != 'sales data not available'
        )
        SELECT
            shop_code,
            COUNT(*) AS total_serial_numbers,
            COUNT(*) FILTER (WHERE serial_check = 'Y') AS serial_match_y,
            ROUND(COUNT(*) FILTER (WHERE serial_check = 'Y')::NUMERIC / NULLIF(COUNT(*),0) * 100, 1)::TEXT || CHR(37) AS compliance_pct
        FROM base
        GROUP BY shop_code
        ORDER BY shop_code
        """
        try:
            df = pd.read_sql(q_compliance, conn, params={'s': start_str, 'e': end_str})
            conn.close()
            return df if df is not None and not df.empty else None
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            return None

    if metric == "Dup Serials":
        q_dup = """
        WITH base AS (
            SELECT
                DATE(bill_date) AS bill_date,
                COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
                TRIM(serial_number) AS serial_number
            FROM serialno_check_yes_no
            WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
              AND COALESCE(item_code,'') != 'sales data not available'
              AND serial_number IS NOT NULL
              AND TRIM(COALESCE(serial_number,'')) != ''
        ),
        dup_sn AS (
            SELECT serial_number
            FROM base
            GROUP BY serial_number
            HAVING COUNT(*) > 1
        )
        SELECT
            b.shop_code,
            COUNT(*) FILTER (
                WHERE b.serial_number IN (SELECT serial_number FROM dup_sn)
            ) AS metric_value
        FROM base b
        GROUP BY b.shop_code
        HAVING COUNT(*) FILTER (
            WHERE b.serial_number IN (SELECT serial_number FROM dup_sn)
        ) > 0
        ORDER BY metric_value DESC, b.shop_code
        """
        try:
            df = pd.read_sql(q_dup, conn, params={'s': start_str, 'e': end_str})
            conn.close()
            if df is None or df.empty:
                return None
            df = df.rename(columns={'metric_value': 'dup_serials'})
            return df
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            return None

    metric_case = "COUNT(*)"
    if metric == "Compliance %":
        metric_case = "ROUND(COUNT(*) FILTER (WHERE TRIM(serial_check) = 'Y')::NUMERIC / NULLIF(COUNT(*),0) * 100, 1)"
    elif metric == "Dup Serials":
        metric_case = "COUNT(*) FILTER (WHERE (bill_date, shop_code, serial_number) IN (SELECT bill_date, shop_code, serial_number FROM dup_sn))"
    elif metric == "No Serial":
        metric_case = "COUNT(*) FILTER (WHERE serial_number IS NULL OR TRIM(COALESCE(serial_number,'')) = '')"
    elif metric == "Not in WH":
        metric_case = "COUNT(*) FILTER (WHERE TRIM(serial_check) = 'N' AND serial_number IS NOT NULL AND TRIM(COALESCE(serial_number,'')) != '')"
    elif metric == "Serial = IC":
        metric_case = "COUNT(*) FILTER (WHERE serial_number IS NOT NULL AND TRIM(COALESCE(serial_number,'')) != '' AND REGEXP_REPLACE(TRIM(serial_number), '[^A-Za-z0-9]','','g') = REGEXP_REPLACE(TRIM(item_code), '[^A-Za-z0-9]','','g'))"
    elif metric == "Check Serial No":
        metric_case = """COUNT(*) FILTER (
            WHERE TRIM(serial_check) = 'N'
              AND serial_number IS NOT NULL
              AND TRIM(COALESCE(serial_number,'')) != ''
              AND (
                  EXISTS (
                      SELECT 1 FROM serialno_check_yes_no y_same
                      WHERE TRIM(y_same.serial_check) = 'Y'
                        AND TRIM(COALESCE(y_same.item_code,'')) = TRIM(COALESCE(item_code,''))
                        AND DATE(y_same.bill_date) = bill_date
                  )
                  OR EXISTS (
                      SELECT 1 FROM serialno_check_yes_no y_any
                      WHERE TRIM(y_any.serial_check) = 'Y'
                        AND TRIM(COALESCE(y_any.item_code,'')) = TRIM(COALESCE(item_code,''))
                  )
              )
        )"""
    elif metric == "Small (≤6)":
        metric_case = "COUNT(*) FILTER (WHERE serial_number IS NOT NULL AND LENGTH(TRIM(COALESCE(serial_number,''))) <= 6)"

    q = f"""
    WITH base AS (
                SELECT
                        DATE(bill_date) AS bill_date,
                        COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
                        serial_number,
                        serial_check,
                        item_code
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
          AND COALESCE(item_code,'') != 'sales data not available'
    ),
    dup_sn AS (
        SELECT bill_date, shop_code, serial_number FROM base
        WHERE serial_number IS NOT NULL AND TRIM(serial_number) != ''
        GROUP BY bill_date, shop_code, serial_number HAVING COUNT(*) > 1
    )
    SELECT
        bill_date,
        shop_code,
        {metric_case} AS metric_value
    FROM base
    GROUP BY bill_date, shop_code
    ORDER BY shop_code, bill_date DESC
    """
    try:
        df = pd.read_sql(q, conn, params={'s': start_str, 'e': end_str})
        conn.close()
        if df is None or df.empty:
            return None

        pivot = df.pivot_table(
            index=['shop_code'],
            columns='bill_date',
            values='metric_value',
            aggfunc='first'
        )

        full_dates = pd.date_range(start=start_str, end=end_str, freq='D')
        pivot = pivot.reindex(columns=full_dates, fill_value=pd.NA)
        pivot = pivot[sorted(pivot.columns, reverse=True)]
        pivot.columns = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in pivot.columns]
        pivot = pivot.reset_index()

        date_cols = [c for c in pivot.columns if c != 'shop_code']
        if date_cols:
            sort_col = date_cols[0]
            sort_series = pd.to_numeric(
                pivot[sort_col].astype(str).str.replace('%', '', regex=False),
                errors='coerce'
            )
            pivot = (
                pivot.assign(_is_blank=sort_series.isna(), _sort=sort_series.fillna(float('inf')))
                .sort_values(['_is_blank', '_sort', 'shop_code'], ascending=[True, True, True])
                .drop(columns=['_is_blank', '_sort'])
            )

        if metric == "Compliance %":
            for col in date_cols:
                pivot[col] = pivot[col].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "")

        return pivot
    except Exception:
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def load_shop_selling_cashier_metric(shop_code: str, metric: str, start_str: str, end_str: str):
    """Cashier metric by day for selected shop (last 7 days)."""
    conn = get_db_connection()
    if not conn:
        return None

    if metric == "Dup Serials":
        q_dup = """
        WITH base AS (
            SELECT
                DATE(bill_date) AS bill_date,
                COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
                COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown') AS cashier,
                TRIM(serial_number) AS serial_number
            FROM serialno_check_yes_no
            WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
              AND COALESCE(item_code,'') != 'sales data not available'
              AND serial_number IS NOT NULL
              AND TRIM(COALESCE(serial_number,'')) != ''
        ),
        scope_shop AS (
            SELECT *
            FROM base
            WHERE shop_code = %(shop)s
        ),
        dup_sn AS (
            SELECT serial_number
            FROM base
            GROUP BY serial_number
            HAVING COUNT(*) > 1
        )
        SELECT
            s.cashier,
            COUNT(*) FILTER (
                WHERE s.serial_number IN (SELECT serial_number FROM dup_sn)
            ) AS metric_value
        FROM scope_shop s
        GROUP BY s.cashier
        HAVING COUNT(*) FILTER (
            WHERE s.serial_number IN (SELECT serial_number FROM dup_sn)
        ) > 0
        ORDER BY metric_value DESC, s.cashier
        """
        try:
            df = pd.read_sql(q_dup, conn, params={'s': start_str, 'e': end_str, 'shop': shop_code})
            conn.close()
            if df is None or df.empty:
                return None
            df = df.rename(columns={'metric_value': 'dup_serials'})
            return df
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            return None

    metric_case = "COUNT(*)"
    if metric == "Compliance %":
        metric_case = "ROUND(COUNT(*) FILTER (WHERE TRIM(serial_check) = 'Y')::NUMERIC / NULLIF(COUNT(*),0) * 100, 1)"
    elif metric == "Dup Serials":
        metric_case = "COUNT(*) FILTER (WHERE (bill_date, serial_number) IN (SELECT bill_date, serial_number FROM dup_sn))"
    elif metric == "No Serial":
        metric_case = "COUNT(*) FILTER (WHERE serial_number IS NULL OR TRIM(COALESCE(serial_number,'')) = '')"
    elif metric == "Not in WH":
        metric_case = "COUNT(*) FILTER (WHERE TRIM(serial_check) = 'N' AND serial_number IS NOT NULL AND TRIM(COALESCE(serial_number,'')) != '')"
    elif metric == "Serial = IC":
        metric_case = "COUNT(*) FILTER (WHERE serial_number IS NOT NULL AND TRIM(COALESCE(serial_number,'')) != '' AND REGEXP_REPLACE(TRIM(serial_number), '[^A-Za-z0-9]','','g') = REGEXP_REPLACE(TRIM(item_code), '[^A-Za-z0-9]','','g'))"
    elif metric == "Check Serial No":
        metric_case = """COUNT(*) FILTER (
            WHERE TRIM(serial_check) = 'N'
              AND serial_number IS NOT NULL
              AND TRIM(COALESCE(serial_number,'')) != ''
              AND (
                  EXISTS (
                      SELECT 1 FROM serialno_check_yes_no y_same
                      WHERE TRIM(y_same.serial_check) = 'Y'
                        AND TRIM(COALESCE(y_same.item_code,'')) = TRIM(COALESCE(item_code,''))
                        AND DATE(y_same.bill_date) = bill_date
                  )
                  OR EXISTS (
                      SELECT 1 FROM serialno_check_yes_no y_any
                      WHERE TRIM(y_any.serial_check) = 'Y'
                        AND TRIM(COALESCE(y_any.item_code,'')) = TRIM(COALESCE(item_code,''))
                  )
              )
        )"""
    elif metric == "Small (≤6)":
        metric_case = "COUNT(*) FILTER (WHERE serial_number IS NOT NULL AND LENGTH(TRIM(COALESCE(serial_number,''))) <= 6)"

    q = f"""
    WITH base AS (
                SELECT
                        DATE(bill_date) AS bill_date,
                        COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown') AS cashier,
                        serial_number,
                        serial_check,
                        item_code
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
          AND COALESCE(item_code,'') != 'sales data not available'
          AND COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') = %(shop)s
    ),
    dup_sn AS (
        SELECT bill_date, serial_number FROM base
        WHERE serial_number IS NOT NULL AND TRIM(serial_number) != ''
        GROUP BY bill_date, serial_number HAVING COUNT(*) > 1
    )
    SELECT
        bill_date,
        cashier,
        {metric_case} AS metric_value
    FROM base
    GROUP BY bill_date, cashier
    ORDER BY cashier, bill_date DESC
    """
    try:
        df = pd.read_sql(q, conn, params={'s': start_str, 'e': end_str, 'shop': shop_code})
        conn.close()
        if df is None or df.empty:
            return None
        pivot = df.pivot_table(
            index=['cashier'],
            columns='bill_date',
            values='metric_value',
            aggfunc='first'
        )

        full_dates = pd.date_range(start=start_str, end=end_str, freq='D')
        pivot = pivot.reindex(columns=full_dates, fill_value=pd.NA)
        pivot = pivot[sorted(pivot.columns, reverse=True)]
        pivot.columns = [d.strftime('%Y-%m-%d') if hasattr(d,'strftime') else str(d) for d in pivot.columns]
        pivot = pivot.reset_index()

        date_cols = [c for c in pivot.columns if c != 'cashier']
        if date_cols:
            sort_col = date_cols[0]
            sort_series = pd.to_numeric(
                pivot[sort_col].astype(str).str.replace('%', '', regex=False),
                errors='coerce'
            )
            pivot = (
                pivot.assign(_is_blank=sort_series.isna(), _sort=sort_series.fillna(float('inf')))
                .sort_values(['_is_blank', '_sort', 'cashier'], ascending=[True, True, True])
                .drop(columns=['_is_blank', '_sort'])
            )

        if metric == "Compliance %":
            for col in date_cols:
                pivot[col] = pivot[col].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "")
        return pivot
    except Exception:
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def load_shop_selling_trend(metric: str, start_str: str, end_str: str):
    """Trend: daily totals for Shop Compliance % metrics."""
    conn = get_db_connection()
    if not conn:
        return None

    q = """
    WITH base AS (
        SELECT
            DATE(bill_date) AS bill_date,
            serial_check,
            serial_number,
            item_code
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
          AND COALESCE(item_code,'') != 'sales data not available'
    ),
    dup_sn AS (
        SELECT serial_number
        FROM base
        WHERE serial_number IS NOT NULL AND TRIM(serial_number) != ''
        GROUP BY serial_number
        HAVING COUNT(*) > 1
    )
    SELECT
        bill_date,
        COUNT(*) AS total_sold,
        ROUND(COUNT(*) FILTER (WHERE TRIM(serial_check) = 'Y')::NUMERIC / NULLIF(COUNT(*),0) * 100, 1) AS compliance_pct,
        COUNT(*) FILTER (
            WHERE TRIM(serial_check) = 'N'
              AND serial_number IS NOT NULL
              AND TRIM(COALESCE(serial_number,'')) != ''
        ) AS not_in_wh,
        COUNT(*) FILTER (
            WHERE serial_number IS NULL OR TRIM(COALESCE(serial_number,'')) = ''
        ) AS no_serial,
        COUNT(*) FILTER (
            WHERE serial_number IN (SELECT serial_number FROM dup_sn)
        ) AS dup_serials,
        COUNT(*) FILTER (
            WHERE serial_number IS NOT NULL
              AND LENGTH(TRIM(COALESCE(serial_number,''))) > 0
              AND LENGTH(TRIM(COALESCE(serial_number,''))) <= 6
        ) AS small_serial,
        COUNT(*) FILTER (
            WHERE serial_number IS NOT NULL
              AND TRIM(COALESCE(serial_number,'')) != ''
              AND REGEXP_REPLACE(TRIM(serial_number), '[^A-Za-z0-9]','','g') != ''
              AND REGEXP_REPLACE(TRIM(serial_number), '[^A-Za-z0-9]','','g') = REGEXP_REPLACE(TRIM(item_code), '[^A-Za-z0-9]','','g')
                ) AS serial_is_ic,
                COUNT(*) FILTER (
                        WHERE TRIM(serial_check) = 'N'
                            AND serial_number IS NOT NULL
                            AND TRIM(COALESCE(serial_number,'')) != ''
                            AND (
                                    EXISTS (
                                            SELECT 1
                                            FROM serialno_check_yes_no y_same
                                            WHERE TRIM(y_same.serial_check) = 'Y'
                                                AND TRIM(COALESCE(y_same.item_code,'')) = TRIM(COALESCE(base.item_code,''))
                                                AND DATE(y_same.bill_date) = base.bill_date
                                    )
                                    OR EXISTS (
                                            SELECT 1
                                            FROM serialno_check_yes_no y_any
                                            WHERE TRIM(y_any.serial_check) = 'Y'
                                                AND TRIM(COALESCE(y_any.item_code,'')) = TRIM(COALESCE(base.item_code,''))
                                    )
                            )
                ) AS incorrect_serial
    FROM base
    GROUP BY bill_date
    ORDER BY bill_date
    """
    try:
        df = pd.read_sql(q, conn, params={'s': start_str, 'e': end_str})
        conn.close()
        if df is None or df.empty:
            return None

        metric_col = "total_sold"
        if metric == "Compliance %":
            metric_col = "compliance_pct"
        elif metric == "Dup Serials":
            metric_col = "dup_serials"
        elif metric == "No Serial":
            metric_col = "no_serial"
        elif metric == "Not in WH":
            metric_col = "not_in_wh"
        elif metric == "Serial = IC":
            metric_col = "serial_is_ic"
        elif metric == "Check Serial No":
            metric_col = "incorrect_serial"
        elif metric == "Small (≤6)":
            metric_col = "small_serial"

        return df[["bill_date", metric_col]].rename(columns={"bill_date": "date", metric_col: "total"})
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None


@st.cache_data(ttl=300)
def load_shop_selling_metric_detail(shop_code: str, cashier: str, date_str: str, metric: str):
    """Serial-level detail for a cashier/date and metric."""
    conn = get_db_connection()
    if not conn:
        return None

    safe_shop = str(shop_code).strip() if shop_code is not None else ""
    safe_cashier = str(cashier).strip() if cashier is not None else ""
    safe_date = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(safe_date) or safe_cashier == "":
        try:
            conn.close()
        except Exception:
            pass
        return None
    safe_date = safe_date.strftime("%Y-%m-%d")

    metric_filter = ""
    if metric == "Dup Serials":
        metric_filter = "AND b.serial_number IN (SELECT serial_number FROM dup_sn)"
    elif metric == "No Serial":
        metric_filter = "AND (b.serial_number IS NULL OR TRIM(COALESCE(b.serial_number,'')) = '')"
    elif metric == "Not in WH":
        metric_filter = "AND TRIM(b.serial_check) = 'N' AND b.serial_number IS NOT NULL AND TRIM(COALESCE(b.serial_number,'')) != ''"
    elif metric == "Check Serial No":
        metric_filter = """AND TRIM(b.serial_check) = 'N'
            AND b.serial_number IS NOT NULL
            AND TRIM(COALESCE(b.serial_number,'')) != ''
            AND (y_same.serial_number IS NOT NULL OR y_near.serial_number IS NOT NULL)"""
    elif metric == "Serial = IC":
        metric_filter = "AND b.serial_number IS NOT NULL AND TRIM(COALESCE(b.serial_number,'')) != '' AND REGEXP_REPLACE(TRIM(b.serial_number), '[^A-Za-z0-9]','','g') = REGEXP_REPLACE(TRIM(b.item_code), '[^A-Za-z0-9]','','g')"
    elif metric == "Small (≤6)":
        metric_filter = "AND b.serial_number IS NOT NULL AND LENGTH(TRIM(COALESCE(b.serial_number,''))) <= 6"

    q = f"""
    WITH src AS (
        SELECT
            serial_check,
            serial_number,
            item_code,
            item_name,
            bill_date,
            COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
            COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown') AS cashier
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) = %(d)s
          AND COALESCE(item_code,'') != 'sales data not available'
    ),
    scope_shop_day AS (
        SELECT *
        FROM src
        WHERE COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') = %(shop)s
    ),
    base AS (
        SELECT *
        FROM scope_shop_day
        WHERE COALESCE(NULLIF(TRIM(cashier),''),'Unknown') = %(cashier)s
    ),
    dup_sn AS (
        SELECT serial_number FROM src
        WHERE serial_number IS NOT NULL AND TRIM(serial_number) != ''
        GROUP BY serial_number HAVING COUNT(*) > 1
    )
    SELECT
        TRIM(b.serial_check) AS serial_check,
        b.serial_number,
        b.item_code,
        COALESCE(NULLIF(TRIM(b.item_name),''), 'Unknown') AS item_desc,
        b.bill_date,
        b.shop_code,
        b.cashier AS cashier_name,
        COALESCE(y_same.serial_number, y_near.serial_number) AS corrected_serial_number,
        COALESCE(y_same.cashier_name, y_near.cashier_name) AS corrected_by_cashier,
        COALESCE(y_same.bill_date, y_near.bill_date) AS corrected_from_date,
        CASE
            WHEN y_same.serial_number IS NOT NULL THEN 'Same Date Y'
            WHEN y_near.serial_number IS NOT NULL THEN 'Nearest Date Y'
            ELSE NULL
        END AS correction_source
    FROM base b
    LEFT JOIN LATERAL (
        SELECT
            TRIM(y.serial_number) AS serial_number,
            COALESCE(NULLIF(TRIM(y.cashier_name),''),'Unknown') AS cashier_name,
            DATE(y.bill_date) AS bill_date
        FROM serialno_check_yes_no y
        WHERE TRIM(y.serial_check) = 'Y'
          AND TRIM(COALESCE(y.item_code,'')) = TRIM(COALESCE(b.item_code,''))
          AND DATE(y.bill_date) = DATE(b.bill_date)
          AND y.serial_number IS NOT NULL
          AND TRIM(COALESCE(y.serial_number,'')) != ''
        ORDER BY DATE(y.bill_date) DESC
        LIMIT 1
    ) y_same ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            TRIM(y.serial_number) AS serial_number,
            COALESCE(NULLIF(TRIM(y.cashier_name),''),'Unknown') AS cashier_name,
            DATE(y.bill_date) AS bill_date
        FROM serialno_check_yes_no y
        WHERE TRIM(y.serial_check) = 'Y'
          AND TRIM(COALESCE(y.item_code,'')) = TRIM(COALESCE(b.item_code,''))
          AND y.serial_number IS NOT NULL
          AND TRIM(COALESCE(y.serial_number,'')) != ''
        ORDER BY ABS(DATE(y.bill_date) - DATE(b.bill_date)), DATE(y.bill_date) DESC
        LIMIT 1
    ) y_near ON y_same.serial_number IS NULL
    WHERE 1=1
        {metric_filter}
    ORDER BY b.bill_date DESC, b.serial_check
    """

    fallback_q = f"""
    WITH src AS (
        SELECT
            serial_check,
            serial_number,
            item_code,
            item_name,
            bill_date,
            COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
            COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown') AS cashier
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) = %(d)s
          AND COALESCE(item_code,'') != 'sales data not available'
    ),
    base AS (
        SELECT *
        FROM src
        WHERE COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') = %(shop)s
          AND COALESCE(NULLIF(TRIM(cashier),''),'Unknown') = %(cashier)s
    ),
    dup_sn AS (
        SELECT serial_number FROM src
        WHERE serial_number IS NOT NULL AND TRIM(serial_number) != ''
        GROUP BY serial_number HAVING COUNT(*) > 1
    )
    SELECT
        TRIM(b.serial_check) AS serial_check,
        b.serial_number,
        b.item_code,
        COALESCE(NULLIF(TRIM(b.item_name),''), 'Unknown') AS item_desc,
        b.bill_date,
        b.shop_code,
        b.cashier AS cashier_name,
        COALESCE(y_same.serial_number, y_near.serial_number) AS corrected_serial_number,
        COALESCE(y_same.cashier_name, y_near.cashier_name) AS corrected_by_cashier,
        COALESCE(y_same.bill_date, y_near.bill_date) AS corrected_from_date,
        CASE
            WHEN y_same.serial_number IS NOT NULL THEN 'Same Date Y'
            WHEN y_near.serial_number IS NOT NULL THEN 'Nearest Date Y'
            ELSE NULL
        END AS correction_source
    FROM base b
    LEFT JOIN LATERAL (
        SELECT
            TRIM(y.serial_number) AS serial_number,
            COALESCE(NULLIF(TRIM(y.cashier_name),''),'Unknown') AS cashier_name,
            DATE(y.bill_date) AS bill_date
        FROM serialno_check_yes_no y
        WHERE TRIM(y.serial_check) = 'Y'
          AND TRIM(COALESCE(y.item_code,'')) = TRIM(COALESCE(b.item_code,''))
          AND DATE(y.bill_date) = DATE(b.bill_date)
          AND y.serial_number IS NOT NULL
          AND TRIM(COALESCE(y.serial_number,'')) != ''
        ORDER BY DATE(y.bill_date) DESC
        LIMIT 1
    ) y_same ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            TRIM(y.serial_number) AS serial_number,
            COALESCE(NULLIF(TRIM(y.cashier_name),''),'Unknown') AS cashier_name,
            DATE(y.bill_date) AS bill_date
        FROM serialno_check_yes_no y
        WHERE TRIM(y.serial_check) = 'Y'
          AND TRIM(COALESCE(y.item_code,'')) = TRIM(COALESCE(b.item_code,''))
          AND y.serial_number IS NOT NULL
          AND TRIM(COALESCE(y.serial_number,'')) != ''
        ORDER BY ABS(DATE(y.bill_date) - DATE(b.bill_date)), DATE(y.bill_date) DESC
        LIMIT 1
    ) y_near ON y_same.serial_number IS NULL
    WHERE 1=1
        {metric_filter}
    ORDER BY b.bill_date DESC, b.serial_check
    """
    try:
        params = {'d': safe_date, 'shop': safe_shop, 'cashier': safe_cashier}

        df = pd.read_sql(q, conn, params=params)
        if (df is None or df.empty) and metric == "Compliance %":
            df = pd.read_sql(fallback_q, conn, params={'d': safe_date, 'shop': safe_shop, 'cashier': safe_cashier})
        conn.close()
        return df if df is not None and not df.empty else None
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None


@st.cache_data(ttl=300)
def load_shop_selling_metric_detail_range(shop_code: str, cashier: str, start_str: str, end_str: str, metric: str):
    """Serial-level detail for a cashier over selected date range."""
    conn = get_db_connection()
    if not conn:
        return None

    safe_shop = str(shop_code).strip() if shop_code is not None else ""
    safe_cashier = str(cashier).strip() if cashier is not None else ""
    if safe_cashier == "":
        try:
            conn.close()
        except Exception:
            pass
        return None

    metric_filter = ""
    if metric == "Dup Serials":
        metric_filter = "AND b.serial_number IN (SELECT serial_number FROM dup_sn)"

    q = f"""
    WITH src AS (
        SELECT
            serial_check,
            serial_number,
            item_code,
            item_name,
            bill_date,
            COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
            COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown') AS cashier
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
          AND COALESCE(item_code,'') != 'sales data not available'
    ),
    scope_shop AS (
        SELECT *
        FROM src
        WHERE COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') = %(shop)s
    ),
    base AS (
        SELECT *
        FROM scope_shop
        WHERE COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') = %(shop)s
          AND COALESCE(NULLIF(TRIM(cashier),''),'Unknown') = %(cashier)s
    ),
    dup_sn AS (
        SELECT serial_number
        FROM src
        WHERE serial_number IS NOT NULL AND TRIM(serial_number) != ''
        GROUP BY serial_number
        HAVING COUNT(*) > 1
    )
    SELECT
        TRIM(b.serial_check) AS serial_check,
        b.serial_number,
        b.item_code,
        COALESCE(NULLIF(TRIM(b.item_name),''), 'Unknown') AS item_desc,
        DATE(b.bill_date) AS bill_date,
        b.shop_code,
        b.cashier AS cashier_name
    FROM base b
    WHERE 1=1
      {metric_filter}
    ORDER BY b.bill_date DESC, b.serial_check
    """

    try:
        params = {'s': start_str, 'e': end_str, 'shop': safe_shop, 'cashier': safe_cashier}
        df = pd.read_sql(q, conn, params=params)
        conn.close()
        return df if df is not None and not df.empty else None
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None

    safe_shop = str(shop_code).strip() if shop_code is not None else ""
    safe_cashier = str(cashier).strip() if cashier is not None else ""
    safe_date = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(safe_date) or safe_cashier == "":
        try:
            conn.close()
        except Exception:
            pass
        return None
    safe_date = safe_date.strftime("%Y-%m-%d")

    metric_filter = ""
    if metric == "Compliance %":
        metric_filter = ""
    elif metric == "Dup Serials":
        metric_filter = "AND b.serial_number IN (SELECT serial_number FROM dup_sn)"
    elif metric == "No Serial":
        metric_filter = "AND (b.serial_number IS NULL OR TRIM(COALESCE(b.serial_number,'')) = '')"
    elif metric == "Not in WH":
        metric_filter = "AND TRIM(b.serial_check) = 'N' AND b.serial_number IS NOT NULL AND TRIM(COALESCE(b.serial_number,'')) != ''"
    elif metric == "Serial = IC":
        metric_filter = "AND b.serial_number IS NOT NULL AND TRIM(COALESCE(b.serial_number,'')) != '' AND REGEXP_REPLACE(TRIM(b.serial_number), '[^A-Za-z0-9]','','g') = REGEXP_REPLACE(TRIM(b.item_code), '[^A-Za-z0-9]','','g')"
    elif metric == "Check Serial No":
        metric_filter = """AND TRIM(b.serial_check) = 'N'
            AND b.serial_number IS NOT NULL
            AND TRIM(COALESCE(b.serial_number,'')) != ''
            AND (y_same.serial_number IS NOT NULL OR y_near.serial_number IS NOT NULL)"""
    elif metric == "Small (≤6)":
        metric_filter = "AND b.serial_number IS NOT NULL AND LENGTH(TRIM(COALESCE(b.serial_number,''))) <= 6"

    q = f"""
    WITH src AS (
        SELECT
            serial_check,
            serial_number,
            item_code,
            item_name,
            bill_date,
            COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
            COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown') AS cashier
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) = %(d)s
          AND COALESCE(item_code,'') != 'sales data not available'
    ),
    scope_shop_day AS (
        SELECT *
        FROM src
        WHERE COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') = %(shop)s
    ),
    base AS (
        SELECT *
        FROM scope_shop_day
        WHERE COALESCE(NULLIF(TRIM(cashier),''),'Unknown') = %(cashier)s
    ),
    dup_sn AS (
        SELECT serial_number FROM src
        WHERE serial_number IS NOT NULL AND TRIM(serial_number) != ''
        GROUP BY serial_number
        HAVING COUNT(*) > 1
    )
    SELECT
        TRIM(b.serial_check) AS serial_check,
        b.serial_number,
        b.item_code,
        COALESCE(NULLIF(TRIM(b.item_name),''), 'Unknown') AS item_desc,
        DATE(b.bill_date) AS bill_date,
        b.shop_code,
        b.cashier AS cashier_name
    FROM base b
    WHERE 1=1
      {metric_filter}
    ORDER BY b.bill_date DESC, b.serial_check
    """

    try:
        params = {'s': start_str, 'e': end_str, 'shop': safe_shop, 'cashier': safe_cashier}
        df = pd.read_sql(q, conn, params=params)
        conn.close()
        return df if df is not None and not df.empty else None
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None

    safe_shop = str(shop_code).strip() if shop_code is not None else ""
    safe_cashier = str(cashier).strip() if cashier is not None else ""
    safe_date = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(safe_date) or safe_cashier == "":
        try:
            conn.close()
        except Exception:
            pass
        return None
    safe_date = safe_date.strftime("%Y-%m-%d")

    metric_filter = ""
    if metric == "Compliance %":
        metric_filter = ""
    elif metric == "Dup Serials":
        metric_filter = "AND b.serial_number IN (SELECT serial_number FROM dup_sn)"
    elif metric == "No Serial":
        metric_filter = "AND (b.serial_number IS NULL OR TRIM(COALESCE(b.serial_number,'')) = '')"
    elif metric == "Not in WH":
        metric_filter = "AND TRIM(b.serial_check) = 'N' AND b.serial_number IS NOT NULL AND TRIM(COALESCE(b.serial_number,'')) != ''"
    elif metric == "Check Serial No":
        metric_filter = """AND TRIM(b.serial_check) = 'N'
            AND b.serial_number IS NOT NULL
            AND TRIM(COALESCE(b.serial_number,'')) != ''
            AND (y_same.serial_number IS NOT NULL OR y_near.serial_number IS NOT NULL)"""
    elif metric == "Serial = IC":
        metric_filter = "AND b.serial_number IS NOT NULL AND TRIM(COALESCE(b.serial_number,'')) != '' AND REGEXP_REPLACE(TRIM(b.serial_number), '[^A-Za-z0-9]','','g') = REGEXP_REPLACE(TRIM(b.item_code), '[^A-Za-z0-9]','','g')"
    elif metric == "Small (≤6)":
        metric_filter = "AND b.serial_number IS NOT NULL AND LENGTH(TRIM(COALESCE(b.serial_number,''))) <= 6"

    q = f"""
    WITH src AS (
        SELECT
            serial_check,
            serial_number,
            item_code,
            item_name,
            bill_date,
            COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
            COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown') AS cashier
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) = %(d)s
          AND COALESCE(item_code,'') != 'sales data not available'
    ),
    scope_shop_day AS (
        SELECT *
        FROM src
        WHERE COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') = %(shop)s
    ),
    base AS (
        SELECT *
        FROM scope_shop_day
        WHERE COALESCE(NULLIF(TRIM(cashier),''),'Unknown') = %(cashier)s
    ),
    dup_sn AS (
        SELECT serial_number FROM scope_shop_day
        WHERE serial_number IS NOT NULL AND TRIM(serial_number) != ''
        GROUP BY serial_number HAVING COUNT(*) > 1
    )
    SELECT
        TRIM(b.serial_check) AS serial_check,
        b.serial_number,
        b.item_code,
        COALESCE(NULLIF(TRIM(b.item_name),''), 'Unknown') AS item_desc,
        b.bill_date,
        b.shop_code,
        b.cashier AS cashier_name,
        COALESCE(y_same.serial_number, y_near.serial_number) AS corrected_serial_number,
        COALESCE(y_same.cashier_name, y_near.cashier_name) AS corrected_by_cashier,
        COALESCE(y_same.bill_date, y_near.bill_date) AS corrected_from_date,
        CASE
            WHEN y_same.serial_number IS NOT NULL THEN 'Same Date Y'
            WHEN y_near.serial_number IS NOT NULL THEN 'Nearest Date Y'
            ELSE NULL
        END AS correction_source
    FROM base b
    LEFT JOIN LATERAL (
        SELECT
            TRIM(y.serial_number) AS serial_number,
            COALESCE(NULLIF(TRIM(y.cashier_name),''),'Unknown') AS cashier_name,
            DATE(y.bill_date) AS bill_date
        FROM serialno_check_yes_no y
        WHERE TRIM(y.serial_check) = 'Y'
          AND TRIM(COALESCE(y.item_code,'')) = TRIM(COALESCE(b.item_code,''))
          AND DATE(y.bill_date) = DATE(b.bill_date)
          AND y.serial_number IS NOT NULL
          AND TRIM(COALESCE(y.serial_number,'')) != ''
        ORDER BY DATE(y.bill_date) DESC
        LIMIT 1
    ) y_same ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            TRIM(y.serial_number) AS serial_number,
            COALESCE(NULLIF(TRIM(y.cashier_name),''),'Unknown') AS cashier_name,
            DATE(y.bill_date) AS bill_date
        FROM serialno_check_yes_no y
        WHERE TRIM(y.serial_check) = 'Y'
          AND TRIM(COALESCE(y.item_code,'')) = TRIM(COALESCE(b.item_code,''))
          AND y.serial_number IS NOT NULL
          AND TRIM(COALESCE(y.serial_number,'')) != ''
        ORDER BY ABS(DATE(y.bill_date) - DATE(b.bill_date)), DATE(y.bill_date) DESC
        LIMIT 1
    ) y_near ON y_same.serial_number IS NULL
    WHERE 1=1
        {metric_filter}
    ORDER BY b.bill_date DESC, b.serial_check
    """

    fallback_q = f"""
    WITH src AS (
        SELECT
            serial_check,
            serial_number,
            item_code,
            item_name,
            bill_date,
            COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
            COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown') AS cashier
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) = %(d)s
          AND COALESCE(item_code,'') != 'sales data not available'
    ),
    base AS (
        SELECT *
        FROM src
        WHERE COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') = %(shop)s
          AND COALESCE(NULLIF(TRIM(cashier),''),'Unknown') = %(cashier)s
    ),
    dup_sn AS (
        SELECT serial_number FROM src
        WHERE serial_number IS NOT NULL AND TRIM(serial_number) != ''
        GROUP BY serial_number HAVING COUNT(*) > 1
    )
    SELECT
        TRIM(b.serial_check) AS serial_check,
        b.serial_number,
        b.item_code,
        COALESCE(NULLIF(TRIM(b.item_name),''), 'Unknown') AS item_desc,
        b.bill_date,
        b.shop_code,
        b.cashier AS cashier_name,
        COALESCE(y_same.serial_number, y_near.serial_number) AS corrected_serial_number,
        COALESCE(y_same.cashier_name, y_near.cashier_name) AS corrected_by_cashier,
        COALESCE(y_same.bill_date, y_near.bill_date) AS corrected_from_date,
        CASE
            WHEN y_same.serial_number IS NOT NULL THEN 'Same Date Y'
            WHEN y_near.serial_number IS NOT NULL THEN 'Nearest Date Y'
            ELSE NULL
        END AS correction_source
    FROM base b
    LEFT JOIN LATERAL (
        SELECT
            TRIM(y.serial_number) AS serial_number,
            COALESCE(NULLIF(TRIM(y.cashier_name),''),'Unknown') AS cashier_name,
            DATE(y.bill_date) AS bill_date
        FROM serialno_check_yes_no y
        WHERE TRIM(y.serial_check) = 'Y'
          AND TRIM(COALESCE(y.item_code,'')) = TRIM(COALESCE(b.item_code,''))
          AND DATE(y.bill_date) = DATE(b.bill_date)
          AND y.serial_number IS NOT NULL
          AND TRIM(COALESCE(y.serial_number,'')) != ''
        ORDER BY DATE(y.bill_date) DESC
        LIMIT 1
    ) y_same ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            TRIM(y.serial_number) AS serial_number,
            COALESCE(NULLIF(TRIM(y.cashier_name),''),'Unknown') AS cashier_name,
            DATE(y.bill_date) AS bill_date
        FROM serialno_check_yes_no y
        WHERE TRIM(y.serial_check) = 'Y'
          AND TRIM(COALESCE(y.item_code,'')) = TRIM(COALESCE(b.item_code,''))
          AND y.serial_number IS NOT NULL
          AND TRIM(COALESCE(y.serial_number,'')) != ''
        ORDER BY ABS(DATE(y.bill_date) - DATE(b.bill_date)), DATE(y.bill_date) DESC
        LIMIT 1
    ) y_near ON y_same.serial_number IS NULL
    WHERE 1=1
        {metric_filter}
    ORDER BY b.bill_date DESC, b.serial_check
    """
    try:
        params = {'d': safe_date, 'shop': safe_shop, 'cashier': safe_cashier}
        
        df = pd.read_sql(q, conn, params=params)
        if (df is None or df.empty) and metric == "Compliance %":
            df = pd.read_sql(fallback_q, conn, params={'d': safe_date, 'shop': safe_shop, 'cashier': safe_cashier})
        conn.close()
        return df if df is not None and not df.empty else None
    except Exception as e:
        import traceback
        print(f"ERROR in load_shop_selling_metric_detail: {str(e)}")
        print(traceback.format_exc())
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def load_shop_selling_incorrect_serial_table(start_str: str, end_str: str):
    """Direct table for Check Serial No metric (N rows with Y correction mapping)."""
    conn = get_db_connection()
    if not conn:
        return None

    q = """
    WITH base_n AS (
        SELECT
            TRIM(serial_check) AS serial_check,
            TRIM(serial_number) AS serial_number,
            REGEXP_REPLACE(TRIM(COALESCE(serial_number,'')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$','','g') AS serial_number_clean,
            TRIM(item_code) AS item_code,
            COALESCE(NULLIF(TRIM(item_name),''), 'Unknown') AS item_desc,
            DATE(bill_date) AS bill_date,
            COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
            COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown') AS cashier_name
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
          AND TRIM(serial_check) = 'N'
          AND serial_number IS NOT NULL
          AND TRIM(COALESCE(serial_number,'')) != ''
          AND COALESCE(item_code,'') != 'sales data not available'
    )
    SELECT
        b.serial_check AS "serial check",
        b.serial_number AS "serial number(where N)",
        b.item_code AS "item code",
        b.item_desc AS "item desc",
        b.bill_date AS "billdate",
        b.shop_code AS "shop code",
        b.cashier_name AS "cashier name",
        COALESCE(y_same.serial_number, y_near.serial_number) AS "Correct serial number by other shop",
        COALESCE(y_same.cashier_name, y_near.cashier_name) AS "corrected done by cashier",
        COALESCE(y_same.shop_code, y_near.shop_code) AS "corrected done at shop",
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM serialno_check_yes_no y_hist
                WHERE TRIM(COALESCE(y_hist.serial_check,'')) = 'Y'
                  AND REGEXP_REPLACE(TRIM(COALESCE(y_hist.serial_number,'')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$','','g') = b.serial_number_clean
            ) THEN 'Y'
            ELSE b.serial_check
        END AS "updated/correct serial remark",
        CASE
            WHEN (
                ABS(
                    LENGTH(REGEXP_REPLACE(TRIM(COALESCE(b.serial_number,'')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$','','g'))
                    -
                    LENGTH(REGEXP_REPLACE(TRIM(COALESCE(COALESCE(y_same.serial_number, y_near.serial_number),'')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$','','g'))
                ) >= 3
                OR ABS(
                    LENGTH(REGEXP_REPLACE(REGEXP_REPLACE(TRIM(COALESCE(b.serial_number,'')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$','','g'), '[^0-9]', '', 'g'))
                    -
                    LENGTH(REGEXP_REPLACE(REGEXP_REPLACE(TRIM(COALESCE(COALESCE(y_same.serial_number, y_near.serial_number),'')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$','','g'), '[^0-9]', '', 'g'))
                ) >= 3
                OR ABS(
                    LENGTH(REGEXP_REPLACE(REGEXP_REPLACE(TRIM(COALESCE(b.serial_number,'')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$','','g'), '[^A-Za-z]', '', 'g'))
                    -
                    LENGTH(REGEXP_REPLACE(REGEXP_REPLACE(TRIM(COALESCE(COALESCE(y_same.serial_number, y_near.serial_number),'')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$','','g'), '[^A-Za-z]', '', 'g'))
                ) >= 3
            ) THEN 'wrong serial number entered by cashier'
            ELSE CASE
                WHEN sf.is_frag_match = 1 THEN 'special character(frag serialno)'
                WHEN COALESCE(avs.in_wh_received, 0) = 1 AND COALESCE(avs.in_loading, 0) = 0 THEN 'WH recived but no dispact initiated'
                WHEN COALESCE(avs.in_wh_received, 0) = 0 AND COALESCE(avs.in_loading, 0) = 1 THEN 'WH has not recived but wh Loaded'
                WHEN COALESCE(avs.in_wh_received, 0) = 1 AND COALESCE(avs.in_loading, 0) = 1 AND COALESCE(avs.in_vc_offload, 0) = 0 THEN 'wh reveived wh loaded shop doesnt offloaded'
                WHEN COALESCE(avs.in_wh_received, 0) = 0 AND COALESCE(avs.in_loading, 0) = 0 THEN 'older grn or dispatch_or Serial not exist'
                WHEN COALESCE(wl_stats.total_rows, 0) = 0 THEN 'not find in the loading may by older grn'
                WHEN COALESCE(wl_stats.offloaded_rows, 0) = 0 THEN 'Not offloaded'
                WHEN COALESCE(wl_exact.mismatch_exact_rows, 0) > 0 THEN 'offloading serial mismatch'
                WHEN REGEXP_REPLACE(
                        REGEXP_REPLACE(
                            REGEXP_REPLACE(
                                REGEXP_REPLACE(TRIM(COALESCE(b.serial_number,'')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$','','g'),
                                '[A-Za-z]', 'A', 'g'
                            ),
                            '[0-9]', '9', 'g'
                        ),
                        '[^A-Za-z0-9]', 'X', 'g'
                     )
                     =
                     REGEXP_REPLACE(
                        REGEXP_REPLACE(
                            REGEXP_REPLACE(
                                REGEXP_REPLACE(TRIM(COALESCE(COALESCE(y_same.serial_number, y_near.serial_number),'')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$','','g'),
                                '[A-Za-z]', 'A', 'g'
                            ),
                            '[0-9]', '9', 'g'
                        ),
                        '[^A-Za-z0-9]', 'X', 'g'
                     )
                THEN 'serial number NOt Found Check'
                ELSE 'Incorrect SerialNo Sold'
            END
        END AS "remark"
    FROM base_n b
    LEFT JOIN LATERAL (
        SELECT
            TRIM(y.serial_number) AS serial_number,
            COALESCE(NULLIF(TRIM(y.cashier_name),''),'Unknown') AS cashier_name,
            COALESCE(NULLIF(TRIM(y.shop_code),''),'Unknown') AS shop_code,
            DATE(y.bill_date) AS bill_date
        FROM serialno_check_yes_no y
        WHERE TRIM(y.serial_check) = 'Y'
          AND TRIM(COALESCE(y.item_code,'')) = TRIM(COALESCE(b.item_code,''))
          AND DATE(y.bill_date) = b.bill_date
          AND y.serial_number IS NOT NULL
          AND TRIM(COALESCE(y.serial_number,'')) != ''
        ORDER BY DATE(y.bill_date) DESC
        LIMIT 1
    ) y_same ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            TRIM(y.serial_number) AS serial_number,
            COALESCE(NULLIF(TRIM(y.cashier_name),''),'Unknown') AS cashier_name,
            COALESCE(NULLIF(TRIM(y.shop_code),''),'Unknown') AS shop_code,
            DATE(y.bill_date) AS bill_date
        FROM serialno_check_yes_no y
        WHERE TRIM(y.serial_check) = 'Y'
          AND TRIM(COALESCE(y.item_code,'')) = TRIM(COALESCE(b.item_code,''))
          AND y.serial_number IS NOT NULL
          AND TRIM(COALESCE(y.serial_number,'')) != ''
        ORDER BY ABS(DATE(y.bill_date) - b.bill_date), DATE(y.bill_date) DESC
        LIMIT 1
    ) y_near ON y_same.serial_number IS NULL
    LEFT JOIN LATERAL (
        SELECT 1 AS is_frag_match
        FROM serial_no_dailydata s2
        WHERE TRIM(COALESCE(s2.vc_item_code::TEXT,'')) = TRIM(COALESCE(b.item_code,''))
          AND COALESCE(NULLIF(TRIM(s2.vc_shop_code),''),'Unknown') = b.shop_code
          AND DATE(s2.dt_doc_date) <= b.bill_date
          AND b.serial_number <> b.serial_number_clean
          AND b.serial_number_clean <> ''
          AND (
              TRIM(COALESCE(s2.serial_no,'')) = b.serial_number_clean
              OR REGEXP_REPLACE(TRIM(COALESCE(s2.serial_no,'')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$','','g') = b.serial_number_clean
          )
        LIMIT 1
    ) sf ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            CASE WHEN EXISTS (
                SELECT 1 FROM whreceived_serialno w
                WHERE REGEXP_REPLACE(TRIM(COALESCE(w.serial_no,'')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$','','g') = b.serial_number_clean
            ) THEN 1 ELSE 0 END AS in_wh_received,
            CASE WHEN EXISTS (
                SELECT 1 FROM serial_no_dailydata s
                WHERE REGEXP_REPLACE(TRIM(COALESCE(s.serial_no,'')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$','','g') = b.serial_number_clean
            ) THEN 1 ELSE 0 END AS in_loading,
            CASE WHEN EXISTS (
                SELECT 1 FROM serial_no_dailydata s
                WHERE REGEXP_REPLACE(TRIM(COALESCE(s.vc_serail_no,'')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$','','g') = b.serial_number_clean
            ) THEN 1 ELSE 0 END AS in_vc_offload
    ) avs ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            COUNT(*) AS total_rows,
            COUNT(*) FILTER (
                WHERE s.vc_serail_no IS NOT NULL
                  AND TRIM(COALESCE(s.vc_serail_no,'')) <> ''
            ) AS offloaded_rows,
            COUNT(*) FILTER (
                WHERE s.vc_serail_no IS NOT NULL
                  AND TRIM(COALESCE(s.vc_serail_no,'')) <> ''
                  AND s.serial_no IS NOT NULL
                  AND TRIM(COALESCE(s.serial_no,'')) <> ''
                  AND REGEXP_REPLACE(TRIM(COALESCE(s.vc_serail_no,'')), '[^A-Za-z0-9]','','g')
                      <> REGEXP_REPLACE(TRIM(COALESCE(s.serial_no,'')), '[^A-Za-z0-9]','','g')
            ) AS mismatch_rows
        FROM serial_no_dailydata s
        WHERE TRIM(COALESCE(s.vc_item_code::TEXT,'')) = TRIM(COALESCE(b.item_code,''))
          AND COALESCE(NULLIF(TRIM(s.vc_shop_code),''),'Unknown') = b.shop_code
          AND DATE(s.dt_doc_date) <= b.bill_date
    ) wl_stats ON TRUE
    LEFT JOIN LATERAL (
        SELECT
            COUNT(*) FILTER (
                WHERE s.vc_serail_no IS NOT NULL
                  AND TRIM(COALESCE(s.vc_serail_no,'')) <> ''
                  AND s.serial_no IS NOT NULL
                  AND TRIM(COALESCE(s.serial_no,'')) <> ''
                  AND REGEXP_REPLACE(TRIM(COALESCE(s.serial_no,'')), '^[^A-Za-z0-9]+|[^A-Za-z0-9]+$','','g') = b.serial_number_clean
                  AND REGEXP_REPLACE(TRIM(COALESCE(s.vc_serail_no,'')), '[^A-Za-z0-9]','','g')
                      <> REGEXP_REPLACE(TRIM(COALESCE(s.serial_no,'')), '[^A-Za-z0-9]','','g')
            ) AS mismatch_exact_rows
        FROM serial_no_dailydata s
        WHERE TRIM(COALESCE(s.vc_item_code::TEXT,'')) = TRIM(COALESCE(b.item_code,''))
          AND COALESCE(NULLIF(TRIM(s.vc_shop_code),''),'Unknown') = b.shop_code
          AND DATE(s.dt_doc_date) <= b.bill_date
    ) wl_exact ON TRUE
    WHERE COALESCE(y_same.serial_number, y_near.serial_number) IS NOT NULL
    ORDER BY b.bill_date DESC, b.shop_code, b.cashier_name
    """
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = 0")
        df = pd.read_sql(q, conn, params={'s': start_str, 'e': end_str})
        conn.close()
        return df if df is not None and not df.empty else None
    except Exception as ex:
        st.warning(f"Check Serial No table error: {ex}")
        try:
            conn.close()
        except Exception:
            pass
        return None


# ─────────────────────────────────────────────────────────────
# CSS  (vertical pipeline design)
# ─────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --card-bg: #1A2847;
    --border-color: #2A3A5A;
}

#MainMenu, footer, header { visibility: hidden; }

html, body, .stApp {
    background: #0a0f1e !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    color: #e2e8f0 !important;
}
.block-container {
    padding-top: 1rem !important;
    max-width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}
[data-testid="column"] { padding: 0 3px !important; }

/* ── Primary KPI box (left zone) ── */
.kpi-box {
    border-radius: 12px; padding: 22px 18px 18px;
    display: flex; flex-direction: column; justify-content: center;
    height: 100%; min-height: 185px;
}
.kpi-label {
    font-size: 0.60rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.09em; margin-bottom: 6px;
}
.kpi-value {
    font-size: 2.8rem; font-weight: 800; line-height: 1.05;
    color: #f8fafc; margin-bottom: 4px;
}
.kpi-sub { font-size: 0.74rem; font-weight: 500; }
.kpi-divider { width: 44px; height: 3px; border-radius: 2px; margin: 8px 0 10px; }
.kpi-secondary { font-size: 0.78rem; color: #94a3b8; line-height: 1.75; }

/* ── Quality tile (right zone) ── */
.q-tile {
    border-radius: 8px; padding: 10px 8px 8px;
    text-align: center; display: flex; flex-direction: column;
    justify-content: center; min-height: 76px;
    border: 1px solid rgba(255,255,255,0.07);
}
.q-label {
    font-size: 0.56rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 3px;
}
.q-value { font-size: 1.15rem; font-weight: 800; color: #f1f5f9; line-height: 1.1; }
.q-sub   { font-size: 0.60rem; margin-top: 3px; font-weight: 600; }

/* ── Flow connector between stages ── */
.flow-wrap {
    display: flex; flex-direction: column; align-items: center;
    margin: 0; padding: 0;
}
.flow-line {
    width: 2px; height: 20px;
}
.flow-badge {
    border-radius: 20px; padding: 3px 16px; font-size: 0.67rem; font-weight: 700;
    border: 1px solid; display: inline-flex; align-items: center;
    gap: 6px; white-space: nowrap; background: rgba(255,255,255,0.04);
    border-color: rgba(255,255,255,0.11); color: #94a3b8;
}

/* ── Plotly + DataFrame styling (from serial_tracker_dashboard_new.py) ── */
div[data-testid="stPlotlyChart"],
div[data-testid="stDataFrame"] {
    background: linear-gradient(135deg, var(--card-bg) 0%, rgba(26, 40, 71, 0.8) 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 0.6rem;
    box-shadow: 0 8px 24px rgba(91, 84, 255, 0.1);
}

div[data-testid="stDataFrame"] table,
div[data-testid="stDataFrame"] table *,
.stDataFrame table,
.stDataFrame table *,
[data-testid="stDataFrame"] *,
.dataframe,
.dataframe * {
    font-size: 1.5rem !important;
}

div[data-testid="stDataFrame"] thead th,
.stDataFrame thead th,
[data-testid="stDataFrame"] thead th {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    padding: 1rem !important;
    line-height: 1.5 !important;
}

div[data-testid="stDataFrame"] tbody td,
.stDataFrame tbody td,
[data-testid="stDataFrame"] tbody td {
    font-size: 1.5rem !important;
    padding: 0.875rem !important;
    line-height: 1.5 !important;
}

div[data-testid="stDataFrame"] [data-testid="stDataFrameCell"],
[data-testid="stDataFrameCell"] {
    font-size: 1.5rem !important;
}

div[data-testid="stDataFrame"] div[role="gridcell"],
div[data-testid="stDataFrame"] div[role="columnheader"] {
    font-size: 1.5rem !important;
}

div[data-testid="stDataFrame"] {
    background: #f8fafc !important;
    border: 1px solid rgba(15, 23, 42, 0.12) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
}

div[data-testid="stDataFrame"] thead {
    background: linear-gradient(135deg,
        rgba(139, 92, 246, 0.15) 0%,
        rgba(91, 84, 255, 0.12) 50%,
        rgba(59, 130, 246, 0.1) 100%) !important;
    border-bottom: 2px solid rgba(139, 92, 246, 0.3) !important;
    position: relative !important;
}

div[data-testid="stDataFrame"] thead::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(139, 92, 246, 0.6) 50%,
        transparent 100%);
}

div[data-testid="stDataFrame"] thead th {
    background: transparent !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    border-bottom: none !important;
    padding: 1.25rem 1rem !important;
    font-size: 0.8rem !important;
    text-align: center !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.4);
}

div[data-testid="stDataFrame"] tbody tr {
    border-bottom: 1px solid rgba(15, 23, 42, 0.08) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    background: #f8fafc !important;
}

div[data-testid="stDataFrame"] tbody tr:hover {
    background: #e2e8f0 !important;
    transform: translateX(2px);
    box-shadow: inset 3px 0 0 rgba(139, 92, 246, 0.8);
}

div[data-testid="stDataFrame"] tbody td {
    color: #0f172a !important;
    padding: 1rem !important;
    font-size: 0.95rem !important;
    border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
    text-align: center !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
}


div[data-testid="stDataFrame"] tbody td:first-child {
    font-weight: 700 !important;
    background: #f1f5f9 !important;
    border-right: 1px solid rgba(15, 23, 42, 0.12) !important;
    text-align: center !important;
    padding-left: 1rem !important;
    letter-spacing: 0.5px;
}

div[data-testid="stDataFrame"] tbody td:last-child {
    border-right: none !important;
    font-weight: 600 !important;
}

div[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background: rgba(255, 255, 255, 0.015) !important;
}

/* ── Expander dark theme ── */
div[data-testid="stExpander"] > details {
    background: rgba(3,28,18,0.75) !important;
    border: 1px solid rgba(52,211,153,0.3) !important;
    border-left: 4px solid #34d399 !important;
    border-radius: 8px !important; margin-top: 4px !important;
}
div[data-testid="stExpander"] > details > summary {
    background: rgba(3,36,24,0.90) !important; color: #a7f3d0 !important;
    font-weight: 700 !important; font-size: 0.86rem !important;
    padding: 10px 16px !important; border-radius: 7px !important;
}
div[data-testid="stExpander"] > details[open] > summary {
    border-bottom: 1px solid rgba(52,211,153,0.2) !important;
    border-radius: 7px 7px 0 0 !important; color: #6ee7b7 !important;
}
div[data-testid="stExpander"] > details > div[data-testid="stExpanderDetails"] {
    background: rgba(2,22,14,0.82) !important;
    border-radius: 0 0 7px 7px !important; padding: 12px 16px 14px !important;
}
/* ── Drill view cards ── */
.drill-card {
     background: linear-gradient(160deg, #0c1a3a 0%, #080f22 100%);
     border: 1px solid rgba(59,130,246,0.15);
     border-radius: 14px;
     padding: 16px 16px 8px;
     margin-top: 10px;
     box-shadow: 0 10px 24px rgba(0,0,0,0.25);
}
.drill-table-card {
    padding: 8px 10px 6px !important;
    margin-top: 6px !important;
}
.drill-table-title {
    margin: 0 0 4px 0 !important;
    color: #cbd5e1;
    font-size: 0.86rem;
    line-height: 1.2;
}
.drill-card h3, .drill-card h4 {
     margin-top: 0;
}
.drill-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 4px 0;
}
.stage-icon {
    width: 28px;
    height: 28px;
    border-radius: 9px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.70rem;
    font-weight: 800;
    color: #f8fafc;
    border: 1px solid rgba(255,255,255,0.20);
    box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    letter-spacing: 0.04em;
}
.stage-icon.wh { background: linear-gradient(135deg, #1f4bd1 0%, #2b79ff 100%); }
.stage-icon.sr { background: linear-gradient(135deg, #0f9f78 0%, #18c38e 100%); }
.stage-icon.ss { background: linear-gradient(135deg, #6a38f5 0%, #9b5bff 100%); }

.drill-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-top: 6px;
    margin-bottom: 8px;
}
.drill-nav-right {
    display: flex;
    align-items: center;
    gap: 8px;
}
.drill-nav-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    height: 34px;
    padding: 0 14px;
    border-radius: 10px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    background: linear-gradient(135deg, rgba(148,163,184,0.14) 0%, rgba(71,85,105,0.18) 100%);
    color: #e2e8f0;
    font-size: 0.78rem;
    font-weight: 700;
    text-decoration: none;
    letter-spacing: 0.01em;
    transition: all 0.18s ease;
}
.drill-nav-link:hover {
    border-color: rgba(59,130,246,0.55);
    color: #f8fafc;
    transform: translateY(-1px);
    box-shadow: 0 6px 14px rgba(59,130,246,0.18);
}
.drill-nav-link.primary {
    border: 1px solid rgba(59,130,246,0.5);
    background: linear-gradient(135deg, rgba(30,58,138,0.45) 0%, rgba(59,130,246,0.35) 100%);
    color: #dbeafe;
}
.drill-nav-link.disabled {
    opacity: 0.35;
    pointer-events: none;
}

/* ── Metrics styling ── */
div[data-testid="metric-container"] {
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 12px;
    padding: 1.5rem;
}

div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# VERTICAL DESIGN — HTML BUILDERS
# ─────────────────────────────────────────────────────────────

# Stage palettes: (panel_bg, header_bg, accent, kpi_bg, q_tile_bg, label_color)
BLUE_P   = ("#0c1a3a", "#08143099", "#3b82f6", "rgba(10,30,90,0.80)",  "rgba(12,30,80,0.55)",  "#93c5fd")
TEAL_P   = ("#06222e", "#04182299", "#14b8a6", "rgba(6,44,58,0.80)",   "rgba(8,50,64,0.55)",   "#5eead4")
GREEN_P  = ("#042a1c", "#021a1099", "#10b981", "rgba(4,42,28,0.80)",   "rgba(6,48,32,0.55)",   "#6ee7b7")
VIOLET_P = ("#160d2e", "#0e081e99", "#8b5cf6", "rgba(30,16,60,0.80)",  "rgba(36,20,70,0.55)",  "#c4b5fd")


def stage_header_html(icon: str, title: str, accent: str, header_bg: str,
                      date_str: str, status_ok: bool, extra: str = "") -> str:
    dot   = "🟢" if status_ok else "🔴"
    badge_color = "#10b981" if status_ok else "#ef4444"
    label = "Clean" if status_ok else "Issues"
    return f"""
<div style="background:{header_bg};border-left:4px solid {accent};
            border-radius:10px 10px 0 0;padding:10px 18px;
            display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:18px;">
    <span style="font-size:1.0rem;font-weight:800;color:{accent};">{icon} {title}</span>
    <span style="background:{badge_color}22;border:1px solid {badge_color}44;color:{badge_color};
                 border-radius:20px;padding:2px 11px;font-size:0.70rem;font-weight:700;">
        {dot} {label}
    </span>
    <span style="color:#475569;font-size:0.85rem;">|</span>
    <span style="color:#64748b;font-size:0.76rem;">📅 {date_str}</span>
    {extra}
</div>"""


def flow_connector_html(count_label: str, count_val: str,
                        from_accent: str, to_accent: str) -> str:
    return f"""
<div style="display:flex;flex-direction:column;align-items:center;
            padding:6px 0 2px;gap:3px;">
    <div style="width:2px;height:14px;
                background:linear-gradient({from_accent},{to_accent});border-radius:2px;"></div>
    <div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.10);
                border-radius:20px;padding:3px 14px;font-size:0.72rem;color:#94a3b8;
                font-weight:600;letter-spacing:0.04em;">
        ↓ &nbsp;{count_label}: <span style="color:#f1f5f9;font-weight:700;">{count_val}</span>
    </div>
    <div style="width:2px;height:14px;
                background:linear-gradient({from_accent},{to_accent});border-radius:2px;"></div>
</div>"""


def kpi_box(bg: str, accent: str, label: str, value: str,
            sub_label: str, sub_ok: bool, left_stats: list) -> str:
    sc = "#6ee7b7" if sub_ok else "#fca5a5"
    rows = ""
    for (sl, sv, sok) in left_stats:
        vc = "#f1f5f9" if sok else "#fca5a5"
        ic = "✓" if sok else "⚠"
        rows += (f'<div style="font-size:0.75rem;color:{vc};line-height:1.8;">'
                 f'<span style="color:#64748b;">{sl}</span>'
                 f'<span style="float:right;font-weight:700;">{ic} {sv}</span></div>')
    return f"""
<div style="background:{bg};border:1px solid {accent}44;border-radius:0 0 0 10px;
            padding:16px 14px 14px;min-height:185px;display:flex;flex-direction:column;
            justify-content:center;gap:6px;height:100%;">
    <div style="font-size:0.60rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.10em;color:{accent};margin-bottom:2px;">{label}</div>
    <div style="font-size:2.8rem;font-weight:800;color:#f8fafc;line-height:1.0;">{value}</div>
    <div style="width:44px;height:3px;background:{accent};border-radius:2px;margin:2px 0 4px;"></div>
    <div style="font-size:0.74rem;color:{sc};font-weight:600;margin-bottom:4px;">{sub_label}</div>
    {rows}
</div>"""


def q_tile(bg: str, label_color: str, label: str, value: str,
           sub: str, ok: bool = True, sub_color: str = None) -> str:
    sc = sub_color or ("#6ee7b7" if ok else "#fca5a5")
    bdr = "rgba(255,255,255,0.05)" if ok else "rgba(239,68,68,0.35)"
    return f"""
<div style="background:{bg};border:1px solid {bdr};border-radius:8px;
            padding:10px 8px;text-align:center;min-height:76px;
            display:flex;flex-direction:column;justify-content:center;gap:3px;">
    <div style="font-size:0.60rem;color:{label_color};font-weight:700;
                text-transform:uppercase;letter-spacing:0.06em;">{label}</div>
    <div style="font-size:1.40rem;font-weight:800;color:#f8fafc;line-height:1.15;">{value}</div>
    <div style="font-size:0.68rem;color:{sc};font-weight:500;">{sub}</div>
</div>"""


def _metric_tooltip(stage: str, label: str) -> str:
    stage_map = {
        "🏭 WH Receiving": {
            "Unique %": "Percent of WH receiving records with a valid, non-blank serial.",
            "Unique Serials": "Count of distinct serial numbers received in the warehouse.",
            "In-Batch Dup": "Same serial repeated more than once in the selected receiving records.",
            "WH DB Dup": "Serial already exists in WH history (seen before this date range).",
            "Blank Serials": "Serial field is empty or missing.",
            "Small (≤6)": "Serial length is 6 characters or less (likely invalid).",
        },
        "📦 WH Loading": {
            "Unique %": "Percent of WH loading records with a valid serial.",
            "Duplicates": "Serial numbers that appear under more than one item code in the selected dates.",
            "Blank Serials": "Serial field is empty or missing.",
            "Small (≤6)": "Serial length is 6 characters or less.",
            "IC": "Serial matches item code (likely a placeholder).",
        },
        "🏪 Shop Receiving": {
            "Not Offloaded": "Loaded from WH but not yet received at shop.",
            "Total Offloaded": "Items received at shop (offloaded).",
            "Same-Day": "Received on the same day they were dispatched.",
            "Old Doc Date": "Received now but dispatch date is outside the selected range.",
            "WH→Shop Mismatch": "Serial at shop does not match WH serial.",
            "Unique %": "Percent of shop receiving records with a valid serial.",
            "Blank": "Serial field is empty or missing at shop.",
            "Small (≤6)": "Serial length is 6 characters or less.",
            "IC": "Shop serial matches item code (likely a placeholder).",
            "Duplicates": "Same serial appears more than once in shop receiving.",
        },
        "🛒 Shop Compliance %": {
            "Compliance %": "Percent of sales with a serial captured.",
            "Dup Serials": "Same serial used in more than one sale.",
            "No Serial": "Sale has no serial.",
            "Not in WH": "Serial sold but not found in WH receiving records.",
            "Check Serial No": "Sale marked as N, but same item has Y serial pattern (same date or nearest date).",
            "Serial = IC": "Serial equals item code (likely a placeholder).",
            "Small (≤6)": "Serial length is 6 characters or less.",
        },
    }

    label_map = {
        "IC": "Serial matches item code (likely a placeholder).",
        "Blank": "Serial field is empty or missing.",
        "Same Day": "Offloaded on the same day as dispatch.",
        "1-3 Days": "Offloaded 1 to 3 days after dispatch.",
        ">3 Days": "Offloaded more than 3 days after dispatch.",
    }

    if stage in stage_map and label in stage_map[stage]:
        return stage_map[stage][label]
    if label.startswith("Offloaded ("):
        return "Items offloaded on that day."
    return label_map.get(label)


def _row(label, value_str, ok, accent, bar_pct=None, note=None, stage=None, metric=None, date_range=None,
         label_color=None, value_color=None):
    """Single metric row: dot + label + optional bar + value + optional note.
    If stage and metric provided, value becomes clickable (keeps original color, underline on hover).
    date_range: tuple of (start_str, end_str) to preserve dates in navigation."""
    dot_c  = "#10b981" if ok else "#ef4444"
    val_c  = value_color or ("#6ee7b7" if ok else "#fca5a5")
    lbl_c  = label_color or ("#64748b" if ok else "#94a3b8")
    bar_html = ""
    if bar_pct is not None:
        clip = max(0, min(bar_pct, 100))
        bar_col = "#10b981" if ok else "#ef4444"
        bar_html = (f'<div style="width:42px;height:3px;background:rgba(255,255,255,0.08);'
                    f'border-radius:2px;overflow:hidden;flex-shrink:0;">'  
                    f'<div style="width:{clip:.0f}%;height:100%;background:{bar_col};border-radius:2px;"></div>'
                    f'</div>')
    if note:
        note_color = "#f8fafc" if "WH Loaded" in note else "#475569"
        note_html = f'<span style="font-size:0.58rem;color:{note_color};margin-left:4px;">{note}</span>'
    else:
        note_html = ""
    
    tooltip = _metric_tooltip(stage, label)
    tooltip_attr = f' title="{html.escape(tooltip)}"' if tooltip else ""

    # ✅ FIX: If drillable metric, include date range in query params to preserve selection
    if stage and metric:
        stage_q = quote(stage)
        metric_q = quote(metric)
        # Include date range in URL to preserve user's selection
        date_params = ""
        if date_range and len(date_range) == 2:
            date_params = f"&start_date={date_range[0]}&end_date={date_range[1]}"
        value_html = (
            f'<a href="?stage={stage_q}&metric={metric_q}{date_params}#drill-section" target="_self" '
            f'style="font-size:0.73rem;color:{val_c};font-weight:700;min-width:38px;'
            f'text-align:right;cursor:pointer;transition:all 0.2s;text-decoration:none;" '
            f'onmouseover="this.style.textDecoration=\'underline\';this.style.opacity=\'0.8\';" '
            f'onmouseout="this.style.textDecoration=\'none\';this.style.opacity=\'1\';">'
            f'{value_str}</a>'
        )
    else:
        value_html = f'<span style="font-size:0.73rem;color:{val_c};font-weight:700;min-width:38px;text-align:right;">{value_str}</span>'
    
    return (
        f'<div style="display:flex;align-items:center;gap:5px;padding:4px 0;'
        f'border-bottom:1px solid rgba(255,255,255,0.04);"{tooltip_attr}>'
        f'<span style="width:6px;height:6px;border-radius:50%;background:{dot_c};flex-shrink:0;"></span>'
        f'<span style="flex:1;font-size:0.67rem;color:{lbl_c};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{label}</span>'
        f'{bar_html}'
        f'{value_html}'
        f'{note_html}'
        f'</div>'
    )


def render_drill_down_tab(stage, metric, drilldown_cache, start_s, end_s):
    stage_order = ["🏭 WH Receiving", "📦 WH Loading", "🏪 Shop Receiving", "🛒 Shop Compliance %"]
    stage_metric_order = {
        "🏭 WH Receiving": [
            "Unique %", "Unique Serials", "In-Batch Dup",
            "WH DB Dup", "Blank Serials", "Small (≤6)"
        ],
        "📦 WH Loading": ["Unique %", "Duplicates", "Blank Serials", "Small (≤6)", "IC"],
        "🏪 Shop Receiving": [
            "Not Offloaded", "Total Offloaded", "Same-Day", "Old Doc Date",
            "WH→Shop Mismatch", "Unique %", "Blank", "Small (≤6)", "IC", "Duplicates"
        ],
        "🛒 Shop Compliance %": [
            "Compliance %", "Dup Serials", "No Serial", "Not in WH",
            "Check Serial No", "Serial = IC", "Small (≤6)"
        ],
    }

    if not stage:
        st.info("Select a metric to open drill-down view.")
        return

    if stage == "🛒 Shop Compliance %":
        metrics = stage_metric_order.get(stage, [])
    else:
        cached_metrics = [
            key.split("||", 1)[1]
            for key in drilldown_cache.keys()
            if key.startswith(f"{stage}||")
        ]
        # ✅ OPTIMIZED: If no cached metrics yet, use all defined metrics for stage
        if not cached_metrics:
            cached_metrics = stage_metric_order.get(stage, [])
        ordered_metrics = [
            m for m in stage_metric_order.get(stage, []) if m in cached_metrics
        ]
        extra_metrics = [m for m in cached_metrics if m not in ordered_metrics]
        metrics = ordered_metrics + sorted(extra_metrics)

    if not metrics:
        st.warning("No drill data available for this stage yet.")
        return

    selected_metric = metric if metric in metrics else metrics[0]

    # ✅ OPTIMIZED: Since drilldowns load on-demand, all stages have data
    def stage_has_data(stage_name: str) -> bool:
        return True  # Always True since we load on-demand

    # Navigation controls
    prev_stage = None
    next_stage = None
    if stage in stage_order:
        idx = stage_order.index(stage)
        prev_candidate = stage_order[idx - 1] if idx > 0 else None
        next_candidate = stage_order[idx + 1] if idx < len(stage_order) - 1 else None
        prev_stage = prev_candidate if prev_candidate and stage_has_data(prev_candidate) else None
        next_stage = next_candidate if next_candidate and stage_has_data(next_candidate) else None

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
    nav_cols = st.columns([1.25, 2.2, 1, 1])

    with nav_cols[0]:
        back_clicked = st.button(
            "↩ Back to Pipeline",
            key=f"back_pipeline_{stage}_{selected_metric}",
            type="primary",
            use_container_width=False,
        )
        if back_clicked:
            st.session_state.current_view = 'pipeline'
            st.session_state.selected_stage = None
            st.session_state.selected_metric = None
            st.query_params.clear()
            st.rerun()

    with nav_cols[2]:
        prev_clicked = st.button(
            "← Previous",
            key=f"prev_{stage}_{selected_metric}",
            disabled=not bool(prev_stage),
            use_container_width=False,
        )

    with nav_cols[3]:
        next_clicked = st.button(
            "Next →",
            key=f"next_{stage}_{selected_metric}",
            disabled=not bool(next_stage),
            use_container_width=False,
        )

    if prev_clicked and prev_stage:
        prev_metrics = stage_metric_order.get(prev_stage, [])
        prev_metric = prev_metrics[0] if prev_metrics else metrics[0]
        st.session_state.current_view = 'drill'
        st.session_state.selected_stage = prev_stage
        st.session_state.selected_metric = prev_metric
        st.query_params.clear()
        st.rerun()

    if next_clicked and next_stage:
        next_metrics = stage_metric_order.get(next_stage, [])
        next_metric = next_metrics[0] if next_metrics else metrics[0]
        st.session_state.current_view = 'drill'
        st.session_state.selected_stage = next_stage
        st.session_state.selected_metric = next_metric
        st.query_params.clear()
        st.rerun()

    st.markdown("<div style='height:2px;'></div>", unsafe_allow_html=True)

    stage_label = stage.replace("🏭 ", "").replace("📦 ", "").replace("🏪 ", "").replace("🛒 ", "")
    stage_icons = {
        "🏭 WH Receiving": ("wr", "WR"),
        "📦 WH Loading": ("wh", "WH"),
        "🏪 Shop Receiving": ("sr", "SR"),
        "🛒 Shop Compliance %": ("ss", "SC"),
    }
    icon_class, icon_text = stage_icons.get(stage, ("wh", "ST"))
    icon_html = f"<span class=\"stage-icon {icon_class}\">{icon_text}</span>"
    heading = f"{stage_label} {selected_metric}"

    st.markdown(f"<h2 class=\"drill-heading\">{icon_html}<span>{heading}</span></h2>", unsafe_allow_html=True)
    st.caption(f"Date range: {start_s} to {end_s}")

    cache_key = f"{stage}||{selected_metric}"
    
    # ✅ OPTIMIZED: Lazy-load drilldown on-demand (only when user clicks)
    df = drilldown_cache.get(cache_key)
    if df is None:
        # Load drilldown data on-demand based on stage
        if stage == "🏭 WH Receiving":
            df = load_wh_receiving_drilldown(start_s, end_s, selected_metric)
        elif stage == "📦 WH Loading":
            df = load_wh_loading_drilldown(start_s, end_s, selected_metric)
        elif stage == "🏪 Shop Receiving":
            df = load_shop_receiving_drilldown(start_s, end_s, selected_metric)
        elif stage == "🛒 Shop Compliance %":
            # Shop Compliance has its own stepwise loaders below.
            # Avoid running the heavy generic drill query on every metric click.
            df = pd.DataFrame()

        # Cache the loaded data
        if df is not None and not df.empty:
            drilldown_cache[cache_key] = df
    
    if (df is None or df.empty) and stage != "🛒 Shop Compliance %":
        st.warning("No drill data available for this metric.")
        return

    if df is None:
        df = pd.DataFrame()

    if stage == "🛒 Shop Compliance %" and not df.empty and 'cashier' in df.columns:
        preferred = [c for c in ['cashier', 'shop_code'] if c in df.columns]
        others = [c for c in df.columns if c not in preferred]
        df = df[preferred + others]
    elif stage == "📦 WH Loading" and not df.empty and 'loader' in df.columns:
        preferred = ['loader']
        others = [c for c in df.columns if c not in preferred]
        df = df[preferred + others]

    def _render_styled_table(
        table_df,
        key_suffix: str,
        title_text: str = None,
        height: int = 420,
        enable_select: bool = False,
        enable_filter: bool = True,
        no_conditional_cols: list | None = None,
    ):
        def _sanitize_text(val):
            if pd.isna(val):
                return ""
            text = str(val)
            # Remove control characters that can break frontend JSON/Arrow parsing.
            text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
            return text.strip()

        def _sanitize_df_for_grid(df_in: pd.DataFrame) -> pd.DataFrame:
            safe_df = df_in.copy()
            safe_df.columns = [_sanitize_text(c) for c in safe_df.columns]
            for c in safe_df.columns:
                if pd.api.types.is_object_dtype(safe_df[c]) or pd.api.types.is_string_dtype(safe_df[c]):
                    safe_df[c] = safe_df[c].map(_sanitize_text)
            return safe_df

        safe_key = re.sub("[^A-Za-z0-9_]+", "_", key_suffix)
        st.markdown('<div class="drill-card drill-table-card">', unsafe_allow_html=True)
        if title_text:
            st.markdown(f"<h4 class='drill-table-title'>{title_text}</h4>", unsafe_allow_html=True)

        display_df = table_df.copy()
        display_df = display_df.where(pd.notna(display_df), "")
        display_df = display_df.replace({"nan": "", "NaN": "", "None": ""})

        all_cols = list(display_df.columns)
        if all_cols and enable_filter:
            filter_col_key = f"filter_col_{safe_key}"
            filter_text_key = f"filter_text_{safe_key}"
            default_filter_col = "shop_code" if "shop_code" in all_cols else all_cols[0]
            default_idx = all_cols.index(default_filter_col)

            with st.expander("Filter rows", expanded=False):
                f_cols = st.columns([1, 2])
                with f_cols[0]:
                    selected_filter_col = st.selectbox(
                        "Column",
                        options=all_cols,
                        index=default_idx,
                        key=filter_col_key,
                    )
                with f_cols[1]:
                    filter_text = st.text_input(
                        "Contains",
                        key=filter_text_key,
                        placeholder="e.g. MSS",
                    ).strip()

                if filter_text:
                    display_df = display_df[
                        display_df[selected_filter_col].astype(str).str.contains(filter_text, case=False, na=False)
                    ]
                    display_df = display_df.reset_index(drop=True)
                    st.caption(f"Filtered rows: {len(display_df)}")

        id_cols_local = [c for c in ["loader", "cashier", "shop_code", "shop_name"] if c in display_df.columns]
        value_cols_local = [c for c in display_df.columns if c not in id_cols_local]
        excluded_conditional_cols = set(no_conditional_cols or [])
        conditional_cols_local = [c for c in value_cols_local if c not in excluded_conditional_cols]

        for col in value_cols_local:
            display_df[col] = display_df[col].astype(str).replace({"<NA>": "", "nan": "", "NaN": "", "None": ""})

        styled_df = display_df.style

        has_percent = False
        if value_cols_local:
            for col in value_cols_local:
                if display_df[col].astype(str).str.contains("%", regex=False).any():
                    has_percent = True
                    break
        if "%" in selected_metric:
            has_percent = True

        def _percent_band_style(val):
            raw = str(val).strip()
            if raw.lower() in ("", "nan", "none"):
                return ""
            try:
                num = float(raw.replace("%", ""))
            except Exception:
                return ""
            if num < 90:
                return "background-color: #fecaca; color: #0f172a; font-weight: 700;"
            if num < 95:
                return "background-color: #fef9c3; color: #0f172a; font-weight: 700;"
            if num >= 95:
                return "background-color: #bbf7d0; color: #0f172a; font-weight: 700;"
            return ""

        if has_percent and conditional_cols_local:
            styled_df = styled_df.applymap(_percent_band_style, subset=conditional_cols_local)

        # WH Loading gradient formatting for Duplicates, Small, and IC metrics
        if stage == "📦 WH Loading" and selected_metric in ["Duplicates", "Small (≤6)", "IC"] and conditional_cols_local:
            col_max_map = {}
            col_min_map = {}
            for col in conditional_cols_local:
                numeric_col = pd.to_numeric(display_df[col].astype(str).str.replace(",", ""), errors="coerce")
                col_max_map[col] = numeric_col.max(skipna=True)
                col_min_map[col] = numeric_col.min(skipna=True)

            def _dup_color(value, col_min, col_max):
                raw = str(value).strip()
                if raw.lower() in ("", "nan", "none"):
                    return ""
                try:
                    num = float(raw.replace(",", ""))
                except Exception:
                    return ""
                if col_max is None or pd.isna(col_max) or col_min is None or pd.isna(col_min):
                    return ""
                if col_max == col_min:
                    return "background-color: #fef9c3;"

                ratio = (num - col_min) / (col_max - col_min)
                ratio = max(0.0, min(ratio, 1.0))

                if ratio <= 0.5:
                    t = ratio / 0.5
                    r = int(220 + (254 - 220) * t)
                    g = int(252 + (249 - 252) * t)
                    b = int(231 + (195 - 231) * t)
                else:
                    t = (ratio - 0.5) / 0.5
                    r = int(254 + (254 - 254) * t)
                    g = int(249 + (226 - 249) * t)
                    b = int(195 + (226 - 195) * t)

                return f"background-color: rgb({r}, {g}, {b});"

            def _apply_dup_style(col):
                cmax = col_max_map.get(col.name)
                cmin = col_min_map.get(col.name)
                return col.map(lambda v: _dup_color(v, cmin, cmax))

            styled_df = styled_df.apply(_apply_dup_style, subset=conditional_cols_local)

        def _fmt_cell(value):
            if pd.isna(value):
                return ""
            text = str(value).strip()
            if text == "":
                return ""
            if text.endswith("%"):
                return text
            num = pd.to_numeric(text.replace(",", ""), errors="coerce")
            if pd.isna(num):
                return text
            if float(num).is_integer():
                return f"{int(num):,}"
            return f"{num:,.2f}"

        fmt_subset = {col: _fmt_cell for col in value_cols_local}
        styled_df = styled_df.format(fmt_subset)

        styled_df = styled_df.set_properties(**{"text-align": "center"})
        csv_link = None
        try:
            csv_data = display_df.to_csv(index=False)
            b64 = base64.b64encode(csv_data.encode("utf-8")).decode("utf-8")
            csv_link = (
                f"<div style='display:flex;justify-content:flex-end;margin:2px 0 6px;'>"
                f"<a href='data:text/csv;base64,{b64}' download='{safe_key}.csv' "
                f"style='font-size:0.72rem;color:#93c5fd;text-decoration:underline;'>Download CSV</a>"
                f"</div>"
            )
        except Exception:
            csv_link = None

        safe_grid_df = _sanitize_df_for_grid(display_df)

        if enable_select:
            event = st.dataframe(
                safe_grid_df,
                use_container_width=True,
                hide_index=True,
                height=height,
                key=safe_key,
                on_select="rerun",
                selection_mode="single-cell",
            )
        else:
            # Use plain sanitized dataframe for maximum frontend Arrow compatibility.
            st.dataframe(safe_grid_df, use_container_width=True, hide_index=True, height=height, key=safe_key)
        if csv_link:
            st.markdown(csv_link, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if not enable_select:
            return None

        if event and hasattr(event, "selection"):
            selection = event.selection
            cells = getattr(selection, "cells", []) or []
            if cells:
                row_idx, col_name = cells[0]
                if row_idx < len(safe_grid_df) and col_name in safe_grid_df.columns:
                    return {
                        "row_idx": row_idx,
                        "col_idx": safe_grid_df.columns.get_loc(col_name),
                        "row": safe_grid_df.iloc[row_idx].to_dict(),
                        "col_name": col_name,
                    }

        return None

    def load_wh_loading_detail(metric: str, loader: str = None, shop_code: str = None, date_str: str = None,
                               start_str: str = None, end_str: str = None):
        conn = get_db_connection()
        if not conn:
            return None

        if date_str:
            date_filter = "dt_doc_date::DATE = %(d)s"
            params = {"d": date_str}
        else:
            date_filter = "dt_doc_date::DATE BETWEEN %(s)s AND %(e)s"
            params = {"s": start_str, "e": end_str}

        loader_filter = ""
        if loader:
            loader_filter = "AND TRIM(wh_load_user) = %(loader)s"
            params["loader"] = loader

        shop_filter = ""
        if shop_code:
            shop_filter = "AND TRIM(vc_shop_code) = %(shop)s"
            params["shop"] = shop_code

        metric_filter = ""
        if metric == "Duplicates":
            metric_filter = "AND serial_clean IN (SELECT serial_clean FROM dupes)"
        elif metric == "Blank Serials":
            metric_filter = "AND (serial_no IS NULL OR LENGTH(TRIM(serial_no)) = 0)"
        elif metric == "Small (≤6)":
            metric_filter = "AND serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) <= 6"
        elif metric == "Unique %":
            metric_filter = "AND serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0"
        elif metric == "IC":
            metric_filter = """AND serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
                  AND REGEXP_REPLACE(TRIM(serial_no), '[^A-Za-z0-9]','','g') != ''
                  AND REGEXP_REPLACE(TRIM(serial_no), '[^A-Za-z0-9]','','g') =
                      REGEXP_REPLACE(TRIM(vc_item_code::TEXT), '[^A-Za-z0-9]','','g')"""

        q = f"""
        WITH base AS (
            SELECT
                dt_doc_date::DATE AS doc_date,
                COALESCE(NULLIF(TRIM(wh_load_user),''),'Unknown') AS loader,
                COALESCE(NULLIF(TRIM(vc_wh_code),''),'Unknown') AS wh_code,
                COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
                vc_item_code,
                vc_item_desc,
                TRIM(serial_no) AS serial_clean,
                REGEXP_REPLACE(TRIM(vc_item_code::TEXT), '[^A-Za-z0-9]','','g') AS item_code_clean,
                serial_no,
                dt_mod_date,
                dt_invoice_date,
                shop_sold
            FROM serial_no_dailydata
            WHERE dt_doc_date IS NOT NULL
              AND {date_filter}
              {loader_filter}
              {shop_filter}
        ),
        dupes AS (
            SELECT serial_clean
            FROM base
            WHERE serial_clean IS NOT NULL AND serial_clean != ''
            GROUP BY serial_clean
            HAVING COUNT(DISTINCT NULLIF(item_code_clean,'')) > 1
        )
        SELECT
            serial_no,
            vc_item_code AS item_code,
            vc_item_desc AS item_name,
            wh_code,
            shop_code,
            shop_sold,
            loader,
            doc_date,
            dt_mod_date,
            dt_invoice_date
        FROM base
        WHERE 1=1
            {metric_filter}
        ORDER BY doc_date DESC, loader, shop_code
        """
        try:
            df = pd.read_sql(q, conn, params=params)
            conn.close()
            return df if df is not None and not df.empty else None
        except Exception as ex:
            try:
                conn.close()
            except Exception:
                pass
            st.warning(f"Drilldown error: {ex}")
            return None

    # Analytics chart before the table
    chart_df = df.copy()
    id_cols = [c for c in ["shop_code", "shop_name", "cashier", "loader"] if c in chart_df.columns]
    value_cols = [c for c in chart_df.columns if c not in id_cols]
    numeric_df = chart_df[value_cols].replace('%', '', regex=True).apply(pd.to_numeric, errors="coerce")
    if "%" in selected_metric:
        totals = numeric_df.mean(axis=0).dropna()
    else:
        totals = numeric_df.sum(axis=0).dropna()

    if totals.empty and stage == "🛒 Shop Compliance %":
        trend_df = load_shop_selling_trend(selected_metric, start_s, end_s)
        if trend_df is not None and not trend_df.empty:
            trend_df["date"] = pd.to_datetime(trend_df["date"], errors="coerce")
            trend_df = trend_df.dropna(subset=["date"]).sort_values("date")
            totals = pd.Series(
                trend_df["total"].values,
                index=trend_df["date"].dt.strftime("%Y-%m-%d")
            )

    def _render_card_chart(fig_obj, key=None):
        st.markdown('<div class="drill-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_obj, use_container_width=True, key=key)
        st.markdown('</div>', unsafe_allow_html=True)

    if stage == "🛒 Shop Compliance %" and selected_metric == "Check Serial No":
        st.caption("Using stepwise drilldown (shop → cashier → detail) for faster loading.")

    if not totals.empty:
        chart_data = pd.DataFrame({"date": totals.index, "total": totals.values})
        chart_data["date"] = pd.to_datetime(chart_data["date"], format="%Y-%m-%d", errors="coerce")
        chart_data = chart_data.dropna(subset=["date"]).sort_values("date")

        unique_map = None
        dup_metrics = {"Duplicates", "Dup Serials"}
        if selected_metric in dup_metrics and not chart_data.empty:
            u_start = chart_data["date"].min().date()
            u_end = chart_data["date"].max().date()
            conn_u = get_db_connection()
            if conn_u:
                try:
                    if stage == "📦 WH Loading":
                        u_sql = """
                        SELECT dt_doc_date::DATE AS u_date,
                               COUNT(DISTINCT serial_no) FILTER (
                                   WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
                               ) AS unique_cnt
                        FROM serial_no_dailydata
                        WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
                        GROUP BY dt_doc_date::DATE
                        """
                    elif stage == "🏪 Shop Receiving":
                        u_sql = """
                        SELECT dt_mod_date::DATE AS u_date,
                               COUNT(DISTINCT vc_serail_no) FILTER (
                                   WHERE vc_serail_no IS NOT NULL AND TRIM(COALESCE(vc_serail_no,'')) != ''
                               ) AS unique_cnt
                        FROM serial_no_dailydata
                        WHERE dt_mod_date IS NOT NULL
                          AND dt_mod_date::DATE BETWEEN %(s)s AND %(e)s
                        GROUP BY dt_mod_date::DATE
                        """
                    else:
                        u_sql = """
                        SELECT DATE(bill_date) AS u_date,
                               COUNT(DISTINCT serial_number) FILTER (
                                   WHERE serial_number IS NOT NULL AND TRIM(COALESCE(serial_number,'')) != ''
                               ) AS unique_cnt
                        FROM serialno_check_yes_no
                        WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
                        GROUP BY DATE(bill_date)
                        """
                    u_df = pd.read_sql(u_sql, conn_u, params={"s": u_start, "e": u_end})
                    unique_map = dict(zip(pd.to_datetime(u_df["u_date"]).dt.date, u_df["unique_cnt"]))
                except Exception:
                    unique_map = None
                finally:
                    try:
                        conn_u.close()
                    except Exception:
                        pass

        is_percent_metric = "%" in selected_metric
        if unique_map:
            chart_data["unique_cnt"] = chart_data["date"].dt.date.map(unique_map).fillna(0).astype(int)
            fig = px.line(
                chart_data,
                x="date",
                y="total",
                markers=True,
                custom_data=["unique_cnt"]
            )
            hover_tmpl = (
                "<b>%{x}</b><br>Duplicate Serials: %{y}"
                "<br>Unique Serials: %{customdata[0]}<extra></extra>"
            )
        else:
            fig = px.line(
                chart_data,
                x="date",
                y="total",
                markers=True
            )
            if is_percent_metric:
                hover_tmpl = f"<b>%{{x}}</b><br>{selected_metric}: %{{y:.1f}}%<extra></extra>"
            else:
                hover_tmpl = f"<b>%{{x}}</b><br>{selected_metric}: %{{y}}<extra></extra>"
        fig.update_traces(
            line=dict(color="#5B54FF", width=3, shape="spline"),
            marker=dict(size=10, color="#5B54FF", symbol="diamond",
                        line=dict(color="rgba(255,255,255,0.8)", width=2)),
            hovertemplate=hover_tmpl
        )
        fig.update_layout(
            title=f"{selected_metric} Trend",
            height=250,
            yaxis_title=None,
            xaxis_title=None,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(26,40,71,0.3)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=11, color="#FFFFFF"),
            title_font=dict(size=14, family="Poppins", color="#FFFFFF", weight=600),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="rgba(26,40,71,0.95)", font_size=12,
                            font_family="Inter", font_color="white"),
            xaxis=dict(gridcolor="rgba(91,84,255,0.1)", showgrid=True),
            yaxis=dict(autorange=True, gridcolor="rgba(91,84,255,0.1)", showgrid=True)
        )

        if stage == "🏪 Shop Receiving":
            if selected_metric == "Old Doc Date":
                truck_7d_df = load_truck_7day_summary(end_s)
                if truck_7d_df is not None and not truck_7d_df.empty:
                    truck_7d_df = truck_7d_df.copy()
                    truck_7d_df["date"] = pd.to_datetime(truck_7d_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                    truck_7d_df = truck_7d_df.rename(columns={
                        "date": "date",
                        "wh_trucks_loaded": "WH Trucks Loaded",
                        "shop_trucks_offloaded": "Shop Trucks Offloaded",
                        "offload_pct": "Offload %"
                    })
                    _render_styled_table(
                        truck_7d_df,
                        f"truck_7d_{end_s}",
                        "Last 7 Days Truck Summary",
                        height=270,
                        enable_select=False,
                        enable_filter=False
                    )
                else:
                    st.markdown('<div class="drill-card">', unsafe_allow_html=True)
                    st.info("No truck summary data for last 7 days.")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                cols = st.columns(2)
                with cols[0]:
                    _render_card_chart(fig)

                with cols[1]:
                    pct_df = load_shop_receiving_pct_trend(end_s)
                    if pct_df is not None and not pct_df.empty:
                        pct_df["date"] = pd.to_datetime(pct_df["date"], errors="coerce")
                        pct_df = pct_df.dropna(subset=["date"]).sort_values("date")
                        fig_pct = px.line(pct_df, x="date", y="pct", markers=True)
                        fig_pct.update_traces(
                            line=dict(color="#00D97E", width=3, shape="spline"),
                            marker=dict(size=10, color="#00D97E", symbol="circle",
                                        line=dict(color="rgba(255,255,255,0.8)", width=2)),
                            hovertemplate="<b>%{x}</b><br>(MOD_date present ÷ WH loaded same date): %{y:.1f}%<extra></extra>"
                        )
                        fig_pct.update_layout(
                            title="Shop Same day Received as Dispatched Date",
                            height=250,
                            yaxis_title=None,
                            xaxis_title=None,
                            margin=dict(l=20, r=20, t=40, b=20),
                            plot_bgcolor="rgba(26,40,71,0.3)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter", size=11, color="#FFFFFF"),
                            title_font=dict(size=14, family="Poppins", color="#FFFFFF", weight=600),
                            hovermode="x unified",
                            hoverlabel=dict(bgcolor="rgba(26,40,71,0.95)", font_size=12,
                                            font_family="Inter", font_color="white"),
                            xaxis=dict(gridcolor="rgba(91,84,255,0.1)", showgrid=True),
                            yaxis=dict(autorange=True, gridcolor="rgba(91,84,255,0.1)", showgrid=True)
                        )
                        _render_card_chart(fig_pct)
                    else:
                        truck_shop_df = load_shop_truck_shopwise_for_doc_date(end_s)
                        if truck_shop_df is not None and not truck_shop_df.empty:
                            truck_shop_df = truck_shop_df.rename(columns={
                                "shop_code": "Shop Code",
                                "shop_name": "Shop Name",
                                "wh_trucks_loaded": "WH Trucks Loaded",
                                "wh_qty_loaded": "WH Qty Loaded",
                                "shop_trucks_offloaded": "Shop Trucks Offloaded",
                                "shop_qty_offloaded": "Shop Qty Offloaded",
                            })
                            _render_styled_table(
                                truck_shop_df,
                                f"shop_truck_doc_date_{end_s}",
                                f"Shop-wise Truck Loading/Offloading ({end_s})",
                                height=280,
                                enable_select=False,
                                enable_filter=True
                            )
                        else:
                            st.markdown('<div class="drill-card">', unsafe_allow_html=True)
                            st.info(f"No shop-wise truck loading/offloading data for {end_s}.")
                            st.markdown('</div>', unsafe_allow_html=True)
            
            # Sort Old Doc Date table by yesterday column (descending - high to low)
            if selected_metric == "Old Doc Date" and df is not None and not df.empty:
                date_cols = [c for c in df.columns if c not in ['shop_code', 'shop_name']]
                if date_cols:
                    yesterday_col = sorted(date_cols, reverse=True)[0]  # Latest date (yesterday)
                    # Convert to numeric and sort descending (higher values first)
                    df_sorted = df.copy()
                    df_sorted[yesterday_col] = pd.to_numeric(df_sorted[yesterday_col], errors='coerce')
                    df = df_sorted.sort_values(by=yesterday_col, ascending=False, na_position='last').reset_index(drop=True)

                sel_shop_key = "sr_old_doc_shop"
                sel_date_key = "sr_old_doc_date"
                if st.session_state.get(sel_shop_key):
                    if st.button("← Back to All Shops", key="sr_old_doc_back"):
                        st.session_state.pop(sel_shop_key, None)
                        st.session_state.pop(sel_date_key, None)
                        st.rerun()

                filtered_df = df
                selected_shop = st.session_state.get(sel_shop_key)
                if selected_shop:
                    filtered_df = df[df["shop_code"] == selected_shop].copy()

                shop_sel = _render_styled_table(
                    filtered_df,
                    f"drill_{stage}_{selected_metric}",
                    enable_select=True
                )
                if shop_sel:
                    shop_code = shop_sel["row"].get("shop_code")
                    date_col = shop_sel["col_name"]
                    if shop_code and date_col and date_col not in ["shop_code", "shop_name"]:
                        st.session_state[sel_shop_key] = shop_code
                        st.session_state[sel_date_key] = date_col
                        st.rerun()

                selected_shop = st.session_state.get(sel_shop_key)
                selected_date = st.session_state.get(sel_date_key)
                if selected_shop and selected_date:
                    detail_df = load_shop_receiving_old_doc_detail(start_s, end_s, selected_shop, selected_date)
                    if detail_df is not None and not detail_df.empty:
                        st.markdown(
                            f"**Old Doc Date Detail** — Shop: {selected_shop} · Offloading Date: {selected_date}",
                            unsafe_allow_html=True,
                        )
                        _render_styled_table(
                            detail_df,
                            f"old_doc_detail_{selected_shop}_{selected_date}",
                            "Duplicate/Detail Rows",
                            height=320,
                            enable_select=False,
                            enable_filter=True
                        )
                    else:
                        st.info("No detail rows found for the selected shop/date.")
            else:
                _render_styled_table(df, f"drill_{stage}_{selected_metric}")
        elif stage == "📦 WH Loading":
            shop_df = load_wh_loading_shop_drilldown(start_s, end_s, selected_metric)
            
            compact_cols = st.columns([1.0, 1.2, 1.2])
            
            with compact_cols[0]:
                # Trend chart on top
                _render_card_chart(fig)
                
                # Monthly Summary Table below trend chart (only for Duplicates, Small, IC)
                if selected_metric in ["Duplicates", "Small (≤6)", "IC"]:
                    monthly_df = load_wh_loading_monthly_summary(start_s, end_s, selected_metric)
                    if monthly_df is not None and not monthly_df.empty:
                        # Format the dataframe for horizontal display
                        display_df = monthly_df.copy()
                        
                        # Keep original month (YYYY-MM) for mapping back
                        month_mapping = dict(zip(
                            pd.to_datetime(display_df['month'] + '-01').dt.strftime('%b %Y'),
                            display_df['month']
                        ))
                        
                        display_df['month'] = pd.to_datetime(display_df['month'] + '-01').dt.strftime('%b %Y')
                        
                        # Calculate thresholds for conditional formatting
                        values = display_df['metric_value'].values
                        if len(values) > 0:
                            min_val = values.min()
                            max_val = values.max()
                            
                            # Define thresholds (high = red, medium = yellow, low = green)
                            # For these metrics, higher is BAD (red), lower is GOOD (green)
                            if max_val > min_val:
                                threshold_high = min_val + (max_val - min_val) * 0.66
                                threshold_low = min_val + (max_val - min_val) * 0.33
                            else:
                                threshold_high = max_val
                                threshold_low = min_val
                            
                            # Format values with colored circles
                            def format_with_circle(val):
                                if pd.isna(val):
                                    return ""
                                num = int(val)
                                if val >= threshold_high:
                                    circle = "🔴"  # High = Red (bad)
                                elif val >= threshold_low:
                                    circle = "🟡"  # Medium = Yellow
                                else:
                                    circle = "🟢"  # Low = Green (good)
                                return f"{num} {circle}"
                            
                            display_df['metric_value'] = display_df['metric_value'].apply(format_with_circle)
                        
                        # Transpose: months as columns, metric value as single row
                        display_df = display_df.set_index('month').T
                        display_df.index = [selected_metric]  # Set row label as metric name
                        
                        st.markdown('<div class="drill-card" style="margin-top: 0.25rem; padding: 0.5rem;">', unsafe_allow_html=True)
                        st.markdown(f'<h4 class="drill-table-title" style="font-size: 0.75rem; margin: 0 0 0.25rem 0; padding: 0;">Monthly {selected_metric}</h4>', unsafe_allow_html=True)
                        
                        monthly_display_df = display_df.copy()
                        monthly_display_df.columns = [re.sub(r"[\x00-\x1f\x7f]", " ", str(c)).strip() for c in monthly_display_df.columns]
                        monthly_display_df.index = [re.sub(r"[\x00-\x1f\x7f]", " ", str(i)).strip() for i in monthly_display_df.index]
                        monthly_display_df = monthly_display_df.applymap(
                            lambda v: re.sub(r"[\x00-\x1f\x7f]", " ", str(v)).strip() if pd.notna(v) else ""
                        )

                        monthly_event = st.dataframe(
                            monthly_display_df,
                            use_container_width=True,
                            hide_index=False,
                            height=100,
                            key=f"monthly_{selected_metric}",
                            on_select="rerun",
                            selection_mode="single-cell"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Store monthly selection in session state for rendering below
                        if monthly_event and hasattr(monthly_event, "selection"):
                            selection = monthly_event.selection
                            cells = getattr(selection, "cells", []) or []
                            if cells:
                                row_idx, col_name = cells[0]
                                # col_name is the month (e.g., "Feb 2026")
                                if col_name in month_mapping:
                                    st.session_state[f'monthly_selected_{selected_metric}'] = {
                                        'month_display': col_name,
                                        'month_yyyymm': month_mapping[col_name]
                                    }
                                    # Clear loader/shop selections when monthly is clicked
                                    prev_loader_key = f"wh_prev_loader_sel_{selected_metric}"
                                    prev_shop_key = f"wh_prev_shop_sel_{selected_metric}"
                                    scope_state_key = f"wh_active_scope_{selected_metric}"
                                    if prev_loader_key in st.session_state:
                                        del st.session_state[prev_loader_key]
                                    if prev_shop_key in st.session_state:
                                        del st.session_state[prev_shop_key]
                                    if scope_state_key in st.session_state:
                                        del st.session_state[scope_state_key]
            
            with compact_cols[1]:
                loader_sel = _render_styled_table(
                    df,
                    f"drill_loader_{stage}_{selected_metric}",
                    f"Loader ({start_s} to {end_s})",
                    height=360,
                    enable_select=True
                )
            with compact_cols[2]:
                if shop_df is not None and not shop_df.empty:
                    shop_sel = _render_styled_table(
                        shop_df,
                        f"drill_shopwise_{stage}_{selected_metric}",
                        f"Shop Code ({start_s} to {end_s})",
                        height=360,
                        enable_select=True
                    )
                else:
                    shop_sel = None
                    st.markdown('<div class="drill-card">', unsafe_allow_html=True)
                    st.info("No shop-wise data available.")
                    st.markdown('</div>', unsafe_allow_html=True)

            def _sel_sig(sel_obj, id_key):
                if not sel_obj:
                    return None
                row = sel_obj.get("row", {}) or {}
                return (sel_obj.get("row_idx"), sel_obj.get("col_name"), str(row.get(id_key, "")))

            scope_state_key = f"wh_active_scope_{selected_metric}"
            prev_loader_key = f"wh_prev_loader_sel_{selected_metric}"
            prev_shop_key = f"wh_prev_shop_sel_{selected_metric}"

            loader_sig = _sel_sig(loader_sel, "loader")
            shop_sig = _sel_sig(shop_sel, "shop_code")
            prev_loader_sig = st.session_state.get(prev_loader_key)
            prev_shop_sig = st.session_state.get(prev_shop_key)

            loader_changed = loader_sig is not None and loader_sig != prev_loader_sig
            shop_changed = shop_sig is not None and shop_sig != prev_shop_sig

            # Clear monthly selection if loader or shop table is clicked
            monthly_sel_key = f'monthly_selected_{selected_metric}'
            if loader_changed or shop_changed:
                if monthly_sel_key in st.session_state:
                    del st.session_state[monthly_sel_key]

            if shop_changed and not loader_changed:
                st.session_state[scope_state_key] = "shop"
            elif loader_changed and not shop_changed:
                st.session_state[scope_state_key] = "loader"
            elif shop_changed and loader_changed:
                st.session_state[scope_state_key] = "shop"

            if loader_sig is not None:
                st.session_state[prev_loader_key] = loader_sig
            if shop_sig is not None:
                st.session_state[prev_shop_key] = shop_sig

            active_scope = st.session_state.get(scope_state_key)
            active_sel = None
            if active_scope == "shop" and shop_sel:
                active_sel = shop_sel
            elif active_scope == "loader" and loader_sel:
                active_sel = loader_sel
            elif shop_sel:
                active_scope = "shop"
                active_sel = shop_sel
            elif loader_sel:
                active_scope = "loader"
                active_sel = loader_sel

            if active_sel:
                row = active_sel["row"]
                col_name = active_sel["col_name"]
                loader_val = row.get("loader") if active_scope == "loader" else None
                shop_val = row.get("shop_code") if active_scope == "shop" else None

                if col_name in ("loader", "shop_code"):
                    date_str = None
                    scope_label = f"full range ({start_s} to {end_s})"
                else:
                    date_str = str(col_name)
                    scope_label = date_str

                detail_df = load_wh_loading_detail(
                    selected_metric,
                    loader=loader_val,
                    shop_code=shop_val,
                    date_str=date_str,
                    start_str=start_s,
                    end_str=end_s,
                )

                if detail_df is not None and not detail_df.empty:
                    st.markdown(
                        f"**WH Loading Drilldown** — metric: `{selected_metric}`, scope: {scope_label}",
                        unsafe_allow_html=True,
                    )
                    st.dataframe(detail_df, use_container_width=True, hide_index=True, height=320)

                    csv_data = detail_df.to_csv(index=False)
                    b64 = base64.b64encode(csv_data.encode("utf-8")).decode("utf-8")
                    st.markdown(
                        f"<div style='display:flex;justify-content:flex-end;margin:4px 0 8px;'>"
                        f"<a href='data:text/csv;base64,{b64}' download='wh_loading_drilldown.csv' "
                        f"style='font-size:0.72rem;color:#93c5fd;text-decoration:underline;'>Download CSV</a>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No drilldown rows for the selected cell.")
            
            # Handle monthly drill-down if a month was selected
            monthly_sel_key = f'monthly_selected_{selected_metric}'
            if monthly_sel_key in st.session_state and st.session_state[monthly_sel_key]:
                monthly_sel = st.session_state[monthly_sel_key]
                month_display = monthly_sel['month_display']
                selected_month = monthly_sel['month_yyyymm']
                
                # Calculate month start and end dates
                month_start = pd.to_datetime(selected_month + '-01')
                month_end = (month_start + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
                month_start_str = month_start.strftime('%Y-%m-%d')
                
                # Load detail for this month
                monthly_detail_df = load_wh_loading_detail(
                    selected_metric,
                    loader=None,
                    shop_code=None,
                    date_str=None,
                    start_str=month_start_str,
                    end_str=month_end
                )
                
                if monthly_detail_df is not None and not monthly_detail_df.empty:
                    st.markdown(
                        f"**Monthly Drilldown** — {month_display} ({selected_metric})",
                        unsafe_allow_html=True,
                    )
                    st.dataframe(monthly_detail_df, use_container_width=True, hide_index=True, height=320)
                    
                    csv_data = monthly_detail_df.to_csv(index=False)
                    b64 = base64.b64encode(csv_data.encode("utf-8")).decode("utf-8")
                    st.markdown(
                        f"<div style='display:flex;justify-content:flex-end;margin:4px 0 8px;'>"
                        f"<a href='data:text/csv;base64,{b64}' download='monthly_drilldown_{month_display.replace(' ', '_')}.csv' "
                        f"style='font-size:0.72rem;color:#93c5fd;text-decoration:underline;'>Download CSV</a>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info(f"No detail records found for {month_display}.")
        elif stage == "🛒 Shop Compliance %":
            flow_start = start_s
            flow_end = end_s

            flow_key = f"ss_flow_{selected_metric}"
            if st.session_state.get(flow_key) != selected_metric:
                st.session_state["ss_step"] = "shop"
                st.session_state["ss_shop"] = None
                st.session_state["ss_cashier"] = None
                st.session_state["ss_date"] = None
                st.session_state[flow_key] = selected_metric

            step = st.session_state.get("ss_step", "shop")

            if step == "cashier":
                if st.button("← Back to Shops", key=f"ss_back_shop_{selected_metric}"):
                    st.session_state["ss_step"] = "shop"
                    st.session_state["ss_cashier"] = None
                    st.session_state["ss_date"] = None
                    st.rerun()
            elif step in ("detail", "detail_range"):
                if st.button("← Back to Cashiers", key=f"ss_back_cashier_{selected_metric}"):
                    st.session_state["ss_step"] = "cashier"
                    st.session_state["ss_date"] = None
                    st.rerun()

            if step == "shop":
                shop_df = load_shop_selling_shop_metric(selected_metric, flow_start, flow_end)
                if shop_df is None or shop_df.empty:
                    st.info("No shop data for selected metric.")
                else:
                    shop_sel = _render_styled_table(
                        shop_df,
                        f"ss_shop_{selected_metric}_{flow_start}_{flow_end}",
                        f"Shop Code — {selected_metric} ({flow_start} to {flow_end})",
                        height=320,
                        enable_select=True,
                        enable_filter=(selected_metric != "Compliance %"),
                        no_conditional_cols=(
                            ["total_serial_numbers", "serial_match_y"]
                            if selected_metric == "Compliance %"
                            else None
                        ),
                    )
                    if shop_sel:
                        st.session_state["ss_shop"] = shop_sel["row"].get("shop_code")
                        st.session_state["ss_step"] = "cashier"
                        st.rerun()

            elif step == "cashier":
                selected_shop = st.session_state.get("ss_shop")
                cashier_df = load_shop_selling_cashier_metric(selected_shop, selected_metric, flow_start, flow_end)
                if cashier_df is None or cashier_df.empty:
                    st.info("No cashier data for selected shop.")
                else:
                    cashier_sel = _render_styled_table(
                        cashier_df,
                        f"ss_cashier_{selected_shop}_{selected_metric}_{flow_start}_{flow_end}",
                        f"Cashier — {selected_metric} ({flow_start} to {flow_end}) — {selected_shop}",
                        height=360,
                        enable_select=True,
                        enable_filter=(selected_metric != "Compliance %")
                    )
                    if cashier_sel:
                        cashier_name = cashier_sel["row"].get("cashier")
                        date_col = cashier_sel["col_name"]
                        if selected_metric == "Dup Serials" and cashier_name:
                            st.session_state["ss_cashier"] = cashier_name
                            st.session_state["ss_step"] = "detail_range"
                            st.rerun()
                        elif cashier_name and date_col and date_col != "cashier":
                            st.session_state["ss_cashier"] = cashier_name
                            st.session_state["ss_date"] = date_col
                            st.session_state["ss_step"] = "detail"
                            st.rerun()

            elif step == "detail":
                selected_shop = st.session_state.get("ss_shop")
                cashier_name = st.session_state.get("ss_cashier")
                date_col = st.session_state.get("ss_date")
                
                detail_df = load_shop_selling_metric_detail(selected_shop, cashier_name, date_col, selected_metric)
                
                if detail_df is None or detail_df.empty:
                    st.info("No serial detail for selected cashier/date.")
                else:
                    st.markdown(
                        f"**Serial Detail** — {selected_shop} · {cashier_name} · {date_col} · {selected_metric}",
                        unsafe_allow_html=True,
                    )
                    # For Compliance %, show summary before detail
                    if selected_metric == "Compliance %":
                        if "serial_check" in detail_df.columns:
                            verified_count = (detail_df["serial_check"] == "Y").sum()
                            total_count = len(detail_df)
                            compliance_pct = (verified_count / total_count * 100) if total_count > 0 else 0
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Verified", f"{verified_count}")
                            with col2:
                                st.metric("Total", f"{total_count}")
                            with col3:
                                st.metric("Compliance %", f"{compliance_pct:.1f}%")
                    _render_styled_table(
                        detail_df,
                        f"ss_detail_{selected_shop}_{cashier_name}_{date_col}",
                        "Serial Detail",
                        height=360,
                        enable_select=False
                    )
            elif step == "detail_range":
                selected_shop = st.session_state.get("ss_shop")
                cashier_name = st.session_state.get("ss_cashier")
                detail_df = load_shop_selling_metric_detail_range(selected_shop, cashier_name, flow_start, flow_end, selected_metric)
                if detail_df is None or detail_df.empty:
                    st.info("No serial detail for selected cashier in selected range.")
                else:
                    st.markdown(
                        f"**Serial Detail** — {selected_shop} · {cashier_name} · {flow_start} to {flow_end} · {selected_metric}",
                        unsafe_allow_html=True,
                    )
                    _render_styled_table(
                        detail_df,
                        f"ss_detail_range_{selected_shop}_{cashier_name}_{flow_start}_{flow_end}",
                        "Serial Detail",
                        height=360,
                        enable_select=False
                    )
        else:
            _render_card_chart(fig)
            _render_styled_table(df, f"drill_{stage}_{selected_metric}")
    else:
        st.info("No numeric data available for the trend chart.")



# ─────────────────────────────────────────────────────────────
# SLAB TREND TABLE
# ─────────────────────────────────────────────────────────────
SLAB_COLS = ["Before WH Loaded", "0 Days", "1-3 Days", "4-7 Days", "8-10 Days", ">10 Days"]
SLAB_SQL  = {
    "Before WH Loaded": "(dt_mod_date::DATE - dt_doc_date::DATE) < 0",
    "0 Days":           "(dt_mod_date::DATE - dt_doc_date::DATE) = 0",
    "1-3 Days":         "(dt_mod_date::DATE - dt_doc_date::DATE) BETWEEN 1 AND 3",
    "4-7 Days":         "(dt_mod_date::DATE - dt_doc_date::DATE) BETWEEN 4 AND 7",
    "8-10 Days":        "(dt_mod_date::DATE - dt_doc_date::DATE) BETWEEN 8 AND 10",
    ">10 Days":         "(dt_mod_date::DATE - dt_doc_date::DATE) > 10",
}


def _compliance_cell_style(val):
    """
    Green → Amber → Red background gradient.
    Text colour = darkest shade of the SAME hue as the background
    (background RGB × 0.28), so the text is always the darkest tint
    of whatever colour the cell shows.
    """
    try:
        v = float(str(val).replace('%', '').strip())
    except (ValueError, TypeError):
        return ''
    if v != v or not (0 <= v <= 100):   # NaN / Inf guard
        return ''

    # ── Determine background RGB ──────────────────────────────
    if v >= 95:
        bg_r, bg_g, bg_b, bg_a = 4, 120, 87, 0.55
    elif v >= 85:
        ratio  = (v - 85) / 10
        bg_r   = int(10  + 20  * (1 - ratio))
        bg_g   = int(140 + 35  * ratio)
        bg_b   = int(60  + 27  * ratio)
        bg_a   = 0.45
    elif v >= 70:
        ratio  = (v - 70) / 15
        bg_r   = int(200 * (1 - ratio) + 10 * ratio)
        bg_g   = int(130 * (1 - ratio) + 120 * ratio)
        bg_b   = 10
        bg_a   = 0.42
    else:
        ratio  = min(v / 70, 1.0)
        bg_r   = int(220 - 20 * ratio)
        bg_g   = int(40  + 60 * ratio)
        bg_b   = 20
        bg_a   = 0.48

    # ── Text = darkest visible shade of the same hue (55 % brightness) ──
    # 55% keeps it clearly green / amber / red (not near-black like 28%)
    # while still providing good contrast against the semi-transparent bg.
    tx_r = max(4, int(bg_r * 0.55))
    tx_g = max(4, int(bg_g * 0.55))
    tx_b = max(4, int(bg_b * 0.55))

    return (f'background-color: rgba({bg_r},{bg_g},{bg_b},{bg_a}); '
            f'color: rgb({tx_r},{tx_g},{tx_b}); '
            f'font-weight: 800')


def load_shop_compliance_pivot(end_str: str):
    """
    Returns a pivot: rows = shop_code, columns = month labels (Jan 2026 …),
    values = compliance % (numeric float).
    """
    conn = get_db_connection()
    if not conn:
        return None
    try:
        end_dt    = pd.to_datetime(end_str, errors='coerce')
        if pd.isna(end_dt):
            return None
        ytd_start = end_dt.replace(month=1, day=1).strftime('%Y-%m-%d')

        q = """
        SELECT
            COALESCE(NULLIF(TRIM(shop_code), ''), 'Unknown')      AS shop_code,
            DATE_TRUNC('month', DATE(bill_date))::DATE            AS month_start,
            ROUND(
                COUNT(*) FILTER (WHERE TRIM(serial_check) = 'Y')::NUMERIC
                / NULLIF(COUNT(*), 0) * 100, 1
            )                                                     AS compliance_pct
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
          AND COALESCE(item_code, '') != 'sales data not available'
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
        df = pd.read_sql(q, conn, params={'s': ytd_start, 'e': end_str})
        conn.close()
        if df is None or df.empty:
            return None

        df['month_label'] = pd.to_datetime(df['month_start']).dt.strftime('%b %Y')
        pivot = df.pivot(index='shop_code', columns='month_label', values='compliance_pct')

        # Sort columns chronologically
        month_order = pd.to_datetime(
            df['month_label'].unique(), format='%b %Y'
        ).sort_values()
        ordered_cols = [d.strftime('%b %Y') for d in month_order if d.strftime('%b %Y') in pivot.columns]
        pivot = pivot[ordered_cols]
        pivot.index.name = 'Shop'
        return pivot.reset_index()
    except Exception:
        try: conn.close()
        except Exception: pass
        return None


# ═══════════════════════════════════════════════════════════════════
# EDA — DATA LOADERS
# ═══════════════════════════════════════════════════════════════════

_DARK_BG   = "rgba(15,20,45,0.55)"
_GRID_CLR  = "rgba(148,163,184,0.12)"
_FONT_CLR  = "#e2e8f0"
_PLOT_FONT = dict(family="Inter", size=11, color=_FONT_CLR)
_LAYOUT    = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=_DARK_BG,
    font=_PLOT_FONT,
    margin=dict(l=10, r=10, t=36, b=10),
    hoverlabel=dict(bgcolor="rgba(15,20,45,0.95)", font_color="white",
                    font_family="Inter", font_size=12),
)


@st.cache_data(ttl=300)
def eda_shop_ranking(start_s: str, end_s: str):
    conn = get_db_connection()
    if not conn: return None
    q = """
        SELECT
            COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
            COUNT(*)                                        AS total_serials,
            SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END) AS total_yes,
            ROUND(SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END)
                  ::NUMERIC / NULLIF(COUNT(*),0)*100,1)    AS compliance_pct
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
          AND COALESCE(item_code,'') != 'sales data not available'
        GROUP BY 1
        ORDER BY compliance_pct ASC NULLS FIRST
    """
    try:
        df = pd.read_sql(q, conn, params={'s': start_s, 'e': end_s})
        conn.close(); return df
    except Exception:
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def eda_cashier_ranking(start_s: str, end_s: str, min_serials: int = 10):
    conn = get_db_connection()
    if not conn: return None
    q = """
        SELECT
            COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown') AS cashier,
            COALESCE(NULLIF(TRIM(shop_code),''),'?')           AS shop_code,
            COUNT(*)                                           AS total_serials,
            SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END) AS total_yes,
            ROUND(SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END)
                  ::NUMERIC / NULLIF(COUNT(*),0)*100,1)        AS compliance_pct
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
          AND cashier_name IS NOT NULL AND TRIM(cashier_name) <> ''
          AND COALESCE(item_code,'') != 'sales data not available'
        GROUP BY 1, 2
        HAVING COUNT(*) >= %(min)s
        ORDER BY compliance_pct ASC NULLS FIRST
    """
    try:
        df = pd.read_sql(q, conn, params={'s': start_s, 'e': end_s, 'min': min_serials})
        conn.close(); return df
    except Exception:
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def eda_shop_date_heatmap(days: int = 14):
    conn = get_db_connection()
    if not conn: return None
    cutoff = (datetime.today().date() - timedelta(days=days)).strftime('%Y-%m-%d')
    q = """
        SELECT
            COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
            DATE(bill_date)                                 AS activity_date,
            COUNT(*)                                        AS total_serials,
            ROUND(SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END)
                  ::NUMERIC / NULLIF(COUNT(*),0)*100,1)    AS compliance_pct
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) >= %(c)s
          AND COALESCE(item_code,'') != 'sales data not available'
        GROUP BY 1, 2
        ORDER BY 2, 1
    """
    try:
        df = pd.read_sql(q, conn, params={'c': cutoff})
        conn.close(); return df
    except Exception:
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def eda_dow_pattern(start_s: str, end_s: str):
    conn = get_db_connection()
    if not conn: return None
    q = """
        SELECT
            EXTRACT(DOW FROM DATE(bill_date))::INT    AS dow,
            TO_CHAR(DATE(bill_date), 'Dy')            AS day_abbr,
            COUNT(*)                                   AS total_serials,
            ROUND(SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END)
                  ::NUMERIC / NULLIF(COUNT(*),0)*100,1) AS compliance_pct
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
          AND COALESCE(item_code,'') != 'sales data not available'
        GROUP BY 1, 2
        ORDER BY 1
    """
    try:
        df = pd.read_sql(q, conn, params={'s': start_s, 'e': end_s})
        conn.close(); return df
    except Exception:
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def eda_weekly_trend(weeks: int = 10):
    conn = get_db_connection()
    if not conn: return None
    cutoff = (datetime.today().date() - timedelta(weeks=weeks)).strftime('%Y-%m-%d')
    q = """
        SELECT
            DATE_TRUNC('week', DATE(bill_date))::DATE  AS week_start,
            COUNT(*)                                    AS total_serials,
            SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END) AS total_yes,
            ROUND(SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END)
                  ::NUMERIC / NULLIF(COUNT(*),0)*100,1) AS compliance_pct
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) >= %(c)s
          AND COALESCE(item_code,'') != 'sales data not available'
        GROUP BY 1
        ORDER BY 1
    """
    try:
        df = pd.read_sql(q, conn, params={'c': cutoff})
        conn.close(); return df
    except Exception:
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def eda_issue_anatomy(start_s: str, end_s: str):
    conn = get_db_connection()
    if not conn: return None
    q = """
        SELECT
            SUM(CASE WHEN UPPER(TRIM(serial_check))='Y'  THEN 1 ELSE 0 END) AS compliant,
            SUM(CASE WHEN UPPER(TRIM(serial_check))='N'
                      AND (serial_number IS NULL OR TRIM(serial_number)='') THEN 1 ELSE 0 END) AS no_serial,
            SUM(CASE WHEN UPPER(TRIM(serial_check))='N'
                      AND serial_number IS NOT NULL AND TRIM(serial_number)<>''
                      AND LENGTH(TRIM(serial_number)) <= 6                  THEN 1 ELSE 0 END) AS small_serial,
            SUM(CASE WHEN UPPER(TRIM(serial_check))='N'
                      AND serial_number IS NOT NULL AND TRIM(serial_number)<>''
                      AND LENGTH(TRIM(serial_number)) > 6                   THEN 1 ELSE 0 END) AS not_in_wh,
            COUNT(*) AS grand_total
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
          AND COALESCE(item_code,'') != 'sales data not available'
    """
    try:
        df = pd.read_sql(q, conn, params={'s': start_s, 'e': end_s})
        conn.close()
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception:
        try: conn.close()
        except: pass
        return {}


@st.cache_data(ttl=300)
def eda_issue_daily_trend(days: int = 14):
    conn = get_db_connection()
    if not conn: return None
    cutoff = (datetime.today().date() - timedelta(days=days)).strftime('%Y-%m-%d')
    q = """
        SELECT
            DATE(bill_date)                                                          AS dt,
            SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END)          AS compliant,
            SUM(CASE WHEN UPPER(TRIM(serial_check))='N'
                      AND (serial_number IS NULL OR TRIM(serial_number)='') THEN 1 ELSE 0 END) AS no_serial,
            SUM(CASE WHEN UPPER(TRIM(serial_check))='N'
                      AND serial_number IS NOT NULL AND TRIM(serial_number)<>''
                      AND LENGTH(TRIM(serial_number)) <= 6 THEN 1 ELSE 0 END)       AS small_serial,
            SUM(CASE WHEN UPPER(TRIM(serial_check))='N'
                      AND serial_number IS NOT NULL AND TRIM(serial_number)<>''
                      AND LENGTH(TRIM(serial_number)) > 6  THEN 1 ELSE 0 END)       AS not_in_wh
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) >= %(c)s
          AND COALESCE(item_code,'') != 'sales data not available'
        GROUP BY 1
        ORDER BY 1
    """
    try:
        df = pd.read_sql(q, conn, params={'c': cutoff})
        conn.close(); return df
    except Exception:
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def eda_dept_grp_summary(start_s: str, end_s: str):
    conn = get_db_connection()
    if not conn: return None
    q = """
        SELECT dept, grp,
            SUM(total_serials) AS total_serials,
            SUM(total_yes)     AS total_yes,
            ROUND(SUM(total_yes)::NUMERIC/NULLIF(SUM(total_serials),0)*100,1) AS compliance_pct
        FROM mv_dept_compliance
        WHERE activity_date BETWEEN %(s)s AND %(e)s
          AND dept <> 'Unclassified'
        GROUP BY dept, grp
        ORDER BY compliance_pct ASC NULLS FIRST
    """
    try:
        df = pd.read_sql(q, conn, params={'s': start_s, 'e': end_s})
        conn.close(); return df
    except Exception:
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def eda_cashier_daily_trend(days: int = 30):
    """Daily compliance % per cashier — used for trend / volatility / anomaly scoring."""
    conn = get_db_connection()
    if not conn: return None
    cutoff = (datetime.today().date() - timedelta(days=days)).strftime('%Y-%m-%d')
    q = """
        SELECT
            COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown') AS cashier,
            COALESCE(NULLIF(TRIM(shop_code),''),'?')           AS shop_code,
            DATE(bill_date)                                    AS dt,
            COUNT(*)                                           AS total_scans,
            SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END) AS yes_scans,
            ROUND(SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END)
                  ::NUMERIC / NULLIF(COUNT(*),0)*100,1)        AS compliance_pct
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) >= %(c)s
          AND cashier_name IS NOT NULL AND TRIM(cashier_name) <> ''
          AND COALESCE(item_code,'') != 'sales data not available'
        GROUP BY 1, 2, 3
        HAVING COUNT(*) >= 3
        ORDER BY cashier, dt
    """
    try:
        df = pd.read_sql(q, conn, params={'c': cutoff})
        conn.close(); return df
    except Exception:
        try: conn.close()
        except: pass
        return None


@st.cache_data(ttl=300)
def eda_cashier_week_comparison():
    """This week vs prior week compliance per cashier — for delta / declining trend detection."""
    conn = get_db_connection()
    if not conn: return None
    q = """
        SELECT
            COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown') AS cashier,
            COALESCE(NULLIF(TRIM(shop_code),''),'?')           AS shop_code,
            SUM(CASE WHEN DATE(bill_date) >= CURRENT_DATE - 7  THEN 1 ELSE 0 END)                                          AS rec_total,
            SUM(CASE WHEN DATE(bill_date) >= CURRENT_DATE - 7
                      AND UPPER(TRIM(serial_check))='Y'        THEN 1 ELSE 0 END)                                          AS rec_yes,
            SUM(CASE WHEN DATE(bill_date) BETWEEN CURRENT_DATE-14 AND CURRENT_DATE-8 THEN 1 ELSE 0 END)                    AS prev_total,
            SUM(CASE WHEN DATE(bill_date) BETWEEN CURRENT_DATE-14 AND CURRENT_DATE-8
                      AND UPPER(TRIM(serial_check))='Y'        THEN 1 ELSE 0 END)                                          AS prev_yes,
            COUNT(DISTINCT DATE(bill_date))                                                                                 AS active_days,
            COUNT(*)                                                                                                        AS total_scans_30d
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) >= CURRENT_DATE - 14
          AND cashier_name IS NOT NULL AND TRIM(cashier_name) <> ''
          AND COALESCE(item_code,'') != 'sales data not available'
        GROUP BY 1, 2
        HAVING SUM(CASE WHEN DATE(bill_date) >= CURRENT_DATE - 7 THEN 1 ELSE 0 END) >= 5
        ORDER BY cashier
    """
    try:
        df = pd.read_sql(q, conn, params={})
        conn.close(); return df
    except Exception:
        try: conn.close()
        except: pass
        return None


# ═══════════════════════════════════════════════════════════════════
# EDA — RENDER
# ═══════════════════════════════════════════════════════════════════

def _eda_kpi_card(label, value, sub, color="#9b5bff", alert=False):
    border = "#ef4444" if alert else color
    bg = "rgba(239,68,68,0.08)" if alert else "rgba(26,32,60,0.6)"
    return f"""
    <div style="background:{bg};border:1px solid {border}30;border-radius:12px;
                padding:16px 14px;text-align:center;min-height:96px;">
      <div style="font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;
                  color:#94a3b8;font-weight:700;margin-bottom:6px;">{label}</div>
      <div style="font-size:1.85rem;font-weight:900;color:{border};
                  line-height:1.1;font-variant-numeric:tabular-nums;">{value}</div>
      <div style="font-size:0.65rem;color:#64748b;margin-top:4px;">{sub}</div>
    </div>"""


def render_eda_intelligence(start_s: str, end_s: str):
    """20-year analyst EDA: shop / cashier / time / issue / dept intelligence."""

    st.markdown(
        '<div style="font-size:0.72rem;font-weight:800;color:#7c86a1;letter-spacing:0.12em;'
        'text-transform:uppercase;margin:28px 0 4px;">Compliance Intelligence Center — EDA</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:0.68rem;color:#475569;margin-bottom:10px;">'
        f'Date window: <b style="color:#94a3b8">{start_s}</b> → <b style="color:#94a3b8">{end_s}</b> '
        f'| Trend charts use rolling last 14 days</div>',
        unsafe_allow_html=True,
    )

    tab_health, tab_shop, tab_cashier, tab_time, tab_issue, tab_dept = st.tabs([
        "🏥 Health", "🏪 Shops", "👥 Cashiers",
        "📅 Time Patterns", "🔬 Issues", "📦 Dept/Group"
    ])

    # ── load data (shared across tabs) ─────────────────────────
    shop_df    = eda_shop_ranking(start_s, end_s)
    cashier_df = eda_cashier_ranking(start_s, end_s)
    anatomy    = eda_issue_anatomy(start_s, end_s)
    weekly_df  = eda_weekly_trend(10)
    dept_df    = eda_dept_grp_summary(start_s, end_s)

    # ── helpers ─────────────────────────────────────────────────
    def pct_color(v):
        if v is None: return "#64748b"
        if v >= 90: return "#10b981"
        if v >= 75: return "#f59e0b"
        return "#ef4444"

    # ══════════════════════════════════════════════════════════
    # TAB 1 — HEALTH CHECK
    # ══════════════════════════════════════════════════════════
    with tab_health:
        grand_total = int(anatomy.get('grand_total') or 0)
        compliant   = int(anatomy.get('compliant')   or 0)
        overall_pct = round(compliant / grand_total * 100, 1) if grand_total else 0

        shops_below_80  = int((shop_df['compliance_pct'] < 80).sum()) if shop_df is not None and not shop_df.empty else 0
        shops_at_100    = int((shop_df['compliance_pct'] >= 99.9).sum()) if shop_df is not None and not shop_df.empty else 0
        total_shops     = len(shop_df) if shop_df is not None else 0
        cashiers_below50 = int((cashier_df['compliance_pct'] < 50).sum()) if cashier_df is not None and not cashier_df.empty else 0

        # Week-over-week delta
        wow_delta = None
        if weekly_df is not None and len(weekly_df) >= 2:
            wow_delta = float(weekly_df.iloc[-1]['compliance_pct']) - float(weekly_df.iloc[-2]['compliance_pct'])

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5, gap="small")
        kpis = [
            (c1, "Overall Compliance", f"{overall_pct:.1f}%",
             f"{compliant:,} / {grand_total:,} serials",
             pct_color(overall_pct), overall_pct < 80),
            (c2, "Shops Below 80%", str(shops_below_80),
             f"of {total_shops} active shops",
             "#ef4444" if shops_below_80 > 0 else "#10b981", shops_below_80 > 3),
            (c3, "Shops at 100%", str(shops_at_100),
             "perfect compliance",
             "#10b981", False),
            (c4, "Cashiers < 50%", str(cashiers_below50),
             "high-risk cashiers (≥10 scans)",
             "#ef4444" if cashiers_below50 > 0 else "#10b981", cashiers_below50 > 0),
            (c5, "Week-on-Week",
             (f"+{wow_delta:.1f}%" if wow_delta and wow_delta >= 0 else f"{wow_delta:.1f}%") if wow_delta is not None else "N/A",
             "vs prior week",
             "#10b981" if wow_delta and wow_delta >= 0 else "#ef4444",
             wow_delta is not None and wow_delta < -2),
        ]
        for col, lbl, val, sub, clr, alrt in kpis:
            with col:
                st.markdown(_eda_kpi_card(lbl, val, sub, clr, alrt), unsafe_allow_html=True)

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

        # Gauge + weekly trend side by side
        g1, g2 = st.columns([1, 2], gap="large")

        with g1:
            # Build "vs prior week" label with actual dates
            _prior_week_lbl = ""
            _cur_week_lbl   = ""
            if weekly_df is not None and len(weekly_df) >= 2:
                try:
                    _cur_ws  = pd.Timestamp(weekly_df.iloc[-1]['week_start'])
                    _prev_ws = pd.Timestamp(weekly_df.iloc[-2]['week_start'])
                    _cur_week_lbl  = f"Week of {_cur_ws.strftime('%d %b')}"
                    _prior_week_lbl= f"vs week of {_prev_ws.strftime('%d %b')}"
                except Exception:
                    _prior_week_lbl = "vs prior calendar week"
            st.markdown(
                f'<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">'
                f'Compliance Score</div>'
                f'<div style="font-size:0.60rem;color:#475569;margin-bottom:4px;">'
                f'{_cur_week_lbl} &nbsp;·&nbsp; delta {_prior_week_lbl}</div>',
                unsafe_allow_html=True,
            )
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=overall_pct,
                delta={'reference': float(weekly_df.iloc[-2]['compliance_pct']) if weekly_df is not None and len(weekly_df) >= 2 else overall_pct,
                       'valueformat': '.1f', 'suffix': '%',
                       'increasing': {'color': '#10b981'},
                       'decreasing': {'color': '#ef4444'}},
                number={'suffix': '%', 'font': {'size': 38, 'color': pct_color(overall_pct), 'family': 'Inter'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#475569',
                             'tickfont': {'color': '#94a3b8', 'size': 9}},
                    'bar': {'color': pct_color(overall_pct), 'thickness': 0.28},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 75],  'color': 'rgba(239,68,68,0.12)'},
                        {'range': [75, 90], 'color': 'rgba(245,158,11,0.12)'},
                        {'range': [90, 100],'color': 'rgba(16,185,129,0.12)'},
                    ],
                    'threshold': {'line': {'color': '#f8fafc', 'width': 2}, 'value': 90, 'thickness': 0.75},
                },
            ))
            gauge.update_layout(**{**_LAYOUT, 'height': 230, 'margin': dict(l=16, r=16, t=20, b=10)})
            st.plotly_chart(gauge, use_container_width=True, key="eda_gauge")

            # Alert box
            if overall_pct < 75:
                st.markdown('<div style="background:rgba(239,68,68,0.12);border:1px solid #ef444440;'
                            'border-radius:8px;padding:10px 12px;font-size:0.70rem;color:#fca5a5;">'
                            '<b>CRITICAL:</b> Compliance below 75%. Immediate investigation required.</div>',
                            unsafe_allow_html=True)
            elif overall_pct < 90:
                st.markdown('<div style="background:rgba(245,158,11,0.10);border:1px solid #f59e0b30;'
                            'border-radius:8px;padding:10px 12px;font-size:0.70rem;color:#fcd34d;">'
                            '<b>ATTENTION:</b> Compliance below target (90%). Review flagged shops.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div style="background:rgba(16,185,129,0.10);border:1px solid #10b98130;'
                            'border-radius:8px;padding:10px 12px;font-size:0.70rem;color:#6ee7b7;">'
                            '<b>ON TRACK:</b> Compliance above 90% threshold. Keep monitoring.</div>',
                            unsafe_allow_html=True)

        with g2:
            st.markdown('<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;'
                        'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">'
                        'Weekly Compliance Trend (10 Weeks)</div>', unsafe_allow_html=True)
            if weekly_df is not None and not weekly_df.empty:
                weekly_df = weekly_df.copy()
                weekly_df['compliance_pct'] = pd.to_numeric(weekly_df['compliance_pct'], errors='coerce')
                weekly_df['week_label'] = pd.to_datetime(weekly_df['week_start']).dt.strftime('%d %b')
                clrs = [pct_color(v) for v in weekly_df['compliance_pct']]
                fig_wk = go.Figure()
                fig_wk.add_hrect(y0=90, y1=100, fillcolor="rgba(16,185,129,0.06)", line_width=0)
                fig_wk.add_hrect(y0=75, y1=90, fillcolor="rgba(245,158,11,0.06)", line_width=0)
                fig_wk.add_hline(y=90, line_dash="dot", line_color="rgba(16,185,129,0.5)",
                                 line_width=1.5, annotation_text="Target 90%",
                                 annotation_font_color="#10b981", annotation_font_size=9)
                fig_wk.add_trace(go.Bar(
                    x=weekly_df['week_label'], y=weekly_df['compliance_pct'],
                    marker_color=clrs, marker_line_width=0,
                    text=[f"{v:.1f}%" for v in weekly_df['compliance_pct']],
                    textposition='outside', textfont=dict(size=9, color=_FONT_CLR),
                    hovertemplate="Week of %{x}<br>Compliance: %{y:.1f}%<extra></extra>",
                ))
                fig_wk.update_layout(**{**_LAYOUT, 'height': 230,
                                        'yaxis': dict(range=[max(0, weekly_df['compliance_pct'].min() - 10), 103],
                                                      showgrid=True, gridcolor=_GRID_CLR, ticksuffix='%'),
                                        'xaxis': dict(showgrid=False),
                                        'showlegend': False,
                                        'margin': dict(l=10, r=10, t=24, b=10)})
                st.plotly_chart(fig_wk, use_container_width=True, key="eda_weekly")
            else:
                st.info("Weekly trend data not available.")

        # Action checklist
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        if shop_df is not None and not shop_df.empty:
            worst_shops = shop_df.nsmallest(3, 'compliance_pct')
            actions = []
            for _, r in worst_shops.iterrows():
                pct = r['compliance_pct']
                if pct < 60:
                    actions.append(f"🚨 <b>{r['shop_code']}</b> — {pct:.1f}% compliance. URGENT: audit cashier practices immediately.")
                elif pct < 80:
                    actions.append(f"⚠️ <b>{r['shop_code']}</b> — {pct:.1f}% compliance. Schedule retraining within 48 hours.")
                else:
                    actions.append(f"📋 <b>{r['shop_code']}</b> — {pct:.1f}% compliance. Monitor closely this week.")
            if actions:
                html_list = "".join(f'<li style="margin-bottom:6px;font-size:0.72rem;color:#cbd5e1;">{a}</li>' for a in actions)
                st.markdown(
                    f'<div style="background:rgba(15,23,42,0.6);border:1px solid #1e293b;border-radius:10px;'
                    f'padding:14px 16px;">'
                    f'<div style="font-size:0.68rem;font-weight:800;color:#94a3b8;text-transform:uppercase;'
                    f'letter-spacing:0.08em;margin-bottom:10px;">Action Items — Worst 3 Shops</div>'
                    f'<ul style="margin:0;padding-left:16px;">{html_list}</ul></div>',
                    unsafe_allow_html=True,
                )

    # ══════════════════════════════════════════════════════════
    # TAB 2 — SHOP INTELLIGENCE
    # ══════════════════════════════════════════════════════════
    with tab_shop:
        if shop_df is None or shop_df.empty:
            st.info("No shop data for the selected period.")
        else:
            shop_df = shop_df.copy()
            shop_df['compliance_pct'] = pd.to_numeric(shop_df['compliance_pct'], errors='coerce').fillna(0)
            shop_df['color'] = shop_df['compliance_pct'].apply(pct_color)

            sh1, sh2 = st.columns([1.6, 1], gap="large")

            with sh1:
                st.markdown('<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;'
                            'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">'
                            'All Shops — Compliance Ranking (worst → best)</div>',
                            unsafe_allow_html=True)
                bar_h = max(360, len(shop_df) * 26)
                fig_shops = go.Figure(go.Bar(
                    x=shop_df['compliance_pct'],
                    y=shop_df['shop_code'],
                    orientation='h',
                    marker_color=shop_df['color'],
                    marker_line_width=0,
                    text=[f"{v:.1f}%" for v in shop_df['compliance_pct']],
                    textposition='outside',
                    textfont=dict(size=9, color=_FONT_CLR),
                    customdata=shop_df[['total_serials', 'total_yes']].values,
                    hovertemplate=(
                        "<b>%{y}</b><br>Compliance: %{x:.1f}%<br>"
                        "Total Serials: %{customdata[0]:,}<br>Compliant: %{customdata[1]:,}<extra></extra>"
                    ),
                ))
                fig_shops.add_vline(x=90, line_dash="dot", line_color="rgba(16,185,129,0.6)",
                                    line_width=1.5, annotation_text="90% target",
                                    annotation_font_color="#10b981", annotation_font_size=9,
                                    annotation_position="top right")
                fig_shops.add_vline(x=75, line_dash="dot", line_color="rgba(245,158,11,0.5)",
                                    line_width=1, annotation_text="75%",
                                    annotation_font_color="#f59e0b", annotation_font_size=9,
                                    annotation_position="top right")
                fig_shops.update_layout(**{**_LAYOUT,
                    'height': bar_h,
                    'xaxis': dict(range=[0, 112], ticksuffix='%', showgrid=True, gridcolor=_GRID_CLR),
                    'yaxis': dict(automargin=True, tickfont=dict(size=9)),
                    'showlegend': False,
                })
                st.plotly_chart(fig_shops, use_container_width=True, key="eda_shop_bar")

            with sh2:
                # Best 5
                st.markdown('<div style="font-size:0.70rem;color:#10b981;font-weight:700;'
                            'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">'
                            'Top 5 — Best Compliance</div>', unsafe_allow_html=True)
                best5 = shop_df.nlargest(5, 'compliance_pct')
                for _, r in best5.iterrows():
                    st.markdown(
                        f'<div style="background:rgba(16,185,129,0.08);border-left:3px solid #10b981;'
                        f'border-radius:6px;padding:6px 10px;margin-bottom:4px;'
                        f'font-size:0.72rem;color:#e2e8f0;">'
                        f'<b>{r["shop_code"]}</b> &nbsp; '
                        f'<span style="color:#10b981;font-weight:800;">{r["compliance_pct"]:.1f}%</span>'
                        f'<span style="color:#64748b;font-size:0.65rem;"> ({int(r["total_serials"]):,} scans)</span></div>',
                        unsafe_allow_html=True,
                    )
                st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

                # Worst 5
                st.markdown('<div style="font-size:0.70rem;color:#ef4444;font-weight:700;'
                            'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">'
                            'Bottom 5 — Needs Attention</div>', unsafe_allow_html=True)
                worst5 = shop_df.nsmallest(5, 'compliance_pct')
                for _, r in worst5.iterrows():
                    st.markdown(
                        f'<div style="background:rgba(239,68,68,0.08);border-left:3px solid #ef4444;'
                        f'border-radius:6px;padding:6px 10px;margin-bottom:4px;'
                        f'font-size:0.72rem;color:#e2e8f0;">'
                        f'<b>{r["shop_code"]}</b> &nbsp; '
                        f'<span style="color:#ef4444;font-weight:800;">{r["compliance_pct"]:.1f}%</span>'
                        f'<span style="color:#64748b;font-size:0.65rem;"> ({int(r["total_serials"]):,} scans)</span></div>',
                        unsafe_allow_html=True,
                    )

                # Distribution
                st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
                st.markdown('<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;'
                            'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">'
                            'Compliance Distribution</div>', unsafe_allow_html=True)
                bins = [0, 60, 75, 85, 90, 95, 100.001]
                labels = ['<60%', '60-75%', '75-85%', '85-90%', '90-95%', '95-100%']
                bin_counts = pd.cut(shop_df['compliance_pct'], bins=bins, labels=labels, right=False).value_counts().reindex(labels)
                dist_colors = ['#ef4444', '#f97316', '#f59e0b', '#84cc16', '#10b981', '#059669']
                fig_dist = go.Figure(go.Bar(
                    x=labels, y=bin_counts.values,
                    marker_color=dist_colors, marker_line_width=0,
                    text=bin_counts.values, textposition='outside',
                    textfont=dict(size=9, color=_FONT_CLR),
                ))
                fig_dist.update_layout(**{**_LAYOUT, 'height': 180,
                    'xaxis': dict(showgrid=False, tickfont=dict(size=9)),
                    'yaxis': dict(showgrid=True, gridcolor=_GRID_CLR, title='Shops'),
                    'margin': dict(l=10, r=10, t=10, b=10),
                })
                st.plotly_chart(fig_dist, use_container_width=True, key="eda_shop_dist")

            # Shop × Date heatmap
            st.markdown('<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;'
                        'text-transform:uppercase;letter-spacing:0.08em;margin:14px 0 4px;">'
                        'Shop × Day Heatmap — Last 14 Days (compliance %)</div>',
                        unsafe_allow_html=True)
            heatmap_df = eda_shop_date_heatmap(14)
            if heatmap_df is not None and not heatmap_df.empty:
                hm_pivot = heatmap_df.pivot(index='shop_code', columns='activity_date', values='compliance_pct')
                hm_pivot.columns = [pd.Timestamp(c).strftime('%d %b') for c in hm_pivot.columns]
                fig_hm = go.Figure(go.Heatmap(
                    z=hm_pivot.values,
                    x=list(hm_pivot.columns),
                    y=list(hm_pivot.index),
                    colorscale=[[0, '#7f1d1d'], [0.4, '#854d0e'], [0.7, '#166534'], [1, '#064e3b']],
                    zmin=0, zmax=100,
                    text=[[f"{v:.1f}%" if not pd.isna(v) else "—" for v in row] for row in hm_pivot.values],
                    texttemplate="%{text}",
                    textfont=dict(size=8, color="white"),
                    hovertemplate="Shop: %{y}<br>Date: %{x}<br>Compliance: %{z:.1f}%<extra></extra>",
                    colorbar=dict(ticksuffix='%', tickfont=dict(color='#94a3b8', size=9),
                                  len=0.8, thickness=12),
                ))
                hm_h = max(300, len(hm_pivot) * 22)
                fig_hm.update_layout(**{**_LAYOUT, 'height': hm_h,
                    'xaxis': dict(side='top', tickfont=dict(size=9), showgrid=False),
                    'yaxis': dict(automargin=True, tickfont=dict(size=9)),
                })
                st.plotly_chart(fig_hm, use_container_width=True, key="eda_heatmap")
            else:
                st.info("Heatmap data not available.")

    # ══════════════════════════════════════════════════════════
    # TAB 3 — CASHIER INTELLIGENCE  (anomaly + pattern)
    # ══════════════════════════════════════════════════════════
    with tab_cashier:
        daily_df  = eda_cashier_daily_trend(30)
        week_df   = eda_cashier_week_comparison()

        if daily_df is None or daily_df.empty:
            st.info("No cashier daily trend data available.")
        else:
            daily_df = daily_df.copy()
            daily_df['compliance_pct'] = pd.to_numeric(daily_df['compliance_pct'], errors='coerce').fillna(0)

            # ── Build cashier summary with anomaly signals ──────────────
            grp = daily_df.groupby(['cashier', 'shop_code'])
            summary_rows = []
            for (cashier, shop), g in grp:
                g = g.sort_values('dt')
                pcts = g['compliance_pct'].tolist()
                total_scans = int(g['total_scans'].sum())
                total_yes   = int(g['yes_scans'].sum())
                avg_pct     = round(total_yes / total_scans * 100, 1) if total_scans else 0
                volatility  = round(pd.Series(pcts).std(), 1) if len(pcts) > 1 else 0
                zero_days   = int((pd.Series(pcts) == 0).sum())
                active_days = len(pcts)

                # Trend: compare first half vs second half
                mid = len(pcts) // 2
                first_half = pd.Series(pcts[:mid]).mean() if mid > 0 else avg_pct
                second_half = pd.Series(pcts[mid:]).mean() if mid > 0 else avg_pct
                trend_delta = round(second_half - first_half, 1)
                trend_dir   = "Improving" if trend_delta > 3 else ("Declining" if trend_delta < -3 else "Stable")

                # Week delta from week_df
                week_row = None
                if week_df is not None and not week_df.empty:
                    match = week_df[(week_df['cashier'] == cashier) & (week_df['shop_code'] == shop)]
                    if not match.empty:
                        r = match.iloc[0]
                        rec_pct  = round(r['rec_yes'] / r['rec_total'] * 100, 1) if r['rec_total'] > 0 else None
                        prev_pct = round(r['prev_yes'] / r['prev_total'] * 100, 1) if r['prev_total'] > 0 else None
                        week_delta = round(rec_pct - prev_pct, 1) if rec_pct is not None and prev_pct is not None else None
                    else:
                        week_delta = None
                else:
                    week_delta = None

                # Impact: non-compliant scan count (volume × failure rate)
                non_comp_count = total_scans - total_yes
                impact_score   = non_comp_count  # higher = more damage

                # Priority score (higher = needs more attention)
                #   40% low compliance, 25% week decline, 20% volatility, 15% zero-days
                score = (
                    0.40 * max(0, 100 - avg_pct) +
                    0.25 * (max(0, -(week_delta or 0)) * 5) +
                    0.20 * min(volatility, 50) +
                    0.15 * (zero_days * 20)
                )

                # Anomaly flags
                flags = []
                if avg_pct < 60:                        flags.append("CRITICAL")
                if (week_delta or 0) < -10:             flags.append("SHARP DROP")
                elif (week_delta or 0) < -5:            flags.append("DECLINING")
                if volatility > 35:                     flags.append("VOLATILE")
                if zero_days >= 2:                      flags.append("ZERO DAYS")
                if avg_pct >= 95:                       flags.append("EXCELLENT")
                elif avg_pct >= 90 and (week_delta or 0) >= 0: flags.append("ON TRACK")

                summary_rows.append({
                    'cashier': cashier, 'shop_code': shop,
                    'avg_pct': avg_pct, 'total_scans': total_scans,
                    'non_comp': non_comp_count,
                    'volatility': volatility, 'zero_days': zero_days,
                    'active_days': active_days,
                    'trend_dir': trend_dir, 'trend_delta': trend_delta,
                    'week_delta': week_delta, 'score': score,
                    'flags': ", ".join(flags) if flags else "OK",
                    'impact': impact_score,
                })

            summary = pd.DataFrame(summary_rows).sort_values('score', ascending=False)
            needs_attention = summary[~summary['flags'].str.contains("EXCELLENT|ON TRACK|OK", na=False)]
            improving       = summary[summary['flags'].str.contains("IMPROVING", na=False)] if 'IMPROVING' in summary['flags'].values else summary[summary['trend_dir'] == 'Improving']

            # ── SECTION 1: Priority Improvement List ────────────────────
            FLAG_STYLE = {
                "CRITICAL":   ("#ef4444", "#7f1d1d"),
                "SHARP DROP": ("#dc2626", "#7f1d1d"),
                "DECLINING":  ("#f97316", "#7c2d12"),
                "VOLATILE":   ("#f59e0b", "#78350f"),
                "ZERO DAYS":  ("#a855f7", "#3b0764"),
                "EXCELLENT":  ("#10b981", "#064e3b"),
                "ON TRACK":   ("#22d3ee", "#164e63"),
                "OK":         ("#64748b", "#1e293b"),
            }

            top_priority = needs_attention.head(30) if not needs_attention.empty else summary.head(30)

            # Build one compact HTML table — single render call
            th = ("font-size:0.60rem;color:#475569;font-weight:700;text-transform:uppercase;"
                  "letter-spacing:0.07em;padding:5px 8px;border-bottom:1px solid #1e293b;"
                  "white-space:nowrap;")
            td = "font-size:0.68rem;padding:4px 8px;color:#cbd5e1;white-space:nowrap;"

            rows_html = ""
            for i, (_, r) in enumerate(top_priority.iterrows()):
                flags_list   = [f.strip() for f in str(r['flags']).split(',')]
                primary_flag = flags_list[0] if flags_list else "OK"
                clr, dark    = FLAG_STYLE.get(primary_flag, FLAG_STYLE["OK"])

                # Compliance % bar
                pct     = r['avg_pct']
                bar_w   = max(0, min(100, pct))
                pct_clr = ("#10b981" if pct >= 90 else "#f59e0b" if pct >= 75 else "#ef4444")

                # WoW delta
                wd = r['week_delta']
                if wd is not None:
                    wow_clr = "#10b981" if wd >= 0 else "#ef4444"
                    wow_txt = f'{"▲" if wd >= 0 else "▼"}{abs(wd):.1f}%'
                else:
                    wow_clr, wow_txt = "#64748b", "—"

                # Trend arrow
                t_icon = ("▲" if r['trend_dir'] == "Improving" else
                          "▼" if r['trend_dir'] == "Declining" else "→")
                t_clr  = ("#10b981" if r['trend_dir'] == "Improving" else
                           "#ef4444" if r['trend_dir'] == "Declining" else "#94a3b8")

                # Flag pills (all flags, compact)
                pills = "".join(
                    f'<span style="background:{FLAG_STYLE.get(f, FLAG_STYLE["OK"])[1]};color:{FLAG_STYLE.get(f, FLAG_STYLE["OK"])[0]};'
                    f'font-size:0.55rem;padding:1px 5px;border-radius:3px;margin-left:3px;font-weight:700;">{f}</span>'
                    for f in flags_list if f and f != "OK"
                )

                row_bg = "rgba(239,68,68,0.05)" if i % 2 == 0 else "transparent"
                rows_html += (
                    f'<tr style="background:{row_bg};border-bottom:1px solid rgba(255,255,255,0.03);">'
                    f'<td style="{td}color:#475569;">{i+1}</td>'
                    f'<td style="{td}"><span style="color:#e2e8f0;font-weight:600;">{r["cashier"]}</span>'
                    f' <span style="color:#475569;font-size:0.60rem;">[{r["shop_code"]}]</span></td>'
                    f'<td style="{td}">'
                    f'<div style="display:flex;align-items:center;gap:5px;">'
                    f'<div style="width:54px;height:5px;background:#1e293b;border-radius:3px;">'
                    f'<div style="width:{bar_w}%;height:100%;background:{pct_clr};border-radius:3px;"></div></div>'
                    f'<span style="color:{pct_clr};font-weight:800;">{pct:.1f}%</span>'
                    f'</div></td>'
                    f'<td style="{td}color:{wow_clr};font-weight:700;">{wow_txt}</td>'
                    f'<td style="{td}color:{t_clr};font-weight:700;">{t_icon} {r["trend_dir"]}</td>'
                    f'<td style="{td}color:#94a3b8;">{int(r["total_scans"]):,}</td>'
                    f'<td style="{td}color:#ef4444;">{int(r["non_comp"]):,}</td>'
                    f'<td style="{td}color:#f59e0b;">{r["volatility"]:.0f}%</td>'
                    f'<td style="{td}color:#a855f7;">{int(r["zero_days"])}</td>'
                    f'<td style="{td}">{pills}</td>'
                    f'</tr>'
                )

            st.markdown(
                f'<div style="font-size:0.62rem;font-weight:800;color:#94a3b8;letter-spacing:0.1em;'
                f'text-transform:uppercase;margin-bottom:6px;">'
                f'Priority Improvement List — {len(top_priority)} cashiers needing attention</div>'
                f'<div style="overflow-x:auto;border:1px solid #1e293b;border-radius:8px;">'
                f'<table style="width:100%;border-collapse:collapse;">'
                f'<thead><tr style="background:rgba(15,23,42,0.8);">'
                f'<th style="{th}">#</th>'
                f'<th style="{th}">Cashier</th>'
                f'<th style="{th}">Compliance</th>'
                f'<th style="{th}">WoW</th>'
                f'<th style="{th}">Trend</th>'
                f'<th style="{th}">Scans</th>'
                f'<th style="{th}">Non-Y</th>'
                f'<th style="{th}">Volatility</th>'
                f'<th style="{th}">0-Days</th>'
                f'<th style="{th}">Flags</th>'
                f'</tr></thead>'
                f'<tbody>{rows_html}</tbody>'
                f'</table></div>',
                unsafe_allow_html=True,
            )

            # ── SECTION 2: Cashier × Date Heatmap ───────────────────────
            st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;text-transform:uppercase;'
                'letter-spacing:0.08em;margin-bottom:4px;">'
                'Cashier × Day Pattern Heatmap — Last 14 Days (worst 35 cashiers by priority)</div>',
                unsafe_allow_html=True,
            )
            hm_cashiers = top_priority['cashier'].tolist()[:35]
            hm_data = daily_df[
                (daily_df['cashier'].isin(hm_cashiers)) &
                (pd.to_datetime(daily_df['dt']).dt.date >= (datetime.today().date() - timedelta(days=14)))
            ].copy()
            hm_data['label'] = hm_data['cashier'] + ' [' + hm_data['shop_code'] + ']'
            hm_data['dt_str'] = pd.to_datetime(hm_data['dt']).dt.strftime('%d %b')

            if not hm_data.empty:
                hm_pivot = hm_data.pivot_table(index='label', columns='dt_str',
                                               values='compliance_pct', aggfunc='mean')
                # Order columns chronologically
                date_range_14 = pd.date_range(
                    start=datetime.today().date() - timedelta(days=14),
                    end=datetime.today().date() - timedelta(days=1)
                )
                date_labels_14 = [d.strftime('%d %b') for d in date_range_14]
                ordered_cols = [c for c in date_labels_14 if c in hm_pivot.columns]
                hm_pivot = hm_pivot.reindex(columns=ordered_cols)

                z_text = [[f"{v:.0f}%" if not pd.isna(v) else "—" for v in row] for row in hm_pivot.values]
                fig_chm = go.Figure(go.Heatmap(
                    z=hm_pivot.values,
                    x=list(hm_pivot.columns),
                    y=list(hm_pivot.index),
                    colorscale=[
                        [0.0,  '#7f1d1d'], [0.3, '#991b1b'],
                        [0.5,  '#854d0e'], [0.75, '#166534'],
                        [1.0,  '#064e3b']
                    ],
                    zmin=0, zmax=100,
                    text=z_text,
                    texttemplate="%{text}",
                    textfont=dict(size=7, color="white"),
                    hovertemplate="<b>%{y}</b><br>%{x}<br>Compliance: %{z:.1f}%<extra></extra>",
                    colorbar=dict(ticksuffix='%', tickfont=dict(color='#94a3b8', size=9),
                                  len=0.7, thickness=10, title=dict(text='%', font=dict(color='#94a3b8', size=9))),
                    xgap=1, ygap=1,
                ))
                fig_chm.update_layout(**{**_LAYOUT, 'height': max(400, len(hm_pivot) * 20),
                    'xaxis': dict(side='top', tickfont=dict(size=9), showgrid=False),
                    'yaxis': dict(automargin=True, tickfont=dict(size=9)),
                    'margin': dict(l=10, r=20, t=30, b=10),
                })
                st.plotly_chart(fig_chm, use_container_width=True, key="eda_cashier_heatmap")
            else:
                st.info("Not enough recent data for cashier heatmap.")

            # ── SECTION 3: Trend Lines for Worst 10 ─────────────────────
            st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
            tr1, tr2 = st.columns([1.6, 1], gap="large")

            with tr1:
                st.markdown(
                    '<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;text-transform:uppercase;'
                    'letter-spacing:0.08em;margin-bottom:4px;">'
                    'Daily Trend — Worst 10 (30 days) — spot who is recovering vs worsening</div>',
                    unsafe_allow_html=True,
                )
                worst10 = top_priority.head(10)['cashier'].tolist()
                trend_data = daily_df[daily_df['cashier'].isin(worst10)].copy()
                trend_data['dt'] = pd.to_datetime(trend_data['dt'])
                trend_data['label'] = trend_data['cashier'] + ' [' + trend_data['shop_code'] + ']'

                if not trend_data.empty:
                    # Color each line by their trend direction
                    line_colors = {
                        r['cashier']: (
                            '#ef4444' if r['trend_dir'] == 'Declining' else
                            '#10b981' if r['trend_dir'] == 'Improving' else
                            '#f59e0b'
                        )
                        for _, r in summary[summary['cashier'].isin(worst10)].iterrows()
                    }
                    fig_trend_c = go.Figure()
                    fig_trend_c.add_hline(y=90, line_dash="dot",
                                          line_color="rgba(16,185,129,0.5)", line_width=1.5)
                    for lbl in trend_data['label'].unique():
                        sub = trend_data[trend_data['label'] == lbl].sort_values('dt')
                        cashier_name_only = lbl.split(' [')[0]
                        clr = line_colors.get(cashier_name_only, '#9b5bff')
                        fig_trend_c.add_trace(go.Scatter(
                            x=sub['dt'], y=sub['compliance_pct'],
                            name=lbl, mode='lines+markers',
                            line=dict(color=clr, width=2, shape='spline'),
                            marker=dict(size=5, color=clr),
                            hovertemplate=f"<b>{lbl}</b><br>%{{x|%d %b}}: %{{y:.1f}}<extra></extra>",
                        ))
                    fig_trend_c.update_layout(**{**_LAYOUT, 'height': 360,
                        'yaxis': dict(range=[0, 105], ticksuffix='%',
                                      showgrid=True, gridcolor=_GRID_CLR),
                        'xaxis': dict(showgrid=False, tickformat='%d %b'),
                        'legend': dict(font=dict(size=8, color=_FONT_CLR), orientation='v',
                                       x=1.01, y=1, bgcolor='rgba(0,0,0,0)'),
                        'hovermode': 'closest',
                    })
                    st.plotly_chart(fig_trend_c, use_container_width=True, key="eda_cashier_trends")
                    st.markdown(
                        '<div style="font-size:0.62rem;color:#475569;">🔴 Declining &nbsp;🟡 Stable &nbsp;🟢 Improving</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No trend data for worst cashiers.")

            with tr2:
                st.markdown(
                    '<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;text-transform:uppercase;'
                    'letter-spacing:0.08em;margin-bottom:6px;">Anomaly Summary</div>',
                    unsafe_allow_html=True,
                )
                flag_counts = {}
                for f_list in summary['flags']:
                    for f in str(f_list).split(','):
                        f = f.strip()
                        if f and f != 'OK':
                            flag_counts[f] = flag_counts.get(f, 0) + 1

                _flag_icon = {
                    "CRITICAL": "🔴", "SHARP DROP": "⬇", "DECLINING": "📉",
                    "VOLATILE": "⚡", "ZERO DAYS": "⬛", "EXCELLENT": "🏆",
                    "ON TRACK": "✅",
                }
                for flag_name, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
                    clr, dark = FLAG_STYLE.get(flag_name, ("#64748b", "#1e293b"))
                    icon = _flag_icon.get(flag_name, "•")
                    st.markdown(
                        f'<div style="background:{dark};border:1px solid {clr}30;border-radius:6px;'
                        f'padding:7px 12px;margin-bottom:4px;display:flex;justify-content:space-between;'
                        f'align-items:center;">'
                        f'<span style="font-size:0.72rem;color:{clr};font-weight:700;">{icon} {flag_name}</span>'
                        f'<span style="font-size:1.1rem;font-weight:900;color:{clr};">{count}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

                # Compliance distribution histogram
                st.markdown(
                    '<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;text-transform:uppercase;'
                    'letter-spacing:0.08em;margin-bottom:4px;">Cashier Compliance Distribution</div>',
                    unsafe_allow_html=True,
                )
                fig_hist2 = go.Figure(go.Histogram(
                    x=summary['avg_pct'], nbinsx=20,
                    marker_color='#9b5bff', marker_line_width=0.5,
                    marker_line_color='rgba(0,0,0,0.3)',
                    hovertemplate="Range: %{x:.0f}%<br>Cashiers: %{y}<extra></extra>",
                ))
                fig_hist2.add_vline(x=90, line_dash="dot", line_color="#10b981", line_width=1.5)
                fig_hist2.add_vline(x=75, line_dash="dot", line_color="#f59e0b", line_width=1)
                fig_hist2.update_layout(**{**_LAYOUT, 'height': 180,
                    'xaxis': dict(ticksuffix='%', showgrid=False),
                    'yaxis': dict(showgrid=True, gridcolor=_GRID_CLR),
                    'margin': dict(l=10, r=10, t=10, b=30),
                })
                st.plotly_chart(fig_hist2, use_container_width=True, key="eda_cashier_hist2")

            # ── SECTION 4: Stats footer ──────────────────────────────────
            st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)
            total_c = len(summary)
            above90 = (summary['avg_pct'] >= 90).sum()
            declining_c = (summary['trend_dir'] == 'Declining').sum()
            critical_c  = (summary['avg_pct'] < 60).sum()
            high_vol_c  = (summary['volatility'] > 35).sum()
            st.markdown(
                f'<div style="background:rgba(15,23,42,0.5);border:1px solid #1e293b;border-radius:8px;'
                f'padding:10px 16px;font-size:0.68rem;color:#94a3b8;">'
                f'Cashiers analysed: <b style="color:#e2e8f0">{total_c}</b> &nbsp;|&nbsp; '
                f'Above 90%: <b style="color:#10b981">{above90}</b> &nbsp;|&nbsp; '
                f'Declining trend: <b style="color:#f97316">{declining_c}</b> &nbsp;|&nbsp; '
                f'Critical (<60%): <b style="color:#ef4444">{critical_c}</b> &nbsp;|&nbsp; '
                f'Volatile (>35% std): <b style="color:#f59e0b">{high_vol_c}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════
    # TAB 4 — TIME PATTERNS
    # ══════════════════════════════════════════════════════════
    with tab_time:
        dow_df = eda_dow_pattern(start_s, end_s)

        tp1, tp2 = st.columns([1, 1], gap="large")

        with tp1:
            st.markdown('<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;'
                        'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">'
                        'Day-of-Week Compliance Pattern</div>', unsafe_allow_html=True)
            if dow_df is not None and not dow_df.empty:
                dow_df['compliance_pct'] = pd.to_numeric(dow_df['compliance_pct'], errors='coerce').fillna(0)
                dow_df['color'] = dow_df['compliance_pct'].apply(pct_color)
                day_order = {0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'}
                dow_df['day_name'] = dow_df['dow'].map(day_order)
                dow_df = dow_df.sort_values('dow')
                fig_dow = go.Figure(go.Bar(
                    x=dow_df['day_name'], y=dow_df['compliance_pct'],
                    marker_color=dow_df['color'], marker_line_width=0,
                    text=[f"{v:.1f}%" for v in dow_df['compliance_pct']],
                    textposition='outside', textfont=dict(size=10, color=_FONT_CLR),
                    customdata=dow_df['total_serials'].values,
                    hovertemplate="%{x}<br>Compliance: %{y:.1f}%<br>Serials: %{customdata:,}<extra></extra>",
                ))
                fig_dow.add_hline(y=90, line_dash="dot", line_color="rgba(16,185,129,0.6)",
                                  line_width=1.5, annotation_text="90%",
                                  annotation_font_color="#10b981", annotation_font_size=9)
                fig_dow.update_layout(**{**_LAYOUT, 'height': 280,
                    'yaxis': dict(range=[max(0, dow_df['compliance_pct'].min()-10), 105],
                                  ticksuffix='%', showgrid=True, gridcolor=_GRID_CLR),
                    'xaxis': dict(showgrid=False),
                    'showlegend': False,
                })
                st.plotly_chart(fig_dow, use_container_width=True, key="eda_dow")

                # Insight
                worst_day = dow_df.loc[dow_df['compliance_pct'].idxmin()]
                best_day  = dow_df.loc[dow_df['compliance_pct'].idxmax()]
                st.markdown(
                    f'<div style="font-size:0.68rem;color:#64748b;padding:4px 0;">'
                    f'Worst day: <b style="color:#ef4444">{worst_day["day_name"]} ({worst_day["compliance_pct"]:.1f}%)</b> &nbsp;|&nbsp; '
                    f'Best day: <b style="color:#10b981">{best_day["day_name"]} ({best_day["compliance_pct"]:.1f}%)</b>'
                    f'</div>', unsafe_allow_html=True)
            else:
                st.info("Day-of-week data not available.")

        with tp2:
            st.markdown('<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;'
                        'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">'
                        'Week-over-Week Compliance Trend</div>', unsafe_allow_html=True)
            if weekly_df is not None and not weekly_df.empty:
                weekly_df2 = weekly_df.copy()
                weekly_df2['compliance_pct'] = pd.to_numeric(weekly_df2['compliance_pct'], errors='coerce')
                weekly_df2['week_label'] = pd.to_datetime(weekly_df2['week_start']).dt.strftime('%d %b')
                weekly_df2['delta'] = weekly_df2['compliance_pct'].diff()
                fig_wk2 = go.Figure()
                fig_wk2.add_hrect(y0=90, y1=100, fillcolor="rgba(16,185,129,0.06)", line_width=0)
                fig_wk2.add_trace(go.Scatter(
                    x=weekly_df2['week_label'], y=weekly_df2['compliance_pct'],
                    mode='lines+markers+text',
                    line=dict(color='#9b5bff', width=3, shape='spline'),
                    marker=dict(size=8, color=[pct_color(v) for v in weekly_df2['compliance_pct']],
                                line=dict(color='rgba(255,255,255,0.4)', width=1.5)),
                    text=[f"{v:.1f}%" for v in weekly_df2['compliance_pct']],
                    textposition='top center', textfont=dict(size=9, color=_FONT_CLR),
                    hovertemplate="Week: %{x}<br>Compliance: %{y:.1f}%<extra></extra>",
                ))
                fig_wk2.add_hline(y=90, line_dash="dot", line_color="rgba(16,185,129,0.5)", line_width=1.5)
                fig_wk2.update_layout(**{**_LAYOUT, 'height': 280,
                    'yaxis': dict(range=[max(0, weekly_df2['compliance_pct'].min()-10), 105],
                                  ticksuffix='%', showgrid=True, gridcolor=_GRID_CLR),
                    'xaxis': dict(showgrid=False),
                })
                st.plotly_chart(fig_wk2, use_container_width=True, key="eda_wk_line")
            else:
                st.info("Weekly trend data not available.")

        # Volume profile
        st.markdown('<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;'
                    'text-transform:uppercase;letter-spacing:0.08em;margin:14px 0 4px;">'
                    'Scan Volume by Day of Week</div>', unsafe_allow_html=True)
        if dow_df is not None and not dow_df.empty:
            fig_vol = go.Figure(go.Bar(
                x=dow_df['day_name'], y=dow_df['total_serials'],
                marker_color='rgba(155,91,255,0.7)', marker_line_width=0,
                text=dow_df['total_serials'], textposition='outside',
                textfont=dict(size=9, color=_FONT_CLR),
                hovertemplate="%{x}<br>Serials: %{y:,}<extra></extra>",
            ))
            fig_vol.update_layout(**{**_LAYOUT, 'height': 160,
                'yaxis': dict(showgrid=True, gridcolor=_GRID_CLR, title='Serials'),
                'xaxis': dict(showgrid=False),
                'margin': dict(l=10, r=10, t=10, b=10),
            })
            st.plotly_chart(fig_vol, use_container_width=True, key="eda_vol_dow")

    # ══════════════════════════════════════════════════════════
    # TAB 5 — ISSUE ANATOMY
    # ══════════════════════════════════════════════════════════
    with tab_issue:
        issue_trend_df = eda_issue_daily_trend(14)
        grand_total = int(anatomy.get('grand_total') or 0)
        compliant   = int(anatomy.get('compliant')   or 0)
        no_serial   = int(anatomy.get('no_serial')   or 0)
        small_ser   = int(anatomy.get('small_serial') or 0)
        not_in_wh   = int(anatomy.get('not_in_wh')   or 0)
        non_comp    = grand_total - compliant

        is1, is2 = st.columns([1, 1.6], gap="large")

        with is1:
            st.markdown('<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;'
                        'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">'
                        'Issue Breakdown (period total)</div>', unsafe_allow_html=True)
            labels = ['Compliant (Y)', 'Not In WH', 'Small/Invalid Serial', 'No Serial Entered']
            values = [compliant, not_in_wh, small_ser, no_serial]
            clrs   = ['#10b981', '#ef4444', '#f97316', '#f59e0b']
            fig_donut = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.62,
                marker=dict(colors=clrs, line=dict(color='rgba(0,0,0,0.2)', width=1)),
                textinfo='percent', textfont=dict(size=11, color='white'),
                hovertemplate="%{label}<br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
            ))
            fig_donut.add_annotation(
                text=f"<b>{round(compliant/grand_total*100,1) if grand_total else 0:.1f}%</b><br>Compliant",
                x=0.5, y=0.5, font=dict(size=14, color='#e2e8f0', family='Inter'),
                showarrow=False, align='center',
            )
            fig_donut.update_layout(**{**_LAYOUT, 'height': 300,
                'legend': dict(orientation='v', x=1.02, y=0.5, font=dict(size=10, color=_FONT_CLR)),
                'margin': dict(l=10, r=10, t=20, b=10),
            })
            st.plotly_chart(fig_donut, use_container_width=True, key="eda_donut")

            # Stat boxes
            stat_items = [
                ("Not In WH", not_in_wh, grand_total, "#ef4444",
                 "Serial entered but not found in WH records — potential fake or mistyped serials"),
                ("Small / Invalid", small_ser, grand_total, "#f97316",
                 "Serial ≤6 chars — likely item code typed instead of serial"),
                ("No Serial", no_serial, grand_total, "#f59e0b",
                 "Field left empty — cashier skipped serial entry entirely"),
            ]
            for lbl, cnt, tot, clr, tip in stat_items:
                p = round(cnt / tot * 100, 1) if tot else 0
                st.markdown(
                    f'<div style="background:rgba(15,23,42,0.5);border-left:3px solid {clr};'
                    f'border-radius:6px;padding:8px 12px;margin-bottom:6px;">'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="font-size:0.70rem;color:#94a3b8;font-weight:700;">{lbl}</span>'
                    f'<span style="font-size:0.80rem;font-weight:900;color:{clr};">{p:.1f}%</span></div>'
                    f'<div style="font-size:0.62rem;color:#475569;margin-top:2px;">{cnt:,} records — {tip}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with is2:
            st.markdown('<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;'
                        'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">'
                        'Issue Trend — Last 14 Days (stacked area)</div>',
                        unsafe_allow_html=True)
            if issue_trend_df is not None and not issue_trend_df.empty:
                it = issue_trend_df.copy()
                it['dt'] = pd.to_datetime(it['dt']).dt.strftime('%d %b')
                fig_area = go.Figure()
                for col, clr, nm in [
                    ('not_in_wh',  '#ef4444', 'Not In WH'),
                    ('small_serial','#f97316','Small/Invalid'),
                    ('no_serial',  '#f59e0b', 'No Serial'),
                    ('compliant',  '#10b981', 'Compliant'),
                ]:
                    fig_area.add_trace(go.Scatter(
                        x=it['dt'], y=it[col], name=nm,
                        stackgroup='one', mode='lines',
                        line=dict(color=clr, width=0.5),
                        fillcolor=clr.replace('#', 'rgba(') + ',0.7)' if False else clr,
                        hovertemplate=f"{nm}: %{{y:,}}<extra></extra>",
                    ))
                fig_area.update_layout(**{**_LAYOUT, 'height': 300,
                    'yaxis': dict(showgrid=True, gridcolor=_GRID_CLR, title='Serials'),
                    'xaxis': dict(showgrid=False),
                    'legend': dict(orientation='h', y=-0.2, font=dict(size=10, color=_FONT_CLR)),
                    'hovermode': 'x unified',
                })
                st.plotly_chart(fig_area, use_container_width=True, key="eda_issue_area")

                # compliance line over issues
                st.markdown('<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;'
                            'text-transform:uppercase;letter-spacing:0.08em;margin:10px 0 4px;">'
                            'Daily Compliance % vs. Non-Compliant Count</div>',
                            unsafe_allow_html=True)
                it['total'] = it['compliant'] + it['not_in_wh'] + it['small_serial'] + it['no_serial']
                it['non_comp'] = it['not_in_wh'] + it['small_serial'] + it['no_serial']
                it['comp_pct'] = (it['compliant'] / it['total'].replace(0, None) * 100).round(1)

                fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
                fig_dual.add_trace(go.Bar(
                    x=it['dt'], y=it['non_comp'],
                    name='Non-Compliant Count', marker_color='rgba(239,68,68,0.5)',
                    hovertemplate="Non-compliant: %{y:,}<extra></extra>",
                ), secondary_y=False)
                fig_dual.add_trace(go.Scatter(
                    x=it['dt'], y=it['comp_pct'],
                    name='Compliance %', mode='lines+markers',
                    line=dict(color='#9b5bff', width=2.5),
                    marker=dict(size=6, color='#9b5bff'),
                    hovertemplate="Compliance: %{y:.1f}%<extra></extra>",
                ), secondary_y=True)
                fig_dual.update_layout(**{**_LAYOUT, 'height': 220,
                    'legend': dict(orientation='h', y=-0.3, font=dict(size=10, color=_FONT_CLR)),
                    'hovermode': 'x unified',
                    'margin': dict(l=10, r=40, t=10, b=10),
                })
                fig_dual.update_yaxes(title_text="Non-Compliant", showgrid=True,
                                      gridcolor=_GRID_CLR, secondary_y=False)
                fig_dual.update_yaxes(title_text="Compliance %", ticksuffix='%',
                                      showgrid=False, secondary_y=True)
                st.plotly_chart(fig_dual, use_container_width=True, key="eda_dual_axis")
            else:
                st.info("Issue trend data not available.")

    # ══════════════════════════════════════════════════════════
    # TAB 6 — DEPT / GROUP ANALYSIS
    # ══════════════════════════════════════════════════════════
    with tab_dept:
        if dept_df is None or dept_df.empty:
            st.info("Department data not available. Run setup_dept_compliance_mv.py first.")
        else:
            dept_df = dept_df.copy()
            dept_df['compliance_pct'] = pd.to_numeric(dept_df['compliance_pct'], errors='coerce').fillna(0)

            level_choice = st.radio("View by:", ["Dept", "Group (within Dept)"],
                                    horizontal=True, key="eda_dept_level")

            if level_choice == "Dept":
                agg = dept_df.groupby('dept', as_index=False).agg(
                    total_serials=('total_serials', 'sum'),
                    total_yes=('total_yes', 'sum')
                )
                agg['compliance_pct'] = (agg['total_yes'] / agg['total_serials'].replace(0, None) * 100).round(1).fillna(0)
                y_col, title_lbl = 'dept', 'Department'
            else:
                agg = dept_df.copy()
                agg['label'] = agg['grp'] + '  [' + agg['dept'] + ']'
                y_col, title_lbl = 'label', 'Group [Dept]'

            agg = agg.sort_values('compliance_pct')
            agg['color'] = agg['compliance_pct'].apply(pct_color)

            dep1, dep2 = st.columns([1.5, 1], gap="large")

            with dep1:
                bar_h = max(320, len(agg) * 26)
                fig_dept2 = go.Figure(go.Bar(
                    x=agg['compliance_pct'], y=agg[y_col],
                    orientation='h', marker_color=agg['color'], marker_line_width=0,
                    text=[f"{v:.1f}%" for v in agg['compliance_pct']],
                    textposition='outside', textfont=dict(size=9, color=_FONT_CLR),
                    customdata=agg['total_serials'].values,
                    hovertemplate="%{y}<br>Compliance: %{x:.1f}%<br>Serials: %{customdata:,}<extra></extra>",
                ))
                fig_dept2.add_vline(x=90, line_dash="dot", line_color="rgba(16,185,129,0.6)", line_width=1.5)
                fig_dept2.update_layout(**{**_LAYOUT, 'height': bar_h,
                    'xaxis': dict(range=[0, 112], ticksuffix='%', showgrid=True, gridcolor=_GRID_CLR),
                    'yaxis': dict(automargin=True, tickfont=dict(size=9)),
                    'showlegend': False,
                })
                st.plotly_chart(fig_dept2, use_container_width=True, key="eda_dept_bar")

            with dep2:
                st.markdown('<div style="font-size:0.70rem;color:#94a3b8;font-weight:700;'
                            'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">'
                            'Summary Table</div>', unsafe_allow_html=True)
                disp = agg[[y_col, 'total_serials', 'total_yes', 'compliance_pct']].rename(columns={
                    y_col: title_lbl,
                    'total_serials': 'Serials',
                    'total_yes': 'Compliant',
                    'compliance_pct': 'Compliance %',
                })
                st.dataframe(
                    disp, use_container_width=True, hide_index=True,
                    column_config={
                        title_lbl: st.column_config.TextColumn(title_lbl, width='medium'),
                        'Serials': st.column_config.NumberColumn('Serials', format='%d'),
                        'Compliant': st.column_config.NumberColumn('Compliant', format='%d'),
                        'Compliance %': st.column_config.ProgressColumn(
                            'Compliance %', format='%.1f%%', min_value=0, max_value=100),
                    }
                )

                # 7-day trend for top depts
                st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
                dept_trend_data = get_dept_compliance_data(
                    (datetime.today().date() - timedelta(days=6)).strftime('%Y-%m-%d'),
                    (datetime.today().date() - timedelta(days=1)).strftime('%Y-%m-%d')
                )
                if dept_trend_data[1] is not None and not dept_trend_data[1].empty:
                    tr = dept_trend_data[1].copy()
                    tr['yes_pct'] = (tr['total_yes'] / tr['total_serials'].replace({0: None}) * 100).round(1).fillna(0)
                    tr['activity_date'] = pd.to_datetime(tr['activity_date']).dt.strftime('%d %b')
                    top_depts_list = agg.nlargest(5, 'total_serials')[y_col].tolist()
                    color_col = 'dept'
                    tr_filtered = tr[tr[color_col].isin(
                        [x.split('  [')[0] if '  [' in x else x for x in top_depts_list]
                    )]
                    if not tr_filtered.empty:
                        fig_dt = px.line(tr_filtered, x='activity_date', y='yes_pct',
                                         color=color_col, markers=True,
                                         labels={'yes_pct': 'Compliance %', 'activity_date': 'Date'})
                        fig_dt.update_traces(line=dict(width=2), marker=dict(size=6))
                        fig_dt.update_layout(**{**_LAYOUT, 'height': 200,
                            'yaxis': dict(ticksuffix='%', range=[0, 105], showgrid=True, gridcolor=_GRID_CLR),
                            'xaxis': dict(showgrid=False),
                            'legend': dict(font=dict(size=9, color=_FONT_CLR), x=0, y=1.1, orientation='h'),
                            'margin': dict(l=10, r=10, t=28, b=10),
                        })
                        st.plotly_chart(fig_dt, use_container_width=True, key="eda_dept_trend")


TITLE_CSS_LOCAL = (
    "font-size:0.80rem;font-weight:700;color:#94a3b8;"
    "letter-spacing:0.06em;text-transform:uppercase;"
)

def render_shop_compliance_pivot(end_s: str):
    """Shop-wise Compliance % pivot table — rows = shop, columns = months."""
    st.markdown(
        f'<div style="{TITLE_CSS_LOCAL}">🏪 Shop-wise Compliance % — Month on Month</div>',
        unsafe_allow_html=True,
    )

    pivot = load_shop_compliance_pivot(end_s)
    if pivot is None or pivot.empty:
        st.info("No shop compliance data available for the selected period.")
        return

    month_cols = [c for c in pivot.columns if c != 'Shop']

    # ── Per-shop Overall average ───────────────────────────────
    num_pivot = pivot[month_cols].apply(pd.to_numeric, errors='coerce')
    pivot['Overall'] = num_pivot.mean(axis=1).round(1)

    all_val_cols = month_cols + ['Overall']

    # ── Summary row (all shops combined) pinned at top ────────
    summary_row = {'Shop': '📊 All Shops'}
    for col in month_cols:
        summary_row[col] = round(float(num_pivot[col].mean()), 1) if num_pivot[col].notna().any() else float('nan')
    summary_row['Overall'] = round(float(pivot['Overall'].mean()), 1)

    shop_rows  = pivot[['Shop'] + all_val_cols].sort_values('Shop').reset_index(drop=True)
    summary_df = pd.DataFrame([summary_row])
    display_df = pd.concat([summary_df, shop_rows], ignore_index=True)

    # ── Styler ─────────────────────────────────────────────────
    def _full_style(df):
        styled = df.style

        # Format all value columns as "XX.X%"
        fmt_all = {c: (lambda v: f"{v:.1f}%" if pd.notna(v) and v == v else "—")
                   for c in all_val_cols}
        styled = styled.format(fmt_all)

        # Conditional colour + dark text for every value cell
        for col in all_val_cols:
            try:
                styled = styled.map(_compliance_cell_style, subset=[col])
            except AttributeError:
                styled = styled.applymap(_compliance_cell_style, subset=[col])

        # Shop column: black text, bold, left-aligned
        styled = styled.set_properties(
            subset=['Shop'],
            **{'font-weight': '800',
               'color': '#000000',
               'text-align': 'left',
               'white-space': 'nowrap',
               'font-size': '0.83rem'}
        )
        styled = styled.set_properties(
            subset=all_val_cols,
            **{'text-align': 'center', 'font-size': '0.83rem', 'min-width': '72px'}
        )

        # Highlight the summary row (index 0) with a slightly stronger bg
        styled = styled.set_properties(
            **{'border-bottom': '2px solid rgba(148,163,184,0.3)',
               'font-weight': '800'},
            subset=pd.IndexSlice[0, :]
        )
        return styled

    row_h = 36
    tbl_h = min(row_h * (len(display_df) + 1) + 42, 560)

    st.dataframe(
        _full_style(display_df),
        use_container_width=True,
        height=tbl_h,
        hide_index=True,
    )

    # Summary legend
    st.markdown("""
<div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:6px;font-size:0.72rem;color:#64748b;">
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;
        background:rgba(4,120,87,0.55);margin-right:4px;"></span>≥ 95 % Excellent</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;
        background:rgba(10,140,87,0.45);margin-right:4px;"></span>85–94 % Good</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;
        background:rgba(200,130,10,0.42);margin-right:4px;"></span>70–84 % Needs Attention</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;
        background:rgba(220,40,20,0.48);margin-right:4px;"></span>&lt; 70 % Critical</span>
</div>""", unsafe_allow_html=True)


def _slab_cell_style(val, col_name, col_max):
    if col_name == "Before WH Loaded":
        if val == 0:
            return 'background-color: rgba(4,44,32,0.55); color: #6ee7b7'
        ratio = min(val / max(col_max, 1), 1.0)
        return f'background-color: rgba(180,0,{int(60*(1-ratio))},0.45); color: #ffe4e4; font-weight:700'
    if col_max == 0 or val == 0:
        return 'background-color: rgba(4,44,32,0.55); color: #6ee7b7'
    ratio = val / col_max
    r = int(220 * ratio)
    g = int(175 * (1 - ratio * 0.55))
    b = int(55 * (1 - ratio))
    return f'background-color: rgba({r},{g},{b},0.38); color: #f1f5f9; font-weight:600'


def render_slab_trend(d_start, d_end):
    slab_start = d_end - timedelta(days=10)
    tbl_key   = f"vslab_tbl_{d_end.strftime('%Y%m%d')}"
    click_key = f"vslab_click_{d_end.strftime('%Y%m%d')}"

    with st.expander("Offloading Time Slab", expanded=False):
        df = load_slab_trend(slab_start.strftime('%Y-%m-%d'), d_end.strftime('%Y-%m-%d'))
        if df is None or df.empty:
            st.info("No slab data for this range.")
            return

        raw_dates = list(df['mod_date'])
        disp = df.copy()
        disp['mod_date'] = pd.to_datetime(disp['mod_date']).dt.strftime('%d %b %Y (%a)')
        disp = disp.rename(columns={'mod_date': 'MOD Date'}).set_index('MOD Date')

        def _style(d):
            s = d.style.format("{:,}")
            for c in SLAB_COLS + ["Total"]:
                if c in d.columns:
                    mx = int(d[c].max())
                    fn = lambda v, cn=c, cm=mx: _slab_cell_style(v, cn, cm)
                    try:
                        s = s.map(fn, subset=[c])
                    except AttributeError:
                        s = s.applymap(fn, subset=[c])
            return s.set_properties(**{'text-align': 'center', 'font-size': '0.85rem'})

        st.dataframe(
            _style(disp),
            use_container_width=True,
            height=min(44 * (len(disp) + 1) + 38, 480),
            on_select="rerun",
            selection_mode="single-cell",
            key=tbl_key
        )

        sel_state = st.session_state.get(tbl_key, {})
        sel_data  = sel_state.get('selection', {}) if isinstance(sel_state, dict) else {}
        sel_rows  = sel_data.get('rows', [])
        sel_cols  = sel_data.get('columns', [])
        if sel_rows and 0 <= sel_rows[0] < len(raw_dates):
            st.session_state[click_key] = (
                raw_dates[sel_rows[0]],
                sel_cols[0] if (sel_cols and sel_cols[0] in SLAB_COLS) else None
            )

        dd_date, dd_slab = st.session_state.get(click_key, (None, None))
        if dd_date is not None:
            dd_str  = dd_date.strftime('%Y-%m-%d') if hasattr(dd_date, 'strftime') else str(dd_date)
            dd_disp = pd.to_datetime(dd_str).strftime('%d %b %Y (%a)')
            st.markdown(f"""
<div style="background:rgba(2,22,14,0.80);border-left:4px solid #059669;
            padding:6px 14px;margin-top:6px;border-radius:6px;">
    <span style="font-size:0.82rem;color:#34d399;font-weight:600;">
        🔍 Shop Drilldown — {dd_disp} &nbsp;|&nbsp; {dd_slab if dd_slab else "All Slabs"}
    </span>
</div>""", unsafe_allow_html=True)

            where_parts = [
                "dt_mod_date IS NOT NULL", "dt_doc_date IS NOT NULL",
                f"dt_mod_date::DATE = '{dd_str}'"
            ]
            if dd_slab and dd_slab in SLAB_SQL:
                where_parts.append(SLAB_SQL[dd_slab])

            conn = get_db_connection()
            if conn:
                try:
                    dd_df = pd.read_sql(f"""
                        SELECT
                            COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown')                              AS "Shop",
                            COUNT(*) FILTER (WHERE (dt_mod_date::DATE-dt_doc_date::DATE) < 0)              AS "Before WH Loaded",
                            COUNT(*) FILTER (WHERE (dt_mod_date::DATE-dt_doc_date::DATE) = 0)              AS "0 Days",
                            COUNT(*) FILTER (WHERE (dt_mod_date::DATE-dt_doc_date::DATE) BETWEEN 1 AND 3)  AS "1-3 Days",
                            COUNT(*) FILTER (WHERE (dt_mod_date::DATE-dt_doc_date::DATE) BETWEEN 4 AND 7)  AS "4-7 Days",
                            COUNT(*) FILTER (WHERE (dt_mod_date::DATE-dt_doc_date::DATE) BETWEEN 8 AND 10) AS "8-10 Days",
                            COUNT(*) FILTER (WHERE (dt_mod_date::DATE-dt_doc_date::DATE) > 10)             AS ">10 Days",
                            COUNT(*)                                                                        AS "Total"
                        FROM serial_no_dailydata
                        WHERE {' AND '.join(where_parts)}
                        GROUP BY vc_shop_code
                        ORDER BY "Total" DESC
                    """, conn)
                    conn.close()
                    if dd_df is not None and not dd_df.empty:
                        st.dataframe(dd_df, use_container_width=True, hide_index=True, height=280)
                    else:
                        st.info("No shop data for selected cell.")
                except Exception as ex:
                    try: conn.close()
                    except: pass
                    st.warning(f"Drilldown error: {ex}")


# ─────────────────────────────────────────────────────────────
# MAIN  — vertical pipeline + horizontal quality metrics strip
# ─────────────────────────────────────────────────────────────
def main():
    inject_css()

    # ── Date Filter ────────────────────────────────────────────
    today     = datetime.today().date()
    yesterday = today - timedelta(days=1)
    min_date  = datetime(2024, 1, 1).date()

    # Default to the most recent date with data in DB (not always yesterday)
    @st.cache_data(ttl=120, show_spinner=False)
    def _get_sn_max_date():
        try:
            conn = get_db_connection()
            if not conn:
                return None
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(DATE(bill_date)) FROM serialno_check_yes_no WHERE bill_date IS NOT NULL")
                row = cur.fetchone()
            conn.close()
            return row[0] if row and row[0] else None
        except Exception:
            return None

    _sn_max_date = _get_sn_max_date()
    _default_date = _sn_max_date if _sn_max_date else yesterday

    # ✅ FIX: Restore date from query params first (for navigation), then session state
    params = st.query_params
    qp_start = params.get('start_date')
    qp_end = params.get('end_date')

    if qp_start and qp_end:
        try:
            saved = (
                datetime.strptime(str(qp_start) if isinstance(qp_start, list) else qp_start, '%Y-%m-%d').date(),
                datetime.strptime(str(qp_end) if isinstance(qp_end, list) else qp_end, '%Y-%m-%d').date()
            )
        except:
            saved = st.session_state.get('vdate_range', (_default_date, _default_date))
    else:
        saved = st.session_state.get('vdate_range', (_default_date, _default_date))
    
    if not isinstance(saved, tuple) or len(saved) != 2:
        saved = (yesterday, yesterday)

    hdr_col, date_col, update_col, refresh_col = st.columns([0.60, 0.20, 0.12, 0.08])
    with date_col:
        st.markdown(
            '<div style="text-align:right;font-size:0.72rem;color:#6ee7b7;'
            'font-weight:600;margin-bottom:2px;">Date Range (GRN / Doc Date)</div>',
            unsafe_allow_html=True
        )
        selection = st.date_input(
            "vdate_pick", value=(saved[0], saved[1]),
            min_value=min_date, max_value=today,
            key="vdate_pick",
            label_visibility="collapsed"
        )

    with update_col:
        st.markdown('<div style="margin-top:18px;"></div>', unsafe_allow_html=True)
        _sn_already = False
        _sn = None
        _sn_target = yesterday          # default
        try:
            import sn_sync as _sn
            # File is named with today's date (e.g. shop_serial_no - 13-MAY-26.csv)
            # Try today first; fall back to yesterday if today's file doesn't exist
            _today_file = _sn.find_sn_file(today)
            _yest_file  = _sn.find_sn_file(yesterday)
            if _today_file:
                _sn_target = today
            elif _yest_file:
                _sn_target = yesterday
            else:
                _sn_target = today      # still show today if neither found

            _sn_already = _sn.date_exists_in_db(_sn_target)
        except Exception:
            _sn = None

        _btn_lbl  = "✅ Updated" if _sn_already else "⬆ Update Data"
        _btn_help = (f"Data for {_sn_target.strftime('%d %b')} already loaded. Click to force re-import."
                     if _sn_already else
                     f"Load {_sn_target.strftime('%d %b %Y')} from network share and send newsletter")
        if st.button(_btn_lbl, key="sn_update_btn", use_container_width=True, help=_btn_help):
            st.session_state["sn_sync_trigger"]    = True
            st.session_state["sn_sync_force"]      = _sn_already
            st.session_state["sn_sync_target_date"]= _sn_target

    with refresh_col:
        st.markdown('<div style="margin-top:18px;"></div>', unsafe_allow_html=True)
        if st.button("🔄", help="Clear cache & reload fresh data", key="refresh_cache"):
            canceled_count, terminated_count = cancel_dashboard_db_activity()
            st.session_state['refresh_db_cleanup_msg'] = (
                f"🔄 Refresh completed · DB cleanup: canceled {canceled_count} active query(s), terminated {terminated_count} idle session(s)."
            )
            st.cache_data.clear()
            st.rerun()

    # ── SN Sync trigger ────────────────────────────────────────────────────────
    if st.session_state.pop("sn_sync_trigger", False) and _sn is not None:
        _force       = st.session_state.pop("sn_sync_force", False)
        _target_date = st.session_state.pop("sn_sync_target_date", yesterday)
        _date_lbl    = _target_date.strftime("%d %b %Y")

        if not _force and _sn.date_exists_in_db(_target_date):
            st.info(f"Data for {_date_lbl} already loaded. Click **⬆ Update Data** again to force re-import.")
        else:
            with st.status(f"Updating serial data for {_date_lbl}…", expanded=True) as _st:
                try:
                    _st.write(f"🔍 Searching for file on \\\\10.10.0.30\\mis …")
                    _csv = _sn.find_sn_file(_target_date)
                    if not _csv:
                        _st.update(label=f"❌ File not found for {_date_lbl}", state="error")
                        st.error(f"shop_serial_no - {_target_date.strftime('%d-%b-%y').upper()}.csv not found on \\\\10.10.0.30\\mis")
                    else:
                        _st.write(f"📄 Found: {os.path.basename(_csv)}")
                        _ok, _msg, _rows = _sn.upload_sn_file(
                            _target_date, _csv,
                            status_fn=lambda m: _st.write(m)
                        )
                        if _ok:
                            _st.update(label=f"✅ {_rows:,} rows loaded for {_date_lbl}", state="complete")
                            _st.write("📧 Sending newsletter…")
                            _nl_ok, _nl_msg = _sn.send_sn_newsletter(_target_date)
                            if _nl_ok:
                                _st.write(f"📧 Newsletter sent ✓")
                            else:
                                _st.write(f"⚠ Newsletter error: {_nl_msg}")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            _st.update(label="❌ Upload failed", state="error")
                            st.error(f"❌ {_msg}")
                except Exception as _e:
                    _st.update(label="❌ Error", state="error")
                    st.error(str(_e))

    st.session_state.pop('refresh_db_cleanup_msg', None)  # clear silently — footer removed

    if isinstance(selection, tuple) and len(selection) == 2:
        d_start, d_end = selection
    else:
        d_start = d_end = selection if isinstance(selection, type(yesterday)) else yesterday
    if d_start > d_end:
        d_start, d_end = d_end, d_start
    st.session_state['vdate_range'] = (d_start, d_end)

    start_s  = d_start.strftime('%Y-%m-%d')
    end_s    = d_end.strftime('%Y-%m-%d')
    date_str = (d_end.strftime('%d %b %Y')
                if d_start == d_end
                else f"{d_start.strftime('%d %b %Y')} → {d_end.strftime('%d %b %Y')}")

    # ── Dashboard Title ────────────────────────────────────────
    with hdr_col:
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;padding:4px 0 6px 0;">
    <img src="https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg"
         style="height:34px;border-radius:50%;border:2px solid #3b82f6;">
    <div>
        <div style="font-size:1.3rem;font-weight:800;color:#f1f5f9;letter-spacing:0.02em;">
            MELCOM · Serial Pipeline — Vertical View
        </div>
        <div style="font-size:0.76rem;color:#64748b;">
            WH Receiving ↓ WH Loading ↓ Shop Receiving ↓ SHOP Compliance % &nbsp;|&nbsp; {date_str}
        </div>
    </div>
</div>""", unsafe_allow_html=True)

    # ── Load all data ──────────────────────────────────────────
    with st.spinner("Loading pipeline data…"):
        wr = load_wh_receiving(start_s, end_s)
        wl = load_wh_loading(start_s, end_s)
        sr = load_shop_receiving(start_s, end_s)
        truck_sum = load_truck_summary(start_s, end_s)
        ss = load_shop_selling(start_s, end_s)

    def gi(d, k): return int(d.get(k, 0)) if d else 0
    def gf(d, k): return float(d.get(k, 0)) if d else 0.0

    # ✅ OPTIMIZED: Removed eager pre-loading of all 22 drilldowns
    # Drilldowns now load on-demand when user clicks a metric (saves 95% of queries)
    drilldown_cache = {}  # Empty cache - drilldowns loaded lazily in render_drill_down_tab

    # ── Compute all stage values upfront ──────────────────────
    wr_total = gi(wr, 'total_received')
    wr_uniq  = gi(wr, 'unique_serials')
    wr_dup   = gi(wr, 'duplicate_count')
    wr_blank = gi(wr, 'blank_serials')
    wr_small = gi(wr, 'small_serials')
    wr_whdup = gi(wr, 'wh_db_duplicates')
    wr_upct  = gf(wr, 'unique_percentage')
    wr_ok    = (wr_total == 0) or (wr_dup == 0 and wr_blank == 0 and wr_small == 0 and wr_whdup == 0)

    wl_total = gi(wl, 'loaded_total')
    wl_uniq  = gi(wl, 'loaded_unique')
    wl_dup   = gi(wl, 'loaded_dup')
    wl_blank = gi(wl, 'loaded_blank')
    wl_small = gi(wl, 'loaded_small')
    wl_ic    = gi(wl, 'loaded_serial_is_ic')
    wl_upct  = gf(wl, 'unique_percentage')
    wl_ok    = (wl_total == 0) or (wl_dup == 0 and wl_blank == 0 and wl_small == 0 and wl_ic == 0)

    sr_total  = gi(sr, 'sr_total')
    sr_uniq   = gi(sr, 'sr_unique')
    sr_blank  = gi(sr, 'sr_blank')
    sr_small  = gi(sr, 'sr_small')
    sr_dup    = gi(sr, 'sr_dup')
    sr_ic     = gi(sr, 'sr_ic')
    sr_same   = gi(sr, 'same_day')
    sr_old    = gi(sr, 'old_doc')
    sr_match  = gi(sr, 'wh_match')
    sr_mmatch = gi(sr, 'wh_mismatch')
    sr_notoff = gi(sr, 'not_offloaded')
    sr_day = load_shop_receiving_day_breakdown(end_s)
    sr_day_total = gi(sr_day, 'total_offloaded')
    sr_day_same = gi(sr_day, 'same_day')
    sr_day_1_3 = gi(sr_day, 'days_1_3')
    sr_day_gt_3 = gi(sr_day, 'days_gt_3')
    sr_upct   = pct(sr_uniq, sr_total)
    mm_pct    = pct(sr_match, sr_match + sr_mmatch) if (sr_match + sr_mmatch) > 0 else 100.0
    sr_ok     = (sr_total == 0) or (sr_dup == 0 and sr_blank == 0 and sr_mmatch == 0 and sr_old == 0 and sr_notoff == 0)

    wh_truck_count = gi(truck_sum, 'wh_unique_trucks')
    shop_truck_count = gi(truck_sum, 'shop_unique_trucks')
    wh_truck_qty = gi(truck_sum, 'wh_truck_loaded_qty')
    shop_truck_off_qty = gi(truck_sum, 'shop_truck_offloaded_qty')

    ss_total = gi(ss, 'total_sold')
    ss_yes   = gi(ss, 'compliance_yes')
    ss_dup   = gi(ss, 'dup_sold')
    ss_noser = gi(ss, 'no_serial')
    ss_small = gi(ss, 'small_serial')
    ss_notwh = gi(ss, 'not_in_wh')
    ss_seqmiss = gi(ss, 'incorrect_serial_count')
    ss_ic    = gi(ss, 'serial_is_ic')
    ss_pct_v = pct(ss_yes, ss_total)
    ss_ok    = (ss_total == 0) or (ss_dup == 0 and ss_noser == 0 and ss_notwh == 0 and ss_seqmiss == 0 and ss_ic == 0 and ss_pct_v == 100.0)

    # Read query params once (drill only)
    # Note: params already read earlier for date range
    def _qp_value(val):
        if isinstance(val, list):
            return val[0] if val else None
        return val
    qp_stage = _qp_value(params.get("stage"))
    qp_metric = _qp_value(params.get("metric"))

    # ── Build stage rows ──────────────────────────────────────
    # ✅ FIX: Pass date range to _stage_panel for metric links
    date_range_tuple = (start_s, end_s)  # Current date range for preserving in links
    
    def _stage_panel(icon, title, subtitle, total_str, status_ok, accent,
                     grad_start, grad_end, rows_html, card_key, show_total=True):
        raw_total = html.unescape(str(total_str))
        
        # Check if this is a percentage value - preserve it as-is
        if '%' in raw_total:
            total_str = raw_total.strip()
            # Extract numeric part for badge logic (ignore % sign)
            try:
                total_num = float(re.sub(r'[^\d.]', '', raw_total))
            except Exception:
                total_num = 0
        else:
            if re.fullmatch(r"\s*[\d,]+\s*", raw_total):
                total_str = raw_total.strip()
            else:
                between_tags = re.search(r">\s*([\d][\d,]*)\s*<", raw_total)
                if between_tags:
                    total_str = between_tags.group(1)
                else:
                    plain_text = re.sub(r"<[^>]*>", " ", raw_total)
                    fallback = re.search(r"\b\d[\d,]*\b", plain_text)
                    total_str = fallback.group(0) if fallback else "0"

            try:
                total_num = int(total_str.replace(",", ""))
                total_str = fmt(total_num)
            except Exception:
                total_num = 0
                total_str = "0"

        if not show_total:
            badge_c   = "#10b981" if status_ok else "#ef4444"
            badge_lbl = "🟢 Clean" if status_ok else "🔴 Issues"
        elif total_num == 0:
            badge_c   = "#64748b"
            badge_lbl = "⚪ No Data"
        elif status_ok:
            badge_c   = "#10b981"
            badge_lbl = "🟢 Clean"
        else:
            badge_c   = "#ef4444"
            badge_lbl = "🔴 Issues"
        if subtitle:
            if "<" in str(subtitle) and ">" in str(subtitle):
                sub_html = f'<div style="margin-top:1px;">{subtitle}</div>'
            else:
                sub_html = f'<div style="font-size:0.58rem;color:#475569;margin-top:1px;">{subtitle}</div>'
        else:
            sub_html = ""
        total_html = (
            f'<div style="font-size:2.3rem;font-weight:900;color:#f8fafc;line-height:1.0;'
            f'margin:4px 0 7px;font-variant-numeric:tabular-nums;">{total_str}</div>'
            if show_total else ''
        )

        return (
            f'<div style="flex:1;min-width:0;background:linear-gradient(160deg,{grad_start} 0%,{grad_end} 100%);'
            f'border:1px solid {accent}28;border-radius:14px;overflow:hidden;">'
            f'<details style="margin:0;padding:0;" data-card="{card_key}">'
            f'<summary style="list-style:none;background:linear-gradient(90deg,{accent}1a,transparent);'
            f'border-bottom:1px solid {accent}18;padding:14px 14px 12px;text-align:center;cursor:pointer;outline:none;">'
            f'<div style="font-size:1.9rem;line-height:1.1;margin-bottom:3px;">{icon}</div>'
            f'<div style="font-size:0.60rem;letter-spacing:0.12em;text-transform:uppercase;'
            f'color:{accent};font-weight:800;margin-bottom:5px;">{title}</div>'
            f'{sub_html}'
            f'{total_html}'
            f'<div style="display:inline-block;background:{badge_c}1a;border:1px solid {badge_c}40;'
            f'border-radius:20px;padding:3px 12px;font-size:0.65rem;'
            f'color:{badge_c};font-weight:700;">{badge_lbl}</div>'
            f'<div style="font-size:0.58rem;color:#94a3b8;margin-top:6px;">Click card to expand/collapse</div>'
            f'</summary>'
            f'<div style="padding:10px 12px 12px;">{rows_html}</div>'
            f'</details>'
            f'</div>'
        )

    def _connector(from_c, to_c, label, count):
        return f"""
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
            flex-shrink:0;width:54px;padding-top:44px;gap:3px;">
  <div style="font-size:0.58rem;color:#334155;text-align:center;white-space:nowrap;
              line-height:1.3;">{label}<br><span style="color:#475569;font-weight:700;">{count}</span></div>
  <div style="position:relative;width:36px;height:2px;
              background:linear-gradient(90deg,{from_c},{to_c});border-radius:1px;">
    <div style="position:absolute;right:-7px;top:-5px;color:{to_c};
                font-size:0.75rem;line-height:1;">▶</div>
  </div>
</div>"""

    # ── Build each stage's rows ────────────────────────────────
    wr_rows = "".join([
       _row("Unique %",        f"{wr_upct:.1f}%",  wr_upct==100, BLUE_P[2],  wr_upct,
           stage="🏭 WH Receiving", metric="Unique %", date_range=date_range_tuple),
       _row("Unique Serials",  fmt(wr_uniq),        wr_uniq==wr_total, BLUE_P[2],
           pct(wr_uniq, wr_total) if wr_total else 100,
           note=f"/ {fmt(wr_total)}", stage="🏭 WH Receiving", metric="Unique Serials", date_range=date_range_tuple),
       _row("In-Batch Dup",    fmt(wr_dup),         wr_dup==0,  BLUE_P[2],
           100-pct(wr_dup, wr_total) if wr_dup else None,
           stage="🏭 WH Receiving", metric="In-Batch Dup", date_range=date_range_tuple),
       _row("WH DB Dup",       fmt(wr_whdup),       wr_whdup==0, BLUE_P[2],
           100-pct(wr_whdup, wr_total) if wr_whdup else None,
           stage="🏭 WH Receiving", metric="WH DB Dup", date_range=date_range_tuple),
       _row("Blank Serials",   fmt(wr_blank),       wr_blank==0, BLUE_P[2],
           100-pct(wr_blank, wr_total) if wr_blank else None,
           stage="🏭 WH Receiving", metric="Blank Serials", date_range=date_range_tuple),
       _row("Small (≤6)",      fmt(wr_small),       wr_small==0, BLUE_P[2],
           100-pct(wr_small, wr_total) if wr_small else None,
           stage="🏭 WH Receiving", metric="Small (≤6)", date_range=date_range_tuple),
    ])

    wl_rows = "".join([
        _row("Unique %",        f"{wl_upct:.1f}%",  wl_upct==100, TEAL_P[2],  wl_upct, 
              stage="📦 WH Loading", metric="Unique %", date_range=date_range_tuple,
              label_color="#c7fff2", value_color="#f8fafc"),
        _row("Duplicates",      fmt(wl_dup),         wl_dup==0,  TEAL_P[2],
             100-pct(wl_dup, wl_total) if wl_dup else None, 
               stage="📦 WH Loading", metric="Duplicates", date_range=date_range_tuple,
               label_color="#c7fff2", value_color="#f8fafc"),
        _row("Blank Serials",   fmt(wl_blank),       wl_blank==0, TEAL_P[2],
             100-pct(wl_blank, wl_total) if wl_blank else None, 
               stage="📦 WH Loading", metric="Blank Serials", date_range=date_range_tuple,
               label_color="#c7fff2", value_color="#f8fafc"),
        _row("Small (≤6)",      fmt(wl_small),       wl_small==0, TEAL_P[2],
             100-pct(wl_small, wl_total) if wl_small else None, 
               stage="📦 WH Loading", metric="Small (≤6)", date_range=date_range_tuple,
               label_color="#c7fff2", value_color="#f8fafc"),
       _row("IC",              fmt(wl_ic),          wl_ic==0, TEAL_P[2],
           100-pct(wl_ic, wl_total) if wl_ic else None,
           stage="📦 WH Loading", metric="IC", date_range=date_range_tuple),
    ])

    sr_rows = "".join([
        _row("Not Offloaded",   fmt(sr_notoff),      sr_notoff==0, GREEN_P[2],
             100-pct(sr_notoff, wl_total) if sr_notoff else None, 
             stage="🏪 Shop Receiving", metric="Not Offloaded", date_range=date_range_tuple),
        _row("Total Offloaded", fmt(sr_total),        True, GREEN_P[2],
               pct(sr_total, wl_total) if wl_total else 100,
               note=f"{pct(sr_total, wl_total):.1f}% of WH Loaded" if wl_total else None,
               stage="🏪 Shop Receiving", metric="Total Offloaded", date_range=date_range_tuple),
        _row("Same-Day",        fmt(sr_same),         True, GREEN_P[2],
             pct(sr_same, sr_total) if sr_total else 100, 
             stage="🏪 Shop Receiving", metric="Same-Day", date_range=date_range_tuple),
       _row("WH→Shop Mismatch",fmt(sr_mmatch),       sr_mmatch==0, GREEN_P[2],
           100-pct(sr_mmatch, sr_total) if sr_mmatch else None, 
           stage="🏪 Shop Receiving", metric="WH→Shop Mismatch", date_range=date_range_tuple),
        _row("Unique %",        f"{sr_upct:.1f}%",   sr_upct==100, GREEN_P[2],  sr_upct, 
            stage="🏪 Shop Receiving", metric="Unique %", date_range=date_range_tuple),
       _row("Blank",           fmt(sr_blank),        sr_blank==0, GREEN_P[2],
           100-pct(sr_blank, sr_total) if sr_blank else None),
       _row("Small (≤6)",      fmt(sr_small),        sr_small==0, GREEN_P[2],
           100-pct(sr_small, sr_total) if sr_small else None,
           stage="🏪 Shop Receiving", metric="Small (≤6)", date_range=date_range_tuple),
       _row("IC",              fmt(sr_ic),           sr_ic==0, GREEN_P[2],
           100-pct(sr_ic, sr_total) if sr_ic else None,
           stage="🏪 Shop Receiving", metric="IC", date_range=date_range_tuple),
       _row("Duplicates",      fmt(sr_dup),          sr_dup==0, GREEN_P[2],
           100-pct(sr_dup, sr_total) if sr_dup else None),
    ])

    ss_rows = "".join([
        _row("Compliance %",    f"{ss_pct_v:.1f}%",  ss_pct_v==100, VIOLET_P[2],  ss_pct_v,
             note=f"{fmt(ss_yes)}/{fmt(ss_total)}", 
             stage="🛒 Shop Compliance %", metric="Compliance %", date_range=date_range_tuple),
        _row("Dup Serials",     fmt(ss_dup),          ss_dup==0, VIOLET_P[2],
             100-pct(ss_dup, ss_total) if ss_dup else None, 
             stage="🛒 Shop Compliance %", metric="Dup Serials", date_range=date_range_tuple),
        _row("No Serial",       fmt(ss_noser),        ss_noser==0, VIOLET_P[2],
             100-pct(ss_noser, ss_total) if ss_noser else None, 
             stage="🛒 Shop Compliance %", metric="No Serial", date_range=date_range_tuple),
        _row("Not in WH",       fmt(ss_notwh),        ss_notwh==0, VIOLET_P[2],
             100-pct(ss_notwh, ss_total) if ss_notwh else None, 
             stage="🛒 Shop Compliance %", metric="Not in WH", date_range=date_range_tuple),
           _row("Check Serial No", fmt(ss_seqmiss), ss_seqmiss==0, VIOLET_P[2],
             100-pct(ss_seqmiss, ss_total) if ss_seqmiss else None, 
               stage="🛒 Shop Compliance %", metric="Check Serial No", date_range=date_range_tuple),
        _row("IC",              fmt(ss_ic),           ss_ic==0, VIOLET_P[2],
             100-pct(ss_ic, ss_total) if ss_ic else None, 
             stage="🛒 Shop Compliance %", metric="Serial = IC", date_range=date_range_tuple),
        _row("Small (≤6)",      fmt(ss_small),        ss_small==0, VIOLET_P[2],
             100-pct(ss_small, ss_total) if ss_small else None, 
             stage="🛒 Shop Compliance %", metric="Small (≤6)", date_range=date_range_tuple),
    ])

    # ── Assemble infographic ───────────────────────────────────
    wl_subtitle = (
        f'<div style="font-size:0.64rem;color:#e2e8f0;font-weight:700;line-height:1.2;">'
        f'by doc date · 🚚 WH trucks loaded: <span style="color:#f8fafc;font-weight:800;">{fmt(wh_truck_count)}</span>'
        f'</div>'
    )

    sr_subtitle = (
        f'<div style="display:flex;align-items:center;justify-content:center;gap:10px;">'
        f'  <span style="font-size:1.14rem;color:#d1fae5;font-weight:900;line-height:1;">🚚 {fmt(wh_truck_count)}</span>'
        f'  <span style="font-size:1.20rem;color:#86efac;font-weight:900;line-height:1;">→</span>'
        f'  <span style="font-size:1.14rem;color:#dcfce7;font-weight:900;line-height:1;">🏪 {fmt(shop_truck_count)}</span>'
        f'</div>'
        f'<div style="font-size:0.56rem;color:#a7f3d0;margin-top:2px;font-weight:700;">WH trucks loaded → trucks reached shop</div>'
        f'<div style="font-size:0.84rem;color:#f0fdf4;margin-top:2px;font-weight:900;line-height:1.1;">'
        f'qty carried: <span style="color:#86efac;">{fmt(wh_truck_qty)}</span>'
        f' · qty offloaded: <span style="color:#bbf7d0;">{fmt(shop_truck_off_qty)}</span>'
        f'</div>'
    )

    infographic = (
        '<div style="display:flex;align-items:flex-start;gap:0;width:100%;'
        'margin-top:10px;overflow-x:auto;">'
        + _stage_panel("🏭", "WH RECEIVING", "", wr_total, wr_ok,
                       BLUE_P[2], "#0c1a3a", "#080f22",
                       wr_rows,
                       "wr")
        + '<div style="width:54px;flex-shrink:0;"></div>'
        + _stage_panel("📦", "WH LOADING", wl_subtitle, wl_total, wl_ok,
                       TEAL_P[2], "#06222e", "#03141a",
                       wl_rows,
                       "wl")
        + _connector(TEAL_P[2], GREEN_P[2], "Offloaded", fmt(sr_total))
        + _stage_panel("🏪", "SHOP RECEIVING", sr_subtitle,
                   sr_total, sr_ok,
                       GREEN_P[2], "#042a1c", "#021610",
                       sr_rows,
                       "sr", show_total=False)
        + _connector(GREEN_P[2], VIOLET_P[2], "Sold", fmt(ss_total))
        + _stage_panel("🛒", "SHOP Compliance %", f"by bill date · Total: {fmt(ss_total)}",
                       f"{ss_pct_v:.1f}%", ss_ok,
                       VIOLET_P[2], "#160d2e", "#0d0818",
                       ss_rows,
                       "ss")
        + '</div>'
    )

    # ── Initialize session state for hidden drilldown tab ────────
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'pipeline'
    if 'selected_stage' not in st.session_state:
        st.session_state.selected_stage = None
    if 'selected_metric' not in st.session_state:
        st.session_state.selected_metric = None

    if qp_stage and qp_metric:
        st.session_state.current_view = 'drill'
        st.session_state.selected_stage = unquote(qp_stage)
        st.session_state.selected_metric = unquote(qp_metric)
        st.query_params.clear()
    elif st.session_state.current_view != 'drill':
        st.session_state.current_view = 'pipeline'
        st.session_state.selected_stage = None
        st.session_state.selected_metric = None

    # ══════════════════════════════════════════════════════════
    # MONTHLY OFFLOADING % TREND
    # ══════════════════════════════════════════════════════════
    render_monthly_offloading_chart(start_s, end_s)

    # ── VERTICAL PIPELINE SECTION (User selection) ───────────
    st.markdown("### 📊 Vertical Pipeline")

    # ── CONDITIONAL RENDERING: Pipeline or Drill-Down Tab ────────
    if st.session_state.current_view == 'pipeline':
        st.markdown(infographic, unsafe_allow_html=True)
    else:
        # Keep the same pipeline block on drill pages for metric switching
        st.markdown(infographic, unsafe_allow_html=True)
        st.markdown('<div id="drill-section"></div>', unsafe_allow_html=True)
        render_drill_down_tab(
            st.session_state.selected_stage,
            st.session_state.selected_metric,
            drilldown_cache,
            start_s,
            end_s,
        )


    # ══════════════════════════════════════════════════════════
    # EDA INTELLIGENCE CENTER
    # ══════════════════════════════════════════════════════════
    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    render_eda_intelligence(start_s, end_s)

    # ══════════════════════════════════════════════════════════
    # SHOP-WISE COMPLIANCE % — MONTH-ON-MONTH TABLE
    # ══════════════════════════════════════════════════════════
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    render_shop_compliance_pivot(end_s)
    
    


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
