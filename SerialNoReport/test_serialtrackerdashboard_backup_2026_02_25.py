"""
test_serialtrackerdashboard.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Revamped single-view Serial Tracker Dashboard.
Four pipeline stages visible in one scroll:
  WH Receiving → WH Loading → Shop Receiving → Shop Selling
"""

import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
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

def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
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
            SELECT serial_no
            FROM whreceived_serialno
            WHERE grn_date BETWEEN %(s)s AND %(e)s
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
def load_wh_loading(start_str: str, end_str: str):
    conn = get_db_connection()
    if not conn:
        return {}
    q = """
    SELECT
        COUNT(*)                                                                 AS loaded_total,
        COUNT(DISTINCT serial_no) FILTER (
            WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
        )                                                                        AS loaded_unique,
        COUNT(*) FILTER (
            WHERE serial_no IS NULL OR LENGTH(TRIM(COALESCE(serial_no,''))) = 0
        )                                                                        AS loaded_blank,
                COUNT(*) FILTER (
                        WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
                            AND LENGTH(TRIM(serial_no)) <= 6
                )                                                                        AS loaded_small,
        COUNT(*) FILTER (
            WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
              AND REGEXP_REPLACE(TRIM(serial_no), '[^A-Za-z0-9]','','g') != ''
              AND REGEXP_REPLACE(TRIM(serial_no), '[^A-Za-z0-9]','','g') =
                  REGEXP_REPLACE(TRIM(vc_item_code::TEXT), '[^A-Za-z0-9]','','g')
        )                                                                        AS loaded_serial_is_ic,
        COUNT(*) FILTER (
            WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
        ) - COUNT(DISTINCT serial_no) FILTER (
            WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
        )                                                                        AS loaded_dup,
        CASE
            WHEN COUNT(*) > 0
            THEN ROUND(COUNT(DISTINCT serial_no) FILTER (
                WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
            )::NUMERIC / COUNT(*) * 100, 2)
            ELSE 0
        END                                                                      AS unique_percentage
    FROM serial_no_dailydata
    WHERE dt_doc_date::DATE BETWEEN %(s)s AND %(e)s
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
def load_shop_receiving_pct_trend(end_str: str):
    """Trend: For each loaded date, % with MOD_date present out of WH loaded on same date."""
    conn = get_db_connection()
    if not conn:
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
def load_shop_selling(start_str: str, end_str: str):
    conn = get_db_connection()
    if not conn:
        return {}
    q = """
    WITH base AS (
        SELECT item_code, serial_number, serial_check
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
        st.error(f"Shop Selling query error: {ex}")
        try: conn.close()
        except: pass
        return {}


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
                COUNT(*) FILTER (WHERE dt_mod_date IS NOT NULL) AS shop_received
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


def render_monthly_offloading_chart(start_s: str, end_s: str):
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align:center;margin:0 0 6px 0;'>📈 Monthly Offloading % (WH Loaded vs Shop Received)</h3>",
        unsafe_allow_html=True,
    )
    st.caption("Formula: (Rows with MOD_date present ÷ Rows loaded by WH on DOC date month) × 100")

    monthly_df = load_monthly_offloading_pct(start_s, end_s)
    if monthly_df is None or monthly_df.empty:
        st.info("No monthly offloading data for selected year-to-date range.")
        return

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
            "<br>Shop Received (MOD_date present): %{customdata[1]}<extra></extra>"
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
        height=280,
        yaxis_title="Offloading %",
        xaxis_title="Month",
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(26,40,71,0.3)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11, color="#FFFFFF"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(26,40,71,0.95)", font_size=12,
                        font_family="Inter", font_color="white"),
        xaxis=dict(gridcolor="rgba(91,84,255,0.1)", showgrid=True),
        yaxis=dict(autorange=True, gridcolor="rgba(91,84,255,0.1)", showgrid=True)
    )
    g_left, g_mid, g_right = st.columns([0.2, 0.30, 0.50])
    with g_mid:
        st.plotly_chart(fig, use_container_width=True, key="monthly_offloading_pct")


@st.cache_data(ttl=300)
def load_wh_loading_drilldown(start_str: str, end_str: str, metric: str):
    """Drilldown: WH Loading by loader + daily breakdown for last 7 days."""
    conn = get_db_connection()
    if not conn:
        return None
    # ✅ OPTIMIZED: Using DISTINCT ON instead of ROW_NUMBER() window function
    q = f"""
    WITH daily_data AS (
        SELECT
            dt_doc_date::DATE AS doc_date,
            COALESCE(NULLIF(TRIM(wh_load_user),''),'Unknown') AS loader,
            COUNT(*) AS total_records,
            COUNT(DISTINCT serial_no) FILTER (WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0) AS unique_serials,
            COUNT(*) FILTER (WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0) AS loaded_with_serial,
            COUNT(*) FILTER (WHERE LENGTH(TRIM(COALESCE(serial_no,''))) <= 6 AND serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0) AS small_serials,
            COUNT(*) FILTER (WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0) - 
            COUNT(DISTINCT serial_no) FILTER (WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0) AS duplicates,
            COUNT(*) FILTER (WHERE serial_no IS NULL OR LENGTH(TRIM(serial_no)) = 0) AS blank_count
        FROM serial_no_dailydata
        WHERE dt_doc_date::DATE BETWEEN ('{end_str}'::DATE - 6) AND '{end_str}'::DATE
        GROUP BY dt_doc_date::DATE, COALESCE(NULLIF(TRIM(wh_load_user),''),'Unknown')
    )
    SELECT
        loader,
        doc_date,
        CASE
            WHEN '{metric}' = 'Unique %' THEN ROUND(unique_serials::NUMERIC / NULLIF(total_records, 0) * 100, 1)::TEXT || '%'
            WHEN '{metric}' = 'Duplicates' THEN duplicates::TEXT
            WHEN '{metric}' = 'Blank Serials' THEN blank_count::TEXT
            WHEN '{metric}' = 'Small (≤6)' THEN small_serials::TEXT
            ELSE total_records::TEXT
        END AS metric_value
    FROM daily_data
    WHERE doc_date BETWEEN ('{end_str}'::DATE - 6) AND '{end_str}'::DATE
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
def load_wh_loading_shop_drilldown(end_str: str, metric: str):
    """WH Loading drilldown: shop-code wise previous 7 days."""
    conn = get_db_connection()
    if not conn:
        return None

    q = f"""
    WITH daily_data AS (
        SELECT
            dt_doc_date::DATE AS doc_date,
            COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
            COUNT(*) AS total_records,
            COUNT(DISTINCT serial_no) FILTER (
                WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
            ) AS unique_serials,
            COUNT(*) FILTER (
                WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
            ) AS loaded_with_serial,
            COUNT(*) FILTER (
                WHERE serial_no IS NULL OR LENGTH(TRIM(COALESCE(serial_no,''))) = 0
            ) AS blank_count,
            COUNT(*) FILTER (
                WHERE serial_no IS NOT NULL AND LENGTH(TRIM(COALESCE(serial_no,''))) <= 6
            ) AS small_serials,
            COUNT(*) FILTER (
                WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
            ) - COUNT(DISTINCT serial_no) FILTER (
                WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
            ) AS duplicates
        FROM serial_no_dailydata
        WHERE dt_doc_date::DATE BETWEEN ('{end_str}'::DATE - 6) AND '{end_str}'::DATE
        GROUP BY dt_doc_date::DATE, vc_shop_code
    )
    SELECT
        shop_code,
        doc_date,
        CASE
            WHEN '{metric}' = 'Unique %' THEN ROUND(unique_serials::NUMERIC / NULLIF(total_records, 0) * 100, 1)::TEXT || '%'
            WHEN '{metric}' = 'Duplicates' THEN duplicates::TEXT
            WHEN '{metric}' = 'Blank Serials' THEN blank_count::TEXT
            WHEN '{metric}' = 'Small (≤6)' THEN small_serials::TEXT
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
def load_shop_receiving_drilldown(start_str: str, end_str: str, metric: str):
    """Drilldown: Shop Receiving by shop + daily breakdown."""
    conn = get_db_connection()
    if not conn:
        return None
    if metric == "Old Doc Date":
        q = f"""
        WITH daily_data AS (
            SELECT
                                (dt_mod_date::DATE) AS mod_date,
                COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
                COALESCE(shop_name, 'Unknown Shop') AS shop_name,
                COUNT(*) FILTER (
                                        WHERE dt_doc_date IS NOT NULL
                                            AND dt_doc_date::DATE < dt_mod_date::DATE
                ) AS old_doc_date
            FROM serial_no_dailydata
                        WHERE dt_mod_date IS NOT NULL
                            AND dt_mod_date::DATE BETWEEN ('{end_str}'::DATE - 6) AND '{end_str}'::DATE
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
                COUNT(*) FILTER (WHERE serial_no IS NULL OR LENGTH(TRIM(serial_no)) = 0) AS no_serial
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
                WHEN '{metric}' = 'WH→Shop Mismatch' THEN no_serial::TEXT
                ELSE total_offloaded::TEXT
            END AS metric_value
        FROM daily_data
        ORDER BY shop_code, mod_date DESC
        """
    try:
        df = pd.read_sql(q, conn)
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
def load_shop_selling_drilldown(start_str: str, end_str: str, metric: str):
    """Drilldown: Shop Selling by shop + daily breakdown."""
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
        st.error(f"Shop Selling drilldown error: {ex}")
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
    """Drilldown: Shop Selling by shop & cashier for selected metric."""
    conn = get_db_connection()
    if not conn:
        return None

    q = f"""
    WITH daily_data AS (
        SELECT
            DATE(bill_date) AS bill_date,
            COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
            COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown') AS cashier,
            COUNT(*) AS total_sold,
            COUNT(*) FILTER (WHERE TRIM(serial_check) = 'Y') AS compliance_yes,
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
                WHERE serial_number IS NOT NULL
                  AND LENGTH(TRIM(COALESCE(serial_number,''))) > 0
                  AND LENGTH(TRIM(COALESCE(serial_number,''))) <= 6
            ) AS small_serial,
            COUNT(DISTINCT serial_number) FILTER (
                WHERE serial_number IS NOT NULL AND TRIM(COALESCE(serial_number,'')) != ''
            ) AS unique_serials,
            COUNT(*) FILTER (
                WHERE serial_number IS NOT NULL AND TRIM(COALESCE(serial_number,'')) != ''
            ) AS with_serial,
            COUNT(*) FILTER (
                WHERE serial_number IS NOT NULL
                  AND TRIM(COALESCE(serial_number,'')) != ''
                AND REGEXP_REPLACE(TRIM(serial_number), '[^A-Za-z0-9]','','g') != ''
                AND REGEXP_REPLACE(TRIM(serial_number), '[^A-Za-z0-9]','','g') = REGEXP_REPLACE(TRIM(item_code), '[^A-Za-z0-9]','','g')
            ) AS serial_is_ic
        FROM serialno_check_yes_no
        WHERE DATE(bill_date) BETWEEN %(s)s AND %(e)s
          AND COALESCE(item_code,'') != 'sales data not available'
        GROUP BY shop_code, cashier_name, DATE(bill_date)
    )
    SELECT
        shop_code,
        cashier,
        bill_date,
        CASE
            WHEN '{metric}' = 'Compliance %' THEN compliance_pct::TEXT || '%'
            WHEN '{metric}' = 'Not in WH' THEN not_in_wh::TEXT
            WHEN '{metric}' = 'Dup Serials' THEN (with_serial - unique_serials)::TEXT
            WHEN '{metric}' = 'No Serial' THEN no_serial::TEXT
            WHEN '{metric}' = 'Serial = IC' THEN serial_is_ic::TEXT
            WHEN '{metric}' = 'Small (≤6)' THEN small_serial::TEXT
            ELSE total_sold::TEXT
        END AS metric_value
    FROM daily_data
    ORDER BY shop_code, cashier, bill_date DESC
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
    """Trend: daily totals for Shop Selling metrics."""
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
        ) AS serial_is_ic
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
    if metric == "Compliance %":
        metric_filter = ""
    elif metric == "Dup Serials":
        metric_filter = "AND b.serial_number IN (SELECT serial_number FROM dup_sn)"
    elif metric == "No Serial":
        metric_filter = "AND (serial_number IS NULL OR TRIM(COALESCE(serial_number,'')) = '')"
    elif metric == "Not in WH":
        metric_filter = "AND TRIM(serial_check) = 'N' AND serial_number IS NOT NULL AND TRIM(COALESCE(serial_number,'')) != ''"
    elif metric == "Serial = IC":
        metric_filter = "AND serial_number IS NOT NULL AND TRIM(COALESCE(serial_number,'')) != '' AND REGEXP_REPLACE(TRIM(serial_number), '[^A-Za-z0-9]','','g') = REGEXP_REPLACE(TRIM(item_code), '[^A-Za-z0-9]','','g')"
    elif metric == "Small (≤6)":
        metric_filter = "AND serial_number IS NOT NULL AND LENGTH(TRIM(COALESCE(serial_number,''))) <= 6"

    q = f"""
    WITH src AS (
        SELECT
            serial_check,
            serial_number,
            item_code,
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
        b.bill_date,
        b.shop_code,
        b.cashier AS cashier_name
    FROM base b
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
        SELECT serial_number FROM base
        WHERE serial_number IS NOT NULL AND TRIM(serial_number) != ''
        GROUP BY serial_number HAVING COUNT(*) > 1
    )
    SELECT
        TRIM(b.serial_check) AS serial_check,
        b.serial_number,
        b.item_code,
        b.bill_date,
        b.shop_code,
        b.cashier AS cashier_name
    FROM base b
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


def _row(label, value_str, ok, accent, bar_pct=None, note=None, stage=None, metric=None, date_range=None):
    """Single metric row: dot + label + optional bar + value + optional note.
    If stage and metric provided, value becomes clickable (keeps original color, underline on hover).
    date_range: tuple of (start_str, end_str) to preserve dates in navigation."""
    dot_c  = "#10b981" if ok else "#ef4444"
    val_c  = "#6ee7b7" if ok else "#fca5a5"
    lbl_c  = "#64748b" if ok else "#94a3b8"
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
        f'border-bottom:1px solid rgba(255,255,255,0.04);">'
        f'<span style="width:6px;height:6px;border-radius:50%;background:{dot_c};flex-shrink:0;"></span>'
        f'<span style="flex:1;font-size:0.67rem;color:{lbl_c};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{label}</span>'
        f'{bar_html}'
        f'{value_html}'
        f'{note_html}'
        f'</div>'
    )


def render_drill_down_tab(stage, metric, drilldown_cache, start_s, end_s):
    stage_order = ["🏭 WH Receiving", "📦 WH Loading", "🏪 Shop Receiving", "🛒 Shop Selling"]
    stage_metric_order = {
        "🏭 WH Receiving": [
            "Unique %", "Unique Serials", "In-Batch Dup",
            "WH DB Dup", "Blank Serials", "Small (≤6)"
        ],
        "📦 WH Loading": ["Unique %", "Duplicates", "Blank Serials", "Small (≤6)"],
        "🏪 Shop Receiving": [
            "Not Offloaded", "Total Offloaded", "Same-Day", "Old Doc Date",
            "WH→Shop Mismatch", "Unique %", "Blank", "Small (≤6)", "Duplicates"
        ],
        "🛒 Shop Selling": [
            "Compliance %", "Dup Serials", "No Serial", "Not in WH",
            "Serial = IC", "Small (≤6)"
        ],
    }

    if not stage:
        st.info("Select a metric to open drill-down view.")
        return

    if stage == "🛒 Shop Selling":
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
        "🛒 Shop Selling": ("ss", "SS"),
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
        elif stage == "🛒 Shop Selling":
            df = load_shop_selling_drilldown(start_s, end_s, selected_metric)
        
        # Cache the loaded data
        if df is not None and not df.empty:
            drilldown_cache[cache_key] = df
    
    if (df is None or df.empty) and stage != "🛒 Shop Selling":
        st.warning("No drill data available for this metric.")
        return

    if df is None:
        df = pd.DataFrame()

    if stage == "🛒 Shop Selling" and not df.empty and 'cashier' in df.columns:
        preferred = [c for c in ['cashier', 'shop_code'] if c in df.columns]
        others = [c for c in df.columns if c not in preferred]
        df = df[preferred + others]
    elif stage == "📦 WH Loading" and not df.empty and 'loader' in df.columns:
        preferred = ['loader']
        others = [c for c in df.columns if c not in preferred]
        df = df[preferred + others]

    def _render_styled_table(table_df, key_suffix: str, title_text: str = None, height: int = 420, enable_select: bool = False, enable_filter: bool = True):
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

        if has_percent and value_cols_local:
            styled_df = styled_df.applymap(_percent_band_style, subset=value_cols_local)

        if stage == "📦 WH Loading" and selected_metric == "Duplicates" and value_cols_local:
            col_max_map = {}
            col_min_map = {}
            for col in value_cols_local:
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

            styled_df = styled_df.apply(_apply_dup_style, subset=value_cols_local)

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

        if enable_select:
            event = st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                height=height,
                key=safe_key,
                on_select="rerun",
                selection_mode="single-cell",
            )
        else:
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=height, key=safe_key)
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
                if row_idx < len(display_df) and col_name in display_df.columns:
                    return {
                        "row_idx": row_idx,
                        "col_idx": display_df.columns.get_loc(col_name),
                        "row": display_df.iloc[row_idx].to_dict(),
                        "col_name": col_name,
                    }

        return None

    def load_wh_loading_detail(metric: str, loader: str = None, shop_code: str = None, date_str: str = None,
                               start_str: str = None, end_str: str = None):
        conn = get_db_connection()
        if not conn:
            return None

        if date_str:
            date_filter = "doc_date = %(d)s"
            params = {"d": date_str}
        else:
            date_filter = "doc_date BETWEEN %(s)s AND %(e)s"
            params = {"s": start_str, "e": end_str}

        loader_filter = ""
        if loader:
            loader_filter = "AND loader = %(loader)s"
            params["loader"] = loader

        shop_filter = ""
        if shop_code:
            shop_filter = "AND shop_code = %(shop)s"
            params["shop"] = shop_code

        metric_filter = ""
        if metric == "Duplicates":
            metric_filter = "AND TRIM(serial_no) IN (SELECT serial_no FROM dupes)"
        elif metric == "Blank Serials":
            metric_filter = "AND (serial_no IS NULL OR LENGTH(TRIM(serial_no)) = 0)"
        elif metric == "Small (≤6)":
            metric_filter = "AND serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) <= 6"
        elif metric == "Unique %":
            metric_filter = "AND serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0"

        q = f"""
        WITH base AS (
            SELECT
                dt_doc_date::DATE AS doc_date,
                COALESCE(NULLIF(TRIM(wh_load_user),''),'Unknown') AS loader,
                COALESCE(NULLIF(TRIM(vc_wh_code),''),'Unknown') AS wh_code,
                COALESCE(NULLIF(TRIM(vc_shop_code),''),'Unknown') AS shop_code,
                vc_item_code,
                vc_item_desc,
                serial_no,
                dt_mod_date,
                dt_invoice_date,
                shop_sold
            FROM serial_no_dailydata
            WHERE dt_doc_date IS NOT NULL
        ),
        dupes AS (
            SELECT doc_date, TRIM(serial_no) AS serial_no
            FROM base
            WHERE serial_no IS NOT NULL AND LENGTH(TRIM(serial_no)) > 0
            GROUP BY doc_date, TRIM(serial_no)
            HAVING COUNT(*) > 1
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
        WHERE {date_filter}
            {loader_filter}
            {shop_filter}
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

    if totals.empty and stage == "🛒 Shop Selling":
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
                    st.markdown('<div class="drill-card">', unsafe_allow_html=True)
                    st.info("No data for Shop Received % trend.")
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
            else:
                _render_styled_table(df, f"drill_{stage}_{selected_metric}")
        elif stage == "📦 WH Loading":
            shop_df = load_wh_loading_shop_drilldown(end_s, selected_metric)
            compact_cols = st.columns([1.0, 1.2, 1.2])
            with compact_cols[0]:
                _render_card_chart(fig)
            with compact_cols[1]:
                loader_sel = _render_styled_table(
                    df,
                    f"drill_loader_{stage}_{selected_metric}",
                    "Loader (Previous 7 Days)",
                    height=360,
                    enable_select=True
                )
            with compact_cols[2]:
                if shop_df is not None and not shop_df.empty:
                    shop_sel = _render_styled_table(
                        shop_df,
                        f"drill_shopwise_{stage}_{selected_metric}",
                        "Shop Code (Previous 7 Days)",
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
                    scope_label = "all dates shown"
                else:
                    date_str = str(col_name)
                    scope_label = date_str

                detail_df = load_wh_loading_detail(
                    selected_metric,
                    loader=loader_val,
                    shop_code=shop_val,
                    date_str=date_str,
                    start_str=(pd.to_datetime(end_s).date() - timedelta(days=6)).strftime("%Y-%m-%d"),
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
        elif stage == "🛒 Shop Selling":
            last7_start = (pd.to_datetime(end_s).date() - timedelta(days=6)).strftime("%Y-%m-%d")

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
            elif step == "detail":
                if st.button("← Back to Cashiers", key=f"ss_back_cashier_{selected_metric}"):
                    st.session_state["ss_step"] = "cashier"
                    st.session_state["ss_date"] = None
                    st.rerun()

            if step == "shop":
                shop_df = load_shop_selling_shop_metric(selected_metric, last7_start, end_s)
                if shop_df is None or shop_df.empty:
                    st.info("No shop data for selected metric.")
                else:
                    shop_sel = _render_styled_table(
                        shop_df,
                        f"ss_shop_{selected_metric}_{end_s}",
                        f"Shop Code — {selected_metric} (Last 7 Days)",
                        height=320,
                        enable_select=True,
                        enable_filter=(selected_metric != "Compliance %")
                    )
                    if shop_sel:
                        st.session_state["ss_shop"] = shop_sel["row"].get("shop_code")
                        st.session_state["ss_step"] = "cashier"
                        st.rerun()

            elif step == "cashier":
                selected_shop = st.session_state.get("ss_shop")
                cashier_df = load_shop_selling_cashier_metric(selected_shop, selected_metric, last7_start, end_s)
                if cashier_df is None or cashier_df.empty:
                    st.info("No cashier data for selected shop.")
                else:
                    cashier_sel = _render_styled_table(
                        cashier_df,
                        f"ss_cashier_{selected_shop}_{selected_metric}_{end_s}",
                        f"Cashier — {selected_metric} (Last 7 Days) — {selected_shop}",
                        height=360,
                        enable_select=True,
                        enable_filter=(selected_metric != "Compliance %")
                    )
                    if cashier_sel:
                        cashier_name = cashier_sel["row"].get("cashier")
                        date_col = cashier_sel["col_name"]
                        if cashier_name and date_col and date_col != "cashier":
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

    with st.expander(
        f"⏱ Offloading Time Slab Trend  |  MOD Date: "
        f"{slab_start.strftime('%d %b %Y')} → {d_end.strftime('%d %b %Y')}",
        expanded=False
    ):
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
    yesterday = (datetime.today() - timedelta(days=1)).date()
    min_date  = datetime(2024, 1, 1).date()

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
            saved = st.session_state.get('vdate_range', (yesterday, yesterday))
    else:
        saved = st.session_state.get('vdate_range', (yesterday, yesterday))
    
    if not isinstance(saved, tuple) or len(saved) != 2:
        saved = (yesterday, yesterday)

    hdr_col, date_col, refresh_col = st.columns([0.72, 0.20, 0.08])
    with date_col:
        st.markdown(
            '<div style="text-align:right;font-size:0.72rem;color:#6ee7b7;'
            'font-weight:600;margin-bottom:2px;">Date Range (GRN / Doc Date)</div>',
            unsafe_allow_html=True
        )
        selection = st.date_input(
            "vdate_pick", value=(saved[0], saved[1]),
            min_value=min_date, max_value=yesterday,
            key="vdate_pick",
            label_visibility="collapsed"
        )

    with refresh_col:
        st.markdown('<div style="margin-top:18px;"></div>', unsafe_allow_html=True)
        if st.button("🔄", help="Clear cache & reload fresh data", key="refresh_cache"):
            st.cache_data.clear()
            st.rerun()

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
            WH Receiving ↓ WH Loading ↓ Shop Receiving ↓ Shop Selling &nbsp;|&nbsp; {date_str}
        </div>
    </div>
</div>""", unsafe_allow_html=True)

    # ── Load all data ──────────────────────────────────────────
    with st.spinner("Loading pipeline data…"):
        wr = load_wh_receiving(start_s, end_s)
        wl = load_wh_loading(start_s, end_s)
        sr = load_shop_receiving(start_s, end_s)
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

    ss_total = gi(ss, 'total_sold')
    ss_yes   = gi(ss, 'compliance_yes')
    ss_dup   = gi(ss, 'dup_sold')
    ss_noser = gi(ss, 'no_serial')
    ss_small = gi(ss, 'small_serial')
    ss_notwh = gi(ss, 'not_in_wh')
    ss_ic    = gi(ss, 'serial_is_ic')
    ss_pct_v = pct(ss_yes, ss_total)
    ss_ok    = (ss_total == 0) or (ss_dup == 0 and ss_noser == 0 and ss_notwh == 0 and ss_ic == 0 and ss_pct_v == 100.0)

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
                     grad_start, grad_end, rows_html, card_key):
        raw_total = html.unescape(str(total_str))
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

        if total_num == 0:
            badge_c   = "#64748b"
            badge_lbl = "⚪ No Data"
        elif status_ok:
            badge_c   = "#10b981"
            badge_lbl = "🟢 Clean"
        else:
            badge_c   = "#ef4444"
            badge_lbl = "🔴 Issues"
        sub_html = (f'<div style="font-size:0.58rem;color:#475569;margin-top:1px;">{subtitle}</div>'
                    if subtitle else "")
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
            f'<div style="font-size:2.3rem;font-weight:900;color:#f8fafc;line-height:1.0;'
            f'margin:4px 0 7px;font-variant-numeric:tabular-nums;">{total_str}</div>'
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
            stage="📦 WH Loading", metric="Unique %", date_range=date_range_tuple),
        _row("Duplicates",      fmt(wl_dup),         wl_dup==0,  TEAL_P[2],
             100-pct(wl_dup, wl_total) if wl_dup else None, 
             stage="📦 WH Loading", metric="Duplicates", date_range=date_range_tuple),
        _row("Blank Serials",   fmt(wl_blank),       wl_blank==0, TEAL_P[2],
             100-pct(wl_blank, wl_total) if wl_blank else None, 
             stage="📦 WH Loading", metric="Blank Serials", date_range=date_range_tuple),
        _row("Small (≤6)",      fmt(wl_small),       wl_small==0, TEAL_P[2],
             100-pct(wl_small, wl_total) if wl_small else None, 
             stage="📦 WH Loading", metric="Small (≤6)", date_range=date_range_tuple),
       _row("IC",              fmt(wl_ic),          wl_ic==0, TEAL_P[2],
           100-pct(wl_ic, wl_total) if wl_ic else None),
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
        _row("Old Doc Date",    fmt(sr_old),          sr_old==0, GREEN_P[2],
             100-pct(sr_old, sr_total) if sr_old else None, 
             stage="🏪 Shop Receiving", metric="Old Doc Date", date_range=date_range_tuple),
       _row("WH→Shop Mismatch",fmt(sr_mmatch),       sr_mmatch==0, GREEN_P[2],
           100-pct(sr_mmatch, sr_total) if sr_mmatch else None, 
           stage="🏪 Shop Receiving", metric="WH→Shop Mismatch", date_range=date_range_tuple),
        _row("Unique %",        f"{sr_upct:.1f}%",   sr_upct==100, GREEN_P[2],  sr_upct, 
            stage="🏪 Shop Receiving", metric="Unique %", date_range=date_range_tuple),
       _row("Blank",           fmt(sr_blank),        sr_blank==0, GREEN_P[2],
           100-pct(sr_blank, sr_total) if sr_blank else None),
       _row("Small (≤6)",      fmt(sr_small),        sr_small==0, GREEN_P[2],
           100-pct(sr_small, sr_total) if sr_small else None),
       _row("Duplicates",      fmt(sr_dup),          sr_dup==0, GREEN_P[2],
           100-pct(sr_dup, sr_total) if sr_dup else None),
       _row(f"Offloaded ({d_end.strftime('%d %b')})", fmt(sr_day_total), True, GREEN_P[2],
           pct(sr_day_total, wl_total) if wl_total else None),
       _row("Same Day", fmt(sr_day_same), True, GREEN_P[2],
           pct(sr_day_same, sr_day_total) if sr_day_total else None),
       _row("1-3 Days", fmt(sr_day_1_3), True, GREEN_P[2],
           pct(sr_day_1_3, sr_day_total) if sr_day_total else None),
       _row(">3 Days", fmt(sr_day_gt_3), True, GREEN_P[2],
           pct(sr_day_gt_3, sr_day_total) if sr_day_total else None),
    ])

    ss_rows = "".join([
        _row("Compliance %",    f"{ss_pct_v:.1f}%",  ss_pct_v==100, VIOLET_P[2],  ss_pct_v,
             note=f"{fmt(ss_yes)}/{fmt(ss_total)}", 
             stage="🛒 Shop Selling", metric="Compliance %", date_range=date_range_tuple),
        _row("Dup Serials",     fmt(ss_dup),          ss_dup==0, VIOLET_P[2],
             100-pct(ss_dup, ss_total) if ss_dup else None, 
             stage="🛒 Shop Selling", metric="Dup Serials", date_range=date_range_tuple),
        _row("No Serial",       fmt(ss_noser),        ss_noser==0, VIOLET_P[2],
             100-pct(ss_noser, ss_total) if ss_noser else None, 
             stage="🛒 Shop Selling", metric="No Serial", date_range=date_range_tuple),
        _row("Not in WH",       fmt(ss_notwh),        ss_notwh==0, VIOLET_P[2],
             100-pct(ss_notwh, ss_total) if ss_notwh else None, 
             stage="🛒 Shop Selling", metric="Not in WH", date_range=date_range_tuple),
        _row("IC",              fmt(ss_ic),           ss_ic==0, VIOLET_P[2],
             100-pct(ss_ic, ss_total) if ss_ic else None, 
             stage="🛒 Shop Selling", metric="Serial = IC", date_range=date_range_tuple),
        _row("Small (≤6)",      fmt(ss_small),        ss_small==0, VIOLET_P[2],
             100-pct(ss_small, ss_total) if ss_small else None, 
             stage="🛒 Shop Selling", metric="Small (≤6)", date_range=date_range_tuple),
    ])

    # ── Assemble infographic ───────────────────────────────────
    infographic = (
        '<div style="display:flex;align-items:flex-start;gap:0;width:100%;'
        'margin-top:10px;overflow-x:auto;">'
        + _stage_panel("🏭", "WH RECEIVING", "", wr_total, wr_ok,
                       BLUE_P[2], "#0c1a3a", "#080f22",
                       wr_rows,
                       "wr")
        + '<div style="width:54px;flex-shrink:0;"></div>'
        + _stage_panel("📦", "WH LOADING", "", wl_total, wl_ok,
                       TEAL_P[2], "#06222e", "#03141a",
                       wl_rows,
                       "wl")
        + _connector(TEAL_P[2], GREEN_P[2], "Offloaded", fmt(sr_total))
        + _stage_panel("🏪", "SHOP RECEIVING", "by WH doc / loaded date",
                       sr_total, sr_ok,
                       GREEN_P[2], "#042a1c", "#021610",
                       sr_rows,
                       "sr")
        + _connector(GREEN_P[2], VIOLET_P[2], "Sold", fmt(ss_total))
        + _stage_panel("🛒", "SHOP SELLING", "by bill date",
                       ss_total, ss_ok,
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
    
    # Remove the legacy drill display - no longer needed
    # ════════════════════════════════════════════════════════════
    
    # ══════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════
    # OFFLOADING SLAB TREND
    # ══════════════════════════════════════════════════════════
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    render_slab_trend(d_start, d_end)
    
    
    # ══════════════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════════════
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    st.markdown(f"""
<div style="margin-top:20px;padding:10px 20px;background:rgba(255,255,255,0.03);
            border:1px solid rgba(255,255,255,0.06);border-radius:10px;
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
    <span style="font-size:0.68rem;color:#475569;">
        🏭 WH Receiving &nbsp;↓&nbsp; 📦 WH Loading &nbsp;↓&nbsp; 🏪 Shop Receiving &nbsp;↓&nbsp; 🛒 Shop Selling
    </span>
    <span style="font-size:0.68rem;color:#334155;">
        DB: WH · Port 3307 · Refreshed: {datetime.now().strftime('%d %b %Y %H:%M')}
    </span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
