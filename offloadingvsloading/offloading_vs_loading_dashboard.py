import streamlit as st
import psycopg2
from datetime import datetime, timedelta, date
from psycopg2.extras import RealDictCursor, execute_values
import html
import plotly.graph_objects as go
import streamlit.components.v1 as components
import io
import base64
import os
import re
import warnings

import pandas as pd

try:
    import wms_sync as _wms_sync
    WMS_SYNC_AVAILABLE = True
except Exception:
    WMS_SYNC_AVAILABLE = False

# ── LVO upload column list (matches home_dashboard TABLE_CONFIGS) ──────────────
_LVO_COLUMNS = [
    "date", "shop_code", "vehicle_no", "item_code", "item_name",
    "qty_loaded", "value_loaded", "qty_offloaded", "value_offloaded",
    "diff_qty", "diff_val", "price", "diff",
]

try:
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    OPENPYXL_AVAILABLE = True
except Exception:
    Workbook = None
    PatternFill = None
    Font = None
    Alignment = None
    OPENPYXL_AVAILABLE = False

st.set_page_config(
    page_title="Loading vs Offloading Dashboard",
    page_icon="https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DB_CONFIG = {
    "host": "localhost",
    "port": 3307,
    "user": "postgres",
    "password": "hello",
    "dbname": "WH",
}

USERS_DB_NAME = "users"
CENTURY_DB_NAME = "century_penetration"
REQUIRED_TABLE_ACCESS = "offloading_vs_loading"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        #MainMenu, footer, header { visibility: hidden; }
        html, body, .stApp {
            background: #0a0f1e !important;
            color: #e2e8f0 !important;
            font-family: 'Inter', 'Segoe UI', sans-serif !important;
        }
        .block-container {
            padding-top: 0.7rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
        .dashboard-title {
            color: #f1f5f9;
            font-size: 1.3rem;
            font-weight: 800;
            line-height: 1.15;
        }
        .dashboard-sub {
            color: #94a3b8;
            font-size: 0.78rem;
            font-weight: 500;
        }
        .section-title {
            color: #dbeafe;
            font-size: 1rem;
            font-weight: 700;
            margin: 0.5rem 0 0.5rem;
        }
        div[data-testid="stTextInput"] label p,
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stPasswordInput"] label p,
        div[data-testid="stTextInput"] input,
        div[data-testid="stPasswordInput"] input {
            color: #f8fafc !important;
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stPasswordInput"] input {
            background: rgba(15, 23, 42, 0.8) !important;
            border: 1px solid rgba(59, 130, 246, 0.35) !important;
        }
        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            color: #ffffff !important;
            background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%) !important;
            border: 1px solid rgba(191, 219, 254, 0.65) !important;
            font-weight: 700 !important;
        }
        /* Reset Password popover toggle button — dark navy to match dashboard */
        button[data-testid="stPopoverButton"] {
            color: #ffffff !important;
            background: linear-gradient(135deg, #0c1a3a 0%, #112244 100%) !important;
            border: 1px solid rgba(59, 130, 246, 0.40) !important;
            font-weight: 600 !important;
        }
        button[data-testid="stPopoverButton"]:hover {
            background: linear-gradient(135deg, #1a2f5a 0%, #1e3872 100%) !important;
            border-color: rgba(96, 165, 250, 0.60) !important;
        }

        /* ── Popover dropdown panel — dark dashboard theme ── */
        div[data-testid="stPopover"] > div,
        div[data-testid="stPopoverContent"],
        [data-baseweb="popover"] [role="dialog"],
        [data-baseweb="popover"] > div {
            background: linear-gradient(160deg, #0c1a3a 0%, #080f22 100%) !important;
            border: 1px solid rgba(59, 130, 246, 0.30) !important;
            border-radius: 10px !important;
        }
        /* All text inside the popover panel */
        div[data-testid="stPopoverContent"] *,
        [data-baseweb="popover"] [role="dialog"] * {
            color: #f8fafc !important;
        }
        /* Input fields inside popover */
        div[data-testid="stPopoverContent"] input,
        [data-baseweb="popover"] [role="dialog"] input {
            background: rgba(15, 23, 42, 0.85) !important;
            border: 1px solid rgba(59, 130, 246, 0.35) !important;
            color: #f8fafc !important;
        }
        /* Labels inside popover */
        div[data-testid="stPopoverContent"] label p,
        [data-baseweb="popover"] [role="dialog"] label p {
            color: #cbd5e1 !important;
        }
        /* Submit button inside popover */
        div[data-testid="stPopoverContent"] button,
        [data-baseweb="popover"] [role="dialog"] button {
            color: #ffffff !important;
            background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%) !important;
            border: 1px solid rgba(191, 219, 254, 0.55) !important;
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_db_connection(dbname: str | None = None):
    cfg = dict(DB_CONFIG)
    if dbname:
        cfg["dbname"] = dbname
    return psycopg2.connect(**cfg)


@st.cache_data(ttl=300, show_spinner=False)
def _wms_check_file_status(target_date: date) -> str:
    """
    Returns:
      'ok'            — file exists on network share
      'missing'       — share accessible but file not there
      'network_error' — share not reachable
    Cached for 5 minutes so it doesn't hammer the network share on every rerun.
    """
    if not WMS_SYNC_AVAILABLE:
        return "ok"
    try:
        found = _wms_sync.find_wms_file(target_date)
        return "ok" if found else "missing"
    except OSError:
        return "network_error"


def init_auth_state() -> None:
    if "ovl_authenticated" not in st.session_state:
        st.session_state["ovl_authenticated"] = False
    if "ovl_user" not in st.session_state:
        st.session_state["ovl_user"] = None
    if "ovl_is_master_user" not in st.session_state:
        st.session_state["ovl_is_master_user"] = False
    if "ovl_allowed_shops" not in st.session_state:
        st.session_state["ovl_allowed_shops"] = []


@st.cache_data(ttl=300, show_spinner=False)
def authenticate_user(login_name: str, password: str) -> dict | None:
    query = """
        SELECT employee_id, full_name, table_access, is_active
        FROM users
        WHERE (
                UPPER(TRIM(COALESCE(full_name, ''))) = UPPER(TRIM(%s))
             OR UPPER(TRIM(COALESCE(employee_id, ''))) = UPPER(TRIM(%s))
        )
          AND password = %s
          AND LOWER(COALESCE(is_active, 'false')) = 'true'
        LIMIT 1
    """
    with get_db_connection(USERS_DB_NAME) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (login_name.strip(), login_name.strip(), password))
            row = cur.fetchone()
    return dict(row) if row else None


def check_dashboard_access(user: dict) -> bool:
    table_access = str((user or {}).get("table_access", "") or "")
    allowed = {x.strip().lower() for x in table_access.split(",") if x.strip()}
    return bool(
        "all" in allowed
        or REQUIRED_TABLE_ACCESS in allowed
        or "offloading_loading" in allowed
        or "offloading_vs_loading_dashboard" in allowed
    )


def get_user_shop_scope(employee_id: str, full_name: str) -> tuple[bool, list[str]]:
    emp = str(employee_id or "").strip().upper()
    name = str(full_name or "").strip().upper()
    query = """
        WITH latest_upload AS (
            SELECT MAX(upload_date) AS max_upload_date
            FROM opsmgr
        )
        SELECT UPPER(TRIM(COALESCE(shopcode, ''))) AS shop_code
        FROM opsmgr
        WHERE upload_date = (SELECT max_upload_date FROM latest_upload)
          AND (SELECT max_upload_date FROM latest_upload) IS NOT NULL
          AND UPPER(TRIM(COALESCE(opsmanagername, ''))) IN (%s, %s)
          AND TRIM(COALESCE(shopcode, '')) <> ''
    """

    with get_db_connection(CENTURY_DB_NAME) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (emp, name))
            rows = cur.fetchall()

    shops = sorted({str(r[0] or "").strip().upper() for r in rows if r and str(r[0] or "").strip()})
    is_master = any(s == "ALL" for s in shops)
    scoped = [s for s in shops if s != "ALL"]
    return is_master, scoped


def get_latest_opsmgr_upload_date() -> date | None:
    query = """
        SELECT MAX(upload_date)
        FROM opsmgr
    """
    with get_db_connection(CENTURY_DB_NAME) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()
    if not row:
        return None
    return row[0]


def change_password(login_name: str, new_password: str) -> tuple[bool, str]:
    if not login_name.strip() or not new_password:
        return False, "Please fill all password fields."
    if len(new_password) < 4:
        return False, "New password must be at least 4 characters."
    resolve_q = """
                SELECT employee_id
        FROM users
                WHERE (
                                UPPER(TRIM(COALESCE(full_name, ''))) = UPPER(TRIM(%s))
                         OR UPPER(TRIM(COALESCE(employee_id, ''))) = UPPER(TRIM(%s))
                )
          AND LOWER(COALESCE(is_active, 'false')) = 'true'
        LIMIT 1
    """
    update_q = """
        UPDATE users
        SET password = %s
        WHERE employee_id = %s
    """
    with get_db_connection(USERS_DB_NAME) as conn:
        with conn.cursor() as cur:
            cur.execute(resolve_q, (login_name.strip(), login_name.strip()))
            ok_row = cur.fetchone()
            if not ok_row:
                return False, "Invalid User Name / User ID or inactive account."
            target_employee_id = str(ok_row[0] or "").strip()
            if not target_employee_id:
                return False, "Unable to resolve user account for password reset."
            cur.execute(update_q, (new_password, target_employee_id))
        conn.commit()

    authenticate_user.clear()
    return True, "Password updated successfully."


def render_login_gate() -> bool:
    st.markdown(
        """
        <div style='max-width:520px;margin:28px auto 0 auto;padding:16px;border:1px solid rgba(59,130,246,0.25);border-radius:12px;background:linear-gradient(160deg,#0c1a3a 0%,#080f22 100%);'>
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>
                <img src='https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg' style='height:38px;width:38px;border-radius:50%;border:1px solid rgba(148,163,184,0.45);' />
                <div style='font-size:1.1rem;font-weight:800;color:#f8fafc;'>Offloading vs Loading - Login</div>
            </div>
            <div style='font-size:0.82rem;color:#cbd5e1;margin-bottom:2px;'>Sign in with User Name and password.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, center_col, right_col = st.columns([1.3, 1.5, 1.3], gap="small")
    with center_col:
        with st.form("ovl_login_form", clear_on_submit=False):
            ops_manager_name = st.text_input("User Name")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

    if submitted:
        user = authenticate_user(ops_manager_name, password)
        if not user:
            st.error("Invalid login credentials or inactive account.")
            return False
        if not check_dashboard_access(user):
            st.error("You do not have access to this dashboard. Ask admin to update users.table_access.")
            return False

        # Super user: table_access='all' → auto master, no opsmgr entry required
        _ta = str((user or {}).get("table_access", "") or "")
        _is_super = "all" in {x.strip().lower() for x in _ta.split(",") if x.strip()}
        if _is_super:
            is_master, allowed_shops = True, []
        else:
            is_master, allowed_shops = get_user_shop_scope(
                str(user.get("employee_id", "") or ""),
                str(user.get("full_name", "") or ""),
            )
        if not _is_super and not is_master and not allowed_shops:
            st.error("No shop mapping found in century_penetration.opsmgr for this user.")
            return False

        st.session_state["ovl_authenticated"] = True
        st.session_state["ovl_user"] = user
        st.session_state["ovl_is_master_user"] = is_master
        st.session_state["ovl_allowed_shops"] = allowed_shops
        st.rerun()

    with center_col:
        with st.popover("Reset Password"):
            with st.form("ovl_reset_pwd_login_form", clear_on_submit=True):
                emp = st.text_input("User Name", key="ovl_reset_emp")
                new_pwd = st.text_input("New Password", type="password", key="ovl_reset_new")
                cnf_pwd = st.text_input("Confirm New Password", type="password", key="ovl_reset_confirm")
                do_reset = st.form_submit_button("Update Password", use_container_width=True)

            if do_reset:
                if new_pwd != cnf_pwd:
                    st.error("New password and confirm password do not match.")
                else:
                    ok, msg = change_password(emp, new_pwd)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    return bool(st.session_state.get("ovl_authenticated", False))


def render_logged_in_sidebar() -> bool:
    user = st.session_state.get("ovl_user") or {}
    full_name = str(user.get("full_name") or "").strip()
    employee_id = str(user.get("employee_id") or "").strip()
    is_master = bool(st.session_state.get("ovl_is_master_user", False))
    allowed_shops = st.session_state.get("ovl_allowed_shops", []) or []

    with st.sidebar:
        st.markdown("### User Access")
        st.caption(f"Name: {full_name or 'Unknown'}")
        st.caption(f"User ID: {employee_id or 'Unknown'}")
        if is_master:
            st.success("Role: Master User (ALL shops)")
        else:
            st.info(f"Role: Ops Manager ({len(allowed_shops)} mapped shop(s))")
            if allowed_shops:
                st.caption("Shops: " + ", ".join(allowed_shops))

        with st.expander("Reset Password", expanded=False):
            with st.form("ovl_reset_pwd_loggedin_form", clear_on_submit=True):
                login_name = st.text_input("User Name", key="ovl_lg_reset_name")
                new_pwd = st.text_input("New Password", type="password", key="ovl_lg_reset_new")
                cnf_pwd = st.text_input("Confirm New Password", type="password", key="ovl_lg_reset_confirm")
                do_reset = st.form_submit_button("Update Password", use_container_width=True)

            if do_reset:
                if new_pwd != cnf_pwd:
                    st.error("New password and confirm password do not match.")
                else:
                    reset_name = str(login_name or "").strip() or full_name or employee_id
                    ok, msg = change_password(reset_name, new_pwd)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

        if st.button("Logout", use_container_width=True):
            st.session_state["ovl_authenticated"] = False
            st.session_state["ovl_user"] = None
            st.session_state["ovl_is_master_user"] = False
            st.session_state["ovl_allowed_shops"] = []
            st.rerun()
            return True

    return False


def apply_shop_scope(rows: list[dict], allowed_shops: list[str], is_master: bool) -> list[dict]:
    if is_master:
        return rows
    allowed = {str(s or "").strip().upper() for s in (allowed_shops or []) if str(s or "").strip()}
    if not allowed:
        return []
    return [
        r for r in rows
        if str(r.get("shop_code", "") or "").strip().upper() in allowed
    ]


def corrected_diff_val_sql() -> str:
    """Return SQL expression that normalizes legacy diff_val sign issues."""
    return """
        CASE
            WHEN COALESCE(diff_val, 0) = 0
                 AND COALESCE(diff_qty, 0) <> 0
                 AND COALESCE(price, 0) <> 0
                THEN COALESCE(diff_qty, 0) * COALESCE(price, 0)
            WHEN COALESCE(diff_qty, 0) < 0
                 AND COALESCE(diff_val, 0) > 0
                 AND COALESCE(price, 0) > 0
                THEN COALESCE(diff_qty, 0) * COALESCE(price, 0)
            WHEN COALESCE(diff_qty, 0) > 0
                 AND COALESCE(diff_val, 0) < 0
                 AND COALESCE(price, 0) > 0
                THEN COALESCE(diff_qty, 0) * COALESCE(price, 0)
            ELSE COALESCE(diff_val, 0)
        END
    """


def get_live_date_bounds_uncached() -> tuple:
    """Read bounds directly from DB to recover from stale cache states."""
    query = """
        SELECT COUNT(*) AS row_count, MIN(date) AS min_date, MAX(date) AS max_date
        FROM offloading_vs_loading
        WHERE date IS NOT NULL
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()
    if not row:
        return 0, None, None
    return int(row[0] or 0), row[1], row[2]


@st.cache_data(ttl=300)
def load_date_bounds() -> tuple:
    query = """
        SELECT MIN(date) AS min_date, MAX(date) AS max_date
        FROM offloading_vs_loading
        WHERE date IS NOT NULL
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()
    if not row or row[0] is None or row[1] is None:
        return None, None
    return row[0], row[1]


@st.cache_data(ttl=300)
def load_offloading_vs_loading(start_date, end_date) -> list[dict]:
    diff_val_expr = "((COALESCE(qty_offloaded, 0) - COALESCE(qty_loaded, 0)) * COALESCE(price, 0))"
    query = f"""
        SELECT
            date,
            shop_code,
            vehicle_no,
            item_code,
            item_name,
            COALESCE(qty_loaded, 0)::numeric AS qty_loaded,
            (COALESCE(qty_loaded, 0) * COALESCE(price, 0))::numeric AS value_loaded,
            COALESCE(qty_offloaded, 0)::numeric AS qty_offloaded,
            (COALESCE(qty_offloaded, 0) * COALESCE(price, 0))::numeric AS value_offloaded,
            (COALESCE(qty_offloaded, 0) - COALESCE(qty_loaded, 0))::numeric AS diff_qty,
            ({diff_val_expr})::numeric AS diff_val
        FROM offloading_vs_loading
        WHERE date BETWEEN %(s)s AND %(e)s
        ORDER BY date DESC, shop_code, vehicle_no, item_code
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, {"s": start_date, "e": end_date})
            return [dict(r) for r in cur.fetchall()]


@st.cache_data(ttl=300)
def load_monthly_summary(start_date, end_date) -> list[dict]:
    diff_val_expr = "((COALESCE(qty_offloaded, 0) - COALESCE(qty_loaded, 0)) * COALESCE(price, 0))"
    query = f"""
        SELECT
            to_char(date_trunc('month', date), 'YYYY-MM') AS month_key,
            to_char(date_trunc('month', date), 'Mon YYYY') AS month_label,
            SUM(COALESCE(qty_offloaded, 0) - COALESCE(qty_loaded, 0))::numeric AS diff_qty,
            SUM({diff_val_expr})::numeric AS diff_val
        FROM offloading_vs_loading
        WHERE date BETWEEN %(s)s AND %(e)s
        GROUP BY 1, 2
        ORDER BY month_key
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, {"s": start_date, "e": end_date})
            return [dict(r) for r in cur.fetchall()]


@st.cache_data(ttl=300)
def load_monthly_summary_all() -> list[dict]:
    diff_val_expr = "((COALESCE(qty_offloaded, 0) - COALESCE(qty_loaded, 0)) * COALESCE(price, 0))"
    query = f"""
        SELECT
            to_char(date_trunc('month', date), 'YYYY-MM') AS month_key,
            to_char(date_trunc('month', date), 'Mon YYYY') AS month_label,
            SUM(COALESCE(qty_offloaded, 0) - COALESCE(qty_loaded, 0))::numeric AS diff_qty,
            SUM({diff_val_expr})::numeric AS diff_val
        FROM offloading_vs_loading
        WHERE date IS NOT NULL
        GROUP BY 1, 2
        ORDER BY month_key
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            return [dict(r) for r in cur.fetchall()]


# ─────────────────────────────────────────────────────────────
# EDA DATA LOADERS
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_discrepancy_summary(start_date, end_date) -> dict:
    """Breakdown: excess / short / balanced counts and values."""
    diff_val_expr = "((COALESCE(qty_offloaded,0)-COALESCE(qty_loaded,0))*COALESCE(price,0))"
    q = f"""
        SELECT
            COUNT(*) FILTER (WHERE ({diff_val_expr}) > 0)  AS excess_count,
            COUNT(*) FILTER (WHERE ({diff_val_expr}) < 0)  AS short_count,
            COUNT(*) FILTER (WHERE ({diff_val_expr}) = 0)  AS balanced_count,
            SUM(CASE WHEN ({diff_val_expr})>0 THEN ({diff_val_expr}) ELSE 0 END) AS excess_val,
            SUM(CASE WHEN ({diff_val_expr})<0 THEN ABS({diff_val_expr}) ELSE 0 END) AS short_val,
            SUM(ABS({diff_val_expr})) AS total_abs_val,
            COUNT(*) AS total_lines,
            COUNT(DISTINCT shop_code) AS shop_count,
            COUNT(DISTINCT vehicle_no) AS vehicle_count,
            COUNT(DISTINCT item_code) AS item_count
        FROM offloading_vs_loading
        WHERE date BETWEEN %(s)s AND %(e)s
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, {"s": start_date, "e": end_date})
            row = cur.fetchone()
    return dict(row) if row else {}


@st.cache_data(ttl=300)
def load_daily_trend(start_date, end_date) -> list[dict]:
    """Daily loaded, offloaded and diff totals."""
    q = """
        SELECT
            date,
            SUM(COALESCE(qty_loaded, 0))::numeric    AS qty_loaded,
            SUM(COALESCE(qty_offloaded, 0))::numeric AS qty_offloaded,
            SUM((COALESCE(qty_offloaded,0)-COALESCE(qty_loaded,0))*COALESCE(price,0))::numeric AS diff_val,
            COUNT(*) AS lines
        FROM offloading_vs_loading
        WHERE date BETWEEN %(s)s AND %(e)s
        GROUP BY date
        ORDER BY date
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, {"s": start_date, "e": end_date})
            return [dict(r) for r in cur.fetchall()]


@st.cache_data(ttl=300)
def load_top_discrepancy_items(start_date, end_date, limit: int = 20) -> list[dict]:
    """Top items by absolute difference value."""
    diff_val_expr = "((COALESCE(qty_offloaded,0)-COALESCE(qty_loaded,0))*COALESCE(price,0))"
    q = f"""
        SELECT
            item_code,
            item_name,
            SUM(COALESCE(qty_loaded,0))::numeric    AS qty_loaded,
            SUM(COALESCE(qty_offloaded,0))::numeric AS qty_offloaded,
            SUM(COALESCE(qty_offloaded,0)-COALESCE(qty_loaded,0))::numeric AS diff_qty,
            SUM({diff_val_expr})::numeric            AS diff_val,
            SUM(ABS({diff_val_expr}))::numeric       AS abs_diff_val,
            COUNT(DISTINCT shop_code)                AS shop_count,
            COUNT(*)                                 AS line_count
        FROM offloading_vs_loading
        WHERE date BETWEEN %(s)s AND %(e)s
          AND item_code IS NOT NULL AND TRIM(item_code) <> ''
        GROUP BY item_code, item_name
        ORDER BY abs_diff_val DESC
        LIMIT %(lim)s
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, {"s": start_date, "e": end_date, "lim": limit})
            return [dict(r) for r in cur.fetchall()]


@st.cache_data(ttl=300)
def load_vehicle_summary(start_date, end_date) -> list[dict]:
    """Vehicle-level discrepancy summary."""
    diff_val_expr = "((COALESCE(qty_offloaded,0)-COALESCE(qty_loaded,0))*COALESCE(price,0))"
    q = f"""
        SELECT
            COALESCE(NULLIF(TRIM(vehicle_no),''),'Unknown') AS vehicle_no,
            COUNT(DISTINCT shop_code)                        AS shops_served,
            COUNT(*)                                         AS lines,
            SUM(COALESCE(qty_loaded,0))::numeric             AS qty_loaded,
            SUM(COALESCE(qty_offloaded,0))::numeric          AS qty_offloaded,
            SUM({diff_val_expr})::numeric                    AS diff_val,
            SUM(ABS({diff_val_expr}))::numeric               AS abs_diff_val,
            SUM(CASE WHEN ({diff_val_expr})<0
                THEN ABS({diff_val_expr}) ELSE 0 END)::numeric AS short_val,
            SUM(CASE WHEN ({diff_val_expr})>0
                THEN ({diff_val_expr}) ELSE 0 END)::numeric   AS excess_val
        FROM offloading_vs_loading
        WHERE date BETWEEN %(s)s AND %(e)s
          AND vehicle_no IS NOT NULL AND TRIM(vehicle_no) <> ''
        GROUP BY 1
        ORDER BY abs_diff_val DESC
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, {"s": start_date, "e": end_date})
            return [dict(r) for r in cur.fetchall()]


@st.cache_data(ttl=300)
def load_shop_date_grid(start_date, end_date) -> list[dict]:
    """Shop × date diff_val for heatmap (last 14 days max)."""
    diff_val_expr = "((COALESCE(qty_offloaded,0)-COALESCE(qty_loaded,0))*COALESCE(price,0))"
    q = f"""
        SELECT
            COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
            date,
            SUM({diff_val_expr})::numeric AS diff_val
        FROM offloading_vs_loading
        WHERE date BETWEEN %(s)s AND %(e)s
        GROUP BY shop_code, date
        ORDER BY shop_code, date
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, {"s": start_date, "e": end_date})
            return [dict(r) for r in cur.fetchall()]


# ─────────────────────────────────────────────────────────────
# NEWSLETTER — HTML BUILDER + EMAIL SENDER
# ─────────────────────────────────────────────────────────────

_SMTP_HOST            = "mail.melcomgroup.com"
_SMTP_PORT            = 25
_NEWSLETTER_FROM      = "mis.manager@melcomgroup.com"
_NEWSLETTER_TO        = "mis.manager@melcomgroup.com"
_MELCOM_LOGO          = "https://melcom.com/skin/frontend/melcom/default/images/logo.png"
_DASHBOARD_URL        = os.getenv("DASHBOARD_URL", "http://10.10.0.30:8502")


def _bar_html(value: float, max_val: float, color: str, label: str, fmt_val: str) -> str:
    """CSS progress-bar row for email canvas infographic."""
    pct = min(100, round(abs(value) / max(max_val, 1) * 100))
    return (
        f'<tr>'
        f'<td style="padding:4px 8px 4px 0;font-size:12px;color:#cbd5e1;'
        f'white-space:nowrap;width:35%;">{label}</td>'
        f'<td style="padding:4px 0;">'
        f'<table width="100%" cellpadding="0" cellspacing="0">'
        f'<tr><td style="background:#1e293b;border-radius:4px;overflow:hidden;height:14px;">'
        f'<div style="width:{pct}%;height:14px;background:{color};border-radius:4px;"></div>'
        f'</td></tr></table></td>'
        f'<td style="padding:4px 0 4px 8px;font-size:12px;font-weight:700;color:{color};'
        f'white-space:nowrap;width:22%;text-align:right;">{fmt_val}</td>'
        f'</tr>'
    )


def _build_lvo_newsletter_html(
    summary: dict,
    shop_rows: list[dict],
    shop_meta_map: dict,
    items_rows: list[dict],
    veh_rows: list[dict],
    daily_rows: list[dict],
    start_date,
    end_date,
) -> str:
    """Build rich HTML newsletter — email-safe table-based infographics."""
    import html as _html

    def _f(v, dec=0):
        try: return f"{float(v):,.{dec}f}"
        except: return "0"

    def _esc(s): return _html.escape(str(s or ""))

    excess_val  = float(summary.get("excess_val")  or 0)
    short_val   = float(summary.get("short_val")   or 0)
    bal_cnt     = int(summary.get("balanced_count") or 0)
    short_cnt   = int(summary.get("short_count")   or 0)
    excess_cnt  = int(summary.get("excess_count")  or 0)
    total_lines = int(summary.get("total_lines")   or 0)
    match_rate  = round(bal_cnt / total_lines * 100, 1) if total_lines else 0
    net_diff    = excess_val - short_val
    net_clr     = "#22c55e" if net_diff >= 0 else "#ef4444"
    net_lbl     = "Net Excess" if net_diff >= 0 else "Net Shortage"

    date_str = f"{start_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')}"
    sent_ts  = datetime.now().strftime("%d %b %Y, %H:%M")

    # ─── Fetch yesterday's compliance + bottom 5 cashiers from serialno_check_yes_no ──
    _yest_compliance = None
    _yest_total      = 0
    _yest_yes        = 0
    _bottom5_yest    = []
    _bottom5_mtd     = []
    try:
        _yest = (datetime.today() - timedelta(days=1)).date()
        _mtd  = _yest.replace(day=1)
        _conn_sn = get_db_connection()
        with _conn_sn.cursor() as _cur:
            _cur.execute("""
                SELECT COUNT(*), SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END)
                FROM serialno_check_yes_no WHERE DATE(bill_date) = %s
            """, (_yest,))
            _r = _cur.fetchone()
            _yest_total = int(_r[0] or 0)
            _yest_yes   = int(_r[1] or 0)
            _yest_compliance = round(_yest_yes / _yest_total * 100, 1) if _yest_total else 0

            _cur.execute("""
                SELECT COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown'),
                       COALESCE(NULLIF(TRIM(shop_code),''),'?'),
                       COUNT(*),
                       ROUND(SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END)
                             ::NUMERIC/NULLIF(COUNT(*),0)*100,1)
                FROM serialno_check_yes_no
                WHERE DATE(bill_date) = %s
                  AND cashier_name IS NOT NULL AND TRIM(cashier_name) <> ''
                GROUP BY 1, 2 HAVING COUNT(*) >= 3
                ORDER BY 4 ASC NULLS FIRST LIMIT 5
            """, (_yest,))
            _bottom5_yest = [{"cashier": r[0], "shop": r[1],
                               "total": int(r[2] or 0), "pct": float(r[3] or 0)}
                             for r in _cur.fetchall()]

            _cur.execute("""
                SELECT COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown'),
                       COALESCE(NULLIF(TRIM(shop_code),''),'?'),
                       COUNT(*),
                       ROUND(SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END)
                             ::NUMERIC/NULLIF(COUNT(*),0)*100,1)
                FROM serialno_check_yes_no
                WHERE DATE(bill_date) BETWEEN %s AND %s
                  AND cashier_name IS NOT NULL AND TRIM(cashier_name) <> ''
                GROUP BY 1, 2 HAVING COUNT(*) >= 10
                ORDER BY 4 ASC NULLS FIRST LIMIT 5
            """, (_mtd, _yest))
            _bottom5_mtd = [{"cashier": r[0], "shop": r[1],
                              "total": int(r[2] or 0), "pct": float(r[3] or 0)}
                            for r in _cur.fetchall()]
        _conn_sn.close()
    except Exception:
        pass

    def _comp_clr(p): return "#22c55e" if p >= 90 else "#f59e0b" if p >= 75 else "#ef4444"

    def _cashier_rows(rows):
        if not rows:
            return ('<tr bgcolor="#0f1829"><td colspan="5" style="padding:10px 16px;'
                    'font-size:11px;color:#475569;font-style:italic;">No data</td></tr>')
        html = ""
        for i, r in enumerate(rows):
            pct = float(r.get("pct", 0))
            clr = _comp_clr(pct)
            bg  = "#0f1829" if i % 2 == 0 else "#111d35"
            gap = pct - 90
            gap_txt = f"{'▲' if gap >= 0 else '▼'}{abs(gap):.1f}%"
            gap_clr = "#22c55e" if gap >= 0 else "#ef4444"
            html += (
                f'<tr bgcolor="{bg}" style="background-color:{bg};">'
                f'<td style="padding:8px 10px 8px 14px;font-size:11px;color:#e2e8f0;">'
                f'{_esc(r.get("cashier",""))}</td>'
                f'<td style="padding:8px 10px;font-size:11px;color:#60a5fa;text-align:center;">'
                f'{_esc(r.get("shop",""))}</td>'
                f'<td style="padding:8px 10px;font-size:11px;color:#94a3b8;text-align:center;">'
                f'{int(r.get("total",0)):,}</td>'
                f'<td style="padding:8px 10px;font-size:13px;font-weight:900;'
                f'color:{clr};text-align:center;">{pct:.1f}%</td>'
                f'<td style="padding:8px 14px;font-size:11px;font-weight:700;'
                f'color:{gap_clr};text-align:right;">{gap_txt}</td>'
                f'</tr>'
            )
        return html

    _yest_str = (datetime.today() - timedelta(days=1)).strftime("%d %b %Y")
    _mtd_str  = (datetime.today() - timedelta(days=1)).replace(day=1).strftime("%d %b")

    # Pre-compute compliance block — avoids nested f-strings in the HTML template
    if _yest_compliance is not None:
        _ycomp_clr = _comp_clr(_yest_compliance)
        _gauge_w   = max(1, min(99, round(_yest_compliance)))
        _gauge_rem = 100 - _gauge_w
        _tgt_txt   = "&#9989; On Target" if _yest_compliance >= 90 else f"&#9888; {abs(_yest_compliance - 90):.1f}% below"
        _tgt_clr   = "#22c55e" if _yest_compliance >= 90 else "#ef4444"
        _yest_compliance_block = (
            f'<tr><td bgcolor="#0a1628" style="background-color:#0a1628;padding:14px 24px;'
            f'border-top:3px solid {_ycomp_clr};">'
            f'<table width="100%" cellpadding="0" cellspacing="0" border="0"><tr>'
            f'<td width="55%" style="vertical-align:middle;">'
            f'<div style="font-size:10px;color:#7c3aed;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:5px;">Yesterday Serial Compliance</div>'
            f'<div style="font-size:38px;font-weight:900;color:{_ycomp_clr};'
            f'font-family:Arial,sans-serif;line-height:1;">{_yest_compliance:.1f}%</div>'
            f'<div style="font-size:11px;color:#64748b;margin-top:3px;">'
            f'{_yest_yes:,} compliant / {_yest_total:,} scans &bull; {_yest_str}</div></td>'
            f'<td width="45%" style="vertical-align:middle;padding-left:14px;'
            f'border-left:1px solid #1e3a5f;text-align:center;">'
            f'<div style="font-size:10px;color:#64748b;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:5px;">vs 90% Target</div>'
            f'<div style="font-size:22px;font-weight:900;font-family:Arial,sans-serif;color:{_tgt_clr};">'
            f'{_tgt_txt}</div></td></tr></table>'
            f'<table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-top:10px;">'
            f'<tr><td width="{_gauge_w}%" bgcolor="{_ycomp_clr}" style="background-color:{_ycomp_clr};'
            f'height:8px;border-radius:4px 0 0 4px;font-size:1px;line-height:1px;">&nbsp;</td>'
            f'<td width="{_gauge_rem}%" bgcolor="#1e293b" style="background-color:#1e293b;'
            f'height:8px;border-radius:0 4px 4px 0;font-size:1px;line-height:1px;">&nbsp;</td>'
            f'</tr></table></td></tr>'
        )
    else:
        _yest_compliance_block = ""

    # ─── email-safe horizontal bar using <td width> ───────────────────────────
    def _bar(value: float, max_val: float, fill_clr: str, bg_clr: str = "#1e293b") -> str:
        pct = max(1, min(99, round(abs(value) / max(abs(max_val), 1) * 100)))
        rem = 100 - pct
        return (
            f'<table width="100%" cellpadding="0" cellspacing="0" border="0" role="presentation">'
            f'<tr>'
            f'<td width="{pct}%" bgcolor="{fill_clr}" style="background-color:{fill_clr};'
            f'height:12px;border-radius:3px 0 0 3px;font-size:1px;line-height:1px;">&nbsp;</td>'
            f'<td width="{rem}%" bgcolor="{bg_clr}" style="background-color:{bg_clr};'
            f'height:12px;border-radius:0 3px 3px 0;font-size:1px;line-height:1px;">&nbsp;</td>'
            f'</tr></table>'
        )

    # ─── KPI card cell ────────────────────────────────────────────────────────
    def _kpi_td(icon, label, value, sub, clr, border_clr):
        return (
            f'<td width="25%" style="padding:6px;">'
            f'<table width="100%" cellpadding="0" cellspacing="0" border="0">'
            f'<tr><td bgcolor="#1a2744" style="background-color:#1a2744;border:1px solid {border_clr};'
            f'border-left:4px solid {clr};border-radius:8px;padding:14px 10px;text-align:center;">'
            f'<div style="font-size:22px;margin-bottom:4px;">{icon}</div>'
            f'<div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:6px;">{label}</div>'
            f'<div style="font-size:20px;font-weight:900;color:{clr};font-family:Arial,sans-serif;'
            f'font-variant-numeric:tabular-nums;">{value}</div>'
            f'<div style="font-size:11px;color:#64748b;margin-top:4px;">{sub}</div>'
            f'</td></tr></table></td>'
        )

    # ─── Section header ───────────────────────────────────────────────────────
    def _section_hdr(icon, title, clr):
        return (
            f'<tr><td bgcolor="#111827" style="background-color:#111827;'
            f'border-top:3px solid {clr};padding:12px 24px;">'
            f'<span style="font-size:16px;">{icon}</span>'
            f' <span style="font-size:13px;font-weight:700;color:#f1f5f9;'
            f'letter-spacing:0.5px;font-family:Arial,sans-serif;">{title}</span>'
            f'</td></tr>'
        )

    # ─── Shop chart rows ──────────────────────────────────────────────────────
    worst_shops = sorted(
        shop_rows, key=lambda x: float(x.get("diff_val") or 0)
    )[:10]
    max_shop_abs = max((abs(float(r.get("diff_val") or 0)) for r in worst_shops), default=1)
    shop_rows_html = ""
    for i, r in enumerate(worst_shops):
        dv   = float(r.get("diff_val") or 0)
        clr  = "#ef4444" if dv < 0 else "#22c55e" if dv > 0 else "#94a3b8"
        code = str(r.get("shop_code", "") or "")
        name = shop_meta_map.get(code.upper(), {}).get("shop_name", "")[:20]
        lbl  = f"{code} {name}".strip()
        row_bg = "#0f1829" if i % 2 == 0 else "#111d35"
        shop_rows_html += (
            f'<tr bgcolor="{row_bg}" style="background-color:{row_bg};">'
            f'<td width="28%" style="padding:9px 10px 9px 16px;font-size:12px;'
            f'color:#e2e8f0;font-family:Arial,sans-serif;font-weight:600;'
            f'white-space:nowrap;">{_esc(lbl)}</td>'
            f'<td width="48%" style="padding:9px 10px;">{_bar(dv, max_shop_abs, clr)}</td>'
            f'<td width="24%" style="padding:9px 16px 9px 6px;font-size:13px;'
            f'font-weight:800;color:{clr};font-family:Arial,sans-serif;'
            f'text-align:right;white-space:nowrap;">GH₵ {dv:,.0f}</td>'
            f'</tr>'
        )

    # ─── Item chart rows ──────────────────────────────────────────────────────
    top_items = items_rows[:10]
    max_item_abs = max((abs(float(r.get("diff_val") or 0)) for r in top_items), default=1)
    item_rows_html = ""
    for i, r in enumerate(top_items):
        dv   = float(r.get("diff_val") or 0)
        clr  = "#ef4444" if dv < 0 else "#22c55e"
        name = str(r.get("item_name") or r.get("item_code") or "")[:32]
        row_bg = "#0f1829" if i % 2 == 0 else "#111d35"
        item_rows_html += (
            f'<tr bgcolor="{row_bg}" style="background-color:{row_bg};">'
            f'<td width="34%" style="padding:8px 10px 8px 16px;font-size:11px;'
            f'color:#cbd5e1;font-family:Arial,sans-serif;white-space:nowrap;'
            f'overflow:hidden;">{_esc(name)}</td>'
            f'<td width="42%" style="padding:8px 10px;">{_bar(dv, max_item_abs, clr)}</td>'
            f'<td width="24%" style="padding:8px 16px 8px 6px;font-size:12px;'
            f'font-weight:800;color:{clr};font-family:Arial,sans-serif;'
            f'text-align:right;white-space:nowrap;">GH₵ {dv:,.0f}</td>'
            f'</tr>'
        )

    # ─── Daily spark using table cells (email-safe) ───────────────────────────
    spark_cells = ""
    recent = daily_rows[-14:] if len(daily_rows) > 14 else daily_rows
    max_spark = max((abs(float(r.get("diff_val") or 0)) for r in recent), default=1)
    bar_h_max = 50
    for r in recent:
        dv    = float(r.get("diff_val") or 0)
        h     = max(3, round(abs(dv) / max_spark * bar_h_max))
        pad   = bar_h_max - h
        clr   = "#ef4444" if dv < 0 else "#22c55e"
        d_lbl = pd.Timestamp(r["date"]).strftime("%d") if r.get("date") else ""
        spark_cells += (
            f'<td width="7%" style="text-align:center;vertical-align:bottom;padding:0 2px;">'
            f'<table cellpadding="0" cellspacing="0" border="0" width="100%">'
            f'<tr><td height="{pad}" style="font-size:1px;line-height:1px;">&nbsp;</td></tr>'
            f'<tr><td bgcolor="{clr}" height="{h}" style="background-color:{clr};'
            f'border-radius:2px 2px 0 0;font-size:1px;line-height:1px;">&nbsp;</td></tr>'
            f'<tr><td style="font-size:9px;color:#64748b;padding-top:3px;'
            f'text-align:center;">{d_lbl}</td></tr>'
            f'</table></td>'
        )

    # ─── Vehicle rows ─────────────────────────────────────────────────────────
    veh_html = ""
    for i, r in enumerate(veh_rows[:8]):
        dv  = float(r.get("diff_val") or 0)
        clr = "#ef4444" if dv < 0 else "#22c55e" if dv > 0 else "#94a3b8"
        row_bg = "#0f1829" if i % 2 == 0 else "#111d35"
        short_v  = float(r.get("short_val")  or 0)
        excess_v = float(r.get("excess_val") or 0)
        veh_html += (
            f'<tr bgcolor="{row_bg}" style="background-color:{row_bg};">'
            f'<td style="padding:9px 14px;font-size:12px;color:#e2e8f0;'
            f'font-family:Arial,sans-serif;font-weight:600;">'
            f'&#x1F69A; {_esc(r.get("vehicle_no",""))}</td>'
            f'<td style="padding:9px 10px;font-size:12px;color:#94a3b8;'
            f'text-align:center;">{int(r.get("shops_served",0))}</td>'
            f'<td style="padding:9px 10px;font-size:12px;color:#ef4444;'
            f'font-weight:700;text-align:right;">GH₵ {short_v:,.0f}</td>'
            f'<td style="padding:9px 10px;font-size:12px;color:#22c55e;'
            f'font-weight:700;text-align:right;">GH₵ {excess_v:,.0f}</td>'
            f'<td style="padding:9px 14px;font-size:13px;color:{clr};'
            f'font-weight:900;text-align:right;">GH₵ {dv:,.0f}</td>'
            f'</tr>'
        )

    # ─── Status line (balanced %) via table bar ───────────────────────────────
    bal_pct  = max(1, min(99, round(bal_cnt  / max(total_lines, 1) * 100)))
    sh_pct   = max(0, min(99 - bal_pct, round(short_cnt  / max(total_lines, 1) * 100)))
    ex_pct   = max(0, 100 - bal_pct - sh_pct)

    html_out = f"""<!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>News Letter-Loading vs Offloading</title>
</head>
<body style="margin:0;padding:0;background-color:#020817;font-family:Arial,'Helvetica Neue',Helvetica,sans-serif;">

<table width="100%" cellpadding="0" cellspacing="0" border="0" bgcolor="#020817"
       style="background-color:#020817;">
<tr><td align="center" style="padding:24px 12px;">

<!-- ═══════════ OUTER WRAPPER ═══════════ -->
<table width="620" cellpadding="0" cellspacing="0" border="0"
       style="max-width:620px;border-radius:16px;overflow:hidden;
              border:1px solid #1e3a5f;">

  <!-- ██ HEADER ██ -->
  <tr>
    <td bgcolor="#0c1a3a"
        style="background-color:#0c1a3a;
               background-image:linear-gradient(135deg,#0c1a3a 0%,#1d4ed8 50%,#0c1a3a 100%);
               padding:28px 24px 20px;text-align:center;border-radius:16px 16px 0 0;
               border-bottom:3px solid #3b82f6;">
      <img src="{_MELCOM_LOGO}" width="150" alt="Melcom"
           style="display:block;margin:0 auto 14px auto;max-height:52px;
                  width:auto;border:0;outline:none;">
      <div style="font-size:24px;font-weight:900;color:#ffffff;letter-spacing:-0.5px;
                  font-family:Arial,sans-serif;margin-bottom:6px;">
        Loading vs Offloading Report
      </div>
      <div style="font-size:13px;color:#93c5fd;font-family:Arial,sans-serif;margin-bottom:4px;">
        {date_str}
      </div>
      <div style="font-size:11px;color:#60a5fa;font-family:Arial,sans-serif;">
        Generated on {sent_ts}
      </div>
    </td>
  </tr>

  <!-- ██ 4 KPI CARDS ██ -->
  <tr>
    <td bgcolor="#0a1628" style="background-color:#0a1628;padding:16px 12px 8px;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr>
          {_kpi_td("&#128308;","Total Shortage",f"GH₵ {short_val:,.0f}",
                   f"{short_cnt:,} lines","#ef4444","#7f1d1d")}
          {_kpi_td("&#129000;","Total Excess",f"GH₵ {excess_val:,.0f}",
                   f"{excess_cnt:,} lines","#f59e0b","#78350f")}
          {_kpi_td("&#9989;","Lines Balanced",f"{match_rate:.1f}%",
                   f"{bal_cnt:,} of {total_lines:,}","#22c55e","#14532d")}
          {_kpi_td("&#128200;","Net Difference",f"GH₵ {abs(net_diff):,.0f}",
                   net_lbl,net_clr,"#1e3a5f")}
        </tr>
      </table>
    </td>
  </tr>

  <!-- ██ COVERAGE STRIP ██ -->
  <tr>
    <td bgcolor="#0f172a" style="background-color:#0f172a;padding:10px 24px;
        border-top:1px solid #1e3a5f;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr>
          <td style="text-align:center;padding:6px;">
            <span style="font-size:20px;">&#127978;</span><br>
            <span style="font-size:18px;font-weight:900;color:#60a5fa;">{_f(summary.get("shop_count",0))}</span><br>
            <span style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Shops</span>
          </td>
          <td style="text-align:center;padding:6px;border-left:1px solid #1e3a5f;">
            <span style="font-size:20px;">&#128666;</span><br>
            <span style="font-size:18px;font-weight:900;color:#a78bfa;">{_f(summary.get("vehicle_count",0))}</span><br>
            <span style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Vehicles</span>
          </td>
          <td style="text-align:center;padding:6px;border-left:1px solid #1e3a5f;">
            <span style="font-size:20px;">&#128230;</span><br>
            <span style="font-size:18px;font-weight:900;color:#fb923c;">{_f(summary.get("item_count",0))}</span><br>
            <span style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Items</span>
          </td>
          <td style="text-align:center;padding:6px;border-left:1px solid #1e3a5f;">
            <span style="font-size:20px;">&#128203;</span><br>
            <span style="font-size:18px;font-weight:900;color:#e2e8f0;">{total_lines:,}</span><br>
            <span style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Total Lines</span>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- ██ YESTERDAY COMPLIANCE SCORE (Serial Tracker) ██ -->
  {_yest_compliance_block}

  <!-- ██ BOTTOM 5 CASHIERS — YESTERDAY ██ -->
  {_section_hdr("&#128101;",f"Bottom 5 Cashiers — Yesterday ({_yest_str})","#ef4444")}
  <tr>
    <td bgcolor="#0d1526" style="background-color:#0d1526;padding:0;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr bgcolor="#111827" style="background-color:#111827;">
          <td style="padding:7px 10px 7px 14px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;letter-spacing:1px;">Cashier</td>
          <td style="padding:7px 10px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;text-align:center;">Store</td>
          <td style="padding:7px 10px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;text-align:center;">Scans</td>
          <td style="padding:7px 10px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;text-align:center;">Score</td>
          <td style="padding:7px 14px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;text-align:right;">vs 90%</td>
        </tr>
        {_cashier_rows(_bottom5_yest)}
      </table>
    </td>
  </tr>

  <!-- ██ BOTTOM 5 CASHIERS — MTD ██ -->
  {_section_hdr("&#128101;",f"Bottom 5 Cashiers — MTD from {_mtd_str}","#f97316")}
  <tr>
    <td bgcolor="#0d1526" style="background-color:#0d1526;padding:0;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr bgcolor="#111827" style="background-color:#111827;">
          <td style="padding:7px 10px 7px 14px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;letter-spacing:1px;">Cashier</td>
          <td style="padding:7px 10px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;text-align:center;">Store</td>
          <td style="padding:7px 10px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;text-align:center;">Scans</td>
          <td style="padding:7px 10px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;text-align:center;">Score</td>
          <td style="padding:7px 14px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;text-align:right;">vs 90%</td>
        </tr>
        {_cashier_rows(_bottom5_mtd)}
      </table>
    </td>
  </tr>

  <!-- ██ DAILY SPARK CHART ██ -->
  {_section_hdr("&#128202;","Daily Net Difference (Last 14 Days)","#3b82f6")}
  <tr>
    <td bgcolor="#0d1526" style="background-color:#0d1526;padding:16px 24px 8px;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr>{spark_cells}</tr>
      </table>
      <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-top:8px;">
        <tr>
          <td style="font-size:10px;color:#fca5a5;font-family:Arial,sans-serif;">
            &#9646; Shortage
          </td>
          <td style="font-size:10px;color:#86efac;font-family:Arial,sans-serif;text-align:right;">
            &#9646; Excess
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- ██ TOP SHOPS ██ -->
  {_section_hdr("&#127978;","Top Shops by Net Difference Value","#f59e0b")}
  <tr>
    <td bgcolor="#0d1526" style="background-color:#0d1526;padding:0;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr bgcolor="#111827" style="background-color:#111827;">
          <td width="28%" style="padding:7px 10px 7px 16px;font-size:10px;color:#64748b;
              font-weight:700;text-transform:uppercase;letter-spacing:1px;">Shop</td>
          <td width="48%" style="padding:7px 10px;font-size:10px;color:#64748b;
              font-weight:700;text-transform:uppercase;letter-spacing:1px;">Variance</td>
          <td width="24%" style="padding:7px 16px 7px 6px;font-size:10px;color:#64748b;
              font-weight:700;text-transform:uppercase;letter-spacing:1px;text-align:right;">Value</td>
        </tr>
        {shop_rows_html}
      </table>
    </td>
  </tr>

  <!-- ██ TOP ITEMS ██ -->
  {_section_hdr("&#128230;","Top Items by Discrepancy Value","#a78bfa")}
  <tr>
    <td bgcolor="#0d1526" style="background-color:#0d1526;padding:0;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr bgcolor="#111827" style="background-color:#111827;">
          <td width="34%" style="padding:7px 10px 7px 16px;font-size:10px;color:#64748b;
              font-weight:700;text-transform:uppercase;letter-spacing:1px;">Item</td>
          <td width="42%" style="padding:7px 10px;font-size:10px;color:#64748b;
              font-weight:700;text-transform:uppercase;letter-spacing:1px;">Variance</td>
          <td width="24%" style="padding:7px 16px 7px 6px;font-size:10px;color:#64748b;
              font-weight:700;text-transform:uppercase;letter-spacing:1px;text-align:right;">Value</td>
        </tr>
        {item_rows_html}
      </table>
    </td>
  </tr>

  <!-- ██ VEHICLE TABLE ██ -->
  {_section_hdr("&#128666;","Vehicle Analysis","#fb923c")}
  <tr>
    <td bgcolor="#0d1526" style="background-color:#0d1526;padding:0;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr bgcolor="#111827" style="background-color:#111827;">
          <td style="padding:7px 14px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;letter-spacing:1px;">Vehicle</td>
          <td style="padding:7px 10px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;letter-spacing:1px;text-align:center;">Shops</td>
          <td style="padding:7px 10px;font-size:10px;color:#ef4444;font-weight:700;
              text-transform:uppercase;letter-spacing:1px;text-align:right;">Short</td>
          <td style="padding:7px 10px;font-size:10px;color:#22c55e;font-weight:700;
              text-transform:uppercase;letter-spacing:1px;text-align:right;">Excess</td>
          <td style="padding:7px 14px;font-size:10px;color:#94a3b8;font-weight:700;
              text-transform:uppercase;letter-spacing:1px;text-align:right;">Net</td>
        </tr>
        {veh_html}
      </table>
    </td>
  </tr>

  <!-- ██ VIEW DASHBOARD CTA ██ -->
  <tr>
    <td bgcolor="#0a1628" style="background-color:#0a1628;padding:20px 24px;
        text-align:center;border-top:1px solid #1e3a5f;">
      <div style="font-size:13px;color:#94a3b8;font-family:Arial,sans-serif;margin-bottom:14px;">
        View the full interactive dashboard for drill-downs, filters and detailed analysis.
      </div>
      <a href="{_DASHBOARD_URL}"
         style="display:inline-block;background-color:#1d4ed8;color:#ffffff;
                font-size:14px;font-weight:700;font-family:Arial,sans-serif;
                padding:13px 32px;border-radius:8px;text-decoration:none;
                border:1px solid #3b82f6;letter-spacing:0.3px;">
        &#128279; View Full Dashboard
      </a>
    </td>
  </tr>

  <!-- ██ FOOTER ██ -->
  <tr>
    <td bgcolor="#060d1a" style="background-color:#060d1a;padding:16px 24px;
        text-align:center;border-top:1px solid #1e3a5f;border-radius:0 0 16px 16px;">
      <img src="{_MELCOM_LOGO}" width="90" alt="Melcom"
           style="display:block;margin:0 auto 10px auto;max-height:30px;
                  width:auto;border:0;outline:none;">
      <div style="font-size:11px;color:#475569;font-family:Arial,sans-serif;line-height:1.6;">
        Melcom Group &bull; WH Analytics Dashboard &bull; {sent_ts}<br>
        <span style="color:#334155;">
          This report is auto-generated. Do not reply to this email.
        </span>
      </div>
    </td>
  </tr>

</table>
<!-- ═══════════ END WRAPPER ═══════════ -->

</td></tr>
</table>
</body>
</html>"""
    return html_out



def _send_lvo_newsletter(
    summary: dict,
    shop_rows: list[dict],
    shop_meta_map: dict,
    items_rows: list[dict],
    veh_rows: list[dict],
    daily_rows: list[dict],
    start_date,
    end_date,
    excel_bytes: bytes | None = None,
) -> tuple[bool, str]:
    """Build and send the LVO newsletter with optional Excel attachment."""
    import smtplib
    from email import encoders as _enc
    from email.mime.base import MIMEBase
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    try:
        html_body = _build_lvo_newsletter_html(
            summary, shop_rows, shop_meta_map, items_rows, veh_rows, daily_rows,
            start_date, end_date,
        )
        subject = "News Letter-Loading vs Offloading"
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"]    = _NEWSLETTER_FROM
        msg["To"]      = _NEWSLETTER_TO
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        # ── Excel attachment ─────────────────────────────────────────────
        if excel_bytes:
            xl_part = MIMEBase(
                "application",
                "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            xl_part.set_payload(excel_bytes)
            _enc.encode_base64(xl_part)
            xl_name = f"LVO_Detail_{end_date.strftime('%d%b%Y')}.xlsx"
            xl_part.add_header("Content-Disposition",
                               f'attachment; filename="{xl_name}"')
            msg.attach(xl_part)

        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=20) as server:
            server.sendmail(_NEWSLETTER_FROM, [_NEWSLETTER_TO], msg.as_string())

        return True, f"Newsletter sent to {_NEWSLETTER_TO}"
    except Exception as e:
        return False, f"Email failed: {e}"


# ─────────────────────────────────────────────────────────────
# EDA RENDER
# ─────────────────────────────────────────────────────────────

_OVL_BG    = "rgba(12,26,58,0.55)"
_OVL_GRID  = "rgba(148,163,184,0.10)"
_OVL_FONT  = "#e2e8f0"
_OVL_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=_OVL_BG,
    font=dict(family="Inter, Segoe UI, sans-serif", size=11, color=_OVL_FONT),
    margin=dict(l=10, r=10, t=36, b=10),
    hoverlabel=dict(bgcolor="rgba(8,15,34,0.96)", font_color="white",
                    font_family="Inter", font_size=12),
)


def _fmt(v, dec=0):
    try:
        return f"{float(v):,.{dec}f}"
    except Exception:
        return "0"


def _kpi(label, value, sub="", color="#3b82f6", alert=False):
    bord = "#ef4444" if alert else color
    bg   = "rgba(239,68,68,0.07)" if alert else "rgba(12,26,58,0.8)"
    return (
        f'<div style="background:{bg};border:1px solid {bord}30;border-radius:12px;'
        f'padding:14px 12px;text-align:center;min-height:90px;">'
        f'<div style="font-size:0.60rem;letter-spacing:0.09em;text-transform:uppercase;'
        f'color:#64748b;font-weight:700;margin-bottom:5px;">{label}</div>'
        f'<div style="font-size:1.7rem;font-weight:900;color:{bord};line-height:1.1;'
        f'font-variant-numeric:tabular-nums;">{value}</div>'
        f'<div style="font-size:0.62rem;color:#475569;margin-top:3px;">{sub}</div>'
        f'</div>'
    )


def render_eda_analytics(
    data: list[dict],
    shop_rows: list[dict],
    shop_meta_map: dict,
    start_date,
    end_date,
    is_master: bool = False,
) -> None:

    summary    = load_discrepancy_summary(start_date, end_date)
    daily_rows = load_daily_trend(start_date, end_date)
    daily_df   = pd.DataFrame(daily_rows)
    items_rows = load_top_discrepancy_items(start_date, end_date, 20)
    items_df   = pd.DataFrame(items_rows)
    veh_rows   = load_vehicle_summary(start_date, end_date)
    veh_df     = pd.DataFrame(veh_rows)
    hm_df      = pd.DataFrame(load_shop_date_grid(start_date, end_date))

    def _fv(k, dec=0):
        try: return _fmt(float(summary.get(k) or 0), dec)
        except: return "0"

    excess_val  = float(summary.get("excess_val")  or 0)
    short_val   = float(summary.get("short_val")   or 0)
    total_abs   = float(summary.get("total_abs_val") or 0)
    excess_cnt  = int(summary.get("excess_count")  or 0)
    short_cnt   = int(summary.get("short_count")   or 0)
    bal_cnt     = int(summary.get("balanced_count") or 0)
    total_lines = int(summary.get("total_lines")   or 0)
    match_rate  = round(bal_cnt / total_lines * 100, 1) if total_lines else 0

    # ── Section header + newsletter button ───────────────────────────────────────
    hdr_col, btn_col = st.columns([5, 1], gap="small")
    with hdr_col:
        st.markdown(
            '<div style="font-size:0.65rem;font-weight:800;color:#475569;letter-spacing:0.12em;'
            'text-transform:uppercase;margin:18px 0 6px;">Intelligence Center — EDA</div>',
            unsafe_allow_html=True,
        )
    with btn_col:
        pass  # Newsletter now sent automatically via Update Data button

    # ── EDA KPI row ─────────────────────────────────────────────────────────────
    k1,k2,k3,k4,k5 = st.columns(5, gap="small")
    kpis = [
        (k1, "Total Shortage Value",  f"GH₵ {_fv('short_val',2)}",
         f"{short_cnt:,} short lines",  "#ef4444", short_val > excess_val),
        (k2, "Total Excess Value",    f"GH₵ {_fv('excess_val',2)}",
         f"{excess_cnt:,} excess lines", "#f59e0b", False),
        (k3, "Lines Balanced",        f"{match_rate:.1f}%",
         f"{bal_cnt:,} of {total_lines:,} lines", "#10b981", match_rate < 50),
        (k4, "Shops with Shortages",
         str(sum(1 for r in shop_rows if float(r.get("short_val") or 0) > 0)),
         f"of {len(shop_rows)} total shops", "#8b5cf6", False),
        (k5, "Vehicles Tracked",      _fv("vehicle_count"),
         f"across {_fv('shop_count')} shops", "#22d3ee", False),
    ]
    for col, lbl, val, sub, clr, alrt in kpis:
        with col:
            st.markdown(_kpi(lbl, val, sub, clr, alrt), unsafe_allow_html=True)

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

    # ── EDA Tabs (tables hidden here — in Data Tables tab only) ──────────────────
    tab_ov, tab_shop, tab_item, tab_veh = st.tabs([
        "📊 Overview", "🏪 Shop Intelligence",
        "📦 Item Analysis", "🚚 Vehicle Analysis",
    ])

    # ══════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════
    with tab_ov:
        ov1, ov2 = st.columns([1, 1.6], gap="large")

        with ov1:
            # Donut: excess / short / balanced
            st.markdown(
                '<div style="font-size:0.68rem;color:#94a3b8;font-weight:700;text-transform:uppercase;'
                'letter-spacing:0.08em;margin-bottom:4px;">Discrepancy Breakdown</div>',
                unsafe_allow_html=True,
            )
            labels = ["Balanced", "Short Received", "Excess Received"]
            values = [bal_cnt, short_cnt, excess_cnt]
            clrs   = ["#10b981", "#ef4444", "#f59e0b"]
            fig_d  = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.60,
                marker=dict(colors=clrs, line=dict(color="rgba(0,0,0,0.2)", width=1)),
                textinfo="percent", textfont=dict(size=11, color="white"),
                hovertemplate="%{label}<br>Lines: %{value:,}<br>Share: %{percent}<extra></extra>",
            ))
            fig_d.add_annotation(
                text=f"<b>{match_rate:.0f}%</b><br>Balanced",
                x=0.5, y=0.5,
                font=dict(size=14, color="#e2e8f0", family="Inter"),
                showarrow=False, align="center",
            )
            fig_d.update_layout(**{**_OVL_LAYOUT, "height": 280,
                "legend": dict(orientation="v", x=1.01, y=0.5,
                               font=dict(size=10, color=_OVL_FONT)),
                "margin": dict(l=10, r=10, t=16, b=10),
            })
            st.plotly_chart(fig_d, use_container_width=True, key="ovl_donut")

            # Value breakdown cards
            for lbl, val, clr in [
                ("Total Short Value",   f"GH₵ {short_val:,.2f}",   "#ef4444"),
                ("Total Excess Value",  f"GH₵ {excess_val:,.2f}",  "#f59e0b"),
                ("Net Difference",
                 f"GH₵ {excess_val - short_val:+,.2f}",
                 "#10b981" if excess_val >= short_val else "#ef4444"),
            ]:
                st.markdown(
                    f'<div style="background:rgba(12,26,58,0.7);border-left:3px solid {clr};'
                    f'border-radius:6px;padding:7px 12px;margin-bottom:4px;">'
                    f'<div style="font-size:0.62rem;color:#64748b;font-weight:700;">{lbl}</div>'
                    f'<div style="font-size:0.95rem;font-weight:800;color:{clr};">{val}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with ov2:
            st.markdown(
                '<div style="font-size:0.68rem;color:#94a3b8;font-weight:700;text-transform:uppercase;'
                'letter-spacing:0.08em;margin-bottom:4px;">Daily Loaded vs Offloaded (Qty)</div>',
                unsafe_allow_html=True,
            )
            if not daily_df.empty:
                daily_df["date"] = pd.to_datetime(daily_df["date"])
                daily_df["qty_loaded"]    = pd.to_numeric(daily_df["qty_loaded"],    errors="coerce").fillna(0)
                daily_df["qty_offloaded"] = pd.to_numeric(daily_df["qty_offloaded"], errors="coerce").fillna(0)
                daily_df["diff_val"]      = pd.to_numeric(daily_df["diff_val"],      errors="coerce").fillna(0)
                daily_df["date_lbl"] = daily_df["date"].dt.strftime("%d %b")

                fig_daily = go.Figure()
                fig_daily.add_trace(go.Bar(
                    x=daily_df["date_lbl"], y=daily_df["qty_loaded"],
                    name="WH Loaded", marker_color="rgba(59,130,246,0.75)",
                    hovertemplate="Loaded: %{y:,.0f}<extra></extra>",
                ))
                fig_daily.add_trace(go.Bar(
                    x=daily_df["date_lbl"], y=daily_df["qty_offloaded"],
                    name="Shop Received", marker_color="rgba(16,185,129,0.75)",
                    hovertemplate="Received: %{y:,.0f}<extra></extra>",
                ))
                fig_daily.update_layout(**{**_OVL_LAYOUT, "height": 280,
                    "barmode": "group",
                    "xaxis": dict(showgrid=False, tickfont=dict(size=9)),
                    "yaxis": dict(showgrid=True, gridcolor=_OVL_GRID),
                    "legend": dict(orientation="h", y=-0.2, font=dict(size=10, color=_OVL_FONT)),
                })
                st.plotly_chart(fig_daily, use_container_width=True, key="ovl_daily_bar")

            # Diff value trend
            st.markdown(
                '<div style="font-size:0.68rem;color:#94a3b8;font-weight:700;text-transform:uppercase;'
                'letter-spacing:0.08em;margin-bottom:4px;margin-top:8px;">Daily Net Difference Value (GH₵)</div>',
                unsafe_allow_html=True,
            )
            if not daily_df.empty:
                clrs_bar = ["#ef4444" if v < 0 else "#f59e0b" for v in daily_df["diff_val"]]
                fig_dv = go.Figure(go.Bar(
                    x=daily_df["date_lbl"], y=daily_df["diff_val"],
                    marker_color=clrs_bar, marker_line_width=0,
                    text=[f"{v:,.0f}" for v in daily_df["diff_val"]],
                    textposition="outside", textfont=dict(size=8, color=_OVL_FONT),
                    hovertemplate="Diff: GH₵ %{y:,.2f}<extra></extra>",
                ))
                fig_dv.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
                fig_dv.update_layout(**{**_OVL_LAYOUT, "height": 200,
                    "xaxis": dict(showgrid=False, tickfont=dict(size=9)),
                    "yaxis": dict(showgrid=True, gridcolor=_OVL_GRID),
                    "margin": dict(l=10, r=10, t=10, b=10),
                })
                st.plotly_chart(fig_dv, use_container_width=True, key="ovl_diff_bar")
            else:
                st.info("No daily trend data.")

        # ── Table 1: Exception Shopwise Difference ──────────────────────────
        st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
        render_shop_summary_table(shop_rows, shop_meta_map)

        # ── Table 2: Loading vs Offloading (Detail) ──────────────────────────
        available_shop_codes = sorted({
            str(r.get("shop_code", "") or "").strip().upper()
            for r in data
            if str(r.get("shop_code", "") or "").strip()
        })
        selected_shop, selected_receive, selected_items = render_detail_filter_header(
            available_shop_codes, shop_meta_map, data
        )
        filtered_detail = data if selected_shop == "All" else [
            r for r in data
            if str(r.get("shop_code", "") or "").strip().upper() == selected_shop
        ]
        if selected_receive == "Short Received":
            filtered_detail = [r for r in filtered_detail if float(r.get("diff_val") or 0) < 0]
        elif selected_receive == "Excess Received":
            filtered_detail = [r for r in filtered_detail if float(r.get("diff_val") or 0) > 0]
        if selected_items:
            sel_set = set(selected_items)
            filtered_detail = [
                r for r in filtered_detail
                if str(r.get("item_name", "") or "").strip() in sel_set
            ]
        render_detail_table(filtered_detail, shop_meta_map, export_rows=data)

    # ══════════════════════════════════════════
    # TAB 2 — SHOP INTELLIGENCE
    # ══════════════════════════════════════════
    with tab_shop:
        if not shop_rows:
            st.info("No shop data.")
        else:
            sh_df = pd.DataFrame(shop_rows)
            for c in ["short_val","excess_val","diff_val","qty_loaded","qty_offloaded"]:
                if c in sh_df.columns:
                    sh_df[c] = pd.to_numeric(sh_df[c], errors="coerce").fillna(0)
            sh_df["abs_diff"] = sh_df["diff_val"].abs()
            sh_df["shop_label"] = sh_df["shop_code"].apply(
                lambda c: f"{c} ({shop_meta_map.get(str(c).upper(),{}).get('shop_name','')[:15]})"
                if shop_meta_map.get(str(c).upper(), {}).get("shop_name") else str(c)
            )

            sh1, sh2 = st.columns([1.6, 1], gap="large")

            with sh1:
                st.markdown(
                    '<div style="font-size:0.68rem;color:#94a3b8;font-weight:700;text-transform:uppercase;'
                    'letter-spacing:0.08em;margin-bottom:4px;">'
                    'Shop Ranking — Net Difference Value (worst first)</div>',
                    unsafe_allow_html=True,
                )
                top_n  = sh_df.nlargest(min(25, len(sh_df)), "abs_diff")
                bar_clr = ["#ef4444" if v < 0 else "#f59e0b" for v in top_n["diff_val"]]
                fig_sh = go.Figure(go.Bar(
                    x=top_n["diff_val"], y=top_n["shop_label"],
                    orientation="h", marker_color=bar_clr, marker_line_width=0,
                    text=[f"GH₵ {v:,.0f}" for v in top_n["diff_val"]],
                    textposition="outside", textfont=dict(size=9, color=_OVL_FONT),
                    customdata=top_n[["short_val","excess_val"]].values,
                    hovertemplate=(
                        "<b>%{y}</b><br>Net: GH₵ %{x:,.2f}<br>"
                        "Short: GH₵ %{customdata[0]:,.2f}<br>"
                        "Excess: GH₵ %{customdata[1]:,.2f}<extra></extra>"
                    ),
                ))
                fig_sh.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)
                fig_sh.update_layout(**{**_OVL_LAYOUT,
                    "height": max(360, len(top_n) * 26),
                    "xaxis": dict(showgrid=True, gridcolor=_OVL_GRID, title="Net Diff Value (GH₵)"),
                    "yaxis": dict(automargin=True, tickfont=dict(size=9)),
                    "showlegend": False,
                })
                st.plotly_chart(fig_sh, use_container_width=True, key="ovl_shop_bar")

            with sh2:
                # Worst 5 by short
                st.markdown(
                    '<div style="font-size:0.68rem;color:#ef4444;font-weight:700;text-transform:uppercase;'
                    'letter-spacing:0.08em;margin-bottom:5px;">Most Short (top 5)</div>',
                    unsafe_allow_html=True,
                )
                worst5 = sh_df.nlargest(5, "short_val")
                for _, r in worst5.iterrows():
                    name = shop_meta_map.get(str(r["shop_code"]).upper(), {}).get("shop_name", "")
                    mgr  = shop_meta_map.get(str(r["shop_code"]).upper(), {}).get("manager_name", "")
                    st.markdown(
                        f'<div style="background:rgba(239,68,68,0.08);border-left:3px solid #ef4444;'
                        f'border-radius:6px;padding:6px 10px;margin-bottom:4px;">'
                        f'<div style="color:#fca5a5;font-weight:700;font-size:0.72rem;">'
                        f'{r["shop_code"]} {f"({name[:20]})" if name else ""}</div>'
                        f'<div style="color:#64748b;font-size:0.63rem;">'
                        f'Short: <b style="color:#ef4444">GH₵ {float(r["short_val"]):,.0f}</b>'
                        f'{f" | Mgr: {mgr[:18]}" if mgr else ""}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

                # Balanced shops
                balanced_shops = sh_df[sh_df["diff_val"].abs() < 1]
                st.markdown(
                    f'<div style="font-size:0.68rem;color:#10b981;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:0.08em;margin-bottom:5px;">Balanced Shops ({len(balanced_shops)})</div>',
                    unsafe_allow_html=True,
                )
                if not balanced_shops.empty:
                    pills = " ".join(
                        f'<span style="background:rgba(16,185,129,0.12);color:#6ee7b7;'
                        f'font-size:0.60rem;padding:2px 7px;border-radius:4px;margin:2px;'
                        f'display:inline-block;">{r["shop_code"]}</span>'
                        for _, r in balanced_shops.head(20).iterrows()
                    )
                    st.markdown(f'<div style="line-height:2;">{pills}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div style="font-size:0.68rem;color:#64748b;">No fully balanced shops</div>',
                        unsafe_allow_html=True,
                    )

            # Shop × Date Heatmap
            if not hm_df.empty:
                st.markdown(
                    '<div style="font-size:0.68rem;color:#94a3b8;font-weight:700;text-transform:uppercase;'
                    'letter-spacing:0.08em;margin:14px 0 4px;">'
                    'Shop × Date — Net Difference Value Heatmap</div>',
                    unsafe_allow_html=True,
                )
                hm_df["date"]     = pd.to_datetime(hm_df["date"])
                hm_df["diff_val"] = pd.to_numeric(hm_df["diff_val"], errors="coerce").fillna(0)
                hm_df["date_lbl"] = hm_df["date"].dt.strftime("%d %b")
                pivot_hm = hm_df.pivot_table(
                    index="shop_code", columns="date_lbl", values="diff_val", aggfunc="sum"
                )
                # Order columns chronologically
                all_dates = sorted(hm_df["date"].unique())
                date_lbls = [pd.Timestamp(d).strftime("%d %b") for d in all_dates]
                ordered_cols = [c for c in date_lbls if c in pivot_hm.columns]
                pivot_hm = pivot_hm.reindex(columns=ordered_cols)

                abs_max = max(1.0, float(pivot_hm.abs().max().max() or 1))
                fig_hm = go.Figure(go.Heatmap(
                    z=pivot_hm.values,
                    x=list(pivot_hm.columns),
                    y=list(pivot_hm.index),
                    colorscale=[
                        [0.0, "#7f1d1d"], [0.4, "#991b1b"],
                        [0.5, "#1e3a3a"],
                        [0.6, "#14532d"], [1.0, "#064e3b"]
                    ],
                    zmid=0, zmin=-abs_max, zmax=abs_max,
                    text=[[f"{v:,.0f}" if not pd.isna(v) else "—" for v in row]
                          for row in pivot_hm.values],
                    texttemplate="%{text}",
                    textfont=dict(size=8, color="white"),
                    hovertemplate="<b>%{y}</b> | %{x}<br>GH₵ %{z:,.2f}<extra></extra>",
                    colorbar=dict(tickfont=dict(color="#94a3b8", size=9), len=0.8, thickness=10),
                    xgap=1, ygap=1,
                ))
                hm_h = max(300, len(pivot_hm) * 22)
                fig_hm.update_layout(**{**_OVL_LAYOUT, "height": hm_h,
                    "xaxis": dict(side="top", tickfont=dict(size=9), showgrid=False),
                    "yaxis": dict(automargin=True, tickfont=dict(size=9)),
                })
                st.plotly_chart(fig_hm, use_container_width=True, key="ovl_heatmap")

    # ══════════════════════════════════════════
    # TAB 3 — ITEM ANALYSIS
    # ══════════════════════════════════════════
    with tab_item:
        if items_df.empty:
            st.info("No item data.")
        else:
            for c in ["diff_val","abs_diff_val","qty_loaded","qty_offloaded","diff_qty"]:
                if c in items_df.columns:
                    items_df[c] = pd.to_numeric(items_df[c], errors="coerce").fillna(0)

            it1, it2 = st.columns([1.8, 1], gap="large")

            with it1:
                st.markdown(
                    '<div style="font-size:0.68rem;color:#94a3b8;font-weight:700;text-transform:uppercase;'
                    'letter-spacing:0.08em;margin-bottom:4px;">'
                    'Top 20 Items by Absolute Discrepancy Value</div>',
                    unsafe_allow_html=True,
                )
                items_sorted = items_df.nlargest(20, "abs_diff_val")
                items_sorted["item_label"] = items_sorted.apply(
                    lambda r: f"{r['item_code']} — {str(r.get('item_name','') or '')[:35]}", axis=1
                )
                bar_clrs = ["#ef4444" if v < 0 else "#f59e0b" for v in items_sorted["diff_val"]]
                fig_it = go.Figure(go.Bar(
                    x=items_sorted["diff_val"],
                    y=items_sorted["item_label"],
                    orientation="h",
                    marker_color=bar_clrs, marker_line_width=0,
                    text=[f"GH₵ {v:,.0f}" for v in items_sorted["diff_val"]],
                    textposition="outside", textfont=dict(size=8, color=_OVL_FONT),
                    customdata=items_sorted[["abs_diff_val","shop_count","line_count"]].values,
                    hovertemplate=(
                        "<b>%{y}</b><br>Net diff: GH₵ %{x:,.2f}<br>"
                        "Abs diff: GH₵ %{customdata[0]:,.2f}<br>"
                        "Shops: %{customdata[1]}<br>Lines: %{customdata[2]}<extra></extra>"
                    ),
                ))
                fig_it.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)
                fig_it.update_layout(**{**_OVL_LAYOUT,
                    "height": max(360, len(items_sorted) * 26),
                    "xaxis": dict(showgrid=True, gridcolor=_OVL_GRID, title="Net Diff Value (GH₵)"),
                    "yaxis": dict(automargin=True, tickfont=dict(size=9)),
                    "showlegend": False,
                })
                st.plotly_chart(fig_it, use_container_width=True, key="ovl_item_bar")

            with it2:
                st.markdown(
                    '<div style="font-size:0.68rem;color:#94a3b8;font-weight:700;text-transform:uppercase;'
                    'letter-spacing:0.08em;margin-bottom:6px;">Item Summary Table</div>',
                    unsafe_allow_html=True,
                )
                th = ("font-size:0.59rem;color:#475569;font-weight:700;text-transform:uppercase;"
                      "letter-spacing:0.06em;padding:5px 7px;border-bottom:1px solid #1e293b;"
                      "white-space:nowrap;")
                td = "font-size:0.66rem;padding:4px 7px;color:#cbd5e1;white-space:nowrap;"

                rows_html = ""
                for i, (_, r) in enumerate(items_sorted.head(15).iterrows()):
                    dv   = float(r["diff_val"])
                    clr  = "#ef4444" if dv < 0 else "#f59e0b" if dv > 0 else "#10b981"
                    rows_html += (
                        f'<tr style="background:{"rgba(12,26,58,0.5)" if i%2==0 else "transparent"};'
                        f'border-bottom:1px solid rgba(255,255,255,0.03);">'
                        f'<td style="{td}">{str(r["item_code"])[:12]}</td>'
                        f'<td style="{td}">{str(r.get("item_name","") or "")[:25]}</td>'
                        f'<td style="{td}color:{clr};font-weight:800;">GH₵ {dv:,.0f}</td>'
                        f'<td style="{td}color:#94a3b8;">{int(r["shop_count"])}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<div style="overflow-x:auto;border:1px solid #1e293b;border-radius:8px;">'
                    f'<table style="width:100%;border-collapse:collapse;">'
                    f'<thead><tr style="background:rgba(15,23,42,0.8);">'
                    f'<th style="{th}">Code</th><th style="{th}">Item</th>'
                    f'<th style="{th}">Net Val</th><th style="{th}">Shops</th>'
                    f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
                    unsafe_allow_html=True,
                )

    # ══════════════════════════════════════════
    # TAB 4 — VEHICLE ANALYSIS
    # ══════════════════════════════════════════
    with tab_veh:
        if veh_df.empty:
            st.info("No vehicle data for this period.")
        else:
            for c in ["diff_val","abs_diff_val","short_val","excess_val","qty_loaded","qty_offloaded"]:
                if c in veh_df.columns:
                    veh_df[c] = pd.to_numeric(veh_df[c], errors="coerce").fillna(0)

            v1, v2 = st.columns([1.6, 1], gap="large")

            with v1:
                st.markdown(
                    '<div style="font-size:0.68rem;color:#94a3b8;font-weight:700;text-transform:uppercase;'
                    'letter-spacing:0.08em;margin-bottom:4px;">'
                    'Vehicle Ranking — Absolute Discrepancy (GH₵)</div>',
                    unsafe_allow_html=True,
                )
                top_veh = veh_df.nlargest(min(20, len(veh_df)), "abs_diff_val")
                veh_clrs = ["#ef4444" if v < 0 else "#f59e0b" for v in top_veh["diff_val"]]
                fig_v = go.Figure(go.Bar(
                    x=top_veh["diff_val"],
                    y=top_veh["vehicle_no"],
                    orientation="h",
                    marker_color=veh_clrs, marker_line_width=0,
                    text=[f"GH₵ {v:,.0f}" for v in top_veh["diff_val"]],
                    textposition="outside", textfont=dict(size=9, color=_OVL_FONT),
                    customdata=top_veh[["short_val","excess_val","shops_served","lines"]].values,
                    hovertemplate=(
                        "<b>%{y}</b><br>Net: GH₵ %{x:,.2f}<br>"
                        "Short: GH₵ %{customdata[0]:,.2f} | Excess: GH₵ %{customdata[1]:,.2f}<br>"
                        "Shops: %{customdata[2]} | Lines: %{customdata[3]}<extra></extra>"
                    ),
                ))
                fig_v.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)
                fig_v.update_layout(**{**_OVL_LAYOUT,
                    "height": max(320, len(top_veh) * 28),
                    "xaxis": dict(showgrid=True, gridcolor=_OVL_GRID, title="Net Diff Value (GH₵)"),
                    "yaxis": dict(automargin=True, tickfont=dict(size=9)),
                    "showlegend": False,
                })
                st.plotly_chart(fig_v, use_container_width=True, key="ovl_veh_bar")

            with v2:
                st.markdown(
                    '<div style="font-size:0.68rem;color:#94a3b8;font-weight:700;text-transform:uppercase;'
                    'letter-spacing:0.08em;margin-bottom:5px;">Vehicle Summary Table</div>',
                    unsafe_allow_html=True,
                )
                th = ("font-size:0.59rem;color:#475569;font-weight:700;text-transform:uppercase;"
                      "letter-spacing:0.06em;padding:5px 7px;border-bottom:1px solid #1e293b;white-space:nowrap;")
                td = "font-size:0.66rem;padding:4px 7px;color:#cbd5e1;white-space:nowrap;"
                rows_html = ""
                for i, (_, r) in enumerate(top_veh.iterrows()):
                    dv  = float(r["diff_val"])
                    clr = "#ef4444" if dv < 0 else "#f59e0b" if dv > 0 else "#10b981"
                    rows_html += (
                        f'<tr style="background:{"rgba(12,26,58,0.5)" if i%2==0 else "transparent"};'
                        f'border-bottom:1px solid rgba(255,255,255,0.03);">'
                        f'<td style="{td}">{r["vehicle_no"]}</td>'
                        f'<td style="{td}color:{clr};font-weight:800;">GH₵ {dv:,.0f}</td>'
                        f'<td style="{td}color:#64748b;">GH₵ {float(r["short_val"]):,.0f}</td>'
                        f'<td style="{td}color:#f59e0b;">GH₵ {float(r["excess_val"]):,.0f}</td>'
                        f'<td style="{td}color:#94a3b8;">{int(r["shops_served"])}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<div style="overflow-x:auto;border:1px solid #1e293b;border-radius:8px;">'
                    f'<table style="width:100%;border-collapse:collapse;">'
                    f'<thead><tr style="background:rgba(15,23,42,0.8);">'
                    f'<th style="{th}">Vehicle</th><th style="{th}">Net</th>'
                    f'<th style="{th}">Short</th><th style="{th}">Excess</th>'
                    f'<th style="{th}">Shops</th>'
                    f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
                    unsafe_allow_html=True,
                )

                # Short vs Excess donut for vehicles
                st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
                total_short_v  = float(veh_df["short_val"].sum())
                total_excess_v = float(veh_df["excess_val"].sum())
                fig_vd = go.Figure(go.Pie(
                    labels=["Short Received", "Excess Received"],
                    values=[total_short_v, total_excess_v],
                    hole=0.55,
                    marker=dict(colors=["#ef4444","#f59e0b"],
                                line=dict(color="rgba(0,0,0,0.2)", width=1)),
                    textinfo="percent", textfont=dict(size=10, color="white"),
                ))
                fig_vd.update_layout(**{**_OVL_LAYOUT, "height": 200,
                    "margin": dict(l=10, r=10, t=10, b=10),
                    "legend": dict(orientation="h", y=-0.2, font=dict(size=9, color=_OVL_FONT)),
                })
                st.plotly_chart(fig_vd, use_container_width=True, key="ovl_veh_donut")



# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_shop_summary(start_date, end_date) -> list[dict]:
    diff_val_expr = "((COALESCE(qty_offloaded, 0) - COALESCE(qty_loaded, 0)) * COALESCE(price, 0))"
    query = f"""
        SELECT
            shop_code,
            SUM(COALESCE(qty_loaded, 0))::numeric AS qty_loaded,
            SUM(COALESCE(qty_loaded, 0) * COALESCE(price, 0))::numeric AS value_loaded,
            SUM(COALESCE(qty_offloaded, 0))::numeric AS qty_offloaded,
            SUM(COALESCE(qty_offloaded, 0) * COALESCE(price, 0))::numeric AS value_offloaded,
            SUM(CASE WHEN (COALESCE(qty_offloaded, 0) - COALESCE(qty_loaded, 0)) < 0 THEN ABS(COALESCE(qty_offloaded, 0) - COALESCE(qty_loaded, 0)) ELSE 0 END)::numeric AS short_qty,
            SUM(CASE WHEN ({diff_val_expr}) < 0 THEN ABS({diff_val_expr}) ELSE 0 END)::numeric AS short_val,
            SUM(CASE WHEN (COALESCE(qty_offloaded, 0) - COALESCE(qty_loaded, 0)) > 0 THEN (COALESCE(qty_offloaded, 0) - COALESCE(qty_loaded, 0)) ELSE 0 END)::numeric AS excess_qty,
            SUM(CASE WHEN ({diff_val_expr}) > 0 THEN ({diff_val_expr}) ELSE 0 END)::numeric AS excess_val,
            SUM(COALESCE(qty_offloaded, 0) - COALESCE(qty_loaded, 0))::numeric AS diff_qty,
            SUM({diff_val_expr})::numeric AS diff_val
        FROM offloading_vs_loading
        WHERE date BETWEEN %(s)s AND %(e)s
        GROUP BY shop_code
        ORDER BY ABS(SUM({diff_val_expr})) DESC, ABS(SUM(diff_qty)) DESC
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, {"s": start_date, "e": end_date})
            return [dict(r) for r in cur.fetchall()]


@st.cache_data(ttl=300)
def load_shop_meta_map() -> dict[str, dict[str, str]]:
    query = """
        WITH mgr AS (
            SELECT
                UPPER(TRIM(shop_code)) AS shop_code,
                MAX(TRIM(shop_description)) AS shop_name,
                STRING_AGG(DISTINCT TRIM(shop_manager_name), ', ' ORDER BY TRIM(shop_manager_name)) AS manager_name
            FROM shopmgrname
            WHERE is_current = TRUE
              AND TRIM(COALESCE(shop_code, '')) <> ''
            GROUP BY UPPER(TRIM(shop_code))
        ),
        stg AS (
            SELECT
                UPPER(TRIM(shop_code)) AS shop_code,
                MAX(TRIM(shop_name)) AS shop_name
            FROM offloading_loading_staging
            WHERE TRIM(COALESCE(shop_code, '')) <> ''
            GROUP BY UPPER(TRIM(shop_code))
        )
        SELECT
            COALESCE(mgr.shop_code, stg.shop_code) AS shop_code,
            COALESCE(NULLIF(mgr.shop_name, ''), stg.shop_name, '') AS shop_name,
            COALESCE(mgr.manager_name, '') AS manager_name
        FROM stg
        FULL OUTER JOIN mgr ON mgr.shop_code = stg.shop_code
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()

    result: dict[str, dict[str, str]] = {}
    for r in rows:
        code = str(r.get("shop_code", "") or "").strip().upper()
        if not code:
            continue
        result[code] = {
            "shop_name": str(r.get("shop_name", "") or "").strip(),
            "manager_name": str(r.get("manager_name", "") or "").strip(),
        }
    return result


def render_metrics(rows: list[dict], start_date, end_date) -> None:
    total_wh_loaded_qty = sum(float(r.get("qty_loaded") or 0) for r in rows)
    total_shop_receiving_qty = sum(float(r.get("qty_offloaded") or 0) for r in rows)
    total_diff_qty = sum(float(r.get("diff_qty") or 0) for r in rows)
    total_diff_val = sum(float(r.get("diff_val") or 0) for r in rows)

    st.markdown(
        f"<div style='color:#94a3b8;font-size:0.8rem;margin:0.2rem 0 0.6rem;'>Selected Range: {start_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')}</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    cards = [
        ("WH Loaded Qty", f"{total_wh_loaded_qty:,.0f}"),
        ("Shop Receiving Qty", f"{total_shop_receiving_qty:,.0f}"),
        ("Final Difference QTY", f"{total_diff_qty:,.0f}"),
        ("Final Difference Value", f"{total_diff_val:,.2f}"),
    ]

    for col, (label, value) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div style="background:linear-gradient(160deg,#0c1a3a 0%,#080f22 100%);border:1px solid rgba(59,130,246,0.22);border-radius:14px;padding:12px;">
                    <div style="color:#94a3b8;font-size:0.70rem;text-transform:uppercase;font-weight:700;letter-spacing:0.05em;">{label}</div>
                    <div style="color:#f8fafc;font-size:1.4rem;font-weight:800;line-height:1.15;margin-top:2px;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_monthly_graph(monthly_rows: list[dict]) -> None:
    if not monthly_rows:
        st.info("No monthly data for selected range.")
        return

    x_vals = [r.get("month_label", "") for r in monthly_rows]
    diff_val = [float(r.get("diff_val") or 0) for r in monthly_rows]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=diff_val,
            mode="lines+markers",
            name="Final Difference Value",
            line=dict(color="#f59e0b", width=3),
            marker=dict(size=7, color="#f59e0b"),
        )
    )

    fig.update_layout(
        title="Month-on-Month Difference Value",
        xaxis_title="Month",
        yaxis_title="Final Difference Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,26,51,0.65)",
        font=dict(color="#e2e8f0"),
        margin=dict(l=30, r=30, t=50, b=30),
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_sortable_html_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    sort_values: list[list[str]],
    table_id: str,
    min_width_px: int = 1120,
    row_link_values: list[str] | None = None,
    row_link_param: str = "detail_shop",
    show_title: bool = True,
    cell_styles: dict[tuple[int, int], str] | None = None,
    enable_download_hover: bool = False,
    download_file_name: str = "table_export.csv",
    excel_base64: str | None = None,
    total_row: list[str] | None = None,
) -> None:
    if not rows:
        st.info("No rows to display.")
        return

    header_html = "".join(f"<th data-col='{idx}'>{html.escape(h)} ⬍</th>" for idx, h in enumerate(headers))
    body_parts = []
    for row_idx, row in enumerate(rows):
        cells = []
        row_shop = None
        if row_link_values and row_idx < len(row_link_values):
            link_value = str(row_link_values[row_idx] or "").strip()
            if link_value:
                row_shop = link_value
        for col_idx, val in enumerate(row):
            sval = sort_values[row_idx][col_idx]
            style_attr = ""
            if cell_styles:
                style_txt = cell_styles.get((row_idx, col_idx), "")
                if style_txt:
                    style_attr = f" style='{style_txt}'"
            if row_shop:
                cells.append(
                    f"<td data-sort='{html.escape(str(sval))}' data-shop='{html.escape(row_shop)}' class='shop-click-cell' {style_attr}>{html.escape(str(val))}</td>"
                )
            else:
                cells.append(f"<td data-sort='{html.escape(str(sval))}'{style_attr}>{html.escape(str(val))}</td>")
        body_parts.append("<tr>" + "".join(cells) + "</tr>")
    body_html = "".join(body_parts)

    if show_title:
        st.markdown(
            f"<div class='section-title'>{title}</div>",
            unsafe_allow_html=True,
        )

    download_button_html = ""
    if enable_download_hover:
        download_button_html = f"<button id='{table_id}-download' class='table-download-btn' title='Download CSV'>⭳</button>"

    tfoot_html = ""
    if total_row is not None:
        tfoot_cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in total_row)
        tfoot_html = f"<tfoot><tr>{tfoot_cells}</tr></tfoot>"

    table_html = f"""
        <style>
        .table-shell-{table_id} {{
            position: relative;
        }}
        .table-download-btn {{
            position: absolute;
            top: 8px;
            right: 10px;
            z-index: 20;
            border: 1px solid rgba(96, 165, 250, 0.95);
            background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
            color: #ffffff;
            border-radius: 8px;
            font-size: 13px;
            line-height: 1;
            padding: 7px 9px;
            cursor: pointer;
            opacity: 0.92;
            transition: opacity .18s ease;
            box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.28), 0 8px 20px rgba(37, 99, 235, 0.45);
        }}
        .table-shell-{table_id}:hover .table-download-btn {{
            opacity: 1;
        }}
        .table-download-btn:hover {{
            background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
            transform: translateY(-1px);
            box-shadow: 0 0 0 2px rgba(147, 197, 253, 0.40), 0 10px 24px rgba(29, 78, 216, 0.55);
        }}
        #{table_id}-wrap {{
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 14px;
            overflow: auto;
            max-height: 620px;
            background: linear-gradient(135deg, #111a33 0%, #0b1226 100%);
        }}
        #{table_id} {{
            width: 100%;
            border-collapse: collapse;
            min-width: {min_width_px}px;
            font-family: Inter, Segoe UI, sans-serif;
        }}
        #{table_id} th {{
            background: rgba(30, 58, 138, 0.55);
            color: #e2e8f0;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            text-align: center;
            padding: 8px;
            border-bottom: 1px solid rgba(139, 92, 246, 0.35);
            position: sticky;
            top: 0;
            z-index: 2;
            white-space: nowrap;
            cursor: pointer;
            user-select: none;
        }}
        #{table_id} td {{
            color: #e2e8f0;
            font-size: 0.8rem;
            text-align: center;
            padding: 7px;
            border-bottom: 1px solid rgba(71, 85, 105, 0.28);
            white-space: nowrap;
        }}
        #{table_id} tbody tr:hover {{
            background: rgba(99, 102, 241, 0.12);
        }}
        #{table_id} td:nth-child(6) {{
            text-align: center;
            max-width: 420px;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        #{table_id} td a:hover {{
            text-decoration: underline;
        }}
        #{table_id} td.shop-click-cell {{
            cursor: pointer;
        }}
        #{table_id} td.shop-click-cell:hover {{
            background: rgba(59, 130, 246, 0.18);
        }}
        #{table_id} tfoot td {{
            background: rgba(15, 30, 80, 0.95);
            color: #f8fafc;
            font-weight: 800;
            font-size: 0.8rem;
            text-align: center;
            border-top: 2px solid rgba(139, 92, 246, 0.6);
            position: sticky;
            bottom: 0;
            z-index: 1;
            white-space: nowrap;
        }}
        </style>
        <div class="table-shell-{table_id}">
            {download_button_html}
            <div id="{table_id}-wrap">
                <table id="{table_id}">
                    <thead><tr>{header_html}</tr></thead>
                    <tbody>{body_html}</tbody>
                    {tfoot_html}
                </table>
            </div>
        </div>
        <script>
        (() => {{
            const table = document.getElementById('{table_id}');
            if (!table) return;
            const headers = table.querySelectorAll('thead th');
            const tbody = table.querySelector('tbody');
            let sortState = {{ col: -1, asc: true }};

            const parseVal = (raw) => {{
                const cleaned = String(raw || '').replace(/,/g, '').trim();
                const n = Number(cleaned);
                return Number.isNaN(n) ? cleaned.toLowerCase() : n;
            }};

            headers.forEach((th, idx) => {{
                th.addEventListener('click', () => {{
                    const rows = Array.from(tbody.querySelectorAll('tr'));
                    const asc = sortState.col === idx ? !sortState.asc : true;
                    rows.sort((a, b) => {{
                        const av = parseVal(a.children[idx].getAttribute('data-sort'));
                        const bv = parseVal(b.children[idx].getAttribute('data-sort'));
                        if (av < bv) return asc ? -1 : 1;
                        if (av > bv) return asc ? 1 : -1;
                        return 0;
                    }});
                    rows.forEach(r => tbody.appendChild(r));
                    sortState = {{ col: idx, asc }};
                }});
            }});

            const clickableCells = table.querySelectorAll('td.shop-click-cell[data-shop]');
            clickableCells.forEach((td) => {{
                td.addEventListener('click', () => {{
                    const shop = String(td.getAttribute('data-shop') || '').trim();
                    if (!shop) return;
                    try {{
                        const baseUrl = new URL(window.parent.location.href);
                        baseUrl.searchParams.set('{row_link_param}', shop);
                        window.parent.location.href = baseUrl.toString();
                    }} catch (e) {{
                        const fallback = `?{row_link_param}=${{encodeURIComponent(shop)}}`;
                        window.parent.location.href = fallback;
                    }}
                }});
            }});

            const downloadBtn = document.getElementById('{table_id}-download');
            if (downloadBtn) {{
                downloadBtn.addEventListener('click', () => {{
                    const link = document.createElement('a');
                    const excelB64 = '{excel_base64 or ""}';
                    if (excelB64) {{
                        link.href = `data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,${{excelB64}}`;
                    }} else {{
                        const csvEscape = (v) => `"${{String(v ?? '').replace(/"/g, '""')}}"`;
                        const headerVals = Array.from(table.querySelectorAll('thead th')).map(th => th.innerText.replace('⬍', '').trim());
                        const bodyVals = Array.from(table.querySelectorAll('tbody tr')).map(tr =>
                            Array.from(tr.querySelectorAll('td')).map(td => td.innerText.trim())
                        );
                        const csvLines = [headerVals, ...bodyVals].map(row => row.map(csvEscape).join(','));
                        const csv = csvLines.join('\\n');
                        const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
                        link.href = URL.createObjectURL(blob);
                    }}
                    link.download = '{download_file_name}';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    if (!excelB64) URL.revokeObjectURL(link.href);
                }});
            }}
        }})();
        </script>
        """
    extra = 40 if total_row is not None else 0
    viewport_height = min(640, max(240, 86 + len(rows) * 30 + extra))
    components.html(table_html, height=viewport_height, scrolling=False)


def render_detail_table(
    rows: list[dict],
    shop_meta_map: dict[str, dict[str, str]],
    export_rows: list[dict] | None = None,
) -> None:
    headers = [
        "Date",
        "Shop Manager",
        "Shop Name",
        "Vehicle No",
        "Item Code",
        "Item Name",
        "WH Loaded Qty",
        "WH Loaded Value",
        "Shop Receiving Qty",
        "Shop Receiving Value",
        "Final Difference QTY",
        "Final Difference Value",
    ]

    def num_fmt(v, decimals=2):
        try:
            return f"{float(v):,.{decimals}f}"
        except Exception:
            return "0.00" if decimals == 2 else "0"

    table_rows = []
    sort_rows = []

    def diff_cell_style(raw_value: float) -> str:
        try:
            num = float(raw_value)
        except Exception:
            num = 0.0

        if num == 0:
            return "background: rgba(74, 222, 128, 0.20); color: #dcfce7; font-weight: 700;"

        cap = 100000000.0
        ratio = min(abs(num), cap) / cap
        alpha = 0.12 + (0.36 * ratio)
        return f"background: rgba(248, 113, 113, {alpha:.3f}); color: #fecaca; font-weight: 700;"

    def _interpolate_hex(c1: str, c2: str, t: float) -> str:
        t = max(0.0, min(1.0, t))
        r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
        r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"{r:02X}{g:02X}{b:02X}"

    def _excel_diff_fill(raw_value: float):
        if not OPENPYXL_AVAILABLE:
            return None
        try:
            num = float(raw_value)
        except Exception:
            num = 0.0

        if num == 0:
            color_hex = "DCFCE7"
        else:
            ratio = min(abs(num), 100000000.0) / 100000000.0
            color_hex = _interpolate_hex("FEE2E2", "B91C1C", ratio)
        return PatternFill(fill_type="solid", start_color=f"FF{color_hex}", end_color=f"FF{color_hex}")

    def build_detail_excel_bytes(table_data: list[list[str]], original_rows: list[dict]) -> bytes:
        if not OPENPYXL_AVAILABLE:
            return b""
        wb = Workbook()

        categories: dict[str, list[tuple[list[str], dict]]] = {
            "Excess Received": [],
            "Short Received": [],
            "Balanced": [],
        }

        for idx, row_vals in enumerate(table_data):
            src = original_rows[idx] if idx < len(original_rows) else {}
            diff_qty_raw = float(src.get("diff_qty") or 0)
            diff_val_raw = float(src.get("diff_val") or 0)
            if diff_qty_raw > 0 or (diff_qty_raw == 0 and diff_val_raw > 0):
                categories["Excess Received"].append((row_vals, src))
            elif diff_qty_raw < 0 or (diff_qty_raw == 0 and diff_val_raw < 0):
                categories["Short Received"].append((row_vals, src))
            else:
                categories["Balanced"].append((row_vals, src))

        def _write_sheet(ws, grouped_rows: list[tuple[list[str], dict]]):
            qty_number_fmt = "#,##0"
            val_number_fmt = "#,##0.00"

            for col_idx, col_name in enumerate(headers, start=1):
                cell = ws.cell(row=1, column=col_idx, value=col_name)
                cell.font = Font(bold=True, color="FFE2E8F0")
                cell.fill = PatternFill(fill_type="solid", start_color="FF1E3A8A", end_color="FF1E3A8A")
                cell.alignment = Alignment(horizontal="center", vertical="center")

            # Add per-sheet totals row directly under the header.
            total_qty_loaded = sum(float(src.get("qty_loaded") or 0) for _, src in grouped_rows)
            total_value_loaded = sum(float(src.get("value_loaded") or 0) for _, src in grouped_rows)
            total_qty_offloaded = sum(float(src.get("qty_offloaded") or 0) for _, src in grouped_rows)
            total_value_offloaded = sum(float(src.get("value_offloaded") or 0) for _, src in grouped_rows)
            total_diff_qty = sum(float(src.get("diff_qty") or 0) for _, src in grouped_rows)
            total_diff_val = sum(float(src.get("diff_val") or 0) for _, src in grouped_rows)

            totals_row = [
                "TOTAL", "", "", "", "", "",
                total_qty_loaded,
                total_value_loaded,
                total_qty_offloaded,
                total_value_offloaded,
                total_diff_qty,
                total_diff_val,
            ]

            for col_idx, val in enumerate(totals_row, start=1):
                c = ws.cell(row=2, column=col_idx, value=val)
                c.font = Font(bold=True)
                c.alignment = Alignment(horizontal="center", vertical="center")
                c.fill = PatternFill(fill_type="solid", start_color="FFE5E7EB", end_color="FFE5E7EB")

            ws.cell(row=2, column=7).number_format = qty_number_fmt
            ws.cell(row=2, column=8).number_format = val_number_fmt
            ws.cell(row=2, column=9).number_format = qty_number_fmt
            ws.cell(row=2, column=10).number_format = val_number_fmt
            ws.cell(row=2, column=11).number_format = qty_number_fmt
            ws.cell(row=2, column=12).number_format = val_number_fmt

            ws.cell(row=2, column=11).fill = _excel_diff_fill(total_diff_qty)
            ws.cell(row=2, column=12).fill = _excel_diff_fill(total_diff_val)

            for row_num, (row_vals, src) in enumerate(grouped_rows, start=3):
                numeric_values = {
                    7: float(src.get("qty_loaded") or 0),
                    8: float(src.get("value_loaded") or 0),
                    9: float(src.get("qty_offloaded") or 0),
                    10: float(src.get("value_offloaded") or 0),
                    11: float(src.get("diff_qty") or 0),
                    12: float(src.get("diff_val") or 0),
                }
                for col_idx, val in enumerate(row_vals, start=1):
                    if col_idx in numeric_values:
                        val = numeric_values[col_idx]
                    c = ws.cell(row=row_num, column=col_idx, value=val)
                    c.alignment = Alignment(horizontal="center", vertical="center")
                    if col_idx in (7, 9, 11):
                        c.number_format = qty_number_fmt
                    elif col_idx in (8, 10, 12):
                        c.number_format = val_number_fmt

                qty_fill = _excel_diff_fill(float(src.get("diff_qty") or 0))
                val_fill = _excel_diff_fill(float(src.get("diff_val") or 0))
                ws.cell(row=row_num, column=11).fill = qty_fill
                ws.cell(row=row_num, column=12).fill = val_fill
                ws.cell(row=row_num, column=11).font = Font(bold=True)
                ws.cell(row=row_num, column=12).font = Font(bold=True)

            widths = [14, 28, 24, 14, 12, 42, 14, 16, 18, 18, 12, 14]
            for idx, w in enumerate(widths, start=1):
                ws.column_dimensions[chr(64 + idx)].width = w

        first = True
        for sheet_name in ["Excess Received", "Short Received", "Balanced"]:
            if first:
                ws = wb.active
                ws.title = sheet_name
                first = False
            else:
                ws = wb.create_sheet(title=sheet_name)
            _write_sheet(ws, categories[sheet_name])

        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()

    cell_styles: dict[tuple[int, int], str] = {}

    for r in rows:
        shop_code = str(r.get("shop_code", "") or "").strip().upper()
        shop_meta = shop_meta_map.get(shop_code, {})
        manager_name = str(shop_meta.get("manager_name", "") or "").strip()
        shop_name = str(shop_meta.get("shop_name", "") or "").strip()
        date_text = r.get("date").strftime("%d %b %Y") if r.get("date") else ""
        diff_qty_raw = float(r.get("diff_qty") or 0)
        diff_val_raw = float(r.get("diff_val") or 0)
        row_display = [
            date_text,
            manager_name,
            shop_name,
            str(r.get("vehicle_no", "") or ""),
            str(r.get("item_code", "") or ""),
            str(r.get("item_name", "") or ""),
            num_fmt(r.get("qty_loaded", 0), 0),
            num_fmt(r.get("value_loaded", 0), 2),
            num_fmt(r.get("qty_offloaded", 0), 0),
            num_fmt(r.get("value_offloaded", 0), 2),
            num_fmt(diff_qty_raw, 0),
            num_fmt(diff_val_raw, 2),
        ]
        row_sort = [
            str(r.get("date") or ""),
            manager_name,
            shop_name,
            str(r.get("vehicle_no", "") or ""),
            str(r.get("item_code", "") or ""),
            str(r.get("item_name", "") or ""),
            str(float(r.get("qty_loaded") or 0)),
            str(float(r.get("value_loaded") or 0)),
            str(float(r.get("qty_offloaded") or 0)),
            str(float(r.get("value_offloaded") or 0)),
            str(diff_qty_raw),
            str(diff_val_raw),
        ]
        row_idx = len(table_rows)
        table_rows.append(row_display)
        sort_rows.append(row_sort)
        cell_styles[(row_idx, 10)] = diff_cell_style(diff_qty_raw)
        cell_styles[(row_idx, 11)] = diff_cell_style(diff_val_raw)

    detail_total_row = [
        "TOTAL", "", "", "", "", "",
        num_fmt(sum(float(r.get("qty_loaded") or 0) for r in rows), 0),
        num_fmt(sum(float(r.get("value_loaded") or 0) for r in rows), 2),
        num_fmt(sum(float(r.get("qty_offloaded") or 0) for r in rows), 0),
        num_fmt(sum(float(r.get("value_offloaded") or 0) for r in rows), 2),
        num_fmt(sum(float(r.get("diff_qty") or 0) for r in rows), 0),
        num_fmt(sum(float(r.get("diff_val") or 0) for r in rows), 2),
    ]

    excel_b64 = None
    download_name = "offloading_vs_loading_detail.csv"
    if OPENPYXL_AVAILABLE:
        export_source_rows = export_rows if export_rows is not None else rows
        export_table_rows = []
        for r in export_source_rows:
            shop_code = str(r.get("shop_code", "") or "").strip().upper()
            shop_meta = shop_meta_map.get(shop_code, {})
            manager_name = str(shop_meta.get("manager_name", "") or "").strip()
            shop_name = str(shop_meta.get("shop_name", "") or "").strip()
            date_text = r.get("date").strftime("%d %b %Y") if r.get("date") else ""
            diff_qty_raw = float(r.get("diff_qty") or 0)
            diff_val_raw = float(r.get("diff_val") or 0)
            export_table_rows.append([
                date_text,
                manager_name,
                shop_name,
                str(r.get("vehicle_no", "") or ""),
                str(r.get("item_code", "") or ""),
                str(r.get("item_name", "") or ""),
                num_fmt(r.get("qty_loaded", 0), 0),
                num_fmt(r.get("value_loaded", 0), 2),
                num_fmt(r.get("qty_offloaded", 0), 0),
                num_fmt(r.get("value_offloaded", 0), 2),
                num_fmt(diff_qty_raw, 0),
                num_fmt(diff_val_raw, 2),
            ])

        excel_bytes = build_detail_excel_bytes(export_table_rows, export_source_rows)
        excel_b64 = base64.b64encode(excel_bytes).decode("utf-8") if excel_bytes else None
        if excel_b64:
            download_name = "offloading_vs_loading_detail.xlsx"

    render_sortable_html_table(
        title="Loading vs Offloading (Detail)",
        headers=headers,
        rows=table_rows,
        sort_values=sort_rows,
        table_id="ovl-detail-table",
        min_width_px=1120,
        show_title=False,
        cell_styles=cell_styles,
        enable_download_hover=True,
        download_file_name=download_name,
        excel_base64=excel_b64,
        total_row=detail_total_row,
    )


def render_shop_summary_table(rows: list[dict], shop_meta_map: dict[str, dict[str, str]]) -> None:
    if not rows:
        st.info("No rows to display.")
        return

    def num_fmt(v, decimals=2):
        try:
            return f"{float(v):,.{decimals}f}"
        except Exception:
            return "0.00" if decimals == 2 else "0"

    table_rows = []
    sort_rows = []
    for r in rows:
        shop_code = str(r.get("shop_code", "") or "").strip().upper()
        shop_meta = shop_meta_map.get(shop_code, {})
        manager_name = str(shop_meta.get("manager_name", "") or "").strip()
        shop_name = str(shop_meta.get("shop_name", "") or "").strip()
        short_qty = float(r.get("short_qty") or 0)
        short_val = float(r.get("short_val") or 0)
        excess_qty = float(r.get("excess_qty") or 0)
        excess_val = float(r.get("excess_val") or 0)
        diff_qty = float(r.get("diff_qty") or 0)
        diff_val = float(r.get("diff_val") or 0)
        table_rows.append([
            manager_name,
            shop_name,
            num_fmt(short_qty, 0),
            num_fmt(short_val, 2),
            num_fmt(excess_qty, 0),
            num_fmt(excess_val, 2),
            num_fmt(diff_qty, 0),
            num_fmt(diff_val, 2),
        ])
        sort_rows.append([
            manager_name,
            shop_name,
            str(short_qty),
            str(short_val),
            str(excess_qty),
            str(excess_val),
            str(diff_qty),
            str(diff_val),
        ])

    shop_total_row = [
        "TOTAL",
        "",
        num_fmt(sum(float(r.get("short_qty") or 0) for r in rows), 0),
        num_fmt(sum(float(r.get("short_val") or 0) for r in rows), 2),
        num_fmt(sum(float(r.get("excess_qty") or 0) for r in rows), 0),
        num_fmt(sum(float(r.get("excess_val") or 0) for r in rows), 2),
        num_fmt(sum(float(r.get("diff_qty") or 0) for r in rows), 0),
        num_fmt(sum(float(r.get("diff_val") or 0) for r in rows), 2),
    ]

    render_sortable_html_table(
        title="Exception Shopwise Difference",
        headers=["Shop Manager", "Shop Name", "SHOP RECVD SHORT QTY", "SHOP RECVD SHORT VALUE", "SHOP RECVD EXCESS QTY", "SHOP RECVD EXCESS VALUE", "Final Difference QTY", "Final Difference Value"],
        rows=table_rows,
        sort_values=sort_rows,
        table_id="ovl-shop-table",
        min_width_px=1220,
        total_row=shop_total_row,
    )


def render_detail_filter_header(
    available_shop_codes: list[str],
    shop_meta_map: dict[str, dict[str, str]],
    detail_rows: list[dict],
    clicked_shop: str | None = None,
) -> tuple[str, str, list[str]]:
    pending_action = st.session_state.pop("detail_item_filter_action", None)
    if pending_action == "clear":
        # Apply search reset before widget instantiation to avoid Streamlit state mutation errors.
        st.session_state["detail_item_search"] = ""

    options = ["All"] + sorted(available_shop_codes)
    preferred = str(clicked_shop or st.session_state.get("detail_shop_filter", "All") or "All").strip().upper()
    if preferred not in options:
        preferred = "All"

    receive_options = ["All", "Short Received", "Excess Received"]
    receive_preferred = str(st.session_state.get("detail_receive_filter", "All") or "All")
    if receive_preferred not in receive_options:
        receive_preferred = "All"

    title_col, search_col, receive_col, filter_col = st.columns([0.44, 0.24, 0.14, 0.18], gap="small")
    with search_col:
        item_search = st.text_input(
            "Search Item Name",
            key="detail_item_search",
            placeholder="Type item name e.g. indomie",
            label_visibility="collapsed",
        ).strip()
    with receive_col:
        selected_receive = st.selectbox(
            "Receive Filter",
            options=receive_options,
            index=receive_options.index(receive_preferred),
            key="detail_receive_filter",
            label_visibility="collapsed",
        )
    with filter_col:
        selected = st.selectbox(
            "Detail Shop",
            options=options,
            index=options.index(preferred),
            key="detail_shop_filter",
            label_visibility="collapsed",
        )

    title_text = "Loading vs Offloading (Detail)"
    if selected != "All":
        meta = shop_meta_map.get(selected, {})
        shop_name = str(meta.get("shop_name", "") or "").strip()
        manager_name = str(meta.get("manager_name", "") or "").strip()
        if shop_name:
            title_text = f"Loading vs Offloading (Detail) — {selected} ({shop_name})"
        else:
            title_text = f"Loading vs Offloading (Detail) — {selected}"
        if manager_name:
            title_text = f"{title_text} | Manager: {manager_name}"
    with title_col:
        st.markdown(f"<div class='section-title'>{html.escape(title_text)}</div>", unsafe_allow_html=True)

    scoped_rows = detail_rows if selected == "All" else [
        row for row in detail_rows
        if str(row.get("shop_code", "") or "").strip().upper() == selected
    ]

    if selected_receive == "Short Received":
        scoped_rows = [row for row in scoped_rows if float(row.get("diff_val") or 0) < 0]
    elif selected_receive == "Excess Received":
        scoped_rows = [row for row in scoped_rows if float(row.get("diff_val") or 0) > 0]

    all_item_names = sorted({
        str(row.get("item_name", "") or "").strip()
        for row in scoped_rows
        if str(row.get("item_name", "") or "").strip()
    })
    if item_search:
        matching_item_names = [name for name in all_item_names if item_search.lower() in name.lower()]
    else:
        matching_item_names = all_item_names

    if pending_action == "select_all":
        st.session_state["detail_item_filter"] = matching_item_names
    elif pending_action == "clear":
        st.session_state["detail_item_filter"] = []

    existing_selection = st.session_state.get("detail_item_filter", []) or []
    valid_selection = [name for name in existing_selection if name in matching_item_names]
    if valid_selection != existing_selection:
        st.session_state["detail_item_filter"] = valid_selection

    selected_items: list[str] = []
    if item_search or valid_selection:
        picker_col, select_all_col, clear_col = st.columns([0.76, 0.12, 0.12], gap="small")
        with picker_col:
            selected_items = st.multiselect(
                "Matching Item Names",
                options=matching_item_names,
                key="detail_item_filter",
                placeholder="Select one or more matching item names",
                help="Type part of an item name above to narrow the list, then select one or more matches.",
            )
        with select_all_col:
            st.markdown('<div style="margin-top: 1.75rem;"></div>', unsafe_allow_html=True)
            if st.button("Select All", key="detail_item_filter_all", use_container_width=True, disabled=not matching_item_names):
                st.session_state["detail_item_filter_action"] = "select_all"
                st.rerun()
        with clear_col:
            st.markdown('<div style="margin-top: 1.75rem;"></div>', unsafe_allow_html=True)
            if st.button("Clear", key="detail_item_filter_clear", use_container_width=True, disabled=not (item_search or selected_items)):
                st.session_state["detail_item_filter_action"] = "clear"
                st.rerun()

        if item_search:
            st.caption(f"{len(matching_item_names):,} item name(s) matched '{item_search}'.")

    return selected, selected_receive, selected_items


def render_header_with_date_filter(min_date, max_date):
    def _coerce_date(val):
        if isinstance(val, date):
            return val
        if isinstance(val, datetime):
            return val.date()
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val).date()
            except Exception:
                return None
        return None

    def _clamp_date(val, lo, hi):
        if val is None:
            return None
        if lo is not None and val < lo:
            return lo
        if hi is not None and val > hi:
            return hi
        return val

    yesterday = (datetime.today() - timedelta(days=1)).date()
    if max_date and max_date <= yesterday:
        default_end = max_date
    else:
        default_end = yesterday
    default_end = _clamp_date(default_end, min_date, max_date)
    default_start = min_date
    default_start = _clamp_date(default_start, min_date, max_date)

    if default_start is None and default_end is not None:
        default_start = default_end
    if default_end is None and default_start is not None:
        default_end = default_start

    saved_raw = st.session_state.get("offload_date_range", (default_start, default_end))
    if isinstance(saved_raw, (tuple, list)):
        parsed = [_coerce_date(v) for v in saved_raw]
        parsed = [p for p in parsed if p is not None]
        if len(parsed) >= 2:
            saved = (parsed[0], parsed[1])
        elif len(parsed) == 1:
            saved = (parsed[0], parsed[0])
        else:
            saved = (default_start, default_end)
    else:
        parsed_one = _coerce_date(saved_raw)
        saved = (parsed_one, parsed_one) if parsed_one else (default_start, default_end)

    saved = (
        _clamp_date(saved[0], min_date, max_date),
        _clamp_date(saved[1], min_date, max_date),
    )
    if saved[0] is None and saved[1] is not None:
        saved = (saved[1], saved[1])
    elif saved[1] is None and saved[0] is not None:
        saved = (saved[0], saved[0])
    elif saved[0] is None and saved[1] is None:
        saved = (default_start, default_end)

    if saved[0] > saved[1]:
        saved = (saved[1], saved[0])

    is_master = st.session_state.get("ovl_is_master_user", False)
    if is_master and WMS_SYNC_AVAILABLE:
        hdr_col, date_col, update_col, refresh_col, logout_col = st.columns([0.48, 0.20, 0.13, 0.08, 0.08], gap="small")
    else:
        update_col = None
        hdr_col, date_col, refresh_col, logout_col = st.columns([0.62, 0.20, 0.09, 0.09])

    with hdr_col:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:12px;padding:4px 0 8px 0;">
                <img src="https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg"
                     style="height:34px;border-radius:50%;border:2px solid #3b82f6;">
                <div>
                    <div class="dashboard-title">Loading vs Offloading Dashboard</div>
                    <div class="dashboard-sub">WH Database · Tabular reconciliation view</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with date_col:
        st.markdown(
            '<div style="text-align:right;font-size:0.72rem;color:#6ee7b7;font-weight:600;margin-bottom:2px;">Date Range</div>',
            unsafe_allow_html=True,
        )
        selection = st.date_input(
            "offload_date_pick",
            value=(saved[0], saved[1]),
            min_value=min_date,
            max_value=max_date,
            key="offload_date_pick",
            label_visibility="collapsed",
        )

    today     = datetime.today().date()
    yesterday = today - timedelta(days=1)

    # ── Monday: also upload Saturday (yesterday-1) ───────────────────────────
    # yesterday = Sunday on Monday; Saturday is two days ago
    if today.weekday() == 0:  # 0 = Monday
        saturday = yesterday - timedelta(days=1)
        _dates_to_upload = [saturday, yesterday]   # Saturday then Sunday
    else:
        _dates_to_upload = [yesterday]

    # ── Show any pending email notification that survived st.rerun() ──────────
    _pending = st.session_state.pop("_lvo_email_notify", None)
    if _pending:
        _kind, _msg = _pending
        if _kind in ("toast", "success"):
            st.toast(_msg, icon="✅")   # auto-fades (~10 s)
        else:
            st.warning(_msg)

    if update_col is not None:
        with update_col:
            st.markdown('<div style="margin-top:18px;"></div>', unsafe_allow_html=True)
            already_loaded = all(_wms_sync.date_exists_in_db(d) for d in _dates_to_upload)
            if today.weekday() == 0:
                _date_label_btn = f"Sat {_dates_to_upload[0].strftime('%d %b')} + Sun {_dates_to_upload[1].strftime('%d %b')}"
            else:
                _date_label_btn = yesterday.strftime('%d %b %Y')
            btn_label = "✅ Updated" if already_loaded else "⬆ Update Data"
            btn_help  = (
                f"Data for {_date_label_btn} is already loaded. Click to force re-import."
                if already_loaded else
                f"Load {_date_label_btn} data from network share and send email"
            )
            if st.button(btn_label, key="wms_update_btn", use_container_width=True, help=btn_help):
                st.session_state["wms_sync_trigger"] = True
                st.session_state["wms_sync_force"] = already_loaded
                st.rerun()

        # ── File-missing banner (shown below header, full width) ──────────────
        if not already_loaded:
            _file_status = _wms_check_file_status(yesterday)
            if _file_status == "network_error":
                st.markdown(
                    f"""
                    <div style="background:rgba(248,81,73,.12);border:1px solid rgba(248,81,73,.4);
                                border-radius:10px;padding:10px 16px;margin:4px 0 8px 0;
                                display:flex;align-items:center;gap:10px;">
                      <span style="font-size:18px;">🔌</span>
                      <span style="color:#f85149;font-weight:600;font-size:13px;">
                        Network share <code>\\\\10.10.0.30\\mis</code> is not accessible.
                        Check VPN / network connectivity.
                      </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif _file_status == "missing":
                _expected = f"WMS_shopdiff_{yesterday.strftime('%d_%m_%Y')}.csv"
                st.markdown(
                    f"""
                    <div style="background:rgba(255,193,7,.10);border:1px solid rgba(255,193,7,.4);
                                border-radius:10px;padding:10px 16px;margin:4px 0 8px 0;
                                display:flex;align-items:center;gap:10px;">
                      <span style="font-size:18px;">⚠️</span>
                      <span style="color:#ffc107;font-weight:600;font-size:13px;">
                        File not yet available on network share —
                        <code>{_expected}</code> not found in
                        <code>\\\\10.10.0.30\\mis</code>.
                        Dashboard data may be incomplete for {yesterday.strftime('%d %b %Y')}.
                      </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with refresh_col:
        st.markdown('<div style="margin-top:18px;"></div>', unsafe_allow_html=True)
        if st.button("🔄", key="offload_refresh", help="Clear cache and reload"):
            st.cache_data.clear()
            st.rerun()

    with logout_col:
        st.markdown('<div style="margin-top:18px;"></div>', unsafe_allow_html=True)
        if st.button("Logout", key="header_logout_btn", use_container_width=True, help="Sign out"):
            st.session_state["ovl_authenticated"] = False
            st.session_state["ovl_user"] = None
            st.session_state["ovl_is_master_user"] = False
            st.session_state["ovl_allowed_shops"] = []
            st.rerun()

    # ── CTA: Update Data — same pipeline as home_dashboard LVO_offloading_vs_loading ──
    if st.session_state.pop("wms_sync_trigger", False) and WMS_SYNC_AVAILABLE:
        force = st.session_state.pop("wms_sync_force", False)

        if not force and all(_wms_sync.date_exists_in_db(d) for d in _dates_to_upload):
            st.info(
                f"Data for {_date_label_btn} is already loaded. "
                "Click **⬆ Update Data** again to force re-import."
            )
        else:
            _status_title = (
                f"Updating data for {_date_label_btn}…"
            )
            with st.status(_status_title, expanded=True) as _sync_status:
                try:
                    _total_rows   = 0
                    _loaded_dates = []
                    _last_csv_path = None

                    _col_sql      = ", ".join(f'"{c}"' for c in _LVO_COLUMNS)
                    _null         = lambda v: None if pd.isnull(v) else v
                    _hist_col_sql = ", ".join(
                        f'"{c}"' for c in _LVO_COLUMNS + ["history_action", "history_at", "source_file"]
                    )

                    # ── Ensure table structure once ───────────────────────────
                    _conn_init = psycopg2.connect(**DB_CONFIG)
                    try:
                        _wms_sync._ensure_loadingvsoffloading_tables(_conn_init)
                        _conn_init.commit()
                    finally:
                        _conn_init.close()

                    # ── Loop over each date to upload ─────────────────────────
                    for _target_date in _dates_to_upload:
                        _d_label = _target_date.strftime('%d %b %Y')

                        # Step 1: find CSV
                        _sync_status.write(
                            f"🔍 [{_d_label}] Searching for "
                            f"`WMS_shopdiff_{_target_date.strftime('%d_%m_%Y')}.csv`…"
                        )
                        try:
                            _csv_path = _wms_sync.find_wms_file(_target_date)
                        except OSError as _e:
                            _sync_status.update(label="❌ Network share not accessible", state="error")
                            st.error(str(_e))
                            raise

                        if not _csv_path:
                            _sync_status.write(
                                f"⚠️ [{_d_label}] File not found — skipping this date."
                            )
                            continue

                        _sync_status.write(f"📄 [{_d_label}] Found: `{os.path.basename(_csv_path)}`")
                        _last_csv_path = _csv_path

                        # Step 2: read CSV
                        _sync_status.write(f"📊 [{_d_label}] Reading CSV…")
                        _df_raw = pd.read_csv(_csv_path, encoding="utf-8-sig", dtype=str)

                        # Step 3: normalise
                        _df_norm = _wms_sync._prepare_lvo_offloading_df(
                            _df_raw, os.path.basename(_csv_path)
                        )
                        _df_norm = _wms_sync._rename_by_expected_columns(_df_norm, _LVO_COLUMNS)
                        _missing = [c for c in _LVO_COLUMNS if c not in _df_norm.columns]
                        if _missing:
                            raise ValueError(f"Missing columns after normalisation: {', '.join(_missing)}")
                        _df_up = _df_norm[_LVO_COLUMNS].copy()
                        _df_up["date"] = pd.to_datetime(_df_up["date"], errors="coerce").dt.date
                        for _nc in ["qty_loaded", "value_loaded", "qty_offloaded",
                                    "value_offloaded", "diff_qty", "diff_val"]:
                            _df_up[_nc] = pd.to_numeric(_df_up[_nc], errors="coerce").fillna(0)
                        _df_up = _df_up[_df_up["date"] == _target_date].copy()
                        if _df_up.empty:
                            _sync_status.write(
                                f"⚠️ [{_d_label}] No rows found after normalisation — skipping."
                            )
                            continue
                        _sync_status.write(
                            f"✅ [{_d_label}] Normalised **{len(_df_up):,} rows**"
                        )

                        # Step 4: DB operations
                        _conn = psycopg2.connect(**DB_CONFIG)
                        _conn.autocommit = False
                        try:
                            with _conn.cursor() as _cur:
                                _cur.execute(
                                    "DELETE FROM public.offloading_vs_loading WHERE date = %s",
                                    (_target_date,),
                                )
                                _deleted = _cur.rowcount
                            if _deleted:
                                _sync_status.write(
                                    f"🗑️ [{_d_label}] Removed **{_deleted}** stale rows"
                                )

                            _values = [
                                tuple(_null(row.get(c)) for c in _LVO_COLUMNS)
                                for _, row in _df_up.iterrows()
                            ]
                            with _conn.cursor() as _cur:
                                execute_values(
                                    _cur,
                                    f"INSERT INTO public.offloading_vs_loading ({_col_sql}) VALUES %s",
                                    _values,
                                    page_size=5000,
                                )
                                _cur.execute(
                                    "UPDATE public.offloading_vs_loading "
                                    "SET diff = diff_qty WHERE diff IS DISTINCT FROM diff_qty"
                                )

                            _now_ts = datetime.now()
                            _hist_values = [
                                tuple(_null(row.get(c)) for c in _LVO_COLUMNS)
                                + ("auto_sync", _now_ts, os.path.basename(_csv_path))
                                for _, row in _df_up.iterrows()
                            ]
                            with _conn.cursor() as _cur:
                                execute_values(
                                    _cur,
                                    f"INSERT INTO public.offloading_vs_loading_history ({_hist_col_sql}) VALUES %s",
                                    _hist_values,
                                    page_size=5000,
                                )
                                _cur.execute(
                                    "UPDATE public.offloading_vs_loading_history "
                                    "SET diff = diff_qty "
                                    "WHERE history_action = 'auto_sync' "
                                    "AND history_at >= NOW() - INTERVAL '10 minutes'"
                                )

                            _conn.commit()
                            _sync_status.write(
                                f"✅ [{_d_label}] Committed **{len(_values):,} rows**"
                            )
                            _total_rows += len(_values)
                            _loaded_dates.append(_target_date)

                        except Exception:
                            _conn.rollback()
                            raise
                        finally:
                            _conn.close()

                    # ── Summary label ─────────────────────────────────────────
                    _dates_summary = " + ".join(d.strftime('%d %b %Y') for d in _loaded_dates)
                    _sync_status.update(
                        label=f"✅ {_total_rows:,} rows loaded for {_dates_summary or _date_label_btn}",
                        state="complete",
                    )

                    # ── Email: standard + newsletter (both with Excel attachment) ──
                    if _loaded_dates:
                        _email_date  = _loaded_dates[-1]
                        _date_start  = min(_dates_to_upload)
                        _sync_status.write("📧 Building Excel report…")
                        try:
                            _excel = _wms_sync.build_excel_bytes(_email_date)
                        except Exception as _xe:
                            _excel = None
                            _sync_status.write(f"⚠️ Excel build error: {_xe}")

                        # Standard operational email (existing flow)
                        _sync_status.write("📧 Sending standard email…")
                        try:
                            _wms_sync.send_email(_email_date, _total_rows, _excel)
                            _sync_status.write("📧 Standard email sent ✓")
                        except Exception as _email_err:
                            _sync_status.write(f"⚠️ Standard email error: {_email_err}")

                        # Newsletter with Excel attachment
                        _sync_status.write("📰 Sending newsletter with attachment…")
                        try:
                            _nl_sum   = load_discrepancy_summary(_date_start, _email_date)
                            _nl_shops = load_shop_summary(_date_start, _email_date)
                            _nl_items = load_top_discrepancy_items(_date_start, _email_date, 20)
                            _nl_vehs  = load_vehicle_summary(_date_start, _email_date)
                            _nl_daily = load_daily_trend(_date_start, _email_date)
                            _nl_meta  = load_shop_meta_map()
                            _nl_ok, _nl_msg = _send_lvo_newsletter(
                                _nl_sum, _nl_shops, _nl_meta,
                                _nl_items, _nl_vehs, _nl_daily,
                                _date_start, _email_date,
                                excel_bytes=_excel,
                            )
                            if _nl_ok:
                                _sync_status.write("📰 Newsletter sent ✓")
                                st.session_state["_lvo_email_notify"] = (
                                    "toast",
                                    f"✅ Data loaded, email & newsletter sent — "
                                    f"{_total_rows:,} rows for {_dates_summary}",
                                )
                            else:
                                _sync_status.write(f"⚠️ Newsletter: {_nl_msg}")
                                st.session_state["_lvo_email_notify"] = (
                                    "warning",
                                    f"Data loaded ({_total_rows:,} rows). Newsletter failed: {_nl_msg}",
                                )
                        except Exception as _nl_err:
                            _sync_status.write(f"⚠️ Newsletter error: {_nl_err}")
                            st.session_state["_lvo_email_notify"] = (
                                "warning",
                                f"Data loaded ({_total_rows:,} rows). Newsletter error: {_nl_err}",
                            )

                        st.cache_data.clear()
                        st.rerun()

                except Exception as _err:
                    _sync_status.update(label="❌ Upload failed", state="error")
                    st.error(f"❌ {_err}")

    if isinstance(selection, tuple) and len(selection) == 2:
        d_start = _coerce_date(selection[0])
        d_end = _coerce_date(selection[1])
        if d_start is None and d_end is None:
            d_start, d_end = default_start, default_end
        elif d_start is None:
            d_start = d_end
        elif d_end is None:
            d_end = d_start
    elif isinstance(selection, (tuple, list)) and len(selection) == 1:
        only_date = _coerce_date(selection[0]) or default_end
        d_start = d_end = only_date
    else:
        single = _coerce_date(selection) or default_end
        d_start = d_end = single

    if d_start > d_end:
        d_start, d_end = d_end, d_start

    d_start = _clamp_date(d_start, min_date, max_date)
    d_end = _clamp_date(d_end, min_date, max_date)

    st.session_state["offload_date_range"] = (d_start, d_end)
    return d_start, d_end


def main() -> None:
    inject_css()
    init_auth_state()

    if not st.session_state.get("ovl_authenticated", False):
        render_login_gate()
        return

    if render_logged_in_sidebar():
        return

    user = st.session_state.get("ovl_user") or {}
    _ta = str((user or {}).get("table_access", "") or "")
    _is_super = "all" in {x.strip().lower() for x in _ta.split(",") if x.strip()}
    if _is_super:
        is_master_user = True
        allowed_shops: list[str] = []
    else:
        is_master_user, allowed_shops = get_user_shop_scope(
            str(user.get("employee_id", "") or ""),
            str(user.get("full_name", "") or ""),
        )
    st.session_state["ovl_is_master_user"] = is_master_user
    st.session_state["ovl_allowed_shops"] = allowed_shops

    if not _is_super and not is_master_user and not allowed_shops:
        st.error("No shop mapping found in latest century_penetration.opsmgr upload for this user.")
        return

    latest_upload_date = get_latest_opsmgr_upload_date()

    min_date, max_date = load_date_bounds()
    if min_date is None or max_date is None:
        live_count, live_min, live_max = get_live_date_bounds_uncached()
        if live_count > 0 and live_min is not None and live_max is not None:
            # One-time self-heal for stale Streamlit cache after uploads/restores.
            if not st.session_state.get("offload_bounds_cache_refreshed", False):
                st.session_state["offload_bounds_cache_refreshed"] = True
                st.cache_data.clear()
                st.rerun()
            min_date, max_date = live_min, live_max
        else:
            st.error("No data found in `offloading_vs_loading`. Run setup script first.")
            return
    else:
        st.session_state["offload_bounds_cache_refreshed"] = False

    start_date, end_date = render_header_with_date_filter(min_date, max_date)

    data = load_offloading_vs_loading(start_date, end_date)
    data = apply_shop_scope(data, allowed_shops, is_master_user)
    if not data:
        st.info("No records for selected date range and access scope.")
        return

    render_metrics(data, start_date, end_date)
    shop_rows = load_shop_summary(start_date, end_date)
    shop_rows = apply_shop_scope(shop_rows, allowed_shops, is_master_user)
    shop_meta_map = load_shop_meta_map()

    if is_master_user:
        if latest_upload_date:
            st.caption(f"Access Scope: ALL shops (master user) | opsmgr upload_date: {latest_upload_date}")
        else:
            st.caption("Access Scope: ALL shops (master user)")
    else:
        if latest_upload_date:
            st.caption(f"Access Scope: {len(allowed_shops)} mapped shop(s) from latest opsmgr upload_date: {latest_upload_date}")
        else:
            st.caption(f"Access Scope: {len(allowed_shops)} mapped shop(s) from century_penetration.opsmgr")

    # ── EDA Intelligence Center (tables live inside tab_tbl tab) ────────────────
    render_eda_analytics(data, shop_rows, shop_meta_map, start_date, end_date,
                         is_master=is_master_user)


if __name__ == "__main__":
    main()
