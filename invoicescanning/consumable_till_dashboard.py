import streamlit as st
import pandas as pd
import psycopg2
import psycopg2.pool
import html
import base64
import json
import re
from urllib.parse import urlencode, quote
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import calendar
import io
import logging
import os
import socket
import uuid
import getpass
from contextlib import contextmanager
from psycopg2.extras import RealDictCursor, Json

st.set_page_config(
    page_title="Invoice Scanning Dashboard",
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

logger = logging.getLogger(__name__)

CONSUMABLE_TILLS = [
    ("SPN", 34, "Kids Play"),
    ("SPN", 32, "Ice-Cream"),
    ("SPN", 33, "Backery"),
    ("LFS", 23, "Ice-Cream"),
    ("LFS", 26, "Arcadia"),
    ("MSS", 13, "Icream"),
    ("MSS", 14, "Juice Till"),
    ("MM1", 1, "Fresh Juice"),
    ("MM1", 13, "Ice Cream"),
    ("MM2", 10, "Fresh Juice"),
    ("MM2", 9, "Bakery"),
    ("WHL", 14, "Juice Bar"),
]

SHOP_NAME_MAP = {
     "ACH": "ACHIMOTA",
     "AF2": "AFLAO",
     "AFI": "MATAHEKO",
     "AFL": "ASHONGMAN",
     "AKE": "ASIMANKESE",
     "AMA": "AMASAMAN",
     "ASF": "ASSIN FOSU",
     "ASH": "ASHAIMAN",
     "BIB": "BIBIANI",
     "BOL": "BOLGATANGA",
     "BRE": "BEREKUM",
     "CAP": "CAPE COAST",
     "CLC": "WA SHOP",
     "DNS": "DANSOMAN",
     "EL2": "EAST LEGON - SPECIALTY",
     "ELS": "EAST LEGON",
     "FAR": "FAREAHA",
     "GBA": "GBAWE",
     "HAA": "HAATSO",
     "HAM": "HAMPTON SQUARE",
     "HOE": "HOHOE",
     "HOV": "HO VOLTA",
     "KA2": "KASOA",
     "KAS": "KASS",
     "KCS": "MINI-DOME",
     "KF2": "KOFORIDUA",
     "KFH": "KOFORIDUA HOME",
     "KS2": "KUMASI ADIEBEBA",
     "KS3": "KUMASI TANOSO ABUAKHWA",
     "KS4": "KUMASI HENE",
     "KS5": "KUMASI MANHYIA",
     "KS7": "KUMASI SANTASI",
     "KS8": "KUMASI TAFO",
     "KSI": "KUMASI ADUM",
     "KSO": "KUMASI SUAME",
     "KSS": "KISSEMAN SHOP",
     "LCC": "ACCRA OPERA",
     "LFS": "PLUS KANESHIE",
     "M01": "LASHIBI",
     "M07": "COMMUNITY 25",
     "MAS": "ADENTA",
     "MDN": "MADINA",
     "MHT": "HOME - TEMA",
     "MKL": "LAPAZ",
     "M03": "LABONE - MINI",
     "M06": "KASOA MINI",
     "MM1": "ACCRA MALL",
     "MM2": "ACHIMOTA MALL",
     "MM3": "KUMASI MALL",
     "MSS": "EAST LEGON BOUNDARY RD",
     "MUS": "UPSA",
     "NAN": "NANAKROM",
     "NKW": "NKAWKAW",
     "OBS": "BAWKU",
     "OLE": "OLEBU (ABLEKUMA)",
     "SD2": "SWEDRU",
     "SPN": "SPINTEX MALL",
     "SPX": "SPINTEX",
     "SU2": "SUNIYANI",
     "SUN": "LABADI",
     "SWO": "SEFWI WIASO",
     "TC2": "TECHIMAN NEW(GYARKO)",
     "TCH": "TECHIMAN",
     "TKD": "TAKORADI",
     "TKW": "TARKWA",
     "TM2": "TEPA",
     "TMA": "MANKESSIM STORE",
     "TML": "TAMALE",
     "TMP": "PLUS TEMA",
     "TSN": "TESHIE NUNGUA",
     "WHL": "WEIJA SHOP",
     "WNC": "WENCHI",


}

IMPLEMENTED_SHOPS = [
    "ACH", "ADB", "AFI", "AFL", "AMA", "ASH", "EL2", "ELS", "FAR", "GBA", "HAA", "HAM", "KA2",
    "KAS", "KS2", "KS3", "KS4", "KS5", "KS7", "KS8", "KSI", "KSO", "KSS", "LCC", "LFS", "M01", "M03",
    "M05", "M06", "M07", "MAS", "MDN", "MM1", "MM2", "MM3", "MSS", "NAN", "OLE", "SPN", "TMP", "WHL",
    "KCS",
]

ALL_SHOPS = [
 "ACH", "AF2", "AFI", "AFL", "AKE", "AMA", "ASF", "ASH", "BIB",
 "BOL", "BRE", "CAP", "CLC", "DNS", "EL2", "ELS", "FAR", "GBA",
 "HAA", "HAM", "HOE", "HOV", "KA2", "KAS", "KCS", "KF2", "KFH",
 "KS2", "KS3", "KS4", "KS5", "KS7", "KS8", "KSI", "KSO", "KSS",
 "LCC", "LFS", "M01", "M07", "MAS", "MDN", "MHT", "MKL", "MM1",
 "MM2", "MM3", "MSS", "MUS", "NAN", "NKW", "OBS", "OLE", "SD2",
 "SPN", "SPX", "SU2", "SUN", "SWO", "TC2", "TCH", "TKD", "TKW",
 "TM2", "TMA", "TML", "TMP", "TSN", "WHL", "WNC",

]

IMPLEMENTED_SHOPS_SET = {shop.strip().upper() for shop in IMPLEMENTED_SHOPS}
TOTAL_SHOPS_SET = {shop.strip().upper() for shop in ALL_SHOPS}
IMPLEMENTATION_TOTAL_SHOPS = 70

ALERT_ERROR_TYPE_ORDER = [
    "Test Bill",
    "Bill Date Mismatch",
    "Duplicate",
    "Wrong Shop",
    "High Bill Amount",
]


def inject_css():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

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
.dashboard-title {
    background: linear-gradient(135deg, #1e3a8a 0%, #5b54ff 50%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
    text-align: center;
}
.dashboard-subtitle {
    color: #cbd5e1;
    font-size: 0.9rem;
    margin-bottom: 1rem;
    text-align: center;
}
.kpi-card {
    background: linear-gradient(160deg, #0c1a3a 0%, #080f22 100%);
    border: 1px solid rgba(59,130,246,0.22);
    border-radius: 14px;
    padding: 14px;
    min-height: 110px;
}
.kpi-label {
    color: #94a3b8;
    font-size: 0.72rem;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.kpi-value {
    color: #f8fafc;
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 0.15rem;
}
.kpi-sub {
    color: #cbd5e1;
    font-size: 0.78rem;
}
.section-title {
    color: #dbeafe;
    font-size: 1rem;
    font-weight: 700;
    margin: 0.6rem 0 0.4rem;
}
.kpi-heading-center {
    text-align: center;
    color: #e2e8f0;
    font-size: 1.22rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    margin: 0.22rem 0 0.45rem;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stMarkdownContainer"] .owner-table-wrap {
    border: 1px solid rgba(91, 84, 255, 0.36);
    border-radius: 14px;
    overflow-x: auto;
    background: linear-gradient(145deg, #0f1b3d 0%, #0a132b 55%, #081126 100%);
    box-shadow: 0 10px 26px rgba(8, 15, 34, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.04);
}
.owner-table {
    width: 100%;
    border-collapse: collapse;
    min-width: 1250px;
}
.owner-table th {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.82) 0%, rgba(91, 84, 255, 0.66) 100%);
    color: #e8efff;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    text-align: center;
    padding: 10px 8px;
    border-bottom: 1px solid rgba(99, 102, 241, 0.45);
    white-space: nowrap;
}
.owner-table td {
    color: #e2e8f0;
    font-size: 0.84rem;
    text-align: center;
    padding: 8px;
    border-bottom: 1px solid rgba(71, 85, 105, 0.34);
}
.owner-table tbody tr:nth-child(even) {
    background: rgba(148, 163, 184, 0.04);
}
.owner-table tbody tr:hover {
    background: rgba(91, 84, 255, 0.16);
}
.owner-table td.left {
    text-align: left;
    white-space: nowrap;
    font-weight: 600;
}
[data-testid="stDataFrame"] {
    background: linear-gradient(135deg, #111a33 0%, #0b1226 100%) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 14px !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )
@st.cache_resource
def _get_connection_pool():
    return psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        dbname=DB_CONFIG["dbname"],
    )


@contextmanager
def get_db_connection():
    pool = _get_connection_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)

@st.cache_data(ttl=3600)
def load_shop_name_map() -> dict:
    """Load shop code → shop description from shopmgrname table."""
    def _clean_shop_name(name: str) -> str:
        txt = str(name or "").strip()
        # Remove leading brand prefix from display names (e.g., "Melcom Spintex" -> "Spintex").
        txt = re.sub(r"^\s*melcom\s+", "", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT shop_code, shop_description FROM shopmgrname WHERE is_current = TRUE"
            )
            return {row[0]: _clean_shop_name(row[1]) for row in cur.fetchall()}


def ensure_shop_tracking_table():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS invoice_shop_tracking (
                    shop_code TEXT PRIMARY KEY,
                    first_seen_at TIMESTAMP NOT NULL,
                    last_seen_at TIMESTAMP NOT NULL
                )
                """
            )
        conn.commit()


def ensure_usage_events_table():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS invoice_dashboard_user_events (
                    id BIGSERIAL PRIMARY KEY,
                    event_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    event_name VARCHAR(100) NOT NULL,
                    event_type VARCHAR(30) NOT NULL DEFAULT 'interaction',
                    session_id VARCHAR(64),
                    system_login_id VARCHAR(150),
                    ip_address VARCHAR(64),
                    event_details JSONB
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_invoice_dash_events_time
                ON invoice_dashboard_user_events(event_time DESC)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_invoice_dash_events_session
                ON invoice_dashboard_user_events(session_id)
                """
            )
        conn.commit()


def ensure_cashier_sessions_mv_exists() -> None:
    """Create cashier-session MV on demand to prevent runtime query failures."""
    create_sql = """
    CREATE MATERIALIZED VIEW mv_wh_erp_cashier_sessions_daily AS
    WITH base AS (
        SELECT
            e.invdate::date AS bill_date,
            UPPER(TRIM(e.store_code)) AS shop_code,
            UPPER(TRIM(COALESCE(e.cashier, ''))) AS cashier_name,
            CASE
                WHEN NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '') IS NULL THEN NULL
                ELSE NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '')::int
            END AS till_no,
            COALESCE(e.amt, 0)::numeric AS amt
        FROM erpdata e
        WHERE NULLIF(TRIM(COALESCE(e.cashier, '')), '') IS NOT NULL
          AND NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '') IS NOT NULL
          AND NULLIF(TRIM(COALESCE(e.invno::text, '')), '') IS NOT NULL
    )
    SELECT
        bill_date,
        shop_code,
        cashier_name,
        till_no,
        MAX(CASE WHEN amt = 0.01 THEN 1 ELSE 0 END) AS has_test_bill
    FROM base
    GROUP BY bill_date, shop_code, cashier_name, till_no
    WITH DATA;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.mv_wh_erp_cashier_sessions_daily')")
            exists = cur.fetchone()[0] is not None
            if exists:
                return

            logger.warning("Missing mv_wh_erp_cashier_sessions_daily. Creating it automatically.")
            cur.execute(create_sql)
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uix_mv_wh_erp_cashier_sessions
                ON mv_wh_erp_cashier_sessions_daily (bill_date, shop_code, cashier_name, till_no)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_mv_wh_erp_cashier_sessions_shop
                ON mv_wh_erp_cashier_sessions_daily (bill_date, shop_code)
                """
            )
        conn.commit()


def get_local_ipv4() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip:
            return ip
    except Exception:
        pass

    return ""


def get_os_login_identity() -> str:
    domain = str(os.getenv("USERDOMAIN", "") or "").strip()
    username = str(os.getenv("USERNAME", "") or os.getenv("USER", "") or "").strip()
    if domain and username:
        return f"{domain}\\{username}"
    if username:
        return username
    try:
        return str(getpass.getuser() or "").strip()
    except Exception:
        return ""


def get_client_context() -> dict:
    headers = {}
    try:
        headers_obj = getattr(st, 'context', None)
        if headers_obj and hasattr(headers_obj, 'headers') and headers_obj.headers:
            headers = dict(headers_obj.headers)
    except Exception:
        headers = {}

    ip = headers.get('X-Forwarded-For', '') or headers.get('X-Real-Ip', '')
    if ',' in ip:
        ip = ip.split(',')[0].strip()

    auth_login = (
        headers.get('Remote-User', '')
        or headers.get('X-Remote-User', '')
        or headers.get('X-Authenticated-User', '')
    )

    return {
        'ip_address': ip,
        'auth_login': auth_login,
    }


def ensure_identity_captured() -> tuple[str, str]:
    if 'usage_system_login_id' not in st.session_state:
        st.session_state.usage_system_login_id = ''
    if 'usage_ip_address' not in st.session_state:
        st.session_state.usage_ip_address = ''

    context = get_client_context()

    if not st.session_state.usage_system_login_id:
        st.session_state.usage_system_login_id = (
            str(context.get('auth_login', '') or '').strip()
            or get_os_login_identity()
        )

    if not st.session_state.usage_ip_address:
        st.session_state.usage_ip_address = (
            str(context.get('ip_address', '') or '').strip()
            or get_local_ipv4()
        )

    return (
        str(st.session_state.get('usage_system_login_id', '') or '').strip(),
        str(st.session_state.get('usage_ip_address', '') or '').strip(),
    )


def log_usage_event(event_name: str, event_type: str = 'interaction', event_details: dict | None = None):
    try:
        ensure_usage_events_table()
        if 'usage_event_session_id' not in st.session_state:
            st.session_state.usage_event_session_id = uuid.uuid4().hex

        system_login_id, ip_address = ensure_identity_captured()
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO invoice_dashboard_user_events
                    (event_time, event_name, event_type, session_id, system_login_id, ip_address, event_details)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        datetime.now(),
                        event_name,
                        event_type,
                        st.session_state.usage_event_session_id,
                        system_login_id,
                        ip_address,
                        Json(event_details or {}),
                    ),
                )
            conn.commit()
    except Exception as exc:
        logger.warning(f"Usage event logging failed for {event_name}: {exc}")


def get_usage_summary(days_back: int = 30) -> dict:
    ensure_usage_events_table()
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                WITH session_rollup AS (
                    SELECT
                        session_id,
                        MIN(event_time) AS session_start,
                        MAX(event_time) AS session_end,
                        COUNT(*) AS session_events,
                        COUNT(*) FILTER (WHERE event_type IN ('click', 'download', 'filter')) AS session_clicks
                    FROM invoice_dashboard_user_events
                    WHERE event_time >= NOW() - (%s || ' days')::INTERVAL
                    GROUP BY session_id
                )
                SELECT
                    COALESCE((SELECT COUNT(*) FROM invoice_dashboard_user_events WHERE event_time >= NOW() - (%s || ' days')::INTERVAL), 0) AS total_events,
                    COALESCE((SELECT COUNT(DISTINCT session_id) FROM invoice_dashboard_user_events WHERE event_time >= NOW() - (%s || ' days')::INTERVAL), 0) AS total_sessions,
                    COALESCE((SELECT COUNT(*) FROM invoice_dashboard_user_events WHERE event_time >= NOW() - (%s || ' days')::INTERVAL AND event_type IN ('click', 'download', 'filter')), 0) AS total_event_clicks,
                    COALESCE((SELECT SUM(EXTRACT(EPOCH FROM (session_end - session_start))) FROM session_rollup), 0) AS total_time_spent_seconds,
                    COALESCE((SELECT AVG(EXTRACT(EPOCH FROM (session_end - session_start))) FROM session_rollup), 0) AS avg_session_seconds
                """,
                (str(days_back), str(days_back), str(days_back), str(days_back)),
            )
            return dict(cur.fetchone() or {})


def get_usage_events_report(days_back: int = 30) -> pd.DataFrame:
    ensure_usage_events_table()
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    event_time,
                    event_name,
                    event_type,
                    COALESCE(NULLIF(system_login_id, ''), 'unknown') AS system_login_id,
                    COALESCE(NULLIF(ip_address, ''), 'unknown') AS ip_address,
                    session_id,
                    event_details
                FROM invoice_dashboard_user_events
                WHERE event_time >= NOW() - (%s || ' days')::INTERVAL
                ORDER BY event_time DESC
                """,
                (str(days_back),),
            )
            return pd.DataFrame(cur.fetchall())


def update_and_get_new_shops(current_shop_codes: list[str]) -> set[str]:
    if not current_shop_codes:
        return set()

    ensure_shop_tracking_table()
    normalized = sorted({str(s).strip().upper() for s in current_shop_codes if str(s).strip()})
    if not normalized:
        return set()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO invoice_shop_tracking (shop_code, first_seen_at, last_seen_at)
                SELECT shop_code, NOW(), NOW()
                FROM unnest(%s::text[]) AS t(shop_code)
                ON CONFLICT (shop_code)
                DO UPDATE SET last_seen_at = EXCLUDED.last_seen_at
                """,
                (normalized,),
            )

            cur.execute(
                """
                SELECT shop_code
                FROM invoice_shop_tracking
                WHERE first_seen_at >= NOW() - INTERVAL '1 day'
                  AND shop_code = ANY(%s::text[])
                """,
                (normalized,),
            )
            new_rows = cur.fetchall()

            if len(normalized) > 5 and len(new_rows) == len(normalized):
                cur.execute(
                    """
                    UPDATE invoice_shop_tracking
                    SET first_seen_at = NOW() - INTERVAL '2 days'
                    WHERE shop_code = ANY(%s::text[])
                    """,
                    (normalized,),
                )
                conn.commit()
                return set()
        conn.commit()
    return {row[0] for row in new_rows}


def _build_live_scan_agg_cte(group_by_columns: list[str], extra_where: str = "") -> str:
    group_select = ",\n            ".join(group_by_columns)
    group_by = ", ".join(group_by_columns)
    using_clause = ", ".join(group_by_columns)
    extra_where_sql = f"\n        {extra_where.strip()}" if extra_where and extra_where.strip() else ""
    # Convert "WHERE cond" → "AND cond" for appending inside an existing WHERE clause
    extra_and_sql = (
        "\n          " + extra_where.strip().replace("WHERE ", "AND ", 1)
    ) if extra_where.strip() else ""
    # For consumable_agg subquery, qualify shop_code references with table prefix "iu."
    extra_and_sql_qualified = re.sub(r'\bshop_code\b', 'iu.shop_code', extra_and_sql) if extra_and_sql else ""
    
    return f"""
    consumable_tills AS (
        SELECT * FROM (VALUES
            ('SPN', 34), ('SPN', 32), ('SPN', 33),
            ('LFS', 23), ('LFS', 26),
            ('MSS', 13), ('MSS', 14),
            ('MM1', 1), ('MM1', 13),
            ('MM2', 10), ('MM2', 9),
            ('WHL', 14)
        ) AS t(shop_code, till_no)
    ),
    invoice_union AS (
        SELECT
            i.invdate::date AS bill_date,
            UPPER(TRIM(i.store_code)) AS shop_code,
            TRIM(COALESCE(i.invno::text, '')) AS invno,
            COALESCE(i.amt, 0)::numeric AS amt,
            CASE
                WHEN NULLIF(REGEXP_REPLACE(COALESCE(i.tillno::text, ''), '[^0-9]', '', 'g'), '') IS NULL THEN NULL
                ELSE NULLIF(REGEXP_REPLACE(COALESCE(i.tillno::text, ''), '[^0-9]', '', 'g'), '')::int
            END AS till_no,

            1 AS src_priority
        FROM invoices i
        WHERE NULLIF(TRIM(COALESCE(i.invno::text, '')), '') IS NOT NULL
          AND i.invdate::date BETWEEN %(s)s AND %(e)s
                    AND i.duplicate = 1

        UNION ALL

        SELECT
            m.invdate::date AS bill_date,
            UPPER(TRIM(m.store_code)) AS shop_code,
            TRIM(COALESCE(m.invno::text, '')) AS invno,
            COALESCE(m.amt, 0)::numeric AS amt,
            CASE
                WHEN NULLIF(REGEXP_REPLACE(COALESCE(m.tillno::text, ''), '[^0-9]', '', 'g'), '') IS NULL THEN NULL
                ELSE NULLIF(REGEXP_REPLACE(COALESCE(m.tillno::text, ''), '[^0-9]', '', 'g'), '')::int
            END AS till_no,
            2 AS src_priority
        FROM invoices_manager m
        WHERE NULLIF(TRIM(COALESCE(m.invno::text, '')), '') IS NOT NULL
          AND m.invdate::date BETWEEN %(s)s AND %(e)s
    ),
    manager_agg AS (
        -- Count distinct invoice numbers handed over to manager per shop within date range
        SELECT
            UPPER(TRIM(m.store_code)) AS shop_code,
            COUNT(DISTINCT NULLIF(TRIM(COALESCE(m.invno::text, '')), ''))::bigint AS handover_test_bill_to_manager
        FROM invoices_manager m
        WHERE m.invdate::date BETWEEN %(s)s AND %(e)s
          AND UPPER(TRIM(COALESCE(m.store_code, ''))) = ANY(%(shops)s)
          AND NULLIF(TRIM(COALESCE(m.invno::text, '')), '') IS NOT NULL
        GROUP BY UPPER(TRIM(m.store_code))
    )
            FROM invoice_union iu
            INNER JOIN consumable_tills ct ON ct.shop_code = iu.shop_code AND ct.till_no = iu.till_no
            WHERE iu.src_priority = 1{extra_and_sql_qualified}
        ) consumable_matches
        GROUP BY {group_by}
    ),
    scan_agg AS (
        SELECT
            {using_clause},
            COALESCE(r.scan_nob, 0)            AS scan_nob,
            COALESCE(r.scanned_nob_ghs, 0)     AS scanned_nob_ghs,
            COALESCE(c.consumable_till_nob, 0) AS consumable_till_nob,
            COALESCE(c.consumable_nob_ghs, 0)  AS consumable_nob_ghs
        FROM raw_invoice_agg r
        FULL OUTER JOIN consumable_agg c USING ({using_clause})
    ),
    """


@st.cache_data(ttl=300)
def load_owner_view(start_date: date, end_date: date) -> pd.DataFrame:
    query = f"""
    WITH erp_agg AS (
        SELECT
            shop_code,
            SUM(erp_nob)     AS erp_nob,
            SUM(erp_nob_ghs) AS erp_nob_ghs,
            SUM(test_bills)  AS test_bills
        FROM mv_wh_erp_daily
        WHERE bill_date BETWEEN %(s)s AND %(e)s
        GROUP BY shop_code
    ),
    {_build_live_scan_agg_cte(['shop_code'])}
    session_flags AS (
        SELECT
            UPPER(TRIM(e.store_code)) AS shop_code,
            UPPER(TRIM(COALESCE(e.cashier, ''))) AS cashier_name,
            NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '')::int AS till_no,
            MAX(CASE WHEN COALESCE(e.amt, 0) = 0.01 THEN 1 ELSE 0 END) AS has_test_bill
        FROM erpdata e
        WHERE e.invdate::date BETWEEN %(s)s AND %(e)s
          AND NULLIF(TRIM(COALESCE(e.store_code, '')), '') IS NOT NULL
          AND NULLIF(TRIM(COALESCE(e.cashier, '')), '') IS NOT NULL
          AND NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '') IS NOT NULL
          AND NULLIF(TRIM(COALESCE(e.invno::text, '')), '') IS NOT NULL
        GROUP BY UPPER(TRIM(e.store_code)), UPPER(TRIM(COALESCE(e.cashier, ''))), NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '')::int
    ),
    test_not_generated_agg AS (
        SELECT shop_code,
               COUNT(*) FILTER (WHERE has_test_bill = 0) AS test_bill_not_generated
        FROM session_flags
        GROUP BY shop_code
    ),
    alerts_agg AS (
        SELECT
            shop_code,
            SUM(alert_count) FILTER (WHERE a_type_normalized = 'Test Bill')         AS alert_test_bill_scanned,
            SUM(alert_count) FILTER (WHERE a_type_normalized = 'Bill Date Mismatch') AS alert_bill_date_mismatched,
            SUM(alert_count) FILTER (WHERE a_type_normalized = 'Duplicate')          AS alert_duplicate,
            SUM(alert_count) FILTER (WHERE a_type_normalized = 'Wrong Shop')         AS alert_wrong_shop,
            SUM(alert_count) FILTER (WHERE a_type_normalized = 'High Bill Amount')   AS alert_high_value_bill
        FROM mv_wh_alerts_daily
        WHERE alert_date BETWEEN %(s)s AND %(e)s
        GROUP BY shop_code
    ),
    all_shops AS (
        SELECT DISTINCT shop_code FROM mv_wh_erp_daily       WHERE bill_date BETWEEN %(s)s AND %(e)s
        UNION
        SELECT DISTINCT shop_code FROM scan_agg
        UNION
        SELECT DISTINCT shop_code FROM mv_wh_alerts_daily    WHERE alert_date BETWEEN %(s)s AND %(e)s
    )
    SELECT
        s.shop_code,
        COALESCE(e.erp_nob, 0)             AS erp_nob,
        COALESCE(sc.scan_nob, 0)            AS scan_nob,
        COALESCE(sc.consumable_till_nob, 0) AS consumable_till_nob,
        COALESCE(e.test_bills, 0)           AS test_bills,
        COALESCE(tn.test_bill_not_generated, 0) AS test_bill_not_generated,
        (COALESCE(sc.scan_nob, 0) + COALESCE(sc.consumable_till_nob, 0) + COALESCE(e.test_bills, 0)) AS total_accounted_bills,
        GREATEST(COALESCE(e.erp_nob, 0) - (COALESCE(sc.scan_nob, 0) + COALESCE(sc.consumable_till_nob, 0) + COALESCE(e.test_bills, 0)), 0) AS bills_not_scanned,
        CASE
            WHEN COALESCE(e.erp_nob, 0) > 0
            THEN ROUND((COALESCE(sc.scan_nob, 0)::numeric / e.erp_nob) * 100, 2)
            ELSE 0
        END AS bill_pct,
        COALESCE(e.erp_nob_ghs, 0)          AS erp_erp_nob_ghs,
        COALESCE(sc.scanned_nob_ghs, 0)     AS scanned_nob_ghs,
        COALESCE(sc.consumable_nob_ghs, 0)  AS consumable_nob_ghs,
        (COALESCE(e.erp_nob_ghs, 0) - (COALESCE(sc.scanned_nob_ghs, 0) + COALESCE(sc.consumable_nob_ghs, 0))) AS diff_ghs,
        CASE
            WHEN COALESCE(e.erp_nob_ghs, 0) > 0
            THEN ROUND((COALESCE(sc.scanned_nob_ghs, 0) / e.erp_nob_ghs) * 100, 2)
            ELSE 0
        END AS diff_pct,
        COALESCE(a.alert_test_bill_scanned, 0)     AS test_bill_scanned,
        COALESCE(a.alert_bill_date_mismatched, 0)  AS bill_date_mismatched,
        COALESCE(a.alert_duplicate, 0)             AS duplicate,
        COALESCE(a.alert_wrong_shop, 0)            AS wrong_shop,
        COALESCE(a.alert_high_value_bill, 0)       AS high_value_bill
    FROM all_shops s
    LEFT JOIN erp_agg e              ON e.shop_code  = s.shop_code
    LEFT JOIN scan_agg sc            ON sc.shop_code = s.shop_code
    LEFT JOIN test_not_generated_agg tn ON tn.shop_code = s.shop_code
    LEFT JOIN alerts_agg a           ON a.shop_code  = s.shop_code
    WHERE s.shop_code IS NOT NULL AND TRIM(s.shop_code) <> ''
      AND s.shop_code NOT IN ('G01','G02','G03','G04','G05','G06','G07','G08','G09','G10','G11','G12','INV','SPX','SEL')
    ORDER BY s.shop_code
    """

    with get_db_connection() as conn:
        return pd.read_sql(query, conn, params={"s": start_date, "e": end_date})


@st.cache_data(ttl=300)
def load_shopwise_test_bill_analysis(start_date: date, end_date: date) -> pd.DataFrame:
    query = """
    WITH shops AS (
        SELECT unnest(%(shops)s::text[]) AS shop_code
    ),
    -- erpdata is the master source for all session/bill metrics (has every cashier who transacted)
    erp_rows AS (
        SELECT
            e.invdate::date AS bill_date,
            UPPER(TRIM(e.store_code)) AS shop_code,
            UPPER(TRIM(COALESCE(e.cashier, ''))) AS cashier_name,
            NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '')::int AS till_no,
            COALESCE(e.amt, 0)::numeric AS amt,
            TRIM(COALESCE(e.invno::text, '')) AS invno
        FROM erpdata e
        WHERE e.invdate::date BETWEEN %(s)s AND %(e)s
          AND UPPER(TRIM(e.store_code)) = ANY(%(shops)s)
          AND NULLIF(TRIM(COALESCE(e.cashier, '')), '') IS NOT NULL
          AND NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '') IS NOT NULL
          AND NULLIF(TRIM(COALESCE(e.invno::text, '')), '') IS NOT NULL
    ),
    -- One row per cashier+till session per day (used for Cashier Login, Not Generated counts)
    erp_sessions AS (
        SELECT
            bill_date,
            shop_code,
            cashier_name,
            till_no,
            MAX(CASE WHEN amt = 0.01 THEN 1 ELSE 0 END) AS has_test_bill
        FROM erp_rows
        GROUP BY bill_date, shop_code, cashier_name, till_no
    ),
    cashier_agg AS (
        SELECT
            shop_code,
            COUNT(*)::bigint AS cashier_login,
            COUNT(*) FILTER (WHERE has_test_bill = 0)::bigint AS test_bill_not_generated
        FROM erp_sessions
        GROUP BY shop_code
    ),
    generated_agg AS (
        SELECT
            shop_code,
            -- Count total invno where amt = 0.01 (each invoice counted once)
            COUNT(*) FILTER (
                WHERE COALESCE(amt, 0) = 0.01
                  AND NULLIF(TRIM(COALESCE(invno, '')), '') IS NOT NULL
            )::bigint AS test_bills_generated
        FROM erp_rows
        GROUP BY shop_code
    ),
    unique_generated_agg AS (
        SELECT
            shop_code,
            COUNT(*) FILTER (WHERE has_test_bill = 1)::bigint AS unique_test_bills_generated
        FROM erp_sessions
        GROUP BY shop_code
    ),
    manager_rows AS (
        SELECT
            m.invdate::date AS bill_date,
            UPPER(TRIM(COALESCE(m.store_code, ''))) AS shop_code,
            UPPER(TRIM(COALESCE(m.cashier, ''))) AS cashier_name,
            NULLIF(REGEXP_REPLACE(COALESCE(m.tillno::text, ''), '[^0-9]', '', 'g'), '')::int AS till_no,
            NULLIF(TRIM(COALESCE(m.invno::text, '')), '') AS invno
        FROM invoices_manager m
        WHERE m.invdate::date BETWEEN %(s)s AND %(e)s
          AND UPPER(TRIM(COALESCE(m.store_code, ''))) = ANY(%(shops)s)
    ),
    manager_sessions AS (
        SELECT
            bill_date,
            shop_code,
            cashier_name,
            till_no,
            MAX(CASE WHEN invno IS NOT NULL THEN 1 ELSE 0 END) AS has_handover_bill
        FROM manager_rows
        WHERE cashier_name <> ''
          AND till_no IS NOT NULL
        GROUP BY bill_date, shop_code, cashier_name, till_no
    ),
    manager_agg AS (
        SELECT
            shop_code,
            -- Count unique invoice numbers handed over to manager (invno not null)
            COUNT(*) FILTER (WHERE has_handover_bill = 1)::bigint AS handover_test_bill_to_manager
        FROM manager_sessions
        GROUP BY shop_code
    )
    SELECT
        s.shop_code,
        COALESCE(c.cashier_login, 0)::bigint                       AS cashier_login,
        COALESCE(c.test_bill_not_generated, 0)::bigint             AS test_bill_not_generated,
        COALESCE(g.test_bills_generated, 0)::bigint                AS test_bills_generated,
        COALESCE(ug.unique_test_bills_generated, 0)::bigint        AS unique_test_bills_generated,
        CASE
            WHEN COALESCE(c.cashier_login, 0) > 0
            THEN ROUND((COALESCE(c.test_bill_not_generated, 0)::numeric / c.cashier_login) * 100, 2)
            ELSE 0
        END AS not_generated_bill_pct,
        CASE
            WHEN COALESCE(c.cashier_login, 0) > 0
            THEN ROUND((COALESCE(ug.unique_test_bills_generated, 0)::numeric / c.cashier_login) * 100, 2)
            ELSE 0
        END AS generated_test_bill_pct,
        COALESCE(m.handover_test_bill_to_manager, 0)::bigint       AS handover_test_bill_to_manager,
        CASE
            WHEN COALESCE(ug.unique_test_bills_generated, 0) > 0
            THEN ROUND((COALESCE(m.handover_test_bill_to_manager, 0)::numeric / ug.unique_test_bills_generated) * 100, 2)
            ELSE 0
        END AS bill_handover_pct,
        (COALESCE(ug.unique_test_bills_generated, 0) - COALESCE(m.handover_test_bill_to_manager, 0))::bigint AS missing_test_bill
    FROM shops s
    LEFT JOIN cashier_agg c  ON c.shop_code = s.shop_code
    LEFT JOIN generated_agg g ON g.shop_code = s.shop_code
    LEFT JOIN unique_generated_agg ug ON ug.shop_code = s.shop_code
    LEFT JOIN manager_agg m  ON m.shop_code = s.shop_code
    ORDER BY s.shop_code
    """

    with get_db_connection() as conn:
        return pd.read_sql(
            query,
            conn,
            params={
                "s": start_date,
                "e": end_date,
                "shops": sorted(list(IMPLEMENTED_SHOPS_SET)),
            },
        )

@st.cache_data(ttl=300)
def load_shopwise_test_bill_drilldown(
    shop_code: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    query = """
    SELECT
        UPPER(TRIM(e.store_code)) AS shop_code,
        UPPER(TRIM(COALESCE(e.cashier, ''))) AS cashier_name,
        e.invdate::date AS invdate,
        TRIM(COALESCE(e.invno::text, '')) AS invno,
        COALESCE(e.amt, 0)::numeric AS amt,
        CASE
            WHEN m.invno IS NOT NULL THEN 'Y'
            ELSE 'N'
        END AS handed_over_to_manager
    FROM erpdata e
    LEFT JOIN (
        SELECT DISTINCT TRIM(COALESCE(invno::text, '')) AS invno
        FROM invoices_manager
        WHERE invdate::date BETWEEN %(s)s AND %(e)s
          AND UPPER(TRIM(store_code)) = %(shop)s
          AND NULLIF(TRIM(COALESCE(invno::text, '')), '') IS NOT NULL
    ) m
        ON TRIM(COALESCE(e.invno::text, '')) = m.invno
    WHERE e.invdate::date BETWEEN %(s)s AND %(e)s
      AND UPPER(TRIM(e.store_code)) = %(shop)s
      AND NULLIF(TRIM(COALESCE(e.invno::text, '')), '') IS NOT NULL
      AND COALESCE(e.amt, 0) = 0.01
    ORDER BY e.invdate DESC, e.invno DESC
    """

    with get_db_connection() as conn:
        return pd.read_sql(
            query,
            conn,
            params={"s": start_date, "e": end_date, "shop": (shop_code or "").strip().upper()},
        )


def render_shopwise_test_bill_drilldown_table(
    selected_shop: str,
    start_date: date,
    end_date: date,
    shop_name_map: dict,
    title_prefix: str,
    table_key_prefix: str,
):
    drilldown_df = load_shopwise_test_bill_drilldown(selected_shop, start_date, end_date)
    if drilldown_df is None or drilldown_df.empty:
        st.info(f"No drilldown records found for {selected_shop} ({start_date} -> {end_date}).")
        return

    shop_name = shop_name_map.get(selected_shop, selected_shop)
    drilldown_df["shop_name"] = shop_name
    detail_display = drilldown_df.rename(
        columns={
            "shop_name": "Shop Name",
            "cashier_name": "Cashier name",
            "invdate": "InvDate",
            "invno": "InvNo",
            "amt": "Amt",
            "handed_over_to_manager": "Handed Over",
        }
    )[["Shop Name", "Cashier name", "InvDate", "InvNo", "Amt", "Handed Over"]]

    detail_headers = list(detail_display.columns)
    detail_rows: list[list[str]] = []
    detail_sort_rows: list[list[str]] = []
    detail_cell_styles: dict[tuple[int, int], str] = {}
    handed_over_col_idx = detail_headers.index("Handed Over") if "Handed Over" in detail_headers else -1

    for drow_idx, (_, drow) in enumerate(detail_display.iterrows()):
        inv_date_val = drow["InvDate"]
        inv_date_disp = ""
        if pd.notna(inv_date_val):
            try:
                inv_date_disp = pd.to_datetime(inv_date_val).strftime("%Y-%m-%d")
            except Exception:
                inv_date_disp = str(inv_date_val)

        handed_over_val = str(drow.get("Handed Over", "N")).strip().upper()
        row_vals = [
            str(drow["Shop Name"]),
            str(drow["Cashier name"]),
            inv_date_disp,
            str(drow["InvNo"]),
            f"{float(drow['Amt']):,.2f}",
            handed_over_val,
        ]
        detail_rows.append(row_vals)
        detail_sort_rows.append(
            [
                _sort_token(drow["Shop Name"]),
                _sort_token(drow["Cashier name"]),
                _sort_token(drow["InvDate"]),
                _sort_token(drow["InvNo"]),
                _sort_token(drow["Amt"]),
                _sort_token(handed_over_val),
            ]
        )

        if handed_over_col_idx >= 0:
            if handed_over_val == "Y":
                detail_cell_styles[(drow_idx, handed_over_col_idx)] = (
                    "background: rgba(74, 222, 128, 0.22); color: #dcfce7; font-weight: 700;"
                )
            else:
                detail_cell_styles[(drow_idx, handed_over_col_idx)] = (
                    "background: rgba(248, 113, 113, 0.22); color: #fecaca; font-weight: 700;"
                )

    _, dd_period = _period_labels(start_date, end_date)
    shop_label = f"{selected_shop} - {shop_name}"
    st.markdown(
        f"<div class='kpi-heading-center' style='margin-top:0px; margin-bottom:0.35rem;'>"
        f"{title_prefix}: {shop_label} • {dd_period}</div>",
        unsafe_allow_html=True,
    )
    render_sortable_html_table(
        title="",
        headers=detail_headers,
        rows=detail_rows,
        sort_values=detail_sort_rows,
        table_id=f"{table_key_prefix}_{selected_shop}_{start_date}_{end_date}",
        show_title=False,
        title_class="section-title",
        min_width_px=980,
        left_align_cols={0, 1, 3},
        cell_styles=detail_cell_styles,
        enable_download_hover=True,
        download_file_name=f"{table_key_prefix}_{selected_shop}_{start_date}_{end_date}.csv",
        wrapper_max_height_px=400,
        table_width="100%",
        table_layout="fixed",
        overflow_x="auto",
        header_white_space="normal",
        header_word_break="break-word",
        cell_white_space="nowrap",
        cell_word_break="normal",
        text_overflow="ellipsis",
        header_font_size="10px",
        cell_font_size="10.5px",
        header_padding="8px 6px",
        cell_padding="7px 6px",
    )


def render_shopwise_test_bill_analysis(start_date: date, end_date: date, key_suffix: str = ""):
    raw_df = load_shopwise_test_bill_analysis(start_date, end_date)
    if raw_df is None or raw_df.empty:
        st.info("No shop-wise test bill analysis data for selected date range.")
        return

    shop_name_map = load_shop_name_map()
    raw_df["shop_name"] = raw_df["shop_code"].map(shop_name_map).fillna(raw_df["shop_code"])

    display_df = raw_df.rename(
        columns={
            "shop_code": "Shop Code",
            "shop_name": "Shop Name",
            "cashier_login": "Cashier Login",
            "test_bill_not_generated": "Cashier Not Generated Test Bills",
            "test_bills_generated": "Cashier Generated Test Bills (Raw)",
            "unique_test_bills_generated": "Cashier Generated Test Bills",
            "not_generated_bill_pct": "Not Generated Bill %",
            "generated_test_bill_pct": "Generated Test Bill %",
            "handover_test_bill_to_manager": "Unique bill handover to manager",
            "bill_handover_pct": "Bill handover %",
            "missing_test_bill": "Missing test bill",
        }
    )

    raw_test_generated_lookup = pd.to_numeric(
        display_df.get("Cashier Generated Test Bills (Raw)"), errors="coerce"
    ).fillna(0)

    unique_test_generated_lookup = pd.to_numeric(
        display_df.get("Cashier Generated Test Bills"), errors="coerce"
    ).fillna(0)

    cashier_login_series = pd.to_numeric(display_df.get("Cashier Login"), errors="coerce").fillna(0)

    ordered_cols = [
        "Shop Code",
        "Shop Name",
        "Cashier Login",
        "Cashier Not Generated Test Bills",
        "Cashier Generated Test Bills",
        "Unique bill handover to manager",
        "Missing test bill",
        "Not Generated Bill %",
        "Generated Test Bill %",
        "Bill handover %",
    ]
    display_df = display_df[[c for c in ordered_cols if c in display_df.columns]]

    headers = list(display_df.columns)
    rows: list[list[str]] = []
    sort_rows: list[list[str]] = []
    cell_titles: dict[tuple[int, int], str] = {}
    cell_styles: dict[tuple[int, int], str] = {}
    pct_cols = {"Not Generated Bill %", "Generated Test Bill %", "Bill handover %"}

    def _pct_style(col_name: str, pct_value: float) -> str:
        # Match One-View Summary Bill % color scheme exactly.
        style_val = pct_value
        if col_name in {"Generated Test Bill %", "Bill handover %"} and pct_value > 100:
            # Formatting-only override requested by user; display value remains unchanged.
            style_val = 0.0

        if style_val > 100:
            return "background: rgba(125, 211, 252, 0.25); color: #e0f2fe; font-weight: 700;"
        if style_val < 70:
            return "background: rgba(248, 113, 113, 0.18); color: #fecaca; font-weight: 600;"
        if style_val < 90:
            return "background: rgba(250, 204, 21, 0.18); color: #fef9c3; font-weight: 600;"
        return "background: rgba(74, 222, 128, 0.18); color: #dcfce7; font-weight: 600;"

    for row_idx, (_, row) in enumerate(display_df.iterrows()):
        r = []
        sr = []

        cashier_login = float(row["Cashier Login"])
        test_not_generated = float(row["Cashier Not Generated Test Bills"])
        test_generated = float(row["Cashier Generated Test Bills"])
        unique_test_generated = float(unique_test_generated_lookup.loc[row.name])
        handover_to_manager = float(row["Unique bill handover to manager"])
        bill_handover_pct = float(row["Bill handover %"])
        missing_test_bill = float(row["Missing test bill"])
        not_gen_pct = float(row["Not Generated Bill %"])
        gen_pct = float(row["Generated Test Bill %"])

        tooltip_map = {
            "Cashier Login": (
                "Count of cashier+till sessions from erpdata for selected period."
            ),
            "Cashier Not Generated Test Bills": (
                "Count of cashier+till sessions (from erpdata) that do not have any test bill (amt = 0.01)."
            ),
            "Cashier Generated Test Bills": "Count of unique cashier+till sessions with at least one test bill (amt = 0.01).",
            "Not Generated Bill %": (
                f"(Cashier Not Generated Test Bills ({test_not_generated:,.0f}) / Cashier Login ({cashier_login:,.0f})) x 100 = {not_gen_pct:.2f}%"
                if cashier_login > 0 else "Cashier Login is 0, so Not Generated Bill % = 0"
            ),
            "Generated Test Bill %": (
                f"(Cashier Generated Test Bills ({unique_test_generated:,.0f}) / Cashier Login ({cashier_login:,.0f})) x 100 = {gen_pct:.2f}%"
                if cashier_login > 0 else "Cashier Login is 0, so Generated Test Bill % = 0"
            ),
            "Unique bill handover to manager": (
                f"Count of distinct invoices (invno) recorded in invoices_manager for the selected period and shop = {handover_to_manager:,.0f}."
            ),
            "Bill handover %": (
                f"(Unique bill handover to manager ({handover_to_manager:,.0f}) / Cashier Generated Test Bills ({unique_test_generated:,.0f})) x 100 = {bill_handover_pct:.2f}%"
                if unique_test_generated > 0 else "Cashier Generated Test Bills is 0, so Bill handover % = 0"
            ),
            "Missing test bill": (
                f"Cashier Generated Test Bills ({unique_test_generated:,.0f}) - Unique bill handover to manager ({handover_to_manager:,.0f}) = {missing_test_bill:,.0f}"
            ),
        }

        for col_idx, col in enumerate(headers):
            val = row[col]
            if col in pct_cols:
                disp = f"{float(val):.2f}%"
            elif col in {"Shop Code", "Shop Name"}:
                disp = str(val)
            else:
                disp = f"{int(float(val)):,}"

            r.append(disp)
            sr.append(_sort_token(val))
            if col in tooltip_map:
                cell_titles[(row_idx, col_idx)] = tooltip_map[col]
            if col in pct_cols:
                cell_styles[(row_idx, col_idx)] = _pct_style(col, float(val))
        rows.append(r)
        sort_rows.append(sr)

    totals = {
        "Cashier Login": float(pd.to_numeric(display_df["Cashier Login"], errors="coerce").fillna(0).sum()),
        "Cashier Not Generated Test Bills": float(pd.to_numeric(display_df["Cashier Not Generated Test Bills"], errors="coerce").fillna(0).sum()),
        "Cashier Generated Test Bills": float(unique_test_generated_lookup.sum()),
        "Unique bill handover to manager": float(pd.to_numeric(display_df["Unique bill handover to manager"], errors="coerce").fillna(0).sum()),
        "Missing test bill": float(pd.to_numeric(display_df["Missing test bill"], errors="coerce").fillna(0).sum()),
    }
    totals["Not Generated Bill %"] = (
        (totals["Cashier Not Generated Test Bills"] / totals["Cashier Login"]) * 100 if totals["Cashier Login"] > 0 else 0
    )
    totals["Generated Test Bill %"] = (
        (totals["Cashier Generated Test Bills"] / totals["Cashier Login"]) * 100 if totals["Cashier Login"] > 0 else 0
    )
    totals["Bill handover %"] = (
        (totals["Unique bill handover to manager"] / totals["Cashier Generated Test Bills"]) * 100 if totals.get("Cashier Generated Test Bills", 0) > 0 else 0
    )

    total_row: list[str] = []
    for col in headers:
        if col == "Shop Code":
            total_row.append("TOTAL")
        elif col == "Shop Name":
            total_row.append("ALL SHOPS")
        elif col in pct_cols:
            total_row.append(f"{totals[col]:.2f}%")
        else:
            total_row.append(f"{int(totals[col]):,}")

    header_classes = {
        idx: "pct-header" for idx, col in enumerate(headers) if col in pct_cols
    }

    render_sortable_html_table(
        title="",
        headers=headers,
        rows=rows,
        sort_values=sort_rows,
        table_id=f"shopwise_test_bill_{start_date}_{end_date}",
        show_title=False,
        title_class="section-title",
        min_width_px=1420,
        header_classes=header_classes,
        left_align_cols={0, 1},
        cell_styles=cell_styles,
        cell_titles=cell_titles,
        row_link_values=[str(v) for v in display_df["Shop Code"].fillna("")],
        row_link_param="tb_shop",
        row_link_extra_params={"tb_start": str(start_date), "tb_end": str(end_date)},
        enable_download_hover=True,
        download_file_name=f"shopwise_test_bill_analysis_{start_date}_{end_date}.csv",
        total_row=total_row,
        wrapper_max_height_px=460,
        table_width="100%",
        table_layout="fixed",
        overflow_x="hidden",
        header_white_space="normal",
        header_word_break="break-word",
        cell_white_space="nowrap",
        cell_word_break="normal",
        text_overflow="ellipsis",
        header_font_size="10px",
        cell_font_size="11px",
        header_padding="8px 6px",
        cell_padding="7px 6px",
    )
    dropdown_shop_options = [""] + sorted(display_df["Shop Code"].dropna().astype(str).unique().tolist())
    st.markdown(
        "<div style='color:#FFFFFF;font-weight:600;margin:0 0 4px 0;'>Select Shop Code For Details</div>",
        unsafe_allow_html=True,
    )
    selected_dropdown_shop = st.selectbox(
        "Select Shop Code For Details",
        options=dropdown_shop_options,
        index=0,
        key=f"shopwise_test_bill_dropdown_{start_date}_{end_date}{key_suffix}",
        label_visibility="collapsed",
    )

    if selected_dropdown_shop:
        render_shopwise_test_bill_drilldown_table(
            selected_shop=str(selected_dropdown_shop).strip().upper(),
            start_date=start_date,
            end_date=end_date,
            shop_name_map=shop_name_map,
            title_prefix="Shop Code Drilldown",
            table_key_prefix="shopwise_test_bill_dropdown_drilldown",
        )

    # ---- Drilldown: read shop + dates from URL query params set by JS row-click ----
    def _qp_scalar(key: str, default: str = "") -> str:
        try:
            val = st.query_params.get(key, default)
        except Exception:
            return default
        if isinstance(val, list):
            if not val:
                return default
            val = val[0]
        return str(val or default).strip()

    _tb_shop = _qp_scalar("tb_shop", "").upper()
    _tb_start_str = _qp_scalar("tb_start", "")
    _tb_end_str = _qp_scalar("tb_end", "")

    try:
        dd_start = datetime.strptime(_tb_start_str, "%Y-%m-%d").date() if _tb_start_str else start_date
    except ValueError:
        dd_start = start_date
    try:
        dd_end = datetime.strptime(_tb_end_str, "%Y-%m-%d").date() if _tb_end_str else end_date
    except ValueError:
        dd_end = end_date

    selected_shop = _tb_shop

    if not selected_shop:
        return

    # Back button clears the query params and reruns cleanly
    if st.button("← Back to summary", key="tb_drilldown_back_btn"):
        try:
            for _p in ("tb_shop", "tb_start", "tb_end"):
                if _p in st.query_params:
                    del st.query_params[_p]
        except Exception:
            pass
        st.rerun()

    render_shopwise_test_bill_drilldown_table(
        selected_shop=selected_shop,
        start_date=dd_start,
        end_date=dd_end,
        shop_name_map=shop_name_map,
        title_prefix="Drilldown",
        table_key_prefix="shopwise_test_bill_drilldown",
    )


def to_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    shop_name_map = load_shop_name_map()
    out["shop_name"] = out["shop_code"].map(shop_name_map).fillna(out["shop_code"])

    out = out[
        [
            "shop_code", "shop_name",
            "erp_nob", "scan_nob", "consumable_till_nob", "test_bills", "test_bill_not_generated", "total_accounted_bills", "bills_not_scanned", "bill_pct",
            "erp_erp_nob_ghs", "scanned_nob_ghs", "consumable_nob_ghs", "diff_ghs", "diff_pct",
            "test_bill_scanned", "bill_date_mismatched", "duplicate", "wrong_shop", "high_value_bill",
        ]
    ]

    out.columns = [
        "Shop Code",
        "Shop Name",
        "ERP NOB",
        "Scan NOB",
        "Consumable Till NOB",
        "ERP Test Bills",
        "Cashier Not Generated Test Bills",
        "Total Accounted Bills",
        "Bills not scanned",
        "Scanned bill %",
        "ERP ERP NOB (GHS)",
        "Scanned NOB (GHS)",
        "Consumable NOB (GHS)",
        "Diff (GHS)",
        "Scanned GHS %",
        "Test Bill Scanned",
        "Bill Date Mismatched",
        "Duplicate",
        "Wrong Shop",
        "High Value Bill",
    ]
    return out


def get_column_tooltip_config() -> dict:
    return {
        "Shop Code": st.column_config.TextColumn("Shop Code", help="Store code"),
        "Shop Name": st.column_config.TextColumn("Shop Name", help="Store name"),
        "ERP NOB": st.column_config.NumberColumn("ERP NOB", help="Count of bills from `erpdata.invno`"),
        "Scan NOB": st.column_config.NumberColumn("Scan NOB", help="Count of bills from scanned invoices (excluding mapped consumable tills)"),
        "Consumable Till NOB": st.column_config.NumberColumn("Consumable Till NOB", help="Count of scanned bills where till is in consumable till mapping"),
        "ERP Test Bills": st.column_config.NumberColumn(
            "ERP Test Bills",
            help="Count of ERP invoices where amount = 0.01 (each invoice counted once)",
        ),
        "Cashier Not Generated Test Bills": st.column_config.NumberColumn(
            "Cashier Not Generated Test Bills",
            help="Count of cashier+till sessions that do NOT have any invoice with amt = 0.01",
        ),
        "Cashier Generated Test Bills": st.column_config.NumberColumn(
            "Cashier Generated Test Bills",
            help="Count of invoices (erpdata.invno) where amt = 0.01 (each invoice counted once)",
        ),
        "Total Accounted Bills": st.column_config.NumberColumn("Total Accounted Bills", help="Scan NOB + Consumable Till NOB + ERP Test Bills"),
        "Bills not scanned": st.column_config.NumberColumn("Bills not scanned", help="ERP NOB - Total Accounted Bills"),
        "Scanned bill %": st.column_config.NumberColumn("Scanned bill %", help="(Scan NOB / ERP NOB) × 100", format="%.2f%%"),
        "ERP ERP NOB (GHS)": st.column_config.NumberColumn("ERP ERP NOB (GHS)", help="Sum of `erpdata.amt`", format="%.0f"),
        "Scanned NOB (GHS)": st.column_config.NumberColumn("Scanned NOB (GHS)", help="Sum of scanned invoice amount excluding consumable tills", format="%.0f"),
        "Consumable NOB (GHS)": st.column_config.NumberColumn("Consumable NOB (GHS)", help="Sum of scanned invoice amount from consumable tills", format="%.2f"),
        "Diff (GHS)": st.column_config.NumberColumn("Diff (GHS)", help="ERP ERP NOB (GHS) - (Scanned NOB (GHS) + Consumable NOB (GHS))", format="%.0f"),
        "Scanned GHS %": st.column_config.NumberColumn("Scanned GHS %", help="(Scanned NOB (GHS) / ERP ERP NOB (GHS)) × 100", format="%.2f%%"),
        "Test Bill Scanned": st.column_config.NumberColumn("Test Bill Scanned", help="Alert count where `a_type = 'test bill'`"),
        "Bill Date Mismatched": st.column_config.NumberColumn("Bill Date Mismatched", help="Alert count where `a_type` indicates bill date mismatch"),
        "Duplicate": st.column_config.NumberColumn("Duplicate", help="Alert count where `a_type` is 'Duplicate' or 'Duplicate Bill'"),
        "Wrong Shop": st.column_config.NumberColumn("Wrong Shop", help="Alert count where `a_type = 'Wrong Sho' or 'Wrong Shop'`"),
        "High Value Bill": st.column_config.NumberColumn("High Value Bill", help="Alert count where `a_type = 'High Bill Amount'`"),
    }


def get_column_help_text() -> dict:
    return {
        "Shop Code": "Store code",
        "Shop Name": "Store name",
        "ERP NOB": "Count of bills from erpdata.invno",
        "Scan NOB": "Count of bills from scanned invoices (excluding mapped consumable tills)",
        "Consumable Till NOB": "Count of scanned bills where till is in consumable till mapping",
        "ERP Test Bills": "Count of ERP invoices where amount = 0.01 (each invoice counted once)",
        "Cashier Not Generated Test Bills": "Count of cashier+till sessions that do NOT have any invoice with amt = 0.01",
        "Cashier Generated Test Bills": "Count of invoices (erpdata.invno) where amt = 0.01 (each invoice counted once)",
        "Total Accounted Bills": "Scan NOB + Consumable Till NOB + ERP Test Bills",
        "Bills not scanned": "ERP NOB - Total Accounted Bills",
        "Scanned bill %": "(Scan NOB / ERP NOB) × 100",
        "ERP ERP NOB (GHS)": "Sum of erpdata.amt",
        "Scanned NOB (GHS)": "Sum of scanned invoice amount excluding consumable tills",
        "Consumable NOB (GHS)": "Sum of scanned invoice amount from consumable tills",
        "Diff (GHS)": "ERP ERP NOB (GHS) - [Scanned NOB (GHS) + Consumable NOB (GHS)]",
        "Scanned GHS %": "(Scanned NOB (GHS) / ERP ERP NOB (GHS)) × 100",
        "Test Bill Scanned": "Alert count where a_type = 'test bill'",
        "Unique bill handover to manager": "Count of distinct invoices (invno) in invoices_manager for the selected period and shop",
        "Bill Date Mismatched": "Alert count where a_type indicates bill date mismatch",
        "Duplicate": "Alert count where a_type is Duplicate/Duplicate Bill",
        "Wrong Shop": "Alert count where a_type is Wrong Shop/Wrong Sho",
        "High Value Bill": "Alert count where a_type = 'High Bill Amount'",
    }


def build_cell_tooltips(display_df: pd.DataFrame) -> pd.DataFrame:
    tips = pd.DataFrame(pd.NA, index=display_df.index, columns=display_df.columns)
    col_erp_test = "ERP Test Bills" if "ERP Test Bills" in display_df.columns else "Test Bills"

    for idx, row in display_df.iterrows():
        scan_nob = int(row["Scan NOB"])
        consumable_nob = int(row["Consumable Till NOB"])
        test_bills = int(row[col_erp_test])
        total_accounted = int(row["Total Accounted Bills"])
        erp_nob = int(row["ERP NOB"])
        not_scanned = int(row["Bills not scanned"])
        bill_pct = float(row["Bill %"])

        erp_ghs = float(row["ERP ERP NOB (GHS)"])
        scanned_ghs = float(row["Scanned NOB (GHS)"])
        consumable_ghs = float(row["Consumable NOB (GHS)"])
        diff_ghs = float(row["Diff (GHS)"])
        diff_pct = float(row["Diff in %"])

        tips.at[idx, "Total Accounted Bills"] = (
            f"Scan NOB ({scan_nob:,}) + Consumable Till NOB ({consumable_nob:,}) + ERP Test Bills ({test_bills:,}) = {total_accounted:,}"
        )
        tips.at[idx, "Bills not scanned"] = (
            f"ERP NOB ({erp_nob:,}) - Total Accounted Bills ({total_accounted:,}) = {not_scanned:,}"
        )
        tips.at[idx, "Bill %"] = (
            f"(Scan NOB ({scan_nob:,}) / ERP NOB ({erp_nob:,})) × 100 = {bill_pct:.2f}%"
            if erp_nob > 0 else "ERP NOB is 0, so Bill % = 0"
        )
        tips.at[idx, "Diff (GHS)"] = (
            f"ERP ERP NOB ({erp_ghs:,.2f}) - [Scanned NOB ({scanned_ghs:,.2f}) + Consumable NOB ({consumable_ghs:,.2f})] = {diff_ghs:,.2f}"
        )
        tips.at[idx, "Diff in %"] = (
            f"(Scanned NOB (GHS) ({scanned_ghs:,.2f}) / ERP ERP NOB ({erp_ghs:,.2f})) × 100 = {diff_pct:.2f}%"
            if erp_ghs != 0 else "ERP ERP NOB (GHS) is 0, so Diff in % = 0"
        )

    return tips


def _sort_token(value) -> str:
        if value is None:
                return ""
        try:
                if pd.isna(value):
                        return ""
        except Exception:
                pass

        if isinstance(value, bool):
                return "1" if value else "0"
        if isinstance(value, int):
                return str(value)
        if isinstance(value, float):
                if pd.isna(value):
                        return ""
                return f"{value:.12f}"
        if isinstance(value, (datetime, pd.Timestamp)):
                return value.isoformat()
        return str(value)


def render_sortable_html_table(
        title: str,
        headers: list[str],
        rows: list[list[str]],
        sort_values: list[list[str]],
        table_id: str,
        min_width_px: int = 1120,
        show_title: bool = True,
        title_class: str = "section-title",
        title_style: str = "",
        header_titles: list[str] | None = None,
        header_classes: dict[int, str] | None = None,
        left_align_cols: set[int] | None = None,
        cell_styles: dict[tuple[int, int], str] | None = None,
        cell_titles: dict[tuple[int, int], str] | None = None,
        cell_link_hrefs: dict[tuple[int, int], str] | None = None,
        row_link_values: list[str] | None = None,
        row_link_param: str = "detail_shop",
        row_link_extra_params: dict | None = None,
        enable_download_hover: bool = False,
        download_file_name: str = "table_export.csv",
        excel_base64: str | None = None,
        total_row: list[str] | None = None,
        wrapper_max_height_px: int = 620,
        table_width: str = "100%",
        table_layout: str = "fixed",
        overflow_x: str = "auto",
        header_white_space: str = "normal",
        header_word_break: str = "break-word",
        cell_white_space: str = "nowrap",
        cell_word_break: str = "normal",
        text_overflow: str = "ellipsis",
        header_font_size: str = "10.5px",
        cell_font_size: str = "11.5px",
        header_padding: str = "8px 6px",
        cell_padding: str = "7px 6px",
        extra_css: str = "",
) -> None:
        if not rows:
                st.info("No rows to display.")
                return

        header_titles = header_titles or [""] * len(headers)
        header_classes = header_classes or {}
        left_align_cols = left_align_cols or set()
        cell_styles = cell_styles or {}
        cell_titles = cell_titles or {}
        cell_link_hrefs = cell_link_hrefs or {}

        header_html = "".join(
                (
                        f"<th data-col='{idx}' class='{html.escape(header_classes.get(idx, ''))}'"
                        f" title='{html.escape(str(header_titles[idx] if idx < len(header_titles) else ''))}'>"
                        f"{html.escape(str(h))} ⬍</th>"
                )
                for idx, h in enumerate(headers)
        )

        body_parts = []
        for row_idx, row in enumerate(rows):
                row_key = ""
                if row_link_values and row_idx < len(row_link_values):
                        row_key = str(row_link_values[row_idx] or "").strip()

                # Build anchor href once per row (used inside every cell)
                row_nav_href = ""
                if row_key:
                        _nav_params: dict = {row_link_param: row_key}
                        if row_link_extra_params:
                                _nav_params.update({k: str(v) for k, v in row_link_extra_params.items()})
                        row_nav_href = "?" + urlencode(_nav_params)

                row_class = "owner-clickable-row" if row_nav_href else ""
                row_attr = f" class='{row_class}'" if row_class else ""

                cells = []
                for col_idx, val in enumerate(row):
                        sval = ""
                        if row_idx < len(sort_values) and col_idx < len(sort_values[row_idx]):
                                sval = str(sort_values[row_idx][col_idx])

                        style_txt = cell_styles.get((row_idx, col_idx), "")
                        style_attr = f" style='{style_txt}'" if style_txt else ""

                        title_txt = cell_titles.get((row_idx, col_idx), "")
                        title_attr = f" title='{html.escape(title_txt)}'" if title_txt else ""

                        cell_class = "left" if col_idx in left_align_cols else ""
                        class_attr = f" class='{cell_class}'" if cell_class else ""

                        cell_content = html.escape(str(val))
                        cell_nav_href = cell_link_hrefs.get((row_idx, col_idx), "")
                        if cell_nav_href:
                            cell_content = (
                                f"<a class='row-nav-link cell-nav-link' href='{html.escape(cell_nav_href)}'"
                                f" target='_parent'>{cell_content}</a>"
                            )
                        elif row_nav_href:
                                cell_content = (
                                        f"<a class='row-nav-link' href='{html.escape(row_nav_href)}'"
                                        f" target='_parent'>{cell_content}</a>"
                                )

                        cells.append(
                                f"<td data-sort='{html.escape(sval)}'{class_attr}{title_attr}{style_attr}>{cell_content}</td>"
                        )
                body_parts.append("<tr" + row_attr + ">" + "".join(cells) + "</tr>")

        body_html = "".join(body_parts)

        tfoot_html = ""
        if total_row is not None:
                tfoot_cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in total_row)
                tfoot_html = f"<tfoot><tr>{tfoot_cells}</tr></tfoot>"

        _extra_params_js = json.dumps(row_link_extra_params or {})

        if show_title and title:
                style_attr = f" style='{title_style}'" if title_style else ""
                st.markdown(
                        f"<div class='{title_class}'{style_attr}>{title}</div>",
                        unsafe_allow_html=True,
                )

        download_button_html = ""
        if enable_download_hover:
                download_button_html = (
                        f"<button id='{table_id}-download' class='table-download-btn' title='Download'>⭳</button>"
                )

        table_html = f"""
        <style>
            .owner-table-shell-{table_id} {{
                position: relative;
                width: 100%;
            }}
            .owner-table-shell-{table_id} .table-download-btn {{
                position: absolute;
                top: 8px;
                right: 10px;
                z-index: 20;
                border: 1px solid rgba(91, 84, 255, 0.42);
                background: linear-gradient(135deg, rgba(30, 58, 138, 0.9) 0%, rgba(91, 84, 255, 0.85) 100%);
                color: #eef2ff;
                border-radius: 8px;
                font-size: 12px;
                line-height: 1;
                padding: 6px 8px;
                cursor: pointer;
                opacity: 0;
                pointer-events: none;
                transition: opacity .18s ease;
            }}
            .owner-table-shell-{table_id}:hover .table-download-btn {{
                opacity: 1;
                pointer-events: auto;
            }}
            .owner-table-shell-{table_id} .table-download-btn:focus {{
                opacity: 1;
                pointer-events: auto;
                outline: 2px solid #a78bfa;
                outline-offset: 1px;
            }}
            #{table_id}-wrap {{
                border: 1px solid rgba(91, 84, 255, 0.36);
                border-radius: 14px;
                overflow-y: auto;
                overflow-x: {overflow_x};
                max-height: {wrapper_max_height_px}px;
                background: linear-gradient(145deg, #0f1b3d 0%, #0a132b 55%, #081126 100%);
                box-shadow: 0 10px 24px rgba(8, 15, 34, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.04);
            }}
            #{table_id} {{
                width: {table_width};
                border-collapse: collapse;
                table-layout: {table_layout};
                min-width: {min_width_px}px;
                font-family: Inter, Segoe UI, sans-serif;
            }}
            #{table_id} th {{
                background: linear-gradient(135deg, rgba(30, 58, 138, 0.82) 0%, rgba(91, 84, 255, 0.66) 100%);
                color: #e8efff;
                font-size: {header_font_size};
                text-transform: uppercase;
                letter-spacing: 0.04em;
                text-align: center;
                padding: {header_padding};
                border-bottom: 1px solid rgba(99, 102, 241, 0.45);
                white-space: {header_white_space};
                word-break: {header_word_break};
                overflow: hidden;
                text-overflow: {text_overflow};
                cursor: pointer;
                user-select: none;
                position: sticky;
                top: 0;
                z-index: 6;
            }}
            #{table_id} th.pct-header {{
                background: linear-gradient(135deg, rgba(124, 58, 237, 0.78) 0%, rgba(217, 70, 239, 0.5) 100%);
                color: #f5d0fe;
                font-weight: 800;
            }}
            #{table_id} td {{
                color: #e2e8f0;
                font-size: {cell_font_size};
                text-align: center;
                padding: {cell_padding};
                border-bottom: 1px solid rgba(71, 85, 105, 0.34);
                white-space: {cell_white_space};
                word-break: {cell_word_break};
                overflow: hidden;
                text-overflow: {text_overflow};
            }}
            #{table_id} td.left {{
                text-align: left;
                font-weight: 600;
            }}
            #{table_id} tbody tr:nth-child(even) {{
                background: rgba(148, 163, 184, 0.04);
            }}
            #{table_id} tbody tr:hover {{
                background: rgba(91, 84, 255, 0.16);
            }}
            #{table_id} tr.owner-clickable-row td {{
                cursor: pointer;
            }}
            #{table_id} tr.owner-clickable-row:hover td {{
                background: rgba(59, 130, 246, 0.18);
            }}
            #{table_id} .row-nav-link {{
                color: inherit;
                text-decoration: none;
                display: block;
                width: 100%;
            }}
            #{table_id} .cell-nav-link {{
                text-decoration: underline;
                text-decoration-color: rgba(191, 219, 254, 0.55);
                text-underline-offset: 2px;
            }}
            #{table_id} tfoot td {{
                background: rgba(15, 30, 80, 0.95);
                color: #f8fafc;
                font-weight: 800;
                border-top: 2px solid rgba(139, 92, 246, 0.6);
                position: sticky;
                bottom: 0;
                z-index: 5;
            }}
            {extra_css}
        </style>

        <div class='owner-table-shell-{table_id}'>
            {download_button_html}
            <div id='{table_id}-wrap'>
                <table id='{table_id}'>
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
                    const txt = String(raw ?? '').trim();
                    const normalized = txt.replace(/,/g, '');
                    const n = Number(normalized);
                    if (normalized !== '' && !Number.isNaN(n)) {{
                        return {{ t: 'n', v: n }};
                    }}
                    return {{ t: 's', v: txt.toLowerCase() }};
                }};

                headers.forEach((th, idx) => {{
                    th.addEventListener('click', () => {{
                        const rows = Array.from(tbody.querySelectorAll('tr'));
                        const asc = sortState.col === idx ? !sortState.asc : true;

                        rows.sort((a, b) => {{
                            const aCell = a.children[idx];
                            const bCell = b.children[idx];
                            const av = parseVal(aCell ? aCell.getAttribute('data-sort') : '');
                            const bv = parseVal(bCell ? bCell.getAttribute('data-sort') : '');

                            if (av.t === 'n' && bv.t === 'n') {{
                                return asc ? av.v - bv.v : bv.v - av.v;
                            }}
                            if (av.v < bv.v) return asc ? -1 : 1;
                            if (av.v > bv.v) return asc ? 1 : -1;
                            return 0;
                        }});

                        rows.forEach(r => tbody.appendChild(r));
                        sortState = {{ col: idx, asc }};
                        headers.forEach((h, i) => {{
                            const baseLabel = h.innerText.replace('⬍', '').replace('↑', '').replace('↓', '').trim();
                            h.innerText = i === idx ? `${{baseLabel}} ${{asc ? '↑' : '↓'}}` : `${{baseLabel}} ⬍`;
                        }});
                    }});
                }});

                // Fix row-nav-link hrefs: srcdoc iframes resolve relative URLs against 'about:blank'.
                // Rewrite each link's href to an absolute URL based on the parent's current location.
                try {{
                    const parentBase = window.parent.location.href;
                    table.querySelectorAll('a.row-nav-link').forEach(function(a) {{
                        const rel = a.getAttribute('href') || '';
                        if (rel && !rel.match(/^https?:\/\//)) {{
                            try {{
                                a.href = new URL(rel, parentBase).toString();
                            }} catch (_e) {{}}
                        }}
                    }});
                }} catch (_e) {{}}

                // Row navigation handled by <a target="_parent"> anchors in each cell — no JS needed.

                const downloadBtn = document.getElementById('{table_id}-download');
                if (downloadBtn) {{
                    downloadBtn.addEventListener('click', (evt) => {{
                        evt.stopPropagation();
                        const link = document.createElement('a');
                        const excelB64 = '{excel_base64 or ""}';
                        if (excelB64) {{
                            link.href = `data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,${{excelB64}}`;
                        }} else {{
                            const csvEscape = (v) => `"${{String(v ?? '').replace(/"/g, '""')}}"`;
                            const headerVals = Array.from(table.querySelectorAll('thead th')).map(th => th.innerText.replace('⬍', '').replace('↑', '').replace('↓', '').trim());
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

        extra = 38 if total_row is not None else 0
        # Keep iframe height close to visible wrapper height to avoid large blank gaps between sections.
        estimated_body = min(wrapper_max_height_px, max(120, len(rows) * 30))
        viewport_height = min(wrapper_max_height_px + 120, max(220, 76 + estimated_body + extra))
        components.html(table_html, height=viewport_height, scrolling=False)




def render_summary_html_table(
    display_df: pd.DataFrame,
    new_shop_codes: set[str] | None = None,
):
    new_shop_codes = {s.strip().upper() for s in (new_shop_codes or set())}
    col_help = get_column_help_text()

    def _resolve_col(*names: str) -> str:
        for name in names:
            if name in display_df.columns:
                return name
        return names[0]

    col_total_accounted = _resolve_col("Total Accounted Bills", "Total Accounted(Scanned & Test Bills)")
    col_bill_pct = _resolve_col("Scanned bill %", "Bill %", "Scanned Bill %")
    col_diff_pct = _resolve_col("Scanned GHS %", "Diff in %")
    col_erp_test = _resolve_col("ERP Test Bills", "Test Bills")
    col_no_test = _resolve_col("Cashier Not Generated Test Bills", "Test bill not generated")

    calc_cols = {col_total_accounted, "Bills not scanned", col_bill_pct, "Diff (GHS)", col_diff_pct}
    compact_header_map = {
        "Shop Code": "Shop",
        "Shop Name": "Name",
        "ERP NOB": "ERP NOB",
        "Scan NOB": "Scan NOB",
        "Consumable Till NOB": "Consumable Till NOB",
        "ERP Test Bills": "ERP Test Bills",
        "Test Bills": "ERP Test Bills",
        "Cashier Not Generated Test Bills": "No Test bill",
        "Test bill not generated": "No Test bill",
        "Total Accounted Bills": "Accounted Bills",
        "Total Accounted(Scanned & Test Bills)": "Accounted Bills",
        "Bills not scanned": "Not Scanned",
        "Bill %": "Scanned bill %",
        "Scanned Bill %": "Scanned bill %",
        "Scanned bill %": "Scanned bill %",
        "ERP ERP NOB (GHS)": "ERP GHS",
        "Scanned NOB (GHS)": "Scanned GHS",
        "Consumable NOB (GHS)": "Consumable GHS",
        "Diff (GHS)": "Diff GHS",
        "Diff in %": "Scanned GHS %",
        "Scanned GHS %": "Scanned GHS %",
        "Test Bill Scanned": "Test Alert",
        "Bill Date Mismatched": "Date Mismatch",
        "Duplicate": "Duplicate",
        "Wrong Shop": "Wrong Shop",
        "High Value Bill": "High Value",
    }

    def fmt_val(col: str, val):
        if col in {col_bill_pct, col_diff_pct}:
            return f"{float(val):.2f}%"
        if col in {"ERP ERP NOB (GHS)", "Scanned NOB (GHS)", "Diff (GHS)"}:
            return f"{float(val):,.0f}"
        if col in {"Consumable NOB (GHS)"}:
            return f"{float(val):,.2f}"
        if isinstance(val, (int, float)) and col not in {"Shop Code", "Shop Name"}:
            try:
                return f"{int(val):,}"
            except Exception:
                return str(val)
        return str(val)

    headers = [compact_header_map.get(col, col) for col in display_df.columns]
    rows = []
    sort_rows = []
    cell_styles: dict[tuple[int, int], str] = {}
    cell_titles: dict[tuple[int, int], str] = {}

    def bill_pct_style(val: float) -> str:
        if val > 100:
            return "background: rgba(125, 211, 252, 0.25); color: #e0f2fe; font-weight: 700;"
        if val < 70:
            return "background: rgba(248, 113, 113, 0.18); color: #fecaca; font-weight: 600;"
        if val < 90:
            return "background: rgba(250, 204, 21, 0.18); color: #fef9c3; font-weight: 600;"
        return "background: rgba(74, 222, 128, 0.18); color: #dcfce7; font-weight: 600;"

    def diff_pct_style(val: float) -> str:
        if val > 100:
            return "background: rgba(125, 211, 252, 0.22); color: #e0f2fe; font-weight: 700;"
        if val < 70:
            return "background: rgba(248, 113, 113, 0.18); color: #fecaca; font-weight: 600;"
        if val < 90:
            return "background: rgba(250, 204, 21, 0.18); color: #fef9c3; font-weight: 600;"
        return "background: rgba(74, 222, 128, 0.18); color: #dcfce7; font-weight: 600;"

    for row_idx, (_, row) in enumerate(display_df.iterrows()):
        is_new_shop = str(row["Shop Code"]).strip().upper() in new_shop_codes
        t_scan = int(row["Scan NOB"])
        t_cons = int(row["Consumable Till NOB"])
        t_test = int(row[col_erp_test])
        t_total = int(row[col_total_accounted])
        t_erp = int(row["ERP NOB"])
        t_not = int(row["Bills not scanned"])
        t_bill_pct = float(row[col_bill_pct])

        t_erp_ghs = float(row["ERP ERP NOB (GHS)"])
        t_scanned_ghs = float(row["Scanned NOB (GHS)"])
        t_cons_ghs = float(row["Consumable NOB (GHS)"])
        t_diff_ghs = float(row["Diff (GHS)"])
        t_diff_pct = float(row[col_diff_pct])

        calc_tip_map = {
            col_total_accounted: f"Scan NOB ({t_scan:,}) + Consumable Till NOB ({t_cons:,}) + ERP Test Bills ({t_test:,}) = {t_total:,}",
            "Bills not scanned": f"ERP NOB ({t_erp:,}) - Total Accounted Bills ({t_total:,}) = {t_not:,}",
            col_bill_pct: f"(Scan NOB ({t_scan:,}) / ERP NOB ({t_erp:,})) × 100 = {t_bill_pct:.2f}%" if t_erp > 0 else "ERP NOB is 0, so Bill % = 0",
            "Diff (GHS)": f"ERP ERP NOB ({t_erp_ghs:,.2f}) - [Scanned NOB ({t_scanned_ghs:,.2f}) + Consumable NOB ({t_cons_ghs:,.2f})] = {t_diff_ghs:,.2f}",
            col_diff_pct: f"(Scanned NOB (GHS) ({t_scanned_ghs:,.2f}) / ERP ERP NOB ({t_erp_ghs:,.2f})) × 100 = {t_diff_pct:.2f}%" if t_erp_ghs != 0 else "ERP ERP NOB (GHS) is 0, so Diff in % = 0",
        }

        display_row = []
        sort_row = []
        for col_idx, col in enumerate(display_df.columns):
            val = fmt_val(col, row[col])
            if col == "Shop Code" and is_new_shop:
                val = f"{val} 🆕"
            title = calc_tip_map[col] if col in calc_cols else ""
            style_attr = ""
            if col == col_bill_pct:
                style_attr = f' style="{bill_pct_style(float(row[col]))}"'
            elif col == col_diff_pct:
                style_attr = f' style="{diff_pct_style(float(row[col]))}"'
            elif col == "Shop Code" and is_new_shop:
                style_attr = ' style="background: rgba(34, 197, 94, 0.20); color: #86efac; font-weight: 800;"'
            if title:
                cell_titles[(row_idx, col_idx)] = title
            if style_attr:
                cell_styles[(row_idx, col_idx)] = style_attr.replace(' style="', '').rstrip('"')
            display_row.append(val)
            sort_row.append(_sort_token(row[col]))

        rows.append(display_row)
        sort_rows.append(sort_row)

    header_titles = [col_help.get(col, "") for col in display_df.columns]
    header_classes = {
        idx: "pct-header" for idx, col in enumerate(display_df.columns) if col in {col_bill_pct, col_diff_pct}
    }

    # Bottom totals row: aggregate numeric columns and recompute percentage metrics at overall level.
    sum_cols = {
        "ERP NOB",
        "Scan NOB",
        "Consumable Till NOB",
        col_erp_test,
        col_no_test,
        col_total_accounted,
        "Bills not scanned",
        "ERP ERP NOB (GHS)",
        "Scanned NOB (GHS)",
        "Consumable NOB (GHS)",
        "Diff (GHS)",
        "Test Bill Scanned",
        "Bill Date Mismatched",
        "Duplicate",
        "Wrong Shop",
        "High Value Bill",
    }
    totals: dict[str, float] = {}
    for col in sum_cols:
        if col in display_df.columns:
            totals[col] = float(pd.to_numeric(display_df[col], errors="coerce").fillna(0).sum())

    erp_nob_total = totals.get("ERP NOB", 0.0)
    accounted_total = totals.get(col_total_accounted, 0.0)
    scan_nob_total = totals.get("Scan NOB", 0.0)
    erp_ghs_total = totals.get("ERP ERP NOB (GHS)", 0.0)
    scanned_ghs_total = totals.get("Scanned NOB (GHS)", 0.0)

    totals[col_bill_pct] = (scan_nob_total / erp_nob_total * 100.0) if erp_nob_total > 0 else 0.0
    totals[col_diff_pct] = (scanned_ghs_total / erp_ghs_total * 100.0) if erp_ghs_total > 0 else 0.0

    total_row: list[str] = []
    for col in display_df.columns:
        if col == "Shop Code":
            total_row.append("TOTAL")
        elif col == "Shop Name":
            total_row.append("ALL SHOPS")
        elif col in totals:
            total_row.append(fmt_val(col, totals[col]))
        else:
            total_row.append("")

    table_dom_id = f"summary_tbl_{abs(hash(tuple(display_df.columns)))}"
    render_sortable_html_table(
        title="",
        headers=headers,
        rows=rows,
        sort_values=sort_rows,
        table_id=table_dom_id,
        show_title=False,
        min_width_px=1420,
        header_titles=header_titles,
        header_classes=header_classes,
        left_align_cols={0, 1},
        cell_styles=cell_styles,
        cell_titles=cell_titles,
        enable_download_hover=True,
        download_file_name="invoice_summary_table.csv",
        total_row=total_row,
        wrapper_max_height_px=460,
        table_width="100%",
        table_layout="fixed",
        overflow_x="hidden",
        header_white_space="normal",
        header_word_break="break-word",
        text_overflow="ellipsis",
        header_font_size="10px",
        cell_font_size="11px",
        header_padding="8px 6px",
        cell_padding="7px 6px",
        extra_css=(
            f"#{table_dom_id} th:nth-child(2), #{table_dom_id} td:nth-child(2) {{"
            "width: clamp(140px, 18vw, 230px);"
            "max-width: 230px;"
            "white-space: nowrap;"
            "overflow: hidden;"
            "text-overflow: ellipsis;"
            "}}"
        ),
    )


def render_styled_html_table(
    display_df: pd.DataFrame,
    title: str = "",
    auto_content_width: bool = False,
    wrap_header_text: bool = False,
    wrap_cell_text: bool = False,
    compact_mode: bool = False,
    title_class: str = "section-title",
    align_left: bool = False,
    align_title_with_table: bool = False,
    header_font_size_override: str | None = None,
    body_font_size_override: str | None = None,
    body_padding_override: str | None = None,
    min_width_px_override: int | None = None,
    table_layout_override: str | None = None,
    overflow_x_override: str | None = None,
):
    if display_df is None or display_df.empty:
        return

    title_style = ""
    if align_title_with_table and auto_content_width and align_left:
        title_style = "text-align:left; width:fit-content; margin:0 0 0.75rem 0;"

    def fmt_val(col: str, val):
        if isinstance(val, (int, float)):
            if isinstance(val, float) and not float(val).is_integer():
                return f"{float(val):,.2f}"
            return f"{int(val):,}"
        return str(val)

    headers = [str(col) for col in display_df.columns]

    value_cols = [c for c in display_df.columns if c != "a_type"]
    all_vals = pd.to_numeric(display_df[value_cols].stack(), errors="coerce").fillna(0) if value_cols else pd.Series([0])
    min_count = int(all_vals.min()) if not all_vals.empty else 0
    max_count = int(all_vals.max()) if not all_vals.empty else 0
    mid_count = min_count + (max_count - min_count) * 0.5

    def _count_style(val: float) -> str:
        if max_count == min_count:
            return "background-color: rgba(250, 204, 21, 0.16); color: #fef9c3; font-weight: 600;"
        if val >= mid_count + (max_count - min_count) * 0.25:
            return "background-color: rgba(248, 113, 113, 0.16); color: #fecaca; font-weight: 600;"
        if val >= mid_count:
            return "background-color: rgba(250, 204, 21, 0.16); color: #fef9c3; font-weight: 600;"
        return "background-color: rgba(74, 222, 128, 0.16); color: #dcfce7; font-weight: 600;"

    rows = []
    sort_rows = []
    cell_styles: dict[tuple[int, int], str] = {}
    for row_idx, (_, row) in enumerate(display_df.iterrows()):
        display_row = []
        sort_row = []
        for col_idx, col in enumerate(display_df.columns):
            val = fmt_val(col, row[col])
            style_attr = ""
            if col in value_cols:
                try:
                    style_attr = _count_style(float(row[col]))
                except Exception:
                    style_attr = ""
            if style_attr:
                cell_styles[(row_idx, col_idx)] = style_attr
            display_row.append(val)
            sort_row.append(_sort_token(row[col]))
        rows.append(display_row)
        sort_rows.append(sort_row)

    viewport_height = min(420, max(220, 90 + len(display_df) * 34))
    inner_table_height = max(200, viewport_height - 22)

    wrapper_display = "block"
    table_width = "auto" if auto_content_width else "100%"
    table_layout = "auto" if auto_content_width else "fixed"
    overflow_x = "auto" if auto_content_width else "hidden"
    text_overflow = "clip" if auto_content_width else "ellipsis"
    wrapper_width = "fit-content" if auto_content_width else "100%"
    wrapper_margin = "0" if (auto_content_width and align_left) else ("0 auto" if auto_content_width else "0")
    header_white_space = "normal" if wrap_header_text else "nowrap"
    header_word_break = "break-word" if wrap_header_text else "normal"
    cell_white_space = "normal" if wrap_cell_text else "nowrap"
    cell_word_break = "break-word" if wrap_cell_text else "normal"
    header_font_size = "10px" if compact_mode else "10.5px"
    cell_font_size = "10.5px" if compact_mode else "11.5px"
    header_padding = "7px 6px" if compact_mode else "8px 6px"
    cell_padding = "6px 6px" if compact_mode else "7px 6px"

    if header_font_size_override:
        header_font_size = header_font_size_override
    if body_font_size_override:
        cell_font_size = body_font_size_override
    if body_padding_override:
        cell_padding = body_padding_override
    if table_layout_override:
        table_layout = table_layout_override
    if overflow_x_override:
        overflow_x = overflow_x_override

    table_dom_id = f"styled_owner_table_{abs(hash((title, tuple(display_df.columns), len(display_df))))}"

    download_file_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in (title or "table")).strip("_") or "table"
    download_file_name = f"{download_file_name}.csv"

    min_width_px = max(720, len(display_df.columns) * (120 if auto_content_width else 95))
    if min_width_px_override is not None:
        min_width_px = max(260, int(min_width_px_override))
    render_sortable_html_table(
        title=title,
        headers=headers,
        rows=rows,
        sort_values=sort_rows,
        table_id=table_dom_id,
        min_width_px=min_width_px,
        show_title=bool(title),
        title_class=title_class,
        title_style=title_style,
        left_align_cols={display_df.columns.get_loc("a_type")} if "a_type" in display_df.columns else set(),
        cell_styles=cell_styles,
        enable_download_hover=True,
        download_file_name=download_file_name,
        wrapper_max_height_px=inner_table_height,
        table_width=table_width,
        table_layout=table_layout,
        overflow_x=overflow_x,
        header_white_space=header_white_space,
        header_word_break=header_word_break,
        cell_white_space=cell_white_space,
        cell_word_break=cell_word_break,
        text_overflow=text_overflow,
        header_font_size=header_font_size,
        cell_font_size=cell_font_size,
        header_padding=header_padding,
        cell_padding=cell_padding,
    )


def _is_full_month_selection(start_date: date, end_date: date) -> bool:
    if start_date.year != end_date.year or start_date.month != end_date.month:
        return False
    last_day = calendar.monthrange(start_date.year, start_date.month)[1]
    return start_date.day == 1 and end_date.day == last_day


def _period_labels(start_date: date, end_date: date) -> tuple[str, str]:
    if _is_full_month_selection(start_date, end_date):
        month_label = start_date.strftime("%b'%y")
        return f"Below data is for • {month_label}", month_label
    if start_date == end_date:
        day_label = start_date.strftime("%d %b %Y")
        return f"Below data is for • {day_label}", day_label
    range_label = f"{start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}"
    compact_range = f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
    return f"Below data is for • {range_label}", compact_range


def render_kpis(df: pd.DataFrame, start_date: date, end_date: date):
    heading_label, period_compact = _period_labels(start_date, end_date)
    st.markdown(f"<div class='kpi-heading-center'>{heading_label}</div>", unsafe_allow_html=True)

    erp = int(df["erp_nob"].sum()) if not df.empty else 0
    accounted = int(df["total_accounted_bills"].sum()) if not df.empty else 0
    not_scanned = int(df["bills_not_scanned"].sum()) if not df.empty else 0
    bill_pct = round((accounted / erp) * 100, 2) if erp > 0 else 0.0

    total_diff = float(df["diff_ghs"].sum()) if not df.empty else 0.0
    total_alerts = int(df[["test_bill_scanned", "bill_date_mismatched", "duplicate", "wrong_shop", "high_value_bill"]].sum().sum()) if not df.empty else 0
    implemented_count = len(IMPLEMENTED_SHOPS_SET)
    total_shop_count = IMPLEMENTATION_TOTAL_SHOPS
    implementation_pct = round((implemented_count / total_shop_count) * 100, 2) if total_shop_count else 0.0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cards = [
        ("ERP NOB", f"{erp:,}", "Total ERP bills"),
        ("Accounted Bills", f"{accounted:,}", f"{bill_pct:.2f}% coverage"),
        ("Bills not scanned", f"{not_scanned:,}", "Needs follow-up"),
        (f"Diff (GHS) ({period_compact})", f"{total_diff:,.2f}", "ERP - (Scan + Consumable)"),
        ("Alerts", f"{total_alerts:,}", "Date range total"),
        ("Implementation %", f"{implementation_pct:.2f}%", f"{implemented_count}/{total_shop_count} shops"),
    ]

    for col, (label, value, sub) in zip([c1, c2, c3, c4, c5, c6], cards):
        with col:
            st.markdown(
                f"""
                <div class='kpi-card'>
                    <div class='kpi-label'>{label}</div>
                    <div class='kpi-value'>{value}</div>
                    <div class='kpi-sub'>{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


@st.cache_data(ttl=300)
def load_daily_diff_series(start_date: date, end_date: date) -> pd.DataFrame:
    query = f"""
    WITH calendar_days AS (
        SELECT generate_series(%(s)s::date, %(e)s::date, interval '1 day')::date AS bill_date
    ),
    erp_agg AS (
        SELECT bill_date, SUM(erp_nob_ghs) AS erp_ghs
        FROM mv_wh_erp_daily
        WHERE bill_date BETWEEN %(s)s AND %(e)s
          AND shop_code IS NOT NULL
          AND TRIM(shop_code) <> ''
          AND shop_code NOT IN ('G01','G02','G03','G04','G05','G06','G07','G08','G09','G10','G11','G12','INV','SPX','SEL')
        GROUP BY bill_date
    ),
    {_build_live_scan_agg_cte(['bill_date'], "WHERE shop_code IS NOT NULL AND TRIM(shop_code) <> '' AND shop_code NOT IN ('G01','G02','G03','G04','G05','G06','G07','G08','G09','G10','G11','G12','INV','SPX','SEL')").rstrip().rstrip(',')}
    SELECT
        d.bill_date,
        COALESCE(e.erp_ghs, 0)                                                       AS erp_ghs,
        COALESCE(s.scanned_nob_ghs, 0)                                               AS scanned_ghs,
        COALESCE(s.consumable_nob_ghs, 0)                                            AS consumable_ghs,
        (COALESCE(e.erp_ghs, 0) - (COALESCE(s.scanned_nob_ghs, 0) + COALESCE(s.consumable_nob_ghs, 0))) AS diff_ghs
    FROM calendar_days d
    LEFT JOIN erp_agg  e ON e.bill_date = d.bill_date
    LEFT JOIN scan_agg s ON s.bill_date = d.bill_date
    ORDER BY d.bill_date
    """

    with get_db_connection() as conn:
        return pd.read_sql(
            query,
            conn,
            params={"s": start_date, "e": end_date, "shops": sorted(list(IMPLEMENTED_SHOPS_SET))},
        )

def render_diff_trend_graphs(anchor_end_date: date):
    month_start = anchor_end_date.replace(day=1)
    month_window_start = (pd.Timestamp(month_start) - pd.DateOffset(months=5)).date()
    last_10_start = anchor_end_date - timedelta(days=9)
    current_week_start = anchor_end_date - timedelta(days=anchor_end_date.weekday())
    anchor_sunday = current_week_start - timedelta(days=1)
    last_4w_start = anchor_sunday - timedelta(days=20)
    series_start = min(month_window_start, last_10_start, last_4w_start)

    trend_df = load_daily_diff_series(series_start, anchor_end_date)
    if trend_df is None or trend_df.empty:
        return

    trend_df["bill_date"] = pd.to_datetime(trend_df["bill_date"]).dt.date
    trend_df["diff_ghs"] = pd.to_numeric(trend_df["diff_ghs"], errors="coerce").fillna(0.0)

    month_df = trend_df[(trend_df["bill_date"] >= month_window_start) & (trend_df["bill_date"] <= anchor_end_date)].copy()
    month_df["month"] = pd.to_datetime(month_df["bill_date"]).dt.to_period("M").astype(str)
    month_df = (
        month_df.groupby("month", as_index=False)["diff_ghs"]
        .sum()
        .sort_values("month")
    )
    month_df["month_label"] = pd.to_datetime(month_df["month"] + "-01").dt.strftime("%b %Y")
    last_10_df = trend_df[(trend_df["bill_date"] >= last_10_start) & (trend_df["bill_date"] <= anchor_end_date)].copy()
    last_10_df["day_label"] = pd.to_datetime(last_10_df["bill_date"]).dt.strftime("%d %b")

    rolling_rows = []
    for i in range(2, -1, -1):
        week_end = anchor_sunday - timedelta(days=7 * i)
        week_start = week_end - timedelta(days=6)
        week_sum = float(
            trend_df[(trend_df["bill_date"] >= week_start) & (trend_df["bill_date"] <= week_end)]["diff_ghs"].sum()
        )
        week_no = int(pd.Timestamp(week_start).isocalendar().week)
        rolling_rows.append({
            "window": f"{week_start.strftime('%d %b')} - {week_end.strftime('%d %b')}",
            "window_label": f"W{week_no}<br>{week_start.strftime('%d %b')}<br>{week_end.strftime('%d %b')}",
            "diff_ghs": week_sum,
        })

    current_week_sum = float(
        trend_df[(trend_df["bill_date"] >= current_week_start) & (trend_df["bill_date"] <= anchor_end_date)]["diff_ghs"].sum()
    )
    current_week_no = int(pd.Timestamp(current_week_start).isocalendar().week)
    rolling_rows.append({
        "window": f"{current_week_start.strftime('%d %b')} - {anchor_end_date.strftime('%d %b')} (Current)",
        "window_label": f"W{current_week_no}<br>{current_week_start.strftime('%d %b')}<br>{anchor_end_date.strftime('%d %b')}*",
        "diff_ghs": current_week_sum,
    })

    rolling_4w_df = pd.DataFrame(rolling_rows)

    return month_df, last_10_df, rolling_4w_df


def _apply_soft_graph_style(fig, height=320, xaxis_kwargs=None, yaxis_kwargs=None):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=48, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="#ffffff"),
        title_font=dict(color="#ffffff", size=14),
    )
    x_base = {
        "showgrid": False,
        "tickfont": dict(color="#ffffff", size=10),
        "title_font": dict(color="#ffffff", size=11),
        "automargin": True,
        "tickangle": -30,
    }
    y_base = {
        "gridcolor": "rgba(148,163,184,0.18)",
        "zeroline": False,
        "tickfont": dict(color="#ffffff", size=10),
        "title_font": dict(color="#ffffff", size=11),
        "automargin": True,
    }
    if xaxis_kwargs:
        x_base.update(xaxis_kwargs)
    if yaxis_kwargs:
        y_base.update(yaxis_kwargs)
    fig.update_xaxes(**x_base)
    fig.update_yaxes(**y_base)


def render_month_diff_graph(month_df: pd.DataFrame, anchor_end_date: date):
    if month_df is None or month_df.empty:
        st.info("No month-wise data.")
        return
    fig_month = px.bar(month_df, x="month_label", y="diff_ghs", title=f"Month-wise Diff (GHS) • up to {anchor_end_date.strftime('%d %b %Y')}")
    fig_month.update_traces(marker_color="#8b5cf6", marker_line_width=0, opacity=0.9)
    fig_month.update_layout(xaxis_title="Month", yaxis_title="Diff (GHS)")
    _apply_soft_graph_style(
        fig_month,
        height=300,
        xaxis_kwargs={
            "tickmode": "array",
            "tickvals": month_df["month_label"].tolist(),
            "ticktext": month_df["month_label"].tolist(),
            "tickangle": -35,
        },
    )
    st.plotly_chart(fig_month, use_container_width=True)


def render_last10_diff_graph(last_10_df: pd.DataFrame):
    if last_10_df is None or last_10_df.empty:
        st.info("No last-10-days data.")
        return
    fig_10d = px.bar(last_10_df, x="day_label", y="diff_ghs", title="Last 10 Days Diff (GHS)")
    fig_10d.update_traces(marker_color="#3b82f6", marker_line_width=0, opacity=0.9)
    fig_10d.update_layout(xaxis_title="Date", yaxis_title="Diff (GHS)")
    tickvals = last_10_df["day_label"].tolist()
    ticktext = tickvals
    _apply_soft_graph_style(
        fig_10d,
        height=300,
        xaxis_kwargs={
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": ticktext,
            "tickangle": -35,
        },
    )
    st.plotly_chart(fig_10d, use_container_width=True)


def render_weekly_diff_graph(rolling_4w_df: pd.DataFrame):
    if rolling_4w_df is None or rolling_4w_df.empty:
        return

    fig_4w = px.bar(rolling_4w_df, x="window_label", y="diff_ghs", title="Rolling Last 4 Weeks Diff (GHS) • Mon-Sun")
    fig_4w.update_traces(marker_color="#f59e0b", marker_line_width=0, opacity=0.9)
    fig_4w.update_layout(xaxis_title="Week (Monday - Sunday)", yaxis_title="Diff (GHS)")
    _apply_soft_graph_style(
        fig_4w,
        height=300,
        xaxis_kwargs={
            "tickmode": "array",
            "tickvals": rolling_4w_df["window_label"].tolist(),
            "ticktext": rolling_4w_df["window_label"].tolist(),
            "tickangle": 0,
        },
    )
    st.plotly_chart(fig_4w, use_container_width=True)


@st.cache_data(ttl=300)
def load_bill_handover_last10(end_date: date) -> pd.DataFrame:
    """Daily avg Bill Handover % across all shops for the last 10 days."""
    start_10 = end_date - timedelta(days=9)
    query = """
    WITH days AS (
        SELECT generate_series(%(s)s::date, %(e)s::date, '1 day'::interval)::date AS bill_date
    ),
    erp_sessions AS (
        SELECT
            e.invdate::date AS bill_date,
            UPPER(TRIM(e.store_code)) AS shop_code,
            UPPER(TRIM(COALESCE(e.cashier, ''))) AS cashier_name,
            NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '')::int AS till_no,
            MAX(CASE WHEN e.amt = 0.01 THEN 1 ELSE 0 END) AS has_test_bill
        FROM erpdata e
        WHERE e.invdate::date BETWEEN %(s)s AND %(e)s
          AND UPPER(TRIM(e.store_code)) = ANY(%(shops)s)
          AND NULLIF(TRIM(COALESCE(e.cashier, '')), '') IS NOT NULL
          AND NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '') IS NOT NULL
          AND NULLIF(TRIM(COALESCE(e.invno::text, '')), '') IS NOT NULL
        GROUP BY e.invdate::date, UPPER(TRIM(e.store_code)),
                 UPPER(TRIM(COALESCE(e.cashier, ''))),
                 NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '')::int
    ),
    unique_gen AS (
        SELECT bill_date, shop_code,
               COUNT(*) FILTER (WHERE has_test_bill = 1) AS unique_generated
        FROM erp_sessions
        GROUP BY bill_date, shop_code
    ),
    mgr_sessions AS (
        SELECT
            m.invdate::date AS bill_date,
            UPPER(TRIM(COALESCE(m.store_code, ''))) AS shop_code,
            UPPER(TRIM(COALESCE(m.cashier, ''))) AS cashier_name,
            NULLIF(REGEXP_REPLACE(COALESCE(m.tillno::text, ''), '[^0-9]', '', 'g'), '')::int AS till_no,
            MAX(CASE WHEN NULLIF(TRIM(COALESCE(m.invno::text, '')), '') IS NOT NULL THEN 1 ELSE 0 END) AS has_handover
        FROM invoices_manager m
        WHERE m.invdate::date BETWEEN %(s)s AND %(e)s
          AND UPPER(TRIM(COALESCE(m.store_code, ''))) = ANY(%(shops)s)
          AND NULLIF(TRIM(COALESCE(m.cashier, '')), '') IS NOT NULL
          AND NULLIF(REGEXP_REPLACE(COALESCE(m.tillno::text, ''), '[^0-9]', '', 'g'), '') IS NOT NULL
        GROUP BY m.invdate::date, UPPER(TRIM(COALESCE(m.store_code, ''))),
                 UPPER(TRIM(COALESCE(m.cashier, ''))),
                 NULLIF(REGEXP_REPLACE(COALESCE(m.tillno::text, ''), '[^0-9]', '', 'g'), '')::int
    ),
    mgr_agg AS (
        SELECT bill_date, shop_code,
               COUNT(*) FILTER (WHERE has_handover = 1) AS handover_count
        FROM mgr_sessions
        GROUP BY bill_date, shop_code
    ),
    daily AS (
        SELECT
            d.bill_date,
            COALESCE(SUM(ug.unique_generated), 0) AS total_generated,
            COALESCE(SUM(ma.handover_count), 0)   AS total_handover
        FROM days d
        LEFT JOIN unique_gen ug ON ug.bill_date = d.bill_date
        LEFT JOIN mgr_agg   ma ON ma.bill_date = d.bill_date AND ma.shop_code = ug.shop_code
        GROUP BY d.bill_date
    )
    SELECT
        bill_date,
        total_generated,
        total_handover,
        CASE WHEN total_generated > 0
             THEN ROUND((total_handover::numeric / total_generated) * 100, 1)
             ELSE 0
        END AS handover_pct
    FROM daily
    ORDER BY bill_date
    """
    with get_db_connection() as conn:
        return pd.read_sql(query, conn, params={
            "s": start_10, "e": end_date,
            "shops": sorted(list(IMPLEMENTED_SHOPS_SET)),
        })


@st.cache_data(ttl=300)
def load_alert_type_comparison(end_date: date) -> pd.DataFrame:
    selected_date = end_date
    yesterday_date = end_date - timedelta(days=1)
    month_start = end_date.replace(day=1)

    query = """
    WITH alert_types AS (
        SELECT * FROM (VALUES
            ('Test Bill', 1),
            ('Bill Date Mismatch', 2),
            ('Duplicate', 3),
            ('Wrong Shop', 4),
            ('High Bill Amount', 5)
        ) AS t(a_type, sort_order)
    ),
    counts AS (
        SELECT
            a_type_normalized                                                              AS a_type,
            SUM(alert_count) FILTER (WHERE alert_date = %(selected_date)s)::bigint        AS selected_date_total,
            SUM(alert_count) FILTER (WHERE alert_date = %(yesterday_date)s)::bigint       AS yesterday_total,
            SUM(alert_count)::bigint                                                      AS month_to_date_total
        FROM mv_wh_alerts_daily
        WHERE alert_date BETWEEN %(month_start)s AND %(selected_date)s
          AND shop_code = ANY(%(shops)s)
        GROUP BY a_type_normalized
    )
    SELECT
        t.a_type,
        COALESCE(c.selected_date_total, 0)::bigint  AS selected_date_total,
        COALESCE(c.yesterday_total, 0)::bigint       AS yesterday_total,
        COALESCE(c.month_to_date_total, 0)::bigint   AS month_to_date_total
    FROM alert_types t
    LEFT JOIN counts c ON c.a_type = t.a_type
    ORDER BY t.sort_order
    """

    with get_db_connection() as conn:
        return pd.read_sql(
            query,
            conn,
            params={
                "month_start": month_start,
                "selected_date": selected_date,
                "yesterday_date": yesterday_date,
                "shops": sorted(list(IMPLEMENTED_SHOPS_SET)),
            },
        )


@st.cache_data(ttl=300)
def load_alert_type_drilldown(end_date: date, selected_error_type: str) -> pd.DataFrame:
    month_start = end_date.replace(day=1)

    query = """
    WITH normalized_alerts AS (
        SELECT
            UPPER(TRIM(COALESCE(a_store_code, ''))) AS shop_code,
            COALESCE(
                NULLIF(TRIM(COALESCE(to_jsonb(alerts) ->> 'cashier', to_jsonb(alerts) ->> 'cashier_name', to_jsonb(alerts) ->> 'a_cashier')), ''),
                '0'
            ) AS cashier,
            COALESCE(
                NULLIF(TRIM(COALESCE(to_jsonb(alerts) ->> 'device_info', to_jsonb(alerts) ->> 'deviceinfo', to_jsonb(alerts) ->> 'device', to_jsonb(alerts) ->> 'a_device_info')), ''),
                '0'
            ) AS device_info,
            COALESCE(scanned_date::date, a_entrytime::date) AS scanned_date,
            CASE
                WHEN LOWER(TRIM(a_type)) IN ('bill date mismatched', 'bill date mismatch') THEN 'Bill Date Mismatch'
                WHEN LOWER(TRIM(a_type)) = 'test bill' THEN 'Test Bill'
                WHEN LOWER(TRIM(a_type)) IN ('duplicate', 'duplicate bill') THEN 'Duplicate'
                WHEN LOWER(TRIM(a_type)) IN ('wrong shop', 'wrong sho') THEN 'Wrong Shop'
                WHEN LOWER(TRIM(a_type)) = 'high bill amount' THEN 'High Bill Amount'
                ELSE INITCAP(TRIM(COALESCE(a_type, 'Unknown')))
            END AS error_type,
            COALESCE(NULLIF(TRIM(a_type), ''), '0') AS a_type_raw,
            COALESCE(scanned_date::date, a_entrytime::date) AS alert_date
        FROM alerts
        WHERE UPPER(TRIM(a_store_code)) = ANY(%(shops)s)
    )
    SELECT
        shop_code AS "Shop Code",
        cashier AS "Cashier",
        error_type AS "Error Type",
        scanned_date AS "Scanned Date",
        device_info AS "Device Info",
        a_type_raw AS "a_type (Error Type)"
    FROM normalized_alerts
    WHERE error_type = %(selected_error_type)s
      AND alert_date BETWEEN %(month_start)s AND %(end_date)s
    ORDER BY scanned_date DESC, shop_code
    """

    with get_db_connection() as conn:
        return pd.read_sql(
            query,
            conn,
            params={
                "selected_error_type": selected_error_type,
                "month_start": month_start,
                "end_date": end_date,
                "shops": sorted(list(IMPLEMENTED_SHOPS_SET)),
            },
        )



def render_alert_error_by_type(end_date: date):
    alerts_df = load_alert_type_comparison(end_date)

    selected_label = end_date.strftime("%d %b %Y")
    yday_label     = (end_date - timedelta(days=1)).strftime("%d %b %Y")
    mtd_label      = f"{end_date.strftime('%b')} MTD"

    if alerts_df.empty:
        st.info("No alert errors found for selected date/month context.")
        return

    # ── Map column names to display labels ───────────────────────────────────
    col_map = {
        "selected_date_total": f"Selected Date ({selected_label})",
        "yesterday_total":     yday_label,
        "month_to_date_total": f"Month-to-Date ({mtd_label})",
    }
    num_cols = list(col_map.keys())   # raw column names with numbers

    # ── Build clickable HTML table ────────────────────────────────────────────
    # Clicking any numeric cell stores (error_type, column_label) in session state
    ss_key = "alert_type_drill"

    header_cells = "<th>Alert Error Type</th>" + "".join(
        f"<th>{col_map[c]}</th>" for c in num_cols
    )

    body_rows = ""
    for _, row in alerts_df.iterrows():
        a_type = str(row["a_type"])
        body_rows += f"<tr><td class='ae-label'>{a_type}</td>"
        for c in num_cols:
            val = int(row[c]) if pd.notna(row[c]) else 0
            col_label = col_map[c]
            # Each clickable number fires a Streamlit query param + JS postMessage
            body_rows += (
                f"<td class='ae-num ae-clickable' "
                f"data-type='{a_type}' data-col='{col_label}' "
                f"onclick=\"drillAlert('{a_type}','{col_label}')\">{val:,}</td>"
            )
        body_rows += "</tr>"

    # Current drilldown selection indicator
    current_drill = st.session_state.get(ss_key)
    drill_badge = ""
    if current_drill:
        drill_badge = (
            f"<div class='ae-drill-badge'>🔍 {current_drill['error_type']} "
            f"· {current_drill['col']}"
            f" <span class='ae-clear' onclick=\"clearAlert()\">✕</span></div>"
        )

    table_html = f"""
<style>
.ae-wrap {{ width:100%; font-family:'Inter','Segoe UI',sans-serif;
    border:1.5px solid #38bdf8; border-radius:7px;
    overflow:hidden; padding:4px 4px 2px 4px;
    box-sizing:border-box;
}}
.ae-wrap table {{
    width:100%; border-collapse:collapse;
    font-size:10.5px; table-layout:fixed;
}}
.ae-wrap th {{
    background:#1e2a3a; color:#ffffff;
    padding:5px 6px; font-size:10px; font-weight:600;
    text-align:center;
    border-bottom:2px solid #38bdf8;
    border-right:1px solid #334155;
    word-wrap:break-word; white-space:normal;
}}
.ae-wrap th:last-child {{ border-right:none; }}
.ae-wrap td {{
    padding:5px 6px;
    border-bottom:1px solid #2d3a4a;
    border-right:1px solid #1e2a3a;
    text-align:center; color:#ffffff; font-size:10.5px;
}}
.ae-wrap td:last-child {{ border-right:none; }}
.ae-label {{ text-align:left !important; color:#ffffff; }}
.ae-num {{ color:#38bdf8; font-weight:600; }}
.ae-clickable {{ cursor:pointer; }}
.ae-clickable:hover {{ background:#1e3a5f !important; color:#fff !important; border-radius:3px; }}
.ae-section-title {{
    font-size:11px; font-weight:700; color:#ffffff;
    text-transform:uppercase; letter-spacing:.05em;
    margin-bottom:4px;
}}
.ae-drill-badge {{
    font-size:10px; color:#fbbf24; background:#1a2535;
    border:1px solid #2d3a4a; border-radius:4px;
    padding:3px 8px; margin-top:4px; display:inline-block;
}}
.ae-clear {{ cursor:pointer; margin-left:6px; color:#f87171; font-weight:700; }}
</style>
<div class="ae-wrap">
  <div class="ae-section-title">Alert Error TYPE</div>
  {drill_badge}
  <table>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{body_rows}</tbody>
  </table>
</div>
<script>
function drillAlert(errType, colLabel) {{
    // Show inline loading feedback immediately
    var existing = document.querySelector('.ae-drill-badge');
    if (existing) {{
        existing.innerHTML = '⏳ Loading ' + errType + '…';
    }} else {{
        var badge = document.createElement('div');
        badge.className = 'ae-drill-badge';
        badge.innerHTML = '⏳ Loading ' + errType + '…';
        var wrap = document.querySelector('.ae-wrap');
        if (wrap) wrap.insertBefore(badge, wrap.querySelector('table'));
    }}
    // Navigate parent to new URL — this triggers a full Streamlit rerun
    const url = new URL(window.parent.location.href);
    url.searchParams.set('alert_drill_type', errType);
    url.searchParams.set('alert_drill_col',  colLabel);
    url.searchParams.delete('alert_drill_clear');
    window.parent.location.href = url.toString();
}}
function clearAlert() {{
    // Navigate to URL without drill params — triggers rerun, Python clears state
    const url = new URL(window.parent.location.href);
    url.searchParams.delete('alert_drill_type');
    url.searchParams.delete('alert_drill_col');
    url.searchParams.set('alert_drill_clear', '1');
    window.parent.location.href = url.toString();
}}
</script>
"""
    # Render the table — dynamic height: title(22) + header(28) + each row(26) + badge(30 if active) + padding(18)
    _n_rows = len(alerts_df) if alerts_df is not None else 0
    _badge_h = 32 if current_drill else 0
    _table_h = 22 + 28 + _n_rows * 26 + _badge_h + 20
    components.html(table_html, height=_table_h, scrolling=False)

    # ── Bill Handover % — Last 10 Days line chart ─────────────────────────────
    try:
        ho_df = load_bill_handover_last10(end_date)
    except Exception:
        ho_df = pd.DataFrame()

    if not ho_df.empty:
        ho_df["bill_date"] = pd.to_datetime(ho_df["bill_date"])
        ho_df["day_label"] = ho_df["bill_date"].dt.strftime("%d %b")
        ho_df["handover_pct"] = pd.to_numeric(ho_df["handover_pct"], errors="coerce").fillna(0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ho_df["day_label"],
            y=ho_df["handover_pct"],
            mode="lines+markers",
            line=dict(color="#38bdf8", width=2.5, shape="spline", smoothing=1.2),
            marker=dict(size=5, color="#38bdf8", line=dict(color="#0a0f1e", width=1)),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.08)",
            hovertemplate="%{x}<br><b>%{y:.1f}%</b><extra></extra>",
        ))
        fig.update_layout(
            title=dict(
                text="Bill Handover % — Last 10 Days",
                font=dict(size=11, color="#ffffff", family="Inter"),
                x=0, xanchor="left",
            ),
            height=160,
            margin=dict(l=4, r=4, t=28, b=4),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showgrid=False, zeroline=False,
                tickfont=dict(size=9, color="#ffffff"),
                tickangle=0,
            ),
            yaxis=dict(
                showgrid=True, gridcolor="#1e2a3a", zeroline=False,
                tickfont=dict(size=9, color="#ffffff"),
                ticksuffix="%",
                range=[0, max(110, ho_df["handover_pct"].max() * 1.15)],
            ),
            showlegend=False,
        )
        st.markdown("""
        <style>
        div[data-testid="stPlotlyChart"] {
            border: 1.5px solid #38bdf8 !important;
            border-radius: 7px;
            overflow: hidden;
            padding: 2px;
        }
        </style>""", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Sync query params → session state on every rerun ───────────────────
    try:
        qp = st.query_params
        drill_type  = qp.get("alert_drill_type", None)
        drill_col   = qp.get("alert_drill_col",  None)
        drill_clear = qp.get("alert_drill_clear", None)
        if drill_clear:
            # ✕ was clicked — wipe session state and clean up the URL flag
            st.session_state.pop(ss_key, None)
            try:
                del st.query_params["alert_drill_clear"]
            except Exception:
                pass
        elif drill_type and str(drill_type).strip():
            # Cell was clicked — store which error type + column was selected
            st.session_state[ss_key] = {
                "error_type": str(drill_type).strip(),
                "col":        str(drill_col).strip() if drill_col else "",
            }
        else:
            # No drill params in URL at all → clear any stale session state
            # so drilldown never appears on a fresh page load
            st.session_state.pop(ss_key, None)
    except Exception:
        pass


def render_alert_error_type_drilldown(end_date: date):
    alerts_df = load_alert_type_comparison(end_date)
    if alerts_df.empty:
        st.info("No alert errors found for selected date/month context.")
        return

    available_types = [t for t in ALERT_ERROR_TYPE_ORDER if t in set(alerts_df["a_type"].astype(str))]
    if not available_types:
        st.info("No drilldown error types available.")
        return

    selected_error_type = st.selectbox(
        "Drilldown Error Type",
        options=available_types,
        index=0,
        key=f"alert_error_type_drill_{end_date}",
    )

    drill_df = load_alert_type_drilldown(end_date, selected_error_type)
    if drill_df.empty:
        st.info("No drilldown rows for selected error type in current month-to-date window.")
        return

    render_styled_html_table(
        drill_df,
        title="Alert Error TYPE Drilldown",
        auto_content_width=False,
        wrap_header_text=True,
        wrap_cell_text=True,
        compact_mode=True,
        title_class="section-title",
        align_left=False,
        align_title_with_table=False,
        min_width_px_override=320,
        table_layout_override="fixed",
        overflow_x_override="hidden",
        header_font_size_override="10px",
        body_font_size_override="10.5px",
        body_padding_override="6px 6px",
    )


def render_top10_shop_max_value_diff(df: pd.DataFrame):
    if df is None or df.empty:
        return

    top_df = df.copy()
    top_df["shop_name"] = top_df["shop_code"].map(load_shop_name_map()).fillna(top_df["shop_code"])
    top_df["diff_ghs"] = pd.to_numeric(top_df["diff_ghs"], errors="coerce").fillna(0)
    top_df["diff_pct"] = pd.to_numeric(top_df["diff_pct"], errors="coerce").fillna(0)

    top_df = top_df.sort_values("diff_ghs", ascending=False).head(10)

    display_df = top_df[["shop_code", "shop_name", "diff_ghs"]].copy()
    display_df.columns = [
        "Shop Code",
        "Shop Name",
        "Diff (GHS)",
    ]
    display_df["Diff (GHS)"] = pd.to_numeric(display_df["Diff (GHS)"], errors="coerce").fillna(0).round(0).astype(int)

    render_styled_html_table(
        display_df,
        title="Top 10 shop having max difference in values",
        auto_content_width=False,
        wrap_header_text=True,
        wrap_cell_text=True,
        compact_mode=True,
        title_class="section-title",
        align_left=False,
        align_title_with_table=False,
        min_width_px_override=320,
        table_layout_override="fixed",
        overflow_x_override="hidden",
        header_font_size_override="10px",
        body_font_size_override="10.5px",
        body_padding_override="6px 6px",
    )


def _summarize_diff_metrics(start_date, end_date) -> tuple[float, float]:
    period_df = load_owner_view(start_date, end_date)
    if period_df is None or period_df.empty:
        return 0.0, 0.0
    qty_diff = float(pd.to_numeric(period_df["bills_not_scanned"], errors="coerce").fillna(0).sum())
    value_diff = float(pd.to_numeric(period_df["diff_ghs"], errors="coerce").fillna(0).sum())
    return qty_diff, value_diff


def _build_anomaly_alerts(df: pd.DataFrame, start_date: date, end_date: date) -> list[dict]:
    if df is None or df.empty:
        return []

    work = df.copy()
    for col in ["erp_nob", "bill_pct", "diff_ghs", "bills_not_scanned", "duplicate"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)

    p95_diff = float(work["diff_ghs"].quantile(0.95)) if "diff_ghs" in work.columns else 0.0
    p95_not_scanned = float(work["bills_not_scanned"].quantile(0.95)) if "bills_not_scanned" in work.columns else 0.0
    p95_duplicate = float(work["duplicate"].quantile(0.95)) if "duplicate" in work.columns else 0.0

    alerts = []

    high_diff = work[(work["diff_ghs"] >= p95_diff) & (work["diff_ghs"] > 0)].sort_values("diff_ghs", ascending=False).head(10)
    for _, row in high_diff.iterrows():
        alerts.append({
            "shop_code": str(row.get("shop_code", "")).strip().upper(),
            "rule": "HIGH_DIFF_GHS",
            "value": float(row.get("diff_ghs", 0)),
        })

    high_not_scanned = work[(work["bills_not_scanned"] >= p95_not_scanned) & (work["bills_not_scanned"] > 0)].sort_values("bills_not_scanned", ascending=False).head(10)
    for _, row in high_not_scanned.iterrows():
        alerts.append({
            "shop_code": str(row.get("shop_code", "")).strip().upper(),
            "rule": "HIGH_BILLS_NOT_SCANNED",
            "value": float(row.get("bills_not_scanned", 0)),
        })

    over_coverage = work[(work["erp_nob"] >= 50) & (work["bill_pct"] > 100)].sort_values("bill_pct", ascending=False).head(10)
    for _, row in over_coverage.iterrows():
        alerts.append({
            "shop_code": str(row.get("shop_code", "")).strip().upper(),
            "rule": "OVER_100_BILL_PERCENT",
            "value": float(row.get("bill_pct", 0)),
        })

    duplicate_spike = work[(work["duplicate"] >= p95_duplicate) & (work["duplicate"] > 0)].sort_values("duplicate", ascending=False).head(10)
    for _, row in duplicate_spike.iterrows():
        alerts.append(
            {
                "shop_code": str(row.get("shop_code", "")).strip().upper(),
                "rule": "HIGH_DUPLICATE_ALERTS",
                "value": float(row.get("duplicate", 0)),
            }
        )

    st.session_state["latest_anomaly_alerts"] = alerts
    return alerts


def _emit_anomaly_alerts_silent(df: pd.DataFrame, start_date: date, end_date: date) -> None:
    alerts = _build_anomaly_alerts(df, start_date, end_date)
    if not alerts:
        return
    logger.info("Generated %s anomaly alerts for %s to %s", len(alerts), start_date, end_date)


def main():
    inject_css()

    ensure_usage_events_table()
    if 'usage_event_session_id' not in st.session_state:
        st.session_state.usage_event_session_id = uuid.uuid4().hex
    if 'usage_session_started_at' not in st.session_state:
        st.session_state.usage_session_started_at = datetime.now().isoformat()
    if 'usage_app_open_logged' not in st.session_state:
        st.session_state.usage_app_open_logged = False
    if 'usage_last_date_range' not in st.session_state:
        st.session_state.usage_last_date_range = ''

    system_login_id, ip_address = ensure_identity_captured()

    if not st.session_state.usage_app_open_logged:
        log_usage_event(
            'app_open',
            event_type='session',
            event_details={
                'system_login_id': system_login_id,
                'ip_address': ip_address,
            },
        )
        st.session_state.usage_app_open_logged = True

    st.markdown(
        "<div class='dashboard-title'>Invoice Scanning Dashboard</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='dashboard-subtitle'>One-view bill accountability across ERP, scanned invoices, consumable tills, and alerts.</div>",
        unsafe_allow_html=True,
    )

    today = datetime.today().date()
    yesterday = today - timedelta(days=1)

    with st.sidebar:
        st.markdown("### Usage Tracking")
        st.caption(f"Auto-detected system login: {system_login_id or 'unknown'}")
        st.caption(f"Auto-detected IP: {ip_address or 'unknown'}")
        usage_days = st.slider(
            "Usage report period (days)",
            min_value=1,
            max_value=90,
            value=30,
            step=1,
            key='invoice_usage_report_days',
        )
        usage_summary = get_usage_summary(usage_days)
        current_session_seconds = max(
            0,
            int((datetime.now() - datetime.fromisoformat(st.session_state.usage_session_started_at)).total_seconds()),
        )
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Events", f"{int(usage_summary.get('total_events', 0) or 0):,}")
            st.metric("Event Clicks", f"{int(usage_summary.get('total_event_clicks', 0) or 0):,}")
            st.metric("Time Spent", f"{round(float(usage_summary.get('total_time_spent_seconds', 0) or 0) / 60, 1):,.1f} min")
        with c2:
            st.metric("Sessions", f"{int(usage_summary.get('total_sessions', 0) or 0):,}")
            st.metric("Avg Session", f"{round(float(usage_summary.get('avg_session_seconds', 0) or 0) / 60, 1):,.1f} min")
            st.metric("Current Session", f"{round(current_session_seconds / 60, 1):,.1f} min")

        with st.expander("Management Usage Report", expanded=False):
            usage_report_df = get_usage_events_report(usage_days)
            if usage_report_df.empty:
                st.info("No usage events found for selected period.")
            else:
                usage_display_df = usage_report_df.head(200).copy()
                usage_headers = [str(col) for col in usage_display_df.columns]
                usage_rows: list[list[str]] = []
                usage_sort_rows: list[list[str]] = []

                for _, usage_row in usage_display_df.iterrows():
                    row_display: list[str] = []
                    row_sort: list[str] = []
                    for col in usage_display_df.columns:
                        raw_val = usage_row[col]
                        if isinstance(raw_val, (datetime, pd.Timestamp)):
                            disp_val = raw_val.strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            disp_val = "" if pd.isna(raw_val) else str(raw_val)
                        row_display.append(disp_val)
                        row_sort.append(_sort_token(raw_val))
                    usage_rows.append(row_display)
                    usage_sort_rows.append(row_sort)

                usage_excel_buffer = io.BytesIO()
                with pd.ExcelWriter(usage_excel_buffer, engine="openpyxl") as writer:
                    usage_display_df.to_excel(writer, index=False, sheet_name="Usage Report")
                usage_excel_b64 = base64.b64encode(usage_excel_buffer.getvalue()).decode("ascii")

                render_sortable_html_table(
                    title="",
                    headers=usage_headers,
                    rows=usage_rows,
                    sort_values=usage_sort_rows,
                    table_id=f"usage_report_table_{usage_days}",
                    show_title=False,
                    min_width_px=1100,
                    left_align_cols={0, 3, 4, 5},
                    enable_download_hover=True,
                    download_file_name=f"invoice_dashboard_usage_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    excel_base64=usage_excel_b64,
                    wrapper_max_height_px=240,
                    table_width="100%",
                    table_layout="auto",
                    overflow_x="auto",
                    header_white_space="nowrap",
                    header_word_break="normal",
                    text_overflow="ellipsis",
                    header_font_size="10px",
                    cell_font_size="10.5px",
                    header_padding="6px 8px",
                    cell_padding="6px 8px",
                )

                st.download_button(
                    label="Download Usage CSV",
                    data=usage_report_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"invoice_dashboard_usage_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    with st.container():
        hdr_col, date_col, refresh_col = st.columns([0.72, 0.20, 0.08])
        with hdr_col:
            st.markdown("")
        with date_col:
            selected_range = st.date_input("Date Range", value=(yesterday, yesterday))
        with refresh_col:
            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
            refresh = st.button("Refresh", help="Clear cache and reload fresh data", use_container_width=True)

    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
    else:
        start_date = selected_range
        end_date = selected_range

    current_range_key = f"{start_date.isoformat()}|{end_date.isoformat()}"
    if st.session_state.usage_last_date_range != current_range_key:
        log_usage_event(
            'date_range_changed',
            event_type='filter',
            event_details={'start_date': start_date.isoformat(), 'end_date': end_date.isoformat()},
        )
        st.session_state.usage_last_date_range = current_range_key

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
        return

    if refresh:
        log_usage_event('refresh_clicked', event_type='click')
        st.cache_data.clear()

    ensure_cashier_sessions_mv_exists()

    data = load_owner_view(start_date, end_date)

    for col in [
        "erp_nob", "scan_nob", "consumable_till_nob", "test_bills", "test_bill_not_generated",
        "total_accounted_bills", "bills_not_scanned", "bill_pct", "erp_erp_nob_ghs", "scanned_nob_ghs",
        "consumable_nob_ghs", "diff_ghs", "diff_pct", "test_bill_scanned", "bill_date_mismatched",
        "duplicate", "wrong_shop", "high_value_bill",
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    if data.empty:
        st.warning("No data found for selected date range.")
        return

    _emit_anomaly_alerts_silent(data, start_date, end_date)

    render_kpis(data, start_date, end_date)
    month_df, last_10_df, _ = render_diff_trend_graphs(end_date)

    st.markdown("<div class='kpi-heading-center'>Diff + Shop & Alert Snapshot</div>", unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns([1, 1, 1, 1], gap="small")
    with p1:
        render_month_diff_graph(month_df, end_date)
    with p2:
        render_last10_diff_graph(last_10_df)
    with p3:
        render_top10_shop_max_value_diff(data)
    with p4:
        render_alert_error_by_type(end_date)

    st.markdown("<div style='height: 0px;'></div>", unsafe_allow_html=True)

    _, period_compact = _period_labels(start_date, end_date)
    st.markdown(
        f"<div class='kpi-heading-center' style='margin-top:-8px; margin-bottom:0.35rem;'>One-View Summary • {period_compact}</div>",
        unsafe_allow_html=True,
    )
    summary_data = data.copy()
    st.caption(f"Showing all shops from KPI scope ({len(summary_data)} shops)")

    display_df = to_display(summary_data)

    update_and_get_new_shops(display_df["Shop Code"].tolist())

    render_summary_html_table(display_df)

    st.markdown(
        f"<div class='kpi-heading-center' style='margin-top:0px; margin-bottom:0.35rem;'>Shop wise Test Bill analysis • {period_compact}</div>",
        unsafe_allow_html=True,
    )
    render_shopwise_test_bill_analysis(start_date, end_date)

    # ── Alert drilldown → appears after Shop wise Test Bill analysis ──────────
    _alert_drill = st.session_state.get("alert_type_drill")
    if _alert_drill and _alert_drill.get("error_type"):
        _dt = _alert_drill["error_type"]
        _cl = _alert_drill.get("col", "")
        st.markdown(
            f"<div class='kpi-heading-center' style='margin-top:8px; margin-bottom:0.3rem;'>"
            f"🔍 Alert Drilldown — <span style='color:#38bdf8'>{_dt}</span>"
            f"{(' · ' + _cl) if _cl else ''}"
            f"</div>",
            unsafe_allow_html=True,
        )
        with st.spinner(f"Loading {_dt} drilldown…"):
            drill_df = load_alert_type_drilldown(end_date, _dt)
        if not drill_df.empty:
            render_styled_html_table(
                drill_df,
                title=f"Alert Detail — {_dt}",
                auto_content_width=False,
                wrap_header_text=True,
                wrap_cell_text=True,
                compact_mode=True,
                title_class="section-title",
                align_left=False,
                align_title_with_table=False,
                min_width_px_override=320,
                table_layout_override="auto",
                overflow_x_override="auto",
                header_font_size_override="10px",
                body_font_size_override="10.5px",
                body_padding_override="5px 6px",
            )
        else:
            st.info(f"No detail rows found for '{_dt}' in current month.")
        st.markdown("<div style='height: 0px;'></div>", unsafe_allow_html=True)

    csv_data = summary_data.copy()
    csv_data["shop_name"] = csv_data["shop_code"].map(load_shop_name_map()).fillna(csv_data["shop_code"])


if __name__ == "__main__":
    main()
