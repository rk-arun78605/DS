"""
Century Stock Penetration Dashboard
Real-time analytics for CENTURY brand stock positioning across shops
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import html
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, Json
from contextlib import contextmanager
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import io
import os
import uuid
import getpass
import socket

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    DB_CONFIG = {
        'host': 'localhost',
        'port': 3307,
        'user': 'postgres',
        'password': 'hello',
        'database': 'century_penetration'
    }
    
    PAGE_TITLE = "Century Stock Penetration"
    PAGE_ICON = "📊"
    LOGO_URL = "https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg"
    
    # Stock status colors
    COLORS = {
        'UnderStock': '#f5576c',
        'OverStock': '#667eea',
        'Balanced': '#00d2ff'
    }


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# DATABASE CONNECTION
# ============================================================

@st.cache_resource
def get_connection_pool():
    """Create connection pool"""
    return psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=5,
        **Config.DB_CONFIG
    )


@contextmanager
def get_db_connection():
    """Get connection from pool"""
    pool = get_connection_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


def get_dashboard_refresh_marker() -> str:
    """Return a lightweight marker that changes whenever the main MV is refreshed."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COALESCE(MAX(refreshed_at)::text, '')
            FROM mv_century_penetration
            """
        )
        row = cursor.fetchone()
    return str(row[0] or '') if row else ''


def sync_dashboard_cache_with_database() -> None:
    """Clear cached dashboard queries when the source MV refresh timestamp changes."""
    marker_key = 'century_dashboard_refresh_marker'
    try:
        current_marker = get_dashboard_refresh_marker()
    except Exception:
        return

    previous_marker = st.session_state.get(marker_key)
    if previous_marker is None:
        st.session_state[marker_key] = current_marker
        return

    if previous_marker != current_marker:
        st.session_state[marker_key] = current_marker
        st.cache_data.clear()
        st.rerun()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_column_config():
    """
    Get column configuration with tooltips showing formulas for derived columns
    """
    return {
        'sih': st.column_config.NumberColumn(
            'SIH',
            help='📦 Stock In Hand - Current shop inventory',
            format='%d'
        ),
        'sit': st.column_config.NumberColumn(
            'SIT',
            help='🚚 Stock In Transit - Items en route to shop',
            format='%d'
        ),
        'total_stock': st.column_config.NumberColumn(
            'Total Stock',
            help='📊 Total Stock = SIH + SIT',
            format='%d'
        ),
        'sales_30d': st.column_config.NumberColumn(
            'Last 30d',
            help='📈 Sales (Yesterday-30 to Yesterday) - Last 30 days sales excluding today',
            format='%d'
        ),
        'sales_60d': st.column_config.NumberColumn(
            'Last 60d',
            help='📈 Sales (Yesterday-60 to Yesterday) - Last 60 days sales excluding today',
            format='%d'
        ),
        'sales_90d': st.column_config.NumberColumn(
            'Last 90d',
            help='📈 Sales (Yesterday-90 to Yesterday) - Last 90 days sales excluding today',
            format='%d'
        ),
        'sales_365d': st.column_config.NumberColumn(
            'Last 365d',
            help='📈 Sales (Yesterday-365 to Yesterday) - Last year sales excluding today',
            format='%d'
        ),
        'ros': st.column_config.NumberColumn(
            'ROS',
            help='⚡ Rate of Sales = Sales (Last 90 days) ÷ 90 - Average daily sales rate',
            format='%.2f'
        ),
        'req_30_days': st.column_config.NumberColumn(
            'Req 30 Days',
            help='🎯 Optimum Stock = ROS (Average per day of last 90 days) × (30 maintenance days + lead days per shop policy). DAILY shops (lead=7): ROS×37 | ALTERNATE DAYS shops (lead=14): ROS×44',
            format='%.2f'
        ),
        'stock_variance': st.column_config.NumberColumn(
            'Stock Variance',
            help='📊 Stock Variance = (SIH + SIT) - Optimum Stock (30 days + shop lead days)',
            format='%.2f'
        ),
        'pack_size': st.column_config.NumberColumn(
            'Pack Size',
            help='📦 Pack Size - Tolerance threshold for stock status classification',
            format='%d'
        ),
        'days_of_stock': st.column_config.NumberColumn(
            'Days of Stock',
            help='📅 Days of Stock = (SIH + SIT) ÷ ROS - How many days current stock will last',
            format='%.1f'
        ),
        'stock_status': st.column_config.TextColumn(
            'Status',
            help='⚠️ Policy Status: UnderStock if stock < (Optimum × threshold), OverStock if stock > Optimum. Threshold is 30% for 7-day lead, 50% for 14-day lead.'
        ),
        'min_nu': st.column_config.NumberColumn(
            'Min',
            help='📉 Minimum Stock Level - Reorder point',
            format='%d'
        ),
        'max_nu': st.column_config.NumberColumn(
            'Max',
            help='📈 Maximum Stock Level - Upper limit',
            format='%d'
        ),
        'reorder_qty': st.column_config.NumberColumn(
            'Reorder Qty',
            help='🔄 Reorder Quantity - Order quantity when below min',
            format='%d'
        ),
        'selling_price': st.column_config.NumberColumn(
            'Price',
            help='💰 Selling Price',
            format='%.2f'
        ),
        'value_30d': st.column_config.NumberColumn(
            'Value 30d',
            help='💵 Sales Value (Last 30 days) = Sum of (Quantity × Price)',
            format='%.2f'
        ),
        'value_60d': st.column_config.NumberColumn(
            'Value 60d',
            help='💵 Sales Value (Last 60 days) = Sum of (Quantity × Price)',
            format='%.2f'
        ),
        'value_90d': st.column_config.NumberColumn(
            'Value 90d',
            help='💵 Sales Value (Last 90 days) = Sum of (Quantity × Price)',
            format='%.2f'
        ),
        'value_365d': st.column_config.NumberColumn(
            'Value 365d',
            help='💵 Sales Value (Last 365 days) = Sum of (Quantity × Price)',
            format='%.2f'
        )
    }


def _to_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


@st.cache_data(ttl=600)
def get_shop_policy_lookup() -> dict[str, dict[str, float]]:
    """Load per-shop policy values from MV columns for reliable tooltip formulas."""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT
                shop_code,
                MAX(lead) AS lead,
                MAX(min_threshold_pct) AS min_threshold_pct
            FROM mv_century_penetration
            WHERE shop_code IS NOT NULL
            GROUP BY shop_code
            """
        )
        rows = cursor.fetchall()

    lookup: dict[str, dict[str, float]] = {}
    for r in rows:
        code = str(r.get('shop_code', '') or '').strip().upper()
        if not code:
            continue
        lookup[code] = {
            'lead': _to_float(r.get('lead', None), 14.0),
            'min_threshold_pct': _to_float(r.get('min_threshold_pct', None), 50.0),
        }
    return lookup


def _resolve_row_policy_values(row: pd.Series, policy_lookup: dict[str, dict[str, float]] | None = None) -> tuple[float, float, float]:
    """Resolve lead/min-threshold/optimum-days from row first, then shop policy lookup."""
    shop_code = str(row.get('shop_code', '') or '').strip().upper()
    policy = (policy_lookup or {}).get(shop_code, {})

    lead_days = _to_float(
        row.get('lead', row.get('lead_days', policy.get('lead', 14.0))),
        _to_float(policy.get('lead', 14.0), 14.0),
    )

    min_pct_raw = _to_float(
        row.get(
            'min_threshold',
            row.get(
                'min_threshold_pct',
                policy.get('min_threshold_pct', 30.0 if lead_days <= 7 else 50.0),
            ),
        ),
        _to_float(policy.get('min_threshold_pct', 30.0 if lead_days <= 7 else 50.0), 50.0),
    )

    optimum_days = _to_float(
        row.get('optimum_stock', row.get('optimum_stock_days', 30.0)),
        30.0,
    )

    return lead_days, min_pct_raw, optimum_days


def with_formula_tooltips(df: pd.DataFrame):
    """
    Add row-level formula tooltips to derived metric cells so users can hover and
    see calculation steps with actual numbers.
    """
    if df is None or df.empty:
        return df

    tooltip_cols = {
        'total_stock',
        'req_30_days',
        'stock_variance',
        'min_threshold_qty',
        'days_of_stock',
        'stock_status',
    }
    if not any(col in df.columns for col in tooltip_cols):
        return df

    tips = pd.DataFrame('', index=df.index, columns=df.columns)
    policy_lookup = get_shop_policy_lookup()

    for idx, row in df.iterrows():
        sih = _to_float(row.get('sih', 0), 0.0)
        sit = _to_float(row.get('sit', 0), 0.0)
        total_stock = _to_float(row.get('total_stock', sih + sit), sih + sit)
        ros = _to_float(row.get('ros', 0), 0.0)
        lead_days, min_pct_raw, optimum_days = _resolve_row_policy_values(row, policy_lookup)
        optimum_30 = _to_float(
            row.get('req_30_days', row.get('optimum_stock_30d', ros * (optimum_days + lead_days))),
            ros * (optimum_days + lead_days),
        )
        variance = _to_float(row.get('stock_variance', total_stock - optimum_30), total_stock - optimum_30)

        min_pct_ratio = (min_pct_raw / 100.0) if min_pct_raw > 1 else min_pct_raw
        min_pct_display = min_pct_ratio * 100.0
        min_threshold_qty = _to_float(
            row.get('min_threshold_qty', optimum_30 * min_pct_ratio),
            optimum_30 * min_pct_ratio,
        )

        days_of_stock = _to_float(row.get('days_of_stock', 0), 0.0)
        stock_status = str(row.get('stock_status', '') or '').strip()

        if 'total_stock' in tips.columns:
            tips.at[idx, 'total_stock'] = f"SIH ({sih:,.0f}) + SIT ({sit:,.0f}) = {total_stock:,.0f}"

        if 'req_30_days' in tips.columns:
            tips.at[idx, 'req_30_days'] = (
                f"Optimum Stock = ROS ({ros:,.2f}) × ({optimum_days:.0f} + {lead_days:.0f} lead days) = {optimum_30:,.2f}"
            )

        if 'min_threshold_qty' in tips.columns:
            tips.at[idx, 'min_threshold_qty'] = (
                f"Min Threshold Qty = Optimum ({optimum_30:,.2f}) × {min_pct_display:.0f}% = {min_threshold_qty:,.2f}"
            )

        if 'stock_variance' in tips.columns:
            tips.at[idx, 'stock_variance'] = (
                f"Stock Variance = Total Stock ({total_stock:,.0f}) - Optimum ({optimum_30:,.2f}) = {variance:,.2f}"
            )

        if 'days_of_stock' in tips.columns:
            if ros > 0:
                tips.at[idx, 'days_of_stock'] = (
                    f"Days of Stock = Total Stock ({total_stock:,.0f}) ÷ ROS ({ros:,.2f}) = {days_of_stock:,.1f}"
                )
            else:
                tips.at[idx, 'days_of_stock'] = "Days of Stock unavailable because ROS is 0"

        if 'stock_status' in tips.columns:
            tips.at[idx, 'stock_status'] = (
                f"UnderStock if {total_stock:,.0f} < {min_threshold_qty:,.2f}; "
                f"OverStock if {total_stock:,.0f} > {optimum_30:,.2f}; "
                f"else Balanced. Current: {stock_status or 'N/A'}"
            )

    return df


def _format_cell_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}"
    return str(value)


def _format_header_label(column_name: str) -> str:
    label = str(column_name).strip().replace('_', ' ')
    if not label:
        return ""

    words = [w for w in label.split(' ') if w]
    upper_keep = {
        'sih': 'SIH',
        'sit': 'SIT',
        'ros': 'ROS',
        'wh': 'WH',
        'qty': 'Qty',
        'd': 'd',
    }

    out = []
    for word in words:
        key = word.lower()
        if key in upper_keep:
            out.append(upper_keep[key])
        elif word.isupper() and len(word) <= 4:
            out.append(word)
        else:
            out.append(word.capitalize())

    return ' '.join(out)


def _row_formula_tooltips(row: pd.Series) -> dict[str, str]:
    sih = _to_float(row.get('sih', 0), 0.0)
    sit = _to_float(row.get('sit', 0), 0.0)
    total_stock = _to_float(row.get('total_stock', sih + sit), sih + sit)
    sales_90d_raw = row.get('sales_90d', None)
    if sales_90d_raw is None:
        sales_90d = _to_float(row.get('ros', 0), 0.0) * 90
    else:
        sales_90d = _to_float(sales_90d_raw, 0.0)
    ros = _to_float(row.get('ros', sales_90d / 90 if sales_90d else 0), 0.0)
    lead_days, min_pct_raw, optimum_days = _resolve_row_policy_values(row, get_shop_policy_lookup())
    optimum_30 = _to_float(
        row.get('req_30_days', row.get('optimum_stock_30d', ros * (optimum_days + lead_days))),
        ros * (optimum_days + lead_days),
    )
    variance = _to_float(row.get('stock_variance', total_stock - optimum_30), total_stock - optimum_30)
    min_pct_ratio = min_pct_raw / 100.0 if min_pct_raw > 1 else min_pct_raw
    min_pct_display = min_pct_ratio * 100
    min_threshold_qty = _to_float(row.get('min_threshold_qty', optimum_30 * min_pct_ratio), optimum_30 * min_pct_ratio)
    days_of_stock = _to_float(row.get('days_of_stock', 0), 0.0)
    status = str(row.get('stock_status', '') or '').strip()

    tips = {
        'total_stock': f"SIH ({sih:,.0f}) + SIT ({sit:,.0f}) = {total_stock:,.0f}",
        'ros': f"ROS = Sales 90d ({sales_90d:,.0f}) ÷ 90 = {ros:,.2f}",
        'req_30_days': (
            f"Optimum Stock = ROS ({ros:,.2f}) × ({optimum_days:.0f} + {lead_days:.0f} lead days) = {optimum_30:,.2f}"
        ),
        'min_threshold_qty': f"Min Threshold Qty = Optimum ({optimum_30:,.2f}) × {min_pct_display:.0f}% = {min_threshold_qty:,.2f}",
        'stock_variance': f"Stock Variance = Total Stock ({total_stock:,.0f}) - Optimum ({optimum_30:,.2f}) = {variance:,.2f}",
        'stock_status': (
            f"UnderStock if {total_stock:,.0f} < {min_threshold_qty:,.2f}; "
            f"OverStock if {total_stock:,.0f} > {optimum_30:,.2f}; else Balanced. Current: {status or 'N/A'}"
        ),
    }

    if ros > 0:
        tips['days_of_stock'] = f"Days of Stock = Total Stock ({total_stock:,.0f}) ÷ ROS ({ros:,.2f}) = {days_of_stock:,.1f}"
    else:
        tips['days_of_stock'] = "Days of Stock unavailable because ROS is 0"

    return tips


def render_dataframe_with_tooltips(df: pd.DataFrame, height: int, column_config=None):
    if df is None or df.empty:
        st.dataframe(df, use_container_width=True, height=height, column_config=column_config)
        return

    derived_columns = {
        'total_stock', 'ros', 'req_30_days', 'min_threshold_qty', 'stock_variance', 'days_of_stock', 'stock_status'
    }
    has_derived = any(c in df.columns for c in derived_columns)

    if not has_derived:
        st.dataframe(df, use_container_width=True, height=height, column_config=column_config)
        return

    table_id = f"century_formula_tbl_{uuid.uuid4().hex[:8]}"
    headers = list(df.columns)

    header_html = "".join([
        f"<th data-col='{idx}'>{html.escape(_format_header_label(str(h)))} ⬍</th>"
        for idx, h in enumerate(headers)
    ])
    row_html_list = []

    for _, row in df.iterrows():
        tips = _row_formula_tooltips(row)
        tds = []
        for col in headers:
            raw_val = row.get(col, "")
            disp = _format_cell_value(raw_val)
            title = tips.get(col, "") if col in derived_columns else ""
            title_attr = f" title='{html.escape(title)}'" if title else ""
            tds.append(f"<td{title_attr}>{html.escape(str(disp))}</td>")
        row_html_list.append("<tr>" + "".join(tds) + "</tr>")

    html_table = f"""
    <style>
      #{table_id}_wrap {{
        max-height: {height - 20}px;
        overflow: auto;
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 10px;
      }}
      #{table_id} {{
        width: 100%;
        border-collapse: collapse;
        font-family: Inter, Segoe UI, sans-serif;
        font-size: 12px;
      }}
      #{table_id} th {{
        position: sticky;
        top: 0;
        z-index: 2;
        background: #1e293b;
        color: #e2e8f0;
        padding: 7px 8px;
        text-align: left;
        border-bottom: 1px solid rgba(148,163,184,0.30);
        white-space: nowrap;
                cursor: pointer;
                user-select: none;
      }}
      #{table_id} td {{
        padding: 6px 8px;
        border-bottom: 1px solid rgba(148,163,184,0.12);
                color: #111827;
                background: #ffffff;
        white-space: nowrap;
      }}
      #{table_id} tr:hover td {{
        background: rgba(99,102,241,0.10);
      }}
    </style>
    <div id="{table_id}_wrap">
      <table id="{table_id}">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{''.join(row_html_list)}</tbody>
      </table>
    </div>
        <script>
            (() => {{
                const table = document.getElementById('{table_id}');
                if (!table) return;
                const headers = table.querySelectorAll('thead th');
                const tbody = table.querySelector('tbody');
                const sortState = {{ col: -1, asc: true }};

                const parseValue = (text) => {{
                    const raw = String(text ?? '').trim();
                    const normalized = raw.replace(/,/g, '').replace(/%/g, '');
                    const num = Number(normalized);
                    if (!Number.isNaN(num) && normalized !== '') return {{ type: 'num', val: num }};
                    return {{ type: 'text', val: raw.toLowerCase() }};
                }};

                headers.forEach((header, index) => {{
                    header.addEventListener('click', () => {{
                        const rows = Array.from(tbody.querySelectorAll('tr'));
                        const asc = sortState.col === index ? !sortState.asc : true;

                        rows.sort((a, b) => {{
                            const aCell = a.children[index] ? a.children[index].innerText : '';
                            const bCell = b.children[index] ? b.children[index].innerText : '';
                            const av = parseValue(aCell);
                            const bv = parseValue(bCell);

                            if (av.type === 'num' && bv.type === 'num') {{
                                return asc ? av.val - bv.val : bv.val - av.val;
                            }}

                            if (av.val < bv.val) return asc ? -1 : 1;
                            if (av.val > bv.val) return asc ? 1 : -1;
                            return 0;
                        }});

                        rows.forEach(row => tbody.appendChild(row));
                        sortState.col = index;
                        sortState.asc = asc;

                        headers.forEach((h, i) => {{
                            const base = h.innerText.replace('⬍', '').replace('↑', '').replace('↓', '').trim();
                            h.innerText = i === index ? `${{base}} ${{asc ? '↑' : '↓'}}` : `${{base}} ⬍`;
                        }});
                    }});
                }});
            }})();
        </script>
    """
    components.html(html_table, height=height, scrolling=False)


def get_ga_measurement_id() -> str:
        """Fetch GA4 Measurement ID from Streamlit secrets or environment."""
        try:
                analytics_cfg = st.secrets.get('analytics', {})
                measurement_id = analytics_cfg.get('ga_measurement_id', '')
                if measurement_id:
                        return measurement_id
        except Exception:
                pass
        return os.getenv('GA_MEASUREMENT_ID', '').strip()


def inject_ga_tracking(measurement_id: str, app_user_id: str = ''):
        """Inject GA4 script and capture browser + anonymized client IP.

        Notes:
        - Browser login IDs (Chrome/Firefox account email) are not accessible to web apps.
        - Sends anonymized IP and browser family as custom event params.
        """
        if not measurement_id:
                return

        safe_user_id = (app_user_id or '').replace("'", "")
        components.html(
                f"""
                <script async src="https://www.googletagmanager.com/gtag/js?id={measurement_id}"></script>
                <script>
                    window.dataLayer = window.dataLayer || [];
                    function gtag(){{dataLayer.push(arguments);}}
                    gtag('js', new Date());
                    gtag('config', '{measurement_id}', {{
                        'anonymize_ip': true,
                        'send_page_view': true
                    }});

                    (async function() {{
                        const userAgent = navigator.userAgent || '';
                        let browser = 'Other';
                        if (/Edg\//.test(userAgent)) browser = 'Edge';
                        else if (/Chrome\//.test(userAgent) && !/Edg\//.test(userAgent)) browser = 'Chrome';
                        else if (/Firefox\//.test(userAgent)) browser = 'Firefox';
                        else if (/Safari\//.test(userAgent) && !/Chrome\//.test(userAgent)) browser = 'Safari';

                        let anonymizedIp = 'unknown';
                        try {{
                            const response = await fetch('https://api.ipify.org?format=json');
                            const data = await response.json();
                            const ip = (data && data.ip) ? data.ip : '';
                            if (ip.includes('.')) {{
                                const parts = ip.split('.');
                                if (parts.length === 4) anonymizedIp = `${{parts[0]}}.${{parts[1]}}.${{parts[2]}}.0`;
                            }} else if (ip.includes(':')) {{
                                const segments = ip.split(':').slice(0, 4);
                                anonymizedIp = segments.join(':') + '::';
                            }}
                        }} catch (e) {{}}

                        gtag('event', 'century_dashboard_session_start', {{
                            browser_family: browser,
                            anonymized_ip: anonymizedIp,
                            app_user_id: '{safe_user_id}'
                        }});
                    }})();
                </script>
                """,
                height=0,
                width=0,
        )


def ensure_user_events_table():
    """Create usage-events table for management reporting."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS century_user_events (
                id BIGSERIAL PRIMARY KEY,
                event_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                event_date DATE,
                event_clock TIME,
                event_name VARCHAR(100) NOT NULL,
                session_id VARCHAR(64),
                app_user_id VARCHAR(100),
                system_login_id VARCHAR(150),
                ip_address VARCHAR(64),
                browser VARCHAR(50),
                user_agent TEXT,
                page_name VARCHAR(100) DEFAULT 'Century Stock Penetration',
                event_details JSONB
            )
        """)
        cur.execute("""
            ALTER TABLE century_user_events
            ADD COLUMN IF NOT EXISTS event_date DATE
        """)
        cur.execute("""
            ALTER TABLE century_user_events
            ADD COLUMN IF NOT EXISTS event_clock TIME
        """)
        cur.execute("""
            ALTER TABLE century_user_events
            ADD COLUMN IF NOT EXISTS system_login_id VARCHAR(150)
        """)
        cur.execute("""
            UPDATE century_user_events
            SET event_date = COALESCE(event_date, event_time::date),
                event_clock = COALESCE(event_clock, event_time::time)
            WHERE event_date IS NULL OR event_clock IS NULL
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_century_user_events_time
            ON century_user_events(event_time DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_century_user_events_name
            ON century_user_events(event_name)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_century_user_events_user
            ON century_user_events(app_user_id)
        """)
        conn.commit()
        cur.close()


def get_client_context() -> dict:
    """Capture request metadata when available from Streamlit context."""
    headers = {}
    try:
        headers_obj = getattr(st, 'context', None)
        if headers_obj and hasattr(headers_obj, 'headers') and headers_obj.headers:
            headers = dict(headers_obj.headers)
    except Exception:
        headers = {}

    user_agent = headers.get('User-Agent', '')
    browser = 'Other'
    ua = user_agent.lower()
    if 'edg/' in ua:
        browser = 'Edge'
    elif 'chrome/' in ua and 'edg/' not in ua:
        browser = 'Chrome'
    elif 'firefox/' in ua:
        browser = 'Firefox'
    elif 'safari/' in ua and 'chrome/' not in ua:
        browser = 'Safari'

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
        'browser': browser,
        'user_agent': user_agent,
        'auth_login': auth_login
    }


def get_local_ipv4() -> str:
    """Best-effort local IPv4 for on-prem dashboard access and usage reporting."""
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

    return ''


def get_os_login_identity() -> str:
    """Best-effort OS login identity, preferring Windows DOMAIN\\USERNAME."""
    domain = str(os.getenv('USERDOMAIN', '') or '').strip()
    username = str(os.getenv('USERNAME', '') or os.getenv('USER', '') or '').strip()

    if domain and username:
        return f"{domain}\\{username}"
    if username:
        return username

    try:
        login_name = str(getpass.getuser() or '').strip()
        if login_name:
            return login_name
    except Exception:
        pass

    return ''


def get_app_user_id() -> str:
    """Get application user id from session state when available."""
    app_user = st.session_state.get('user')
    if isinstance(app_user, dict):
        return str(app_user.get('employee_id') or app_user.get('username') or '')
    return ''


def get_system_login_id() -> str:
    captured_login = str(st.session_state.get('captured_login_id', '')).strip()
    if captured_login:
        return captured_login

    context = get_client_context()
    if context.get('auth_login'):
        return str(context['auth_login'])

    app_user_id = get_app_user_id()
    if app_user_id:
        return app_user_id

    os_login = get_os_login_identity()
    if os_login:
        return os_login

    return ''


def ensure_login_id_captured() -> bool:
    """Ensure each session has a user-specific login id captured without manual input."""
    if 'captured_login_id' not in st.session_state:
        st.session_state.captured_login_id = ''

    if 'captured_ip_address' not in st.session_state:
        st.session_state.captured_ip_address = ''

    # Auto-seed from authenticated headers/app user/OS login when available
    if not st.session_state.captured_login_id:
        auto_login = ''
        context = get_client_context()
        if context.get('auth_login'):
            auto_login = str(context['auth_login']).strip()
        elif get_app_user_id():
            auto_login = get_app_user_id().strip()
        else:
            auto_login = get_os_login_identity().strip()

        if auto_login:
            st.session_state.captured_login_id = auto_login

    if not st.session_state.captured_ip_address:
        context = get_client_context()
        ip_address = str(context.get('ip_address', '') or '').strip() or get_local_ipv4().strip()
        if ip_address:
            st.session_state.captured_ip_address = ip_address

    if st.session_state.captured_login_id:
        tracking_ip = str(st.session_state.get('captured_ip_address', '') or '').strip() or 'unknown'
        st.success(f"Tracking as: {st.session_state.captured_login_id}")
        st.caption(f"IP Address: {tracking_ip}")
        return True

    fallback_name = get_os_login_identity().strip()
    if fallback_name:
        st.session_state.captured_login_id = fallback_name
        tracking_ip = str(st.session_state.get('captured_ip_address', '') or '').strip() or 'unknown'
        st.success(f"Tracking as: {fallback_name}")
        st.caption(f"IP Address: {tracking_ip}")
        return True

    st.info("Tracking will use available system identity and IP automatically.")
    return True


def backfill_current_session_identity():
    """Update already-logged events in the active session that still show unknown identity values."""
    try:
        ensure_user_events_table()
        if 'event_session_id' not in st.session_state:
            return

        session_id = str(st.session_state.get('event_session_id', '') or '').strip()
        system_login_id = get_system_login_id().strip()
        context = get_client_context()
        ip_address = str(context.get('ip_address', '') or '').strip() or str(st.session_state.get('captured_ip_address', '') or '').strip() or get_local_ipv4().strip()

        if not session_id or (not system_login_id and not ip_address):
            return

        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE century_user_events
                SET
                    system_login_id = CASE
                        WHEN %s <> '' AND COALESCE(NULLIF(system_login_id, ''), 'unknown') = 'unknown' THEN %s
                        ELSE system_login_id
                    END,
                    ip_address = CASE
                        WHEN %s <> '' AND COALESCE(NULLIF(ip_address, ''), 'unknown') = 'unknown' THEN %s
                        ELSE ip_address
                    END
                WHERE session_id = %s
                """,
                (
                    system_login_id,
                    system_login_id,
                    ip_address,
                    ip_address,
                    session_id,
                )
            )
            conn.commit()
            cur.close()
    except Exception as e:
        logger.warning(f"Session identity backfill failed: {e}")


def log_user_event(event_name: str, event_details: dict | None = None):
    """Insert one user event row into century_user_events."""
    try:
        ensure_user_events_table()
        if 'event_session_id' not in st.session_state:
            st.session_state.event_session_id = uuid.uuid4().hex

        context = get_client_context()
        ip_address = str(context.get('ip_address', '') or '').strip() or str(st.session_state.get('captured_ip_address', '') or '').strip() or get_local_ipv4()
        payload = dict(event_details or {})
        if get_system_login_id().strip() and 'system_login_id' not in payload:
            payload['system_login_id'] = get_system_login_id().strip()
        if ip_address and 'ip_address' not in payload:
            payload['ip_address'] = ip_address

        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO century_user_events
                (event_time, event_date, event_clock, event_name, session_id, app_user_id, system_login_id, ip_address, browser, user_agent, event_details)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    datetime.now(),
                    datetime.now().date(),
                    datetime.now().time().replace(microsecond=0),
                    event_name,
                    st.session_state.event_session_id,
                    get_app_user_id(),
                    get_system_login_id(),
                    ip_address,
                    context.get('browser', 'Other'),
                    context.get('user_agent', ''),
                    Json(payload)
                )
            )
            conn.commit()
            cur.close()
    except Exception as e:
        logger.warning(f"Event logging failed for '{event_name}': {e}")


def get_user_events_report(days_back: int = 30) -> pd.DataFrame:
    """Detailed events report for management export."""
    ensure_user_events_table()
    with get_db_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT
                event_time,
                event_date,
                event_clock,
                event_name,
                COALESCE(NULLIF(app_user_id, ''), 'anonymous') AS app_user_id,
                COALESCE(NULLIF(system_login_id, ''), 'unknown') AS system_login_id,
                COALESCE(NULLIF(ip_address, ''), 'unknown') AS ip_address,
                browser,
                event_details
            FROM century_user_events
            WHERE event_time >= NOW() - (%s || ' days')::INTERVAL
            ORDER BY event_time DESC
            """,
            (str(days_back),)
        )
        rows = cur.fetchall()
        cur.close()
        return pd.DataFrame(rows)


def get_user_events_summary(days_back: int = 30) -> dict:
    """Summary metrics for management usage report."""
    ensure_user_events_table()
    with get_db_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT
                COUNT(*) AS total_events,
                COUNT(DISTINCT session_id) AS total_sessions,
                COUNT(DISTINCT NULLIF(app_user_id, '')) AS known_users,
                COUNT(DISTINCT COALESCE(NULLIF(ip_address, ''), 'unknown')) AS distinct_ips
            FROM century_user_events
            WHERE event_time >= NOW() - (%s || ' days')::INTERVAL
            """,
            (str(days_back),)
        )
        summary = dict(cur.fetchone())
        cur.close()
        return summary


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data(ttl=600)
def get_overview_metrics(status_threshold: int = 10, shop_filter=None):
    """Get high-level overview metrics with configurable threshold"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        where_clause = "WHERE shop_code = %s" if shop_filter else ""
        params = [status_threshold, status_threshold, status_threshold]
        if shop_filter:
            params.extend([shop_filter, shop_filter])
        query = """
            WITH item_status_summary AS (
                -- For each item, calculate percentage of shops in each status
                SELECT 
                    item_code,
                    COUNT(*) as total_shops_for_item,
                    SUM(CASE WHEN stock_status = 'UnderStock' THEN 1 ELSE 0 END) as understock_count,
                    SUM(CASE WHEN stock_status = 'OverStock' THEN 1 ELSE 0 END) as overstock_count,
                    SUM(CASE WHEN stock_status = 'Balanced' THEN 1 ELSE 0 END) as balanced_count,
                    -- Calculate percentages
                    ROUND(100.0 * SUM(CASE WHEN stock_status = 'UnderStock' THEN 1 ELSE 0 END) / COUNT(*), 1) as understock_pct,
                    ROUND(100.0 * SUM(CASE WHEN stock_status = 'OverStock' THEN 1 ELSE 0 END) / COUNT(*), 1) as overstock_pct,
                    ROUND(100.0 * SUM(CASE WHEN stock_status = 'Balanced' THEN 1 ELSE 0 END) / COUNT(*), 1) as balanced_pct
                FROM mv_century_penetration
                {where_clause}
                GROUP BY item_code
            ),
            item_dominant_status AS (
                -- Classify item based on configurable threshold
                SELECT 
                    item_code,
                    CASE 
                        WHEN understock_pct >= %s THEN 'UnderStock'
                        WHEN overstock_pct >= %s THEN 'OverStock'
                        WHEN balanced_pct >= %s THEN 'Balanced'
                        ELSE 'Mixed' -- Items with no dominant status
                    END as dominant_status
                FROM item_status_summary
            ),
            overall_stats AS (
                SELECT 
                    COUNT(DISTINCT item_code) as total_items,
                    ROUND(SUM(sih), 0) as total_sih,
                    ROUND(SUM(sit), 0) as total_sit,
                    ROUND(SUM(req_30_days), 0) as total_requirement,
                    ROUND(AVG(ros), 2) as avg_ros,
                    COUNT(DISTINCT shop_code) as total_shops
                FROM mv_century_penetration
                {where_clause}
            )
            SELECT 
                o.total_items,
                o.total_shops,
                SUM(CASE WHEN ids.dominant_status = 'UnderStock' THEN 1 ELSE 0 END) as understock_items,
                SUM(CASE WHEN ids.dominant_status = 'OverStock' THEN 1 ELSE 0 END) as overstock_items,
                SUM(CASE WHEN ids.dominant_status = 'Balanced' THEN 1 ELSE 0 END) as balanced_items,
                SUM(CASE WHEN ids.dominant_status = 'Mixed' THEN 1 ELSE 0 END) as mixed_items,
                o.total_sih,
                o.total_sit,
                o.total_requirement,
                o.avg_ros
            FROM item_dominant_status ids
            CROSS JOIN overall_stats o
            GROUP BY o.total_items, o.total_shops, o.total_sih, o.total_sit, o.total_requirement, o.avg_ros
        """.format(where_clause=where_clause)
        cursor.execute(query, tuple(params))
        return dict(cursor.fetchone())


@st.cache_data(ttl=600)
def get_shop_wise_summary(shop_filter=None):
    """Get shop-wise stock status summary"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        where_clause = "WHERE shop_code = %s" if shop_filter else ""
        query = """
            SELECT 
                shop_code,
                COUNT(*) as total_items,
                SUM(CASE WHEN stock_status = 'UnderStock' THEN 1 ELSE 0 END) as understock,
                SUM(CASE WHEN stock_status = 'OverStock' THEN 1 ELSE 0 END) as overstock,
                SUM(CASE WHEN stock_status = 'Balanced' THEN 1 ELSE 0 END) as balanced,
                ROUND(SUM(sih), 0) as total_sih,
                ROUND(SUM(sit), 0) as total_sit,
                ROUND(SUM(req_30_days), 0) as total_requirement,
                ROUND(AVG(ros), 2) as avg_ros
            FROM mv_century_penetration
            {where_clause}
            GROUP BY shop_code
            ORDER BY understock DESC
        """.format(where_clause=where_clause)
        if shop_filter:
            cursor.execute(query, (shop_filter,))
        else:
            cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())


@st.cache_data(ttl=600)
def get_latest_whstock_upload_date():
    """Get latest whstock upload date once per cache window."""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT MAX(upload_date) as max_date FROM whstock")
        max_date_row = cursor.fetchone()
        return max_date_row['max_date'] if max_date_row and max_date_row['max_date'] else None


@st.cache_data(ttl=600)
def get_understock_items(limit=2000, shop_filter=None):
    """Get top understock items with warehouse stock"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        max_date = get_latest_whstock_upload_date()
        
        shop_clause = "AND m.shop_code = %s" if shop_filter else ""
        query = f"""
            WITH latest_whstock AS (
                SELECT DISTINCT ON (vc_item_code, wh_code)
                    vc_item_code,
                    wh_code,
                    wh_name,
                    balance_qty,
                    upload_date
                FROM whstock
                WHERE upload_date = %s
                ORDER BY vc_item_code, wh_code, upload_date DESC
            )
            SELECT 
                m.item_code,
                m.item_name,
                m.shop_code,
                m.dept,
                ROUND(m.sih, 0) as sih,
                ROUND(m.sit, 0) as sit,
                ROUND(m.total_stock, 0) as total_stock,
                ROUND(m.sales_30d, 0) as "Last 30d",
                ROUND(m.sales_60d, 0) as "Last 60d",
                ROUND(m.sales_90d, 0) as "Last 90d",
                ROUND(m.ros, 2) as ros,
                ROUND(m.req_30_days, 0) as req_30_days,
                ROUND(m.stock_variance, 0) as stock_variance,
                ROUND(m.pack_size, 0) as pack_size,
                ROUND(m.days_of_stock, 1) as days_of_stock,
                COALESCE(SUM(w.balance_qty), 0) as total_wh_stock,
                COALESCE(SUM(CASE WHEN w.wh_code = 'TS' THEN w.balance_qty ELSE 0 END), 0) as wh_tema,
                COALESCE(SUM(CASE WHEN w.wh_code = 'KA' THEN w.balance_qty ELSE 0 END), 0) as wh_kumasi
            FROM mv_century_penetration m
            LEFT JOIN latest_whstock w ON m.item_code = w.vc_item_code
            WHERE m.stock_status = 'UnderStock'
              {shop_clause}
            GROUP BY m.item_code, m.item_name, m.shop_code, m.dept, m.sih, m.sit, m.total_stock,
                     m.sales_30d, m.sales_60d, m.sales_90d, m.ros, m.req_30_days, 
                     m.stock_variance, m.pack_size, m.days_of_stock
            ORDER BY m.stock_variance ASC
            LIMIT {limit}
        """
        if shop_filter:
            cursor.execute(query, (max_date, shop_filter))
        else:
            cursor.execute(query, (max_date,))
        return pd.DataFrame(cursor.fetchall())


@st.cache_data(ttl=600)
def get_overstock_items(limit=50, shop_filter=None):
    """Get top overstock items with warehouse stock"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        max_date = get_latest_whstock_upload_date()
        
        shop_clause = "AND m.shop_code = %s" if shop_filter else ""
        query = f"""
            WITH latest_whstock AS (
                SELECT DISTINCT ON (vc_item_code, wh_code)
                    vc_item_code,
                    wh_code,
                    wh_name,
                    balance_qty,
                    upload_date
                FROM whstock
                WHERE upload_date = %s
                ORDER BY vc_item_code, wh_code, upload_date DESC
            )
            SELECT 
                m.item_code,
                m.item_name,
                m.shop_code,
                m.dept,
                ROUND(m.sih, 0) as sih,
                ROUND(m.sit, 0) as sit,
                ROUND(m.total_stock, 0) as total_stock,
                ROUND(m.sales_30d, 0) as "Last 30d",
                ROUND(m.sales_60d, 0) as "Last 60d",
                ROUND(m.sales_90d, 0) as "Last 90d",
                ROUND(m.ros, 2) as ros,
                ROUND(m.req_30_days, 0) as req_30_days,
                ROUND(m.stock_variance, 0) as stock_variance,
                ROUND(m.pack_size, 0) as pack_size,
                ROUND(m.days_of_stock, 1) as days_of_stock,
                COALESCE(SUM(w.balance_qty), 0) as total_wh_stock,
                COALESCE(SUM(CASE WHEN w.wh_code = 'TS' THEN w.balance_qty ELSE 0 END), 0) as wh_tema,
                COALESCE(SUM(CASE WHEN w.wh_code = 'KA' THEN w.balance_qty ELSE 0 END), 0) as wh_kumasi
            FROM mv_century_penetration m
            LEFT JOIN latest_whstock w ON m.item_code = w.vc_item_code
            WHERE m.stock_status = 'OverStock'
              {shop_clause}
            GROUP BY m.item_code, m.item_name, m.shop_code, m.dept, m.sih, m.sit, m.total_stock,
                     m.sales_30d, m.sales_60d, m.sales_90d, m.ros, m.req_30_days, 
                     m.stock_variance, m.pack_size, m.days_of_stock
            ORDER BY m.stock_variance DESC
            LIMIT {limit}
        """
        if shop_filter:
            cursor.execute(query, (max_date, shop_filter))
        else:
            cursor.execute(query, (max_date,))
        return pd.DataFrame(cursor.fetchall())


@st.cache_data(ttl=600)
def get_department_analysis(shop_filter=None):
    """Get department-wise analysis"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        where_clause = "WHERE shop_code = %s" if shop_filter else ""
        query = """
            SELECT 
                dept,
                COUNT(DISTINCT item_code) as total_items,
                SUM(CASE WHEN stock_status = 'UnderStock' THEN 1 ELSE 0 END) as understock,
                SUM(CASE WHEN stock_status = 'OverStock' THEN 1 ELSE 0 END) as overstock,
                ROUND(SUM(sih), 0) as total_sih,
                ROUND(SUM(sit), 0) as total_sit,
                ROUND(SUM(req_30_days), 0) as total_requirement,
                ROUND(AVG(ros), 2) as avg_ros
            FROM mv_century_penetration
            {where_clause}
            GROUP BY dept
            ORDER BY understock DESC
        """.format(where_clause=where_clause)
        if shop_filter:
            cursor.execute(query, (shop_filter,))
        else:
            cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())


@st.cache_data(ttl=600)
def get_critical_items(shop_filter=None):
    """Get critical items (no stock, high demand) with warehouse stock"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        max_date = get_latest_whstock_upload_date()
        
        shop_clause = "AND m.shop_code = %s" if shop_filter else ""
        query = """
            WITH latest_whstock AS (
                SELECT DISTINCT ON (vc_item_code, wh_code)
                    vc_item_code,
                    wh_code,
                    wh_name,
                    balance_qty,
                    upload_date
                FROM whstock
                WHERE upload_date = %s
                ORDER BY vc_item_code, wh_code, upload_date DESC
            )
            SELECT 
                m.item_code,
                m.item_name,
                m.shop_code,
                m.dept,
                ROUND(m.sih, 0) as sih,
                ROUND(m.sit, 0) as sit,
                ROUND(m.sales_30d, 0) as "Last 30d",
                ROUND(m.sales_60d, 0) as "Last 60d",
                ROUND(m.sales_90d, 0) as "Last 90d",
                ROUND(m.ros, 2) as ros,
                ROUND(m.req_30_days, 0) as req_30_days,
                ROUND(m.pack_size, 0) as pack_size,
                COALESCE(SUM(w.balance_qty), 0) as total_wh_stock,
                COALESCE(SUM(CASE WHEN w.wh_code = 'TS' THEN w.balance_qty ELSE 0 END), 0) as wh_tema,
                COALESCE(SUM(CASE WHEN w.wh_code = 'KA' THEN w.balance_qty ELSE 0 END), 0) as wh_kumasi
            FROM mv_century_penetration m
            LEFT JOIN latest_whstock w ON m.item_code = w.vc_item_code
            WHERE m.total_stock = 0
              AND m.ros > 1
              {shop_clause}
            GROUP BY m.item_code, m.item_name, m.shop_code, m.dept, m.sih, m.sit,
                     m.sales_30d, m.sales_60d, m.sales_90d, m.ros, m.req_30_days, m.pack_size
            ORDER BY m.ros DESC
            LIMIT 50
        """.format(shop_clause=shop_clause)
        if shop_filter:
            cursor.execute(query, (max_date, shop_filter))
        else:
            cursor.execute(query, (max_date,))
        return pd.DataFrame(cursor.fetchall())


@st.cache_data(ttl=600)
def get_slow_moving_items(shop_filter=None):
    """Get slow moving items with warehouse stock"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        max_date = get_latest_whstock_upload_date()
        
        shop_clause = "AND m.shop_code = %s" if shop_filter else ""
        query = """
            WITH latest_whstock AS (
                SELECT DISTINCT ON (vc_item_code, wh_code)
                    vc_item_code,
                    wh_code,
                    wh_name,
                    balance_qty,
                    upload_date
                FROM whstock
                WHERE upload_date = %s
                ORDER BY vc_item_code, wh_code, upload_date DESC
            )
            SELECT 
                m.item_code,
                m.item_name,
                m.shop_code,
                m.dept,
                ROUND(m.sih, 0) as sih,
                ROUND(m.sit, 0) as sit,
                ROUND(m.sales_30d, 0) as "Last 30d",
                ROUND(m.sales_60d, 0) as "Last 60d",
                ROUND(m.sales_90d, 0) as "Last 90d",
                ROUND(m.ros, 2) as ros,
                ROUND(m.pack_size, 0) as pack_size,
                ROUND(m.days_of_stock, 1) as days_of_stock,
                m.last_sale_date,
                COALESCE(SUM(w.balance_qty), 0) as total_wh_stock,
                COALESCE(SUM(CASE WHEN w.wh_code = 'TS' THEN w.balance_qty ELSE 0 END), 0) as wh_tema,
                COALESCE(SUM(CASE WHEN w.wh_code = 'KA' THEN w.balance_qty ELSE 0 END), 0) as wh_kumasi
            FROM mv_century_penetration m
            LEFT JOIN latest_whstock w ON m.item_code = w.vc_item_code
            WHERE m.ros < 0.5
              AND m.sih > 0
              {shop_clause}
            GROUP BY m.item_code, m.item_name, m.shop_code, m.dept, m.sih, m.sit,
                     m.sales_30d, m.sales_60d, m.sales_90d, m.ros, m.pack_size, 
                     m.days_of_stock, m.last_sale_date
            ORDER BY m.ros ASC, m.days_of_stock DESC
            LIMIT 50
        """.format(shop_clause=shop_clause)
        if shop_filter:
            cursor.execute(query, (max_date, shop_filter))
        else:
            cursor.execute(query, (max_date,))
        return pd.DataFrame(cursor.fetchall())


@st.cache_data(ttl=600)
def get_available_shops(view_name=None, search_term=None):
    """Get available shops in alphabetical order, with optional view-specific filtering."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        where_clause = "WHERE shop_code IS NOT NULL"
        params = []

        view_key = (view_name or "").lower()

        if "understock" in view_key:
            where_clause += " AND stock_status = 'UnderStock'"
        elif "overstock" in view_key:
            where_clause += " AND stock_status = 'OverStock'"
        elif "critical" in view_key:
            where_clause += " AND total_stock = 0 AND ros > 1"
        elif "slow moving" in view_key:
            where_clause += " AND ros < 0.5 AND sih > 0"

        if search_term:
            where_clause += " AND (LOWER(item_code) LIKE LOWER(%s) OR LOWER(item_name) LIKE LOWER(%s))"
            like_term = f"%{search_term}%"
            params.extend([like_term, like_term])

        query = f"""
            SELECT DISTINCT shop_code
            FROM mv_century_penetration
            {where_clause}
            ORDER BY shop_code
        """
        cursor.execute(query, tuple(params))
        return [row[0] for row in cursor.fetchall()]


@st.cache_data(ttl=600)
def search_items(search_term, shop_filter=None):
    """Search items by code or name"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = """
            SELECT 
                item_code,
                item_name,
                shop_code,
                dept,
                stock_status,
                ROUND(sih, 0) as sih,
                ROUND(sit, 0) as sit,
                ROUND(total_stock, 0) as total_stock,
                ROUND(sales_30d, 0) as "Last 30d",
                ROUND(sales_60d, 0) as "Last 60d",
                ROUND(sales_90d, 0) as "Last 90d",
                ROUND(ros, 2) as ros,
                ROUND(req_30_days, 0) as req_30_days,
                ROUND(stock_variance, 0) as stock_variance,
                ROUND(pack_size, 0) as pack_size,
                ROUND(days_of_stock, 1) as days_of_stock,
                ROUND(selling_price, 2) as selling_price
            FROM mv_century_penetration
                 WHERE (LOWER(item_code) LIKE LOWER(%s)
                     OR LOWER(item_name) LIKE LOWER(%s))
                AND (%s IS NULL OR shop_code = %s)
            ORDER BY ros DESC
            LIMIT 100
        """
        like_term = f"%{search_term}%"
        cursor.execute(query, (like_term, like_term, shop_filter, shop_filter))
        return pd.DataFrame(cursor.fetchall())


@st.cache_data(ttl=600)
def get_all_items(limit=2000, shop_filter=None):
    """Get all items regardless of stock status."""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = f"""
            SELECT
                item_code,
                item_name,
                shop_code,
                dept,
                stock_status,
                ROUND(sih, 0) as sih,
                ROUND(sit, 0) as sit,
                ROUND(total_stock, 0) as total_stock,
                ROUND(sales_30d, 0) as "Last 30d",
                ROUND(sales_60d, 0) as "Last 60d",
                ROUND(sales_90d, 0) as "Last 90d",
                ROUND(ros, 2) as ros,
                ROUND(req_30_days, 0) as req_30_days,
                ROUND(stock_variance, 0) as stock_variance,
                ROUND(pack_size, 0) as pack_size,
                ROUND(days_of_stock, 1) as days_of_stock,
                ROUND(selling_price, 2) as selling_price
            FROM mv_century_penetration
            WHERE (%s IS NULL OR shop_code = %s)
            ORDER BY shop_code, ros DESC, item_code
            LIMIT {limit}
        """
        cursor.execute(query, (shop_filter, shop_filter))
        return pd.DataFrame(cursor.fetchall())


@st.cache_data(ttl=600)
def get_all_items_download_full():
    """Get full All Items dataset for download (no filters, no row limit)."""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = """
            SELECT
                item_code,
                item_name,
                shop_code,
                dept,
                stock_status,
                ROUND(sih, 0) as sih,
                ROUND(sit, 0) as sit,
                ROUND(total_stock, 0) as total_stock,
                ROUND(sales_30d, 0) as "Last 30d",
                ROUND(sales_60d, 0) as "Last 60d",
                ROUND(sales_90d, 0) as "Last 90d",
                ROUND(ros, 2) as ros,
                ROUND(req_30_days, 0) as req_30_days,
                ROUND(stock_variance, 0) as stock_variance,
                ROUND(pack_size, 0) as pack_size,
                ROUND(days_of_stock, 1) as days_of_stock,
                ROUND(selling_price, 2) as selling_price
            FROM mv_century_penetration
            ORDER BY shop_code, ros DESC, item_code
        """
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())


@st.cache_data(ttl=60)
def get_target_vs_achieve(limit=None):
    """Get target vs achieve snapshot from the shared view."""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
        query = f"""
            SELECT
                item_code,
                item_name,
                report_month,
                ROUND(item_retail_price_avg, 2) AS item_retail_price_avg,
                ROUND(target_sales_month_pcs, 2) AS target_sales_month_pcs,
                ROUND(target_value_month_ghc, 2) AS target_value_month_ghc,
                ROUND(actual_sales_month_qty, 2) AS actual_sales_month_qty,
                ROUND(actual_sales_month_value_net, 2) AS actual_sales_month_value_net,
                ROUND(shortfall_excess_qty, 2) AS shortfall_excess_qty,
                ROUND(shortfall_excess_value, 2) AS shortfall_excess_value,
                ROUND(wh_stock_tema, 2) AS wh_stock_tema,
                ROUND(wh_stock_kumasi, 2) AS wh_stock_kumasi,
                ROUND(COALESCE(wh_stock_tema, 0) + COALESCE(wh_stock_kumasi, 0), 2) AS stock_in_hand,
                remark,
                report_as_of_date
            FROM mv_target_vs_achieve_century
            ORDER BY CASE WHEN remark = 'Behind Target' THEN 1 ELSE 0 END DESC,
                     shortfall_excess_value DESC,
                     item_code
            {limit_clause}
        """
        cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())


def prepare_target_vs_achieve_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return Target vs Achieve table in requested column order and labels."""
    if df is None or df.empty:
        return pd.DataFrame()

    working_df = df.copy()
    actual_qty = pd.to_numeric(working_df.get("actual_sales_month_qty"), errors="coerce").fillna(0)
    target_qty = pd.to_numeric(working_df.get("target_sales_month_pcs"), errors="coerce").fillna(0)
    actual_value = pd.to_numeric(working_df.get("actual_sales_month_value_net"), errors="coerce").fillna(0)
    target_value = pd.to_numeric(working_df.get("target_value_month_ghc"), errors="coerce").fillna(0)
    # User-requested definition: shortfall/access qty = actual qty - target/month
    working_df["shortfall_excess_qty"] = actual_qty - target_qty
    # Use the same direction for value variance: actual value - target value
    working_df["shortfall_excess_value"] = actual_value - target_value

    display_df = working_df[[
        "item_code",
        "item_name",
        "report_month",
        "item_retail_price_avg",
        "target_sales_month_pcs",
        "stock_in_hand",
        "actual_sales_month_qty",
        "shortfall_excess_qty",
        "target_value_month_ghc",
        "actual_sales_month_value_net",
        "shortfall_excess_value",
    ]].copy()

    display_df = display_df.rename(columns={
        "item_code": "ITEM CODE",
        "item_name": "ITEM NAME",
        "report_month": "REPORT MONTH",
        "item_retail_price_avg": "RETAIL PRICE AVG",
        "target_sales_month_pcs": "TARGET/MONTH (PCS)",
        "stock_in_hand": "STOCK IN HAND",
        "actual_sales_month_qty": "ACTUAL QTY",
        "shortfall_excess_qty": "SHORTFALL/ACCESS QTY",
        "target_value_month_ghc": "TARGET VALUE(GHC)",
        "actual_sales_month_value_net": "ACTUAL VALUE NET",
        "shortfall_excess_value": "SHORTFALL/ACCESS VALUE",
    })
    return display_df


def style_target_vs_achieve_table(display_df: pd.DataFrame):
    """Apply alignment and variance coloring for target vs achieve table."""
    def variance_style(value):
        try:
            num = float(value)
        except Exception:
            return ""

        if num > 0:
            return "background-color: #dcfce7; color: #166534; font-weight: 700; text-align: center;"
        if num < 0:
            return "background-color: #fee2e2; color: #991b1b; font-weight: 700; text-align: center;"
        return "background-color: #e2e8f0; color: #1e293b; font-weight: 700; text-align: center;"

    styled_df = display_df.style.set_properties(**{"text-align": "center"})
    styled_df = styled_df.set_properties(subset=["ITEM CODE", "ITEM NAME"], **{"text-align": "left"})
    styled_df = styled_df.applymap(
        variance_style,
        subset=["SHORTFALL/ACCESS QTY", "SHORTFALL/ACCESS VALUE"],
    )
    return styled_df


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    # PAGE CONFIGURATION
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for tab management
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0

    sync_dashboard_cache_with_database()

    if 'event_session_id' not in st.session_state:
        st.session_state.event_session_id = uuid.uuid4().hex

    # Always ensure schema is current (safe and idempotent)
    ensure_user_events_table()

    # Initialize GA tracking once per session
    if 'ga_initialized' not in st.session_state:
        st.session_state.ga_initialized = False
    if not st.session_state.ga_initialized:
        ga_measurement_id = get_ga_measurement_id()
        app_user = st.session_state.get('user')
        app_user_id = ''
        if isinstance(app_user, dict):
            app_user_id = str(app_user.get('employee_id') or app_user.get('username') or '')
        inject_ga_tracking(ga_measurement_id, app_user_id=app_user_id)
        st.session_state.ga_initialized = True
    
    # CUSTOM CSS
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: rgba(255,255,255,0.05);
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # HEADER
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; align-items: center; gap: 15px;">
            <img src="{Config.LOGO_URL}" style="width: 50px; height: 50px; border-radius: 8px;">
            <div>
                <h1 style="margin: 0;"> Century Stock Penetration Dashboard</h1>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab_options = [
        "🏪 Shop Analysis",
        "⚠️ UnderStock Items",
        "📦 OverStock Items",
        "🚨 Critical Items",
        "🐌 Slow Moving",
        "📂 Department Analysis",
        "🎯 Target vs Achieve",
        "📋 All Items"
    ]

    # SIDEBAR
    with st.sidebar:
        st.markdown("### 👤 User Identity")
        login_ready = ensure_login_id_captured()
        backfill_current_session_identity()
        st.markdown("---")

        st.markdown("### 🔍 Filters & Search")
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            log_user_event('refresh_clicked')
            # Refresh the target vs achieve matview so report_month advances with CURRENT_DATE
            try:
                with get_db_connection() as _conn:
                    _conn.autocommit = True
                    with _conn.cursor() as _cur:
                        _cur.execute("REFRESH MATERIALIZED VIEW mv_target_vs_achieve_century;")
            except Exception:
                pass
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Threshold control
        st.markdown("### ⚙️ Settings")
        status_threshold = st.slider(
            "Status Classification Threshold",
            min_value=0,
            max_value=100,
            value=10,
            step=5,
            help="Minimum % of shops required to classify an item's dominant status"
        )
        st.caption(f"Items are classified as UnderStock/OverStock/Balanced if they have that status in ≥{status_threshold}% of shops")
        
        st.markdown("---")
        
        # Search
        search_term = st.text_input("🔎 Search Item", placeholder="Item code or name...")

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **Century Brand Stock Analysis**
        
        - 📦 Stock levels (SIH + SIT)
        - 📈 Sales trends (30/60/90/365 days)
        - ⚡ Rate of Sales (ROS)
        - 🎯 30-day requirements
        - ⚠️ Stock status alerts
        """)

        st.markdown("---")
        with st.expander("📊 Management Usage Report", expanded=False):
            current_detected_login = get_system_login_id().strip() or 'unknown'
            current_detected_ip = str(st.session_state.get('captured_ip_address', '') or '').strip() or get_local_ipv4().strip() or 'unknown'
            st.caption(f"Auto-detected system login: {current_detected_login}")
            st.caption(f"Auto-detected IP: {current_detected_ip}")

            report_days = st.slider(
                "Report period (days)",
                min_value=1,
                max_value=90,
                value=30,
                step=1,
                key='events_report_days'
            )

            summary = get_user_events_summary(report_days)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Events", f"{summary.get('total_events', 0):,}")
                st.metric("Sessions", f"{summary.get('total_sessions', 0):,}")
            with c2:
                st.metric("Known Users", f"{summary.get('known_users', 0):,}")
                st.metric("Distinct IPs", f"{summary.get('distinct_ips', 0):,}")

            events_df = get_user_events_report(report_days)
            st.caption(f"Showing last {report_days} days")
            st.dataframe(events_df.head(200), use_container_width=True, height=220)

            if not events_df.empty:
                report_csv = events_df.to_csv(index=False)
                if st.download_button(
                    label="📥 Download Full Usage Report",
                    data=report_csv,
                    file_name=f"century_usage_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key='download_usage_report_btn'
                ):
                    log_user_event('usage_report_downloaded', {'days': report_days, 'rows': len(events_df)})

    if 'app_open_logged' not in st.session_state:
        log_user_event('app_open', {
            'page': 'century_dashboard',
            'login_ready': bool(login_ready)
        })
        st.session_state.app_open_logged = True

    def render_tab_shop_filter(view_name: str, key_suffix: str, search_text: str | None = None):
        st.markdown("#### 🏪 Shop Code Filter")
        try:
            shops_for_view = get_available_shops(
                view_name=None,
                search_term=search_text if search_text else None
            )
        except Exception:
            shops_for_view = []

        shop_options = ["All Shops"] + sorted([s for s in shops_for_view if s])
        selected_shop = st.selectbox(
            "Shop Code",
            options=shop_options,
            index=0,
            key=f"shop_code_filter_{key_suffix}",
            help="Shows all shops, regardless of current stock status view"
        )
        return None if selected_shop == "All Shops" else selected_shop
    
    # MAIN CONTENT
    if search_term:
        if st.session_state.get('last_search_term') != search_term:
            log_user_event('search_performed', {'search_term': search_term})
            st.session_state.last_search_term = search_term
        # SEARCH RESULTS
        st.markdown("### 🔍 Search Results")
        shop_filter_value = render_tab_shop_filter("🔍 Search Results", "search", search_term)
        search_results = search_items(search_term, shop_filter=shop_filter_value)
        
        if not search_results.empty:
            render_dataframe_with_tooltips(
                search_results,
                height=500,
                column_config=get_column_config()
            )
            
            # Download button
            try:
                # More robust CSV conversion: convert to string representation first
                csv_buffer = io.StringIO()
                for idx, row in search_results.iterrows():
                    if idx == 0:
                        csv_buffer.write(','.join(str(col) for col in search_results.columns) + '\n')
                    csv_buffer.write(','.join(f'"{str(val).replace(chr(34), chr(34)*2)}"' for val in row) + '\n')
                csv = csv_buffer.getvalue()
            except Exception:
                # Fallback: convert all to string dtype first
                csv_df = search_results.astype(str)
                csv_df.columns = csv_df.columns.astype(str)
                csv = csv_df.to_csv(index=False)
            if st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name=f"century_search_{search_term}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            ):
                log_user_event('search_results_downloaded', {'search_term': search_term, 'rows': len(search_results)})
        else:
            st.warning("No items found matching your search.")
    
    else:
        # Initialize active tab in session state
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = 0
        
        # OVERVIEW METRICS
        st.markdown("### 📊 Overview Metrics")
        metrics = get_overview_metrics(status_threshold)
        st.caption("Showing data for: All Shops")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                "Total Items",
                f"{metrics['total_items']:,}",
                help="Unique CENTURY items across all shops"
            )
        
        with col2:
            st.metric(
                "Total Shops",
                f"{metrics['total_shops']:,}",
                help="Number of shops stocking CENTURY"
            )
        
        with col3:
            st.metric(
                "⚠️ UnderStock",
                f"{metrics['understock_items']:,}",
                delta=f"-{(metrics['understock_items']/metrics['total_items']*100):.1f}%",
                delta_color="inverse",
                help=f"Items understocked in ≥{status_threshold}% of shops"
            )
            if st.button("📋 View Details", key="btn_understock", use_container_width=True):
                log_user_event('understock_details_clicked')
                st.session_state.active_tab = 1
                st.rerun()
        
        with col4:
            st.metric(
                "📦 OverStock",
                f"{metrics['overstock_items']:,}",
                delta=f"+{(metrics['overstock_items']/metrics['total_items']*100):.1f}%",
                delta_color="off",
                help=f"Items overstocked in ≥{status_threshold}% of shops"
            )
            if st.button("📋 View Details", key="btn_overstock", use_container_width=True):
                log_user_event('overstock_details_clicked')
                st.session_state.active_tab = 2
                st.rerun()
        
        with col5:
            st.metric(
                "✅ Balanced",
                f"{metrics['balanced_items']:,}",
                delta=f"{(metrics['balanced_items']/metrics['total_items']*100):.1f}%",
                delta_color="normal",
                help=f"Items balanced in ≥{status_threshold}% of shops"
            )
        
        with col6:
            mixed_count = metrics.get('mixed_items', 0)
            st.metric(
                "🔀 Mixed",
                f"{mixed_count:,}",
                delta=f"{(mixed_count/metrics['total_items']*100):.1f}%",
                delta_color="off",
                help=f"Items with no dominant status (<{status_threshold}% in any category)"
            )
        
        st.markdown("---")
        
        col7, col8, col9, col10 = st.columns(4)
        
        with col7:
            st.metric("Stock In Hand", f"{metrics['total_sih']:,}")
        
        with col8:
            st.metric("Stock In Transit", f"{metrics['total_sit']:,}")
        
        with col9:
            st.metric("30-Day Requirement", f"{metrics['total_requirement']:,}")
        
        with col10:
            st.metric("Avg Rate of Sales", f"{metrics['avg_ros']:.2f}/day")
        
        st.markdown("---")
        
        # TAB SELECTION (using radio for programmatic control)
        
        def on_tab_change():
            """Callback for tab selection changes"""
            st.session_state.active_tab = tab_options.index(st.session_state.tab_selector)
            log_user_event('tab_changed', {'selected_tab': st.session_state.tab_selector})
        
        selected_tab = st.radio(
            "Select View:",
            tab_options,
            index=st.session_state.active_tab,
            horizontal=True,
            label_visibility="collapsed",
            key="tab_selector",
            on_change=on_tab_change
        )
        
        st.markdown("---")
        
        # RENDER SELECTED TAB CONTENT
        if selected_tab == "🏪 Shop Analysis":
            st.markdown("### 🏪 Shop-wise Stock Status")
            shop_filter_value = render_tab_shop_filter("🏪 Shop Analysis", "shop_analysis")
            shop_df = get_shop_wise_summary(shop_filter=shop_filter_value)
            
            if not shop_df.empty:
                # Shop selection for drill-down
                if not shop_filter_value:
                    selected_shop = st.selectbox(
                        "Select Shop for Details",
                        options=["All Shops"] + sorted(shop_df['shop_code'].unique().tolist())
                    )

                    if selected_shop != "All Shops":
                        shop_df = shop_df[shop_df['shop_code'] == selected_shop]
                
                # Stacked bar chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='UnderStock',
                    x=shop_df['shop_code'],
                    y=shop_df['understock'],
                    marker_color=Config.COLORS['UnderStock']
                ))
                
                fig.add_trace(go.Bar(
                    name='OverStock',
                    x=shop_df['shop_code'],
                    y=shop_df['overstock'],
                    marker_color=Config.COLORS['OverStock']
                ))
                
                fig.add_trace(go.Bar(
                    name='Balanced',
                    x=shop_df['shop_code'],
                    y=shop_df['balanced'],
                    marker_color=Config.COLORS['Balanced']
                ))
                
                fig.update_layout(
                    barmode='stack',
                    title="Stock Status by Shop",
                    xaxis_title="Shop Code",
                    yaxis_title="Number of Items",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
             # Data table
                render_dataframe_with_tooltips(shop_df, height=400, column_config=get_column_config())
        
        elif selected_tab == "⚠️ UnderStock Items":
            title_col, filter_col = st.columns([3, 2])
            with title_col:
                st.markdown("### ⚠️ Top UnderStock Items")
            with filter_col:
                understock_shops = get_available_shops(view_name=None)
                understock_shop_options = ["All Shops"] + sorted([s for s in understock_shops if s])
                selected_understock_shop = st.selectbox(
                    "Shop Code",
                    options=understock_shop_options,
                    key="shop_code_filter_understock_inline",
                    label_visibility="collapsed"
                )
            shop_filter_value = None if selected_understock_shop == "All Shops" else selected_understock_shop
            understock_limit = st.selectbox(
                "Rows to load",
                options=[500, 1000, 2000, 5000, 10000],
                index=2,
                key="understock_rows_limit",
                help="Lower row count loads faster"
            )
            understock_df = get_understock_items(limit=understock_limit, shop_filter=shop_filter_value)
            
            if not understock_df.empty:
                # Enhanced column config with warehouse columns
                wh_column_config = get_column_config()
                wh_column_config.update({
                    'total_wh_stock': st.column_config.NumberColumn(
                        '🏭 Total WH Stock',
                        help='Total warehouse stock available',
                        format='%d',
                        width='small'
                    ),
                    'wh_tema': st.column_config.NumberColumn(
                        '🏭 WH Tema',
                        help='Warehouse stock in Tema (TS)',
                        format='%d',
                        width='small'
                    ),
                    'wh_kumasi': st.column_config.NumberColumn(
                        '🏭 WH Kumasi',
                        help='Warehouse stock in Kumasi (KA)',
                        format='%d',
                        width='small'
                    )
                })
                
                render_dataframe_with_tooltips(understock_df, height=500, column_config=wh_column_config)
                
                # Fix pandas CSV conversion: use robust method
                try:
                    csv_buffer = io.StringIO()
                    for idx, row in understock_df.iterrows():
                        if idx == 0:
                            csv_buffer.write(','.join(str(col) for col in understock_df.columns) + '\n')
                        csv_buffer.write(','.join(f'"{str(val).replace(chr(34), chr(34)*2)}"' for val in row) + '\n')
                    csv = csv_buffer.getvalue()
                except Exception:
                    csv_df = understock_df.astype(str)
                    csv_df.columns = csv_df.columns.astype(str)
                    csv = csv_df.to_csv(index=False)
                if st.download_button(
                    label="📥 Download UnderStock Items",
                    data=csv,
                    file_name=f"understock_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                ):
                    log_user_event('understock_downloaded', {'rows': len(understock_df)})
            else:
                st.success("No understock items found!")
        
        elif selected_tab == "📦 OverStock Items":
            st.markdown("### 📦 Top OverStock Items")
            shop_filter_value = render_tab_shop_filter("📦 OverStock Items", "overstock")
            overstock_df = get_overstock_items(shop_filter=shop_filter_value)
            
            if not overstock_df.empty:
                # Enhanced column config with warehouse columns
                wh_column_config = get_column_config()
                wh_column_config.update({
                    'total_wh_stock': st.column_config.NumberColumn(
                        '🏭 Total WH Stock',
                        help='Total warehouse stock available',
                        format='%d',
                        width='small'
                    ),
                    'wh_tema': st.column_config.NumberColumn(
                        '🏭 WH Tema',
                        help='Warehouse stock in Tema (TS)',
                        format='%d',
                        width='small'
                    ),
                    'wh_kumasi': st.column_config.NumberColumn(
                        '🏭 WH Kumasi',
                        help='Warehouse stock in Kumasi (KA)',
                        format='%d',
                        width='small'
                    )
                })
                
                render_dataframe_with_tooltips(overstock_df, height=500, column_config=wh_column_config)
                
                # Fix pandas CSV conversion: use robust method
                try:
                    csv_buffer = io.StringIO()
                    for idx, row in overstock_df.iterrows():
                        if idx == 0:
                            csv_buffer.write(','.join(str(col) for col in overstock_df.columns) + '\n')
                        csv_buffer.write(','.join(f'"{str(val).replace(chr(34), chr(34)*2)}"' for val in row) + '\n')
                    csv = csv_buffer.getvalue()
                except Exception:
                    csv_df = overstock_df.astype(str)
                    csv_df.columns = csv_df.columns.astype(str)
                    csv = csv_df.to_csv(index=False)
                if st.download_button(
                    label="📥 Download OverStock Items",
                    data=csv,
                    file_name=f"overstock_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                ):
                    log_user_event('overstock_downloaded', {'rows': len(overstock_df)})
            else:
                st.success("No overstock items found!")
        
        elif selected_tab == "🚨 Critical Items":
            st.markdown("### 🚨 Critical Items (No Stock + High Demand)")
            shop_filter_value = render_tab_shop_filter("🚨 Critical Items", "critical")
            critical_df = get_critical_items(shop_filter=shop_filter_value)
            
            if not critical_df.empty:
                st.warning(f"⚠️ **{len(critical_df)} critical items** require immediate attention!")
                
                # Enhanced column config with warehouse columns
                wh_column_config = get_column_config()
                wh_column_config.update({
                    'total_wh_stock': st.column_config.NumberColumn(
                        '🏭 Total WH Stock',
                        help='Total warehouse stock available',
                        format='%d',
                        width='small'
                    ),
                    'wh_tema': st.column_config.NumberColumn(
                        '🏭 WH Tema',
                        help='Warehouse stock in Tema (TS)',
                        format='%d',
                        width='small'
                    ),
                    'wh_kumasi': st.column_config.NumberColumn(
                        '🏭 WH Kumasi',
                        help='Warehouse stock in Kumasi (KA)',
                        format='%d',
                        width='small'
                    )
                })
                
                render_dataframe_with_tooltips(critical_df, height=500, column_config=wh_column_config)
                
                # Fix pandas CSV conversion: use robust method
                try:
                    csv_buffer = io.StringIO()
                    for idx, row in critical_df.iterrows():
                        if idx == 0:
                            csv_buffer.write(','.join(str(col) for col in critical_df.columns) + '\n')
                        csv_buffer.write(','.join(f'"{str(val).replace(chr(34), chr(34)*2)}"' for val in row) + '\n')
                    csv = csv_buffer.getvalue()
                except Exception:
                    csv_df = critical_df.astype(str)
                    csv_df.columns = csv_df.columns.astype(str)
                    csv = csv_df.to_csv(index=False)
                if st.download_button(
                    label="📥 Download Critical Items",
                    data=csv,
                    file_name=f"critical_items_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                ):
                    log_user_event('critical_items_downloaded', {'rows': len(critical_df)})
            else:
                st.success("No critical stock-outs with high demand!")
        
        elif selected_tab == "🐌 Slow Moving":
            st.markdown("### 🐌 Slow Moving Items")
            shop_filter_value = render_tab_shop_filter("🐌 Slow Moving", "slow_moving")
            slow_df = get_slow_moving_items(shop_filter=shop_filter_value)
            
            if not slow_df.empty:
                # Enhanced column config with warehouse columns
                wh_column_config = get_column_config()
                wh_column_config.update({
                    'total_wh_stock': st.column_config.NumberColumn(
                        '🏭 Total WH Stock',
                        help='Total warehouse stock available',
                        format='%d',
                        width='small'
                    ),
                    'wh_tema': st.column_config.NumberColumn(
                        '🏭 WH Tema',
                        help='Warehouse stock in Tema (TS)',
                        format='%d',
                        width='small'
                    ),
                    'wh_kumasi': st.column_config.NumberColumn(
                        '🏭 WH Kumasi',
                        help='Warehouse stock in Kumasi (KA)',
                        format='%d',
                        width='small'
                    )
                })
                
                render_dataframe_with_tooltips(slow_df, height=500, column_config=wh_column_config)
                
                # Fix pandas CSV conversion: use robust method
                try:
                    csv_buffer = io.StringIO()
                    for idx, row in slow_df.iterrows():
                        if idx == 0:
                            csv_buffer.write(','.join(str(col) for col in slow_df.columns) + '\n')
                        csv_buffer.write(','.join(f'"{str(val).replace(chr(34), chr(34)*2)}"' for val in row) + '\n')
                    csv = csv_buffer.getvalue()
                except Exception:
                    csv_df = slow_df.astype(str)
                    csv_df.columns = csv_df.columns.astype(str)
                    csv = csv_df.to_csv(index=False)
                if st.download_button(
                    label="📥 Download Slow Moving Items",
                    data=csv,
                    file_name=f"slow_moving_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                ):
                    log_user_event('slow_moving_downloaded', {'rows': len(slow_df)})
            else:
                st.info("No slow moving items found.")
        
        elif selected_tab == "📂 Department Analysis":
            st.markdown("### 📂 Department Analysis")
            shop_filter_value = render_tab_shop_filter("📂 Department Analysis", "department")
            dept_df = get_department_analysis(shop_filter=shop_filter_value)
            
            if not dept_df.empty:
                # Bar chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='UnderStock',
                    x=dept_df['dept'],
                    y=dept_df['understock'],
                    marker_color=Config.COLORS['UnderStock']
                ))
                
                fig.add_trace(go.Bar(
                    name='OverStock',
                    x=dept_df['dept'],
                    y=dept_df['overstock'],
                    marker_color=Config.COLORS['OverStock']
                ))
                
                fig.update_layout(
                    barmode='group',
                    title="Stock Issues by Department",
                    xaxis_title="Department",
                    yaxis_title="Number of Items",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                render_dataframe_with_tooltips(dept_df, height=400, column_config=get_column_config())

        elif selected_tab == "🎯 Target vs Achieve":
            st.markdown("### 🎯 Target vs Achieve")

            try:
                target_df = get_target_vs_achieve()
            except Exception as exc:
                st.error(f"Unable to load target vs achieve view: {exc}")
                target_df = pd.DataFrame()

            if not target_df.empty:
                target_display_df = prepare_target_vs_achieve_display(target_df)
                styled_target_df = style_target_vs_achieve_table(target_display_df)
                st.dataframe(
                    styled_target_df,
                    use_container_width=True,
                    height=520,
                    column_config={
                        "ITEM CODE": st.column_config.TextColumn("ITEM CODE"),
                        "ITEM NAME": st.column_config.TextColumn("ITEM NAME"),
                        "REPORT MONTH": st.column_config.TextColumn("REPORT MONTH"),
                        "RETAIL PRICE AVG": st.column_config.NumberColumn("RETAIL PRICE AVG", format="%.2f"),
                        "TARGET/MONTH (PCS)": st.column_config.NumberColumn("TARGET/MONTH (PCS)", format="%.0f"),
                        "STOCK IN HAND": st.column_config.NumberColumn("STOCK IN HAND", format="%.0f"),
                        "ACTUAL QTY": st.column_config.NumberColumn("ACTUAL QTY", format="%.2f"),
                        "SHORTFALL/ACCESS QTY": st.column_config.NumberColumn("SHORTFALL/ACCESS QTY", format="%.2f"),
                        "TARGET VALUE(GHC)": st.column_config.NumberColumn("TARGET VALUE(GHC)", format="%.2f"),
                        "ACTUAL VALUE NET": st.column_config.NumberColumn("ACTUAL VALUE NET", format="%.2f"),
                        "SHORTFALL/ACCESS VALUE": st.column_config.NumberColumn("SHORTFALL/ACCESS VALUE", format="%.2f"),
                    },
                )

                if st.download_button(
                    label="📥 Download Target vs Achieve",
                    data=target_display_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"target_vs_achieve_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                ):
                    log_user_event('target_vs_achieve_downloaded', {'rows': len(target_display_df)})
            else:
                st.info("No rows found in mv_target_vs_achieve_century.")

        elif selected_tab == "📋 All Items":
            title_col, filter_col = st.columns([3, 2])
            with title_col:
                st.markdown("### 📋 All Items")
            with filter_col:
                shop_filter_value = render_tab_shop_filter("📋 All Items", "all_items")

            all_items_limit = st.selectbox(
                "Rows to load",
                options=[500, 1000, 2000, 5000, 10000],
                index=2,
                key="all_items_rows_limit",
                help="Lower row count loads faster"
            )
            all_items_df = get_all_items(limit=all_items_limit, shop_filter=shop_filter_value)

            if not all_items_df.empty:
                render_dataframe_with_tooltips(
                    all_items_df,
                    height=500,
                    column_config=get_column_config()
                )

                try:
                    all_items_download_df = get_all_items_download_full()
                    csv_buffer = io.StringIO()
                    for idx, row in all_items_download_df.iterrows():
                        if idx == 0:
                            csv_buffer.write(','.join(str(col) for col in all_items_download_df.columns) + '\n')
                        csv_buffer.write(','.join(f'"{str(val).replace(chr(34), chr(34)*2)}"' for val in row) + '\n')
                    csv = csv_buffer.getvalue()
                except Exception:
                    csv_df = get_all_items_download_full().astype(str)
                    csv_df.columns = csv_df.columns.astype(str)
                    csv = csv_df.to_csv(index=False)

                if st.download_button(
                    label="📥 Download All Items (Full)",
                    data=csv,
                    file_name=f"all_items_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                ):
                    log_user_event('all_items_downloaded', {
                        'rows': len(all_items_download_df),
                        'scope': 'full_dataset',
                        'ignores_row_limit': True,
                        'ignores_shop_filter': True,
                    })
            else:
                st.info("No items found for the selected shop.")


if __name__ == "__main__":
    main()
