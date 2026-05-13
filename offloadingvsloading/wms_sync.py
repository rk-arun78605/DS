"""
WMS Shop Diff daily sync — Loading vs Offloading dashboard.

Follows the exact same upload pipeline as home_dashboard.py
(TABLE_CONFIGS["LVO_offloading_vs_loading"]):
  • Column normalisation via _prepare_lvo_offloading_df
  • Table / history table creation via _ensure_loadingvsoffloading_tables
  • Delete-for-date + INSERT via psycopg2 execute_values (idempotent)
  • History row written with history_action = 'auto_sync'
  • Email to mis.manager@melcomgroup.com with Excel attachment

Usage:
    from wms_sync import run_sync
    result = run_sync()                     # syncs yesterday
    result = run_sync(target_date=d, force=True)  # force re-import
"""

import base64
import csv as _csv_mod
import glob
import io
import logging
import os
import re
import smtplib
import tempfile
import time
import warnings
from datetime import date, datetime, timedelta
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor

try:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    OPENPYXL_OK = True
except Exception:
    OPENPYXL_OK = False

# ── Configuration ──────────────────────────────────────────────────────────────
NETWORK_PATH = r"\\10.10.0.30\mis"

DB_CONFIG = {
    "host": "localhost",
    "port": 3307,
    "user": "postgres",
    "password": "hello",
    "dbname": "WH",
}

# Exact column list from home_dashboard TABLE_CONFIGS["LVO_offloading_vs_loading"]
LVO_COLUMNS = [
    "date", "shop_code", "vehicle_no", "item_code", "item_name",
    "qty_loaded", "value_loaded", "qty_offloaded", "value_offloaded",
    "diff_qty", "diff_val", "price", "diff",
]

EMAIL_TO   = "mis.manager@melcomgroup.com"
EMAIL_FROM = "mis.manager@melcomgroup.com"
SMTP_HOST  = "mail.melcomgroup.com"   # update if different
SMTP_PORT  = 25
SMTP_USER  = ""
SMTP_PASS  = ""
DASHBOARD_URL  = "http://10.10.1.79:8522"
DASHBOARD_PORT = 8522

# Credentials for headless screenshot (must have table_access='all' or 'offloading_vs_loading')
SCREENSHOT_USER = "MGMT001"   # update to a valid dashboard user
SCREENSHOT_PASS = "Hello@123" # update to their password

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN NORMALISATION  (identical to home_dashboard._prepare_lvo_offloading_df)
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_header_key(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(name).strip().lower())


def _rename_by_expected_columns(df: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    rename_map = {}
    normalized_actual = {_normalize_header_key(c): c for c in df.columns}
    for expected in expected_columns:
        key = _normalize_header_key(expected)
        if key in normalized_actual:
            actual = normalized_actual[key]
            if actual != expected:
                rename_map[actual] = expected
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _extract_date_from_filename(source_file_name: str) -> Optional[date]:
    if not source_file_name:
        return None
    name = source_file_name.lower()
    patterns = [
        r"(\d{4})[-_](\d{1,2})[-_](\d{1,2})",
        r"(\d{1,2})[-_](\d{1,2})[-_](\d{4})",
        r"(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*(\d{2,4})",
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if not m:
            continue
        try:
            if pat.startswith("(\\d{4})"):
                y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return datetime(y, mm, dd).date()
            if "jan|feb|mar" in pat:
                d = int(m.group(1))
                mon_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                           "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
                y = int(m.group(3))
                if y < 100:
                    y += 2000
                return datetime(y, mon_map[m.group(2)], d).date()
            d, mm, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return datetime(y, mm, d).date()
        except Exception:
            continue
    return None


def _prepare_lvo_offloading_df(df: pd.DataFrame, source_file_name: str = "") -> pd.DataFrame:
    """
    Identical to home_dashboard._prepare_lvo_offloading_df.
    Normalises any WMS_shopdiff CSV column layout to the canonical LVO schema.
    """
    out = df.copy()
    original_cols = list(out.columns)

    def _find_original_col(*keys):
        normalized_to_actual = {_normalize_header_key(col): col for col in original_cols}
        for key in keys:
            actual = normalized_to_actual.get(key)
            if actual is not None:
                return actual
        return None

    alias_map = {
        'date': 'date', 'dateinvoice': 'date', 'date_invoice': 'date',
        'dtdate': 'date', 'offloa': 'date', 'offloadi': 'date',
        'offloadingdate': 'date', 'offload_date': 'date',
        'offloaddate': 'date', 'offload': 'date',
        'shopcode': 'shop_code', 'shop_code': 'shop_code', 'shop': 'shop_code',
        'vehicleno': 'vehicle_no', 'vehicle_no': 'vehicle_no', 'vehiclenumber': 'vehicle_no',
        'itemcode': 'item_code', 'item_code': 'item_code',
        'itemname': 'item_name', 'item_name': 'item_name',
        'qtyloaded': 'qty_loaded', 'qty_loaded': 'qty_loaded',
        'whloadedqty': 'qty_loaded', 'loadedqty': 'qty_loaded',
        'offloadedqty': 'qty_loaded', 'offloadqty': 'qty_loaded',
        'offloadingqty': 'qty_loaded',
        'valueloaded': 'value_loaded', 'value_loaded': 'value_loaded',
        'whloadedvalue': 'value_loaded', 'loadedvalue': 'value_loaded',
        'offloadedvalue': 'value_loaded', 'offloadvalue': 'value_loaded',
        'offloadingvalue': 'value_loaded',
        'qtyoffloaded': 'qty_offloaded', 'qty_offloaded': 'qty_offloaded',
        'shopreceivingqty': 'qty_offloaded', 'receive': 'qty_offloaded',
        'received': 'qty_offloaded', 'receivedqty': 'qty_offloaded',
        'receivingqty': 'qty_offloaded',
        'valueoffloaded': 'value_offloaded', 'value_offloaded': 'value_offloaded',
        'shopreceivingvalue': 'value_offloaded',
        'cartq': 'qty_loaded', 'cart_qty': 'qty_loaded', 'cartonqty': 'qty_loaded',
        'diffqty': 'diff_qty', 'diff_qty': 'diff_qty',
        'diff': 'diff', 'diffval': 'diff_val', 'diff_val': 'diff_val',
        'shopdif': 'diff_val', 'shopdiff': 'diff_val',
        'shopdifference': 'diff_val', 'shopvaluediff': 'diff_val',
        'valuediff': 'diff_val', 'price': 'price',
    }

    rename_map = {}
    for c in original_cols:
        key = _normalize_header_key(c)
        target = alias_map.get(key)
        if target and c != target and target not in out.columns:
            rename_map[c] = target
    if rename_map:
        out = out.rename(columns=rename_map)

    # Explicit business rules: RECEIVED_QTY = warehouse loaded, OFFLOADED_QTY = shop receiving
    offloaded_qty_col = _find_original_col('offloadedqty')
    received_qty_col  = _find_original_col('receivedqty')
    explicit_price_col = _find_original_col('price')
    explicit_diff_col  = _find_original_col('diff')
    if received_qty_col is not None:
        out['qty_loaded'] = pd.to_numeric(df[received_qty_col], errors='coerce').fillna(0)
    if offloaded_qty_col is not None:
        out['qty_offloaded'] = pd.to_numeric(df[offloaded_qty_col], errors='coerce').fillna(0)
    if explicit_price_col is not None:
        out['price'] = pd.to_numeric(df[explicit_price_col], errors='coerce').fillna(0)
    elif len(original_cols) >= 10:
        out['price'] = pd.to_numeric(df[original_cols[9]], errors='coerce').fillna(0)
    if explicit_diff_col is not None:
        out['diff'] = pd.to_numeric(df[explicit_diff_col], errors='coerce').fillna(0)
    elif len(original_cols) >= 11:
        out['diff'] = pd.to_numeric(df[original_cols[10]], errors='coerce').fillna(0)

    if 'diff' in out.columns and 'diff_qty' not in out.columns:
        out['diff_qty'] = pd.to_numeric(out['diff'], errors='coerce')

    for col in ('diff_qty', 'diff_val'):
        if col not in out.columns:
            out[col] = 0

    diff_qty_num = pd.to_numeric(out['diff_qty'], errors='coerce').fillna(0)
    diff_val_num = pd.to_numeric(out['diff_val'], errors='coerce').fillna(0)

    if 'qty_loaded' not in out.columns and 'qty_offloaded' in out.columns:
        out['qty_loaded'] = pd.to_numeric(out['qty_offloaded'], errors='coerce').fillna(0) - diff_qty_num
    if 'qty_offloaded' not in out.columns and 'qty_loaded' in out.columns:
        out['qty_offloaded'] = pd.to_numeric(out['qty_loaded'], errors='coerce').fillna(0) + diff_qty_num
    if 'value_loaded' not in out.columns and 'value_offloaded' in out.columns:
        out['value_loaded'] = pd.to_numeric(out['value_offloaded'], errors='coerce').fillna(0) - diff_val_num
    if 'value_offloaded' not in out.columns and 'value_loaded' in out.columns:
        out['value_offloaded'] = pd.to_numeric(out['value_loaded'], errors='coerce').fillna(0) + diff_val_num

    for req in ('qty_loaded', 'value_loaded', 'qty_offloaded', 'value_offloaded'):
        if req not in out.columns:
            out[req] = 0

    qty_loaded_num    = pd.to_numeric(out.get('qty_loaded',    0), errors='coerce').fillna(0)
    qty_offloaded_num = pd.to_numeric(out.get('qty_offloaded', 0), errors='coerce').fillna(0)

    if 'qty_loaded' in out.columns and 'qty_offloaded' in out.columns:
        out['diff_qty'] = qty_offloaded_num - qty_loaded_num
        diff_qty_num = pd.to_numeric(out['diff_qty'], errors='coerce').fillna(0)

    unit_price = pd.to_numeric(out.get('price', 0), errors='coerce').fillna(0)
    out['price'] = unit_price

    out['value_loaded']    = qty_loaded_num    * unit_price
    out['value_offloaded'] = qty_offloaded_num * unit_price
    out['diff_val'] = pd.to_numeric(out['value_offloaded'], errors='coerce').fillna(0) - \
                      pd.to_numeric(out['value_loaded'],    errors='coerce').fillna(0)

    # Sync 'diff' column
    if 'diff' in out.columns and 'diff_qty' in out.columns:
        out['diff'] = pd.to_numeric(out['diff'], errors='coerce').fillna(
            pd.to_numeric(out['diff_qty'], errors='coerce').fillna(0)
        )

    # Date column
    if 'date' not in out.columns:
        best_parsed, best_count = None, 0
        for col in out.columns:
            key = _normalize_header_key(col)
            if not any(tok in key for tok in ['date', 'offload', 'offloading']):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(out[col], errors='coerce', dayfirst=True, format='mixed')
            except Exception:
                try:
                    parsed = pd.to_datetime(out[col], errors='coerce', dayfirst=True)
                except Exception:
                    continue
            cnt = int(parsed.notna().sum())
            if cnt > best_count:
                best_count, best_parsed = cnt, parsed
        if best_parsed is not None and best_count > 0:
            out['date'] = best_parsed.dt.date

    if 'date' not in out.columns:
        fallback = _extract_date_from_filename(source_file_name)
        if fallback is None:
            raise ValueError(
                "Date column not found. Add a date column or include a date in the filename."
            )
        out['date'] = fallback

    return out


# ══════════════════════════════════════════════════════════════════════════════
# TABLE SETUP  (identical to home_dashboard._ensure_loadingvsoffloading_tables)
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_loadingvsoffloading_tables(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS public.offloading_vs_loading (
            date DATE, shop_code TEXT, vehicle_no TEXT, item_code TEXT, item_name TEXT,
            qty_loaded NUMERIC, value_loaded NUMERIC,
            qty_offloaded NUMERIC, value_offloaded NUMERIC,
            diff_qty NUMERIC, diff_val NUMERIC,
            price NUMERIC, diff NUMERIC
        )
    """)
    cur.execute("ALTER TABLE public.offloading_vs_loading ADD COLUMN IF NOT EXISTS price NUMERIC")
    cur.execute("ALTER TABLE public.offloading_vs_loading ADD COLUMN IF NOT EXISTS diff NUMERIC")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS public.offloading_vs_loading_history (
            id BIGSERIAL PRIMARY KEY,
            date DATE, shop_code TEXT, vehicle_no TEXT, item_code TEXT, item_name TEXT,
            qty_loaded NUMERIC, value_loaded NUMERIC,
            qty_offloaded NUMERIC, value_offloaded NUMERIC,
            diff_qty NUMERIC, diff_val NUMERIC,
            price NUMERIC, diff NUMERIC,
            history_action TEXT NOT NULL,
            history_at TIMESTAMP NOT NULL DEFAULT NOW(),
            source_file TEXT
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_ovl_date_shop ON public.offloading_vs_loading (date, shop_code)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_ovl_item_shop ON public.offloading_vs_loading (item_code, shop_code)"
    )
    conn.commit()
    cur.close()


# ══════════════════════════════════════════════════════════════════════════════
# CORE LOAD  (follows home_dashboard._upload_append_with_history + date-DELETE)
# ══════════════════════════════════════════════════════════════════════════════

def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _load_df_to_db(conn, df_upload: pd.DataFrame, target_date: date, source_file: str) -> int:
    """
    1. Delete existing rows for target_date (idempotent / re-run safe)
    2. INSERT using execute_values (same approach as home_dashboard)
    3. UPDATE diff = diff_qty
    4. Write history rows with history_action='auto_sync'
    """
    columns = LVO_COLUMNS
    col_sql = ', '.join([_quote_ident(c) for c in columns])

    def _null(v):
        return None if pd.isnull(v) else v

    values = [
        tuple(_null(row.get(c)) for c in columns)
        for _, row in df_upload.iterrows()
    ]

    cur = conn.cursor()
    log.info("Deleting existing rows for %s", target_date)
    cur.execute("DELETE FROM public.offloading_vs_loading WHERE date = %s", (target_date,))
    deleted = cur.rowcount

    log.info("Inserting %d rows for %s", len(values), target_date)
    if values:
        execute_values(
            cur,
            f"INSERT INTO public.offloading_vs_loading ({col_sql}) VALUES %s",
            values,
            page_size=5000,
        )
        # Keep diff in sync (identical to home_dashboard)
        cur.execute("""
            UPDATE public.offloading_vs_loading
            SET diff = diff_qty
            WHERE diff IS DISTINCT FROM diff_qty
        """)

        # History — same pattern as _upload_append_with_history
        log.info("Writing %d history rows", len(values))
        now_ts = datetime.now()
        history_values = [
            tuple(_null(row.get(c)) for c in columns) + ('auto_sync', now_ts, source_file)
            for _, row in df_upload.iterrows()
        ]
        history_col_sql = ', '.join(
            [_quote_ident(c) for c in columns + ['history_action', 'history_at', 'source_file']]
        )
        execute_values(
            cur,
            f"INSERT INTO public.offloading_vs_loading_history ({history_col_sql}) VALUES %s",
            history_values,
            page_size=5000,
        )
        cur.execute("""
            UPDATE public.offloading_vs_loading_history
            SET diff = diff_qty
            WHERE history_action = 'auto_sync'
              AND history_at >= NOW() - INTERVAL '10 minutes'
        """)

    conn.commit()
    cur.close()
    log.info("Deleted %d old rows, inserted %d new rows", deleted, len(values))
    return len(values)


# ══════════════════════════════════════════════════════════════════════════════
# FILE DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def find_wms_file(target_date: date) -> str | None:
    if not os.path.exists(NETWORK_PATH):
        raise OSError(f"Network path not accessible: {NETWORK_PATH}")
    for date_str in (
        target_date.strftime("%d_%m_%Y"),
        f"{target_date.day}_{target_date.month:02d}_{target_date.year}",
    ):
        for pat in (
            os.path.join(NETWORK_PATH, f"*WMS_shopdiff*{date_str}*.csv"),
            os.path.join(NETWORK_PATH, f"*wms_shopdiff*{date_str}*.csv"),
        ):
            matches = glob.glob(pat)
            if matches:
                return sorted(matches)[-1]
    return None


def date_exists_in_db(target_date: date) -> bool:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM offloading_vs_loading WHERE date = %s LIMIT 1",
                (target_date,),
            )
            return cur.fetchone() is not None
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# EXCEL ATTACHMENT
# ══════════════════════════════════════════════════════════════════════════════

def _hex_fill(hex_color: str) -> "PatternFill":
    return PatternFill(start_color=hex_color, end_color=hex_color, fill_type="solid")


def build_excel_bytes(target_date: date) -> bytes:
    """
    Three-sheet workbook:
      Sheet 1 — Excess Received  (diff_qty > 0)
      Sheet 2 — Short Received   (diff_qty < 0)
      Sheet 3 — Balanced         (diff_qty = 0)
    Each sheet contains item-level detail with shop manager name,
    sorted by shop then item, with a TOTALS row at the bottom.
    """
    if not OPENPYXL_OK:
        return b""

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                WITH mgr AS (
                    SELECT
                        UPPER(TRIM(shop_code))  AS shop_code,
                        MAX(TRIM(shop_description)) AS shop_name,
                        STRING_AGG(DISTINCT TRIM(shop_manager_name), ', '
                                   ORDER BY TRIM(shop_manager_name)) AS manager_name
                    FROM shopmgrname
                    WHERE is_current = TRUE
                      AND TRIM(COALESCE(shop_code,'')) <> ''
                    GROUP BY UPPER(TRIM(shop_code))
                )
                SELECT
                    COALESCE(mgr.manager_name, o.shop_code) AS shop_manager,
                    COALESCE(mgr.shop_name,    o.shop_code) AS shop_name,
                    o.shop_code,
                    o.vehicle_no,
                    o.item_code,
                    o.item_name,
                    o.qty_loaded,
                    o.value_loaded,
                    o.qty_offloaded,
                    o.value_offloaded,
                    o.diff_qty,
                    o.diff_val,
                    o.price
                FROM offloading_vs_loading o
                LEFT JOIN mgr ON mgr.shop_code = UPPER(TRIM(o.shop_code))
                WHERE o.date = %s
                ORDER BY o.shop_code, o.item_code
            """, (target_date,))
            all_rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    excess   = [r for r in all_rows if float(r.get("diff_qty") or 0) > 0]
    short    = [r for r in all_rows if float(r.get("diff_qty") or 0) < 0]
    balanced = [r for r in all_rows if float(r.get("diff_qty") or 0) == 0]

    # ── Style constants ───────────────────────────────────────────────────────
    date_label  = target_date.strftime("%d %b %Y")
    num_fmt     = "#,##0.00"
    num_fmt0    = "#,##0"
    center      = Alignment(horizontal="center", vertical="center")
    right       = Alignment(horizontal="right",  vertical="center")
    left        = Alignment(horizontal="left",   vertical="center")
    hdr_font    = Font(bold=True, color="FFFFFF", size=10)
    tot_font    = Font(bold=True, size=10)

    SHEET_DEFS = [
        ("Excess Received", excess,   "00B050", "E2EFDA"),  # green tones
        ("Short Received",  short,    "C00000", "FCE4D6"),  # red tones
        ("Balanced",        balanced, "2E75B6", "DEEAF1"),  # blue tones
    ]

    HDRS = [
        "Shop Manager", "Shop Name", "Shop Code", "Vehicle No",
        "Item Code", "Item Name",
        "Loaded Qty", "Loaded Value (GH₵)",
        "Offloaded Qty", "Offloaded Value (GH₵)",
        "Diff Qty", "Diff Value (GH₵)", "Unit Price (GH₵)",
    ]
    COL_WIDTHS = [22, 28, 10, 12, 14, 40, 12, 18, 13, 20, 10, 16, 16]
    NUM_COLS   = {7, 8, 9, 10, 11, 12, 13}   # 1-based indices of numeric columns
    DIFF_COLS  = {11, 12}                     # colour-coded

    wb = Workbook()
    wb.remove(wb.active)   # remove default empty sheet

    for sheet_name, rows, accent_hex, light_hex in SHEET_DEFS:
        ws = wb.create_sheet(title=sheet_name)

        # ── Title row ─────────────────────────────────────────────────────────
        ws.merge_cells(start_row=1, start_column=1,
                       end_row=1, end_column=len(HDRS))
        title_cell = ws.cell(row=1, column=1,
                             value=f"{sheet_name}   ·   {date_label}   "
                                   f"({len(rows):,} items)")
        title_cell.font      = Font(bold=True, color="FFFFFF", size=12)
        title_cell.fill      = _hex_fill(f"0D2B8E")
        title_cell.alignment = center
        ws.row_dimensions[1].height = 22

        # ── Header row ────────────────────────────────────────────────────────
        hdr_fill = _hex_fill("0A1F5C")
        for ci, (h, w) in enumerate(zip(HDRS, COL_WIDTHS), 1):
            cell = ws.cell(row=2, column=ci, value=h)
            cell.font      = hdr_font
            cell.fill      = hdr_fill
            cell.alignment = right if ci in NUM_COLS else center
            ws.column_dimensions[cell.column_letter].width = w
        ws.row_dimensions[2].height = 16

        # ── Data rows ─────────────────────────────────────────────────────────
        row_fill_even = _hex_fill("EBF3FB") if light_hex == "DEEAF1" else _hex_fill(light_hex)
        row_fill_odd  = PatternFill()   # no fill

        for ri, row in enumerate(rows, 3):
            dq = float(row.get("diff_qty") or 0)
            dv = float(row.get("diff_val") or 0)
            vals = [
                row["shop_manager"],
                row["shop_name"],
                row["shop_code"],
                row["vehicle_no"],
                row["item_code"],
                row["item_name"],
                float(row["qty_loaded"]     or 0),
                float(row["value_loaded"]   or 0),
                float(row["qty_offloaded"]  or 0),
                float(row["value_offloaded"]or 0),
                dq,
                dv,
                float(row["price"]          or 0),
            ]
            fill = row_fill_even if ri % 2 == 0 else row_fill_odd
            for ci, v in enumerate(vals, 1):
                cell = ws.cell(row=ri, column=ci, value=v)
                cell.fill = fill
                if ci in NUM_COLS:
                    cell.number_format = num_fmt0 if ci in (7, 9, 11) else num_fmt
                    cell.alignment     = right
                    if ci in DIFF_COLS and v != 0:
                        cell.font = Font(
                            bold=True,
                            color="C00000" if v < 0 else "00B050",
                        )
                else:
                    cell.alignment = left
                    cell.font = Font(size=10)

        # ── Totals row ────────────────────────────────────────────────────────
        if rows:
            tr = len(rows) + 3
            tot_fill = _hex_fill("D9E1F2")
            ws.cell(row=tr, column=1, value="TOTAL").font = Font(bold=True, size=10)
            ws.cell(row=tr, column=1).fill = tot_fill
            for ci in range(2, len(HDRS) + 1):
                cell = ws.cell(row=tr, column=ci)
                cell.fill = tot_fill
                if ci in NUM_COLS:
                    key_map = {
                        7: "qty_loaded",    8: "value_loaded",
                        9: "qty_offloaded", 10: "value_offloaded",
                        11: "diff_qty",     12: "diff_val",
                        13: "price",
                    }
                    if ci in key_map:
                        total = sum(float(r.get(key_map[ci]) or 0) for r in rows)
                        cell.value         = total
                        cell.number_format = num_fmt0 if ci in (7, 9, 11) else num_fmt
                        cell.alignment     = right
                        cell.font          = tot_font
                        if ci in DIFF_COLS and total != 0:
                            cell.font = Font(
                                bold=True, size=10,
                                color="C00000" if total < 0 else "00B050",
                            )

        # ── Freeze panes below header ─────────────────────────────────────────
        ws.freeze_panes = "A3"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD SCREENSHOT  (Selenium headless Chrome)
# ══════════════════════════════════════════════════════════════════════════════

def _take_dashboard_screenshot(target_date: date) -> bytes | None:
    """
    Launch headless Chrome, log into the dashboard, wait for the shopwise
    table to render, take a screenshot, crop from the top of the page to
    the bottom of the first table, and return PNG bytes.
    Returns None on any failure (email will fall back to HTML body).
    """
    try:
        from PIL import Image
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError as e:
        log.warning("Screenshot dependencies missing: %s", e)
        return None

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1440,1600")
    opts.add_argument("--hide-scrollbars")
    opts.add_argument("--force-device-scale-factor=1")
    opts.add_argument("--disable-gpu")

    driver = None
    try:
        driver = webdriver.Chrome(options=opts)
        wait = WebDriverWait(driver, 20)

        # ── 1. Login ──────────────────────────────────────────────────────────
        driver.get(f"http://localhost:{DASHBOARD_PORT}")
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="text"]')))
        time.sleep(0.5)

        driver.find_element(By.CSS_SELECTOR, 'input[type="text"]').send_keys(SCREENSHOT_USER)
        driver.find_element(By.CSS_SELECTOR, 'input[type="password"]').send_keys(SCREENSHOT_PASS)
        # Streamlit form submit button
        driver.find_element(
            By.XPATH,
            '//button[@kind="primaryFormSubmit" or .//p[text()="Login"]]'
        ).click()

        # ── 2. Wait for KPI metrics to appear ─────────────────────────────────
        wait.until(EC.presence_of_element_located(
            (By.XPATH, '//*[@data-testid="stMetricValue"]')
        ))
        time.sleep(3)   # allow all Streamlit components (iframes) to finish rendering

        # ── 3. Locate the bottom of the shopwise table ────────────────────────
        # The table is rendered inside a components.v1.html iframe.
        # Strategy: find the last visible iframe and use its bottom edge.
        crop_bottom = 900   # safe default (covers header + KPIs + ~10 shop rows)
        try:
            iframes = driver.find_elements(By.TAG_NAME, "iframe")
            for iframe in iframes:
                rect = driver.execute_script(
                    "var r = arguments[0].getBoundingClientRect();"
                    "return {top: r.top + window.scrollY, bottom: r.bottom + window.scrollY, h: r.height};",
                    iframe,
                )
                # Only consider visible, content-bearing iframes (height > 50)
                if rect["h"] > 50:
                    candidate = int(rect["bottom"]) + 30
                    if candidate > crop_bottom:
                        crop_bottom = candidate
        except Exception:
            pass

        # Scroll to top before taking screenshot
        driver.execute_script("window.scrollTo(0, 0)")
        time.sleep(0.3)

        # ── 4. Screenshot + crop ──────────────────────────────────────────────
        png_bytes = driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(png_bytes))
        crop_h = min(crop_bottom, img.height)
        if crop_h > 200:
            img = img.crop((0, 0, img.width, crop_h))

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        log.info("Dashboard screenshot captured (%d × %d px)", img.width, crop_h)
        return buf.getvalue()

    except Exception as e:
        log.warning("Dashboard screenshot failed: %s", e)
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# EMAIL
# ══════════════════════════════════════════════════════════════════════════════

def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th','th','th','th','th','th'][n % 10]}"


def format_date_subject(d: date) -> str:
    return f"{_ordinal(d.day)} {d.strftime('%b')} {d.strftime('%y')}"


# ── Dashboard data queries for email body ─────────────────────────────────────

def _get_email_data(target_date: date) -> dict:
    """Fetch KPI totals + shopwise summary for the email HTML body."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # KPI totals
            cur.execute("""
                SELECT
                    SUM(qty_loaded)    AS total_loaded_qty,
                    SUM(qty_offloaded) AS total_offloaded_qty,
                    SUM(diff_qty)      AS total_diff_qty,
                    SUM(diff_val)      AS total_diff_val
                FROM offloading_vs_loading
                WHERE date = %s
            """, (target_date,))
            kpi = dict(cur.fetchone() or {})

            # Shop summary with manager names (same join as dashboard load_shop_meta_map)
            cur.execute("""
                WITH mgr AS (
                    SELECT
                        UPPER(TRIM(shop_code))  AS shop_code,
                        MAX(TRIM(shop_description)) AS shop_name,
                        STRING_AGG(DISTINCT TRIM(shop_manager_name), ', '
                                   ORDER BY TRIM(shop_manager_name)) AS manager_name
                    FROM shopmgrname
                    WHERE is_current = TRUE
                      AND TRIM(COALESCE(shop_code,'')) <> ''
                    GROUP BY UPPER(TRIM(shop_code))
                )
                SELECT
                    COALESCE(mgr.manager_name, s.shop_code)     AS shop_manager,
                    COALESCE(mgr.shop_name,    s.shop_code)     AS shop_name,
                    SUM(CASE WHEN s.diff_qty < 0 THEN ABS(s.diff_qty) ELSE 0 END) AS short_qty,
                    SUM(CASE WHEN s.diff_val < 0 THEN ABS(s.diff_val) ELSE 0 END) AS short_val,
                    SUM(CASE WHEN s.diff_qty > 0 THEN s.diff_qty      ELSE 0 END) AS excess_qty,
                    SUM(CASE WHEN s.diff_val > 0 THEN s.diff_val      ELSE 0 END) AS excess_val,
                    SUM(s.diff_qty) AS final_diff_qty,
                    SUM(s.diff_val) AS final_diff_val
                FROM offloading_vs_loading s
                LEFT JOIN mgr ON mgr.shop_code = UPPER(TRIM(s.shop_code))
                WHERE s.date = %s
                GROUP BY shop_manager, shop_name
                ORDER BY ABS(SUM(s.diff_val)) DESC
            """, (target_date,))
            shops = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    return {
        "total_loaded_qty":    float(kpi.get("total_loaded_qty")    or 0),
        "total_offloaded_qty": float(kpi.get("total_offloaded_qty") or 0),
        "total_diff_qty":      float(kpi.get("total_diff_qty")      or 0),
        "total_diff_val":      float(kpi.get("total_diff_val")      or 0),
        "shops": shops,
    }


def _num(v, decimals: int = 2) -> str:
    try:
        f = float(v or 0)
        return f"{f:,.{decimals}f}"
    except Exception:
        return "0.00"


def _diff_color(v) -> str:
    try:
        return "#ef4444" if float(v or 0) < 0 else ("#22c55e" if float(v or 0) > 0 else "#94a3b8")
    except Exception:
        return "#94a3b8"


def _build_html_fallback(target_date: date, rows_loaded: int) -> str:
    """
    HTML email body built from DB data (used when screenshot is unavailable).
    Mirrors the dashboard layout: header, KPI cards, Exception Shopwise table.
    """
    data = _get_email_data(target_date)
    date_label = target_date.strftime("%d %b %Y")
    date_range = f"{target_date.strftime('%Y/%m/%d')} → {target_date.strftime('%Y/%m/%d')}"
    diff_qty_color = _diff_color(data["total_diff_qty"])
    diff_val_color = _diff_color(data["total_diff_val"])

    # ── Shop rows HTML ────────────────────────────────────────────────────────
    shop_rows_html = ""
    for r in data["shops"]:
        fdq_color = _diff_color(r["final_diff_qty"])
        fdv_color = _diff_color(r["final_diff_val"])
        shop_rows_html += f"""
        <tr style="border-bottom:1px solid #1e3a6e;">
          <td style="padding:8px 10px;color:#cbd5e1;font-size:12px;">{r['shop_manager']}</td>
          <td style="padding:8px 10px;color:#e2e8f0;font-size:12px;font-weight:600;">{r['shop_name']}</td>
          <td style="padding:8px 10px;color:#cbd5e1;font-size:12px;text-align:right;">{_num(r['short_qty'], 0)}</td>
          <td style="padding:8px 10px;color:#cbd5e1;font-size:12px;text-align:right;">{_num(r['short_val'])}</td>
          <td style="padding:8px 10px;color:#cbd5e1;font-size:12px;text-align:right;">{_num(r['excess_qty'], 0)}</td>
          <td style="padding:8px 10px;color:#cbd5e1;font-size:12px;text-align:right;">{_num(r['excess_val'])}</td>
          <td style="padding:8px 10px;font-size:12px;font-weight:700;text-align:right;color:{fdq_color};">{_num(r['final_diff_qty'], 0)}</td>
          <td style="padding:8px 10px;font-size:12px;font-weight:700;text-align:right;color:{fdv_color};">{_num(r['final_diff_val'])}</td>
        </tr>"""

    # Totals row
    t_fdq_color = _diff_color(data["total_diff_qty"])
    t_fdv_color = _diff_color(data["total_diff_val"])
    total_short_qty = sum(float(r.get("short_qty") or 0) for r in data["shops"])
    total_short_val = sum(float(r.get("short_val") or 0) for r in data["shops"])
    total_excess_qty = sum(float(r.get("excess_qty") or 0) for r in data["shops"])
    total_excess_val = sum(float(r.get("excess_val") or 0) for r in data["shops"])

    return f"""<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background-color:#0a1020;font-family:Arial,Helvetica,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0a1020;">
<tr><td align="center" style="padding:20px 10px;">

<table width="820" cellpadding="0" cellspacing="0" style="background-color:#0a1020;max-width:820px;">

  <!-- ── HEADER ── -->
  <tr>
    <td style="padding:0 0 16px 0;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td style="vertical-align:middle;">
            <span style="color:#f1f5f9;font-size:17px;font-weight:800;
                         letter-spacing:-0.3px;">Loading vs Offloading Dashboard</span><br>
            <span style="color:#94a3b8;font-size:11px;">WH Database &nbsp;·&nbsp; Tabular reconciliation view</span>
          </td>
          <td align="right" style="vertical-align:middle;">
            <span style="color:#6ee7b7;font-size:10px;font-weight:700;
                         text-transform:uppercase;letter-spacing:0.5px;">Date Range</span><br>
            <span style="color:#e2e8f0;font-size:13px;font-weight:700;">{date_range}</span>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- ── SELECTED RANGE CAPTION ── -->
  <tr>
    <td style="padding:0 0 14px 0;">
      <span style="color:#94a3b8;font-size:11px;">
        Selected Range: {date_label} → {date_label}
      </span>
    </td>
  </tr>

  <!-- ── KPI CARDS ── -->
  <tr>
    <td style="padding:0 0 20px 0;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td width="25%" style="padding:0 6px 0 0;">
            <table width="100%" cellpadding="12" cellspacing="0"
                   style="background-color:#0c1830;border-radius:8px;
                          border:1px solid #1e3a6e;">
              <tr><td>
                <div style="color:#94a3b8;font-size:9px;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.6px;">WH LOADED QTY</div>
                <div style="color:#e2e8f0;font-size:22px;font-weight:800;
                            margin-top:4px;">{_num(data['total_loaded_qty'], 0)}</div>
              </td></tr>
            </table>
          </td>
          <td width="25%" style="padding:0 6px;">
            <table width="100%" cellpadding="12" cellspacing="0"
                   style="background-color:#0c1830;border-radius:8px;
                          border:1px solid #1e3a6e;">
              <tr><td>
                <div style="color:#94a3b8;font-size:9px;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.6px;">SHOP RECEIVING QTY</div>
                <div style="color:#e2e8f0;font-size:22px;font-weight:800;
                            margin-top:4px;">{_num(data['total_offloaded_qty'], 0)}</div>
              </td></tr>
            </table>
          </td>
          <td width="25%" style="padding:0 6px;">
            <table width="100%" cellpadding="12" cellspacing="0"
                   style="background-color:#0c1830;border-radius:8px;
                          border:1px solid #1e3a6e;">
              <tr><td>
                <div style="color:#94a3b8;font-size:9px;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.6px;">FINAL DIFFERENCE QTY</div>
                <div style="color:{diff_qty_color};font-size:22px;font-weight:800;
                            margin-top:4px;">{_num(data['total_diff_qty'], 0)}</div>
              </td></tr>
            </table>
          </td>
          <td width="25%" style="padding:0 0 0 6px;">
            <table width="100%" cellpadding="12" cellspacing="0"
                   style="background-color:#0c1830;border-radius:8px;
                          border:1px solid #1e3a6e;">
              <tr><td>
                <div style="color:#94a3b8;font-size:9px;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.6px;">FINAL DIFFERENCE VALUE</div>
                <div style="color:{diff_val_color};font-size:22px;font-weight:800;
                            margin-top:4px;">{_num(data['total_diff_val'])}</div>
              </td></tr>
            </table>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- ── EXCEPTION SHOPWISE DIFFERENCE TABLE ── -->
  <tr>
    <td>
      <div style="color:#dbeafe;font-size:13px;font-weight:700;margin-bottom:8px;">
        Exception Shopwise Difference
      </div>
      <table width="100%" cellpadding="0" cellspacing="0"
             style="border-radius:8px;overflow:hidden;border:1px solid #1e3a6e;">
        <thead>
          <tr style="background-color:#0d2b8e;">
            <th style="padding:9px 10px;color:#bfdbfe;font-size:10px;font-weight:700;
                       text-transform:uppercase;letter-spacing:0.4px;text-align:left;">SHOP MANAGER</th>
            <th style="padding:9px 10px;color:#bfdbfe;font-size:10px;font-weight:700;
                       text-transform:uppercase;letter-spacing:0.4px;text-align:left;">SHOP NAME</th>
            <th style="padding:9px 10px;color:#bfdbfe;font-size:10px;font-weight:700;
                       text-transform:uppercase;letter-spacing:0.4px;text-align:right;">SHORT QTY</th>
            <th style="padding:9px 10px;color:#bfdbfe;font-size:10px;font-weight:700;
                       text-transform:uppercase;letter-spacing:0.4px;text-align:right;">SHORT VALUE</th>
            <th style="padding:9px 10px;color:#bfdbfe;font-size:10px;font-weight:700;
                       text-transform:uppercase;letter-spacing:0.4px;text-align:right;">EXCESS QTY</th>
            <th style="padding:9px 10px;color:#bfdbfe;font-size:10px;font-weight:700;
                       text-transform:uppercase;letter-spacing:0.4px;text-align:right;">EXCESS VALUE</th>
            <th style="padding:9px 10px;color:#bfdbfe;font-size:10px;font-weight:700;
                       text-transform:uppercase;letter-spacing:0.4px;text-align:right;">FINAL DIFF QTY</th>
            <th style="padding:9px 10px;color:#bfdbfe;font-size:10px;font-weight:700;
                       text-transform:uppercase;letter-spacing:0.4px;text-align:right;">FINAL DIFF VALUE</th>
          </tr>
        </thead>
        <tbody style="background-color:#0c1830;">
          {shop_rows_html}
          <!-- TOTAL ROW -->
          <tr style="background-color:#0d1f4a;border-top:2px solid #1e3a6e;">
            <td style="padding:9px 10px;color:#e2e8f0;font-size:12px;font-weight:800;">TOTAL</td>
            <td style="padding:9px 10px;"></td>
            <td style="padding:9px 10px;color:#e2e8f0;font-size:12px;font-weight:800;text-align:right;">{_num(total_short_qty, 0)}</td>
            <td style="padding:9px 10px;color:#e2e8f0;font-size:12px;font-weight:800;text-align:right;">{_num(total_short_val)}</td>
            <td style="padding:9px 10px;color:#e2e8f0;font-size:12px;font-weight:800;text-align:right;">{_num(total_excess_qty, 0)}</td>
            <td style="padding:9px 10px;color:#e2e8f0;font-size:12px;font-weight:800;text-align:right;">{_num(total_excess_val)}</td>
            <td style="padding:9px 10px;font-size:12px;font-weight:800;text-align:right;color:{t_fdq_color};">{_num(data['total_diff_qty'], 0)}</td>
            <td style="padding:9px 10px;font-size:12px;font-weight:800;text-align:right;color:{t_fdv_color};">{_num(data['total_diff_val'])}</td>
          </tr>
        </tbody>
      </table>
    </td>
  </tr>

  <!-- ── FOOTER ── -->
  <tr>
    <td style="padding:18px 0 0 0;border-top:1px solid #1e3a6e;margin-top:16px;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td>
            <span style="color:#94a3b8;font-size:11px;">
              Data updated for <strong style="color:#e2e8f0;">{date_label}</strong> &nbsp;·&nbsp;
              {rows_loaded:,} records loaded
            </span><br>
            <a href="{DASHBOARD_URL}" style="color:#60a5fa;font-size:11px;text-decoration:none;">
              {DASHBOARD_URL}
            </a>
            <span style="color:#94a3b8;font-size:10px;"> — accessible on MELCOM AP network only</span>
          </td>
          <td align="right">
            <span style="color:#475569;font-size:10px;">MIS Team &nbsp;·&nbsp; Auto-generated</span>
          </td>
        </tr>
      </table>
    </td>
  </tr>

</table>
</td></tr>
</table>
</body>
</html>"""


def build_email_html(target_date: date, rows_loaded: int) -> str:
    """
    Take an actual screenshot of the dashboard (Selenium headless Chrome).
    Embeds the screenshot as a base64 PNG in a minimal HTML wrapper.
    Falls back to the HTML-table body if screenshot fails.
    """
    date_label = target_date.strftime("%d %b %Y")
    date_range = f"{target_date.strftime('%Y/%m/%d')} → {target_date.strftime('%Y/%m/%d')}"

    screenshot = _take_dashboard_screenshot(target_date)

    if screenshot:
        b64 = base64.b64encode(screenshot).decode("ascii")
        img_tag = (
            f'<img src="data:image/png;base64,{b64}" '
            f'alt="Loading vs Offloading Dashboard — {date_label}" '
            f'style="max-width:100%;border-radius:8px;'
            f'border:1px solid #1e3a6e;display:block;" />'
        )
        log.info("Screenshot embedded in email body")
    else:
        # No screenshot — fall back to the full HTML-table body
        log.info("No screenshot — using HTML fallback body")
        return _build_html_fallback(target_date, rows_loaded)

    return f"""<!DOCTYPE html>
<html>
<body style="margin:0;padding:20px 24px;background-color:#0a1020;
             font-family:Arial,Helvetica,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0"
         style="max-width:1440px;margin:0 auto;">

    <!-- header -->
    <tr><td style="padding:0 0 10px 0;">
      <span style="color:#94a3b8;font-size:12px;">
        Loading vs Offloading Dashboard &nbsp;·&nbsp;
        <strong style="color:#e2e8f0;">{date_range}</strong>
        &nbsp;·&nbsp; {rows_loaded:,} records loaded
      </span>
    </td></tr>

    <!-- screenshot -->
    <tr><td>{img_tag}</td></tr>

    <!-- footer -->
    <tr><td style="padding:14px 0 0 0;border-top:1px solid #1e3a6e;">
      <a href="{DASHBOARD_URL}" style="color:#60a5fa;font-size:11px;
         text-decoration:none;">{DASHBOARD_URL}</a>
      <span style="color:#94a3b8;font-size:10px;">
        &nbsp;— accessible on MELCOM AP network only
      </span>
    </td></tr>

  </table>
</body>
</html>"""


# ── Outlook COM sender (saves to Sent Items automatically) ───────────────────

def _send_via_outlook(subject: str, html_body: str,
                      excel_bytes: bytes | None, target_date: date) -> None:
    import win32com.client
    import tempfile

    outlook = win32com.client.Dispatch("Outlook.Application")
    mail = outlook.CreateItem(0)   # 0 = olMailItem
    mail.To      = EMAIL_TO
    mail.Subject = subject
    mail.HTMLBody = html_body

    tmp_path = None
    if excel_bytes:
        # Fixed filename so the attachment shows as "loading_vs_offloading.xlsx"
        tmp_path = os.path.join(tempfile.gettempdir(), "loading_vs_offloading.xlsx")
        with open(tmp_path, "wb") as _f:
            _f.write(excel_bytes)
        mail.Attachments.Add(tmp_path)

    try:
        mail.Send()   # sends AND saves copy to Outlook Sent Items
        log.info("Email sent via Outlook → %s  (saved to Sent Items)", EMAIL_TO)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ── SMTP fallback ─────────────────────────────────────────────────────────────

def _send_via_smtp(subject: str, html_body: str,
                   excel_bytes: bytes | None, target_date: date) -> None:
    msg = MIMEMultipart("related")
    msg["From"], msg["To"], msg["Subject"] = EMAIL_FROM, EMAIL_TO, subject
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    if excel_bytes:
        filename = f"loading_vs_offloading_detail_{target_date.strftime('%d%b%Y')}.xlsx"
        part = MIMEBase("application",
                        "vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        part.set_payload(excel_bytes)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
        msg.attach(part)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
        if SMTP_USER and SMTP_PASS:
            server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())
    log.info("Email sent via SMTP → %s  subject: %s", EMAIL_TO, subject)


# ── Public entry point ────────────────────────────────────────────────────────

def send_email(target_date: date, rows_loaded: int,
               excel_bytes: bytes | None = None) -> None:
    """
    Build HTML email body and send via Outlook COM (saves to Sent Items).
    Falls back to SMTP if Outlook COM fails.
    Raises RuntimeError with details if BOTH methods fail.
    """
    subject = (
        f"Loading vs Offloading dashboard updated till "
        f"{format_date_subject(target_date)}"
    )

    # Build HTML body — fall back to a plain text body if DB query fails
    try:
        html_body = build_email_html(target_date, rows_loaded)
    except Exception as e_html:
        log.warning("HTML body build failed (%s) — sending plain text fallback", e_html)
        html_body = (
            f"<p>Hello,</p>"
            f"<p>Loading vs Offloading data updated for "
            f"<strong>{target_date.strftime('%d %b %Y')}</strong>.</p>"
            f"<p>{rows_loaded:,} records loaded.</p>"
            f"<p><a href='{DASHBOARD_URL}'>{DASHBOARD_URL}</a></p>"
            f"<p>Regards,<br>MIS Team</p>"
        )

    # ── Try Outlook COM first (saves to Sent Items automatically) ────────────
    outlook_err = None
    try:
        _send_via_outlook(subject, html_body, excel_bytes, target_date)
        log.info("Email sent via Outlook COM → %s", EMAIL_TO)
        return
    except Exception as e:
        outlook_err = e
        log.warning("Outlook COM failed: %s — trying SMTP fallback", e)

    # ── SMTP fallback ─────────────────────────────────────────────────────────
    smtp_err = None
    try:
        _send_via_smtp(subject, html_body, excel_bytes, target_date)
        log.info("Email sent via SMTP → %s", EMAIL_TO)
        return
    except Exception as e:
        smtp_err = e
        log.error("SMTP also failed: %s", e)

    # Both failed — raise so the caller can show the error to the user
    raise RuntimeError(
        f"Email delivery failed.\n"
        f"  Outlook COM: {outlook_err}\n"
        f"  SMTP ({SMTP_HOST}:{SMTP_PORT}): {smtp_err}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_sync(target_date: date | None = None, force: bool = False) -> dict:
    """
    Sync one day's WMS data into offloading_vs_loading.

    Returns:
        {"status": "ok"|"skipped"|"error", "message": str,
         "rows_loaded": int, "date": date, "email_sent": bool}
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    result = {"status": "ok", "message": "", "rows_loaded": 0,
              "date": target_date, "email_sent": False}

    if not force and date_exists_in_db(target_date):
        result["status"]  = "skipped"
        result["message"] = (
            f"Data for {target_date} is already loaded. "
            "Click again to force re-import."
        )
        log.info(result["message"])
        return result

    # ── Find CSV ──────────────────────────────────────────────────────────────
    try:
        csv_path = find_wms_file(target_date)
    except OSError as e:
        result["status"] = "error"
        result["message"] = str(e)
        log.error(result["message"])
        return result

    if not csv_path:
        result["status"]  = "error"
        result["message"] = (
            f"WMS_shopdiff file for {target_date.strftime('%d %b %Y')} "
            f"not found in {NETWORK_PATH}. "
            f"Expected: WMS_shopdiff_{target_date.strftime('%d_%m_%Y')}.csv"
        )
        log.warning(result["message"])
        return result

    log.info("Found CSV: %s", csv_path)

    # ── Read + normalise (home_dashboard pipeline) ────────────────────────────
    try:
        df_raw = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
        df_norm = _prepare_lvo_offloading_df(df_raw, os.path.basename(csv_path))
        df_norm = _rename_by_expected_columns(df_norm, LVO_COLUMNS)
        missing = [c for c in LVO_COLUMNS if c not in df_norm.columns]
        if missing:
            raise ValueError(f"Missing columns after normalisation: {', '.join(missing)}")
        df_upload = df_norm[LVO_COLUMNS].copy()
        # Final type coercion (same as home_dashboard upload_data_to_table)
        df_upload['date'] = pd.to_datetime(df_upload['date'], errors='coerce').dt.date
        for nc in ['qty_loaded','value_loaded','qty_offloaded','value_offloaded','diff_qty','diff_val']:
            df_upload[nc] = pd.to_numeric(df_upload[nc], errors='coerce').fillna(0)
        # Filter to target_date only (skip rows with different/null dates)
        df_upload = df_upload[df_upload['date'] == target_date].copy()
        if df_upload.empty:
            raise ValueError(
                f"After column normalisation, 0 rows have date={target_date}. "
                "Check that the CSV offloading_date column is in DD-Mon-YY format."
            )
    except Exception as e:
        result["status"]  = "error"
        result["message"] = f"CSV parse error: {e}"
        log.error(result["message"], exc_info=True)
        return result

    # ── Write to DB ───────────────────────────────────────────────────────────
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        try:
            _ensure_loadingvsoffloading_tables(conn)
            rows = _load_df_to_db(conn, df_upload, target_date, os.path.basename(csv_path))
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        result["rows_loaded"] = rows
    except Exception as e:
        result["status"]  = "error"
        result["message"] = f"DB load failed: {e}"
        log.error(result["message"], exc_info=True)
        return result

    result["message"] = (
        f"Loaded {rows:,} rows for {target_date.strftime('%d %b %Y')} "
        f"from {os.path.basename(csv_path)}"
    )

    # ── Excel + email ─────────────────────────────────────────────────────────
    try:
        excel_bytes = build_excel_bytes(target_date) if OPENPYXL_OK else None
        send_email(target_date, rows, excel_bytes)
        result["email_sent"] = True
    except Exception as e:
        log.warning("Email failed (data was loaded OK): %s", e)
        result["message"] += f"  ⚠ Email failed: {e}"

    log.info("Sync complete: %s", result["message"])
    return result


if __name__ == "__main__":
    import sys
    target = None
    if len(sys.argv) > 1:
        try:
            target = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        except ValueError:
            print("Usage: python wms_sync.py [YYYY-MM-DD] [--force]")
            sys.exit(1)
    res = run_sync(target_date=target, force="--force" in sys.argv)
    print(f"[{res['status'].upper()}] {res['message']}")
    sys.exit(0 if res["status"] in ("ok", "skipped") else 1)
