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
from psycopg2.extras import RealDictCursor, Json, execute_values
import time
from openpyxl.styles import PatternFill, Font

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

# ── MySQL source config (invcentral) ───────────────────────────────────────────
MYSQL_CONFIG = {
    "host":     "192.168.0.17",
    "user":     "misaccount",
    "password": "Inv@Central@2024",
    "database": "invcentral",
}

# PostgreSQL target config for Delete & Re-Fetch
PG_CONFIG_REFETCH = {
    "host": "localhost", "port": 3307,
    "user": "postgres", "password": "hello", "database": "WH",
}

# Tables fetched from MySQL invcentral
MYSQL_REFETCH_TABLES = ["ALERTS", "INVOICES", "invoices_manager"]
# ERPDATA is sourced from \\10.10.0.30\mis\shopbillcount_YYYYMMDD.csv (not MySQL)


# ── Delete & Re-Fetch helpers (same logic as home_dashboard.py) ────────────────

def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _calculate_adaptive_batch_size(table_name: str, src_rows: int, base: int = 20000) -> int:
    if src_rows < 50000:   return base
    if src_rows < 200000:  return max(5000, base // 2)
    if src_rows < 500000:  return 5000
    return 2000


def _refetch_tables_from_date(
    mysql_config: dict,
    pg_config: dict,
    from_date_str: str,
    tables: list,
    batch_size: int = 20000,
    status_fn=None,
) -> list:
    """
    Identical process to home_dashboard.delete_and_refetch_tables_from_date.
    Deletes rows >= from_date_str from PostgreSQL WH tables, then re-fetches
    from MySQL (invcentral) and re-inserts.

    status_fn(msg) is called for each progress update (used by st.status.write).
    """
    import pymysql
    import pymysql.cursors

    def _log(msg):
        if status_fn:
            status_fn(msg)

    TABLE_DATE_CFG = {
        'alerts':           {'pg_col': 'a_entrytime', 'mysql_col': 'a_ENTRYTIME', 'is_varchar': False},
        'erpdata':          {'pg_col': 'invdate',     'mysql_col': 'invdate',     'is_varchar': True},
        'invoices':         {'pg_col': 'invdate',     'mysql_col': 'invdate',     'is_varchar': True},
        'invoices_manager': {'pg_col': 'invdate',     'mysql_col': 'invdate',     'is_varchar': True},
    }

    from_dt_iso     = from_date_str
    from_dt_compact = from_date_str.replace('-', '')

    results = []
    mysql_conn = pg_conn = None
    try:
        mysql_conn = pymysql.connect(
            host=mysql_config['host'], user=mysql_config['user'],
            password=mysql_config['password'], database=mysql_config['database'],
            charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=20, read_timeout=300,
        )
        pg_conn = psycopg2.connect(**pg_config)

        for tbl in tables:
            tbl_lower = tbl.lower()
            cfg = TABLE_DATE_CFG.get(tbl_lower)
            if not cfg:
                results.append({'table': tbl_lower, 'deleted': 0, 'fetched': 0,
                                 'inserted': 0, 'status': 'skipped', 'error': 'Unknown table'})
                continue

            pg_col    = cfg['pg_col']
            mysql_col = cfg['mysql_col']

            # ── Step 1: Delete from PostgreSQL ────────────────────────────────
            try:
                with pg_conn.cursor() as cur:
                    cur.execute(
                        f'DELETE FROM {_quote_ident(tbl_lower)} '
                        f'WHERE {_quote_ident(pg_col)}::date >= %s',
                        (from_dt_iso,)
                    )
                    deleted = cur.rowcount or 0
                    pg_conn.commit()
            except Exception as de:
                pg_conn.rollback()
                results.append({'table': tbl_lower, 'deleted': 0, 'fetched': 0,
                                 'inserted': 0, 'status': 'error', 'error': f'Delete failed: {de}'})
                _log(f"❌ `{tbl_lower}`: delete error — {de}")
                continue

            _log(f"🗑 `{tbl_lower}`: deleted {deleted:,} rows from {from_dt_iso} onwards")

            # ── Step 2: Column list from MySQL ────────────────────────────────
            mysql_cur = mysql_conn.cursor()
            mysql_cur.execute(
                "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s ORDER BY ORDINAL_POSITION",
                (mysql_config['database'], tbl)
            )
            col_meta = mysql_cur.fetchall()
            if not col_meta:
                results.append({'table': tbl_lower, 'deleted': deleted, 'fetched': 0,
                                 'inserted': 0, 'status': 'error',
                                 'error': f'MySQL table {tbl} not found'})
                _log(f"❌ MySQL table `{tbl}` not found in `{mysql_config['database']}`")
                continue

            columns = [r['COLUMN_NAME'] for r in col_meta]
            pg_cols = [c.lower() for c in columns]

            # ── Step 3: Primary key for conflict handling ──────────────────────
            try:
                with pg_conn.cursor() as cur:
                    cur.execute("""
                        SELECT a.attname FROM pg_index i
                        JOIN pg_attribute a ON a.attrelid = i.indrelid
                                           AND a.attnum = ANY(i.indkey)
                        WHERE i.indrelid = %s::regclass AND i.indisprimary
                    """, (tbl_lower,))
                    pk_cols = [r[0] for r in cur.fetchall()]
            except Exception:
                pk_cols = []

            conflict_sql = (
                "ON CONFLICT ({}) DO NOTHING".format(
                    ', '.join([_quote_ident(c) for c in pk_cols])
                ) if pk_cols else ""
            )

            # Disable triggers on invoices during load
            triggers_disabled = False
            if tbl_lower == 'invoices':
                try:
                    with pg_conn.cursor() as cur:
                        cur.execute("ALTER TABLE public.invoices DISABLE TRIGGER USER")
                    pg_conn.commit()
                    triggers_disabled = True
                    _log("⏸️ `invoices`: triggers disabled for load")
                except Exception:
                    pg_conn.rollback()

            # ── Step 4: Fetch from MySQL + insert into PostgreSQL ─────────────
            select_cols = ', '.join([f'`{c}`' for c in columns])
            if cfg['is_varchar']:
                where_sql, mysql_param = f"WHERE `{mysql_col}` >= %s", from_dt_compact
            else:
                where_sql, mysql_param = f"WHERE DATE(`{mysql_col}`) >= %s", from_dt_iso

            insert_col_sql = ', '.join([_quote_ident(c) for c in pg_cols])
            ins_sql = (f"INSERT INTO {_quote_ident(tbl_lower)} ({insert_col_sql}) "
                       f"VALUES %s {conflict_sql}")

            # Estimate row count for adaptive batch sizing
            try:
                mysql_cur.execute(
                    "SELECT TABLE_ROWS FROM INFORMATION_SCHEMA.TABLES "
                    "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s",
                    (mysql_config['database'], tbl)
                )
                est = mysql_cur.fetchone()
                src_row_est = est['TABLE_ROWS'] if est else 0
            except Exception:
                src_row_est = 0
            eff_batch = _calculate_adaptive_batch_size(tbl, src_row_est, batch_size)

            total_fetched = total_inserted = batch_counter = 0
            id_col = next((c for c in columns if c.lower() == 'id'), None)

            if cfg['is_varchar'] and id_col:
                _log(f"⚡ `{tbl_lower}`: keyset pagination on `{id_col}`")
                last_id = 0
                while True:
                    try:
                        mysql_cur.execute(
                            f"SELECT {select_cols} FROM `{tbl}` "
                            f"WHERE `{mysql_col}` >= %s AND `{id_col}` > %s "
                            f"ORDER BY `{id_col}` ASC LIMIT {eff_batch}",
                            [mysql_param, last_id]
                        )
                        rows = mysql_cur.fetchall() or []
                    except Exception as fe:
                        _log(f"❌ `{tbl_lower}` fetch error: {fe}")
                        break
                    if not rows:
                        break
                    value_rows = [
                        tuple([{k.lower(): v for k, v in r.items()}.get(c) for c in pg_cols])
                        for r in rows
                    ]
                    try:
                        with pg_conn.cursor() as cur:
                            execute_values(cur, ins_sql, value_rows, page_size=min(eff_batch, 5000))
                            pg_conn.commit()
                            total_inserted += max(0, cur.rowcount or 0)
                    except Exception as ie:
                        pg_conn.rollback()
                        _log(f"⚠️ `{tbl_lower}` insert error: {ie}")
                        break
                    total_fetched += len(rows)
                    batch_counter += 1
                    try:
                        last_id = int(rows[-1].get(id_col) or last_id)
                    except Exception:
                        pass
                    if batch_counter % 10 == 0:
                        try: mysql_conn.ping(reconnect=True)
                        except Exception: pass
                    if batch_counter % 5 == 0:
                        _log(f"⏳ `{tbl_lower}`: {total_fetched:,} fetched / {total_inserted:,} inserted…")
            else:
                data_cur = mysql_conn.cursor(pymysql.cursors.SSDictCursor)
                try:
                    data_cur.execute(
                        f"SELECT {select_cols} FROM `{tbl}` {where_sql} ORDER BY `{mysql_col}` ASC",
                        [mysql_param]
                    )
                except Exception as fe:
                    _log(f"❌ `{tbl_lower}` fetch error: {fe}")
                    data_cur.close()
                    results.append({'table': tbl_lower, 'deleted': deleted,
                                    'fetched': 0, 'inserted': 0, 'status': 'error',
                                    'error': f'Fetch failed: {fe}'})
                    continue
                while True:
                    rows = data_cur.fetchmany(eff_batch)
                    if not rows:
                        break
                    value_rows = [
                        tuple([{k.lower(): v for k, v in r.items()}.get(c) for c in pg_cols])
                        for r in rows
                    ]
                    try:
                        with pg_conn.cursor() as cur:
                            execute_values(cur, ins_sql, value_rows, page_size=min(eff_batch, 5000))
                            pg_conn.commit()
                            total_inserted += max(0, cur.rowcount or 0)
                    except Exception as ie:
                        pg_conn.rollback()
                        _log(f"⚠️ `{tbl_lower}` insert error: {ie}")
                        break
                    total_fetched += len(rows)
                    batch_counter += 1
                    if batch_counter % 10 == 0:
                        try: mysql_conn.ping(reconnect=True)
                        except Exception: pass
                    if batch_counter % 5 == 0:
                        _log(f"⏳ `{tbl_lower}`: {total_fetched:,} fetched / {total_inserted:,} inserted…")
                data_cur.close()

            # Re-enable triggers
            if triggers_disabled:
                try:
                    with pg_conn.cursor() as cur:
                        cur.execute("ALTER TABLE public.invoices ENABLE TRIGGER USER")
                    pg_conn.commit()
                    _log("▶️ `invoices`: triggers re-enabled")
                except Exception as te:
                    pg_conn.rollback()
                    _log(f"❌ Failed to re-enable invoices triggers: {te}")

            results.append({'table': tbl_lower, 'deleted': deleted,
                            'fetched': total_fetched, 'inserted': total_inserted,
                            'status': 'success', 'error': None})
            _log(f"✅ `{tbl_lower}`: deleted {deleted:,} · fetched {total_fetched:,} · "
                 f"inserted {total_inserted:,}")

    finally:
        for c in (mysql_conn, pg_conn):
            try: c.close() if c else None
            except Exception: pass

    return results


def _normalize_invoices_duplicates(pg_config: dict) -> int:
    """Recompute invoices.duplicate flags — same logic as home_dashboard."""
    sql = """
        WITH ranked AS (
            SELECT id,
                CASE
                    WHEN fullqrcode IS NULL OR TRIM(fullqrcode) = '' THEN 1
                    WHEN ROW_NUMBER() OVER (PARTITION BY fullqrcode ORDER BY id) = 1 THEN 1
                    ELSE 0
                END AS correct_duplicate
            FROM public.invoices
        )
        UPDATE public.invoices i
        SET duplicate = r.correct_duplicate
        FROM ranked r
        WHERE i.id = r.id AND i.duplicate IS DISTINCT FROM r.correct_duplicate
    """
    with psycopg2.connect(**pg_config) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            updated = cur.rowcount or 0
        conn.commit()
    return updated


def _serialize_row_for_hash(row_dict: dict, ordered_columns: list) -> str:
    """Deterministic string representation of a row for MD5 dedup hash."""
    return '|'.join('' if row_dict.get(c) is None else str(row_dict.get(c))
                    for c in ordered_columns)


# ══════════════════════════════════════════════════════════════════════════════
# POST-UPDATE EMAIL  (Outlook COM → saves to Sent Items)
# ══════════════════════════════════════════════════════════════════════════════

INVOICE_EMAIL_TO   = "mis.manager@melcomgroup.com"
INVOICE_EMAIL_FROM = "mis.manager@melcomgroup.com"
INVOICE_SMTP_HOST  = "mail.melcomgroup.com"
INVOICE_SMTP_PORT  = 25
INVOICE_DASHBOARD_URL = "http://10.10.1.79:8521"   # update port if different

_MONTH_ABBR = {1:'JAN',2:'FEB',3:'MAR',4:'APR',5:'MAY',6:'JUN',
               7:'JUL',8:'AUG',9:'SEP',10:'OCT',11:'NOV',12:'DEC'}

def _invoice_date_label(d: date) -> str:
    """'07-MAY-26' format for email subject."""
    return f"{d.day:02d}-{_MONTH_ABBR[d.month]}-{str(d.year)[-2:]}"


def _refresh_dashboard_mvs(pg_config: dict, status_fn=None) -> None:
    """
    REFRESH MATERIALIZED VIEW CONCURRENTLY for every MV the invoice
    scanning dashboard reads from.  Must be called after data is loaded
    (ERPDATA + MySQL tables) so the MVs reflect the latest rows.
    """
    MVS = [
        'mv_wh_erp_daily',
        'mv_wh_scan_daily',
        'mv_wh_alerts_daily',
        'mv_erpdata_test_bills_daily',
        'mv_wh_erp_cashier_sessions_daily',
    ]
    def _log(msg):
        if status_fn:
            status_fn(msg)

    _ac = psycopg2.connect(**pg_config)
    try:
        _ac.autocommit = True
        with _ac.cursor() as cur:
            for mv in MVS:
                try:
                    cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {mv}")
                    _log(f"✅  {mv}")
                except Exception as e1:
                    try:
                        cur.execute(f"REFRESH MATERIALIZED VIEW {mv}")
                        _log(f"✅  {mv}")
                    except Exception as e2:
                        _log(f"⚠️  {mv}: {e2}")
    finally:
        _ac.close()


def _xl_pct_fill(val: float):
    """openpyxl PatternFill matching dashboard colour scheme: ≥90=green / 70–90=yellow / <70=red."""
    from openpyxl.styles import PatternFill
    if val > 100:                          # over-100 → light blue (same as dashboard)
        return PatternFill("solid", fgColor="DBEAFE")
    if val >= 90:
        return PatternFill("solid", fgColor="C6EFCE")
    if val >= 70:
        return PatternFill("solid", fgColor="FFEB9C")
    return PatternFill("solid", fgColor="FFC7CE")


def _xl_diff_fill(val: float):
    """Diff GHS: negative = green (good), positive = red (bad)."""
    from openpyxl.styles import PatternFill
    if val < 0:
        return PatternFill("solid", fgColor="C6EFCE")
    if val > 0:
        return PatternFill("solid", fgColor="FFC7CE")
    return None


def _xl_write_sheet(ws, title_text: str, n_cols: int,
                    hdrs: list, widths: list, rows_data: list) -> None:
    """Write title row, header row, then data rows to an openpyxl worksheet."""
    from openpyxl.styles import Font, Alignment, PatternFill
    title_col = chr(ord('A') + n_cols - 1)
    ws.merge_cells(f"A1:{title_col}1")
    ws["A1"] = title_text
    ws["A1"].font = Font(bold=True, size=12, color="FFFFFF")
    ws["A1"].fill = PatternFill("solid", fgColor="1F3864")
    ws["A1"].alignment = Alignment(horizontal="center")
    ws.row_dimensions[1].height = 20
    hdr_fill = PatternFill("solid", fgColor="2F5496")
    for ci, (h, w) in enumerate(zip(hdrs, widths), 1):
        c = ws.cell(row=2, column=ci, value=h)
        c.font, c.fill = Font(bold=True, color="FFFFFF", size=10), hdr_fill
        c.alignment = Alignment(horizontal="center")
        ws.column_dimensions[c.column_letter].width = w
    for ri, row_cells in enumerate(rows_data, 3):
        for ci, (val, fmt, align, fill) in enumerate(row_cells, 1):
            cell = ws.cell(row=ri, column=ci, value=val)
            if fmt:
                cell.number_format = fmt
            cell.alignment = Alignment(horizontal=align)
            if fill:
                cell.fill = fill
    ws.freeze_panes = "A3"


def _build_owner_view_excel(report_date: date) -> bytes:
    """
    Owner-view summary Excel — same data as dashboard One-View Summary table.
    Columns: exactly as to_display() + Total Alerts column.
    Sorted by Scan NOB high → low. Conditional formatting on % and GHS columns.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill

    # Use the exact same data pipeline as the dashboard (bypasses Streamlit cache)
    st.cache_data.clear()
    raw_df = load_owner_view(report_date, report_date)

    if raw_df.empty:
        wb = Workbook()
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()

    disp = to_display(raw_df)  # applies shop names + exact column set

    # Add Total Alerts column (sum of all alert types)
    alert_cols = ["Test Bill Scanned", "Bill Date Mismatched", "Duplicate",
                  "Invalid store code", "High Value Bill"]
    disp["Total Alerts"] = disp[[c for c in alert_cols if c in disp.columns]].sum(axis=1)

    # Sort by Scan NOB descending, TOTAL row first if present
    disp = disp.sort_values("Scan NOB", ascending=False)

    # ── Column metadata: (header, width, fmt, align, is_pct, is_diff_ghs) ──
    COL_META = [
        ("Shop",                    8,  None,       "left",   False, False),
        ("Name",                   28,  None,       "left",   False, False),
        ("ERP NOB",                 9,  "#,##0",    "right",  False, False),
        ("Scan NOB",                9,  "#,##0",    "right",  False, False),
        ("Consumable Till NOB",    13,  "#,##0",    "right",  False, False),
        ("ERP Test Bills",         11,  "#,##0",    "right",  False, False),
        ("Cashier Not Generated",  16,  "#,##0",    "right",  False, False),
        ("Total Accounted Bills",  16,  "#,##0",    "right",  False, False),
        ("Bills not scanned",      14,  "#,##0",    "right",  False, False),
        ("Scanned bill %",         12,  "0.00%",    "center", True,  False),
        ("ERP NOB (GHS)",          14,  "#,##0.00", "right",  False, False),
        ("Scanned NOB (GHS)",      15,  "#,##0.00", "right",  False, False),
        ("Consumable NOB (GHS)",   16,  "#,##0.00", "right",  False, False),
        ("Diff (GHS)",             14,  "#,##0.00", "right",  False, True),
        ("Scanned GHS %",          12,  "0.00%",    "center", True,  False),
        ("Total Alerts",           11,  "#,##0",    "right",  False, False),
        ("Test Bill Scanned",      14,  "#,##0",    "right",  False, False),
        ("Bill Date Mismatched",   16,  "#,##0",    "right",  False, False),
        ("Duplicate",              10,  "#,##0",    "right",  False, False),
        ("Invalid store code",     14,  "#,##0",    "right",  False, False),
        ("High Value Bill",        13,  "#,##0",    "right",  False, False),
    ]

    # Map dashboard column names to meta
    DISP_TO_META = {
        "Shop Code":                       0,
        "Shop Name":                       1,
        "ERP NOB":                         2,
        "Scan NOB":                        3,
        "Consumable Till NOB":             4,
        "ERP Test Bills":                  5,
        "Cashier Not Generated Test Bills":6,
        "Total Accounted Bills":           7,
        "Bills not scanned":               8,
        "Scanned bill %":                  9,
        "ERP ERP NOB (GHS)":              10,
        "Scanned NOB (GHS)":              11,
        "Consumable NOB (GHS)":           12,
        "Diff (GHS)":                     13,
        "Scanned GHS %":                  14,
        "Total Alerts":                   15,
        "Test Bill Scanned":              16,
        "Bill Date Mismatched":           17,
        "Duplicate":                      18,
        "Invalid store code":             19,
        "High Value Bill":                20,
    }

    ordered = [c for c in DISP_TO_META if c in disp.columns]
    col_metas = [COL_META[DISP_TO_META[c]] for c in ordered]
    hdrs   = [m[0] for m in col_metas]
    widths = [m[1] for m in col_metas]

    rows_data = []
    for _, row in disp[ordered].iterrows():
        row_cells = []
        for col_name, meta in zip(ordered, col_metas):
            _, _, fmt, align, is_pct, is_diff = meta
            raw = row[col_name]
            try:
                v = float(raw)
            except Exception:
                v = str(raw) if raw is not None else ""
            # Convert pct columns: dashboard stores 0-100, Excel needs 0-1 for 0.00%
            fill = None
            if is_pct and isinstance(v, float):
                fill = _xl_pct_fill(v)
                v = v / 100.0
            elif is_diff and isinstance(v, float):
                fill = _xl_diff_fill(v)
            row_cells.append((v, fmt, align, fill))
        rows_data.append(row_cells)

    wb = Workbook()
    ws = wb.active
    ws.title = "One-View Summary"
    _xl_write_sheet(
        ws,
        f"Invoice Scanning — One View Summary   {report_date.strftime('%d %b %Y')}",
        len(ordered), hdrs, widths, rows_data,
    )
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _build_testbill_excel(report_date: date) -> bytes:
    """
    Shopwise test bill analysis Excel — same data as dashboard test-bill table.
    All 11 columns. Sorted by Bill handover % high → low.
    Conditional formatting on %, numbers as numbers.
    """
    from openpyxl import Workbook

    st.cache_data.clear()
    raw_df = load_shopwise_test_bill_analysis(report_date, report_date)
    if raw_df is None or raw_df.empty:
        wb = Workbook()
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()

    shop_name_map = load_shop_name_map()
    raw_df["shop_name"] = raw_df["shop_code"].map(shop_name_map).fillna(raw_df["shop_code"])

    disp = raw_df.rename(columns={
        "shop_code":                    "Shop Code",
        "shop_name":                    "Shop Name",
        "cashier_login":                "Cashier Login",
        "test_bill_not_generated":      "Cashier Not Generated Test Bills",
        "test_bills_generated":         "Total Test Bill",
        "unique_test_bills_generated":  "Cashier Generated Test Bill",
        "not_generated_bill_pct":       "Not Generated Bill %",
        "generated_test_bill_pct":      "Cashier Generated Test Bill%",
        "handover_test_bill_to_manager":"Unique bill handover to manager",
        "bill_handover_pct":            "Bill handover %",
        "missing_test_bill":            "Missing test bill",
    })

    # Recalculate Cashier Generated Test Bill% from raw values (same as dashboard)
    cl   = pd.to_numeric(disp.get("Cashier Login"), errors="coerce").fillna(0)
    utg  = pd.to_numeric(disp.get("Cashier Generated Test Bill"), errors="coerce").fillna(0)
    disp["Cashier Generated Test Bill%"] = ((utg / cl.replace(0, float("nan"))) * 100).round(2).fillna(0)

    ordered_cols = [
        "Shop Code", "Shop Name", "Cashier Login",
        "Cashier Generated Test Bill", "Cashier Generated Test Bill%",
        "Not Generated Bill %", "Cashier Not Generated Test Bills",
        "Total Test Bill", "Unique bill handover to manager",
        "Missing test bill", "Bill handover %",
    ]
    disp = disp[[c for c in ordered_cols if c in disp.columns]]
    disp = disp.sort_values("Bill handover %", ascending=False)

    PCT_COLS = {"Bill handover %", "Cashier Generated Test Bill%", "Not Generated Bill %"}
    NUM_COLS = {"Cashier Login", "Cashier Generated Test Bill",
                "Cashier Not Generated Test Bills", "Total Test Bill",
                "Unique bill handover to manager", "Missing test bill"}

    WIDTHS = {
        "Shop Code": 9, "Shop Name": 28, "Cashier Login": 13,
        "Cashier Generated Test Bill": 18, "Cashier Generated Test Bill%": 18,
        "Not Generated Bill %": 18, "Cashier Not Generated Test Bills": 22,
        "Total Test Bill": 13, "Unique bill handover to manager": 22,
        "Missing test bill": 14, "Bill handover %": 14,
    }

    hdrs   = list(disp.columns)
    widths = [WIDTHS.get(c, 14) for c in hdrs]

    rows_data = []
    for _, row in disp.iterrows():
        row_cells = []
        for col_name in hdrs:
            raw = row[col_name]
            try:
                v = float(raw)
            except Exception:
                v = str(raw) if raw is not None else ""
            fill = fmt = None
            align = "left"
            if col_name in PCT_COLS and isinstance(v, float):
                fill = _xl_pct_fill(v)
                fmt  = "0.0%"
                align = "center"
                v = v / 100.0
            elif col_name in NUM_COLS and isinstance(v, float):
                fmt   = "#,##0"
                align = "right"
                v = int(v)
            row_cells.append((v, fmt, align, fill))
        rows_data.append(row_cells)

    wb = Workbook()
    ws = wb.active
    ws.title = "Shopwise Test Bill Analysis"
    _xl_write_sheet(
        ws,
        f"Shopwise Test Bill Analysis   {report_date.strftime('%d %b %Y')}",
        len(hdrs), hdrs, widths, rows_data,
    )
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def send_invoice_update_email(report_date: date, owner_excel: bytes, testbill_excel: bytes) -> None:
    """
    Send Invoice Scanning update email via Outlook COM (saves to Sent Items).
    Falls back to SMTP if Outlook is unavailable.
    """
    import tempfile, smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    date_label = _invoice_date_label(report_date)   # '07-MAY-26'
    subject    = f"Invoice Scanning Dashboard_{date_label}"

    html_body = f"""<html><body style="font-family:Arial,sans-serif;font-size:13px;color:#222;">
<p>Hello Mr Arun,</p>
<p>PFA Invoice scanning &amp; Shopwise test bill analysis</p>
<p>
  <a href="{INVOICE_DASHBOARD_URL}" style="color:#1155CC;font-weight:bold;">
    Click here to view the dashboard : - INVOICE SCANNING DASHBOARD
  </a>
</p>
<p>Regards,</p>
<p>
  <strong>Arun Pillai (AP)</strong><br>
  Manager MIS<br>
  2nd Palace Link Road,<br>
  Off Dadeban Road,<br>
  North Industrial Area,<br>
  P. O. Box 3920,<br>
  Accra, Ghana.<br>
  MB: +233 531090913<br>
  Email: <a href="mailto:MIS.Manager@melcomgroup.com">MIS.Manager@melcomgroup.com</a>
       / URL: <a href="http://www.melcomgroup.com">www.melcomgroup.com</a>
</p>
<hr style="border:none;border-top:1px solid #ccc;margin:8px 0;">
<p style="font-size:10px;color:#666;font-style:italic;">
  This mail is confidential and may also be privileged. Please delete it and notify us immediately
  if you are not the intended recipient. You should not copy or use it for any purpose nor disclose
  its contents to any other person.
</p>
</body></html>"""

    _dl = report_date.strftime('%d%b%y')   # e.g. 07May26
    attach1_name = f"INV_summary_Shopwise_{_dl}.xlsx"
    attach2_name = f"Shopwise_test_bill_analysis_{_dl}.xlsx"

    # ── Try Outlook COM first ─────────────────────────────────────────────────
    try:
        import win32com.client as _win32
        tmp1 = tmp2 = None
        outlook = _win32.Dispatch("Outlook.Application")
        mail = outlook.CreateItem(0)
        mail.To       = INVOICE_EMAIL_TO
        mail.Subject  = subject
        mail.HTMLBody = html_body
        # Save Excel files to temp and attach
        for xls, name in [(owner_excel, attach1_name), (testbill_excel, attach2_name)]:
            fd, tmp = tempfile.mkstemp(suffix=f"_{name}")
            try:
                os.write(fd, xls)
            finally:
                os.close(fd)
            mail.Attachments.Add(tmp)
            if tmp1 is None:
                tmp1 = tmp
            else:
                tmp2 = tmp
        mail.Send()
        for t in (tmp1, tmp2):
            try:
                if t: os.unlink(t)
            except Exception:
                pass
        return
    except Exception as _oe:
        logger.warning("Outlook COM failed: %s — trying SMTP", _oe)

    # ── SMTP fallback ─────────────────────────────────────────────────────────
    msg = MIMEMultipart()
    msg["From"], msg["To"], msg["Subject"] = INVOICE_EMAIL_FROM, INVOICE_EMAIL_TO, subject
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    for xls, name in [(owner_excel, attach1_name), (testbill_excel, attach2_name)]:
        part = MIMEBase("application", "vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        part.set_payload(xls)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{name}"')
        msg.attach(part)
    with smtplib.SMTP(INVOICE_SMTP_HOST, INVOICE_SMTP_PORT, timeout=15) as srv:
        srv.sendmail(INVOICE_EMAIL_FROM, [INVOICE_EMAIL_TO], msg.as_string())


ERPDATA_CSV_PATH = r"\\10.10.0.30\mis"

# Explicit column mapping: CSV columns (left-to-right) → erpdata table columns.
# Update this list if the shopbillcount CSV column order changes.
ERPDATA_COPY_COLUMNS = ['invno', 'store_code', 'amt', 'invdate', 'entry_time', 'tillno', 'cashier']


def _sync_erpdata_for_date(target_date: date, pg_config: dict, status_fn=None) -> dict:
    """
    Fast ERPDATA update:
      Autocommit phase  → create index on invdate (makes DELETE fast),
                          drop NOT NULL, disable ALL triggers
      Transaction phase → DELETE for date, COPY streamed direct from file,
                          re-enable triggers
    """
    import os as _os

    date_str = target_date.strftime('%Y%m%d')
    csv_file = f"shopbillcount_{date_str}.csv"
    csv_path = _os.path.join(ERPDATA_CSV_PATH, csv_file)

    def _log(msg):
        if status_fn:
            status_fn(msg)

    result = {
        'table': 'erpdata', 'deleted': 0, 'fetched': 0,
        'inserted': 0, 'status': 'success', 'error': None,
    }

    if not _os.path.exists(ERPDATA_CSV_PATH):
        msg = f"Network share `{ERPDATA_CSV_PATH}` not accessible"
        _log(f"❌ {msg}")
        return {**result, 'status': 'error', 'error': msg}

    if not _os.path.exists(csv_path):
        msg = f"`{csv_file}` not found in `{ERPDATA_CSV_PATH}`"
        _log(f"⚠️ {msg}")
        return {**result, 'status': 'error', 'error': msg}

    _log(f"📄 Found: `{csv_file}`")

    # ── Autocommit phase: schema + index + NOT NULL drop + disable triggers ───
    # All in autocommit so each statement is its own tx — no lock escalation.
    _ac = psycopg2.connect(**pg_config)
    try:
        _ac.autocommit = True
        with _ac.cursor() as _c:
            # Ensure table exists
            _c.execute("""
                CREATE TABLE IF NOT EXISTS public.erpdata (
                    invno TEXT, store_code TEXT, amt TEXT,
                    invdate TEXT, entry_time TEXT, tillno TEXT, cashier TEXT,
                    _synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Drop NOT NULL on data columns
            _c.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema='public' AND table_name='erpdata'
                  AND is_nullable='NO'
                  AND column_name NOT IN ('_synced_at')
            """)
            for (col,) in _c.fetchall():
                try:
                    _c.execute(
                        f"ALTER TABLE public.erpdata "
                        f"ALTER COLUMN {_quote_ident(col)} DROP NOT NULL"
                    )
                except Exception:
                    pass
            # Disable triggers BEFORE delete — removes per-row trigger overhead
            try:
                _c.execute("ALTER TABLE public.erpdata DISABLE TRIGGER ALL")
                _log("⏸️ `erpdata`: triggers disabled")
            except Exception:
                pass
    finally:
        _ac.close()

    pg_conn = psycopg2.connect(**pg_config)
    try:
        # ── Fast DELETE (now uses index) ──────────────────────────────────────
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM public.erpdata WHERE invdate = %s", (date_str,))
            deleted = cur.rowcount or 0
        pg_conn.commit()
        _log(f"🗑 `erpdata`: deleted {deleted:,} existing rows for {date_str}")
        result['deleted'] = deleted

        # ── COPY streamed directly from file (no memory buffering) ────────────
        col_list_sql = ', '.join(_quote_ident(c) for c in ERPDATA_COPY_COLUMNS)
        copy_sql = (
            f"COPY public.erpdata ({col_list_sql}) "
            f"FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')"
        )
        _log(f"📥 COPY INTO erpdata ({', '.join(ERPDATA_COPY_COLUMNS)})…")

        with pg_conn.cursor() as cur, open(csv_path, 'rb') as _f:
            cur.copy_expert(copy_sql, _f)
            inserted = cur.rowcount if (cur.rowcount or 0) > 0 else 0
        pg_conn.commit()

        result['inserted'] = inserted
        result['fetched']  = inserted
        _log(f"✅ `erpdata`: {inserted:,} rows loaded for {date_str}")

    except Exception as e:
        pg_conn.rollback()
        _log(f"❌ `erpdata` COPY failed: {e}")
        result.update({'status': 'error', 'error': str(e)})
    finally:
        # Always re-enable triggers (autocommit, separate connection)
        try:
            _re = psycopg2.connect(**pg_config)
            _re.autocommit = True
            with _re.cursor() as _c:
                _c.execute("ALTER TABLE public.erpdata ENABLE TRIGGER ALL")
            _re.close()
            _log("▶️ `erpdata`: triggers re-enabled")
        except Exception:
            pass
        pg_conn.close()

    return result



logger = logging.getLogger(__name__)


def _timed_read_sql(query, conn, params=None, label: str | None = None) -> pd.DataFrame:
    """Execute `pd.read_sql` and log elapsed time; returns DataFrame unchanged.

    Use `label` to identify the caller in logs.
    """
    start = time.time()
    df = pd.read_sql(query, conn, params=params)
    elapsed = time.time() - start
    if label:
        logger.info("%s: pd.read_sql completed in %.2fs", label, elapsed)
    else:
        logger.info("pd.read_sql completed in %.2fs", elapsed)
    return df

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
    "Invalid store code",
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


def ensure_erpdata_test_bills_mv_exists() -> None:
    """Create materialized view mv_erpdata_test_bills_daily if missing.

    This view stores per-day, per-shop distinct invoice counts from `erpdata` where amt = 0.01.
    """
    create_sql = """
    CREATE MATERIALIZED VIEW mv_erpdata_test_bills_daily AS
    SELECT
        e.invdate::date AS bill_date,
        UPPER(TRIM(e.store_code)) AS shop_code,
        COUNT(DISTINCT NULLIF(TRIM(e.invno::text), '')) AS test_bills_generated
    FROM erpdata e
    WHERE COALESCE(e.amt, 0) = 0.01
      AND NULLIF(TRIM(COALESCE(e.invno::text, '')), '') IS NOT NULL
    GROUP BY e.invdate::date, UPPER(TRIM(e.store_code))
    WITH DATA;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.mv_erpdata_test_bills_daily')")
            exists = cur.fetchone()[0] is not None  
            if exists:
                return

            logger.warning("Missing mv_erpdata_test_bills_daily. Creating it automatically.")
            cur.execute(create_sql)
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_mv_erpdata_test_bills_daily_bill_shop
                ON mv_erpdata_test_bills_daily (bill_date, shop_code)
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
    consumable_agg AS (
        SELECT
            {group_select.replace('shop_code', 'iu.shop_code')},
            COUNT(*) FILTER (WHERE NULLIF(TRIM(COALESCE(iu.invno, '')), '') IS NOT NULL)::bigint AS consumable_till_nob,
            COALESCE(SUM(iu.amt), 0)::numeric AS consumable_nob_ghs
        FROM (
            SELECT * FROM invoice_union iu
            WHERE iu.src_priority = 1{extra_and_sql_qualified}
        ) iu
        INNER JOIN consumable_tills ct ON ct.shop_code = iu.shop_code AND ct.till_no = iu.till_no
        GROUP BY {group_by.replace('shop_code', 'iu.shop_code')}
    ),
    scan_agg AS (
        -- scan_nob = COUNT(DISTINCT invno) from invoices — total bills scanned, all tills
        SELECT
            {group_select},
            COUNT(DISTINCT NULLIF(TRIM(COALESCE(raw_i.invno, '')), ''))::bigint AS scan_nob,
            COALESCE(SUM(raw_i.amt), 0)                                         AS scanned_nob_ghs,
            COALESCE(MAX(c.consumable_till_nob), 0)                             AS consumable_till_nob,
            COALESCE(MAX(c.consumable_nob_ghs), 0)                              AS consumable_nob_ghs
        FROM (
            SELECT bill_date, shop_code, invno, amt
            FROM (
                SELECT DISTINCT ON (invdate::date, UPPER(TRIM(store_code)), TRIM(COALESCE(invno::text, '')))
                    invdate::date                        AS bill_date,
                    UPPER(TRIM(store_code))              AS shop_code,
                    TRIM(COALESCE(invno::text, ''))      AS invno,
                    COALESCE(amt, 0)::numeric            AS amt
                FROM invoices
                WHERE invdate::date BETWEEN %(s)s AND %(e)s
                  AND NULLIF(TRIM(COALESCE(store_code, '')), '') IS NOT NULL
                  AND NULLIF(TRIM(COALESCE(invno::text, '')),  '') IS NOT NULL
                ORDER BY invdate::date, UPPER(TRIM(store_code)), TRIM(COALESCE(invno::text, ''))
            ) _inv
            WHERE 1=1{extra_and_sql}
        ) raw_i
        FULL OUTER JOIN consumable_agg c USING ({using_clause})
        GROUP BY {group_by}
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
            SUM(alert_count) FILTER (WHERE a_type_normalized = 'Test Bill')                              AS alert_test_bill_scanned,
            SUM(alert_count) FILTER (WHERE a_type_normalized = 'Bill Date Mismatch')                     AS alert_bill_date_mismatched,
            SUM(alert_count) FILTER (WHERE a_type_normalized IN ('Wrong Shop', 'Invalid Store Code'))    AS alert_wrong_shop,
            SUM(alert_count) FILTER (WHERE a_type_normalized = 'High Bill Amount')                       AS alert_high_value_bill
        FROM mv_wh_alerts_daily
        WHERE alert_date BETWEEN %(s)s AND %(e)s
        GROUP BY shop_code
    ),
    dup_distinct_agg AS (
        SELECT
            UPPER(TRIM(a_store_code)) AS shop_code,
            COUNT(DISTINCT NULLIF(TRIM(COALESCE(a_invoice::text, '')), '')) AS alert_duplicate_distinct
        FROM alerts
        WHERE LOWER(TRIM(a_type)) IN ('duplicate', 'duplicate bill')
          AND COALESCE(scanned_date::date, a_entrytime::date) BETWEEN %(s)s AND %(e)s
          AND NULLIF(TRIM(COALESCE(a_store_code, '')), '') IS NOT NULL
        GROUP BY UPPER(TRIM(a_store_code))
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
        (COALESCE(sc.scan_nob, 0) + COALESCE(e.test_bills, 0)) AS total_accounted_bills,
        GREATEST(COALESCE(e.erp_nob, 0) - (COALESCE(sc.scan_nob, 0) + COALESCE(e.test_bills, 0)), 0) AS bills_not_scanned,
        CASE
            WHEN COALESCE(e.erp_nob, 0) > 0
            THEN ROUND((COALESCE(sc.scan_nob, 0)::numeric / e.erp_nob) * 100, 2)
            ELSE 0
        END AS bill_pct,
        COALESCE(e.erp_nob_ghs, 0)          AS erp_erp_nob_ghs,
        COALESCE(sc.scanned_nob_ghs, 0)     AS scanned_nob_ghs,
        COALESCE(sc.consumable_nob_ghs, 0)  AS consumable_nob_ghs,
        (COALESCE(e.erp_nob_ghs, 0) - COALESCE(sc.scanned_nob_ghs, 0)) AS diff_ghs,
        CASE
            WHEN COALESCE(e.erp_nob_ghs, 0) > 0
            THEN ROUND((COALESCE(sc.scanned_nob_ghs, 0) / e.erp_nob_ghs) * 100, 2)
            ELSE 0
        END AS diff_pct,
        COALESCE(a.alert_test_bill_scanned, 0)     AS test_bill_scanned,
        COALESCE(a.alert_bill_date_mismatched, 0)  AS bill_date_mismatched,
        COALESCE(d.alert_duplicate_distinct, 0)    AS duplicate,
        COALESCE(a.alert_wrong_shop, 0)            AS wrong_shop,
        COALESCE(a.alert_high_value_bill, 0)       AS high_value_bill
    FROM all_shops s
    LEFT JOIN erp_agg e              ON e.shop_code  = s.shop_code
    LEFT JOIN scan_agg sc            ON sc.shop_code = s.shop_code
    LEFT JOIN test_not_generated_agg tn ON tn.shop_code = s.shop_code
    LEFT JOIN alerts_agg a           ON a.shop_code  = s.shop_code
    LEFT JOIN dup_distinct_agg d     ON d.shop_code  = s.shop_code
    WHERE s.shop_code IS NOT NULL AND TRIM(s.shop_code) <> ''
      AND s.shop_code NOT IN ('G01','G02','G03','G04','G05','G06','G07','G08','G09','G10','G11','G12','INV','SPX','SEL')
    ORDER BY s.shop_code
    """

    with get_db_connection() as conn:
        return _timed_read_sql(
            query,
            conn,
            params={
                "s": start_date,
                "e": end_date,
                "shops": sorted(list(IMPLEMENTED_SHOPS_SET)),
            },
            label="load_owner_view",
        )


@st.cache_data(ttl=300)
def load_shopwise_test_bill_analysis(start_date: date, end_date: date) -> pd.DataFrame:
    # Ensure MV for fast erpdata test-bill counts exists (created on-demand).
    try:
        ensure_erpdata_test_bills_mv_exists()
    except Exception:
        logger.exception("Failed to ensure mv_erpdata_test_bills_daily exists; proceeding without it.")

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
        -- Prefer pre-aggregated materialized view for performance; falls back to erpdata if MV missing.
        SELECT shop_code, SUM(test_bills_generated)::bigint AS test_bills_generated
        FROM mv_erpdata_test_bills_daily
        WHERE bill_date BETWEEN %(s)s AND %(e)s
          AND shop_code = ANY(%(shops)s)
        GROUP BY shop_code
    ),
    unique_generated_agg AS (
        SELECT
            shop_code,
            COUNT(*) FILTER (WHERE has_test_bill = 1)::bigint AS unique_test_bills_generated
        FROM erp_sessions
        GROUP BY shop_code
    ),
    manager_agg AS (
        -- Count distinct invoice numbers handed over to manager.
        -- For WHL we include erpdata where amt = 0.01 (business override found during validation).
        SELECT shop_code,
               COUNT(DISTINCT NULLIF(TRIM(invno), ''))::bigint AS handover_test_bill_to_manager
        FROM (
            SELECT UPPER(TRIM(COALESCE(m.store_code, ''))) AS shop_code,
                   TRIM(COALESCE(m.invno::text, '')) AS invno
            FROM invoices_manager m
            WHERE m.invdate::date BETWEEN %(s)s AND %(e)s
              AND NULLIF(TRIM(COALESCE(m.invno::text, '')), '') IS NOT NULL
              AND UPPER(TRIM(COALESCE(m.store_code, ''))) = ANY(%(shops)s)

            UNION ALL

            -- WHL override: use erpdata invnos with amt = 0.01
            SELECT UPPER(TRIM(e.store_code)) AS shop_code,
                   TRIM(COALESCE(e.invno::text, '')) AS invno
            FROM erpdata e
            WHERE e.invdate::date BETWEEN %(s)s AND %(e)s
              AND COALESCE(e.amt, 0) = 0.01
              AND UPPER(TRIM(e.store_code)) = 'WHL'
              AND NULLIF(TRIM(COALESCE(e.invno::text, '')), '') IS NOT NULL
        ) t
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
            THEN ROUND((COALESCE(g.test_bills_generated, 0)::numeric / c.cashier_login) * 100, 2)
            ELSE 0
        END AS generated_test_bill_pct,
        COALESCE(m.handover_test_bill_to_manager, 0)::bigint       AS handover_test_bill_to_manager,
        CASE
            WHEN COALESCE(g.test_bills_generated, 0) > 0
            THEN ROUND((COALESCE(m.handover_test_bill_to_manager, 0)::numeric / g.test_bills_generated) * 100, 2)
            ELSE 0
        END AS bill_handover_pct,
        (COALESCE(g.test_bills_generated, 0) - COALESCE(m.handover_test_bill_to_manager, 0))::bigint AS missing_test_bill
    FROM shops s
    LEFT JOIN cashier_agg c  ON c.shop_code = s.shop_code
    LEFT JOIN generated_agg g ON g.shop_code = s.shop_code
    LEFT JOIN unique_generated_agg ug ON ug.shop_code = s.shop_code
    LEFT JOIN manager_agg m  ON m.shop_code = s.shop_code
    ORDER BY s.shop_code
    """

    with get_db_connection() as conn:
        return _timed_read_sql(
            query,
            conn,
            params={
                "s": start_date,
                "e": end_date,
                "shops": sorted(list(IMPLEMENTED_SHOPS_SET)),
            },
            label="load_shopwise_test_bill_analysis",
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
        NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '')::int AS till_no,
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
          -- allow empty or 'ALL' to mean no shop filter
          AND (%(shop)s = '' OR %(shop)s = 'ALL' OR UPPER(TRIM(store_code)) = %(shop)s)
          AND NULLIF(TRIM(COALESCE(invno::text, '')), '') IS NOT NULL
    ) m
        ON TRIM(COALESCE(e.invno::text, '')) = m.invno
    WHERE e.invdate::date BETWEEN %(s)s AND %(e)s
      AND (%(shop)s = '' OR %(shop)s = 'ALL' OR UPPER(TRIM(e.store_code)) = %(shop)s)
      AND NULLIF(TRIM(COALESCE(e.invno::text, '')), '') IS NOT NULL
      AND COALESCE(e.amt, 0) = 0.01
    ORDER BY e.invdate DESC, e.invno DESC
    """

    with get_db_connection() as conn:
        return _timed_read_sql(
            query,
            conn,
            params={"s": start_date, "e": end_date, "shop": (shop_code or "").strip().upper()},
            label="load_shopwise_test_bill_drilldown",
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

    # If user requested 'ALL', keep per-row shop_code and map each to a shop name.
    if (selected_shop or '').strip().upper() == 'ALL':
        drilldown_df["shop_name"] = drilldown_df["shop_code"].map(shop_name_map).fillna(drilldown_df["shop_code"])
        shop_name = "ALL SHOPS"
    else:
        shop_name = shop_name_map.get(selected_shop, selected_shop)
        drilldown_df["shop_name"] = shop_name
    detail_display = drilldown_df.rename(
        columns={
            "shop_name": "Shop Name",
            "cashier_name": "Cashier name",
            "till_no": "Till No",
            "invdate": "InvDate",
            "invno": "InvNo",
            "amt": "Amt",
            "handed_over_to_manager": "Handed Over",
        }
    )[["Shop Name", "Cashier name", "Till No", "InvDate", "InvNo", "Amt", "Handed Over"]]

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
        till_no_val = drow.get("Till No", "")
        row_vals = [
            str(drow["Shop Name"]),
            str(drow["Cashier name"]),
            str(till_no_val) if pd.notna(till_no_val) and till_no_val != '' else "",
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
                _sort_token(till_no_val),
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
            "test_bills_generated": "Total Test Bill",
            "unique_test_bills_generated": "Cashier Generated Test Bill",
            "not_generated_bill_pct": "Not Generated Bill %",
            "generated_test_bill_pct": "Generated Test Bill %",
            "handover_test_bill_to_manager": "Unique bill handover to manager",
            "bill_handover_pct": "Bill handover %",
            "missing_test_bill": "Missing test bill",
        }
    )

    raw_test_generated_lookup = pd.to_numeric(
        display_df.get("Cashier Generated Test Bills (Raw)", pd.Series(index=display_df.index)),
        errors="coerce"
    ).fillna(0)

    unique_test_generated_lookup = pd.to_numeric(
        display_df.get("Cashier Generated Test Bill", pd.Series(index=display_df.index)),
        errors="coerce"
    ).fillna(0)

    cashier_login_series = pd.to_numeric(display_df.get("Cashier Login"), errors="coerce").fillna(0)

    display_df["Cashier Generated Test Bill%"] = (
        (unique_test_generated_lookup / cashier_login_series.replace(0, float("nan"))) * 100
    ).round(2).fillna(0)

    ordered_cols = [
        "Shop Code",
        "Shop Name",
        "Cashier Login",
        "Cashier Generated Test Bill",
        "Cashier Generated Test Bill%",
        "Not Generated Bill %",
        "Cashier Not Generated Test Bills",
        "Total Test Bill",
        "Unique bill handover to manager",
        "Missing test bill",
        "Bill handover %",
    ]
    display_df = display_df[[c for c in ordered_cols if c in display_df.columns]]

    headers = list(display_df.columns)
    rows: list[list[str]] = []
    sort_rows: list[list[str]] = []
    cell_titles: dict[tuple[int, int], str] = {}
    cell_styles: dict[tuple[int, int], str] = {}
    pct_cols = {"Not Generated Bill %", "Bill handover %", "Cashier Generated Test Bill%"}

    def _pct_style(col_name: str, pct_value: float) -> str:
        # Match One-View Summary Bill % color scheme exactly.
        style_val = pct_value
        if col_name in {"Bill handover %", "Cashier Generated Test Bill%"} and pct_value > 100:
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
        test_generated = float(row["Total Test Bill"])
        unique_test_generated = float(unique_test_generated_lookup.loc[row.name])
        handover_to_manager = float(row["Unique bill handover to manager"])
        generated = float(unique_test_generated_lookup.loc[row.name])
        handover = float(row["Unique bill handover to manager"])
        cashier_gen_pct = float(row["Cashier Generated Test Bill%"])

        bill_handover_pct = (handover / test_generated * 100) if test_generated > 0 else 0
        bill_handover_pct = min(100, bill_handover_pct)
        missing_test_bill = max(0, float(row["Total Test Bill"]) - float(row["Unique bill handover to manager"]))
        not_gen_pct = float(row["Not Generated Bill %"])

        tooltip_map = {
            "Cashier Login": (
                "Count of cashier+till sessions from erpdata for selected period."
            ),
            "Cashier Generated Test Bill": (
                f"Count of unique cashier+till combinations (from erpdata) that generated at least one test bill (amt = 0.01) = {unique_test_generated:,.0f}."
            ),
            "Cashier Generated Test Bill%": (
                f"(Cashier Generated Test Bill ({unique_test_generated:,.0f}) / Cashier Login ({cashier_login:,.0f})) x 100 = {cashier_gen_pct:.2f}%"
                if cashier_login > 0 else "Cashier Login is 0, so Cashier Generated Test Bill% = 0"
            ),
            "Cashier Not Generated Test Bills": (
                "Count of cashier+till sessions (from erpdata) that do not have any test bill (amt = 0.01)."
            ),
            "Total Test Bill": "Count of invoices (erpdata.invno) where amt = 0.01 (each invoice counted once).",
            "Not Generated Bill %": (
                f"(Cashier Not Generated Test Bills ({test_not_generated:,.0f}) / Cashier Login ({cashier_login:,.0f})) x 100 = {not_gen_pct:.2f}%"
                if cashier_login > 0 else "Cashier Login is 0, so Not Generated Bill % = 0"
            ),
            "Unique bill handover to manager": (
                f"Count of distinct invoice numbers in `invoices_manager` with amt = 0.01 during the selected date range = {handover_to_manager:,.0f}."
            ),
            "Bill handover %": (
                f"(Unique bill handover to manager ({handover_to_manager:,.0f}) / Total Test Bills ({test_generated:,.0f})) x 100 = {bill_handover_pct:.2f}%"
                if test_generated > 0 else "Total Test Bills is 0, so Bill handover % = 0"
            ),
            "Missing test bill": (
                f"Total Test Bill ({test_generated:,.0f}) - Unique bill handover to manager ({handover_to_manager:,.0f}) = {missing_test_bill:,.0f}"
            )
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
        "Cashier Generated Test Bill": float(unique_test_generated_lookup.sum()),
        "Cashier Not Generated Test Bills": float(pd.to_numeric(display_df["Cashier Not Generated Test Bills"], errors="coerce").fillna(0).sum()),
        "Total Test Bill": float(pd.to_numeric(display_df["Total Test Bill"], errors="coerce").fillna(0).sum()),
        "Unique bill handover to manager": float(pd.to_numeric(display_df["Unique bill handover to manager"], errors="coerce").fillna(0).sum()),
        "total_generated": float(unique_test_generated_lookup.sum()),
        "total_handover": float(pd.to_numeric(display_df["Unique bill handover to manager"], errors="coerce").fillna(0).sum()),
    }
    totals["Missing test bill"] = max(0, totals["total_generated"] - totals["total_handover"])
    totals["Not Generated Bill %"] = (
        (totals["Cashier Not Generated Test Bills"] / totals["Cashier Login"]) * 100 if totals["Cashier Login"] > 0 else 0
    )
    totals["Cashier Generated Test Bill%"] = (
        (totals["Cashier Generated Test Bill"] / totals["Cashier Login"]) * 100 if totals["Cashier Login"] > 0 else 0
    )
    totals["Bill handover %"] = (
        (totals["Unique bill handover to manager"] / totals["Total Test Bill"]) * 100 if totals.get("Total Test Bill", 0) > 0 else 0
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
    # Add an explicit 'ALL' option so users can view drilldown across all shops
    dropdown_shop_options = ["", "ALL"] + sorted(display_df["Shop Code"].dropna().astype(str).unique().tolist())
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
        "Invalid store code",
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
        "Invalid store code": st.column_config.NumberColumn("Invalid store code", help="Alert count where `a_type = 'Wrong Sho' or 'Wrong Shop'`"),
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
            "Cashier Generated Test Bills": "Count of invoices in `erpdata.invno` where amt = 0.01 (each invoice counted once). Uses a materialized view for performance when available.",
        "Total Accounted Bills": "Scan NOB + Consumable Till NOB + ERP Test Bills",
        "Bills not scanned": "ERP NOB - Total Accounted Bills",
        "Scanned bill %": "(Scan NOB / ERP NOB) × 100",
        "ERP ERP NOB (GHS)": "Sum of erpdata.amt",
        "Scanned NOB (GHS)": "Sum of scanned invoice amount excluding consumable tills",
        "Consumable NOB (GHS)": "Sum of scanned invoice amount from consumable tills",
        "Diff (GHS)": "ERP ERP NOB (GHS) - [Scanned NOB (GHS) + Consumable NOB (GHS)]",
        "Scanned GHS %": "(Scanned NOB (GHS) / ERP ERP NOB (GHS)) × 100",
        "Test Bill Scanned": "Alert count where a_type = 'test bill'",
        "Bill Date Mismatched": "Alert count where a_type indicates bill date mismatch",
        "Duplicate": "Alert count where a_type is Duplicate/Duplicate Bill",
        "Invalid store code": "Alert count where a_type is Wrong Shop/Wrong Sho",
        "High Value Bill": "Alert count where a_type = 'High Bill Amount'",
        "Unique bill handover to manager": "Count of distinct invoice numbers handed to managers. For shop WHL the count includes `erpdata` invoices with amt = 0.01 (override) to match DB validation; other shops use `invoices_manager`.",
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
                # If caller didn't supply an Excel export, build one in-memory
                if not excel_base64:
                    try:
                        # Build DataFrame from rows for Excel export
                        df_export = pd.DataFrame(rows, columns=headers)
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                            df_export.to_excel(writer, index=False, sheet_name="Sheet1")
                            wb = writer.book
                            ws = writer.sheets["Sheet1"]

                            # Apply basic formatting derived from `cell_styles`
                            def _style_to_fill_font(style_txt: str):
                                # Map a few known background color cues to fills/fonts
                                if not style_txt:
                                    return None, None
                                s = style_txt.lower()
                                if "rgba(248, 113, 113" in s or "#fecaca" in s or "red" in s:
                                    fill = PatternFill(start_color="FFF87171", end_color="FFF87171", fill_type="solid")
                                    font = Font(color="FF000000")
                                    return fill, font
                                if "rgba(74, 222, 128" in s or "#dcfce7" in s or "green" in s:
                                    fill = PatternFill(start_color="FF4ADE80", end_color="FF4ADE80", fill_type="solid")
                                    font = Font(color="FF000000")
                                    return fill, font
                                if "rgba(250, 204, 21" in s or "#fef9c3" in s or "yellow" in s:
                                    fill = PatternFill(start_color="FFFACC15", end_color="FFFACC15", fill_type="solid")
                                    font = Font(color="FF000000")
                                    return fill, font
                                if "rgba(125, 211, 252" in s or "#e0f2fe" in s or "blue" in s:
                                    fill = PatternFill(start_color="FF7DD3FC", end_color="FF7DD3FC", fill_type="solid")
                                    font = Font(color="FF000000")
                                    return fill, font
                                # Fallback: no formatting
                                return None, None

                            # cell_styles keys are (row_idx, col_idx) where row_idx starts at 0
                            for (r_idx, c_idx), style_txt in (cell_styles or {}).items():
                                try:
                                    # Excel rows: header at 1, data starts at 2
                                    excel_row = 2 + int(r_idx)
                                    excel_col = 1 + int(c_idx)
                                    cell = ws.cell(row=excel_row, column=excel_col)
                                    fill, font = _style_to_fill_font(style_txt)
                                    if fill:
                                        cell.fill = fill
                                    if font:
                                        cell.font = font
                                except Exception:
                                    # Non-fatal: continue applying others
                                    pass

                        excel_base64 = base64.b64encode(excel_buffer.getvalue()).decode("ascii")
                    except Exception:
                        excel_base64 = None

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
        "Wrong Shop": "Invalid store code",
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
        "Invalid store code",
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
    # Query erpdata directly so the graph grows as daily CSVs accumulate.
    # Dates with no erpdata get NULL diff_ghs → rendered as gaps, not negative bars.
    query = f"""
    WITH calendar_days AS (
        SELECT generate_series(%(s)s::date, %(e)s::date, interval '1 day')::date AS bill_date
    ),
    erp_agg AS (
        SELECT
            e.invdate::date                                                                    AS bill_date,
            SUM(COALESCE(e.amt, 0)::numeric) FILTER (WHERE COALESCE(e.amt, 0) != 0.01)       AS erp_ghs
        FROM erpdata e
        WHERE e.invdate::date BETWEEN %(s)s AND %(e)s
          AND UPPER(TRIM(COALESCE(e.store_code, ''))) != ''
          AND UPPER(TRIM(COALESCE(e.store_code, ''))) NOT IN ('G01','G02','G03','G04','G05','G06','G07','G08','G09','G10','G11','G12','INV','SPX','SEL')
        GROUP BY e.invdate::date
    ),
    {_build_live_scan_agg_cte(['bill_date'], "WHERE shop_code IS NOT NULL AND TRIM(shop_code) <> '' AND shop_code NOT IN ('G01','G02','G03','G04','G05','G06','G07','G08','G09','G10','G11','G12','INV','SPX','SEL')").rstrip().rstrip(',')}
    SELECT
        d.bill_date,
        e.erp_ghs,
        COALESCE(s.scanned_nob_ghs, 0)                                               AS scanned_ghs,
        COALESCE(s.consumable_nob_ghs, 0)                                            AS consumable_ghs,
        CASE
            WHEN e.erp_ghs IS NOT NULL
            THEN e.erp_ghs - (COALESCE(s.scanned_nob_ghs, 0) + COALESCE(s.consumable_nob_ghs, 0))
            ELSE NULL
        END                                                                           AS diff_ghs
    FROM calendar_days d
    LEFT JOIN erp_agg  e ON e.bill_date = d.bill_date
    LEFT JOIN scan_agg s ON s.bill_date = d.bill_date
    ORDER BY d.bill_date
    """

    with get_db_connection() as conn:
        return _timed_read_sql(
            query,
            conn,
            params={
                "s": start_date,
                "e": end_date,
                "shops": sorted(list(IMPLEMENTED_SHOPS_SET)),
            },
            label="load_daily_diff_series",
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
    trend_df["diff_ghs"] = pd.to_numeric(trend_df["diff_ghs"], errors="coerce")  # keep NaN for dates with no ERP data

    # Month-wise: only include days that have ERP data (non-NaN diff_ghs)
    month_df = (
        trend_df[(trend_df["bill_date"] >= month_window_start) & (trend_df["bill_date"] <= anchor_end_date)]
        .dropna(subset=["diff_ghs"])
        .copy()
    )
    month_df["month"] = pd.to_datetime(month_df["bill_date"]).dt.to_period("M").astype(str)
    month_df = (
        month_df.groupby("month", as_index=False)["diff_ghs"]
        .sum()
        .sort_values("month")
    )
    month_df["month_label"] = pd.to_datetime(month_df["month"] + "-01").dt.strftime("%b %Y")

    # Last-10 days: only days with ERP data
    last_10_df = (
        trend_df[(trend_df["bill_date"] >= last_10_start) & (trend_df["bill_date"] <= anchor_end_date)]
        .dropna(subset=["diff_ghs"])
        .copy()
    )
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
    """
    Bill Handover % per day across all shops for the last 10 days.
    Formula:
      handover_pct = COUNT(DISTINCT invno in invoices_manager)
                     / COUNT(DISTINCT invno in erpdata where amt = 0.01)  × 100
    """
    start_10 = end_date - timedelta(days=9)
    query = """
    WITH days AS (
        SELECT generate_series(%(s)s::date, %(e)s::date, '1 day'::interval)::date AS bill_date
    ),
    total_test_bills AS (
        -- Denominator: distinct invno in erpdata where amt = 0.01 (test bills)
        SELECT
            e.invdate::date                                                             AS bill_date,
            COUNT(DISTINCT NULLIF(TRIM(COALESCE(e.invno::text, '')), ''))::bigint       AS total_generated
        FROM erpdata e
        WHERE e.invdate::date BETWEEN %(s)s AND %(e)s
          AND UPPER(TRIM(e.store_code)) = ANY(%(shops)s)
          AND COALESCE(e.amt, 0) = 0.01
          AND NULLIF(TRIM(COALESCE(e.invno::text, '')), '') IS NOT NULL
        GROUP BY e.invdate::date
    ),
    mgr_agg AS (
        -- Numerator: distinct invno handed to manager (invoices_manager + WHL erpdata override)
        SELECT bill_date,
               COUNT(DISTINCT NULLIF(TRIM(invno), ''))::bigint AS handover_count
        FROM (
            SELECT m.invdate::date                              AS bill_date,
                   TRIM(COALESCE(m.invno::text, ''))           AS invno
            FROM invoices_manager m
            WHERE m.invdate::date BETWEEN %(s)s AND %(e)s
              AND UPPER(TRIM(COALESCE(m.store_code, ''))) = ANY(%(shops)s)
              AND NULLIF(TRIM(COALESCE(m.invno::text, '')), '') IS NOT NULL

            UNION ALL

            -- WHL override: test-bill invnos from erpdata count as handover
            SELECT e.invdate::date                              AS bill_date,
                   TRIM(COALESCE(e.invno::text, ''))           AS invno
            FROM erpdata e
            WHERE e.invdate::date BETWEEN %(s)s AND %(e)s
              AND UPPER(TRIM(e.store_code)) = 'WHL'
              AND COALESCE(e.amt, 0) = 0.01
              AND NULLIF(TRIM(COALESCE(e.invno::text, '')), '') IS NOT NULL
        ) t
        GROUP BY bill_date
    )
    SELECT
        d.bill_date,
        COALESCE(ttb.total_generated, 0)  AS total_generated,
        COALESCE(ma.handover_count,   0)  AS total_handover,
        CASE
            WHEN COALESCE(ttb.total_generated, 0) > 0
            THEN ROUND(
                    COALESCE(ma.handover_count, 0)::numeric
                    / ttb.total_generated * 100,
                 1)
            ELSE 0
        END                               AS handover_pct
    FROM days d
    LEFT JOIN total_test_bills ttb ON ttb.bill_date = d.bill_date
    LEFT JOIN mgr_agg          ma  ON ma.bill_date  = d.bill_date
    ORDER BY d.bill_date
    """
    with get_db_connection() as conn:
        return _timed_read_sql(
            query,
            conn,
            params={
                "s": start_10,
                "e": end_date,
                "shops": sorted(list(IMPLEMENTED_SHOPS_SET)),
            },
            label="load_bill_handover_last10",
        )


@st.cache_data(ttl=300)
def load_alert_type_comparison(end_date: date) -> pd.DataFrame:
    selected_date = end_date
    yesterday_date = end_date - timedelta(days=1)
    month_start = end_date.replace(day=1)

    # Query alerts table directly (not the MV) so stale/unrefreshed MVs never
    # cause zeroes for historical dates.  Inline normalization mirrors mv_wh_alerts_daily.
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
    norm AS (
        SELECT
            UPPER(TRIM(a_store_code)) AS shop_code,
            CASE
                WHEN LOWER(TRIM(a_type)) IN ('bill date mismatched', 'bill date mismatch') THEN 'Bill Date Mismatch'
                WHEN LOWER(TRIM(a_type)) = 'test bill'                                     THEN 'Test Bill'
                WHEN LOWER(TRIM(a_type)) IN ('duplicate', 'duplicate bill')                THEN 'Duplicate'
                WHEN LOWER(TRIM(a_type)) IN ('wrong shop', 'wrong sho', 'invalid store code') THEN 'Wrong Shop'
                WHEN LOWER(TRIM(a_type)) = 'high bill amount'                              THEN 'High Bill Amount'
                ELSE NULL
            END AS a_type_normalized,
            COALESCE(scanned_date::date, a_entrytime::date) AS alert_date,
            NULLIF(TRIM(COALESCE(a_invoice::text, '')), '') AS invno
        FROM alerts
        WHERE NULLIF(TRIM(COALESCE(a_store_code, '')), '') IS NOT NULL
          AND COALESCE(scanned_date::date, a_entrytime::date) BETWEEN %(month_start)s AND %(selected_date)s
          AND UPPER(TRIM(a_store_code)) NOT IN ('G01','G02','G03','G04','G05','G06','G07','G08','G09','G10','G11','G12','INV','SPX','SEL')
    ),
    counts AS (
        SELECT
            a_type_normalized AS a_type,
            CASE
                WHEN a_type_normalized = 'Duplicate'
                THEN COALESCE(NULLIF(COUNT(DISTINCT invno) FILTER (WHERE alert_date = %(selected_date)s), 0),
                              COUNT(*) FILTER (WHERE alert_date = %(selected_date)s))
                ELSE COUNT(*) FILTER (WHERE alert_date = %(selected_date)s)
            END::bigint AS selected_date_total,
            CASE
                WHEN a_type_normalized = 'Duplicate'
                THEN COALESCE(NULLIF(COUNT(DISTINCT invno) FILTER (WHERE alert_date = %(yesterday_date)s), 0),
                              COUNT(*) FILTER (WHERE alert_date = %(yesterday_date)s))
                ELSE COUNT(*) FILTER (WHERE alert_date = %(yesterday_date)s)
            END::bigint AS yesterday_total,
            CASE
                WHEN a_type_normalized = 'Duplicate'
                THEN COALESCE(NULLIF(COUNT(DISTINCT invno), 0), COUNT(*))
                ELSE COUNT(*)
            END::bigint AS month_to_date_total
        FROM norm
        WHERE a_type_normalized IS NOT NULL
        GROUP BY a_type_normalized
    )
    SELECT
        CASE WHEN t.a_type = 'Wrong Shop' THEN 'Invalid store code' ELSE t.a_type END AS a_type,
        COALESCE(c.selected_date_total, 0)::bigint  AS selected_date_total,
        COALESCE(c.yesterday_total, 0)::bigint       AS yesterday_total,
        COALESCE(c.month_to_date_total, 0)::bigint   AS month_to_date_total
    FROM alert_types t
    LEFT JOIN counts c ON c.a_type = t.a_type
    ORDER BY t.sort_order
    """

    with get_db_connection() as conn:
        return _timed_read_sql(
            query,
            conn,
            params={
                "month_start": month_start,
                "selected_date": selected_date,
                "yesterday_date": yesterday_date,
            },
            label="load_alert_type_comparison",
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
                WHEN LOWER(TRIM(a_type)) IN ('wrong shop', 'wrong sho', 'invalid store code') THEN 'Invalid store code'
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
        return _timed_read_sql(
            query,
            conn,
            params={
                "selected_error_type": selected_error_type,
                "month_start": month_start,
                "end_date": end_date,
                "shops": sorted(list(IMPLEMENTED_SHOPS_SET)),
            },
            label="load_alert_type_drilldown",
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
        # Exclude days with no ERP sessions (erpdata not loaded for those dates)
        ho_df = ho_df[ho_df["total_generated"] > 0]

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


# ══════════════════════════════════════════════════════════════════════════════
# EDA ANALYTICS — INVOICE SCANNING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def eda_inv_14day_nob_trend(end_date: date) -> pd.DataFrame:
    """14-day daily ERP NOB, Scan NOB and Scan% using mv_wh_erp_daily."""
    start = end_date - timedelta(days=13)
    _shops = sorted(list(IMPLEMENTED_SHOPS_SET))
    query = """
        WITH erp AS (
            SELECT bill_date, SUM(erp_nob)::bigint AS erp_nob
            FROM mv_wh_erp_daily
            WHERE bill_date BETWEEN %(start)s AND %(end)s
              AND shop_code = ANY(%(shops)s)
            GROUP BY 1
        ),
        scanned AS (
            SELECT bill_date, SUM(scan_nob)::bigint AS scan_nob
            FROM mv_wh_invoices_agg_daily
            WHERE bill_date BETWEEN %(start)s AND %(end)s
              AND shop_code = ANY(%(shops)s)
            GROUP BY 1
        )
        SELECT e.bill_date AS dt,
               e.erp_nob,
               COALESCE(s.scan_nob, 0) AS scan_nob,
               ROUND(COALESCE(s.scan_nob,0)::numeric / NULLIF(e.erp_nob,0) * 100, 1) AS scan_pct
        FROM erp e LEFT JOIN scanned s ON s.bill_date = e.bill_date
        ORDER BY 1
    """
    with get_db_connection() as conn:
        try:
            return _timed_read_sql(query, conn,
                                   params={"start": start, "end": end_date, "shops": _shops},
                                   label="eda_14day_nob_trend")
        except Exception:
            return pd.DataFrame()


@st.cache_data(ttl=300)
def eda_inv_alert_by_shop(end_date: date) -> pd.DataFrame:
    """Shop-wise alert counts for current month-to-date."""
    mtd_start = end_date.replace(day=1)
    _shops = sorted(list(IMPLEMENTED_SHOPS_SET))
    query = """
        SELECT UPPER(TRIM(a_store_code)) AS shop_code,
               COUNT(*) AS total_alerts,
               SUM(CASE WHEN LOWER(TRIM(a_type)) LIKE '%%test%%' THEN 1 ELSE 0 END)      AS test_bill,
               SUM(CASE WHEN LOWER(TRIM(a_type)) LIKE '%%duplicate%%' THEN 1 ELSE 0 END) AS duplicate,
               SUM(CASE WHEN LOWER(TRIM(a_type)) LIKE '%%date%%'
                         OR  LOWER(TRIM(a_type)) LIKE '%%mismatch%%'
                         THEN 1 ELSE 0 END)                                               AS date_mismatch,
               SUM(CASE WHEN LOWER(TRIM(a_type)) LIKE '%%wrong%%'
                         OR  LOWER(TRIM(a_type)) LIKE '%%invalid%%'
                         THEN 1 ELSE 0 END)                                               AS wrong_shop,
               SUM(CASE WHEN LOWER(TRIM(a_type)) LIKE '%%high%%' THEN 1 ELSE 0 END)      AS high_value
        FROM alerts
        WHERE UPPER(TRIM(a_store_code)) = ANY(%(shops)s)
          AND COALESCE(scanned_date::date, a_entrytime::date)
              BETWEEN %(mtd_start)s AND %(end)s
        GROUP BY 1
        ORDER BY total_alerts DESC
    """
    with get_db_connection() as conn:
        try:
            return _timed_read_sql(query, conn,
                                   params={"mtd_start": mtd_start, "end": end_date, "shops": _shops},
                                   label="eda_alert_by_shop")
        except Exception:
            return pd.DataFrame()


def render_inv_eda_analytics(data: pd.DataFrame, start_date: date, end_date: date):
    """Premium infographic EDA — Invoice Scanning."""

    if data.empty:
        st.info("No data for selected date range.")
        return

    # ── Pre-compute all metrics ────────────────────────────────────────────────
    shop_nm = load_shop_name_map()
    df = data.copy()
    df["shop_name"] = df["shop_code"].map(shop_nm).fillna(df["shop_code"])
    for col in ["erp_nob","scan_nob","consumable_till_nob","total_accounted_bills",
                "bills_not_scanned","bill_pct","diff_ghs","diff_pct","erp_erp_nob_ghs",
                "scanned_nob_ghs","consumable_nob_ghs",
                "test_bill_scanned","bill_date_mismatched","duplicate","wrong_shop","high_value_bill"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["total_alerts"] = df[["test_bill_scanned","bill_date_mismatched","duplicate",
                              "wrong_shop","high_value_bill"]].sum(axis=1)

    total_erp     = int(df["erp_nob"].sum())
    total_acc     = int(df["total_accounted_bills"].sum())
    total_not     = int(df["bills_not_scanned"].sum())
    overall_pct   = round(total_acc / total_erp * 100, 1) if total_erp else 0.0
    total_diff    = float(df["diff_ghs"].sum())
    total_alerts  = int(df["total_alerts"].sum())
    shops_green   = int((df["bill_pct"] >= 90).sum())
    shops_yellow  = int(((df["bill_pct"] >= 70) & (df["bill_pct"] < 90)).sum())
    shops_red     = int((df["bill_pct"] < 70).sum())
    over_exposed  = int((df["diff_ghs"] > 0).sum())
    shops_w_alerts= int((df["total_alerts"] > 0).sum())

    def _pcc(v: float) -> str:
        return "#22c55e" if v >= 90 else ("#f59e0b" if v >= 70 else "#ef4444")

    if   overall_pct >= 95: _hl, _hc = "EXCELLENT", "#22c55e"
    elif overall_pct >= 90: _hl, _hc = "GOOD",      "#22c55e"
    elif overall_pct >= 80: _hl, _hc = "FAIR",      "#f59e0b"
    elif overall_pct >= 70: _hl, _hc = "WARNING",   "#f97316"
    else:                   _hl, _hc = "CRITICAL",  "#ef4444"

    # SVG ring gauge
    _r = 50; _circ = 2 * 3.14159 * _r
    _filled = overall_pct / 100 * _circ
    _gauge = (
        f'<svg width="160" height="160" viewBox="0 0 120 120">'
        f'<circle cx="60" cy="60" r="{_r}" fill="none" stroke="#1e2a3a" stroke-width="10"/>'
        f'<circle cx="60" cy="60" r="{_r}" fill="none" stroke="{_hc}" stroke-width="10"'
        f' stroke-dasharray="{_filled:.1f} {_circ:.1f}" stroke-linecap="round"'
        f' transform="rotate(-90 60 60)"/>'
        f'<text x="60" y="57" text-anchor="middle" fill="{_hc}" font-size="19"'
        f' font-weight="900" font-family="Arial">{overall_pct:.1f}%</text>'
        f'<text x="60" y="72" text-anchor="middle" fill="#64748b" font-size="7.5"'
        f' font-family="Arial">Scan Compliance</text>'
        f'</svg>'
    )

    # Sparkline SVG (14-day trend)
    _tdf = eda_inv_14day_nob_trend(end_date)
    _tvals = []
    _tsignal = "→ Stable"
    if not _tdf.empty:
        _tdf["scan_pct"] = pd.to_numeric(_tdf["scan_pct"], errors="coerce").fillna(0)
        _tvals = _tdf["scan_pct"].tolist()
        if len(_tvals) >= 2:
            _tsignal = "↗ Improving" if _tvals[-1] > _tvals[0] else ("↘ Declining" if _tvals[-1] < _tvals[0] else "→ Stable")

    def _sparkline_svg(vals, color="#38bdf8", w=200, h=36):
        if len(vals) < 2:
            return '<span style="color:#475569;font-size:11px;">No trend data</span>'
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx != mn else 1
        pts = []
        for i, v in enumerate(vals):
            x = i / (len(vals) - 1) * w
            y = h - ((v - mn) / rng * h * 0.78 + h * 0.11)
            pts.append(f"{x:.1f},{y:.1f}")
        path_d = "M " + " L ".join(pts)
        lx, ly = float(pts[-1].split(",")[0]), float(pts[-1].split(",")[1])
        lclr = _pcc(vals[-1])
        return (
            f'<svg width="{w}" height="{h}" style="display:block;overflow:visible;">'
            f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>'
            f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="4" fill="{lclr}" stroke="#0a1628" stroke-width="1.5"/>'
            f'</svg>'
        )

    _spark_svg = _sparkline_svg(_tvals)
    _tsignal_clr = "#22c55e" if "↗" in _tsignal else ("#ef4444" if "↘" in _tsignal else "#94a3b8")

    # Test bill data
    _tb = load_shopwise_test_bill_analysis(start_date, end_date)
    _tb_map: dict = {}
    _tb_total_login = _tb_total_hand = 0
    if _tb is not None and not _tb.empty:
        _tb = _tb.copy()
        _tb["bill_handover_pct"] = pd.to_numeric(_tb["bill_handover_pct"], errors="coerce").fillna(0)
        _tb["cashier_login"]     = pd.to_numeric(_tb["cashier_login"], errors="coerce").fillna(0)
        _tb["handover_test_bill_to_manager"] = pd.to_numeric(_tb["handover_test_bill_to_manager"], errors="coerce").fillna(0)
        _tb_map           = dict(zip(_tb["shop_code"].str.upper(), _tb["bill_handover_pct"]))
        _tb_total_login   = int(_tb["cashier_login"].sum())
        _tb_total_hand    = int(_tb["handover_test_bill_to_manager"].sum())
    _tb_overall_pct = round(_tb_total_hand / max(_tb_total_login, 1) * 100, 1)
    _tb_clr = _pcc(_tb_overall_pct)

    # Alert comparison data
    _al_cmp = load_alert_type_comparison(end_date)
    _al_by_type: list = []
    _al_icons = {
        "Test Bill":          ("🧪", "#3b82f6"),
        "Bill Date Mismatch": ("📅", "#f59e0b"),
        "Duplicate":          ("🔄", "#ef4444"),
        "Invalid store code": ("🏪", "#8b5cf6"),
        "High Bill Amount":   ("💸", "#22c55e"),
    }
    if not _al_cmp.empty:
        for _, _row in _al_cmp.iterrows():
            _al_by_type.append({
                "name":  str(_row["a_type"]),
                "today": int(_row.get("selected_date_total", 0) or 0),
                "mtd":   int(_row.get("month_to_date_total", 0) or 0),
            })

    # Financial pre-compute
    _diff_clr = "#22c55e" if total_diff <= 0 else "#ef4444"
    _diff_lbl = "Surplus" if total_diff <= 0 else "Financial Exposure"

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🎯 Command Center", "🏪 Shop Intelligence Matrix", "💰 Financial & Alert Pulse"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Command Center
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        # Row 1: Gauge + 3 intelligence cards
        g_col, k_col = st.columns([0.24, 0.76])

        with g_col:
            st.markdown(
                f'<div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:12px;'
                f'padding:20px 14px;text-align:center;">'
                f'{_gauge}'
                f'<div style="font-size:14px;font-weight:900;color:{_hc};letter-spacing:2px;margin-top:6px;">{_hl}</div>'
                f'<div style="font-size:9px;color:#475569;margin-top:3px;">Scan Health Status</div>'
                f'<div style="display:flex;justify-content:space-around;margin-top:14px;">'
                f'<div style="text-align:center;">'
                f'<div style="width:8px;height:8px;background:#22c55e;border-radius:50%;margin:0 auto 3px;"></div>'
                f'<div style="font-size:8px;color:#64748b;">On Track</div>'
                f'<div style="font-size:14px;font-weight:900;color:#22c55e;">{shops_green}</div>'
                f'</div>'
                f'<div style="text-align:center;">'
                f'<div style="width:8px;height:8px;background:#f59e0b;border-radius:50%;margin:0 auto 3px;"></div>'
                f'<div style="font-size:8px;color:#64748b;">Warning</div>'
                f'<div style="font-size:14px;font-weight:900;color:#f59e0b;">{shops_yellow}</div>'
                f'</div>'
                f'<div style="text-align:center;">'
                f'<div style="width:8px;height:8px;background:#ef4444;border-radius:50%;margin:0 auto 3px;"></div>'
                f'<div style="font-size:8px;color:#64748b;">Critical</div>'
                f'<div style="font-size:14px;font-weight:900;color:#ef4444;">{shops_red}</div>'
                f'</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with k_col:
            kc1, kc2, kc3 = st.columns(3)
            for _col, (_icon, _lbl, _val, _sub, _clr) in zip([kc1, kc2, kc3], [
                ("📦", "ERP Billings", f"{total_erp:,}", f"{total_not:,} not scanned", "#38bdf8"),
                ("💰", _diff_lbl, f"GHS {abs(total_diff):,.0f}", f"{over_exposed} shops exposed", _diff_clr),
                ("📋", "Test Bill Handover", f"{_tb_overall_pct:.1f}%", f"{_tb_total_hand:,} of {_tb_total_login:,} cashiers", _tb_clr),
            ]):
                with _col:
                    st.markdown(
                        f'<div style="background:#0a1628;border:1px solid #1e3a5f;border-left:5px solid {_clr};'
                        f'border-radius:12px;padding:14px 12px;height:100px;">'
                        f'<div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;">{_icon} {_lbl}</div>'
                        f'<div style="font-size:22px;font-weight:900;color:{_clr};margin:6px 0 3px;">{_val}</div>'
                        f'<div style="font-size:9px;color:#64748b;">{_sub}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Sparkline strip
            st.markdown(
                f'<div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;'
                f'padding:10px 14px;margin-top:8px;display:flex;align-items:center;gap:14px;">'
                f'<div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:1px;white-space:nowrap;">📈 14-Day Scan Trend</div>'
                f'<div style="flex:1;">{_spark_svg}</div>'
                f'<div style="font-size:11px;font-weight:700;color:{_tsignal_clr};white-space:nowrap;">{_tsignal}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        # Row 2: Shops at risk + Alert anatomy
        r1, r2 = st.columns(2)

        with r1:
            _at_risk = df[df["bill_pct"] < 90].sort_values("bill_pct").head(12)
            _risk_rows = ""
            for _, _sr in _at_risk.iterrows():
                _sp = float(_sr["bill_pct"])
                _sc = _pcc(_sp)
                _sn = str(_sr.get("shop_name", ""))[:22]
                _badge_lbl = "CRITICAL" if _sp < 70 else "WARNING"
                _badge_bg  = "rgba(239,68,68,0.15)" if _sp < 70 else "rgba(245,158,11,0.15)"
                _badge_clr = "#ef4444" if _sp < 70 else "#f59e0b"
                _not_scan  = int(_sr.get("bills_not_scanned", 0))
                _risk_rows += (
                    f'<div style="display:flex;align-items:center;justify-content:space-between;'
                    f'padding:8px 14px;border-bottom:1px solid #111827;">'
                    f'<div>'
                    f'<span style="font-size:12px;font-weight:700;color:#e2e8f0;">{_sr["shop_code"]}</span>'
                    f'<span style="font-size:9px;color:#475569;margin-left:7px;">{_sn}</span>'
                    f'</div>'
                    f'<div style="display:flex;align-items:center;gap:10px;">'
                    f'<span style="font-size:9px;color:#64748b;">{_not_scan:,} not scanned</span>'
                    f'<span style="font-size:9px;color:{_badge_clr};background:{_badge_bg};'
                    f'padding:2px 8px;border-radius:10px;font-weight:700;border:1px solid {_badge_clr};">{_badge_lbl}</span>'
                    f'<span style="font-size:14px;font-weight:900;color:{_sc};min-width:46px;text-align:right;">{_sp:.1f}%</span>'
                    f'</div></div>'
                )
            _risk_msg = _risk_rows or '<div style="padding:16px;text-align:center;color:#22c55e;font-weight:700;font-size:13px;">✅ All shops on track</div>'
            st.markdown(
                f'<div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:12px;overflow:hidden;">'
                f'<div style="background:#111827;border-top:3px solid #ef4444;padding:10px 16px;">'
                f'<span style="font-size:13px;font-weight:700;color:#f1f5f9;">⚠️ Shops Requiring Attention</span>'
                f'<span style="font-size:9px;color:#64748b;margin-left:8px;">scan% below 90 target</span>'
                f'</div>{_risk_msg}</div>',
                unsafe_allow_html=True,
            )

        with r2:
            _al_rows = ""
            _al_total_mtd = sum(a["mtd"] for a in _al_by_type) or 1
            for _a in _al_by_type:
                _icon, _clr = _al_icons.get(_a["name"], ("⚠️", "#94a3b8"))
                _pct_of_total = round(_a["mtd"] / _al_total_mtd * 100)
                _bar_w = max(1, _pct_of_total)
                _al_rows += (
                    f'<div style="padding:10px 14px;border-bottom:1px solid #111827;">'
                    f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px;">'
                    f'<div style="display:flex;align-items:center;gap:8px;">'
                    f'<span style="font-size:16px;">{_icon}</span>'
                    f'<span style="font-size:11px;color:#e2e8f0;font-weight:600;">{_a["name"]}</span>'
                    f'</div>'
                    f'<div style="text-align:right;">'
                    f'<span style="font-size:16px;font-weight:900;color:{_clr};">{_a["mtd"]:,}</span>'
                    f'<span style="font-size:9px;color:#64748b;margin-left:5px;">MTD</span>'
                    f'&nbsp;&nbsp;<span style="font-size:11px;color:#94a3b8;">today: {_a["today"]:,}</span>'
                    f'</div></div>'
                    f'<div style="height:6px;background:#1e293b;border-radius:3px;overflow:hidden;">'
                    f'<div style="width:{_bar_w}%;height:100%;background:{_clr};border-radius:3px;"></div>'
                    f'</div>'
                    f'<div style="font-size:8px;color:#475569;margin-top:2px;">{_pct_of_total}% of total alerts</div>'
                    f'</div>'
                )
            _al_block = _al_rows or '<div style="padding:16px;color:#64748b;font-size:11px;">No alerts found</div>'
            st.markdown(
                f'<div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:12px;overflow:hidden;">'
                f'<div style="background:#111827;border-top:3px solid #f59e0b;padding:10px 16px;">'
                f'<span style="font-size:13px;font-weight:700;color:#f1f5f9;">🚨 Alert Anatomy — MTD</span>'
                f'<span style="font-size:9px;color:#64748b;margin-left:8px;">{_al_total_mtd:,} total</span>'
                f'</div>{_al_block}</div>',
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Shop Intelligence Matrix
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown(
            '<div style="font-size:11px;color:#64748b;margin-bottom:8px;">'
            'Performance matrix across all shops — color-coded by tier. '
            '<span style="color:#22c55e;">■</span> ≥90% &nbsp;'
            '<span style="color:#f59e0b;">■</span> 70–90% &nbsp;'
            '<span style="color:#ef4444;">■</span> &lt;70%</div>',
            unsafe_allow_html=True,
        )

        # Build matrix rows
        _df_mat = df.copy().sort_values("bill_pct", ascending=True)
        _header_html = (
            '<thead><tr style="background:#0d1b35;">'
            '<th style="padding:9px 12px;font-size:10px;color:#94a3b8;font-weight:700;text-align:left;'
            'border-bottom:2px solid #1e3a5f;white-space:nowrap;">SHOP</th>'
            '<th style="padding:9px 10px;font-size:10px;color:#94a3b8;font-weight:700;text-align:right;'
            'border-bottom:2px solid #1e3a5f;white-space:nowrap;">ERP NOB</th>'
            '<th style="padding:9px 10px;font-size:10px;color:#94a3b8;font-weight:700;text-align:center;'
            'border-bottom:2px solid #1e3a5f;white-space:nowrap;min-width:140px;">SCAN %</th>'
            '<th style="padding:9px 10px;font-size:10px;color:#94a3b8;font-weight:700;text-align:right;'
            'border-bottom:2px solid #1e3a5f;white-space:nowrap;">NOT SCANNED</th>'
            '<th style="padding:9px 10px;font-size:10px;color:#94a3b8;font-weight:700;text-align:right;'
            'border-bottom:2px solid #1e3a5f;white-space:nowrap;">GHS DIFF</th>'
            '<th style="padding:9px 10px;font-size:10px;color:#94a3b8;font-weight:700;text-align:center;'
            'border-bottom:2px solid #1e3a5f;white-space:nowrap;">ALERTS</th>'
            '<th style="padding:9px 10px;font-size:10px;color:#94a3b8;font-weight:700;text-align:center;'
            'border-bottom:2px solid #1e3a5f;white-space:nowrap;min-width:120px;">TEST HANDOVER</th>'
            '<th style="padding:9px 10px;font-size:10px;color:#94a3b8;font-weight:700;text-align:center;'
            'border-bottom:2px solid #1e3a5f;white-space:nowrap;">GRADE</th>'
            '</tr></thead>'
        )

        _rows_html_parts = []
        for _ri, (_, _mr) in enumerate(_df_mat.iterrows()):
            _sp      = float(_mr["bill_pct"])
            _sc      = _pcc(_sp)
            _erp     = int(_mr["erp_nob"])
            _not_s   = int(_mr["bills_not_scanned"])
            _diff    = float(_mr["diff_ghs"])
            _al_cnt  = int(_mr["total_alerts"])
            _tb_pct  = float(_tb_map.get(str(_mr["shop_code"]).upper(), -1))
            _tb_show = f"{_tb_pct:.1f}%" if _tb_pct >= 0 else "—"
            _tb_clr2 = _pcc(_tb_pct) if _tb_pct >= 0 else "#475569"
            _diff_c  = "#22c55e" if _diff <= 0 else "#ef4444"
            _not_c   = "#22c55e" if _not_s == 0 else ("#f59e0b" if _not_s < 20 else "#ef4444")
            _al_c    = "#22c55e" if _al_cnt == 0 else ("#f59e0b" if _al_cnt < 10 else "#ef4444")
            _row_bg  = "rgba(255,255,255,0.013)" if _ri % 2 else "transparent"
            # Grade
            if   _sp >= 95 and _diff <= 0: _grade, _gc = "A+", "#22c55e"
            elif _sp >= 90:                _grade, _gc = "A",  "#22c55e"
            elif _sp >= 80:                _grade, _gc = "B",  "#84cc16"
            elif _sp >= 70:                _grade, _gc = "C",  "#f59e0b"
            elif _sp >= 60:                _grade, _gc = "D",  "#f97316"
            else:                          _grade, _gc = "F",  "#ef4444"
            # Inline scan bar (40px wide represented as divs)
            _bar_pct = max(1, min(100, round(_sp)))
            _sn_short = str(_mr.get("shop_name", ""))[:16]
            _row_html = (
                f'<tr style="background:{_row_bg};border-bottom:1px solid #111827;">'
                f'<td style="padding:8px 12px;">'
                f'<div style="font-size:12px;font-weight:700;color:#e2e8f0;">{_mr["shop_code"]}</div>'
                f'<div style="font-size:9px;color:#475569;">{_sn_short}</div>'
                f'</td>'
                f'<td style="padding:8px 10px;text-align:right;font-size:12px;color:#94a3b8;font-weight:600;">{_erp:,}</td>'
                f'<td style="padding:8px 10px;">'
                f'<div style="display:flex;align-items:center;gap:5px;">'
                f'<div style="flex:1;height:8px;background:#1e293b;border-radius:4px;overflow:hidden;">'
                f'<div style="width:{_bar_pct}%;height:100%;background:{_sc};border-radius:4px;"></div>'
                f'</div>'
                f'<span style="font-size:11px;font-weight:700;color:{_sc};min-width:40px;">{_sp:.1f}%</span>'
                f'</div></td>'
                f'<td style="padding:8px 10px;text-align:right;font-size:12px;font-weight:600;color:{_not_c};">{_not_s:,}</td>'
                f'<td style="padding:8px 10px;text-align:right;font-size:12px;font-weight:600;color:{_diff_c};">GHS {_diff:,.0f}</td>'
                f'<td style="padding:8px 10px;text-align:center;">'
                f'<span style="font-size:11px;font-weight:700;color:{_al_c};background:rgba(255,255,255,0.05);'
                f'padding:2px 8px;border-radius:10px;">{_al_cnt}</span>'
                f'</td>'
                f'<td style="padding:8px 10px;">'
                f'<div style="display:flex;align-items:center;gap:5px;">'
                f'<div style="flex:1;height:7px;background:#1e293b;border-radius:4px;overflow:hidden;">'
                f'<div style="width:{max(1,min(100,round(_tb_pct) if _tb_pct>=0 else 0))}%;height:100%;background:{_tb_clr2};border-radius:4px;"></div>'
                f'</div>'
                f'<span style="font-size:11px;font-weight:700;color:{_tb_clr2};min-width:34px;">{_tb_show}</span>'
                f'</div></td>'
                f'<td style="padding:8px 10px;text-align:center;">'
                f'<span style="font-size:16px;font-weight:900;color:{_gc};background:rgba(255,255,255,0.07);'
                f'width:30px;height:30px;display:inline-flex;align-items:center;justify-content:center;'
                f'border-radius:6px;">{_grade}</span>'
                f'</td>'
                f'</tr>'
            )
            _rows_html_parts.append(_row_html)

        _table_body = "".join(_rows_html_parts)
        _matrix_html = (
            f'<div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:12px;'
            f'overflow:auto;max-height:620px;">'
            f'<table style="width:100%;border-collapse:collapse;font-family:Arial,sans-serif;">'
            f'{_header_html}<tbody>{_table_body}</tbody>'
            f'</table></div>'
        )
        components.html(_matrix_html, height=640, scrolling=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Financial & Alert Pulse
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        # Top exposure scorecard
        _f1, _f2, _f3, _f4 = st.columns(4)
        _tot_erp_ghs = float(df["erp_erp_nob_ghs"].sum())
        _tot_scn_ghs = float(df["scanned_nob_ghs"].sum()) + float(df["consumable_nob_ghs"].sum())
        _ghs_rec_pct = round(_tot_scn_ghs / max(_tot_erp_ghs, 1) * 100, 1)
        for _col2, (_ico2, _lbl2, _val2, _sub2, _c2) in zip([_f1, _f2, _f3, _f4], [
            ("💵", "ERP GHS Value",   f"GHS {_tot_erp_ghs:,.0f}", f"{start_date.strftime('%d %b')} – {end_date.strftime('%d %b')}", "#38bdf8"),
            ("✅", "Scanned GHS",     f"GHS {_tot_scn_ghs:,.0f}", f"{_ghs_rec_pct:.1f}% recovered", _pcc(_ghs_rec_pct)),
            ("📉", "GHS Exposure",    f"GHS {abs(total_diff):,.0f}", _diff_lbl, _diff_clr),
            ("🏪", "Shops Exposed",  f"{over_exposed}",  f"of {len(df)} shops", "#f59e0b" if over_exposed > 0 else "#22c55e"),
        ]):
            with _col2:
                st.markdown(
                    f'<div style="background:#0a1628;border:1px solid #1e3a5f;border-left:5px solid {_c2};'
                    f'border-radius:12px;padding:14px 12px;margin-bottom:10px;">'
                    f'<div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:1px;">{_ico2} {_lbl2}</div>'
                    f'<div style="font-size:19px;font-weight:900;color:{_c2};margin:5px 0 2px;">{_val2}</div>'
                    f'<div style="font-size:9px;color:#64748b;">{_sub2}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Top shops by exposure
        st.markdown(
            '<div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;'
            'letter-spacing:1px;margin:6px 0 8px;">📊 Top 10 Shops by Financial Exposure (GHS Diff)</div>',
            unsafe_allow_html=True,
        )
        _top_exp = df[df["diff_ghs"] > 0].sort_values("diff_ghs", ascending=False).head(10)
        if _top_exp.empty:
            st.success("✅ No shops with positive GHS diff — all shops are in surplus!")
        else:
            _max_diff = float(_top_exp["diff_ghs"].max()) or 1
            _exp_rows = ""
            for _ri2, (_, _er) in enumerate(_top_exp.iterrows()):
                _ed = float(_er["diff_ghs"])
                _ep = round(_ed / _max_diff * 100)
                _en = str(_er.get("shop_name", ""))[:24]
                _grade_val = float(_er["bill_pct"])
                _gc2 = _pcc(_grade_val)
                _exp_rows += (
                    f'<div style="display:flex;align-items:center;gap:12px;padding:9px 14px;'
                    f'border-bottom:1px solid #111827;">'
                    f'<div style="font-size:11px;color:#475569;font-weight:700;min-width:20px;">{_ri2+1}</div>'
                    f'<div style="min-width:120px;">'
                    f'<div style="font-size:12px;font-weight:700;color:#e2e8f0;">{_er["shop_code"]}</div>'
                    f'<div style="font-size:9px;color:#475569;">{_en}</div>'
                    f'</div>'
                    f'<div style="flex:1;height:12px;background:#1e293b;border-radius:6px;overflow:hidden;">'
                    f'<div style="width:{_ep}%;height:100%;background:linear-gradient(90deg,#ef4444,#dc2626);border-radius:6px;"></div>'
                    f'</div>'
                    f'<div style="min-width:110px;text-align:right;">'
                    f'<span style="font-size:13px;font-weight:900;color:#ef4444;">GHS {_ed:,.0f}</span>'
                    f'</div>'
                    f'<div style="min-width:48px;text-align:right;">'
                    f'<span style="font-size:11px;font-weight:700;color:{_gc2};">{_grade_val:.1f}%</span>'
                    f'</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:12px;overflow:hidden;">'
                f'<div style="background:#111827;border-top:3px solid #ef4444;padding:9px 14px;">'
                f'<span style="font-size:12px;font-weight:700;color:#f1f5f9;">Top Financial Exposure by Shop</span>'
                f'</div>{_exp_rows}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        # Alert deep-dive by type — visual cards
        st.markdown(
            '<div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;'
            'letter-spacing:1px;margin:6px 0 8px;">🚨 Alert Deep-Dive — MTD Breakdown</div>',
            unsafe_allow_html=True,
        )
        _al_card_cols = st.columns(len(_al_by_type) if _al_by_type else 1)
        for _ci2, (_ac, _adata) in enumerate(zip(_al_card_cols, _al_by_type)):
            _aicon, _aclr = _al_icons.get(_adata["name"], ("⚠️", "#94a3b8"))
            _a_share = round(_adata["mtd"] / max(sum(a["mtd"] for a in _al_by_type), 1) * 100)
            with _ac:
                st.markdown(
                    f'<div style="background:#0a1628;border:1px solid #1e3a5f;border-top:4px solid {_aclr};'
                    f'border-radius:12px;padding:14px 12px;text-align:center;">'
                    f'<div style="font-size:24px;margin-bottom:6px;">{_aicon}</div>'
                    f'<div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:1px;margin-bottom:8px;">{_adata["name"]}</div>'
                    f'<div style="font-size:26px;font-weight:900;color:{_aclr};">{_adata["mtd"]:,}</div>'
                    f'<div style="font-size:9px;color:#64748b;margin:3px 0 8px;">MTD &nbsp;·&nbsp; today: {_adata["today"]:,}</div>'
                    f'<div style="height:6px;background:#1e293b;border-radius:3px;overflow:hidden;">'
                    f'<div style="width:{_a_share}%;height:100%;background:{_aclr};border-radius:3px;"></div>'
                    f'</div>'
                    f'<div style="font-size:9px;color:#475569;margin-top:4px;">{_a_share}% of total</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# INVOICE NEWSLETTER
# ══════════════════════════════════════════════════════════════════════════════

def _query_inv_newsletter_data(report_date: date) -> dict:
    """Fetch all metrics needed for the invoice scanning newsletter."""
    import psycopg2 as _pg2
    _conn = _pg2.connect(**DB_CONFIG)
    result = {}
    _shops = sorted(list(IMPLEMENTED_SHOPS_SET))
    mtd_start = report_date.replace(day=1)
    d_str     = report_date.strftime("%Y-%m-%d")
    m_str     = mtd_start.strftime("%Y-%m-%d")

    try:
        with _conn.cursor() as cur:

            # ── Yesterday overall ───────────────────────────────────────────
            cur.execute("""
                SELECT SUM(e.erp_nob)::bigint,
                       SUM(COALESCE(s.scan_nob,0))::bigint,
                       SUM(COALESCE(s.consumable_nob,0))::bigint
                FROM mv_wh_erp_daily e
                LEFT JOIN mv_wh_invoices_agg_daily s
                       ON s.bill_date = e.bill_date AND s.shop_code = e.shop_code
                WHERE e.bill_date = %s AND e.shop_code = ANY(%s)
            """, (d_str, _shops))
            row = cur.fetchone()
            y_erp   = int(row[0] or 0)
            y_scan  = int(row[1] or 0)
            y_cons  = int(row[2] or 0)
            y_acc   = y_scan + y_cons
            y_pct   = round(y_acc / y_erp * 100, 1) if y_erp else 0
            result.update({"yest_erp": y_erp, "yest_scan": y_scan, "yest_cons": y_cons,
                           "yest_accounted": y_acc, "yest_scan_pct": y_pct,
                           "yest_date": report_date.strftime("%d %b %Y")})

            # ── MTD overall ─────────────────────────────────────────────────
            cur.execute("""
                SELECT SUM(e.erp_nob)::bigint,
                       SUM(COALESCE(s.scan_nob,0))::bigint,
                       SUM(COALESCE(s.consumable_nob,0))::bigint,
                       SUM(e.erp_nob_ghs)::numeric,
                       SUM(COALESCE(s.scan_nob_ghs,0) + COALESCE(s.consumable_nob_ghs,0))::numeric
                FROM mv_wh_erp_daily e
                LEFT JOIN mv_wh_invoices_agg_daily s
                       ON s.bill_date = e.bill_date AND s.shop_code = e.shop_code
                WHERE e.bill_date BETWEEN %s AND %s AND e.shop_code = ANY(%s)
            """, (m_str, d_str, _shops))
            row = cur.fetchone()
            m_erp  = int(row[0] or 0)
            m_scan = int(row[1] or 0)
            m_cons = int(row[2] or 0)
            m_acc  = m_scan + m_cons
            m_pct  = round(m_acc / m_erp * 100, 1) if m_erp else 0
            m_erp_ghs  = float(row[3] or 0)
            m_acc_ghs  = float(row[4] or 0)
            m_diff_ghs = round(m_erp_ghs - m_acc_ghs, 0)
            result.update({"mtd_erp": m_erp, "mtd_scan": m_scan, "mtd_cons": m_cons,
                           "mtd_accounted": m_acc, "mtd_scan_pct": m_pct,
                           "mtd_diff_ghs": m_diff_ghs, "mtd_erp_ghs": m_erp_ghs,
                           "mtd_start": mtd_start.strftime("%d %b")})

            # ── Bottom 5 shops (yesterday by scan_pct) ───────────────────────
            cur.execute("""
                SELECT e.shop_code,
                       e.erp_nob,
                       COALESCE(s.scan_nob,0) + COALESCE(s.consumable_nob,0) AS accounted,
                       ROUND((COALESCE(s.scan_nob,0)+COALESCE(s.consumable_nob,0))
                             ::numeric / NULLIF(e.erp_nob,0) * 100, 1) AS scan_pct
                FROM mv_wh_erp_daily e
                LEFT JOIN mv_wh_invoices_agg_daily s
                       ON s.bill_date = e.bill_date AND s.shop_code = e.shop_code
                WHERE e.bill_date = %s AND e.shop_code = ANY(%s) AND e.erp_nob > 0
                ORDER BY scan_pct ASC NULLS FIRST
                LIMIT 5
            """, (d_str, _shops))
            result["bottom5_shops"] = [
                {"shop": r[0], "erp_nob": int(r[1] or 0),
                 "accounted": int(r[2] or 0), "pct": float(r[3] or 0)}
                for r in cur.fetchall()
            ]

            # ── Alert counts MTD ─────────────────────────────────────────────
            cur.execute("""
                SELECT
                  SUM(CASE WHEN LOWER(TRIM(a_type)) LIKE '%%test%%'      THEN 1 ELSE 0 END),
                  SUM(CASE WHEN LOWER(TRIM(a_type)) LIKE '%%duplicate%%'  THEN 1 ELSE 0 END),
                  SUM(CASE WHEN LOWER(TRIM(a_type)) LIKE '%%date%%'
                            OR  LOWER(TRIM(a_type)) LIKE '%%mismatch%%'   THEN 1 ELSE 0 END),
                  SUM(CASE WHEN LOWER(TRIM(a_type)) LIKE '%%wrong%%'
                            OR  LOWER(TRIM(a_type)) LIKE '%%invalid%%'    THEN 1 ELSE 0 END),
                  SUM(CASE WHEN LOWER(TRIM(a_type)) LIKE '%%high%%'       THEN 1 ELSE 0 END),
                  COUNT(*)
                FROM alerts
                WHERE UPPER(TRIM(a_store_code)) = ANY(%s)
                  AND COALESCE(scanned_date::date, a_entrytime::date)
                      BETWEEN %s AND %s
            """, (_shops, m_str, d_str))
            row = cur.fetchone()
            result["alerts"] = {
                "test_bill": int(row[0] or 0), "duplicate": int(row[1] or 0),
                "date_mismatch": int(row[2] or 0), "wrong_shop": int(row[3] or 0),
                "high_value": int(row[4] or 0), "total": int(row[5] or 0),
            }

            # ── Test bill handover summary ───────────────────────────────────
            cur.execute("""
                SELECT SUM(cashier_login)::bigint, SUM(handover_count)::bigint
                FROM mv_wh_erp_cashier_sessions_daily
                WHERE bill_date BETWEEN %s AND %s
                  AND shop_code = ANY(%s)
            """, (m_str, d_str, _shops))
            row = cur.fetchone()
            tb_login  = int(row[0] or 0)
            tb_hand   = int(row[1] or 0)
            tb_pct    = round(tb_hand / max(tb_login, 1) * 100, 1)
            result["testbill"] = {"cashier_login": tb_login, "handover": tb_hand, "handover_pct": tb_pct}

    except Exception as _e:
        logger.warning("Newsletter data query error: %s", _e)
    finally:
        _conn.close()

    return result


def _build_inv_newsletter_html(data: dict) -> str:
    import html as _html
    import smtplib

    def _esc(v): return _html.escape(str(v or ""))

    def _bar(pct: float, fill: str, bg: str = "#1e293b") -> str:
        p = max(1, min(99, round(float(pct or 0))))
        r = 100 - p
        return (
            f'<table width="100%" cellpadding="0" cellspacing="0" border="0">'
            f'<tr>'
            f'<td width="{p}%" bgcolor="{fill}" style="background-color:{fill};'
            f'height:10px;font-size:1px;line-height:1px;">&nbsp;</td>'
            f'<td width="{r}%" bgcolor="{bg}" style="background-color:{bg};'
            f'height:10px;font-size:1px;line-height:1px;">&nbsp;</td>'
            f'</tr></table>'
        )

    def _cc(pct: float) -> str:
        if pct >= 90: return "#22c55e"
        if pct >= 70: return "#f59e0b"
        return "#ef4444"

    def _kpi(icon, label, value, sub, clr):
        return (
            f'<td width="25%" style="padding:5px;">'
            f'<table width="100%" cellpadding="0" cellspacing="0">'
            f'<tr><td bgcolor="#1a2744" style="background-color:#1a2744;'
            f'border:1px solid #1e3a5f;border-left:4px solid {clr};'
            f'border-radius:8px;padding:12px 8px;text-align:center;">'
            f'<div style="font-size:18px;margin-bottom:4px;">{icon}</div>'
            f'<div style="font-size:9px;color:#94a3b8;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:4px;">{_esc(label)}</div>'
            f'<div style="font-size:17px;font-weight:900;color:{clr};font-family:Arial,sans-serif;">'
            f'{_esc(value)}</div>'
            f'<div style="font-size:9px;color:#64748b;margin-top:3px;">{_esc(sub)}</div>'
            f'</td></tr></table></td>'
        )

    sent_ts = datetime.now().strftime("%d %b %Y %H:%M")
    yest_pct = float(data.get("yest_scan_pct", 0))
    mtd_pct  = float(data.get("mtd_scan_pct", 0))
    yest_clr = _cc(yest_pct)
    mtd_clr  = _cc(mtd_pct)
    yest_date = data.get("yest_date", "")
    mtd_start = data.get("mtd_start", "")

    # Bottom 5 shops rows
    bot5 = data.get("bottom5_shops", [])
    bot5_rows = ""
    for s in bot5:
        clr = _cc(s["pct"])
        bot5_rows += (
            f'<tr style="border-bottom:1px solid #1e293b;">'
            f'<td style="padding:7px 10px;font-size:12px;color:#e2e8f0;font-weight:600;">'
            f'{_esc(s["shop"])}</td>'
            f'<td style="padding:7px 10px;font-size:12px;color:#94a3b8;text-align:right;">'
            f'{int(s["erp_nob"]):,}</td>'
            f'<td style="padding:7px 10px;font-size:12px;color:#94a3b8;text-align:right;">'
            f'{int(s["accounted"]):,}</td>'
            f'<td style="padding:7px 10px;">'
            f'<table width="100%" cellpadding="0" cellspacing="0">'
            f'<tr>'
            f'<td style="font-size:11px;color:{clr};font-weight:700;white-space:nowrap;'
            f'padding-right:6px;width:36px;">{s["pct"]:.1f}%</td>'
            f'<td width="100%">{_bar(s["pct"], clr)}</td>'
            f'</tr></table></td>'
            f'</tr>'
        )
    if not bot5_rows:
        bot5_rows = '<tr><td colspan="4" style="padding:10px;color:#64748b;text-align:center;font-size:11px;">No shop data</td></tr>'

    # Alerts block
    al = data.get("alerts", {})
    al_total = int(al.get("total", 0))
    _alert_rows_parts = [
        ("Test Bill", al.get("test_bill", 0), "#3b82f6"),
        ("Duplicate",  al.get("duplicate", 0), "#ef4444"),
        ("Date Mismatch", al.get("date_mismatch", 0), "#f59e0b"),
        ("Wrong Shop", al.get("wrong_shop", 0), "#8b5cf6"),
        ("High Value", al.get("high_value", 0), "#22c55e"),
    ]
    alert_rows_html = ""
    for albl, aval, aclr in _alert_rows_parts:
        a_pct = round(aval / max(al_total, 1) * 100)
        alert_rows_html += (
            f'<tr style="border-bottom:1px solid #1e293b;">'
            f'<td style="padding:6px 10px;font-size:12px;color:#e2e8f0;">{_esc(albl)}</td>'
            f'<td style="padding:6px 10px;font-size:12px;color:{aclr};font-weight:700;'
            f'text-align:right;">{int(aval):,}</td>'
            f'<td style="padding:6px 10px;width:120px;">{_bar(a_pct, aclr)}</td>'
            f'</tr>'
        )

    tb = data.get("testbill", {})
    tb_pct    = float(tb.get("handover_pct", 0))
    tb_pct_clr = _cc(tb_pct)
    tb_hand   = int(tb.get("handover", 0))
    tb_login  = int(tb.get("cashier_login", 0))

    mtd_diff = float(data.get("mtd_diff_ghs", 0))
    diff_clr = "#22c55e" if mtd_diff <= 0 else "#ef4444"

    html_out = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background-color:#060d1a;font-family:Arial,Helvetica,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" border="0"
       style="background-color:#060d1a;padding:20px 10px;">
<tr><td align="center">

<table width="680" cellpadding="0" cellspacing="0" border="0"
       style="background-color:#0a1628;border-radius:16px;border:1px solid #1e3a5f;
              max-width:680px;width:100%;">

  <!-- HEADER -->
  <tr>
    <td bgcolor="#060d1a" style="background-color:#060d1a;padding:20px 28px 14px;
        border-radius:16px 16px 0 0;border-bottom:2px solid #1e3a5f;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td>
            <img src="{MELCOM_LOGO}" height="32" alt="Melcom"
                 style="display:block;max-height:32px;width:auto;border:0;">
          </td>
          <td align="right">
            <div style="font-size:10px;color:#475569;font-family:Arial,sans-serif;">
              {sent_ts}
            </div>
          </td>
        </tr>
      </table>
      <div style="font-size:18px;font-weight:900;color:#f1f5f9;margin-top:10px;
                  font-family:Arial,sans-serif;letter-spacing:0.3px;">
        Invoice Scanning — Daily Report
      </div>
      <div style="font-size:12px;color:#64748b;margin-top:4px;">
        Scan compliance, alert analysis &amp; financial impact &nbsp;·&nbsp;
        <span style="color:#38bdf8;">{yest_date}</span>
      </div>
    </td>
  </tr>

  <!-- KPIs ROW -->
  <tr>
    <td bgcolor="#0d1b35" style="background-color:#0d1b35;padding:16px 20px;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          {_kpi("📦", "ERP NOB", f"{data.get('yest_erp',0):,}", yest_date, "#38bdf8")}
          {_kpi("✅", "Accounted", f"{data.get('yest_accounted',0):,}", "Scan + Consumable", "#6366f1")}
          {_kpi("📊", "Scan %", f"{yest_pct:.1f}%", "Yesterday coverage", yest_clr)}
          {_kpi("📅", "MTD Scan %", f"{mtd_pct:.1f}%", f"{mtd_start} – {yest_date.split()[0]} {yest_date.split()[1]}", mtd_clr)}
        </tr>
      </table>
    </td>
  </tr>

  <!-- COMPLIANCE BARS -->
  <tr>
    <td bgcolor="#0a1628" style="background-color:#0a1628;padding:14px 24px;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td style="font-size:12px;color:#e2e8f0;font-weight:600;padding-bottom:5px;">
            Yesterday Scan Compliance
            <span style="color:{yest_clr};font-size:13px;font-weight:900;margin-left:8px;">
              {yest_pct:.1f}%
            </span>
          </td>
        </tr>
        <tr><td>{_bar(yest_pct, yest_clr)}</td></tr>
        <tr>
          <td style="font-size:12px;color:#e2e8f0;font-weight:600;padding-top:10px;padding-bottom:5px;">
            MTD Scan Compliance
            <span style="color:{mtd_clr};font-size:13px;font-weight:900;margin-left:8px;">
              {mtd_pct:.1f}%
            </span>
          </td>
        </tr>
        <tr><td>{_bar(mtd_pct, mtd_clr)}</td></tr>
      </table>
    </td>
  </tr>

  <!-- BOTTOM 5 SHOPS -->
  <tr>
    <td bgcolor="#111827" style="background-color:#111827;border-top:3px solid #ef4444;
        padding:11px 22px 0;">
      <span style="font-size:13px;font-weight:700;color:#f1f5f9;">
        ⚠️ Bottom 5 Shops — Yesterday Compliance
      </span>
    </td>
  </tr>
  <tr>
    <td bgcolor="#0a1628" style="background-color:#0a1628;padding:0 14px 10px;">
      <table width="100%" cellpadding="0" cellspacing="0"
             style="border-collapse:collapse;font-family:Arial,sans-serif;">
        <tr style="background:#111827;">
          <th style="padding:8px 10px;font-size:10px;color:#94a3b8;font-weight:700;
              text-align:left;border-bottom:1px solid #1e293b;">Shop</th>
          <th style="padding:8px 10px;font-size:10px;color:#94a3b8;font-weight:700;
              text-align:right;border-bottom:1px solid #1e293b;">ERP NOB</th>
          <th style="padding:8px 10px;font-size:10px;color:#94a3b8;font-weight:700;
              text-align:right;border-bottom:1px solid #1e293b;">Accounted</th>
          <th style="padding:8px 10px;font-size:10px;color:#94a3b8;font-weight:700;
              border-bottom:1px solid #1e293b;">Scan %</th>
        </tr>
        {bot5_rows}
      </table>
    </td>
  </tr>

  <!-- ALERT SUMMARY -->
  <tr>
    <td bgcolor="#111827" style="background-color:#111827;border-top:3px solid #f59e0b;
        padding:11px 22px 0;">
      <span style="font-size:13px;font-weight:700;color:#f1f5f9;">
        🚨 Alert Summary — MTD &nbsp;·&nbsp;
        <span style="color:#f59e0b;">{al_total:,} total alerts</span>
      </span>
    </td>
  </tr>
  <tr>
    <td bgcolor="#0a1628" style="background-color:#0a1628;padding:0 14px 10px;">
      <table width="100%" cellpadding="0" cellspacing="0"
             style="border-collapse:collapse;font-family:Arial,sans-serif;">
        {alert_rows_html}
      </table>
    </td>
  </tr>

  <!-- FINANCIAL + TEST BILL ROW -->
  <tr>
    <td bgcolor="#0a1628" style="background-color:#0a1628;padding:10px 14px;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td width="49%" bgcolor="#111827" style="background-color:#111827;
              border:1px solid #1e3a5f;border-left:4px solid {diff_clr};
              border-radius:8px;padding:12px 14px;">
            <div style="font-size:10px;color:#94a3b8;font-weight:700;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:6px;">💰 MTD GHS Diff</div>
            <div style="font-size:22px;font-weight:900;color:{diff_clr};font-family:Arial;">
              GHS {mtd_diff:,.0f}
            </div>
            <div style="font-size:10px;color:#64748b;margin-top:4px;">
              ERP: GHS {float(data.get("mtd_erp_ghs",0)):,.0f}
            </div>
          </td>
          <td width="2%"></td>
          <td width="49%" bgcolor="#111827" style="background-color:#111827;
              border:1px solid #1e3a5f;border-left:4px solid {tb_pct_clr};
              border-radius:8px;padding:12px 14px;">
            <div style="font-size:10px;color:#94a3b8;font-weight:700;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:6px;">📋 Test Bill Handover MTD</div>
            <div style="font-size:22px;font-weight:900;color:{tb_pct_clr};font-family:Arial;">
              {tb_pct:.1f}%
            </div>
            <div style="font-size:10px;color:#64748b;margin-top:4px;">
              {tb_hand:,} of {tb_login:,} cashiers handed over
            </div>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- DASHBOARD LINK -->
  <tr>
    <td bgcolor="#0d1b35" style="background-color:#0d1b35;padding:12px 24px;text-align:center;">
      <a href="{INVOICE_DASHBOARD_URL}"
         style="display:inline-block;background:#1e40af;color:#ffffff;font-weight:700;
                font-size:12px;padding:10px 28px;border-radius:8px;text-decoration:none;
                font-family:Arial,sans-serif;letter-spacing:0.3px;">
        🔗 View Invoice Scanning Dashboard
      </a>
    </td>
  </tr>

  <!-- FOOTER -->
  <tr>
    <td bgcolor="#060d1a" style="background-color:#060d1a;padding:14px 24px;
        text-align:center;border-top:1px solid #1e3a5f;border-radius:0 0 16px 16px;">
      <img src="{MELCOM_LOGO}" width="80" alt="Melcom"
           style="display:block;margin:0 auto 8px;max-height:26px;width:auto;border:0;">
      <div style="font-size:11px;color:#475569;font-family:Arial,sans-serif;line-height:1.6;">
        Melcom Group &bull; Invoice Scanning &bull; {sent_ts}<br>
        <span style="color:#334155;">Auto-generated after data update. Do not reply.</span>
      </div>
    </td>
  </tr>

</table>
</td></tr>
</table>
</body>
</html>"""
    return html_out


def send_inv_newsletter(report_date: date, owner_excel: bytes, testbill_excel: bytes) -> tuple[bool, str]:
    """
    Send invoice scanning newsletter HTML with Excel attachments via SMTP.
    Replaces the plain-text email body; attachments are unchanged.
    """
    import smtplib, tempfile
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    try:
        data = _query_inv_newsletter_data(report_date)
        html = _build_inv_newsletter_html(data)

        date_label = _invoice_date_label(report_date)
        subject    = f"News Letter - Invoice Scanning_{date_label}"
        _dl        = report_date.strftime('%d%b%y')
        attach1_name = f"INV_summary_Shopwise_{_dl}.xlsx"
        attach2_name = f"Shopwise_test_bill_analysis_{_dl}.xlsx"

        # ── Try Outlook COM first (saves to Sent Items) ───────────────────
        try:
            import win32com.client as _win32
            outlook = _win32.Dispatch("Outlook.Application")
            mail = outlook.CreateItem(0)
            mail.To       = INVOICE_EMAIL_TO
            mail.Subject  = subject
            mail.HTMLBody = html
            for xls, name in [(owner_excel, attach1_name), (testbill_excel, attach2_name)]:
                fd, tmp = tempfile.mkstemp(suffix=f"_{name}")
                try: os.write(fd, xls)
                finally: os.close(fd)
                mail.Attachments.Add(tmp)
                try: os.unlink(tmp)
                except Exception: pass
            mail.Send()
            return True, f"Newsletter sent via Outlook to {INVOICE_EMAIL_TO}"
        except Exception as _oe:
            logger.warning("Outlook COM failed: %s — trying SMTP", _oe)

        # ── SMTP fallback ─────────────────────────────────────────────────
        msg = MIMEMultipart("mixed")
        msg["From"]    = INVOICE_EMAIL_FROM
        msg["To"]      = INVOICE_EMAIL_TO
        msg["Subject"] = subject
        msg.attach(MIMEText(html, "html", "utf-8"))
        for xls, name in [(owner_excel, attach1_name), (testbill_excel, attach2_name)]:
            part = MIMEBase("application", "vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            part.set_payload(xls)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{name}"')
            msg.attach(part)
        with smtplib.SMTP(INVOICE_SMTP_HOST, INVOICE_SMTP_PORT, timeout=20) as srv:
            srv.sendmail(INVOICE_EMAIL_FROM, [INVOICE_EMAIL_TO], msg.as_string())
        return True, f"Newsletter sent via SMTP to {INVOICE_EMAIL_TO}"

    except Exception as e:
        logger.error("Invoice newsletter send error: %s", e)
        return False, str(e)


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

    # Clear any stale update notification from session state
    st.session_state.pop("_inv_update_notify", None)

    with st.container():
        hdr_col, date_col, update_col, refresh_col = st.columns([0.58, 0.20, 0.14, 0.08])
        with hdr_col:
            st.markdown("")
        with date_col:
            selected_range = st.date_input("Date Range", value=(yesterday, yesterday))
        with update_col:
            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
            if st.button(
                "⬆ Update Data",
                key="inv_update_btn",
                use_container_width=True,
                help=f"Delete & Re-Fetch {yesterday.strftime('%d %b %Y')} data from MySQL (invcentral)",
            ):
                st.session_state["_inv_update_trigger"] = True
                st.rerun()
        with refresh_col:
            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
            refresh = st.button("Refresh", help="Clear cache and reload fresh data", use_container_width=True)

    # ── CTA handler: Delete & Re-Fetch yesterday ─────────────────────────────
    # ERPDATA  → \\10.10.0.30\mis\shopbillcount_YYYYMMDD.csv  (network share)
    # All others → MySQL invcentral (Delete & Re-Fetch from date)
    if st.session_state.pop("_inv_update_trigger", False):
        from_date  = yesterday.strftime("%Y-%m-%d")
        date_label = yesterday.strftime("%d %b %Y")

        # ── Progress bar UI ───────────────────────────────────────────────────
        # Stages and their cumulative completion %:
        # ERPDATA CSV=40 · ALERTS=55 · INVOICES=70 · inv_manager=80
        # Dup flags=85 · Excel=93 · Email=100
        _STAGES = [
            ( 5,  "📂  ERPDATA — reading CSV from network share"),
            (35,  "💾  ERPDATA — loading into database"),
            (48,  "🔌  ALERTS — fetching from MySQL"),
            (61,  "🔌  INVOICES — fetching from MySQL"),
            (72,  "🔌  invoices_manager — fetching from MySQL"),
            (78,  "🔄  Normalising invoice duplicate flags"),
            (86,  "🔃  Refreshing materialized views"),
            (94,  "📊  Building Excel reports"),
            (100, "📧  Sending email"),
        ]
        _total_stages = len(_STAGES)

        _prog_bar  = st.progress(0)
        _stage_box = st.empty()

        def _set_stage(idx: int, override_label: str | None = None):
            pct, label = _STAGES[min(idx, _total_stages - 1)]
            _prog_bar.progress(pct / 100)
            display_label = override_label or label
            pending = _total_stages - idx - 1
            _stage_box.markdown(
                f"""<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                               padding:10px 16px;margin:4px 0;">
                  <div style="color:#e6edf3;font-weight:600;font-size:13px;">{display_label}</div>
                  <div style="color:#8b949e;font-size:11px;margin-top:4px;">
                    Stage {idx + 1} of {_total_stages} &nbsp;·&nbsp;
                    <span style="color:#58a6ff;">{pct}% complete</span>
                    {f"&nbsp;·&nbsp; {pending} step(s) remaining" if pending > 0 else ""}
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

        _all_results = []
        try:
            # ── Stage 0-1: ERPDATA — historical + yesterday ───────────────────
            # Scan the network share for all available shopbillcount CSVs in the
            # past 6 months.  Load any dates not already in erpdata so that the
            # diff-trend graphs have real historical ERP values after MV refresh.
            _set_stage(0)
            import glob as _glob, re as _re

            _six_months_ago = yesterday - timedelta(days=180)
            _all_erp_dates: list[date] = []
            try:
                _pattern = os.path.join(ERPDATA_CSV_PATH, "shopbillcount_*.csv")
                for _fp in _glob.glob(_pattern):
                    _fn = os.path.basename(_fp)
                    _m = _re.match(r"shopbillcount_(\d{8})\.csv", _fn, _re.IGNORECASE)
                    if _m:
                        try:
                            _pd = datetime.strptime(_m.group(1), "%Y%m%d").date()
                            if _six_months_ago <= _pd <= yesterday:
                                _all_erp_dates.append(_pd)
                        except ValueError:
                            pass
                _all_erp_dates.sort()
            except Exception:
                pass

            if not _all_erp_dates:
                _all_erp_dates = [yesterday]

            # Find dates already loaded (skip re-loading, always reload yesterday)
            _existing_erp_dates: set[str] = set()
            try:
                with psycopg2.connect(**PG_CONFIG_REFETCH) as _pg_chk:
                    with _pg_chk.cursor() as _chk_cur:
                        _chk_cur.execute(
                            "SELECT DISTINCT invdate FROM erpdata WHERE invdate >= %s",
                            (_six_months_ago.strftime('%Y%m%d'),),
                        )
                        _existing_erp_dates = {r[0] for r in _chk_cur.fetchall()}
            except Exception:
                pass

            _to_load = [
                d for d in _all_erp_dates
                if d.strftime('%Y%m%d') not in _existing_erp_dates or d == yesterday
            ]
            if not _to_load:
                _to_load = [yesterday]

            _total_erp_inserted = 0
            for _i, _d in enumerate(_to_load):
                _pct_within = int(5 + 30 * _i / max(len(_to_load), 1))
                _prog_bar.progress(_pct_within / 100)
                _stage_box.markdown(
                    f"""<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                                   padding:10px 16px;margin:4px 0;">
                      <div style="color:#e6edf3;font-weight:600;font-size:13px;">
                        📂  ERPDATA — {_d.strftime('%d %b %Y')} ({_i + 1}/{len(_to_load)})
                      </div>
                      <div style="color:#8b949e;font-size:11px;margin-top:4px;">
                        Stage 1 of {_total_stages} &nbsp;·&nbsp;
                        <span style="color:#58a6ff;">{_pct_within}% complete</span>
                        &nbsp;·&nbsp; {_total_stages - 1} step(s) remaining
                      </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                _sub = _sync_erpdata_for_date(_d, PG_CONFIG_REFETCH, status_fn=None)
                if _sub.get('status') == 'success':
                    _total_erp_inserted += _sub.get('inserted', 0)
                if _d == yesterday:
                    _all_results.append(_sub)

            _set_stage(1, f"✅  ERPDATA — {_total_erp_inserted:,} rows loaded ({len(_to_load)} file(s))")

            # ── Stages 2-4: MySQL tables (one stage each) ─────────────────────
            for _si, _tbl in enumerate(MYSQL_REFETCH_TABLES, start=2):
                _set_stage(_si)
                _r = _refetch_tables_from_date(
                    mysql_config=MYSQL_CONFIG,
                    pg_config=PG_CONFIG_REFETCH,
                    from_date_str=from_date,
                    tables=[_tbl],
                    batch_size=20000,
                    status_fn=None,
                )
                _all_results.extend(_r)
                _ins = sum(x.get('inserted', 0) for x in _r)
                _set_stage(_si, f"✅  {_tbl} — {_ins:,} rows loaded")

            # ── Stage 5: Normalise duplicates ─────────────────────────────────
            _set_stage(5)
            try:
                _dup_updated = _normalize_invoices_duplicates(PG_CONFIG_REFETCH)
                _set_stage(5, f"✅  Duplicate flags — {_dup_updated:,} rows updated")
            except Exception as _dup_err:
                _set_stage(5, f"⚠️  Duplicate flag refresh failed: {_dup_err}")

            # ── Stage 6: Refresh materialized views ───────────────────────────
            _set_stage(6)
            try:
                _refresh_dashboard_mvs(PG_CONFIG_REFETCH, status_fn=None)
                _set_stage(6, "✅  Materialized views refreshed")
            except Exception as _mv_err:
                _set_stage(6, f"⚠️  MV refresh: {_mv_err}")

            # ── Summary ───────────────────────────────────────────────────────
            _errors   = [r for r in _all_results if r.get('status') not in ('success','skipped')]
            _ok       = [r for r in _all_results if r.get('status') == 'success']
            total_ins = sum(r.get('inserted', 0) for r in _ok)

            if _errors:
                _prog_bar.progress(1.0)
                _stage_box.error(
                    f"⚠️ {len(_ok)} tables OK · {len(_errors)} error(s): "
                    + "; ".join(r['error'] for r in _errors if r.get('error'))
                )
            else:
                # ── Stage 7: Excel (MVs already refreshed → fresh data) ───────
                _set_stage(7)
                _owner_xls    = _build_owner_view_excel(yesterday)
                _testbill_xls = _build_testbill_excel(yesterday)
                _set_stage(7, "✅  Excel reports built")

                # ── Stage 8: Newsletter (with Excel attachments) ─────────────
                _set_stage(8)
                _nl_ok, _nl_msg = send_inv_newsletter(yesterday, _owner_xls, _testbill_xls)
                if _nl_ok:
                    _email_note = f"  ·  📧 Newsletter sent to {INVOICE_EMAIL_TO} ✓"
                    _set_stage(8, f"✅  Newsletter sent to {INVOICE_EMAIL_TO}")
                else:
                    _email_note = f"  ·  ⚠️ Newsletter failed: {_nl_msg}"
                    _set_stage(8, f"⚠️  Newsletter failed: {_nl_msg}")

                _prog_bar.progress(1.0)
                _stage_box.success(
                    f"✅ All done — **{total_ins:,} rows** for {date_label}{_email_note}"
                )

        except Exception as _upd_err:
            _prog_bar.progress(1.0)
            _stage_box.error(f"❌ Update failed: {_upd_err}")

        st.cache_data.clear()
        st.rerun()

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

    # ── Tabbed layout ─────────────────────────────────────────────────────────
    tab_ov, tab_eda = st.tabs(["📊 Overview", "📈 EDA Analytics"])

    with tab_ov:
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

        # Bill Handover % — Last 10 Days
        try:
            ho_df = load_bill_handover_last10(end_date)
        except Exception:
            ho_df = pd.DataFrame()
        if not ho_df.empty:
            ho_df["bill_date"]    = pd.to_datetime(ho_df["bill_date"])
            ho_df["day_label"]    = ho_df["bill_date"].dt.strftime("%d %b")
            ho_df["handover_pct"] = pd.to_numeric(ho_df["handover_pct"], errors="coerce").fillna(0)
            ho_df = ho_df[ho_df["total_generated"] > 0]
            if not ho_df.empty:
                fig_ho = go.Figure()
                fig_ho.add_trace(go.Scatter(
                    x=ho_df["day_label"], y=ho_df["handover_pct"],
                    mode="lines+markers",
                    line=dict(color="#38bdf8", width=2.5, shape="spline", smoothing=1.2),
                    marker=dict(size=5, color="#38bdf8", line=dict(color="#0a0f1e", width=1)),
                    fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
                    hovertemplate="%{x}<br><b>%{y:.1f}%</b><extra></extra>",
                ))
                fig_ho.update_layout(
                    title=dict(text="Bill Handover % — Last 10 Days",
                               font=dict(size=11, color="#ffffff"), x=0, xanchor="left"),
                    height=160, margin=dict(l=4, r=4, t=28, b=4),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=9, color="#ffffff")),
                    yaxis=dict(showgrid=True, gridcolor="#1e2a3a", zeroline=False,
                               tickfont=dict(size=9, color="#ffffff"), ticksuffix="%",
                               range=[0, max(110, ho_df["handover_pct"].max() * 1.15)]),
                    showlegend=False,
                )
                st.plotly_chart(fig_ho, use_container_width=True, config={"displayModeBar": False})

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

        # ── Alert drilldown ───────────────────────────────────────────────────
        _alert_drill = st.session_state.get("alert_type_drill")
        if _alert_drill and _alert_drill.get("error_type"):
            _dt = _alert_drill["error_type"]
            _cl = _alert_drill.get("col", "")
            _dl_label = f"🔍 Alert Drilldown — {_dt}" + (f" · {_cl}" if _cl else "")
            st.markdown(
                f"<div class='kpi-heading-center' style='margin-top:8px; margin-bottom:0.3rem;'>"
                f"{_dl_label}</div>",
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

        csv_data = summary_data.copy() if 'summary_data' in dir() else data.copy()
        csv_data["shop_name"] = csv_data["shop_code"].map(load_shop_name_map()).fillna(csv_data["shop_code"])

    with tab_eda:
        render_inv_eda_analytics(data, start_date, end_date)


if __name__ == "__main__":
    main()
