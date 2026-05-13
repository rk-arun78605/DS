"""
sn_sync.py — Serial Number Tracking: network-share sync + newsletter
=====================================================================
• find_sn_file(date)        → locate shop_serial_no CSV on //10.10.0.30/mis
• date_exists_in_db(date)   → check if WH already has data for that date
• upload_sn_file(date)      → delete-then-insert into serialno_check_yes_no
• build_sn_newsletter_html  → rich HTML email (email-safe <td> bars)
• send_sn_newsletter        → send via mail.melcomgroup.com:25
"""

from __future__ import annotations

import os
import re
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SHARE_PATH     = r"\\10.10.0.30\mis"
FILE_PREFIX    = "shop_serial_no - "

DB_CONFIG = dict(host="localhost", port=3307, user="postgres",
                 password="hello", database="WH")

SMTP_HOST      = "mail.melcomgroup.com"
SMTP_PORT      = 25
NL_FROM        = "mis.manager@melcomgroup.com"
NL_TO          = "mis.manager@melcomgroup.com"
MELCOM_LOGO    = "https://melcom.com/skin/frontend/melcom/default/images/logo.png"
DASHBOARD_URL  = os.getenv("SN_DASHBOARD_URL", "http://10.10.0.30:8503")

SN_DB_COLS     = ["item_code","item_name","serial_number","shop_code",
                  "bill_no","bill_date","till_number","cashier_name","serial_check"]

# Flexible CSV → DB column mapping
# Note: _normalize_col() fixes the "coode"→"code" typo before matching,
# so "ITEM_COODE" → "ITEM_CODE" automatically. Listed here for clarity.
_COL_ALTS = {
    "ITEM_CODE":     ["ITEM_CODE","ITEM_COODE","ITEMCODE","ITEM CODE","ITEM COODE","PRODUCT_CODE","PRDCODE"],
    "ITEM_NAME":     ["ITEM_NAME","ITEMNAME","ITEM NAME","PRODUCT_NAME","DESCRIPTION","PRODUCT DESC"],
    "SERIAL_NUMBER": ["SERIAL_NUMBER","SERIALNO","SERIAL_NO","SERIAL NO","SERIAL NUMBER","BARCODE","SERIAL"],
    "SHOP_CODE":     ["SHOP_CODE","SHOPCODE","SHOP CODE","STORE_CODE","STORE CODE","LOCATION","SHOP"],
    "BILL_NO":       ["BILL_NO","BILLNO","BILL NO","INVOICE_NO","INVNO","TRANS_NO","RECEIPT_NO"],
    "BILL_DATE":     ["BILL_DATE","BILLDATE","BILL DATE","INVOICE_DATE","TRANS_DATE","DATE","INVDATE"],
    "TILL_NUMBER":   ["TILL_NUMBER","TILLNO","TILL NO","TILL","POS_NO","TILL_NO","TILL NUMBER"],
    "CASHIER_NAME":  ["CASHIER_NAME","CASHIER","OPERATOR","STAFF_NAME","STAFF NAME","USER_NAME","CASHIER NAME"],
    "SERIAL_CHECK":  ["SERIAL_CHECK","SERIALCHECK","STATUS","CHECK","VERIFIED","SERIAL CHECK"],
}


# ─────────────────────────────────────────────────────────────
# FILE DETECTION
# ─────────────────────────────────────────────────────────────

def find_sn_file(target_date: datetime.date) -> str | None:
    """Return full path of shop_serial_no CSV for target_date, or None."""
    for fmt in ["%d-%b-%y", "%d-%B-%y", "%d-%b-%Y", "%d-%B-%Y"]:
        ds   = target_date.strftime(fmt).upper()
        path = os.path.join(SHARE_PATH, f"{FILE_PREFIX}{ds}.csv")
        if os.path.exists(path):
            return path
    # Fallback: glob for any file matching date digits
    day_str = target_date.strftime("%d")
    mon_str = target_date.strftime("%b").upper()
    try:
        for f in os.listdir(SHARE_PATH):
            if FILE_PREFIX.lower() in f.lower() and day_str in f and mon_str in f.upper():
                return os.path.join(SHARE_PATH, f)
    except OSError:
        pass
    return None


def date_exists_in_db(target_date: datetime.date) -> bool:
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM serialno_check_yes_no WHERE DATE(bill_date) = %s",
                (target_date,)
            )
            return int(cur.fetchone()[0] or 0) > 0
    except Exception:
        return False
    finally:
        try: conn.close()
        except: pass


# ─────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────

def _normalize_col(name: str) -> str:
    """Normalize a column name: strip, upper, replace spaces/special chars,
    fix the 'coode' → 'code' typo that appears in some Melcom CSV exports."""
    s = str(name).strip().upper()
    s = s.replace("COODE", "CODE")          # fix typo "Item coode"
    s = re.sub(r"[^A-Z0-9]+", "_", s)      # non-alphanum → underscore
    return s.strip("_")


def upload_sn_file(target_date: datetime.date, csv_path: str,
                   status_fn=None) -> tuple[bool, str, int]:
    """
    Parse CSV → delete existing rows for target_date → insert new rows.
    Returns (success, message, rows_inserted).
    """
    def _log(m):
        if status_fn: status_fn(m)

    try:
        # ── Adaptive CSV read — same scoring approach as home_dashboard.py ──
        def _compact_header(col_name: str) -> str:
            """Compact header for scoring: lowercase, remove non-alphanum, fix typo."""
            s = str(col_name).lower().strip().replace("coode", "code")
            return re.sub(r"[^a-z0-9]+", "", s)

        _expected_compact = {_compact_header(c) for c in SN_DB_COLS}
        _enc_candidates   = ["utf-8-sig", "utf-8", "cp1252", "latin1", "iso-8859-1"]
        _skip_candidates  = [4, 0, 1, 2, 3, 5]
        _sep_candidates   = [None, ",", ";", "\t", "|"]

        best_df, best_score, best_enc, best_skip, best_sep = None, -1, None, None, None

        for enc in _enc_candidates:
            for skip_val in _skip_candidates:
                for sep in _sep_candidates:
                    try:
                        _df = pd.read_csv(
                            csv_path, encoding=enc, skiprows=skip_val,
                            sep=sep, engine="python", on_bad_lines="skip", dtype=str,
                        )
                        found = {_compact_header(c) for c in _df.columns}
                        score = len(_expected_compact & found)
                        if score > best_score:
                            best_score, best_df = score, _df
                            best_enc, best_skip, best_sep = enc, skip_val, sep
                        if score == len(_expected_compact):
                            break
                    except Exception:
                        continue
                if best_score == len(_expected_compact):
                    break
            if best_score == len(_expected_compact):
                break

        df = best_df
        if df is None or df.empty:
            return False, "Could not read CSV — check file format", 0

        _log(f"📄 Read {len(df):,} rows  enc={best_enc}  skiprows={best_skip}  "
             f"sep={repr(best_sep)}  header_score={best_score}/{len(_expected_compact)}")

        # ── Normalize column names (handles 'Item coode' typo + whitespace) ─
        original_cols = df.columns.tolist()
        df.columns    = [_normalize_col(c) for c in original_cols]
        _log(f"🔍 FILE COLUMNS FOUND (normalized): {list(df.columns)}")

        # ── Flexible column mapping → DB column names ────────────────────────
        rename = {}
        for db_col, alts in _COL_ALTS.items():
            db_norm = _normalize_col(db_col)
            if db_norm in df.columns:
                if db_norm != db_col:
                    rename[db_norm] = db_col
                continue                        # already correct
            for alt in alts:
                alt_n = _normalize_col(alt)
                if alt_n in df.columns:
                    rename[alt_n] = db_col
                    break

        if rename:
            df.rename(columns=rename, inplace=True)
            _log(f"🗺 Renamed: {rename}")

        # ── Verify we have all required columns ──────────────────────────────
        db_cols_upper = [c.upper() for c in SN_DB_COLS]
        missing = [c for c in db_cols_upper if c not in df.columns]
        if missing:
            _log(f"⚠ Missing columns: {missing} — available: {list(df.columns)}")
            # Allow partial — missing cols will insert as NULL

        # ── Parse bill_date ──────────────────────────────────────────────────
        if "BILL_DATE" in df.columns:
            s = df["BILL_DATE"].astype(str).str.strip()
            # Try DD/MM/YYYY first (dayfirst=True), then let pandas guess
            parsed = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
            still_null = parsed.isna() & s.notna() & (s != "nan")
            if still_null.any():
                parsed[still_null] = pd.to_datetime(
                    s[still_null], dayfirst=True, errors="coerce"
                )
            df["BILL_DATE"] = parsed.dt.strftime("%Y-%m-%d")
            valid_dates = df["BILL_DATE"].dropna()
            _log(f"📅 Parsed {len(valid_dates):,} dates — sample: {valid_dates.iloc[0] if len(valid_dates) else 'none'}")

        # ── Filter to target date ─────────────────────────────────────────────
        date_iso = target_date.strftime("%Y-%m-%d")
        if "BILL_DATE" in df.columns:
            df_up = df[df["BILL_DATE"] == date_iso].copy()
            if df_up.empty:
                _log(f"⚠ No rows match {date_iso}. Dates in file: {df['BILL_DATE'].dropna().unique()[:5].tolist()}")
                _log("⬆ Uploading all rows regardless of date")
                df_up = df.copy()
            else:
                _log(f"✅ {len(df_up):,} rows match date {date_iso}")
        else:
            df_up = df.copy()

        df_up = df_up.dropna(how="all")
        df_up = df_up[df_up.astype(str).ne("").any(axis=1)]

        if df_up.empty:
            return False, "No data rows after date filter and cleaning", 0

        # ── DB: delete then insert ────────────────────────────────────────────
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM serialno_check_yes_no WHERE DATE(bill_date) = %s",
                    (target_date,)
                )
                deleted = cur.rowcount
            _log(f"🗑 Deleted {deleted} existing rows for {date_iso}")

            def _clean(v, col_name=""):
                if v is None:
                    return None
                s = str(v).strip()          # always strip whitespace
                if s in ("", "nan", "NaT", "None", "NaN"):
                    return None
                # Normalize serial_check to single char Y/N
                if col_name == "serial_check":
                    su = s.upper()
                    if su.startswith("Y"):
                        return "Y"
                    if su.startswith("N"):
                        return "N"
                return s

            # Build values using actual column names (normalized to UPPER)
            vals = []
            for _, row in df_up.iterrows():
                row_tuple = tuple(
                    _clean(row.get(c.upper()), col_name=c)
                    for c in SN_DB_COLS
                )
                vals.append(row_tuple)

            col_sql = ", ".join(SN_DB_COLS)
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    f"INSERT INTO serialno_check_yes_no ({col_sql}) VALUES %s",
                    vals, page_size=5000
                )
            conn.commit()
            _log(f"✅ Inserted {len(vals):,} rows for {date_iso}")

            # Verify
            conn2 = psycopg2.connect(**DB_CONFIG)
            with conn2.cursor() as cur2:
                cur2.execute(
                    "SELECT COUNT(*) FROM serialno_check_yes_no WHERE DATE(bill_date)=%s",
                    (target_date,)
                )
                db_count = cur2.fetchone()[0]
            conn2.close()
            _log(f"🔎 DB verification: {db_count:,} rows now in table for {date_iso}")

            return True, f"Deleted {deleted}, inserted {len(vals)} rows (DB: {db_count})", len(vals)
        except Exception as e:
            conn.rollback()
            _log(f"❌ DB error: {e}")
            return False, f"DB error: {e}", 0
        finally:
            conn.close()

    except Exception as e:
        _log(f"❌ Error: {e}")
        return False, str(e), 0


# ─────────────────────────────────────────────────────────────
# NEWSLETTER DATA QUERIES
# ─────────────────────────────────────────────────────────────

def _get_conn():
    return psycopg2.connect(**DB_CONFIG)


def query_sn_newsletter_data(target_date: datetime.date) -> dict:
    """Fetch all data needed for the serial tracker newsletter."""
    # If target_date has no data, resolve to the most recent bill_date in the DB.
    try:
        _c = _get_conn()
        with _c.cursor() as _cur:
            _cur.execute(
                "SELECT COUNT(*) FROM serialno_check_yes_no WHERE DATE(bill_date) = %s",
                (target_date,)
            )
            _cnt = int(_cur.fetchone()[0] or 0)
            if _cnt == 0:
                _cur.execute(
                    "SELECT MAX(DATE(bill_date)) FROM serialno_check_yes_no "
                    "WHERE bill_date IS NOT NULL"
                )
                _row = _cur.fetchone()
                if _row and _row[0]:
                    target_date = _row[0]
        _c.close()
    except Exception:
        pass

    yesterday = target_date
    mtd_start = yesterday.replace(day=1).strftime("%Y-%m-%d")
    yest_str  = yesterday.strftime("%Y-%m-%d")
    trend_start = (yesterday - timedelta(days=13)).strftime("%Y-%m-%d")

    conn = _get_conn()
    result = {}
    try:
        with conn.cursor() as cur:

            # ── Yesterday overall compliance ────────────────────────────────
            cur.execute("""
                SELECT COUNT(*) AS total,
                    SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END) AS yes_cnt
                FROM serialno_check_yes_no
                WHERE DATE(bill_date) = %s
            """, (yest_str,))
            r = cur.fetchone()
            y_total = int(r[0] or 0)
            y_yes   = int(r[1] or 0)
            result["yest_total"]      = y_total
            result["yest_yes"]        = y_yes
            result["yest_compliance"] = round(y_yes / y_total * 100, 1) if y_total else 0
            result["yest_date"]       = yesterday.strftime("%d %b %Y")

            # ── MTD overall compliance ──────────────────────────────────────
            cur.execute("""
                SELECT COUNT(*) AS total,
                    SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END) AS yes_cnt
                FROM serialno_check_yes_no
                WHERE DATE(bill_date) BETWEEN %s AND %s
            """, (mtd_start, yest_str))
            r = cur.fetchone()
            m_total = int(r[0] or 0)
            m_yes   = int(r[1] or 0)
            result["mtd_total"]      = m_total
            result["mtd_yes"]        = m_yes
            result["mtd_compliance"] = round(m_yes / m_total * 100, 1) if m_total else 0
            result["mtd_start"]      = yesterday.replace(day=1).strftime("%d %b")

            # ── 14-day daily trend ──────────────────────────────────────────
            cur.execute("""
                SELECT DATE(bill_date) AS dt,
                    COUNT(*) AS total,
                    SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END) AS yes_cnt
                FROM serialno_check_yes_no
                WHERE DATE(bill_date) BETWEEN %s AND %s
                GROUP BY 1 ORDER BY 1
            """, (trend_start, yest_str))
            result["daily_trend"] = [
                {"date": r[0], "total": int(r[1] or 0), "yes": int(r[2] or 0)}
                for r in cur.fetchall()
            ]

            # ── Shop compliance (yesterday) ─────────────────────────────────
            cur.execute("""
                SELECT COALESCE(NULLIF(TRIM(shop_code),''),'Unknown') AS shop_code,
                    COUNT(*) AS total,
                    SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END) AS yes_cnt,
                    ROUND(SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END)
                          ::NUMERIC/NULLIF(COUNT(*),0)*100,1) AS compliance_pct
                FROM serialno_check_yes_no
                WHERE DATE(bill_date) = %s
                GROUP BY 1 ORDER BY compliance_pct ASC NULLS FIRST
            """, (yest_str,))
            result["shop_yest"] = [
                {"shop_code": r[0], "total": int(r[1] or 0),
                 "yes": int(r[2] or 0), "pct": float(r[3] or 0)}
                for r in cur.fetchall()
            ]

            # ── Bottom 5 cashiers — yesterday ───────────────────────────────
            cur.execute("""
                SELECT COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown') AS cashier,
                    COALESCE(NULLIF(TRIM(shop_code),''),'?') AS shop_code,
                    COUNT(*) AS total,
                    SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END) AS yes_cnt,
                    ROUND(SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END)
                          ::NUMERIC/NULLIF(COUNT(*),0)*100,1) AS compliance_pct
                FROM serialno_check_yes_no
                WHERE DATE(bill_date) = %s
                  AND cashier_name IS NOT NULL AND TRIM(cashier_name) <> ''
                GROUP BY 1, 2
                HAVING COUNT(*) >= 3
                ORDER BY compliance_pct ASC NULLS FIRST
                LIMIT 5
            """, (yest_str,))
            result["bottom5_yest"] = [
                {"cashier": r[0], "shop": r[1], "total": int(r[2] or 0),
                 "yes": int(r[3] or 0), "pct": float(r[4] or 0)}
                for r in cur.fetchall()
            ]

            # ── Bottom 5 cashiers — MTD ────────────────────────────────────
            cur.execute("""
                SELECT COALESCE(NULLIF(TRIM(cashier_name),''),'Unknown') AS cashier,
                    COALESCE(NULLIF(TRIM(shop_code),''),'?') AS shop_code,
                    COUNT(*) AS total,
                    SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END) AS yes_cnt,
                    ROUND(SUM(CASE WHEN UPPER(TRIM(serial_check))='Y' THEN 1 ELSE 0 END)
                          ::NUMERIC/NULLIF(COUNT(*),0)*100,1) AS compliance_pct
                FROM serialno_check_yes_no
                WHERE DATE(bill_date) BETWEEN %s AND %s
                  AND cashier_name IS NOT NULL AND TRIM(cashier_name) <> ''
                GROUP BY 1, 2
                HAVING COUNT(*) >= 10
                ORDER BY compliance_pct ASC NULLS FIRST
                LIMIT 5
            """, (mtd_start, yest_str))
            result["bottom5_mtd"] = [
                {"cashier": r[0], "shop": r[1], "total": int(r[2] or 0),
                 "yes": int(r[3] or 0), "pct": float(r[4] or 0)}
                for r in cur.fetchall()
            ]

            # ── Top 10 non-compliant items (yesterday) ─────────────────────
            cur.execute("""
                SELECT COALESCE(NULLIF(TRIM(item_name),''),item_code) AS item_label,
                    COUNT(*) AS total,
                    SUM(CASE WHEN UPPER(TRIM(serial_check))='N' THEN 1 ELSE 0 END) AS fail_cnt
                FROM serialno_check_yes_no
                WHERE DATE(bill_date) = %s
                  AND UPPER(TRIM(serial_check)) = 'N'
                  AND item_name IS NOT NULL AND TRIM(item_name) <> ''
                GROUP BY 1
                ORDER BY fail_cnt DESC LIMIT 8
            """, (yest_str,))
            result["top_fail_items"] = [
                {"name": r[0], "total": int(r[1] or 0), "fail": int(r[2] or 0)}
                for r in cur.fetchall()
            ]

    finally:
        conn.close()
    return result


# ─────────────────────────────────────────────────────────────
# NEWSLETTER HTML BUILDER
# ─────────────────────────────────────────────────────────────

def build_sn_newsletter_html(data: dict) -> str:
    import html as _html

    def _esc(v): return _html.escape(str(v or ""))

    def _bar(pct: float, fill: str, bg: str = "#1e293b") -> str:
        p = max(1, min(99, round(pct)))
        r = 100 - p
        return (
            f'<table width="100%" cellpadding="0" cellspacing="0" border="0">'
            f'<tr>'
            f'<td width="{p}%" bgcolor="{fill}" style="background-color:{fill};'
            f'height:12px;border-radius:3px 0 0 3px;font-size:1px;line-height:1px;">&nbsp;</td>'
            f'<td width="{r}%" bgcolor="{bg}" style="background-color:{bg};'
            f'height:12px;border-radius:0 3px 3px 0;font-size:1px;line-height:1px;">&nbsp;</td>'
            f'</tr></table>'
        )

    def _compliance_color(pct: float) -> str:
        if pct >= 90: return "#22c55e"
        if pct >= 75: return "#f59e0b"
        return "#ef4444"

    def _kpi(icon, label, value, sub, clr, border):
        return (
            f'<td width="25%" style="padding:5px;">'
            f'<table width="100%" cellpadding="0" cellspacing="0">'
            f'<tr><td bgcolor="#1a2744" style="background-color:#1a2744;'
            f'border:1px solid {border};border-left:4px solid {clr};'
            f'border-radius:8px;padding:13px 8px;text-align:center;">'
            f'<div style="font-size:20px;margin-bottom:4px;">{icon}</div>'
            f'<div style="font-size:10px;color:#94a3b8;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:5px;">{label}</div>'
            f'<div style="font-size:19px;font-weight:900;color:{clr};'
            f'font-family:Arial,sans-serif;">{value}</div>'
            f'<div style="font-size:10px;color:#64748b;margin-top:3px;">{sub}</div>'
            f'</td></tr></table></td>'
        )

    def _section_hdr(icon, title, clr):
        return (
            f'<tr><td bgcolor="#111827" style="background-color:#111827;'
            f'border-top:3px solid {clr};padding:11px 22px;">'
            f'<span style="font-size:15px;">{icon}</span>'
            f' <span style="font-size:13px;font-weight:700;color:#f1f5f9;'
            f'font-family:Arial,sans-serif;letter-spacing:0.4px;">{title}</span>'
            f'</td></tr>'
        )

    def _cashier_table(rows: list[dict], title: str) -> str:
        if not rows:
            return (
                f'<tr bgcolor="#0f1829" style="background-color:#0f1829;">'
                f'<td colspan="5" style="padding:12px 16px;font-size:12px;'
                f'color:#475569;font-style:italic;">No data available</td></tr>'
            )
        html = ""
        for i, r in enumerate(rows):
            pct  = float(r.get("pct", 0))
            clr  = _compliance_color(pct)
            bg   = "#0f1829" if i % 2 == 0 else "#111d35"
            delta_vs_target = pct - 90
            delta_txt = (f'+{delta_vs_target:.1f}%' if delta_vs_target >= 0
                         else f'{delta_vs_target:.1f}%')
            delta_clr = "#22c55e" if delta_vs_target >= 0 else "#ef4444"
            html += (
                f'<tr bgcolor="{bg}" style="background-color:{bg};">'
                f'<td style="padding:9px 10px 9px 16px;font-size:12px;color:#e2e8f0;'
                f'font-family:Arial,sans-serif;font-weight:600;">'
                f'{_esc(r.get("cashier",""))}</td>'
                f'<td style="padding:9px 10px;font-size:11px;color:#60a5fa;'
                f'font-family:Arial,sans-serif;text-align:center;font-weight:700;">'
                f'{_esc(r.get("shop",""))}</td>'
                f'<td style="padding:9px 10px;font-size:11px;color:#94a3b8;text-align:center;">'
                f'{int(r.get("total",0)):,}</td>'
                f'<td style="padding:9px 10px;font-size:13px;font-weight:900;'
                f'color:{clr};text-align:center;">{pct:.1f}%</td>'
                f'<td style="padding:9px 16px;font-size:11px;font-weight:700;'
                f'color:{delta_clr};text-align:right;">{delta_txt}</td>'
                f'</tr>'
            )
        return html

    # ── Daily spark ───────────────────────────────────────────────────────────
    daily  = data.get("daily_trend", [])
    recent = daily[-14:] if len(daily) > 14 else daily
    max_t  = max((r["total"] for r in recent), default=1)
    spark  = ""
    for r in recent:
        pct  = round(r["yes"] / max(r["total"], 1) * 100, 1)
        h    = max(3, round(r["total"] / max_t * 55))
        pad  = 55 - h
        clr  = _compliance_color(pct)
        lbl  = pd.Timestamp(r["date"]).strftime("%d") if r.get("date") else ""
        spark += (
            f'<td style="text-align:center;vertical-align:bottom;padding:0 2px;">'
            f'<table cellpadding="0" cellspacing="0" border="0" width="100%">'
            f'<tr><td height="{pad}" style="font-size:1px;line-height:1px;">&nbsp;</td></tr>'
            f'<tr><td height="{h}" bgcolor="{clr}" style="background-color:{clr};'
            f'border-radius:2px 2px 0 0;font-size:1px;line-height:1px;">&nbsp;</td></tr>'
            f'<tr><td style="font-size:8px;color:#64748b;padding-top:2px;'
            f'text-align:center;font-family:Arial,sans-serif;">{lbl}</td></tr>'
            f'</table></td>'
        )
        # Tooltip row: compliance %
        spark += (
            f'<!-- {lbl}: {pct}% -->'
        )

    # ── Shop ranking rows ─────────────────────────────────────────────────────
    shops = data.get("shop_yest", [])
    max_shop_total = max((r["total"] for r in shops), default=1)
    shop_rows_html = ""
    for i, r in enumerate(shops[:12]):
        pct = float(r.get("pct", 0))
        clr = _compliance_color(pct)
        bg  = "#0f1829" if i % 2 == 0 else "#111d35"
        shop_rows_html += (
            f'<tr bgcolor="{bg}" style="background-color:{bg};">'
            f'<td width="20%" style="padding:8px 10px 8px 16px;font-size:12px;'
            f'color:#60a5fa;font-weight:700;">{_esc(r.get("shop_code",""))}</td>'
            f'<td width="44%" style="padding:8px 10px;">{_bar(pct, clr)}</td>'
            f'<td width="16%" style="padding:8px 6px;font-size:13px;font-weight:900;'
            f'color:{clr};text-align:center;">{pct:.1f}%</td>'
            f'<td width="20%" style="padding:8px 16px;font-size:11px;color:#64748b;'
            f'text-align:right;">{int(r.get("total",0)):,} scans</td>'
            f'</tr>'
        )

    # ── Top fail items ────────────────────────────────────────────────────────
    fail_items = data.get("top_fail_items", [])
    max_fail   = max((r["fail"] for r in fail_items), default=1)
    fail_html  = ""
    for i, r in enumerate(fail_items):
        pct = round(r["fail"] / max(r["total"], 1) * 100, 1)
        bg  = "#0f1829" if i % 2 == 0 else "#111d35"
        fail_html += (
            f'<tr bgcolor="{bg}" style="background-color:{bg};">'
            f'<td width="46%" style="padding:8px 10px 8px 16px;font-size:11px;'
            f'color:#e2e8f0;">{_esc(str(r.get("name",""))[:35])}</td>'
            f'<td width="34%" style="padding:8px 10px;">'
            f'{_bar(r["fail"] / max_fail * 100, "#ef4444")}</td>'
            f'<td width="20%" style="padding:8px 16px;font-size:12px;font-weight:800;'
            f'color:#ef4444;text-align:right;">{int(r["fail"])} fails</td>'
            f'</tr>'
        )

    # ── Values ───────────────────────────────────────────────────────────────
    yest_pct = float(data.get("yest_compliance", 0))
    mtd_pct  = float(data.get("mtd_compliance", 0))
    yest_clr = _compliance_color(yest_pct)
    mtd_clr  = _compliance_color(mtd_pct)
    wow_delta = yest_pct - mtd_pct
    wow_clr   = "#22c55e" if wow_delta >= 0 else "#ef4444"
    wow_txt   = f"{'▲' if wow_delta >= 0 else '▼'} {abs(wow_delta):.1f}%"

    # Compliance gauge bar (HTML table-based)
    gauge_pct = max(1, min(99, round(yest_pct)))
    gauge_rem = 100 - gauge_pct
    sent_ts   = datetime.now().strftime("%d %b %Y, %H:%M")
    yest_date = data.get("yest_date", "")
    mtd_start = data.get("mtd_start", "")

    # Shops below 90%
    shops_below_90 = sum(1 for r in shops if float(r.get("pct",0)) < 90)
    shops_ok       = sum(1 for r in shops if float(r.get("pct",0)) >= 95)

    html_out = f"""<!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>News Letter - Serial Number Tracking</title>
</head>
<body style="margin:0;padding:0;background-color:#020817;
             font-family:Arial,'Helvetica Neue',Helvetica,sans-serif;">

<table width="100%" cellpadding="0" cellspacing="0" border="0"
       bgcolor="#020817" style="background-color:#020817;">
<tr><td align="center" style="padding:24px 12px;">

<table width="620" cellpadding="0" cellspacing="0" border="0"
       style="max-width:620px;border-radius:16px;overflow:hidden;
              border:1px solid #1e3a5f;">

  <!-- HEADER -->
  <tr>
    <td bgcolor="#0c1a3a"
        style="background-color:#0c1a3a;
               background-image:linear-gradient(135deg,#0c1a3a 0%,#1e3799 50%,#0c1a3a 100%);
               padding:26px 24px 18px;text-align:center;
               border-bottom:3px solid #9b5bff;">
      <img src="{MELCOM_LOGO}" width="150" alt="Melcom"
           style="display:block;margin:0 auto 12px auto;max-height:50px;width:auto;border:0;">
      <div style="font-size:22px;font-weight:900;color:#ffffff;
                  letter-spacing:-0.3px;font-family:Arial,sans-serif;margin-bottom:5px;">
        Serial Number Tracking Report
      </div>
      <div style="font-size:13px;color:#a78bfa;font-family:Arial,sans-serif;margin-bottom:3px;">
        Yesterday: {yest_date} &nbsp;·&nbsp; MTD from {mtd_start}
      </div>
      <div style="font-size:11px;color:#7c3aed;font-family:Arial,sans-serif;">
        Generated {sent_ts}
      </div>
    </td>
  </tr>

  <!-- YESTERDAY COMPLIANCE BANNER -->
  <tr>
    <td bgcolor="#0d1430" style="background-color:#0d1430;padding:16px 24px;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr>
          <td width="60%" style="vertical-align:middle;">
            <div style="font-size:11px;color:#7c3aed;font-weight:700;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
              Yesterday's Compliance Score
            </div>
            <div style="font-size:42px;font-weight:900;color:{yest_clr};
                        font-family:Arial,sans-serif;line-height:1;">
              {yest_pct:.1f}%
            </div>
            <div style="font-size:12px;color:#64748b;margin-top:4px;">
              {int(data.get("yest_yes",0)):,} compliant of {int(data.get("yest_total",0)):,} scans
            </div>
          </td>
          <td width="40%" style="vertical-align:middle;text-align:center;
                                  border-left:1px solid #1e3a5f;padding-left:16px;">
            <div style="font-size:11px;color:#64748b;font-weight:700;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
              MTD Compliance
            </div>
            <div style="font-size:30px;font-weight:900;color:{mtd_clr};
                        font-family:Arial,sans-serif;">{mtd_pct:.1f}%</div>
            <div style="font-size:13px;color:{wow_clr};font-weight:700;margin-top:4px;">
              {wow_txt} vs MTD avg
            </div>
            <div style="font-size:10px;color:#475569;">
              {int(data.get("mtd_yes",0)):,} / {int(data.get("mtd_total",0)):,} scans
            </div>
          </td>
        </tr>
      </table>
      <!-- Compliance gauge bar -->
      <div style="margin-top:12px;font-size:10px;color:#475569;margin-bottom:4px;">
        Compliance vs 100% target
      </div>
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr>
          <td width="{gauge_pct}%" bgcolor="{yest_clr}"
              style="background-color:{yest_clr};height:10px;
                     border-radius:5px 0 0 5px;font-size:1px;line-height:1px;">&nbsp;</td>
          <td width="{gauge_rem}%" bgcolor="#1e293b"
              style="background-color:#1e293b;height:10px;
                     border-radius:0 5px 5px 0;font-size:1px;line-height:1px;">&nbsp;</td>
        </tr>
      </table>
      <table width="100%" cellpadding="0" cellspacing="0" border="0"
             style="margin-top:8px;">
        <tr>
          <td style="font-size:10px;color:{yest_clr};">
            {'&#9989; Above Target (90%)' if yest_pct >= 90 else '&#9888;&#65039; Below Target (90%)'}
          </td>
          <td style="font-size:10px;color:#475569;text-align:right;">
            Target: 90% &nbsp;|&nbsp; Gap: {abs(yest_pct-90):.1f}%
            {'above' if yest_pct >= 90 else 'below'}
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- KPI CARDS -->
  <tr>
    <td bgcolor="#0a1628" style="background-color:#0a1628;padding:12px 10px 8px;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr>
          {_kpi("&#128201;","Yesterday Scans",
                f"{int(data.get('yest_total',0)):,}","total serials","#60a5fa","#1e3a5f")}
          {_kpi("&#128308;","Non-Compliant",
                f"{int(data.get('yest_total',0))-int(data.get('yest_yes',0)):,}",
                "yesterday","#ef4444","#7f1d1d")}
          {_kpi("&#127978;","Shops Below 90%",
                str(shops_below_90),f"of {len(shops)} shops","#f59e0b","#78350f")}
          {_kpi("&#10003;","Shops ≥ 95%",
                str(shops_ok),"excellent performance","#22c55e","#14532d")}
        </tr>
      </table>
    </td>
  </tr>

  <!-- DAILY TREND SPARK -->
  {_section_hdr("&#128202;","14-Day Compliance Trend (bar height = scan volume)","#9b5bff")}
  <tr>
    <td bgcolor="#0d1526" style="background-color:#0d1526;padding:14px 24px 10px;">
      <table cellpadding="0" cellspacing="0" border="0" width="100%">
        <tr>{spark}</tr>
      </table>
      <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-top:8px;">
        <tr>
          <td style="font-size:10px;color:#22c55e;">&#9646; &#8805;90%</td>
          <td style="font-size:10px;color:#f59e0b;text-align:center;">&#9646; 75-90%</td>
          <td style="font-size:10px;color:#ef4444;text-align:right;">&#9646; &lt;75%</td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- SHOP COMPLIANCE RANKING (YESTERDAY) -->
  {_section_hdr("&#127978;","Shop Compliance — Yesterday (worst first)","#f59e0b")}
  <tr>
    <td bgcolor="#0d1526" style="background-color:#0d1526;padding:0;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr bgcolor="#111827" style="background-color:#111827;">
          <td width="20%" style="padding:7px 10px 7px 16px;font-size:10px;color:#64748b;
              font-weight:700;text-transform:uppercase;letter-spacing:1px;">Shop</td>
          <td width="44%" style="padding:7px 10px;font-size:10px;color:#64748b;
              font-weight:700;text-transform:uppercase;letter-spacing:1px;">Compliance</td>
          <td width="16%" style="padding:7px 6px;font-size:10px;color:#64748b;
              font-weight:700;text-transform:uppercase;text-align:center;">Score</td>
          <td width="20%" style="padding:7px 16px;font-size:10px;color:#64748b;
              font-weight:700;text-transform:uppercase;text-align:right;">Scans</td>
        </tr>
        {shop_rows_html}
      </table>
    </td>
  </tr>

  <!-- BOTTOM 5 CASHIERS — YESTERDAY -->
  {_section_hdr("&#128101;","Bottom 5 Cashiers — Yesterday (min. 3 scans)","#ef4444")}
  <tr>
    <td bgcolor="#0d1526" style="background-color:#0d1526;padding:0;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr bgcolor="#111827" style="background-color:#111827;">
          <td style="padding:7px 10px 7px 16px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;letter-spacing:1px;">Cashier</td>
          <td style="padding:7px 10px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;text-align:center;">Store</td>
          <td style="padding:7px 10px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;text-align:center;">Scans</td>
          <td style="padding:7px 10px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;text-align:center;">Score</td>
          <td style="padding:7px 16px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;text-align:right;">vs 90% Target</td>
        </tr>
        {_cashier_table(data.get("bottom5_yest",[]), "Yesterday")}
      </table>
    </td>
  </tr>

  <!-- BOTTOM 5 CASHIERS — MTD -->
  {_section_hdr("&#128101;",f"Bottom 5 Cashiers — MTD from {mtd_start} (min. 10 scans)","#f97316")}
  <tr>
    <td bgcolor="#0d1526" style="background-color:#0d1526;padding:0;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr bgcolor="#111827" style="background-color:#111827;">
          <td style="padding:7px 10px 7px 16px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;letter-spacing:1px;">Cashier</td>
          <td style="padding:7px 10px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;text-align:center;">Store</td>
          <td style="padding:7px 10px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;text-align:center;">Scans</td>
          <td style="padding:7px 10px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;text-align:center;">Score</td>
          <td style="padding:7px 16px;font-size:10px;color:#64748b;font-weight:700;
              text-transform:uppercase;text-align:right;">vs 90% Target</td>
        </tr>
        {_cashier_table(data.get("bottom5_mtd",[]), "MTD")}
      </table>
    </td>
  </tr>

  <!-- TOP FAILING ITEMS -->
  {_section_hdr("&#128230;","Top Items — Most Non-Compliant Serials (Yesterday)","#a78bfa")}
  <tr>
    <td bgcolor="#0d1526" style="background-color:#0d1526;padding:0;">
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        <tr bgcolor="#111827" style="background-color:#111827;">
          <td width="46%" style="padding:7px 10px 7px 16px;font-size:10px;color:#64748b;
              font-weight:700;text-transform:uppercase;letter-spacing:1px;">Item</td>
          <td width="34%" style="padding:7px 10px;font-size:10px;color:#64748b;
              font-weight:700;text-transform:uppercase;">Failures</td>
          <td width="20%" style="padding:7px 16px;font-size:10px;color:#64748b;
              font-weight:700;text-transform:uppercase;text-align:right;">Count</td>
        </tr>
        {fail_html if fail_html else
         '<tr bgcolor="#0f1829"><td colspan="3" style="padding:12px 16px;font-size:12px;'
         'color:#475569;font-style:italic;">No failing items for this date</td></tr>'}
      </table>
    </td>
  </tr>

  <!-- ACTION REQUIRED -->
  <tr>
    <td bgcolor="#0a1628" style="background-color:#0a1628;padding:16px 24px;
        border-top:1px solid #1e3a5f;">
      <div style="font-size:12px;font-weight:700;color:#f1f5f9;margin-bottom:8px;">
        &#9888;&#65039; Action Required
      </div>
      <table width="100%" cellpadding="0" cellspacing="0" border="0">
        {''.join([
            f'<tr><td style="padding:4px 0;">'
            f'<span style="font-size:11px;color:#fca5a5;">&#9679;</span>'
            f'<span style="font-size:11px;color:#cbd5e1;font-family:Arial,sans-serif;">'
            f' Shop <b style="color:#60a5fa">{r.get("shop_code","?")}</b> — '
            f'{float(r.get("pct",0)):.1f}% compliance, '
            f'{int(r.get("total",0)):,} scans — requires immediate coaching</span>'
            f'</td></tr>'
            for r in shops[:3]
            if float(r.get("pct",0)) < 80
        ]) or '<tr><td style="padding:4px 0;font-size:11px;color:#22c55e;">'
             '&#10003; All top shops above 80% threshold — good standing</td></tr>'}
      </table>
      {''.join([
          f'<table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-top:6px;">'
          f'<tr><td style="padding:4px 0;font-size:11px;color:#cbd5e1;">'
          f'<span style="color:#f97316;">&#9679;</span>'
          f' Cashier <b style="color:#fbbf24">{r.get("cashier","?")}</b>'
          f' [{r.get("shop","?")}] — MTD {float(r.get("pct",0)):.1f}% — persistent underperformer'
          f'</td></tr></table>'
          for r in data.get("bottom5_mtd",[])[:2]
          if float(r.get("pct",0)) < 70
      ])}
    </td>
  </tr>

  <!-- CTA -->
  <tr>
    <td bgcolor="#0a1628" style="background-color:#0a1628;padding:16px 24px 20px;
        text-align:center;border-top:1px solid #1e3a5f;">
      <a href="{DASHBOARD_URL}"
         style="display:inline-block;background-color:#7c3aed;color:#ffffff;
                font-size:14px;font-weight:700;font-family:Arial,sans-serif;
                padding:12px 30px;border-radius:8px;text-decoration:none;
                border:1px solid #9b5bff;letter-spacing:0.3px;">
        &#128279; View Serial Tracker Dashboard
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
        Melcom Group &bull; Serial Tracker &bull; {sent_ts}<br>
        <span style="color:#334155;">Auto-generated after file upload. Do not reply.</span>
      </div>
    </td>
  </tr>

</table>
</td></tr>
</table>
</body>
</html>"""
    return html_out


# ─────────────────────────────────────────────────────────────
# SEND
# ─────────────────────────────────────────────────────────────

def send_sn_newsletter(target_date: datetime.date) -> tuple[bool, str]:
    """Send newsletter with the daily CSV file attached (today's date in filename)."""
    from email import encoders as _enc
    from email.mime.base import MIMEBase

    try:
        # Resolve the actual bill_date present in the DB — the filename date
        # (target_date) may differ from the bill_date inside the CSV.
        nl_date = target_date
        try:
            _c = _get_conn()
            with _c.cursor() as _cur:
                _cur.execute(
                    "SELECT MAX(DATE(bill_date)) FROM serialno_check_yes_no "
                    "WHERE bill_date IS NOT NULL"
                )
                _row = _cur.fetchone()
            _c.close()
            if _row and _row[0]:
                nl_date = _row[0]
        except Exception:
            pass  # fall back to target_date

        data    = query_sn_newsletter_data(nl_date)
        html    = build_sn_newsletter_html(data)
        subject = "News Letter - Serial Number Tracking"

        # Use "mixed" so we can attach files
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"]    = NL_FROM
        msg["To"]      = NL_TO
        msg.attach(MIMEText(html, "html", "utf-8"))

        # ── Attach today's CSV file from network share ───────────────────
        csv_path = find_sn_file(target_date)
        if csv_path and os.path.exists(csv_path):
            try:
                with open(csv_path, "rb") as fh:
                    csv_bytes = fh.read()
                csv_part = MIMEBase("text", "csv")
                csv_part.set_payload(csv_bytes)
                _enc.encode_base64(csv_part)
                csv_filename = os.path.basename(csv_path)
                csv_part.add_header("Content-Disposition",
                                    f'attachment; filename="{csv_filename}"')
                msg.attach(csv_part)
            except Exception as _att_err:
                pass  # send without attachment if file unreadable

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as srv:
            srv.sendmail(NL_FROM, [NL_TO], msg.as_string())
        return True, f"Newsletter sent to {NL_TO}"
    except Exception as e:
        return False, str(e)
