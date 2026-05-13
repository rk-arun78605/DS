from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta
from functools import lru_cache
import re

import pandas as pd
import psycopg2
import psycopg2.pool
from shiny import App, reactive, render, ui


DB_CONFIG = {
    "host": "localhost",
    "port": 3307,
    "user": "postgres",
    "password": "hello",
    "dbname": "WH",
}


IMPLEMENTED_SHOPS = [
    "ACH", "AFI", "AFL", "AMA", "ASH", "EL2", "ELS", "FAR", "GBA", "HAA", "HAM", "KA2",
    "KAS", "KS2", "KS3", "KS4", "KS5", "KS7", "KS8", "KSI", "KSO", "KSS", "LCC", "LFS", "M01", "M03",
    "M06", "M07", "MAS", "MDN", "MM1", "MM2", "MM3", "MSS", "NAN", "OLE", "SPN", "TMP", "WHL",
]


_pool: psycopg2.pool.SimpleConnectionPool | None = None


def get_connection_pool() -> psycopg2.pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            dbname=DB_CONFIG["dbname"],
        )
    return _pool


@contextmanager
def get_db_connection():
    pool = get_connection_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


@lru_cache(maxsize=1)
def load_shop_name_map() -> dict[str, str]:
    def clean_shop_name(name: str) -> str:
        text = str(name or "").strip()
        text = re.sub(r"^\s*melcom\s+", "", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT shop_code, shop_description FROM shopmgrname WHERE is_current = TRUE"
            )
            return {row[0]: clean_shop_name(row[1]) for row in cur.fetchall()}


def load_shopwise_test_bill_analysis(start_date: date, end_date: date) -> pd.DataFrame:
    query = """
    WITH shops AS (
        SELECT unnest(%(shops)s::text[]) AS shop_code
    ),
    erp_rows AS (
        SELECT
            UPPER(TRIM(e.store_code)) AS shop_code,
            UPPER(TRIM(COALESCE(e.cashier, ''))) AS cashier_name,
            NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '')::int AS till_no,
            COALESCE(e.amt, 0)::numeric AS amt
        FROM erpdata e
        WHERE e.invdate::date BETWEEN %(s)s AND %(e)s
          AND UPPER(TRIM(e.store_code)) = ANY(%(shops)s)
          AND NULLIF(TRIM(COALESCE(e.cashier, '')), '') IS NOT NULL
          AND NULLIF(REGEXP_REPLACE(COALESCE(e.tillno::text, ''), '[^0-9]', '', 'g'), '') IS NOT NULL
          AND NULLIF(TRIM(COALESCE(e.invno::text, '')), '') IS NOT NULL
    ),
    erp_sessions AS (
        SELECT
            shop_code,
            cashier_name,
            till_no,
            MAX(CASE WHEN amt = 0.01 THEN 1 ELSE 0 END) AS has_test_bill
        FROM erp_rows
        GROUP BY shop_code, cashier_name, till_no
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
            COUNT(*) FILTER (WHERE amt = 0.01)::bigint AS test_bills_generated
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
    manager_agg AS (
        SELECT
            UPPER(TRIM(m.store_code)) AS shop_code,
            COUNT(m.invno)::bigint AS handover_test_bill_to_manager
        FROM invoices_manager m
        WHERE m.invdate::date BETWEEN %(s)s AND %(e)s
          AND UPPER(TRIM(m.store_code)) = ANY(%(shops)s)
          AND NULLIF(TRIM(COALESCE(m.invno::text, '')), '') IS NOT NULL
        GROUP BY UPPER(TRIM(m.store_code))
    )
    SELECT
        s.shop_code,
        COALESCE(c.cashier_login, 0)::bigint AS cashier_login,
        COALESCE(c.test_bill_not_generated, 0)::bigint AS test_bill_not_generated,
        COALESCE(g.test_bills_generated, 0)::bigint AS test_bills_generated,
        COALESCE(ug.unique_test_bills_generated, 0)::bigint AS unique_test_bills_generated,
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
        COALESCE(m.handover_test_bill_to_manager, 0)::bigint AS handover_test_bill_to_manager,
        CASE
            WHEN COALESCE(g.test_bills_generated, 0) > 0
            THEN ROUND((COALESCE(m.handover_test_bill_to_manager, 0)::numeric / g.test_bills_generated) * 100, 2)
            ELSE 0
        END AS bill_handover_pct,
        (COALESCE(g.test_bills_generated, 0) - COALESCE(m.handover_test_bill_to_manager, 0))::bigint AS missing_test_bill
    FROM shops s
    LEFT JOIN cashier_agg c ON c.shop_code = s.shop_code
    LEFT JOIN generated_agg g ON g.shop_code = s.shop_code
    LEFT JOIN unique_generated_agg ug ON ug.shop_code = s.shop_code
    LEFT JOIN manager_agg m ON m.shop_code = s.shop_code
    ORDER BY s.shop_code
    """

    with get_db_connection() as conn:
        return pd.read_sql(
            query,
            conn,
            params={
                "s": start_date,
                "e": end_date,
                "shops": sorted(IMPLEMENTED_SHOPS),
            },
        )


def load_shopwise_test_bill_drilldown(shop_code: str, start_date: date, end_date: date) -> pd.DataFrame:
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
            params={"s": start_date, "e": end_date, "shop": shop_code.strip().upper()},
        )


def normalize_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def format_integer(value) -> str:
    return f"{int(float(value)):,}"


def format_percent(value) -> str:
    return f"{float(value):.2f}%"


def build_summary_display(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    shop_name_map = load_shop_name_map()
    display_df = raw_df.copy()
    display_df["shop_name"] = display_df["shop_code"].map(shop_name_map).fillna(display_df["shop_code"])
    display_df = display_df.rename(
        columns={
            "shop_code": "Shop Code",
            "shop_name": "Shop Name",
            "cashier_login": "Cashier Login",
            "test_bills_generated": "Cashier Generated Test Bills",
            "test_bill_not_generated": "Cashier Not Generated Test Bills",
            "unique_test_bills_generated": "Cashier Unique Test Bills",
            "handover_test_bill_to_manager": "Unique Bill Handover to Manager",
            "missing_test_bill": "Missing test bill",
            "not_generated_bill_pct": "Not Generated Bill %",
            "generated_test_bill_pct": "Generated Test Bill %",
            "bill_handover_pct": "Bill handover %",
        }
    )

    ordered_cols = [
        "Shop Code",
        "Shop Name",
        "Cashier Login",
        "Cashier Generated Test Bills",
        "Cashier Not Generated Test Bills",
        "Cashier Unique Test Bills",
        "Unique Bill Handover to Manager",
        "Missing test bill",
        "Not Generated Bill %",
        "Generated Test Bill %",
        "Bill handover %",
    ]
    display_df = display_df[ordered_cols]

    totals = {
        "Shop Code": "TOTAL",
        "Shop Name": "ALL SHOPS",
        "Cashier Login": float(pd.to_numeric(display_df["Cashier Login"], errors="coerce").fillna(0).sum()),
        "Cashier Generated Test Bills": float(pd.to_numeric(display_df["Cashier Generated Test Bills"], errors="coerce").fillna(0).sum()),
        "Cashier Not Generated Test Bills": float(pd.to_numeric(display_df["Cashier Not Generated Test Bills"], errors="coerce").fillna(0).sum()),
        "Cashier Unique Test Bills": float(pd.to_numeric(display_df["Cashier Unique Test Bills"], errors="coerce").fillna(0).sum()),
        "Unique Bill Handover to Manager": float(pd.to_numeric(display_df["Unique Bill Handover to Manager"], errors="coerce").fillna(0).sum()),
        "Missing test bill": float(pd.to_numeric(display_df["Missing test bill"], errors="coerce").fillna(0).sum()),
    }
    totals["Not Generated Bill %"] = (
        (totals["Cashier Not Generated Test Bills"] / totals["Cashier Login"]) * 100 if totals["Cashier Login"] > 0 else 0
    )
    totals["Generated Test Bill %"] = (
        (totals["Cashier Unique Test Bills"] / totals["Cashier Login"]) * 100 if totals["Cashier Login"] > 0 else 0
    )
    totals["Bill handover %"] = (
        (totals["Unique Bill Handover to Manager"] / totals["Cashier Generated Test Bills"]) * 100
        if totals["Cashier Generated Test Bills"] > 0 else 0
    )

    display_df = pd.concat([display_df, pd.DataFrame([totals])], ignore_index=True)

    pct_cols = {"Not Generated Bill %", "Generated Test Bill %", "Bill handover %"}
    numeric_cols = {
        "Cashier Login",
        "Cashier Generated Test Bills",
        "Cashier Not Generated Test Bills",
        "Cashier Unique Test Bills",
        "Unique Bill Handover to Manager",
        "Missing test bill",
    }
    for col in display_df.columns:
        if col in pct_cols:
            display_df[col] = display_df[col].map(format_percent)
        elif col in numeric_cols:
            display_df[col] = display_df[col].map(format_integer)

    return display_df


def build_drilldown_display(raw_df: pd.DataFrame, shop_name_map: dict[str, str]) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=["Shop Name", "Cashier name", "InvDate", "InvNo", "Amt", "Handed Over"])

    display_df = raw_df.copy()
    display_df["shop_name"] = display_df["shop_code"].map(shop_name_map).fillna(display_df["shop_code"])
    display_df = display_df.rename(
        columns={
            "shop_name": "Shop Name",
            "cashier_name": "Cashier name",
            "invdate": "InvDate",
            "invno": "InvNo",
            "amt": "Amt",
            "handed_over_to_manager": "Handed Over",
        }
    )[["Shop Name", "Cashier name", "InvDate", "InvNo", "Amt", "Handed Over"]]
    display_df["InvDate"] = pd.to_datetime(display_df["InvDate"]).dt.strftime("%Y-%m-%d")
    display_df["Amt"] = pd.to_numeric(display_df["Amt"], errors="coerce").fillna(0).map(lambda x: f"{x:,.2f}")
    return display_df


default_end_date = date.today() - timedelta(days=1)

app_ui = ui.page_fluid(
    ui.tags.style(
        """
        .app-shell { max-width: 1600px; margin: 0 auto; padding: 24px; }
        .app-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
        .app-subtitle { color: #475569; margin-bottom: 1rem; }
        .section-note { color: #64748b; margin-top: 0.5rem; margin-bottom: 1rem; }
        """
    ),
    ui.div(
        {"class": "app-shell"},
        ui.div("Invoice Scanning Dashboard", class_="app-title"),
        ui.div(
            "Shiny port of the Shop-wise Test Bill analysis and drilldown from the Streamlit dashboard.",
            class_="app-subtitle",
        ),
        ui.layout_columns(
            ui.input_date_range(
                "date_range",
                "Date Range",
                start=default_end_date,
                end=default_end_date,
            ),
            ui.input_action_button("refresh", "Refresh Data"),
            ui.output_text("period_text"),
            col_widths=[5, 2, 5],
        ),
        ui.hr(),
        ui.h4("Shop wise Test Bill analysis"),
        ui.div(
            "Generated Test Bill % = Cashier Unique Test Bills / Cashier Login x 100. Bill handover % = Handed over invoices / Generated test bills x 100.",
            class_="section-note",
        ),
        ui.output_data_frame("summary_table"),
        ui.hr(),
        ui.layout_columns(
            ui.input_selectize("selected_shop", "Shop Code Drilldown", choices=[]),
            ui.output_text("drilldown_title"),
            col_widths=[4, 8],
        ),
        ui.output_data_frame("drilldown_table"),
        ui.hr(),
        ui.p("Run this app with: shiny run --reload invoicescanning/testshiny.py"),
        ui.p("Alternative: python -m shiny run --reload invoicescanning/testshiny.py"),
    ),
)


def server(input, output, session):
    @reactive.calc
    def selected_dates() -> tuple[date, date]:
        values = input.date_range()
        if not values or len(values) != 2:
            return default_end_date, default_end_date
        start_date = normalize_date(values[0])
        end_date = normalize_date(values[1])
        if start_date > end_date:
            return end_date, start_date
        return start_date, end_date

    @reactive.calc
    def summary_raw() -> pd.DataFrame:
        input.refresh()
        start_date, end_date = selected_dates()
        return load_shopwise_test_bill_analysis(start_date, end_date)

    @reactive.effect
    def _update_shop_choices() -> None:
        df = summary_raw()
        choices = [""] + sorted(df["shop_code"].dropna().astype(str).unique().tolist())
        selected = input.selected_shop() or ""
        if selected not in choices:
            selected = ""
        ui.update_selectize("selected_shop", choices=choices, selected=selected, session=session)

    @reactive.calc
    def drilldown_raw() -> pd.DataFrame:
        input.refresh()
        selected_shop = (input.selected_shop() or "").strip().upper()
        if not selected_shop:
            return pd.DataFrame(columns=["shop_code", "cashier_name", "invdate", "invno", "amt", "handed_over_to_manager"])
        start_date, end_date = selected_dates()
        return load_shopwise_test_bill_drilldown(selected_shop, start_date, end_date)

    @output
    @render.text
    def period_text() -> str:
        start_date, end_date = selected_dates()
        return f"Selected period: {start_date.isoformat()} to {end_date.isoformat()}"

    @output
    @render.text
    def drilldown_title() -> str:
        selected_shop = (input.selected_shop() or "").strip().upper()
        if not selected_shop:
            return "Select a shop to view drilldown"
        start_date, end_date = selected_dates()
        shop_name = load_shop_name_map().get(selected_shop, selected_shop)
        return f"Drilldown: {selected_shop} - {shop_name} | {start_date.isoformat()} to {end_date.isoformat()}"

    @output
    @render.data_frame
    def summary_table():
        return render.DataGrid(build_summary_display(summary_raw()), filters=True, width="100%")

    @output
    @render.data_frame
    def drilldown_table():
        df = drilldown_raw()
        if df.empty:
            info_df = pd.DataFrame({"Info": ["Select a shop code to load test-bill drilldown rows."]})
            return render.DataGrid(info_df, width="100%")
        return render.DataGrid(build_drilldown_display(df, load_shop_name_map()), filters=True, width="100%")


app = App(app_ui, server)