import streamlit as st
import pandas as pd
import plotly.express as px
import psycopg2
from psycopg2.extras import RealDictCursor
from numerize.numerize import numerize
from datetime import datetime, timedelta

# ====================== MELCOM THEME ======================
MELCOM_BLUE = "#002c6d"
MELCOM_RED = "#ed1b24"
MELCOM_LIGHT = "#f3f6fa"
MELCOM_GRAY = "#f4f4f4"
MELCOM_DARK = "#1b263b"
HILITE_GREEN = "#43aa8b"
HILITE_ORANGE = "#ffa600"

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="PI Dashboard",
    page_icon="https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg",
    layout="wide",
)

# ====================== CUSTOM STYLE ======================
st.markdown(f"""
<style>
.dashboard-header {{
    background: linear-gradient(90deg, {MELCOM_BLUE} 60%, {MELCOM_GRAY} 100%);
    padding: 1rem 0.5rem;
    border-radius: 10px;
    text-align: center;
    font-size: 1.8rem;
    font-weight: 700;
    color: white;
    margin-bottom: 1rem;
}}
.metric-card {{
  border: 2px solid #4a90e2; 
  border-radius: 8px;
  background-color: #f3f6fa;
  padding: 8px 10px;
  box-shadow: 1px 1px 8px rgba(0,0,0,0.08);
  text-align: center;
  font-size: 0.85rem;
  color: #1b263b;
  user-select: none;
  max-width: 180px;
  min-width: 50px;
  aspect-ratio: 4 / 3;
  display: flex;
  flex-direction: column;
  justify-content: center;
  transition: box-shadow 0.3s ease;
}}
.metric-card.chart {{
    max-width: 700px;
    width: 95vw;
    margin: 12px auto 18px auto;
    padding: 18px 20px;
    min-width: 320px;
}}
.metric-card:hover {{
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}}
.metric-card .metric-title {{
    font-weight: 600;
    margin-bottom: 1px;
    font-size: 1rem;
    color: {MELCOM_BLUE};
}}
.metric-card .metric-value {{
    font-weight: 700;
    font-size: 1.3rem;
    color: {MELCOM_DARK};
}}
.metric-card.red {{
    background-color: {MELCOM_RED};
    color: white;
}}
.metric-card.red .metric-title,
.metric-card.red .metric-value {{
    color: white;
}}
.metric-card.dm {{
    background-color: {MELCOM_GRAY};
    color: {MELCOM_RED};
}}
.metric-table {{
    font-size: 0.75rem !important;
    line-height: 1.1 !important;
}}
.stDataFrame, .streamlit-expanderHeader {{
    font-size: 0.75rem !important;
}}
.stDataFrame table th, .stDataFrame table td {{
    padding: 40px 6px !important;
}}
</style>
""", unsafe_allow_html=True)

# ====================== HELPERS ======================
def safe_num(x):
    try:
        return float(x if not hasattr(x, "item") else x.item())
    except:
        return 0.0

def clean_loc(x):
    return str(x).strip().upper() if x else ""

def parse_date(v):
    if pd.isna(v):
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(v).strip(), fmt)
        except ValueError:
            continue
    return None

def calc_summary(data):
    add = safe_num(data["ADDITION"].sum()) if "ADDITION" in data else 0.0
    red = safe_num(data["REDUCE"].sum()) if "REDUCE" in data else 0.0
    return add, red, add - red

# ====================== DB CONNECTION ======================
def get_connection():
    return psycopg2.connect(
        host="localhost",
        user="postgres",
        password="hello",
        database="pi_dashboard",
        port=3307
    )

def check_user(employee_id, password):
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM users WHERE employee_id=%s AND password=%s", (employee_id, password))
        user = cur.fetchone()
        cur.close()
        conn.close()
        return user is not None
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

# ====================== DATA FETCH ======================
@st.cache_data(ttl=600)
def get_data():
    try:
        conn = get_connection()
        df = pd.read_sql("SELECT * FROM pi_main", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return pd.DataFrame()

# ====================== DASHBOARD ======================
def dashboard():
    st.markdown(f"<div class='dashboard-header'><h1 class='dashboard-title'>📊 Melcom PI Dashboard</h1></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='user-welcome'>👤 User: {st.session_state.get('employee_id', '')}</div>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown(f"👋 <span style='color:{MELCOM_BLUE}; font-weight:bold;'>Welcome {st.session_state.get('employee_id', '')}</span>", unsafe_allow_html=True)
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.success("Logged out successfully.")
            st.rerun()
        st.markdown("### ⚙️ Cache Control")
        if st.button("🔄 Clear Cache"):
            st.cache_data.clear()
            st.success("✅ Cache cleared!")
        st.markdown("### 📅 Filters")

        df = get_data()
        if df.empty:
            st.warning("⚠️ No data found in pi_main.")
            st.stop()

        df["TRAN_DATE"] = df["TRAN_DATE"].apply(parse_date)
        df = df.dropna(subset=["TRAN_DATE"])
        df["LOCATION"] = df["LOCATION"].apply(clean_loc)
        for col in ["ADD_QTY", "SUB_QTY", "ADDITION", "REDUCE"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        start = st.date_input("Start Date", df["TRAN_DATE"].min().date())
        end = st.date_input("End Date", df["TRAN_DATE"].max().date())
        filtered = df[(df["TRAN_DATE"] >= pd.to_datetime(start)) & (df["TRAN_DATE"] <= pd.to_datetime(end))]

    # Split damage and non-damage
    dmg_df = filtered[filtered["LOCATION"].isin(["DM", "EX"])]
    nodmg_df = filtered[~filtered["LOCATION"].isin(["DM", "EX"])]

    # Summaries
    total_add, total_red, total_adj = calc_summary(filtered)
    add_nodmg, red_nodmg, adj_nodmg = calc_summary(nodmg_df)
    add_dmg, red_dmg, adj_dmg = calc_summary(dmg_df)

    # ======== METRIC CARDS + LAST 30 DAYS CHART ROW =========
    c3, c4, c5, chart_col = st.columns([1, 1, 1, 2], gap="small")

    with c3:
        st.markdown(f"""
        <div class="metric-card" style= margin-left:1%;margin-top:1%;>
        <div class="metric-title">Total Adjustment</div>
        <div>Add: {numerize(total_add)}</div>
        <div>Reduce: {numerize(total_red)}</div>
        <div>Adj: {numerize(total_adj)}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card" style= margin-left:-47%;margin-top:1%;>
          <div class="metric-title">🧾 Without Damage & Expiry</div>
          <div>Add: {numerize(add_nodmg)}</div>
          <div>Reduce: {numerize(red_nodmg)}</div>
          <div>Adj: {numerize(adj_nodmg)}</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-card dm" style= margin-left:-96%;margin-top:1%;>
          <div class="metric-title">⚠️ Only Damage & Expiry</div>
          <div>Add: {numerize(add_dmg)}</div>
          <div>Reduce: {numerize(red_dmg)}</div>
          <div>Adj: {numerize(adj_dmg)}</div>
        </div>""", unsafe_allow_html=True)

    with chart_col:
        st.markdown("#### Last 30 Days Total Adjustment")
        filter_col, _ = st.columns([1, 8])
        with filter_col:
            type_selected = st.selectbox(
                "",
                ["Warehouse", "Shop"],
                index=0,
                key="wh_shop_filter"
            )
        if type_selected == "Warehouse":
            type_col = "TYPE"
            type_val = "WH"
        else:
            type_col = "TYPE"
            type_val = "SHOP"
        today = datetime.now()
        thirty_days_ago = today - timedelta(days=30)
        filtered_30d = filtered[
            (filtered["TRAN_DATE"] >= thirty_days_ago) &
            (filtered["TRAN_DATE"] <= today) &
            (filtered[type_col].str.upper() == type_val)
        ].copy()
        filtered_30d["Total Adjustment"] = filtered_30d["ADDITION"] - filtered_30d["REDUCE"]
        by_day = filtered_30d.groupby(filtered_30d["TRAN_DATE"].dt.date)["Total Adjustment"].sum().reset_index()
        by_day["LABEL_DATE"] = pd.to_datetime(by_day["TRAN_DATE"]).dt.strftime("%d-%b")
        fixed_width = 700
        fig = px.bar(
            by_day,
            x="LABEL_DATE",
            y="Total Adjustment",
            title="",
            text="Total Adjustment",
            width=fixed_width
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            plot_bgcolor=MELCOM_LIGHT,
            paper_bgcolor=MELCOM_LIGHT,
            margin=dict(t=40, r=0, l=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ========== Department & Ops Manager Summaries ==========
    st.markdown("---")
    col_dept, col_ops = st.columns([1, 1], gap="small")

    with col_dept:
        if "DEPT" in filtered.columns:
            dept_df = filtered.groupby("DEPT", dropna=False).agg({
                "ADD_QTY": "sum", "SUB_QTY": "sum", "ADDITION": "sum", "REDUCE": "sum"
            }).fillna(0)
            dept_df["Total Adjustment"] = dept_df["ADDITION"] - dept_df["REDUCE"]
            display_df = dept_df.copy()
            display_df["ADDITION"] /= 1_000_000
            display_df["REDUCE"] /= 1_000_000
            display_df["ADD_QTY"] /= 1_000
            display_df["SUB_QTY"] /= 1_000
            display_df["Total Adjustment"] /= 1_000_000
            st.subheader("🏬 Department Summary")
            st.dataframe(display_df.style.format({
                "ADDITION": "{:,.2f} M",
                "REDUCE": "{:,.2f} M",
                "ADD_QTY": "{:,.2f} K",
                "SUB_QTY": "{:,.2f} K",
                "Total Adjustment": "{:,.2f} M"
            }), use_container_width=False, width=700)

    with col_ops:
        if "Ops Manager" in filtered.columns:
            ops_df = filtered.groupby("Ops Manager", dropna=False).agg({
                "ADDITION": "sum", "REDUCE": "sum"
            }).fillna(0)
            ops_df["Total Adjustment"] = ops_df["ADDITION"] - ops_df["REDUCE"]
            display_ops = ops_df / 1_000_000
            st.subheader("👷 Ops Manager Summary")
            st.dataframe(display_ops.style.format({
                "ADDITION": "{:,.2f} M",
                "REDUCE": "{:,.2f} M",
                "Total Adjustment": "{:,.2f} M"
            }), use_container_width=False)

    # ========== Monthly Chart ==========
    filtered["MONTH_PERIOD"] = filtered["TRAN_DATE"].dt.to_period("M")
    monthly = filtered.groupby("MONTH_PERIOD")[["ADDITION", "REDUCE"]].sum().reset_index()
    monthly["Total_Adj"] = monthly["ADDITION"] - monthly["REDUCE"]
    monthly["MONTH"] = monthly["MONTH_PERIOD"].dt.strftime("%b'%y")
    monthly = monthly.sort_values("MONTH_PERIOD")
    fig = px.bar(
        monthly,
        x="MONTH",
        y="Total_Adj",
        text=monthly["Total_Adj"].apply(numerize),
        title="📈 Total Adjustment by Month",
        color_discrete_sequence=[MELCOM_BLUE]
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(plot_bgcolor=MELCOM_LIGHT, paper_bgcolor=MELCOM_LIGHT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<hr><center style='color:{MELCOM_BLUE}'>© 2025 Melcom PI Dashboard</center>", unsafe_allow_html=True)

# ====================== MAIN ======================
def login_page():
    st.markdown(f"""
        <div style='text-align:center;padding-top:40px;background:{MELCOM_LIGHT};border-radius:12px;'>
            <img src="https://melcom.com/media/favicon/stores/1/faviconn_162_x_184px.jpg" width="90">
            <h2 style='margin-top:10px;color:{MELCOM_BLUE};'>Melcom PI Dashboard Login</h2>
        </div>
    """, unsafe_allow_html=True)
    with st.form("login_form", clear_on_submit=False):
        employee_id = st.text_input("👤 employee_id")
        password = st.text_input("🔑 Password", type="password")
        submit = st.form_submit_button("Login")
    if submit:
        if check_user(employee_id, password):
            st.session_state["logged_in"] = True
            st.session_state["employee_id"] = employee_id
            st.success("✅ Login successful! Loading dashboard...")
            st.rerun()
        else:
            st.error("❌ Invalid employee_id or password")

if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    login_page()
else:
    dashboard()
