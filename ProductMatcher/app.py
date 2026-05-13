import io
import psycopg2
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from db import (
    get_connection,
    load_itemmaster,
    insert_uploaded_items,
    bulk_insert_itemmaster,
    get_itemmaster_count,
    kill_stale_sessions,
)
from matcher import run_matching, build_index

st.set_page_config(
    page_title="Product Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Kill stale idle connections once per browser session (on every fresh page load / refresh)
if "sessions_cleared" not in st.session_state:
    kill_stale_sessions()
    st.session_state["sessions_cleared"] = True

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif; }
.stApp { background:#0d1117; }
#MainMenu{visibility:hidden;} footer{visibility:hidden;} header{visibility:hidden;}
.stMarkdown p,.stMarkdown li{color:#c9d1d9;}
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4{color:#e6edf3!important;}

.stTextInput>div>div>input{background:#161b22!important;border:1px solid #30363d!important;
  border-radius:10px!important;color:#e6edf3!important;padding:12px 16px!important;font-size:15px!important;}
.stTextInput>div>div>input:focus{border-color:#E31837!important;box-shadow:0 0 0 3px rgba(227,24,55,.15)!important;}
.stTextInput label,.stSelectbox label,.stMultiSelect label{color:#8b949e!important;font-size:12px!important;
  font-weight:600!important;text-transform:uppercase!important;letter-spacing:.5px!important;}

.stSelectbox>div>div{background:#161b22!important;border-color:#30363d!important;border-radius:10px!important;color:#e6edf3!important;}
.stMultiSelect>div>div{background:#161b22!important;border-color:#30363d!important;border-radius:10px!important;}

.stButton>button{background:#21262d!important;color:#e6edf3!important;border:1px solid #30363d!important;
  border-radius:10px!important;font-weight:600!important;transition:all .2s ease!important;}
.stButton>button:hover{background:#30363d!important;border-color:#6e7681!important;transform:translateY(-1px)!important;}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#E31837 0%,#b01028 100%)!important;
  border-color:transparent!important;color:white!important;box-shadow:0 4px 15px rgba(227,24,55,.3)!important;}
.stButton>button[kind="primary"]:hover{box-shadow:0 8px 25px rgba(227,24,55,.5)!important;transform:translateY(-2px)!important;}

.stTabs [data-baseweb="tab-list"]{gap:6px;background:transparent!important;border-bottom:1px solid #30363d;}
.stTabs [data-baseweb="tab-list"] button{background:transparent!important;color:#8b949e!important;
  border:none!important;border-radius:0!important;padding:10px 20px!important;font-weight:500!important;
  border-bottom:2px solid transparent!important;}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"]{color:#E31837!important;
  border-bottom-color:#E31837!important;font-weight:700!important;}

[data-testid="stMetricLabel"]{color:#8b949e!important;font-size:11px!important;
  text-transform:uppercase!important;letter-spacing:.8px!important;}
[data-testid="stMetricValue"]{color:#e6edf3!important;font-weight:800!important;}

.stAlert{border-radius:10px!important;}
.stCheckbox label{color:#e6edf3!important;}
.stCaptionContainer p{color:#6e7681!important;}
hr{border-color:#30363d!important;margin:16px 0!important;}
.stDataFrame{border-radius:12px!important;}
[data-testid="stDataFrame"]{background:#161b22!important;}
.stTextArea textarea{background:#161b22!important;border-color:#30363d!important;
  color:#e6edf3!important;border-radius:10px!important;}
.stProgress>div>div{background-color:#E31837!important;}
[data-testid="stForm"]{background:#161b22;border:1px solid #30363d;border-radius:16px;padding:16px;}
.streamlit-expanderHeader{background:#161b22!important;border-radius:10px!important;color:#e6edf3!important;}
.streamlit-expanderContent{background:#0d1117!important;border-color:#30363d!important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:#0d1117;}
::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:#6e7681;}

.stFileUploader>div{background:#161b22!important;border:1px dashed #30363d!important;
  border-radius:12px!important;color:#8b949e!important;}
.stFileUploader label{color:#8b949e!important;font-size:12px!important;font-weight:600!important;
  text-transform:uppercase!important;letter-spacing:.5px!important;}

.page-header{background:linear-gradient(135deg,#161b22 0%,#1c2128 100%);
  border:1px solid #30363d;border-radius:16px;padding:28px 32px;margin-bottom:24px;
  position:relative;overflow:hidden;}
.page-header::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#E31837,#764ba2);}
.page-header h1{color:#e6edf3;font-size:26px;font-weight:800;margin:0 0 6px 0;}
.page-header p{color:#8b949e;font-size:14px;margin:0;}

.stat-card{background:#161b22;border:1px solid #30363d;border-radius:12px;
  padding:18px 20px;text-align:center;}
.stat-card .val{font-size:28px;font-weight:800;color:#e6edf3;line-height:1;}
.stat-card .lbl{font-size:11px;color:#8b949e;text-transform:uppercase;
  letter-spacing:.8px;margin-top:4px;}
.stat-card.green .val{color:#3fb950;}
.stat-card.blue .val{color:#58a6ff;}
.stat-card.yellow .val{color:#d29922;}
.stat-card.red .val{color:#f85149;}

.section-label{color:#8b949e;font-size:11px;font-weight:600;
  text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _badge(match_type: str) -> str:
    map_ = {
        'EXACT':           ('<span style="background:rgba(63,185,80,.15);color:#3fb950;'
                            'border:1px solid rgba(63,185,80,.3);padding:3px 9px;'
                            'border-radius:20px;font-size:11px;font-weight:700;'
                            'white-space:nowrap;">EXACT</span>'),
        'EXACT_SECONDARY': ('<span style="background:rgba(88,166,255,.15);color:#58a6ff;'
                            'border:1px solid rgba(88,166,255,.3);padding:3px 9px;'
                            'border-radius:20px;font-size:11px;font-weight:700;'
                            'white-space:nowrap;">BARCODE1</span>'),
        'FUZZY':           ('<span style="background:rgba(210,153,34,.15);color:#d29922;'
                            'border:1px solid rgba(210,153,34,.3);padding:3px 9px;'
                            'border-radius:20px;font-size:11px;font-weight:700;'
                            'white-space:nowrap;">FUZZY</span>'),
        'NO_MATCH':        ('<span style="background:rgba(248,81,73,.15);color:#f85149;'
                            'border:1px solid rgba(248,81,73,.3);padding:3px 9px;'
                            'border-radius:20px;font-size:11px;font-weight:700;'
                            'white-space:nowrap;">NO MATCH</span>'),
    }
    return map_.get(match_type, match_type)


def _conf_html(conf: str) -> str:
    colors = {'HIGH': '#3fb950', 'MEDIUM': '#d29922', 'LOW': '#f85149'}
    c = colors.get(conf, '#6e7681')
    return f'<span style="color:{c};font-weight:700;font-size:12px;">{conf}</span>' if conf else '—'


def _score_html(match_pct) -> str:
    if match_pct == '' or match_pct is None:
        return '—'
    pct = float(match_pct)
    if pct > 85:
        bar_color = '#3fb950'
    elif pct >= 65:
        bar_color = '#d29922'
    else:
        bar_color = '#f85149'
    width = min(int(pct), 100)
    return (
        f'<div style="display:flex;align-items:center;gap:8px;min-width:90px;">'
        f'<div style="width:52px;height:5px;background:#21262d;border-radius:3px;overflow:hidden;flex-shrink:0;">'
        f'<div style="width:{width}%;height:100%;background:{bar_color};border-radius:3px;"></div></div>'
        f'<span style="color:#e6edf3;font-size:12px;font-weight:600;">{pct:.1f}%</span>'
        f'</div>'
    )


def _esc(text) -> str:
    return str(text or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _remark_html(remark: str) -> str:
    if not remark:
        return '—'
    return (
        f'<span style="background:rgba(210,153,34,.15);color:#d29922;'
        f'border:1px solid rgba(210,153,34,.3);padding:3px 8px;'
        f'border-radius:6px;font-size:11px;font-weight:600;white-space:nowrap;">'
        f'⚠ {_esc(remark)}</span>'
    )


def build_html_table(df: pd.DataFrame) -> str:
    # Detect how many Top Match columns actually exist in this result
    top_match_cols = sorted(
        [c for c in df.columns if c.startswith('Top Match ')],
        key=lambda c: int(c.split()[-1])
    )

    fixed_cols = [
        ('Supplier Description', '220px'),
        ('Weight / Pack',        '110px'),
        ('Input Barcode',        '120px'),
        ('Input Barcode1',       '110px'),
        ('Match Type',           '100px'),
        ('Match %',              '110px'),
        ('Matched Item Name',    '210px'),
        ('Item Code',            '100px'),
        ('Matched Barcode',      '120px'),
        ('Confidence',           '90px'),
        ('Duplicate Item Codes', '180px'),
        ('Remark',               '260px'),
    ]
    cols = fixed_cols + [(c, '260px') for c in top_match_cols]

    header_cells = ''.join(
        f'<th style="min-width:{w};width:{w};">{c}</th>'
        for c, w in cols
    )

    rows_html = ''
    for _, r in df.iterrows():
        has_dup = bool(r.get('Duplicate Item Codes', ''))
        row_bg  = 'background:rgba(210,153,34,.05);' if has_dup else ''
        rows_html += (
            f'<tr style="{row_bg}">'
            f'<td style="color:#e6edf3;font-weight:500;">{_esc(r.get("Supplier Description",""))}</td>'
            f'<td><span style="background:#21262d;color:#8b949e;padding:2px 7px;'
            f'border-radius:6px;font-size:11px;">{_esc(r.get("Weight / Pack","")) or "—"}</span></td>'
            f'<td style="color:#8b949e;font-family:monospace;font-size:12px;">{_esc(r.get("Input Barcode","")) or "—"}</td>'
            f'<td style="color:#8b949e;font-family:monospace;font-size:12px;">{_esc(r.get("Input Barcode1","")) or "—"}</td>'
            f'<td>{_badge(r.get("Match Type",""))}</td>'
            f'<td>{_score_html(r.get("Match %",""))}</td>'
            f'<td style="color:#e6edf3;font-weight:500;">{_esc(r.get("Matched Item Name","")) or "—"}</td>'
            f'<td style="color:#58a6ff;font-family:monospace;font-size:12px;">{_esc(r.get("Item Code","")) or "—"}</td>'
            f'<td style="color:#8b949e;font-family:monospace;font-size:12px;">{_esc(r.get("Matched Barcode","")) or "—"}</td>'
            f'<td>{_conf_html(r.get("Confidence",""))}</td>'
            f'<td style="color:#d29922;font-family:monospace;font-size:12px;">{_esc(r.get("Duplicate Item Codes","")) or "—"}</td>'
            f'<td>{_remark_html(r.get("Remark",""))}</td>'
            + ''.join(
                f'<td style="color:#6e7681;font-size:12px;">{_esc(r.get(col,"")) or "—"}</td>'
                for col in top_match_cols
            )
            + '</tr>'
        )

    return f"""
<style>
  .pm-wrap {{overflow-x:auto;border-radius:12px;border:1px solid #30363d;}}
  .pm-table {{width:100%;border-collapse:collapse;font-size:13px;font-family:'Inter',sans-serif;}}
  .pm-table thead tr {{background:#161b22;}}
  .pm-table th {{padding:10px 14px;text-align:left;color:#8b949e;font-size:11px;
    font-weight:600;text-transform:uppercase;letter-spacing:.5px;
    border-bottom:1px solid #30363d;white-space:nowrap;}}
  .pm-table td {{padding:10px 14px;border-bottom:1px solid #21262d;
    color:#c9d1d9;vertical-align:middle;}}
  .pm-table tbody tr:hover td {{background:#161b22;}}
  .pm-table tbody tr:last-child td {{border-bottom:none;}}
</style>
<div class="pm-wrap">
  <table class="pm-table">
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
"""


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px 0;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
        <span style="font-size:22px;">🔍</span>
        <span style="color:#e6edf3;font-size:17px;font-weight:800;">Product Matcher</span>
      </div>
      <p style="color:#8b949e;font-size:12px;margin:0;">Barcode + AI fuzzy matching</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # DB health
    try:
        conn = get_connection()
        conn.close()
        st.markdown('<p style="color:#3fb950;font-size:12px;font-weight:600;">● Database connected</p>',
                    unsafe_allow_html=True)
    except psycopg2.OperationalError as e:
        st.markdown('<p style="color:#f85149;font-size:12px;font-weight:600;">● Database offline</p>',
                    unsafe_allow_html=True)
        st.error(str(e))
        st.code("python setup_db.py", language="bash")
        st.stop()

    st.divider()

    # ── File reader (used for both item master and supplier uploads) ────────────
    def _read_supplier_file(f) -> "pd.DataFrame":
        """
        Read CSV or Excel with automatic encoding detection.
        Tries utf-8 → utf-8-sig → cp1252 → latin-1 for CSV files.
        Handles 0xa0 non-breaking spaces common in Windows/Excel exports.
        """
        if not f.name.lower().endswith(".csv"):
            return pd.read_excel(f)
        raw = f.read()
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                import io as _io
                decoded = raw.decode(enc).replace("\xa0", " ")
                return pd.read_csv(_io.StringIO(decoded))
            except (UnicodeDecodeError, Exception):
                pass
        raise ValueError("Could not decode file. Try saving as UTF-8 CSV from Excel.")

    st.markdown('<p class="section-label">Item Master</p>', unsafe_allow_html=True)
    try:
        im_count = get_itemmaster_count()
        st.markdown(
            f'<div class="stat-card" style="margin-bottom:12px;">'
            f'<div class="val">{im_count:,}</div>'
            f'<div class="lbl">Records in master</div></div>',
            unsafe_allow_html=True,
        )
    except Exception:
        im_count = 0
        st.warning("Run setup_db.py first")

    im_file = st.file_uploader(
        "Upload Item Master (Excel / CSV)",
        type=["xlsx", "xls", "csv"],
        key="im_upload",
        help="Required: item_code, item_name, barcode | Optional: status",
    )

    if im_file:
        try:
            im_df = _read_supplier_file(im_file)
            im_df.columns = im_df.columns.str.lower().str.strip().str.replace(r"\s+", "_", regex=True)
            missing_im = {"item_code", "item_name", "barcode"} - set(im_df.columns)
            if missing_im:
                st.error(f"Missing: {missing_im}")
            else:
                st.caption(f"{len(im_df):,} rows detected")
                st.dataframe(im_df.head(5), use_container_width=True)
                if st.button("Import to Item Master", type="primary"):
                    with st.spinner("Importing…"):
                        n = bulk_insert_itemmaster(im_df)
                    st.session_state.pop("im_index", None)  # force index rebuild
                    st.success(f"Upserted {n:,} records")
                    st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    with st.expander("Column guide"):
        st.markdown(
            "**Item Master**\n"
            "- `item_code` — unique key\n"
            "- `item_name` — used for fuzzy match\n"
            "- `barcode` — exact match\n"
            "- `status` — active / inactive\n\n"
            "**Supplier file**\n"
            "- `item_description` — required\n"
            "- `barcode` — optional\n"
            "- `barcode1` — optional alternate",
        )


# ── Page header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h1>🔍 Product Matcher</h1>
  <p>Match supplier products to your item master using barcode lookup &amp; AI fuzzy matching</p>
</div>
""", unsafe_allow_html=True)


# ── Upload supplier file ───────────────────────────────────────────────────────
st.markdown('<p class="section-label">Step 1 — Upload Supplier File</p>', unsafe_allow_html=True)

supplier_file = st.file_uploader(
    "Excel or CSV  ·  Required column: item_description  ·  Optional: barcode, barcode1",
    type=["xlsx", "xls", "csv"],
    key="supplier_upload",
)

if not supplier_file:
    st.markdown("""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin-top:8px;">
      <p style="color:#8b949e;font-size:13px;margin:0 0 12px 0;font-weight:600;">Expected format</p>
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead>
          <tr style="background:#21262d;">
            <th style="padding:8px 12px;color:#8b949e;font-size:11px;text-transform:uppercase;
                letter-spacing:.5px;border-bottom:1px solid #30363d;text-align:left;">barcode</th>
            <th style="padding:8px 12px;color:#8b949e;font-size:11px;text-transform:uppercase;
                letter-spacing:.5px;border-bottom:1px solid #30363d;text-align:left;">barcode1</th>
            <th style="padding:8px 12px;color:#8b949e;font-size:11px;text-transform:uppercase;
                letter-spacing:.5px;border-bottom:1px solid #30363d;text-align:left;">item_description</th>
          </tr>
        </thead>
        <tbody>
          <tr><td style="padding:8px 12px;color:#6e7681;border-bottom:1px solid #21262d;">1234567890123</td>
              <td style="padding:8px 12px;color:#6e7681;border-bottom:1px solid #21262d;">ALT-001</td>
              <td style="padding:8px 12px;color:#c9d1d9;border-bottom:1px solid #21262d;">Coca Cola 500ml Bottle</td></tr>
          <tr><td style="padding:8px 12px;color:#6e7681;border-bottom:1px solid #21262d;">9876543210987</td>
              <td style="padding:8px 12px;color:#6e7681;border-bottom:1px solid #21262d;"></td>
              <td style="padding:8px 12px;color:#c9d1d9;border-bottom:1px solid #21262d;">Pepsi 1Ltr PK6</td></tr>
          <tr><td style="padding:8px 12px;color:#6e7681;"></td>
              <td style="padding:8px 12px;color:#6e7681;">ALT-789</td>
              <td style="padding:8px 12px;color:#c9d1d9;">Lays Classic Salted 50g</td></tr>
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


try:
    sup_df = _read_supplier_file(supplier_file)
except Exception as e:
    st.error(f"Cannot read file: {e}")
    st.stop()

sup_df.columns = sup_df.columns.str.lower().str.strip().str.replace(r"\s+", "_", regex=True)

if "item_description" not in sup_df.columns:
    st.error("File must contain an `item_description` column.")
    st.stop()

for col in ("barcode", "barcode1"):
    if col not in sup_df.columns:
        sup_df[col] = ""

sup_df = sup_df[["barcode", "barcode1", "item_description"]].fillna("").copy()

# Normalise barcode columns — Excel stores numbers as floats (50107452.0 → 50107452)
for col in ("barcode", "barcode1"):
    sup_df[col] = sup_df[col].apply(
        lambda v: (str(v)[:-2] if str(v).endswith(".0") and str(v)[:-2].lstrip("-").isdigit()
                   else str(v)).strip()
    )

# Clear stored results whenever a new file is uploaded
file_sig = f"{supplier_file.name}_{len(sup_df)}"
if st.session_state.get("_file_sig") != file_sig:
    st.session_state.pop("results_df", None)
    st.session_state.pop("match_session_id", None)
    st.session_state["_file_sig"] = file_sig

st.markdown(
    f'<p style="color:#3fb950;font-size:13px;font-weight:600;margin:4px 0 12px 0;">'
    f'✓ Loaded {len(sup_df):,} rows from '
    f'<code style="background:#21262d;padding:2px 6px;border-radius:4px;color:#58a6ff;">'
    f'{supplier_file.name}</code></p>',
    unsafe_allow_html=True,
)

with st.expander("Preview uploaded data", expanded=False):
    st.dataframe(sup_df.head(20), use_container_width=True)


# ── Run matching ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-label" style="margin-top:20px;">Step 2 — Match Products</p>',
            unsafe_allow_html=True)

col_btn, col_hint = st.columns([1, 5])
with col_btn:
    run_btn = st.button("Run Matching", type="primary", use_container_width=True)
with col_hint:
    st.markdown(
        '<p style="color:#6e7681;font-size:12px;padding-top:10px;">'
        'Priority: Exact Barcode → Barcode1 → Fuzzy / Semantic</p>',
        unsafe_allow_html=True,
    )

if run_btn:
    if im_count == 0:
        st.warning("Item Master is empty — import it from the sidebar first.")
        st.stop()

    # Load + index itemmaster once per session; reuse on subsequent runs
    if "im_index" not in st.session_state or len(st.session_state["im_index"]) != 7:
        with st.spinner("Loading item master… (first run builds a disk cache — faster on next refresh)"):
            raw_im = load_itemmaster()
            st.session_state["im_index"] = build_index(raw_im)

    im, norm_names, gram_tokens, discrim_tokens, bc_dict, tfidf_vec, tfidf_mat = st.session_state["im_index"]

    prog  = st.progress(0)
    label = st.empty()

    def progress_cb(pct: float, msg: str = ""):
        prog.progress(min(pct, 1.0))
        if msg:
            label.markdown(
                f'<p style="color:#8b949e;font-size:12px;margin:2px 0;">{msg}</p>',
                unsafe_allow_html=True,
            )

    results_df = run_matching(sup_df, im, norm_names, gram_tokens, discrim_tokens, bc_dict, tfidf_vec, tfidf_mat, progress_cb=progress_cb)

    prog.progress(1.0)
    label.markdown(
        f'<p style="color:#3fb950;font-size:12px;margin:2px 0;font-weight:600;">'
        f'✓ Matched {len(results_df):,} rows</p>',
        unsafe_allow_html=True,
    )

    session_id = insert_uploaded_items(sup_df)
    st.session_state["results_df"]       = results_df
    st.session_state["match_session_id"] = session_id

# ── Results (live — persisted in session state) ────────────────────────────────
if "results_df" not in st.session_state:
    st.stop()

results_df = st.session_state["results_df"]
session_id = st.session_state.get("match_session_id", "export")

# ── Summary metrics ────────────────────────────────────────────────────────────
st.markdown('<p class="section-label" style="margin-top:24px;">Step 3 — Results</p>',
            unsafe_allow_html=True)

total     = len(results_df)
exact     = int((results_df["Match Type"] == "EXACT").sum())
exact_sec = int((results_df["Match Type"] == "EXACT_SECONDARY").sum())
fuzzy_cnt = int((results_df["Match Type"] == "FUZZY").sum())
no_match  = int((results_df["Match Type"] == "NO_MATCH").sum())
dup_bc    = int((results_df["Duplicate Item Codes"] != "").sum())

st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin-bottom:20px;">
  <div class="stat-card"><div class="val">{total:,}</div><div class="lbl">Total Rows</div></div>
  <div class="stat-card green"><div class="val">{exact:,}</div><div class="lbl">Exact Barcode</div></div>
  <div class="stat-card blue"><div class="val">{exact_sec:,}</div><div class="lbl">Barcode1 Match</div></div>
  <div class="stat-card yellow"><div class="val">{fuzzy_cnt:,}</div><div class="lbl">Fuzzy Match</div></div>
  <div class="stat-card red"><div class="val">{no_match:,}</div><div class="lbl">No Match</div></div>
  <div class="stat-card yellow"><div class="val">{dup_bc:,}</div><div class="lbl">Duplicate Barcodes</div></div>
</div>
""", unsafe_allow_html=True)


# ── Live filter bar (no button needed) ────────────────────────────────────────
f_col1, f_col2, f_col3 = st.columns([2, 2, 3])
with f_col1:
    filter_type = st.selectbox(
        "Filter by Match Type",
        ["All", "EXACT", "EXACT_SECONDARY", "FUZZY", "NO_MATCH"],
        key="filter_type",
    )
with f_col2:
    filter_conf = st.selectbox(
        "Filter by Confidence",
        ["All", "HIGH", "MEDIUM", "LOW"],
        key="filter_conf",
    )
with f_col3:
    search_term = st.text_input(
        "Search supplier description (live)",
        key="search_desc",
        placeholder="e.g. Coca Cola  — results update as you type",
    )

display_df = results_df.copy()
if filter_type != "All":
    display_df = display_df[display_df["Match Type"] == filter_type]
if filter_conf != "All":
    display_df = display_df[display_df["Confidence"] == filter_conf]
if search_term.strip():
    display_df = display_df[
        display_df["Supplier Description"].str.contains(search_term.strip(), case=False, na=False)
    ]

st.markdown(
    f'<p style="color:#6e7681;font-size:12px;margin:0 0 8px 0;">'
    f'Showing {len(display_df):,} of {total:,} rows</p>',
    unsafe_allow_html=True,
)

# ── HTML results table ─────────────────────────────────────────────────────────
table_html = build_html_table(display_df)
components.html(table_html, height=min(620, max(200, len(display_df) * 44 + 60)), scrolling=True)


# ── Download ───────────────────────────────────────────────────────────────────
st.divider()
buf = io.StringIO()
results_df.to_csv(buf, index=False)

st.download_button(
    label="⬇  Download Full Results (CSV)",
    data=buf.getvalue(),
    file_name=f"matching_results_{session_id}.csv",
    mime="text/csv",
    type="primary",
)
